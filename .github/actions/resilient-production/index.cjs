'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { StringDecoder } = require('node:string_decoder');

const LOG_LIMIT = 2 * 1024 * 1024;
const TOKEN = /(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|hf_[A-Za-z0-9]{12,}|[0-9]{7,12}:[A-Za-z0-9_-]{25,}|eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,})/g;
const SECRET_KEY = /token|secret|password|passwd|cookie|api.?key|private.?key|authorization|credential/i;

function redact(text, environment = process.env) {
  let result = String(text);
  for (const [name, value] of Object.entries(environment)) {
    if (SECRET_KEY.test(name) && value && value.length >= 4 && value.length <= 65536) {
      result = result.split(value).join('[REDACTED]');
    }
  }
  return result.replace(TOKEN, '[REDACTED]')
    .replace(/(https?:\/\/)[^\s/@:]+:[^\s/@]+@/gi, '$1[REDACTED]@')
    .replace(/([?&]([^=&\s"'<>]{1,128})=)[^&\s"'<>]+/g, (match, prefix, key) => {
      let decoded = key;
      try { decoded = decodeURIComponent(key); } catch { /* malformed query key */ }
      return /token|key|sig|credential|auth|password|secret|cookie|session|jwt|policy|code/i.test(decoded)
        ? prefix + '[REDACTED]' : match;
    })
    .replace(/\b(authorization\s*[:=]\s*(?:bearer\s+)?|cookie\s*[:=]\s*)[^\r\n]+/gi, '$1[REDACTED]');
}

function warning(kind) {
  // Never print arbitrary transport errors; they may contain signed URLs.
  const message = 'Recovery checkpoint ' + kind + '; production will continue.';
  try { require('@actions/core').warning(message); }
  catch { process.stdout.write('::warning::' + message + '\n'); }
}

function parseName(name, runId) {
  const match = /^reliable-checkpoint-(\d+)-(\d+)-(\d+)$/.exec(name || '');
  if (!match || match[1] !== String(runId)) return null;
  const attempt = Number(match[2]), sequence = Number(match[3]);
  if (!Number.isSafeInteger(attempt) || !Number.isSafeInteger(sequence)) return null;
  return { attempt, sequence };
}

function sortCheckpoints(artifacts, runId) {
  return artifacts.filter(item => parseName(item.name, runId)).sort((left, right) => {
    const a = parseName(left.name, runId), b = parseName(right.name, runId);
    return b.attempt - a.attempt || b.sequence - a.sequence;
  });
}

function previousCheckpoint(artifacts, runId, attempt) {
  return sortCheckpoints(artifacts, runId).find(item => parseName(item.name, runId).attempt < Number(attempt));
}

function removableCheckpoints(artifacts, runId, uploadedName) {
  // A successful new upload is always retained even if another attempt races us.
  const older = sortCheckpoints(artifacts, runId).filter(item => item.name !== uploadedName);
  return older.slice(1);
}

class BoundedLog {
  constructor(filename, environment = process.env) {
    this.filename = filename;
    this.environment = environment;
    this.chunks = [];
    this.bytes = 0;
    this.pending = new Map();
    this.decoders = new Map();
    this.dropping = new Set();
  }

  append(text) {
    const data = Buffer.from(redact(text, this.environment));
    if (!data.length) return;
    this.chunks.push(data);
    this.bytes += data.length;
    while (this.bytes > LOG_LIMIT && this.chunks.length) {
      const excess = this.bytes - LOG_LIMIT;
      const first = this.chunks[0];
      if (first.length <= excess) { this.chunks.shift(); this.bytes -= first.length; }
      else { this.chunks[0] = first.subarray(excess); this.bytes -= excess; }
    }
  }

  consume(source, chunk) {
    if (!this.decoders.has(source)) this.decoders.set(source, new StringDecoder('utf8'));
    let text = (this.pending.get(source) || '') + this.decoders.get(source).write(chunk);
    const lines = text.split('\n');
    this.pending.set(source, lines.pop());
    for (const line of lines) {
      if (!this.dropping.has(source)) {
        this.append(line.length > 131072 ? '[oversized log line omitted]\n' : line + '\n');
      }
      this.dropping.delete(source);
    }
    if (this.pending.get(source).length > 131072) {
      if (!this.dropping.has(source)) this.append('[oversized log line omitted]\n');
      this.dropping.add(source);
      this.pending.set(source, '');
    }
  }

  finish() {
    for (const [source, decoder] of this.decoders) {
      if (!this.dropping.has(source)) this.append((this.pending.get(source) || '') + decoder.end());
    }
    this.pending.clear();
    this.decoders.clear();
    this.dropping.clear();
    this.flush();
  }

  flush() {
    const temporary = this.filename + '.tmp-' + process.pid;
    fs.writeFileSync(temporary, Buffer.concat(this.chunks), { mode: 0o600 });
    fs.renameSync(temporary, this.filename);
  }
}

function runBounded(command, args, timeoutMs, options = {}) {
  return new Promise((resolve, reject) => {
    let child;
    try { child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options }); }
    catch { reject(new Error('spawn_unavailable')); return; }
    let stdout = '', stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); }, timeoutMs);
    timer.unref();
    child.stdout.on('data', data => { if (stdout.length < 65536) stdout += data.toString().slice(0, 65536 - stdout.length); });
    child.stderr.on('data', data => { if (stderr.length < 4096) stderr += data.toString().slice(0, 4096 - stderr.length); });
    child.once('error', () => { clearTimeout(timer); reject(new Error('spawn_unavailable')); });
    child.once('close', code => {
      clearTimeout(timer);
      if (code === 0) resolve(stdout);
      else reject(new Error('helper_unavailable'));
    });
  });
}

async function artifactWorker(request) {
  const { DefaultArtifactClient } = require('@actions/artifact');
  const client = new DefaultArtifactClient();
  if (request.operation === 'upload') {
    const uploaded = await client.uploadArtifact(request.name, [request.archive], request.directory,
      { retentionDays: 1, compressionLevel: 0 });
    if (!uploaded.id) throw new Error('upload_unavailable');
    // Delete older snapshots only after the new one is durably accepted.
    try {
      const { artifacts } = await client.listArtifacts();
      for (const old of removableCheckpoints(artifacts, request.runId, request.name)) {
        try { await client.deleteArtifact(old.name); } catch { /* retention is best effort */ }
      }
    } catch { /* successful upload remains usable if listing is unavailable */ }
    return { uploaded: true };
  }
  if (request.operation === 'restore') {
    const { artifacts } = await client.listArtifacts();
    const chosen = previousCheckpoint(artifacts, request.runId, request.attempt);
    if (!chosen) return { restored: false };
    await client.downloadArtifact(chosen.id, { path: request.directory });
    return { restored: true };
  }
  throw new Error('unknown_operation');
}

async function callWorker(request, directory, timeoutMs = 90000, workerScript = __filename) {
  const requestPath = path.join(directory, 'artifact-request.json');
  const resultPath = path.join(directory, 'artifact-result.json');
  fs.writeFileSync(requestPath, JSON.stringify({ ...request, resultPath }), { mode: 0o600 });
  // The artifact toolkit logs progress on stdout. Results use a private file so
  // those messages cannot corrupt the worker protocol or leak signed URLs.
  await runBounded(process.execPath, [workerScript, '--artifact-worker', requestPath], timeoutMs);
  return JSON.parse(fs.readFileSync(resultPath, 'utf8'));
}

function runtimeAvailable() {
  return Boolean(process.env.ACTIONS_RUNTIME_TOKEN && process.env.ACTIONS_RESULTS_URL &&
    /^\d+$/.test(process.env.GITHUB_RUN_ID || '') && /^\d+$/.test(process.env.GITHUB_RUN_ATTEMPT || ''));
}

function tempDirectory() {
  return fs.mkdtempSync(path.join(process.env.RUNNER_TEMP || os.tmpdir(), 'production-checkpoint-'));
}

async function snapshot(python, log) {
  let temporary;
  try {
    log.flush();
    if (!runtimeAvailable()) { warning('upload unavailable'); return; }
    temporary = tempDirectory();
    const archive = path.join(temporary, 'snapshot.tar.gz');
    await runBounded(python, ['-m', 'scripts.checkpoint_snapshot', 'create', '--output', archive], 45000);
    const sequence = String(++snapshot.sequence).padStart(4, '0');
    const name = 'reliable-checkpoint-' + process.env.GITHUB_RUN_ID + '-' +
      process.env.GITHUB_RUN_ATTEMPT + '-' + sequence;
    await callWorker({ operation: 'upload', name, archive, directory: temporary,
      runId: process.env.GITHUB_RUN_ID }, temporary);
    process.stdout.write('Recovery checkpoint uploaded (' + name + ').\n');
  } catch { warning('upload unavailable'); }
  finally { if (temporary) { try { fs.rmSync(temporary, { recursive: true, force: true }); } catch { /* optional cleanup */ } } }
}
snapshot.sequence = 0;

async function restore(python) {
  let temporary;
  try {
    if (!runtimeAvailable()) { warning('restore unavailable'); return; }
    temporary = tempDirectory();
    const result = await callWorker({ operation: 'restore', directory: temporary,
      runId: process.env.GITHUB_RUN_ID, attempt: process.env.GITHUB_RUN_ATTEMPT }, temporary);
    if (!result.restored) {
      process.stdout.write('No earlier-attempt recovery checkpoint in this workflow run.\n');
      return;
    }
    const archive = path.join(temporary, 'snapshot.tar.gz');
    if (!fs.lstatSync(archive).isFile()) throw new Error('invalid_archive');
    await runBounded(python, ['-m', 'scripts.checkpoint_snapshot', 'restore', '--archive', archive], 45000);
    process.stdout.write('Earlier-attempt recovery checkpoint restored.\n');
  } catch { warning('restore unavailable'); }
  finally { if (temporary) { try { fs.rmSync(temporary, { recursive: true, force: true }); } catch { /* optional cleanup */ } } }
}

function signalCode(signal) {
  return signal === 'SIGINT' ? 130 : signal === 'SIGKILL' ? 137 : 143;
}

async function runProduction(python, intervalSeconds, command = null) {
  const directory = path.join(process.cwd(), 'work', 'runner-diagnostics');
  fs.mkdirSync(directory, { recursive: true });
  const log = new BoundedLog(path.join(directory, 'production.log'));
  log.append('Guarded production starting; checkpoint artifacts are best effort.\n');
  // Persist already-warmed state before the long process starts. An immediate
  // host loss can still predate the first upload; no in-process hook can fix that.
  await snapshot(python, log);
  const argv = command || [python, '-m', 'scripts.runner_guard', '--log-dir', 'work/runner-diagnostics',
    '--', 'bash', 'scripts/produce_guarded.sh'];
  const child = spawn(argv[0], argv.slice(1), { stdio: ['ignore', 'pipe', 'pipe'] });
  let checkpoint = null, terminating = false, terminationTimer;
  const checkpointNow = () => {
    if (!checkpoint) checkpoint = snapshot(python, log).finally(() => { checkpoint = null; });
    return checkpoint;
  };
  const interval = setInterval(checkpointNow, intervalSeconds * 1000);
  interval.unref();
  const flushInterval = setInterval(() => { try { log.flush(); } catch { /* diagnostic disk write is optional */ } }, 5000);
  flushInterval.unref();
  child.stdout.on('data', data => { process.stdout.write(data); log.consume('stdout', data); });
  child.stderr.on('data', data => { process.stderr.write(data); log.consume('stderr', data); });
  const onSignal = signal => {
    if (terminating) return;
    terminating = true;
    log.append('Cancellation received; forwarding to the resource guard.\n');
    child.kill(signal);
    // The guard owns all process-tree cleanup. Keep the action bounded if it
    // cannot finish, and begin best-effort persistence while cancellation allows.
    void checkpointNow();
    terminationTimer = setTimeout(() => child.kill('SIGKILL'), 45000);
    terminationTimer.unref();
  };
  const term = () => onSignal('SIGTERM'), interrupt = () => onSignal('SIGINT');
  process.on('SIGTERM', term);
  process.on('SIGINT', interrupt);
  let result;
  try {
    result = await new Promise(resolve => {
      child.once('error', () => resolve(1));
      child.once('close', (code, signal) => resolve(code === null ? signalCode(signal) : code));
    });
  } finally {
    clearInterval(interval);
    clearInterval(flushInterval);
    clearTimeout(terminationTimer);
    // Keep cancellation forwarding in place while the final upload is running.
  }
  try { log.append('Resource guard exited with status ' + result + '.\n'); log.finish(); }
  catch { warning('diagnostic log unavailable'); }
  if (checkpoint) await checkpoint;
  // Flush telemetry and logs after guard exit, including ordinary application
  // failures. Never restart an application solely because it returned nonzero.
  await snapshot(python, log);
  process.removeListener('SIGTERM', term);
  process.removeListener('SIGINT', interrupt);
  return terminating && result === 0 ? 143 : result;
}

async function main() {
  const python = process.env.INPUT_PYTHON || 'python';
  const mode = process.env.INPUT_MODE || 'run';
  if (mode === 'restore') { await restore(python); return 0; }
  if (mode !== 'run') throw new Error('invalid_mode');
  const interval = Number(process.env['INPUT_CHECKPOINT-INTERVAL-SECONDS'] || '900');
  if (!Number.isFinite(interval) || interval < 1 || interval > 3600) throw new Error('invalid_interval');
  let command = null;
  if (process.env['INPUT_COMMAND-JSON']) {
    command = JSON.parse(process.env['INPUT_COMMAND-JSON']);
    if (!Array.isArray(command) || command.length === 0 || command.length > 128 ||
        command.some(item => typeof item !== 'string' || item.includes('\0')) || !command[0]) {
      throw new Error('invalid_command');
    }
  }
  return runProduction(python, interval, command);
}

module.exports = { redact, parseName, sortCheckpoints, previousCheckpoint, removableCheckpoints,
  BoundedLog, runBounded, signalCode, runProduction, callWorker };

if (require.main === module) {
  if (process.argv[2] === '--artifact-worker') {
    let request;
    try { request = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')); }
    catch { process.exitCode = 1; }
    if (request) artifactWorker(request).then(
      result => {
        try { fs.writeFileSync(request.resultPath, JSON.stringify(result), { mode: 0o600 }); }
        catch { process.exit(1); }
        process.exit(0);
      },
      () => process.exit(1));
  } else {
    main().then(code => { process.exitCode = code; }, () => {
      // Runtime failures of the production wrapper must be visible, with no
      // arbitrary error text or command/environment dump.
      process.stderr.write('Guarded production wrapper failed.\n');
      process.exitCode = 1;
    });
  }
}
