'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { redact, previousCheckpoint, removableCheckpoints, BoundedLog, runProduction,
  signalCode, callWorker } = require('./index.cjs');

test('secret values, authorization and signed URL credentials are scrubbed', () => {
  const text = 'known-secret\nhttps://user:password@host/file?sig=private&ok=yes\n' +
    'Authorization: Bearer private-bearer\nCookie: private-cookie\nhf_123456789012345';
  const result = redact(text, { ACCESS_TOKEN: 'known-secret' });
  for (const secret of ['known-secret', 'user:password', 'sig=private', 'private-bearer',
    'private-cookie', 'hf_123456789012345']) assert.equal(result.includes(secret), false);
  assert.equal(result.includes('ok=yes'), true);
});

test('restore selects latest snapshot in earlier attempt of the same run', () => {
  const names = ['reliable-checkpoint-42-1-0002', 'reliable-checkpoint-42-2-0001',
    'reliable-checkpoint-42-2-0009', 'reliable-checkpoint-42-3-0001',
    'reliable-checkpoint-99-2-9999', 'unrelated'];
  const artifacts = names.map((name, index) => ({ name, id: index + 1 }));
  assert.equal(previousCheckpoint(artifacts, '42', '3').name, 'reliable-checkpoint-42-2-0009');
  assert.equal(previousCheckpoint(artifacts, '42', '1'), undefined);
});

test('retention preserves uploaded snapshot plus predecessor and ignores unrelated artifacts', () => {
  const names = ['reliable-checkpoint-42-1-0001', 'reliable-checkpoint-42-2-0001',
    'reliable-checkpoint-42-2-0002', 'warm-cache', 'reliable-checkpoint-99-2-0001'];
  const uploaded = names[2];
  const removals = removableCheckpoints(names.map(name => ({ name })), '42', uploaded);
  assert.deepEqual(removals.map(item => item.name), [names[0]]);
});

test('artifact log masks secrets split between UTF8 chunks and bounds long output', () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'checkpoint-log-test-'));
  try {
    const filename = path.join(temporary, 'production.log');
    const log = new BoundedLog(filename, { SOME_SECRET: 'a-private-secret' });
    log.consume('stdout', Buffer.from('before a-private-'));
    log.consume('stdout', Buffer.from('secret after\n'));
    log.consume('stderr', Buffer.from('prefix ' + 'x'.repeat(140000)));
    log.consume('stderr', Buffer.from('discard this tail\nlast line\n'));
    for (let index = 0; index < 600; index++) log.consume('stdout', Buffer.from('x'.repeat(4096) + '\n'));
    log.consume('stdout', Buffer.from('ending a-private-secret\n'));
    log.finish();
    const output = fs.readFileSync(filename, 'utf8');
    assert.equal(output.includes('a-private-secret'), false);
    assert.equal(output.includes('discard this tail'), false);
    assert.equal(output.includes('ending [REDACTED]'), true);
    assert.ok(fs.statSync(filename).size <= 2 * 1024 * 1024);
  } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
});

test('optional artifact unavailability preserves application exit status', async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'checkpoint-exit-test-'));
  const previous = process.cwd();
  const runtime = process.env.ACTIONS_RUNTIME_TOKEN;
  try {
    process.chdir(temporary);
    delete process.env.ACTIONS_RUNTIME_TOKEN;
    const status = await runProduction('unused-python', 60,
      [process.execPath, '-e', 'process.stdout.write("tiny production\\n");process.exit(17)']);
    assert.equal(status, 17);
    assert.match(fs.readFileSync(path.join(temporary, 'work/runner-diagnostics/production.log'), 'utf8'),
      /Resource guard exited with status 17/);
  } finally {
    process.chdir(previous);
    if (runtime === undefined) delete process.env.ACTIONS_RUNTIME_TOKEN;
    else process.env.ACTIONS_RUNTIME_TOKEN = runtime;
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

test('signal failures map to conventional shell statuses', () => {
  assert.equal(signalCode('SIGINT'), 130);
  assert.equal(signalCode('SIGTERM'), 143);
  assert.equal(signalCode('SIGKILL'), 137);
});


test('worker progress on stdout cannot corrupt its private result protocol', async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'checkpoint-worker-test-'));
  try {
    const worker = path.join(temporary, 'worker.cjs');
    fs.writeFileSync(worker, [
      "const fs = require('node:fs');",
      "const request = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));",
      "console.log('Artifact toolkit progress output'.repeat(3000));",
      "fs.writeFileSync(request.resultPath, JSON.stringify({ uploaded: true }));",
    ].join('\n'));
    const result = await callWorker({ operation: 'upload' }, temporary, 5000, worker);
    assert.deepEqual(result, { uploaded: true });
  } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
});

test('cloud signatures, credentials and encoded credential query keys are scrubbed', () => {
  const result = redact('https://host/file?X-Amz-Signature=amz-private&X-Amz-Credential=credential-private' +
    '&X-Goog-Signature=goog-private&lsig=lsig-private&%74oken=encoded-private&v=public-video', {});
  for (const value of ['amz-private', 'credential-private', 'goog-private', 'lsig-private', 'encoded-private']) {
    assert.equal(result.includes(value), false);
  }
  assert.equal(result.includes('v=public-video'), true);
});
