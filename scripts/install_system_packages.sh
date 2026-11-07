#!/usr/bin/env bash
# Install apt packages without letting a slow, stalled or broken mirror fail or hang the job.
# - `apt-get update` errors (often from an unrelated third-party list) are reported, not fatal.
# - Each step has a hard time limit. The retry switches every mirror (including the runner's
#   apt-mirrors.txt list, which favors Azure) to Ubuntu's main archive; finished downloads are kept.
# - If FFmpeg is still missing, a static *release* build is installed (never a development snapshot).
# Usage: bash scripts/install_system_packages.sh ffmpeg tesseract-ocr ...
set -uo pipefail

packages=("$@")
options=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
log="${RUNNER_TEMP:-/tmp}/apt-install.log"

report() {
  local title="$1"
  local detail
  detail="$(tail -n 6 "$log" 2>/dev/null | tr '\n' ' ' | cut -c1-900)"
  echo "::warning title=$title::$detail"
}

attempt() {
  local limit="$1"
  : > "$log"
  if ! timeout 240 sudo apt-get "${options[@]}" update -qq >>"$log" 2>&1; then
    report "apt-get update reported a problem (continuing)"
  fi
  timeout "$limit" sudo env DEBIAN_FRONTEND=noninteractive apt-get "${options[@]}" install -y --no-install-recommends \
    "${packages[@]}" >>"$log" 2>&1
}

use_main_archive() {
  for file in /etc/apt/apt-mirrors.txt /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    [ -f "$file" ] || continue
    sudo sed -i 's#http://azure.archive.ubuntu.com#http://archive.ubuntu.com#g' "$file"
  done
  if [ -f /etc/apt/apt-mirrors.txt ]; then
    # The runner's mirror list may name Azure plus others; keep only the main archive (and security).
    printf 'http://archive.ubuntu.com/ubuntu/\tpriority:1\n' | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
  fi
}

version_notice() {
  echo "::notice title=FFmpeg installed::$(ffmpeg -version 2>/dev/null | head -n 1 | cut -c1-120)"
}

if attempt 300; then
  version_notice
  exit 0
fi
report "Package install failed on the default mirror; retrying with Ubuntu's main archive"
use_main_archive
if attempt 600; then
  version_notice
  exit 0
fi
report "Package install failed again"

# Last resort for the one package everything needs: a static FFmpeg release build.
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "::warning title=Installing static FFmpeg::apt could not install FFmpeg; using a static release build."
  dir="${RUNNER_TEMP:-/tmp}/ffmpeg-static"
  installed=false
  for url in \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz" \
    "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"; do
    rm -rf "$dir" && mkdir -p "$dir"
    if curl --fail --silent --show-error --location --retry 3 --max-time 300 "$url" -o "$dir/ffmpeg.tar.xz" &&
       tar -xJf "$dir/ffmpeg.tar.xz" -C "$dir"; then
      binary="$(find "$dir" -type f -name ffmpeg -perm -u+x | head -n 1)"
      probe="$(find "$dir" -type f -name ffprobe -perm -u+x | head -n 1)"
      if [ -n "$binary" ] && [ -n "$probe" ]; then
        sudo cp "$binary" "$probe" /usr/local/bin/
        installed=true
        break
      fi
    fi
  done
  if [ "$installed" != true ]; then
    echo "::error title=FFmpeg unavailable::Neither apt nor a static build could install FFmpeg."
    exit 1
  fi
fi
version_notice
# Other packages are optional for the pipeline (OCR, fonts); report what is missing and continue.
for package in "${packages[@]}"; do
  [ "$package" = ffmpeg ] && command -v ffmpeg >/dev/null && continue
  dpkg -s "$package" >/dev/null 2>&1 || echo "::warning title=Optional package missing::$package could not be installed."
done
exit 0
