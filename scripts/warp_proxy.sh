#!/usr/bin/env bash
# Optional fallback route for trailer downloads: a free Cloudflare WARP
# WireGuard tunnel exposed as a local SOCKS5 proxy (userspace, no root VPN).
# YouTube often blocks GitHub datacenter IPs; WARP egress IPs are usually not.
# This is an unofficial, best-effort workaround: on any failure the job simply
# continues without the fallback route.
set -euo pipefail

PORT="${WARP_SOCKS_PORT:-40000}"
DIR="$RUNNER_TEMP/warp"
mkdir -p "$DIR"
cd "$DIR"

asset() {
  # Resolve the newest linux/amd64 release asset of a GitHub repository.
  local auth=()
  [ -n "${GITHUB_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
  curl --fail --silent --show-error --location --retry 3 "${auth[@]}" \
    "https://api.github.com/repos/$1/releases/latest" |
    jq -r --arg pattern "$2" '[.assets[] | select(.name | test($pattern))][0].browser_download_url'
}

wgcf_url="$(asset ViRb3/wgcf 'linux_amd64$')"
proxy_url="$(asset windtf/wireproxy 'linux_amd64\.tar\.gz$')"
[ -n "$wgcf_url" ] && [ "$wgcf_url" != null ] && [ -n "$proxy_url" ] && [ "$proxy_url" != null ] || {
  echo "::warning title=WARP fallback unavailable::Could not resolve wgcf/wireproxy release downloads."
  exit 0
}
curl --fail --silent --show-error --location --retry 3 "$wgcf_url" -o wgcf
curl --fail --silent --show-error --location --retry 3 "$proxy_url" -o wireproxy.tar.gz
tar -xzf wireproxy.tar.gz
chmod +x wgcf wireproxy

./wgcf register --accept-tos > /dev/null
./wgcf generate > /dev/null
{
  cat wgcf-profile.conf
  printf '\n[Socks5]\nBindAddress = 127.0.0.1:%s\n' "$PORT"
} > wireproxy.conf
./wireproxy -c wireproxy.conf > wireproxy.log 2>&1 &

for _ in $(seq 1 30); do
  if trace="$(curl --silent --max-time 8 --socks5-hostname "127.0.0.1:$PORT" https://www.cloudflare.com/cdn-cgi/trace)" &&
     grep -q '^warp=on' <<< "$trace"; then
    echo "YTDLP_FALLBACK_PROXY=socks5h://127.0.0.1:$PORT" >> "$GITHUB_ENV"
    echo "::notice title=WARP fallback ready::Trailer downloads can retry through Cloudflare WARP if YouTube blocks the runner."
    exit 0
  fi
  sleep 2
done
echo "::warning title=WARP fallback unavailable::The WARP tunnel did not come up; downloads use the direct route only."
exit 0
