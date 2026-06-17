#!/usr/bin/env bash
# Poll EC2 algofeast-api logs for high-signal trading events.
set -euo pipefail
KEY="${VIBEFNO_KEY:-/Users/premkumar/Documents/vibefno.pem}"
HOST="${VIBEFNO_HOST:-ec2-user@ec2-3-108-61-102.ap-south-1.compute.amazonaws.com}"
OUT="${TRADE_WATCH_LOG:-/Users/premkumar/Documents/tradehandler-ai-workspace/.trade-watch-alerts.log}"
PATTERN='V2 LIMIT|V2_ORDER|auto_placed|auto_gtt|auto_stale|ExitTrail|Step [0-9]|ALLOW_AUTONOMOUS|ALLOW_READY|PLACED |stale fill|GTT modify|partial exit|tick error'

touch "$OUT"
echo "=== monitor started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$OUT"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  lines="$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=15 "$HOST" \
    "sudo journalctl -u algofeast-api --since '45 sec ago' --no-pager 2>/dev/null" \
    | grep -iE "$PATTERN" || true)" 2>/dev/null || lines=""
  if [[ -z "$lines" ]]; then
    : # SSH quiet fail — retry next cycle
  elif [[ -n "$lines" ]]; then
    {
      echo "--- $ts ---"
      echo "$lines"
    } >> "$OUT"
  fi
  # heartbeat every ~5 min
  minute=$((10#$(date +%M) % 5))
  sec=$((10#$(date +%S) % 60))
  if [[ "$minute" -eq 0 && "$sec" -lt 35 ]]; then
    status="$(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$HOST" \
      "grep -m1 armed /home/ec2-user/algofeast-workspace/algofeast-api/data/v2_strategy_watch.json 2>/dev/null | head -c 80" \
      2>/dev/null || echo "ssh-unavailable")"
    echo "[heartbeat $ts] $status" >> "$OUT"
  fi
  sleep 30
done
