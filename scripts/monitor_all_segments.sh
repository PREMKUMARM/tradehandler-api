#!/usr/bin/env bash
# Poll EC2 algofeast-api logs — all segments: nifty50, sensex, commodity.
set -u
KEY="${VIBEFNO_KEY:-/Users/premkumar/Documents/vibefno.pem}"
HOST="${VIBEFNO_HOST:-ec2-user@ec2-3-108-61-102.ap-south-1.compute.amazonaws.com}"
OUT="${TRADE_WATCH_LOG:-/Users/premkumar/Documents/tradehandler-ai-workspace/.trade-watch-alerts.log}"
WORKDIR="${TRADE_WATCH_WORKDIR:-/Users/premkumar/Documents/tradehandler-ai-workspace}"
API_DIR="/home/ec2-user/algofeast-workspace/algofeast-api"

# High-signal events across all three segments
EVENT_PATTERN='GateAudit:(nifty50|sensex|commodity)|\[V2Watch\]|\[SensexWatch\]|\[CommodityWatch\]|V2 LIMIT entry|Commodity LIMIT entry|V2_ORDER_RESULT|COMMODITY_ORDER_RESULT|auto_placed|auto_gtt|auto_stale|auto_skipped|auto_cancelled|stale fill abort|ExitTrailMonitor|TrailOps|Step [0-9]R|ALLOW_AUTONOMOUS|ALLOW_READY|BLOCKED_GATE|GateAudit:.* PLACED|partial exit|tick error|Orphan GTT|GTT attach'

SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o ServerAliveInterval=20)

ssh_cmd() {
  ssh "${SSH_OPTS[@]}" "$HOST" "$@" 2>/dev/null
}

touch "$OUT"
echo "=== all-segment monitor started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$OUT"

cycle=0
while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cycle=$((cycle + 1))

  raw="$(ssh_cmd "sudo journalctl -u algofeast-api --since '50 sec ago' --no-pager 2>/dev/null" || true)"
  if [[ -n "$raw" ]]; then
    lines="$(echo "$raw" | grep -iE "$EVENT_PATTERN" || true)"
    if [[ -n "$lines" ]]; then
      {
        echo "--- $ts ---"
        # Tag lines by segment when possible
        while IFS= read -r ln; do
          seg="other"
          if echo "$ln" | grep -qiE 'GateAudit:nifty50|V2Watch|V2 LIMIT|V2_ORDER'; then seg="nifty50"
          elif echo "$ln" | grep -qiE 'GateAudit:sensex|SensexWatch'; then seg="sensex"
          elif echo "$ln" | grep -qiE 'GateAudit:commodity|CommodityWatch|Commodity LIMIT|COMMODITY_ORDER'; then seg="commodity"
          elif echo "$ln" | grep -qiE 'ExitTrail|TrailOps'; then seg="trail"
          fi
          echo "[$seg] $ln"
        done <<< "$lines"
      } >> "$OUT"
    fi
  else
    if (( cycle % 10 == 0 )); then
      echo "[heartbeat $ts] ssh-unavailable (retrying)" >> "$OUT"
    fi
  fi

  # Every ~5 min: armed state for all 3 watches
  if (( cycle % 10 == 0 )); then
    snap="$(ssh_cmd "cd $API_DIR && for f in data/v2_strategy_watch.json data/sensex_strategy_watch.json data/commodity_strategy_watch.json; do
      seg=\$(basename \"\$f\" _strategy_watch.json)
      if [ -f \"\$f\" ]; then
        armed=\$(python3 -c \"import json; d=json.load(open('\$f')); print(d.get('armed', False))\" 2>/dev/null || echo '?')
        placed=\$(python3 -c \"import json; d=json.load(open('\$f')); print(d.get('placed_symbol_today') or '-')\" 2>/dev/null || echo '-')
        echo \"\$seg armed=\$armed placed=\$placed\"
      else
        echo \"\$seg missing\"
      fi
    done" || echo "ssh-unavailable")"
    echo "[status $ts]" >> "$OUT"
    echo "$snap" | while IFS= read -r ln; do echo "  $ln"; done >> "$OUT"
  fi

  sleep 30
done
