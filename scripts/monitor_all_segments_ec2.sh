#!/usr/bin/env bash
# Run ON EC2 — tails algofeast-api for all segments (nifty50, sensex, commodity).
# Usage: nohup ./scripts/monitor_all_segments_ec2.sh >> /tmp/segment-events.log 2>&1 &
set -u
OUT="${SEGMENT_EVENTS_LOG:-/tmp/segment-events.log}"
PATTERN='GateAudit:(nifty50|sensex|commodity)|\[V2Watch\]|\[SensexWatch\]|\[CommodityWatch\]|V2 LIMIT entry|Commodity LIMIT entry|V2_ORDER_RESULT|COMMODITY_ORDER_RESULT|auto_placed|auto_gtt|auto_stale|auto_skipped|auto_cancelled|stale fill abort|ExitTrailMonitor|TrailOps|Step [0-9]R|ALLOW_AUTONOMOUS|ALLOW_READY|BLOCKED_GATE| PLACED |partial exit|Orphan GTT|GTT attach'

tag_line() {
  local ln="$1" seg="other"
  if echo "$ln" | grep -qiE 'GateAudit:nifty50|V2Watch|V2 LIMIT|V2_ORDER'; then seg="nifty50"
  elif echo "$ln" | grep -qiE 'GateAudit:sensex|SensexWatch'; then seg="sensex"
  elif echo "$ln" | grep -qiE 'GateAudit:commodity|CommodityWatch|Commodity LIMIT|COMMODITY_ORDER'; then seg="commodity"
  elif echo "$ln" | grep -qiE 'ExitTrail|TrailOps'; then seg="trail"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [$seg] $ln"
}

echo "=== EC2 segment monitor started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$OUT"

sudo journalctl -u algofeast-api -f --no-pager 2>/dev/null | while IFS= read -r line; do
  if echo "$line" | grep -qiE "$PATTERN"; then
    tag_line "$line" >> "$OUT"
  fi
done
