#!/bin/bash
set -uo pipefail

ENV_FILE="$HOME/rag-fastapi/.env"
LOG_FILE="$HOME/weekly_checkin.log"
GCP_PROJECT="rag-mcp-agent-prod"
GCP_KEY_FILE="$HOME/.gcp/monitoring-reader-key.json"
WINDOW_START="2026-07-13T00:00:00Z"

if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
NOW="$TS"

{
echo "===== check-in $TS ====="

LIVE_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:8000/health/live)
READY_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:8000/health/ready)
echo "health/live http=$LIVE_CODE  health/ready http=$READY_CODE"

gcloud auth activate-service-account --key-file="$GCP_KEY_FILE" >/dev/null 2>&1
gcloud config set project "$GCP_PROJECT" >/dev/null 2>&1
TOKEN=$(gcloud auth print-access-token 2>/dev/null)

if [ -z "$TOKEN" ]; then
  echo "gcp uptime: ERROR could not obtain access token"
else
  for CHECK in "rag-mcp-agent-liveness-Rg3RHb3YdoU:live" "rag-mcp-agent-readiness-lpbOzdlQhuc:ready"; do
    CHECK_ID="${CHECK%%:*}"
    LABEL="${CHECK##*:}"
    PCT=$(curl -s -G "https://monitoring.googleapis.com/v3/projects/${GCP_PROJECT}/timeSeries" \
      -H "Authorization: Bearer $TOKEN" \
      --data-urlencode "filter=metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.labels.check_id=\"${CHECK_ID}\"" \
      --data-urlencode "interval.startTime=${WINDOW_START}" \
      --data-urlencode "interval.endTime=${NOW}" \
      --data-urlencode "aggregation.alignmentPeriod=5184000s" \
      --data-urlencode "aggregation.perSeriesAligner=ALIGN_FRACTION_TRUE" \
      --data-urlencode "aggregation.crossSeriesReducer=REDUCE_MEAN" \
      | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    v = d['timeSeries'][0]['points'][0]['value']['doubleValue']
    print(f'{v*100:.3f}')
except Exception as e:
    print(f'ERROR:{e}')
")
    echo "gcp uptime ($LABEL) since window start: ${PCT}%"
  done
fi

if [ -z "${UPTIMEROBOT_API_KEY:-}" ]; then
  echo "uptimerobot: ERROR no API key configured"
else
  curl -s -X POST "https://api.uptimerobot.com/v2/getMonitors" \
    -d "api_key=${UPTIMEROBOT_API_KEY}" \
    -d "format=json" \
    -d "custom_uptime_ratios=1-7-30" \
    | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for m in d.get('monitors', []):
        if '104.198.167.39' in m.get('url', ''):
            print(f\"uptimerobot: {m['friendly_name']} -> 1d/7d/30d = {m['custom_uptime_ratio']}%\")
except Exception as e:
    print(f'uptimerobot: ERROR {e}')
"
fi

echo ""
} >> "$LOG_FILE"
