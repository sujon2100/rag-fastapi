# Monitoring log: 30-day evidence window

Official monitoring window start: 2026-07-13 (GCP Cloud Monitoring checks
and UptimeRobot monitors both created that day).

This is the point from which uptime numbers should be treated as citable.
Everything before it was deployment and bug-fixing work (see "Issues found
and fixed during deployment" in RUNBOOK.md) and should be excluded from the
headline uptime figure in the final writeup - the e2-small crashes and the
resize to e2-medium happened before the window opened.

Target end of window: 2026-08-12 or later, minimum 30 days.

## Where to pull current numbers

- GCP Cloud Monitoring: https://console.cloud.google.com/monitoring/uptime?project=rag-mcp-agent-prod
- UptimeRobot public status page (shared with ai-platform-builder, look for
  the monitors named "RAG-MCP-Agent — Liveness" and
  "RAG-MCP-Agent — Readiness" specifically, not the "AI Platform Builder
  Gateway" ones on the same page): https://stats.uptimerobot.com/2JUsdtF71z
- Public endpoint directly: http://104.198.167.39:8000/health/live and
  /health/ready

This project has no synthetic traffic generator, unlike ai-platform-builder
- only the real uptime-check traffic from GCP and UptimeRobot themselves
hits the deployment. That's a deliberate choice, not an oversight; this log
should not be read as tracking a "requests processed" figure the way
ai-platform-builder's does.

## Check-in log

2026-07-13: monitoring window just started, right after deployment. The VM
was resized from e2-small to e2-medium after two reproducible crashes under
real agent traffic, root-caused to OOM with zero swap (see RUNBOOK.md). One
real alert already fired and self-resolved the same day: a readiness check
failure caused by Debian's unattended-upgrades triggering an automatic
reboot, recovered automatically by the restart: unless-stopped policy
within minutes, no manual intervention needed. Baseline established, no
cumulative uptime percentage to report yet.
