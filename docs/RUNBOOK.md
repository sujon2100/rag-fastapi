# Runbook: local testing and what broke along the way

## Part 1: local, no Docker

### Prerequisites

- Python 3.11 (`python3.11 --version`)
- Ollama installed and running (`ollama serve`, or the menu-bar app)
- A real Pinecone API key (free tier is fine) - `PINECONE_API_KEY=your_key_here`
  in `.env` is a placeholder and will fail fast with
  `PineconeConfigurationError` if left as-is

### Setup

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit in a real PINECONE_API_KEY
ollama pull qwen2.5:1.5b # used by both /rag/query and the agent
```

### Run it

```bash
# terminal 1: the MCP tool server
python -m app.mcp.server

# terminal 2: the FastAPI app
uvicorn app.main:app --reload
```

### Smoke test

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready   # actually checks Pinecone + Ollama

curl -X POST http://localhost:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "anything"}'

curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "ingest a document with source test and text: the sky is blue", "role": "ingest_agent"}'

curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "search the documents for sky", "role": "reader"}'
```

### Tests

```bash
pytest tests/                     # authz + tool-routing logic, no external deps, ~15s
pytest tests/ --run-integration   # spins up a real MCP server subprocess,
                                   # hits real Pinecone and real Ollama
```

## Part 2: Docker Compose

```bash
cp .env.example .env   # real PINECONE_API_KEY
docker compose up -d ollama
docker exec -it $(docker compose ps -q ollama) ollama pull qwen2.5:1.5b
docker compose up -d mcp-server api
```

First start of `api` and `mcp-server` is slow - both import `torch` and
`sentence-transformers` at module load time (for the embedder), and on
Docker Desktop for Mac that took 2.5-5 minutes cold in testing here, versus
about 20 seconds for the same import running directly on the host. That
gap is almost certainly Docker Desktop's macOS virtualization layer being
slow to cold-read the large `.so` files those packages ship - a plain
Linux host (which is what the actual cloud deployment target is) should
not have the same penalty, but it hasn't been measured there yet. Poll
`/health/live` rather than assuming a fixed wait:

```bash
until curl -sf http://localhost:8000/health/live >/dev/null; do sleep 5; done
```

### Tear down

```bash
docker compose down          # containers + network, keeps the ollama volume
docker compose down -v       # also wipes the pulled models
```

## Issues found and fixed while building this

Worth keeping in the record rather than glossing over.

1. **`.env` was never actually loaded.** `app/services/vectorstore.py` read
   `PINECONE_API_KEY` with plain `os.getenv()` - there was no
   `load_dotenv()` call anywhere in the codebase. The key in `.env` was
   silently ignored; `os.getenv("PINECONE_API_KEY")` returned `None`, not
   even the placeholder string. Fixed by adding `load_dotenv()` to the top
   of `vectorstore.py`.

2. **The `.gitignore` had never actually worked.** The tracked file was
   literally named `.gitignore  ` - two trailing spaces baked into the
   filename from whenever it was first created. Git only recognizes a file
   named exactly `.gitignore`; `git check-ignore` confirmed nothing was
   being ignored at all. `.env` had been tracked in git since the very
   first commit as a result (checked the full history - only the
   placeholder key was ever committed, no real secret). Fixed by renaming
   the file and running `git rm --cached .env`.

3. **Hardcoded `localhost:11434` for Ollama.** `app/services/llm_inference.py`
   had `OLLAMA_URL = "http://localhost:11434/api/generate"` hardcoded,
   which resolves fine on a host machine but not inside a Docker container
   where Ollama runs as a separate service (`ollama`, not `localhost`).
   Fixed by reading `OLLAMA_BASE_URL` from the environment, same variable
   name that was already sitting unused in `.env`.

4. **`llama3` does not support Ollama's tool-calling API.** The agent's
   first real run against real Ollama failed with
   `ollama._types.ResponseError: registry.ollama.ai/library/llama3:latest
   does not support tools (status code: 400)`. Not every model on Ollama
   implements the tools/function-calling API; `llama3` (the base model)
   doesn't. Switched the agent specifically to `qwen2.5`, initially
   leaving `/rag/query`'s plain-generation path on `llama3` since it never
   calls tools and didn't strictly need to change. Revisited before
   deployment (see item 8 below) and switched everywhere.

5. **`qwen2.5:0.5b` was too small to reliably fill in both required fields**
   on the two-argument `ingest_document` tool call, in testing here - it
   would sometimes call the tool with a missing or malformed argument. The
   underlying tool and MCP wiring were confirmed correct by calling them
   directly, bypassing the LLM. `qwen2.5:1.5b` handled the same prompts
   correctly in every test run performed. This is a real, honest limit of
   running small models for tool use on CPU, not a code bug - later
   testing also saw the 1.5b model occasionally answer a search question
   without actually calling `search_documents`, so this is a "mostly
   reliable," not "always reliable," part of the system, and that's worth
   stating plainly rather than papering over.

6. **Docker: PID 1 zombie process, container couldn't be stopped or
   recreated.** Without an init process, `uvicorn` as PID 1 didn't reap
   child processes correctly on shutdown, leaving a zombie that
   `docker stop`/`docker kill` couldn't touch (`cannot kill container:
   ... PID ... is zombie`). Needed `docker rm -f` to clear it out. Fixed
   by adding `init: true` to both the `api` and `mcp-server` services in
   `docker-compose.yml`, which runs `tini` as PID 1 instead.

7. **MCP tool results are a list of content blocks, not a plain string.**
   `ainvoke()` on a `langchain_mcp_adapters` tool returns something like
   `[{"type": "text", "text": "...", "id": "..."}]`, not `str`. The first
   version of both the agent's `tools_node` and the integration test
   assumed a plain string and either fed the LLM a noisy Python repr of
   that list (which likely made the small-model reliability problem in
   item 5 worse) or failed a test assertion outright. Fixed in both
   places by extracting the actual text out of the content blocks before
   using it.

8. **Running two different Ollama models risked exceeding a small
   deployment VM's memory.** `/rag/query` was left on `llama3` (a 4.7GB
   model) after item 4, while the agent used `qwen2.5:1.5b`. Ollama keeps
   a model resident in memory for a keep-alive window after use, so if
   both endpoints got hit within that window, both models could end up
   loaded simultaneously - `ai-platform-builder` specifically picked a
   small model (`qwen2.5:0.5b`) to fit on an `e2-small` (2GB RAM), and
   `llama3` alone would already blow past that. Caught before deployment,
   not after - switched `/rag/query` to `qwen2.5:1.5b` too, so the whole
   system now runs on one small model instead of two.

## Part 3: cloud deployment

Live on GCP since 2026-07-13, in its own dedicated project
(`rag-mcp-agent-prod`) rather than reusing `ai-platform-eb2-demo` - a
deliberate choice to keep this as a genuinely separate, independently
verifiable deployment rather than two services riding on one project's
identity.

- Public endpoint: `http://104.198.167.39:8000`
- Project: `rag-mcp-agent-prod` (account sujon2100@gmail.com), billed
  against the same billing account as `ai-platform-eb2-demo`
  (`0156C5-886573-BA959C`)
- VM: `rag-mcp-agent-vm`, zone `us-central1-a`, machine type `e2-medium`
  (see item 10 below for why not `e2-small`)
- Static IP: `rag-mcp-agent-ip`, reserved so the address doesn't change
  on restart
- Boot disk: 50GB `pd-standard` (see item 9 below for why not 30GB)

### One-time account setup

Same billing account as `ai-platform-builder` - no new trial needed.
Before provisioning anything, checked the actual remaining balance in the
console (Billing -> Overview) rather than assuming: **$298.91 of $300
credit remaining, 88 days left (ends 2026-10-08), still on free trial,
not upgraded to a paid account** - confirmed as of 2026-07-13. On a free
trial, GCP does not auto-charge if credit runs out; it suspends resources
instead, so there was no risk of surprise billing before starting.

```bash
gcloud projects create rag-mcp-agent-prod --name="rag-mcp-agent-prod"
gcloud billing projects link rag-mcp-agent-prod --billing-account=0156C5-886573-BA959C
gcloud config set project rag-mcp-agent-prod
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
gcloud services enable compute.googleapis.com monitoring.googleapis.com billingbudgets.googleapis.com
```

### Provisioning

```bash
gcloud compute addresses create rag-mcp-agent-ip --region=us-central1

gcloud compute firewall-rules create allow-ssh-iap \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:22 --source-ranges=35.235.240.0/20 \
  --target-tags=rag-mcp-agent-vm

gcloud compute firewall-rules create allow-gateway-http \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:8000 --source-ranges=0.0.0.0/0 \
  --target-tags=rag-mcp-agent-vm

gcloud compute instances create rag-mcp-agent-vm \
  --zone=us-central1-a --machine-type=e2-medium \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-standard \
  --address=rag-mcp-agent-ip --tags=rag-mcp-agent-vm \
  --metadata-from-file=startup-script=infra/gcp/startup-script.sh
```

`infra/gcp/startup-script.sh` installs Docker CE, the compose plugin, and
`at`/`atd` on first boot - same script `ai-platform-builder` uses.

### Connecting to the VM

```bash
gcloud compute ssh rag-mcp-agent-vm --zone=us-central1-a --tunnel-through-iap
```

### Deploying

```bash
tar --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='venv311' --exclude='.env' -czf /tmp/rag-fastapi.tar.gz .
gcloud compute scp /tmp/rag-fastapi.tar.gz rag-mcp-agent-vm:/tmp/ --tunnel-through-iap

# on the VM:
mkdir -p ~/rag-fastapi
tar -xzf /tmp/rag-fastapi.tar.gz -C ~/rag-fastapi
# write a real .env by hand on the VM - never scp'd, never in git:
#   PINECONE_API_KEY=<real key>
#   OLLAMA_BASE_URL=http://ollama:11434
cd ~/rag-fastapi
sudo docker compose up -d ollama
sudo docker exec rag-fastapi-ollama-1 ollama pull qwen2.5:1.5b
sudo docker compose build api
sudo docker compose up -d mcp-server api
```

### Monitoring

Two layers, same pattern as `ai-platform-builder`:

1. GCP Cloud Monitoring: uptime checks `rag-mcp-agent-liveness` and
   `rag-mcp-agent-readiness` hit `/health/live` and `/health/ready` every
   5 minutes, each with an alert policy emailing sujon2100@gmail.com on
   failure.
2. UptimeRobot (independent third party): two new monitors,
   `RAG-MCP-Agent — Liveness` and `RAG-MCP-Agent — Readiness`, created
   manually via the UptimeRobot console (their API rejects monitor
   *creation* on the free plan with a generic "not allowed to use some
   settings" error, even though read-only API calls work fine - tried
   several parameter combinations before concluding it's a plan
   restriction, not a malformed request). Original plan was a fully
   separate status page from `ai-platform-builder`'s, but the free plan
   only allows one public status page per account, so both projects now
   share the same one: `https://stats.uptimerobot.com/2JUsdtF71z`,
   renamed from "AI Platform Gateway" to "Helal Uddin — Deployments" to
   reflect that. Separation between the two projects is by monitor
   naming instead of by URL - all four monitors are clearly prefixed
   (`AI Platform Builder Gateway — ...` / `RAG-MCP-Agent — ...`), so it's
   unambiguous which entries belong to which deployment even on one
   shared page.

Monitoring window started **2026-07-13**. Same rule as
`ai-platform-builder`: no uptime percentage is citable evidence until a
real amount of time has actually passed.

### Monitoring log

- **2026-07-13 22:43 UTC - readiness alert fired, self-resolved within
  ~15 minutes.** Investigated rather than assumed benign: `dmesg` showed
  no OOM-killer activity, but the systemd journal showed a fresh boot at
  22:32-22:33 UTC with `unattended-upgrades.service - Unattended Upgrades
  Shutdown` immediately before it - Debian's automatic security-update
  reboot, not an application crash. `restart: unless-stopped` brought all
  three containers back up with no manual intervention; the readiness
  check failed for a few minutes during the cold `torch`/Ollama reload
  that follows any restart (same delay documented in Part 2), then
  cleared on its own once warm. Confirmed healthy afterward by re-running
  both health checks manually. Left `unattended-upgrades`'s auto-reboot
  enabled rather than disabling it - the occasional brief, self-healing
  blip this causes is a more honest signal than silencing security
  patching to keep the uptime number clean.

### Budget alerts

Set on the shared billing account (not scoped to a single project), so
one alert path covers both this deployment and `ai-platform-builder`'s:

```bash
gcloud billing budgets create \
  --billing-account=0156C5-886573-BA959C \
  --display-name="Trial credit guardrail" \
  --budget-amount=300USD \
  --threshold-rule=percent=0.1667 \
  --threshold-rule=percent=0.3333 \
  --threshold-rule=percent=0.6667 \
  --threshold-rule=percent=1.0
```

Fires at roughly $50, $100, $200, and $300 of the $300 trial credit,
emailing the billing account's default IAM recipients.

### Issues found and fixed during deployment (2026-07-13)

Worth keeping in the record rather than glossing over - this deployment
surfaced more real problems than the local Docker Compose testing did,
because a small cloud VM behaves differently from a Mac with Docker
Desktop.

9. **The 30GB boot disk ran out of space mid-build and the build
   failed.** `docker compose build mcp-server api` built the identical
   Dockerfile twice under two different image tags (`rag-fastapi-api` and
   `rag-fastapi-mcp-server`), each unpacking its own ~3GB copy of
   torch/sentence-transformers - so the build needed roughly double the
   disk it actually should have. It failed partway through with
   `write ...: no space left on device` while extracting `triton`'s
   shared library. Fixed two ways: changed `docker-compose.yml` so `api`
   builds and tags a single `rag-fastapi:latest` image and `mcp-server`
   just references that same tag instead of building its own copy (halved
   both the build time and the disk footprint), and resized the boot disk
   from 30GB to 50GB for headroom (`gcloud compute disks resize`, then
   `growpart`/`resize2fs` on the VM - `growpart` isn't installed by
   default on Debian, needed `apt-get install cloud-guest-utils` first).

10. **`e2-small` (2GB RAM, no swap) crashed under real traffic - twice.**
    Health checks and even the first `/rag/query` test passed fine on
    `e2-small`, but the very first real `/agent/query` request (which
    runs `mcp-server`'s embedding step and the agent's Ollama tool-calling
    step at the same time) made the entire VM unreachable - not just the
    containers, but SSH and even the guest OS's own network stack
    (`gcloud compute ssh` failed with `failed to connect to backend`, and
    the serial console showed the guest stuck retrying
    `dial tcp 169.254.169.254:80: connect: network is unreachable`,
    consistent with an OOM event taking out something more than just the
    offending container). A `gcloud compute instances reset` brought it
    back, but the exact same request crashed it again within seconds -
    confirmed reproducible, not a one-off. With zero swap on a 2GB
    instance, running `mcp-server`'s embedding model and Ollama loading
    `qwen2.5:1.5b` at the same time had no room to fail gracefully. Fixed
    by resizing to `e2-medium` (4GB RAM): `gcloud compute instances stop`,
    `gcloud compute instances set-machine-type --machine-type=e2-medium`,
    `gcloud compute instances start`. Confirmed fixed by re-running the
    exact request that crashed the VM twice under `e2-small` - it
    succeeded, and the VM stayed reachable afterward. Two cheaper
    alternatives considered and not taken here: adding swap (risked very
    slow responses given the disk is `pd-standard`, not SSD) and merging
    `api`/`mcp-server` into one process to stop loading the embedding
    model twice (a real fix, but a bigger refactor than the deployment
    timeline called for - worth doing later).

11. **No restart policy meant a crash required manual recovery.** The
    original `docker-compose.yml` had no `restart:` key on any service,
    so when the VM reset killed all three containers, they stayed dead
    until someone ran `docker compose up -d` by hand. Added
    `restart: unless-stopped` to all three services. Confirmed working:
    after the `e2-medium` resize and reboot, all three containers came
    back on their own with no manual intervention.
