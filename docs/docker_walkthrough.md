# Docker Compose Walkthrough (Phase 12)

The full deployable app: `db` (Postgres) + `backend` (FastAPI) + `frontend` (static dashboard, nginx), wired together with `docker-compose.yml`. Reuses the exact same enforcement pipeline every earlier phase built (`Classifier`, `PolicyEngine`, `EnforcementEngine`, `Interceptor`) — this phase's only new code is the HTTP surface (`src/toolwarden/service/app.py`) and Postgres-backed drop-ins for Phase 6's `ApprovalQueue` and Phase 2's `LogSink` (`src/toolwarden/service/postgres_approval_queue.py`, `postgres_log_sink.py`).

## Running it

```
export TOOLWARDEN_MODEL_DIR="D:\CyberSecurity\Projects\Apps and Models\ToolWarden"   # or your own model directory
docker compose up -d --build
```

- Backend API: `http://localhost:8000`
- Dashboard: `http://localhost:8080`

`TOOLWARDEN_MODEL_DIR` must point at a directory with the layout described in the README's Setup section (DeBERTa checkpoint, LightGBM booster, `ensemble_stacker.json`) — it's bind-mounted **read-only** into the backend container at `/models`, never baked into the image. This is the storage-split rule (model weights vs. everything else) applied to the container: the image stays small and reproducible, and swapping which trained model is deployed never requires a rebuild.

## Why a real service, not just the pip library wrapped in a container

The most significant design change from every earlier adapter (`guarded_loop.py`'s OpenAI loop, `mcp_proxy/server.py`'s MCP server): **a HOLD can't block an HTTP request waiting for a human.** Phase 9 and 10's demos use a synchronous `on_hold()` callback because they're single Python processes where a human (or a scripted stand-in) can answer inline. There's no human on the other end of a REST call. So here, a HOLD returns immediately:

```json
{"decision": "hold", "reason": "score 0.513 >= hold_threshold 0.5", "score": 0.513, "pending_id": "eb42d9cb-..."}
```

...and the pending item sits in Postgres until a reviewer resolves it via the dashboard (or `POST /v1/approvals/{id}/resolve` directly). This is the honest shape of a real async approval workflow — not a simplification of the demo's callback, a different and more realistic pattern the demo's design couldn't show.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Container healthcheck |
| `POST` | `/v1/tool-calls/request` | Intercept + classify + enforce an outbound request (agent → tool) |
| `POST` | `/v1/tool-calls/result` | Intercept + classify + enforce an inbound result (tool → agent) |
| `GET` | `/v1/approvals/pending` | List items awaiting human review |
| `POST` | `/v1/approvals/{id}/resolve` | Attributed approve/deny — `decided_by` is required, matching Phase 6's audit rule |
| `GET` | `/v1/traffic?limit=50` | Recent intercepted requests/results, newest first |

`detector_mode` (Phase 11: `"ensemble"` or `"deberta_only"`) is set via the `TOOLWARDEN_DETECTOR_MODE` environment variable at container startup, not per-request — the classifier and its loaded weights are shared across all requests in one process.

## What's new vs. reused

**New this phase:**
- `src/toolwarden/service/schema.sql` — Postgres tables for pending/resolved approvals and traffic events.
- `postgres_approval_queue.py` / `postgres_log_sink.py` — duck-type-compatible with Phase 6's `ApprovalQueue` and Phase 2's `LogSink` protocol. `EnforcementEngine` and `Interceptor` run completely unmodified against them — same interface, different backing store.
- `app.py` — the FastAPI surface itself.
- `frontend/` — a vanilla HTML/CSS/JS dashboard (deliberately no Node/npm build step) showing pending approvals and recent traffic, served by nginx, which also reverse-proxies `/api/*` to the backend so the browser only ever talks to one origin (no CORS needed).

**Reused, not rebuilt:** `Classifier`, `PolicyEngine`, `EnforcementEngine`, `Interceptor`, and the `ApprovalRecord`/`PendingApproval`/`ApprovalDecision` dataclasses — all imported directly from the pieces Phase 2, 4, 6, 8, and 11 already built. `resolution_to_decision()` (approval → Decision mapping) was pulled out of `ApprovalQueue` into a shared free function specifically so the Postgres and in-memory implementations can't drift on what "denied" means for a REQUEST vs. a RESULT.

## Real bugs the literal verification caught

Per the lesson from Phase 11's stop gate (verify literally, not by approximation): every piece here was run for real — a real Postgres container, a real `docker build`, real containers started via `docker compose up`, real HTTP requests against the running stack — before this phase was called done. That caught three bugs that authoring-without-running would have shipped invisibly:

1. **`schema.sql` wasn't in the installed package.** `pip install .` doesn't bundle non-`.py` files by default; the backend container crashed on startup with a `FileNotFoundError` for the schema file that plainly existed in the source tree. Fixed via `[tool.setuptools.package-data]` in `pyproject.toml`.
2. **LightGBM needs `libgomp1`**, not present in the `python:3.13-slim` base image — `OSError: libgomp.so.1: cannot open shared object file`. Fixed by installing it via `apt-get` in the Dockerfile. A well-known Docker+LightGBM gotcha in hindsight, invisible until an actual container was run.
3. **A malformed `pending_id` crashed the resolve endpoint with a raw 500** (`psycopg.errors.InvalidTextRepresentation`) instead of a clean 404 — the `id` column is UUID-typed, and Postgres rejects a non-UUID string at the query level before the "not found" logic ever runs. Fixed by validating the id's format in Python first (`PostgresApprovalQueue.resolve`), so a malformed id and a well-formed-but-missing id both surface as the same `UnknownApprovalError` → 404, matching how a real caller should experience "no such approval" regardless of *why* it doesn't exist.

A fourth bug surfaced during the GPU checkpoint specifically — see below, kept separate because it's a different *kind* of gap (a feature silently not doing what its name implied, not a packaging/environment mismatch).

None of these would have been caught by unit tests with mocked dependencies, or by reading the code carefully — they only exist at the boundary between the code and its real packaging/runtime environment, which is exactly what this phase's Docker verification is for.

## GPU passthrough (its own checkpoint)

The default `docker-compose.yml` is CPU-only by design — `docker/backend.Dockerfile` installs the same CPU torch wheel `pip install toolwarden-ai` resolves (see the README's GPU vs. CPU note), so the stack works out of the box on any machine with Docker, no NVIDIA setup required.

GPU acceleration is layered on top via `docker-compose.gpu.yml` + `docker/backend.gpu.Dockerfile` (which reinstalls a CUDA-matched torch build, `torch==2.13.0+cu132`, matching this repo's own training pin):

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### The fourth bug: "the container can see a GPU" and "inference uses it" are different claims

Verifying `torch.cuda.is_available()` returns `True` inside the GPU-enabled container was not the end of this checkpoint — it was a trap. Checking further (does a real classification request actually run faster / on the GPU, not just "could" it) found that `load_fitted_models()` (`src/toolwarden/classifier/evaluate.py`) loaded the DeBERTa checkpoint via `from_pretrained()` and never moved it off CPU. `torch.cuda.is_available()` being `True` says the *hardware and driver* are reachable; it says nothing about whether any given model tensor actually lives there. The classifier was running on CPU inside the "GPU-enabled" container the whole time, silently.

Fixed by moving the model to CUDA in `load_fitted_models()` — but **deliberately gated behind an explicit `TOOLWARDEN_USE_GPU=1` environment variable, not automatic whenever CUDA is available.** This dev machine has a GPU too, and CPU vs. GPU floating-point kernels are not bit-identical — verified directly: the same attack text's ensemble score was `0.9772014446696929` on CPU and `0.9772014547817592` on GPU, differing at the 7th decimal place. Every score published anywhere in this project so far (`docs/classifier_report.md`, `docs/degradation_curve_report.md`, every specific decimal cited in `docs/known_limitations.md`) was computed on CPU, since nothing moved the eval model to GPU before this fix existed. Making GPU use automatic would have silently started perturbing those numbers on this machine's next rerun of anything — Phase 8's benchmark, the demos, this project's own test suite — the moment this phase's Docker work touched shared inference code. `docker-compose.gpu.yml` sets `TOOLWARDEN_USE_GPU=1` for the backend service explicitly; nothing else changes behavior by default.

Verified end to end through the real running container, not just the fix in isolation: `TOOLWARDEN_USE_GPU=1` set via compose → `next(model.parameters()).device` reads `cuda:0` inside `toolwarden-backend-1` → a real `POST /v1/tool-calls/result` through the container's actual exposed port returns the GPU-specific score above, matching the host-side GPU verification exactly (same GPU, same torch build, same kernels).

Verified on this project's dev machine (Docker Desktop, WSL2 backend, RTX 2000 Ada, driver 595.95 / CUDA 13.2): confirmed `docker run --gpus all` exposes the host GPU inside a container at all (`nvidia-smi` runs correctly from inside a throwaway `nvidia/cuda` container) before building the GPU backend image on top of that, then confirmed `torch.cuda.is_available()` is `True` and a real classification call runs on the GPU inside the actual `backend` service container — not just that the image built.

If `docker run --gpus all ... nvidia-smi` fails on a different machine, that's a Docker/WSL2/NVIDIA Container Toolkit host-configuration issue to resolve before this compose override will work — outside what a Dockerfile or compose file alone can fix. The [NVIDIA Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) cover host setup; on Windows with Docker Desktop, GPU support ships with the WSL2 backend and generally needs no separate toolkit install beyond a current NVIDIA driver.
