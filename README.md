# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

**Phase 13 (testing/validation audit) complete.** Full test suite reran fresh (116/116, then 131/131 once the follow-up work below added more), both auto-generated reports (`docs/classifier_report.md`, `docs/degradation_curve_report.md`) reran from scratch and came back byte-identical to what's committed. Audit pass found and fixed six real documentation issues (undercounted bug counts, a stale model-directory claim, a dead link, an unclosed cross-reference, threat-model claims written before Phase 12 existed) — see [docs/known_limitations.md](docs/known_limitations.md) for the full list. One larger finding: `docs/architecture.md` described explainability (SHAP + attention) as attached to every flagged decision, but grepping all three live enforcement paths found it was never actually wired in past Phase 5's offline spot-check — **now fixed and verified live** (see below).

**Explainability wired into the live decision path.** `Classifier.score_and_explain()` computes score and SHAP/attention evidence together; `EnforcementEngine` attaches it to the shared decision object at one point so all three adapters (OpenAI, MCP, the FastAPI service) inherit it without three separate integrations. Surfaced wherever a HOLD is already shown — demo/MCP console output, the Phase 12 dashboard's approvals table and resolve dialog. Verified live across all three: real classifier scores plus real, sensible explanations reproduced through the OpenAI demo, the MCP demo, and a real `docker compose up` stack's actual HTTP responses. Caught one more real packaging bug along the way (`shap` was training-only in `pyproject.toml`, but the live service needs it now — moved to base dependencies).

**The Anthropic adapter is built and unit-tested, but explicitly NOT verified against the real API — do not read this as "working."** `src/toolwarden/demo/guarded_loop_anthropic.py`'s `GuardedAnthropicToolLoop` closes a real gap the Phase 13 audit found: every phase up to this point only ever exercised OpenAI, despite "OpenAI/Anthropic tool_use" being the original locked scope. It reuses `Classifier`/`PolicyEngine`/`EnforcementEngine`/`ApprovalQueue` completely unmodified, same shape as the MCP adapter. 7/7 unit tests pass against a scripted fake Anthropic client (`tests/unit/test_guarded_loop_anthropic.py`), including Claude's multi-tool-call-per-turn batching behavior. But no live call has ever succeeded: the only key available (an AWS Bedrock "Claude Platform on AWS" key, a different auth system from `console.anthropic.com`'s direct API) was rejected by AWS's own gateway with a genuine format error, not a bug in this project's code. Sagar chose to proceed with the OpenAI-only human-testing round rather than chase a working key right now — full honest status, including exactly what is and isn't verified, in [docs/known_limitations.md](docs/known_limitations.md). See [docs/human_testing_guide.md](docs/human_testing_guide.md) for how to complete this verification later.

**Phase 12 (Docker Compose app) complete** — see [docs/docker_walkthrough.md](docs/docker_walkthrough.md). `db` (Postgres) + `backend` (FastAPI) + `frontend` (a no-build-step dashboard) — same enforcement pipeline as every earlier phase, reused unmodified, now reachable over HTTP instead of only as a Python import. The one real architectural difference: a HOLD can't block an HTTP request waiting for a human the way the demos' synchronous `on_hold()` callback does, so it returns immediately with a `pending_id` and a reviewer resolves it later via the dashboard — the honest shape of an async approval workflow, not a simplified callback. Model weights are bind-mounted into the backend container, never baked into the image; GPU passthrough (`docker-compose.gpu.yml`) is layered on top as its own verified checkpoint, not a blocker for the base stack. Verified literally end to end: real `docker compose up`, real HTTP requests against the running containers, byte-identical classifier scores to every earlier phase's verification. That verification caught four real bugs (a missing `schema.sql` in the installed package, a missing system library LightGBM needs, an unhandled malformed-ID crash, and — the GPU checkpoint's own trap — `torch.cuda.is_available()` being `True` inside the "GPU-enabled" container didn't mean inference actually used the GPU, since nothing ever moved the model off CPU) — all fixed, all documented in the walkthrough rather than smoothed over.

**Phase 11 (pip packaging) complete.** `src/toolwarden/__init__.py` now defines the actual public interface (`Classifier`, `PolicyEngine`, `EnforcementEngine`, `ApprovalQueue`, `Interceptor`, the direct-API adapter, and the supporting types) instead of leaving consumers to reach into internal submodules — importing it is cheap (no torch/transformers/lightgbm import as a side effect; see the module docstring). `pyproject.toml` now declares real runtime dependencies for `pip install toolwarden-ai`, split from `requirements.txt`'s exact training-environment pins, with `mcp`/`dev`/`train` as optional extras.

**Stop gate verified literally, not just described:** built the wheel, installed it into a brand-new venv outside this repo, and ran Phase 9's guarded demo loop using only what that install provides. This caught two real bugs a same-repo test would have missed entirely — both fixed before this phase closed: (1) `load_fitted_models()` was re-fitting the ensemble stacker from this repo's gitignored training dataset on *every* call, which a real install can never have; fixed by persisting the fitted stacker as a small JSON artifact (`EnsembleStacker.save`/`load` in `src/toolwarden/classifier/ensemble.py`) instead of re-deriving it, verified byte-identical to the pre-fix scores. (2) the demo scripts' log paths were computed relative to `__file__`, which resolves into the venv's own `site-packages` tree for an installed run instead of somewhere sensible; fixed to resolve relative to the caller's working directory instead. Full fresh-install run (all three scenarios, real OpenAI calls) reproduced byte-identical classifier scores to the dev-checkout run.

Added `detector_mode` to `Classifier` (`"ensemble"` default, `"deberta_only"` available) — a direct, quantified response to the ensemble-combination ablation below: the fitted stacker weighs LightGBM almost as heavily as DeBERTa (deberta_weight≈3.807, lightgbm_weight≈3.791) despite LightGBM being measurably less reliable on held-out data, which suppresses real attacks DeBERTa alone would have caught (3/35 on AgentDojo, 6/300 on the synthetic set — see [docs/degradation_curve_report.md](docs/degradation_curve_report.md)'s "Ensemble-combination ablation" section). The stacker itself isn't retrained yet — that needs a third calibration split that's never touched either benchmark, scoped as its own task (see [docs/known_limitations.md](docs/known_limitations.md)) — `deberta_only` gives a documented way to opt out of the failure mode today, at the cost of LightGBM's contribution entirely rather than a partial reweighting.

**Phase 10 (MCP proxy layer) complete** — see [docs/mcp_proxy_walkthrough.md](docs/mcp_proxy_walkthrough.md). The same enforcement pipeline from Phase 9 (`Classifier`, `PolicyEngine`, `EnforcementEngine`, `ApprovalQueue`, `Interceptor` — reused, not rebuilt) wired into MCP's `on_call_tool` handler (`src/toolwarden/mcp_proxy/server.py`) instead of an inline OpenAI tool-calling loop. Enforcement now runs server-side: the client-side agent loop (`src/toolwarden/mcp_proxy/agent_loop.py`) never imports the classifier or policy engine directly, only sees what a real MCP client would see. Same three scenarios as Phase 9, run through the second locked integration adapter — proves the detection/enforcement behavior isn't an artifact of one protocol.

**Phase 9 (standalone demo agent) complete** — see [docs/demo_walkthrough.md](docs/demo_walkthrough.md). A research-assistant agent that fetches a webpage and summarizes it, wired through ToolWarden's full pipeline chained into one live agent loop for the first time (`src/toolwarden/demo/guarded_loop.py`): every tool-call request/result is intercepted, scored by the real trained ensemble, policy-decided, and enforced. Verified reliable across two independent live runs.

Building the demo surfaced a real, unplanned finding: small, meaning-preserving rewordings of equally-benign sentences swing the classifier's score by 0.5+ with no consistent explanation (e.g. "The weather in Lisbon is sunny today." scores 0.19; "Lisbon Weather Update: Sunny skies are expected..." scores 0.93). This is sharper than Phase 8's AgentDojo precision (0.27-0.30) alone suggested — closer to poor calibration/overfitting than a bounded error rate. Confirmed with Sagar before proceeding: the demo shows this honestly as a third scenario (a genuinely benign page auto-quarantined) rather than quietly picking content that happens to score well. Full disclosure in [docs/known_limitations.md](docs/known_limitations.md).

The three demo scenarios, each grounded in real verified classifier scores: **ALLOW** (benign page, score ~0.23, task completes normally) — **HOLD** (a page carrying a real held-out-derived injected instruction, score ~0.51, run both approved and denied — approving it results in a real `send_email` call delivering the attacker's link, denying it quarantines the content before the agent ever sees it) — **FALSE POSITIVE** (a second genuinely benign page, score ~0.96, auto-quarantined before any human review, making Phase 8's precision number concrete on one readable example).

Phase 8's degradation curve (recall reported separately per source, every point estimate with a 95% CI):

| Source | N | Ensemble recall (95% CI) |
|---|---|---|
| In-distribution test (InjecAgent) | 680 | 1.000 [0.989, 1.000] |
| Held-out novel (AgentDojo) | 132 | 0.914 [0.776, 0.970] |
| Synthetic adversarial (Qwen3.5-9b-generated, N=300) | 300 | 0.980 [0.957, 0.991] |

Full breakdown, lower-bound caveat, and PINT citation in [docs/degradation_curve_report.md](docs/degradation_curve_report.md). Phase 7's local red-teamer is Qwen3.5-9B (swapped from Qwen2.5-14B-Instruct) — see [docs/redteam_generation_report.md](docs/redteam_generation_report.md).

Phases 1-6 stand — see [docs/](docs/) for threat model, architecture, dataset assembly, classifier training, explainability, and the enforcement engine.

## Scope

- **In scope:** tool-call request/response interception, classification of tool-call scope drift and indirect prompt injection in tool output, configurable enforcement policy, human-approval queue.
- **Out of scope:** semantic content of the agent's own text responses, model internals of closed APIs, direct prompt injection from the user's own turns, tool reliability/correctness, network/infra security, tool supply-chain security. Full detail in [docs/threat-model.md](docs/threat-model.md).

## Differentiation

Enforcement-capable agent security tools already exist commercially. The gap this project targets: a reproducible, honest degradation curve showing accuracy loss under adversarial stress vs. clean benchmark performance, measured against a real baseline (ProtectAI DeBERTa-v3-base). Every number reported here is reproducible and comes with its adversarial-condition context — no unqualified "best-in-class" claims.

## Project layout

```
src/         application and library code
tests/       unit / integration / e2e tests
docs/        architecture, threat model, writeups
datasets/    dataset assembly scripts + (gitignored) data
models/      model config/loading code only — actual weight files live outside this repo
```

Model weight files are stored separately at `D:\CyberSecurity\Projects\Apps and Models\ToolWarden\` and are
never committed to this repo — `classifiers\deberta-v3-base-toolwarden\` (the fine-tuned DeBERTa checkpoint),
`classifiers\lightgbm\` (the trained LightGBM booster and the fitted ensemble stacker), and `llm\` (the local
red-teamer GGUF — both the current Qwen3.5-9B and superseded Qwen2.5-14B-Instruct files are kept there side by
side).

## Setup

**Using the library** (guarding your own agent — not training or retraining ToolWarden's classifier):

```
pip install toolwarden-ai
pip install "toolwarden-ai[mcp]"   # only if you're using the MCP adapter
```

**Pointing at your own model files:** set the `TOOLWARDEN_MODEL_DIR` environment variable to a directory containing (see `src/toolwarden/config.py` for the exact expected layout):

```
classifiers/deberta-v3-base-toolwarden/final/   # the fine-tuned DeBERTa checkpoint
classifiers/lightgbm/lightgbm_model.txt         # the trained LightGBM booster
classifiers/lightgbm/ensemble_stacker.json       # the fitted ensemble stacker's weights (3 floats)
```

If unset, it defaults to a path specific to this project's own dev machine — it will not resolve on your machine, so setting it is required, not optional. `ensemble_stacker.json` in particular has to be generated once by someone with this repo's processed training dataset (dev-only, not distributed — `load_fitted_models()` in `src/toolwarden/classifier/evaluate.py` fits it fresh and caches it there the first time it's called in a dev checkout) and then travel with the rest of the directory; without it, `Classifier(detector_mode="ensemble")` raises a `FileNotFoundError` with next steps rather than failing silently. This project doesn't publish pretrained weights yet — Phase 4's checkpoint is currently a local artifact, not distributed.

**GPU vs. CPU:** the base install's `torch` dependency has no pinned CUDA build (a downstream consumer's hardware/driver setup is outside this project's control — see `pyproject.toml`'s comment), so `pip install toolwarden-ai` alone resolves a CPU-only `torch` wheel from PyPI. This is sufficient to actually run inference (verified via a fresh install with no extras) — just slower than the CUDA build this repo's own `requirements.txt` pins for training. For GPU-accelerated inference, install a CUDA-matched `torch` build yourself (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)) before or after installing `toolwarden-ai`.

Minimal usage:

```python
from toolwarden import ApprovalQueue, Classifier, Direction, EnforcementEngine, JsonlFileSink, PolicyEngine

classifier = Classifier(detector_mode="ensemble")  # or "deberta_only" -- see docs/known_limitations.md
policy = PolicyEngine()  # defaults: block >= 0.85, hold >= 0.5
engine = EnforcementEngine(policy=policy, approval_queue=ApprovalQueue(sink=JsonlFileSink("approvals.jsonl")))

tool_output = "..."  # a tool-call result you're about to feed back to your agent
score = classifier.score(tool_output)
outcome = engine.evaluate(Direction.RESULT, {"content": tool_output}, score)  # -> ALLOW / BLOCK / QUARANTINE / HOLD
```

For the two full integration adapters wired end to end (direct OpenAI tool-calling and MCP), see `src/toolwarden/demo/guarded_loop.py` / [docs/demo_walkthrough.md](docs/demo_walkthrough.md) and `src/toolwarden/mcp_proxy/server.py` / [docs/mcp_proxy_walkthrough.md](docs/mcp_proxy_walkthrough.md).

**Developing or retraining this repo's own classifier:**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins this repo's own exact training/benchmarking environment (a CUDA-specific torch build, `agentdojo`, `shap`, `matplotlib`, the local red-teamer client) — broader, and pinned more exactly for reproducibility, than what `pip install toolwarden-ai` needs as a library. `requirements-dev.txt` adds `pytest`/`ruff`.

**Windows troubleshooting:** if `pip install` fails with `WinError 206: The filename or extension is too long`, your venv's path is too deeply nested — `torch`'s installed file tree is large enough to hit Windows' `MAX_PATH` limit (hit this during Phase 11's own verification, inside a nested temp directory). Create the venv somewhere shorter (e.g. `C:\venvs\toolwarden`) rather than several folders deep.

## License

TBD.
