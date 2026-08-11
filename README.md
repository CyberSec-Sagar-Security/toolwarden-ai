# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

**Phase 11 (pip packaging) complete.** `src/toolwarden/__init__.py` now defines the actual public interface (`Classifier`, `PolicyEngine`, `EnforcementEngine`, `ApprovalQueue`, `Interceptor`, the direct-API adapter, and the supporting types) instead of leaving consumers to reach into internal submodules — importing it is cheap (no torch/transformers/lightgbm import as a side effect; see the module docstring). `pyproject.toml` now declares real runtime dependencies for `pip install toolwarden-ai`, split from `requirements.txt`'s exact training-environment pins, with `mcp`/`dev`/`train` as optional extras.

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

Model weight files (DeBERTa checkpoints, local red-teamer GGUF, LightGBM artifacts) are stored separately at
`D:\CyberSecurity\Projects\Apps and Models\ToolWarden\llm\` and are never committed to this repo. Both the
current (Qwen3.5-9B) and superseded (Qwen2.5-14B-Instruct) GGUF files are kept there side by side.

## Setup

**Using the library** (guarding your own agent — not training or retraining ToolWarden's classifier):

```
pip install toolwarden-ai
pip install "toolwarden-ai[mcp]"   # only if you're using the MCP adapter
```

Requires a trained classifier checkpoint under `TOOLWARDEN_MODEL_DIR` (see `src/toolwarden/config.py`) — this project doesn't publish pretrained weights yet; Phase 4's checkpoint is currently a local artifact, not distributed. Minimal usage:

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

## License

TBD.
