# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

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

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(`requirements.txt` will be added as dependencies are introduced phase by phase.)

## License

TBD.
