# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

**Phase 8 (degradation curve benchmark) complete — this is the headline artifact.** See [docs/degradation_curve_report.md](docs/degradation_curve_report.md) and its chart. Recall reported separately per source (never blended), every point estimate with a 95% CI:

| Source | N | Ensemble recall (95% CI) |
|---|---|---|
| In-distribution test (InjecAgent) | 680 | 1.000 [0.989, 1.000] |
| Held-out novel (AgentDojo) | 132 | 0.914 [0.776, 0.970] |
| Synthetic adversarial (Qwen-generated, N=300) | 300 | 0.997 [0.981, 0.999] |

The synthetic set's high recall is itself a finding, not a good result: it's *higher* than AgentDojo's, empirically confirming the report's lower-bound caveat — the real, structurally-disjoint benchmark is harder for this classifier than 300 purpose-built adversarial examples. The classifier's dominant real-world failure mode is actually **precision**, not recall (AgentDojo precision: 0.27-0.30) — over-flagging benign content, which an attack-only synthetic set structurally cannot reveal. Full breakdown (per-model, pretext-subset vs. non-pretext, bootstrap/Wilson CIs) in the report. PINT is cited only (ProtectAI's published 79.14% score) — never framed as an evaluation this project ran, since PINT's real dataset isn't public (see docs/datasets.md).

Phases 1-7 stand — see [docs/](docs/) for threat model, architecture, dataset assembly, classifier training, explainability, enforcement engine, and the red-teamer.

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

Model weight files (DeBERTa checkpoints, Qwen2.5 GGUF, LightGBM artifacts) are stored separately at
`D:\CyberSecurity\Projects\Apps and Models\ToolWarden` and are never committed to this repo.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(`requirements.txt` will be added as dependencies are introduced phase by phase.)

## License

TBD.
