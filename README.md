# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

**Local red-teamer swapped Qwen2.5-14B-Instruct → Qwen3.5-9B (2026-08-11)** for better instruction-following and full 8GB VRAM fit. Fresh two-round review (not inherited from the Qwen2.5 review) found real, disclosed issues along the way — see [docs/redteam_generation_report.md](docs/redteam_generation_report.md): the first pass actually showed *worse* pretext homogeneity than Qwen2.5 (37% vs. 17%), traced to the model over-indexing on example phrases in the prompt itself; fixed and reduced to 5%, with literal-phrase clustering down to 3% (vs. Qwen2.5's 18%, same N=300 scale) and improved novelty (max trigram overlap 0.12, vs. Qwen2.5's same-scale 0.15 — not compared against Qwen2.5's much smaller N=25 pilot batches, which isn't a fair baseline). Obfuscation-instruction-following improved but remains a disclosed weakness. The report also carries a sharpened caveat: these numbers describe a set iteratively curated against the *specific* homogeneity patterns a human reviewer happened to catch, not a natural sample of the generator's organic diversity — real-world attacker diversity is likely understated by more than the raw numbers suggest. Approved by Sagar; Phase 8 rerun below.

**Phase 8 degradation curve, rerun against the Qwen3.5-9b set** (recall reported separately per source, never blended, every point estimate with a 95% CI — full breakdown in [docs/degradation_curve_report.md](docs/degradation_curve_report.md) and its chart):

| Source | N | Ensemble recall (95% CI) |
|---|---|---|
| In-distribution test (InjecAgent) | 680 | 1.000 [0.989, 1.000] |
| Held-out novel (AgentDojo) | 132 | 0.914 [0.776, 0.970] |
| Synthetic adversarial (Qwen3.5-9b-generated, N=300) | 300 | 0.980 [0.957, 0.991] |

Same qualitative story as the first Phase 8 run, now confirmed on an independently-generated synthetic set: AgentDojo remains harder for this classifier than the synthetic attacks (lower recall, much wider CI), and the classifier's dominant real-world failure mode is **precision, not recall** (AgentDojo precision: 0.27-0.30) — over-flagging benign content, which an attack-only synthetic set structurally cannot reveal regardless of which model generated it. The report's lower-bound caveat is now sharper: on top of the disclosed obfuscation/pretext gaps, the low homogeneity numbers themselves are a curation artifact (a human found and removed the specific repetition patterns this model defaulted to) rather than evidence of the generator's natural diversity — real degradation is likely understated by more than the numbers alone suggest. PINT is cited only (ProtectAI's published 79.14% score) — never framed as an evaluation this project ran, since PINT's real dataset isn't public (see docs/datasets.md).

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
