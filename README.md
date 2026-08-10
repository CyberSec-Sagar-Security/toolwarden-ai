# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

Phase 7 (local adversarial red-teamer) complete, scaled to 300 examples. Qwen2.5-14B-Instruct (Q4_K_M, llama.cpp, GPU-offloaded) running locally, driven by cross-product prompt scaffolding over an attack-intent/cover-context/phrasing-strategy taxonomy (full coverage, not random sampling) — see [docs/redteam_generation_report.md](docs/redteam_generation_report.md). Three review rounds, real problems found and disclosed at each: v1→v2 fixed a mislabeling bug (vague non-attacks labeled as injections) and improved phrasing clustering; the v3 scale-up to N=300 surfaced a new finding invisible at N=25 — pretext homogeneity (~17% share a "recent system update" hook regardless of category) that isn't caught by the trigram-overlap or phrase-clustering metrics. Combined with the still-unsolved obfuscation-realism gap, the report explicitly flags Phase 8's degradation numbers as a likely **lower bound** on real degradation, not an accurate estimate. Generated data is permanently flagged `is_synthetic: true` and is not merged into the trainable dataset.

Enforcement engine (Phase 6) and explainability (Phase 5) results stand — see [docs/](docs/). Detection: F1 1.000 in-distribution, ~0.42 on the held-out novel-attack split.

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
