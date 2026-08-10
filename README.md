# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It detects behavior drift and indirect prompt injection in tool output, enforces block/quarantine/human-approval decisions (not just alerts), and publishes an honest, adversarially-measured degradation curve.

## Status

Phase 7 (local adversarial red-teamer) complete. Qwen2.5-14B-Instruct (Q4_K_M, llama.cpp, GPU-offloaded) running locally, driven by cross-product prompt scaffolding over an attack-intent/cover-context/phrasing-strategy taxonomy — see [docs/redteam_generation_report.md](docs/redteam_generation_report.md). Two generations: v1 had real quality problems found by manual review (phrasing clustering, several mislabeled non-attacks, cartoonish obfuscation); the prompt was fixed and regenerated. v2's mislabeling issue is resolved, phrasing clustering improved, obfuscation realism remains a known, disclosed limitation. Novelty confirmed both by a trigram-overlap check against the trainable pool (max 0.08) and manual read. Generated data is permanently flagged `is_synthetic: true` and is not merged into the trainable dataset.

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
