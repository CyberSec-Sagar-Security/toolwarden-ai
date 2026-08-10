# docs

Architecture spec, threat model, and writeup material.

- [threat-model.md](threat-model.md) — STRIDE analysis, scope boundary (what ToolWarden does and does not defend against)
- [architecture.md](architecture.md) — interception-layer diagram, component boundaries, API-first/MCP-second sequencing rationale
- [datasets.md](datasets.md) — dataset sources actually used, why PINT was dropped as a trainable source, split design
- [classifier_report.md](classifier_report.md) — Phase 4 honest metrics (in-distribution vs. held-out novel-attack)
- [explainability_report.md](explainability_report.md) — Phase 5 SHAP/attention spot-check and honest assessment
- [known_limitations.md](known_limitations.md) — explicit tradeoffs, e.g. classifier-timeout handling's availability cost
- [redteam_generation_report.md](redteam_generation_report.md) — Phase 7 local red-teamer output, novelty check, honest v1→v2 quality assessment
