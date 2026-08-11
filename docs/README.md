# docs

Architecture spec, threat model, and writeup material.

- [threat-model.md](threat-model.md) — STRIDE analysis, scope boundary (what ToolWarden does and does not defend against)
- [architecture.md](architecture.md) — interception-layer diagram, component boundaries, API-first/MCP-second sequencing rationale
- [datasets.md](datasets.md) — dataset sources actually used, why PINT was dropped as a trainable source, split design
- [classifier_report.md](classifier_report.md) — Phase 4 honest metrics (in-distribution vs. held-out novel-attack)
- [explainability_report.md](explainability_report.md) — Phase 5 SHAP/attention spot-check and honest assessment
- [known_limitations.md](known_limitations.md) — explicit tradeoffs, e.g. classifier-timeout handling's availability cost
- [redteam_generation_report.md](redteam_generation_report.md) — Phase 7 local red-teamer output (current model: Qwen3.5-9B), novelty check, honest quality assessment. Superseded Qwen2.5-14B-Instruct run archived at `redteam_generation_report_qwen2.5-14b_archive.md`; Qwen3.5's own first-pass (pre-fix) run archived at `redteam_generation_report_qwen3.5-9b_v1_archive.md`.
- [degradation_curve_report.md](degradation_curve_report.md) — Phase 8 headline artifact, currently built on the archived Qwen2.5-14B-Instruct red-team set: recall/F1 per source with 95% CIs, pretext-subset breakdown, honest lower-bound interpretation, PINT citation (not our eval). **Pending rerun** against the new Qwen3.5-9B set once Sagar signs off (see redteam_generation_report.md) — treat current numbers as provisional until then.
