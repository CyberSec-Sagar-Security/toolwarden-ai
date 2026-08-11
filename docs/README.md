# docs

Architecture spec, threat model, and writeup material.

- [threat-model.md](threat-model.md) — STRIDE analysis, scope boundary (what ToolWarden does and does not defend against)
- [architecture.md](architecture.md) — interception-layer diagram, component boundaries, API-first/MCP-second sequencing rationale
- [datasets.md](datasets.md) — dataset sources actually used, why PINT was dropped as a trainable source, split design
- [classifier_report.md](classifier_report.md) — Phase 4 honest metrics (in-distribution vs. held-out novel-attack)
- [explainability_report.md](explainability_report.md) — Phase 5 SHAP/attention spot-check and honest assessment
- [known_limitations.md](known_limitations.md) — explicit tradeoffs, e.g. classifier-timeout handling's availability cost
- [redteam_generation_report.md](redteam_generation_report.md) — Phase 7 local red-teamer output (current model: Qwen3.5-9B), novelty check, honest quality assessment, sharpened curation caveat. Superseded Qwen2.5-14B-Instruct run archived at `redteam_generation_report_qwen2.5-14b_archive.md`; Qwen3.5's own first-pass (pre-fix) run archived at `redteam_generation_report_qwen3.5-9b_v1_archive.md`.
- [degradation_curve_report.md](degradation_curve_report.md) — Phase 8 headline artifact, rerun against the Qwen3.5-9B red-team set (2026-08-11): recall/F1 per source with 95% CIs, pretext-subset breakdown, honest lower-bound interpretation (dynamically generated from the actual run, not hardcoded prose), PINT citation (not our eval).
- [demo_walkthrough.md](demo_walkthrough.md) — Phase 9 standalone demo agent: the interception/classifier/enforcement pipeline chained into a live agent loop for the first time, three scenarios (allow, hold-then-approve/deny, false-positive auto-quarantine), and the classifier-brittleness finding it surfaced along the way.
- [mcp_proxy_walkthrough.md](mcp_proxy_walkthrough.md) — Phase 10: the same enforcement pipeline reached through MCP instead of direct OpenAI tool-calling, proving the behavior isn't an artifact of one integration adapter. Same three scenarios, same reused classifier/policy/queue, enforcement moved server-side.
