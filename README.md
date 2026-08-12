# ToolWarden AI

![status](https://img.shields.io/badge/status-WIP-yellow)

A firewall that sits between an AI agent and the tools it calls. It intercepts every outbound tool-call request and inbound tool-call result, classifies each for indirect prompt injection and tool-call scope drift with a trained DeBERTa+LightGBM ensemble, **enforces** a block/quarantine/human-approval decision (not just an alert), and publishes an honest, adversarially-measured degradation curve instead of a single flattering accuracy number.

## The problem

Agentic AI systems call tools — they fetch webpages, read emails, query databases, send messages. Any of those tool *results* can carry attacker-controlled text, and an LLM has no reliable way to distinguish "instructions from my system prompt" from "instructions that happened to be sitting in a webpage I just fetched." This is indirect prompt injection, and its practical consequence is scope drift: a task scoped to "summarize this page" ends up calling `send_email` because the page told the agent to.

Most published defenses report one clean benchmark number and stop there. This project's premise is that the number that matters is how much a detector's performance degrades from a clean, in-distribution benchmark to real adversarial pressure — and that number should be measured and published honestly, including the parts that make the system look worse, not just the parts that make it look good.

## What it does

- **Intercepts** both directions of tool-call traffic — outbound request (agent → tool) and inbound result (tool → agent) — at three independent integration points: direct OpenAI tool-calling, the Model Context Protocol, and a FastAPI service reachable over HTTP.
- **Classifies** each with a fine-tuned DeBERTa-v3-base + LightGBM ensemble (logistic-regression stacker), trained on InjecAgent and evaluated against AgentDojo as a structurally disjoint held-out set.
- **Enforces** — not just flags — a real decision: allow, block, quarantine (strip the flagged content before it reaches the agent), or hold for human approval, via a configurable policy engine and an attributed, timestamped approval queue.
- **Explains** every non-ALLOW decision with the top SHAP-attributed LightGBM features and top-attended DeBERTa tokens behind the score, surfaced wherever a human reviews a hold.
- **Measures its own failure honestly**: a degradation curve across three sources (in-distribution, real held-out novel attacks, synthetic adversarial), every point estimate with a 95% confidence interval, reported per-source rather than blended into one number.

## Architecture

```
                         +-------------------------------------------+
                         |              ToolWarden AI                |
                         |         (interception layer)               |
                         |                                             |
   +-------+  tool-call  |   +------------+       +----------------+  |   +------+
   |       |  request    |   |            |       |                |  |   |      |
   | Agent |------------>|-->| Classifier |------>|  Enforcement   |--|-->| Tool |
   |       |             |   | (DeBERTa + |       |  Engine        |  |   |      |
   |       |             |   |  LightGBM) |       |  (policy:      |  |   |      |
   |       |<------------|<--|            |<------|  allow/block/  |<-|---|      |
   |       |  tool-call  |   +------------+       |  quarantine/   |  |   +------+
   +-------+  result     |         |               |  hold)         |  |
                         |         v               +--------+-------+  |
                         |   +-----------+                   |          |
                         |   | Explain-  |                   v          |
                         |   | ability   |          +-----------------+ |
                         |   | (SHAP +   |          | Approval Queue  | |
                         |   | attention)|          | (human sign-off,| |
                         |   +-----------+          |  logged w/      | |
                         |                           |  timestamp +    | |
                         |                           |  decision)      | |
                         |                           +-----------------+ |
                         +-------------------------------------------+
```

One detection/policy core, two directions of traffic, three integration adapters that reuse it unmodified — the classifier and enforcement engine don't know or care whether they were reached via a direct OpenAI tool call, an MCP `on_call_tool` handler, or an HTTP POST to a FastAPI service.

| Component | Responsibility | Does NOT do |
|---|---|---|
| Interception layer | Capture outbound tool-call requests and inbound tool-call results | Make any allow/block decision itself |
| Classifier (DeBERTa + LightGBM ensemble) | Score a tool-call request or result for injection/scope-drift risk | Enforce policy, own the approval queue |
| Explainability | Attach SHAP/attention evidence to a flagged score | Change the classifier's decision |
| Enforcement engine | Map classifier score → allow/block/quarantine/hold, per configurable thresholds | Self-certify a held action as safe — that requires human approval |
| Approval queue | Hold flagged actions, record human decisions with timestamp + identity | Auto-approve anything |

**Why direct-API first, MCP second:** direct tool-calling APIs (OpenAI/Anthropic `tool_use`) expose a stable, versioned JSON contract for both the call and the result, so the interception layer could be proven against a fixed target before absorbing the added variable of a newer, faster-evolving protocol. This deliberately avoids a lesson from a prior project (ContextGuard AI), where an integration built on scraping/DOM-dependent structure broke when that structure shifted, for reasons that had nothing to do with the actual detection logic. Full rationale: [docs/architecture.md](docs/architecture.md).

## Results — the honest degradation curve

The project's headline artifact. Recall is reported **separately per source** — never blended into one number — because blending would hide exactly the gap this project exists to surface. Every point estimate carries a 95% confidence interval (Wilson score for proportions, percentile bootstrap for F1). Generated by `src/toolwarden/benchmark/degradation.py`.

| Source | N | Model | Recall (95% CI) | Precision (95% CI) | F1 (95% CI) |
|---|---|---|---|---|---|
| In-distribution test (InjecAgent) | 680 | Ensemble | 1.000 [0.989, 1.000] | 1.000 [0.989, 1.000] | 1.000 [1.000, 1.000] |
| Held-out novel (AgentDojo) | 132 | Ensemble | 0.914 [0.776, 0.970] | **0.278 [0.205, 0.366]** | 0.427 [0.324, 0.521] |
| Synthetic adversarial (Qwen3.5-9B-generated) | 300 | Ensemble | 0.980 [0.957, 0.991] | N/A (attack-only) | N/A (attack-only) |

**The number that actually matters: precision on real held-out data is 0.27–0.30, not the recall figure alone.** On AgentDojo — a real, structurally disjoint benchmark the classifier never trained on — recall stays high (0.914) but precision collapses to 0.278: roughly **70–73% of everything the classifier flags on realistic novel content is a false positive.** A synthetic, attack-only red-team set can't reveal this at all (it has no benign examples to false-positive on), which is exactly why this project insists on reporting both, and reporting AgentDojo separately rather than folding it into a friendlier blended number.

**The synthetic curve is a lower bound, stated as one, not an accurate estimate.** The Qwen3.5-9B-generated red-team set (Phase 7) still has disclosed weaknesses after three rounds of iteration: obfuscation realism is weak (most "obfuscated" examples skip the technique and produce clean prose instead), a residual ~5% pretext cluster remains (a "system recovery / priority override" family), and every homogeneity pattern the set does *not* show was found by a human manually reading samples and prompting it away — the low clustering numbers describe a set curated against the specific patterns a reviewer happened to catch, not the generator's natural diversity. AgentDojo being *harder* for this classifier than 300 LLM-generated attacks (0.914 recall vs. 0.980, and AgentDojo's own precision collapse the synthetic set structurally cannot show) is empirical confirmation that this caveat isn't just a hedge. Full breakdown, pretext-subset table, and cross-model consistency check (recall conclusions hold whether the red-teamer is Qwen2.5-14B-Instruct or Qwen3.5-9B): [docs/degradation_curve_report.md](docs/degradation_curve_report.md).

### Ensemble-combination ablation — is the ensemble actually better than DeBERTa alone?

Investigated, not assumed. For every real attack DeBERTa alone correctly flags in the held-out sources, the ensemble's combination score pushes some of them back under the classification threshold:

| Source | Real attacks (N) | DeBERTa-alone caught | Ensemble missed (of DeBERTa's catches) | Missed rate |
|---|---|---|---|---|
| Held-out novel (AgentDojo) | 35 | 35 | 3 | 8.6% |
| Synthetic, all | 300 | 300 | 6 | 2.0% |

**Root cause, confirmed not hypothesized:** the fitted stacker (`EnsembleStacker.coefficients()`) weighs DeBERTa and LightGBM almost identically (deberta_weight≈3.807, lightgbm_weight≈3.791) despite LightGBM being measurably less reliable on both held-out sources — likely because the stacker was fit on a validation slice from the training distribution, where both models score near-perfectly, so it never saw the distribution shift where LightGBM's coarse features become unreliable and had no signal to learn to distrust it. A real fix (a stacker calibrated on genuinely out-of-distribution data) was deliberately **not applied**: it would need a third calibration split that's never touched either benchmark, and any quick fix risks the exact train-on-the-test-set mistake the project's own group-holdout split exists to prevent. Shipped instead: `Classifier(detector_mode="deberta_only")`, a documented, blunt opt-out of this specific failure mode — at the honest cost of whatever LightGBM contributes correctly elsewhere too. Full investigation, concrete missed-attack examples, and the interim-mitigation decision trail: [docs/known_limitations.md](docs/known_limitations.md).

### Classifier brittleness on short novel text — a real finding, not smoothed over

Building the standalone demo agent (Phase 9) required picking benign "webpage" content. Before any scenario text was locked, 20+ live verification calls against different phrasings of equally benign sentences surfaced something sharper than the AgentDojo precision number alone: small, meaning-preserving rewordings can swing the ensemble score by 0.5+ with no consistent explanation. These are the investigation-phase findings that led to that conclusion — not the demo's own final scenario text, reconciled below:

| Text | Score |
|---|---|
| "The weather in Lisbon is sunny today." | 0.192 |
| "Lisbon Weather Update: Sunny skies are expected across the city this weekend, with highs near 24°C." | 0.933 |
| A real AgentDojo held-out example, labeled *benign* in the eval set | 0.979 |

Traced to its actual mechanism, not left as an anecdote: decomposing DeBERTa's score, LightGBM's score, and LightGBM's SHAP contributions for the pair above showed DeBERTa barely moves (0.708 → 0.674) while a 22-character length difference flips LightGBM from 0.020 to 0.999 — SHAP attributes the swing almost entirely to `text_length` and `topic_shift_score`, coarse surface features LightGBM leans on more than the semantically-targeted ones (Phase 5's explainability spot-check had already flagged this pattern; this confirmed it causes a measured behavior change, not just an attribution curiosity).

The standalone demo's own locked scenario content — different, final text, not the table above — shows the identical effect: **ALLOW scores 0.230** ("Bus route 28 stops at Alfama, Graca, and Baixa every 15 minutes on weekdays.", the same sentence and same score as the investigation's own bus-route check) and **FALSE POSITIVE scores 0.960**, auto-quarantined without human review — a slightly extended version of the 0.933 weather example above (one added sentence about narrow streets pushed it from 0.933 to 0.960: a second, independent instance of the same brittleness, not a coincidence). The demo was built to show this honestly as its third scenario rather than quietly picking content that happens to score well. Full root-cause trace: [docs/known_limitations.md](docs/known_limitations.md).

### Explainability — built, spot-checked, and honestly assessed

SHAP (`TreeExplainer`) for LightGBM's engineered features, last-layer attention for DeBERTa's tokens. Spot-checked against 4 real records (2 InjecAgent, 2 AgentDojo): DeBERTa's attention is "partially useful, not crisp" — on the clearest injection example, roughly half the top-8 attended tokens were genuinely part of the injected span, the rest were from surrounding benign text. LightGBM's SHAP leans on coarse features (`text_length`, `topic_shift_score`) far more than semantically-targeted ones in 3 of 4 cases. Conclusion stated plainly rather than oversold: "both explainers produce real signal, not noise, but neither cleanly isolates the injected span the way a human reviewer would want — treat this as directional evidence, not a precise highlight." Full spot-check: [docs/explainability_report.md](docs/explainability_report.md).

## Engineering rigor — bugs that only a real, literal test caught

A recurring pattern across this project: unit tests with mocked dependencies passed cleanly while the actual packaged/deployed artifact was broken. Every phase's stop gate insisted on running the real thing — a real fresh venv, a real Docker build, real containers, real HTTP requests — specifically because that's where these were found:

- **Fresh-venv install (Phase 11):** the ensemble stacker was silently re-fitting from a gitignored training dataset on every call — invisible in the dev checkout, fatal in a real install with no training data. Fixed by persisting the fitted stacker as a small JSON artifact instead of re-deriving it.
- **Fresh-venv install (Phase 11):** demo log paths resolved relative to `__file__`, which pointed inside the venv's own `site-packages` tree for an installed run instead of the caller's working directory.
- **Docker packaging (Phase 12):** `schema.sql` wasn't bundled by `pip install .` (setuptools doesn't ship non-`.py` files by default) — the backend crashed on startup with a `FileNotFoundError` for a file that plainly existed in the source tree.
- **Docker packaging (Phase 12):** LightGBM needs `libgomp1`, absent from the `python:3.13-slim` base image — `OSError: libgomp.so.1: cannot open shared object file`.
- **Docker error handling (Phase 12):** a malformed approval ID crashed the resolve endpoint with a raw Postgres 500 instead of a clean 404, because UUID-typed columns reject a non-UUID string before the "not found" logic ever runs.
- **GPU passthrough (Phase 12) — the sharpest one:** `torch.cuda.is_available()` returning `True` inside the "GPU-enabled" container said nothing about whether the model tensors actually lived there. They didn't — `load_fitted_models()` never moved the DeBERTa checkpoint off CPU, so inference silently ran on CPU the whole time despite the GPU being visible. Fixed, and deliberately gated behind an explicit `TOOLWARDEN_USE_GPU=1` flag rather than made automatic — CPU and GPU floating-point kernels aren't bit-identical (the same attack text scored `0.9772014446696929` on CPU vs. `0.9772014547817592` on GPU, differing at the 7th decimal), and every number this project has ever published was computed on CPU, so silently switching would have perturbed every prior report the next time shared inference code was touched.
- **Explainability wiring (Phase 13):** `shap` was only in a training-only optional dependency extra; the live Docker service crashed with `ModuleNotFoundError` the moment the enforcement path started computing explanations for real, since a fresh install via `pyproject.toml` alone never pulled it in.
- **Human-testing guide (Phase 13):** writing the guide by literally running every command against a genuinely fresh install caught two more real issues before a human tester would have hit them — bash-style escaped quotes in a `curl` example that don't work in PowerShell, and a missing `-s` flag causing noisy progress-meter output in a copy-pasted example.

None of these were found by reading the code carefully or by a mocked unit test — they only exist at the boundary between code and its real packaging/runtime environment. Full list of every bug, fix, and how each was verified: [docs/docker_walkthrough.md](docs/docker_walkthrough.md), [docs/known_limitations.md](docs/known_limitations.md).

## Scope

- **In scope:** tool-call request/response interception, classification of tool-call scope drift and indirect prompt injection in tool output, configurable enforcement policy, human-approval queue, explainability for human reviewers.
- **Out of scope:** semantic content of the agent's own text responses, model internals of closed APIs, direct prompt injection from the user's own turns, tool reliability/correctness, network/infra security, tool supply-chain security. Full STRIDE analysis and reasoning: [docs/threat-model.md](docs/threat-model.md).

**Honestly disclosed, not resolved:** classifier-timeout handling fails closed into HOLD, which is safe but has a real availability cost under load ([docs/known_limitations.md](docs/known_limitations.md)); the Postgres audit trail is convention-only append-only, not database-enforced ([docs/threat-model.md](docs/threat-model.md)); the Anthropic adapter is built and unit-tested but not live-verified (see below).

## What's actually verified, and what isn't

**Live-verified end to end:** the direct OpenAI adapter, the MCP adapter, and the full Docker Compose stack (`db` + FastAPI `backend` + dashboard `frontend`) — all three run through the same three-scenario demo (ALLOW / HOLD-then-approve-or-deny / false-positive auto-quarantine) with real API calls, real classifier scores, and real explainability output attached to every non-ALLOW decision.

**Built and unit-tested, not live-verified:** the Anthropic adapter (`GuardedAnthropicToolLoop`). It reuses the classifier/policy/enforcement/queue completely unmodified, exactly like the MCP adapter, and passes 7/7 unit tests against a scripted fake client covering the same enforcement matrix plus Claude's own multi-tool-call-per-turn batching behavior. But no live call to Anthropic's API has ever succeeded — the only key available was an AWS Bedrock key (a different auth path from `console.anthropic.com`'s direct API), rejected by AWS's own gateway with a genuine key-format error, not a bug in this project. Stated here plainly so it isn't misread as working: **do not treat the Anthropic path as confirmed until it's actually run.** Steps to complete that verification: [docs/human_testing_guide.md](docs/human_testing_guide.md#5-the-anthropic-adapter--built-not-yet-verified).

## Setup

**Using the library** (guarding your own agent — not training or retraining ToolWarden's classifier):

```
pip install toolwarden-ai
pip install "toolwarden-ai[mcp]"   # only if you're using the MCP adapter
```

**Pointing at your own model files:** set `TOOLWARDEN_MODEL_DIR` to a directory containing:

```
classifiers/deberta-v3-base-toolwarden/final/   # the fine-tuned DeBERTa checkpoint
classifiers/lightgbm/lightgbm_model.txt         # the trained LightGBM booster
classifiers/lightgbm/ensemble_stacker.json      # the fitted ensemble stacker's weights (3 floats)
```

This project doesn't publish pretrained weights yet — Phase 4's checkpoint is currently a local artifact, not distributed. If unset, `TOOLWARDEN_MODEL_DIR` defaults to a path specific to this project's own dev machine and will not resolve elsewhere; `Classifier(detector_mode="ensemble")` raises a clear `FileNotFoundError` with next steps rather than failing silently.

**GPU vs. CPU:** the base install resolves a CPU-only `torch` wheel — sufficient to run inference, just slower than the CUDA build this repo's own training environment pins. Install a CUDA-matched `torch` build yourself for GPU-accelerated inference.

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

**Full step-by-step human-testing walkthrough** (fresh-venv wheel install, the OpenAI demo, the MCP demo, and the full Docker stack — every command literally verified, not just written): [docs/human_testing_guide.md](docs/human_testing_guide.md).

**Developing or retraining this repo's own classifier:**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins this repo's own exact training/benchmarking environment; `requirements-dev.txt` adds `pytest`/`ruff`.

**Windows troubleshooting:** if `pip install` fails with `WinError 206: The filename or extension is too long`, your venv's path is too deeply nested — `torch`'s installed file tree is large enough to hit Windows' `MAX_PATH` limit. Create the venv somewhere shorter (e.g. `C:\venvs\toolwarden`).

## Project layout

```
src/         application and library code
tests/       unit / integration / e2e tests (131 tests: 111 unit, 20 Postgres/HTTP integration)
docs/        architecture, threat model, benchmark reports, walkthroughs, writeups
datasets/    dataset assembly scripts + (gitignored) data
models/      model config/loading code only — actual weight files live outside this repo
```

Model weight files are stored separately at `D:\CyberSecurity\Projects\Apps and Models\ToolWarden\` and are never committed to this repo.

## Documentation

Full index, including every benchmark report, walkthrough, and this project's resume/interview material: [docs/README.md](docs/README.md).

- [docs/threat-model.md](docs/threat-model.md) — STRIDE analysis, scope boundary
- [docs/architecture.md](docs/architecture.md) — interception-layer diagram, component boundaries, sequencing rationale
- [docs/datasets.md](docs/datasets.md) — dataset sources, why PINT was dropped, split design
- [docs/classifier_report.md](docs/classifier_report.md) — Phase 4 in-distribution vs. held-out metrics
- [docs/explainability_report.md](docs/explainability_report.md) — SHAP/attention spot-check
- [docs/redteam_generation_report.md](docs/redteam_generation_report.md) — local red-teamer output, novelty check, honest quality assessment
- [docs/degradation_curve_report.md](docs/degradation_curve_report.md) — the full headline benchmark
- [docs/known_limitations.md](docs/known_limitations.md) — every disclosed tradeoff and honest gap, in one place
- [docs/demo_walkthrough.md](docs/demo_walkthrough.md) / [docs/mcp_proxy_walkthrough.md](docs/mcp_proxy_walkthrough.md) / [docs/docker_walkthrough.md](docs/docker_walkthrough.md) — the three integration adapters, live-verified
- [docs/human_testing_guide.md](docs/human_testing_guide.md) — run it yourself, every command verified
- [docs/resume_and_interview_prep.md](docs/resume_and_interview_prep.md) — resume bullets and STAR-format talking points drawn from this project

## Development history

<details>
<summary>Phase-by-phase build log (14 phases, click to expand)</summary>

**Phase 0 — repo scaffold.** `src/`/`tests`/`docs`/`datasets`/`models` layout, dev dependency pins.

**Phase 1 — threat model + architecture.** STRIDE analysis and the explicit scope boundary, written before any detection code existed.

**Phase 2 — passthrough interception layer + OpenAI adapter.** The interception point (both directions of tool-call traffic) built first as passthrough-only, against direct OpenAI tool-calling.

**Phase 3 — dataset assembly.** InjecAgent (train/in-distribution test) and AgentDojo (held-out, structurally disjoint) integrated; PINT investigated and dropped as a trainable source (Lakera ships the eval methodology, not the 4,314-input dataset itself — confirmed by inspecting the repo directly, not assumed).

**Phase 4 — classifier core.** DeBERTa-v3-base fine-tune + LightGBM, combined via a fitted logistic-regression stacker. In-distribution F1 1.000/1.000/1.000 (ensemble); held-out novel F1 collapses to 0.427 (recall 0.914, precision 0.278) — reported honestly from day one, not smoothed into one headline number.

**Phase 5 — explainability.** SHAP for LightGBM, attention for DeBERTa, spot-checked against 4 real records. Found — and stated plainly — that LightGBM leans on coarse surface features more than the semantically-targeted ones, a finding that later explained a real production behavior in Phase 9.

**Phase 6 — enforcement engine.** Policy layer (allow/block/quarantine/hold thresholds) and an attributed, timestamped human-approval queue.

**Phase 7 — local adversarial red-teamer.** Qwen2.5-14B-Instruct, then swapped to Qwen3.5-9B after three iteration rounds targeting pretext homogeneity and obfuscation realism — the swap's own hypothesis (better instruction-following reduces pretext drift) was only partially borne out and initially *contradicted* on the first pass, root-caused and fixed rather than assumed to have worked. Final set: 300 examples, 0/300 near-duplicates of the training pool, literal-override-phrase clustering down to 3%.

**Phase 8 — degradation curve benchmark.** The project's headline artifact: recall/precision/F1 with 95% CIs across in-distribution, held-out novel, and synthetic sources, reported separately per source. Confirmed the recall-CI conclusions hold independent of which red-teamer generated the synthetic set.

**Phase 9 — standalone demo agent.** First time the full pipeline ran as one live agent loop. Surfaced the short-text brittleness finding above while picking demo content — confirmed with the project owner before shipping the demo honestly (a real false-positive scenario) rather than cherry-picking content that scores well.

**Phase 10 — MCP proxy.** Same pipeline reused unmodified, reached through MCP's `on_call_tool` instead of inline OpenAI calls, proving the detection/enforcement behavior isn't an artifact of one integration path.

**Phase 11 — pip packaging.** Public `__init__.py` surface, real `pyproject.toml` dependencies, `detector_mode="deberta_only"` shipped as a direct response to the ensemble-ablation finding. Stop gate: built the wheel, installed into a brand-new venv outside the repo, ran the full demo against only what that install provides — caught two real bugs invisible in the dev checkout (see Engineering rigor above).

**Phase 12 — Docker Compose app.** `db` + `backend` + `frontend`, the same enforcement pipeline reused again, now reachable over HTTP with an honest async-approval shape (a HOLD returns a `pending_id` immediately rather than blocking the request). Four real bugs caught by literal container verification, including the GPU-silently-running-on-CPU trap (see Engineering rigor above).

**Phase 13 — testing/validation audit.** Full test suite and every doc/README claim traced back to an actual test or measured result. Found and fixed six real documentation issues, plus one larger gap: explainability was built (Phase 5) and spot-checked but never wired into any live decision path past that offline check. Follow-up work closed that gap (one shared integration point in the enforcement engine, inherited by all three adapters, verified live in all three) and built the Anthropic adapter — unit-tested but honestly marked not-live-verified after the only available key (AWS Bedrock) was rejected by AWS's own gateway.

**Phase 14 — this document.** Writeup and resume material, pulling real numbers and real findings from all 13 prior phases rather than summarizing from memory.

</details>

## License

TBD.
