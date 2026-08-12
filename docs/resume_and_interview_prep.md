# Resume and Interview Prep

Material for presenting ToolWarden AI to a hiring manager — resume bullets and STAR-format interview talking points. Every number here traces back to a report in this repo; see the linked doc for the full derivation rather than taking any figure here as the primary source.

## Resume bullets

Pick 3–5 depending on space and the role (security-engineering roles: lead with the threat model/enforcement bullets; ML/applied-AI roles: lead with the classifier/ablation bullets).

- Designed and built an interception firewall for AI agent tool-calling that detects indirect prompt injection and tool-call scope drift and **enforces** block/quarantine/human-approval decisions in real time, across three independent integration paths (direct OpenAI tool-calling, MCP, and a FastAPI service) sharing one detection/policy core.
- Fine-tuned a DeBERTa-v3-base + LightGBM ensemble classifier and measured its real-world degradation honestly: 100% recall/precision in-distribution collapsing to 27.8% precision (70%+ false-positive rate) on a structurally disjoint held-out benchmark (AgentDojo) — reported per-source with 95% confidence intervals rather than blended into one number.
- Diagnosed a measurable weakness in the ensemble's combination logic via a paired ablation against real held-out attacks (found the fitted stacker trusted a less-reliable component almost as much as the more-reliable one, missing 8.6% of attacks the stronger model alone caught) and shipped a documented mitigation (`detector_mode="deberta_only"`) rather than silently patching the trained artifact every other reported number depended on.
- Built and iterated a local adversarial red-teaming pipeline (quantized open-weight LLM) to generate synthetic attack data for evaluation, diagnosing and fixing two rounds of generator template-collapse (a "pretext homogeneity" failure where the model over-indexed on its own prompt examples) via direct inspection rather than assumption.
- Ran every verification claim against the real artifact, not a mock: caught and fixed 8+ real packaging/runtime bugs invisible to unit tests (a silently-CPU-only "GPU-enabled" Docker container, a missing system library, an unbundled schema file, a re-fit-on-every-call model artifact) by literally building the wheel, running the containers, and hitting the live endpoints before calling any phase complete.

## STAR-format interview talking points

### 1. The ensemble was quietly missing real attacks the stronger model alone caught

**Situation:** The project's headline classifier is a DeBERTa+LightGBM ensemble, combined via a fitted logistic-regression stacker. While investigating an unrelated classifier-brittleness finding, a handful of hand-picked examples showed a pattern: DeBERTa would confidently and correctly flag an attack, LightGBM would confidently score the same text as benign, and the combined score would land below the enforcement threshold — a real attack effectively laundered through by a bad LightGBM signal.

**Task:** Determine whether this was a real, measurable problem against the actual held-out benchmarks the project's headline numbers were drawn from, or just an artifact of a small, cherry-picked sample.

**Action:** Wrote a paired ablation: for every real attack in the held-out AgentDojo set and the synthetic red-team set that DeBERTa alone correctly flagged, checked how many the ensemble's combination score pushed back under the classification threshold. Traced the root cause to the stacker's fitted coefficients (DeBERTa and LightGBM weighted almost identically, despite LightGBM being measurably less reliable on held-out data) and to *why*: the stacker was calibrated on a validation slice from the training distribution, where both models score near-perfectly, so it never saw the distribution shift where LightGBM becomes unreliable.

**Result:** Quantified a real, previously invisible cost — the ensemble missed 8.6% of AgentDojo attacks and 2.0% of synthetic attacks that DeBERTa alone would have caught, with no offsetting precision gain the sample size could actually detect. Rather than retraining the stacker on a quick fix (which risked contaminating the one clean held-out generalization signal the project's numbers depend on, or overfitting the fix to the exact 9 examples used to find the problem), shipped a documented, honest opt-out (`detector_mode="deberta_only"`) and flagged the real fix — a genuinely disjoint third calibration split — as scoped future work rather than rushing it in.

### 2. A "GPU-enabled" container that was silently running on CPU the whole time

**Situation:** Adding GPU passthrough to the Docker Compose deployment, the obvious verification step — confirm `torch.cuda.is_available()` returns `True` inside the container — passed cleanly.

**Task:** The project's standing discipline is to verify literally, not by proxy, so the question became: does a real inference call actually *use* the GPU, not just report that one is visible.

**Action:** Checked further and found `load_fitted_models()` loaded the DeBERTa checkpoint via `from_pretrained()` and never moved it off CPU — `cuda.is_available()` being `True` says the hardware and driver are reachable, not that any specific tensor lives there. Fixed the actual model placement, then verified end-to-end through the real running container: set the GPU flag, confirmed `next(model.parameters()).device` read `cuda:0` inside the live container, and hit the actual exposed HTTP endpoint to confirm the score matched a direct host-side GPU run bit-for-bit.

**Result:** Caught a fully silent correctness gap that would have shipped invisibly — the feature's own name was actively misleading about what it did. Also surfaced a second, non-obvious consequence while fixing it: CPU and GPU floating-point kernels aren't bit-identical (the same input scored differently at the 7th decimal place between them), so GPU use was deliberately gated behind an explicit opt-in flag rather than made automatic, to avoid silently perturbing every previously-published benchmark number the next time someone touched shared inference code on a GPU-equipped machine.

### 3. Choosing to show a real classifier weakness in the demo instead of hiding it

**Situation:** Building the first live end-to-end demo agent, the plan was two scenarios — a clean run and an injected-attack run — using a hand-written "clean webpage" and the same content with one injected paragraph. Both scored ~0.98, i.e. the classifier couldn't tell them apart on the content actually chosen for the demo.

**Task:** Understand whether this was a demo-content problem (pick different text) or a real classifier property worth disclosing, and decide how to handle it in a deliverable meant to demonstrate the system working.

**Action:** Ran 20+ live verification calls against genuinely different phrasings of equally benign content and found the swings were real and sharp — a 22-character difference between two harmless sentences about the weather flipped the score from 0.19 to 0.93. Traced the mechanism with the explainability tooling already built in an earlier phase: DeBERTa's own opinion barely moved, but LightGBM's SHAP attribution showed the swing was driven almost entirely by coarse surface features (`text_length`, `topic_shift_score`), confirming a pattern the explainability spot-check had already flagged as a risk but never previously shown to cause a concrete behavior change. Brought this to the project owner before proceeding rather than quietly picking content that happened to score well.

**Result:** The demo shipped with a third scenario — a genuinely benign page that gets auto-quarantined without human review — turning what could have been a hidden flaw into the project's most concrete, honest illustration of its own precision problem, and directly informed a later, larger finding (the ensemble-ablation story above) about which component was actually responsible.

### 4. Auditing your own architecture diagram against the actual code

**Situation:** By the project's testing/validation phase, the architecture doc described explainability (SHAP + attention evidence) as attached to every flagged decision a human reviews.

**Task:** As part of a standing rule that every claim in the docs must trace to an actual test or measured result — not just be asserted — verify that claim against the code, not just re-read the doc and assume it was still accurate.

**Action:** Grepped all three live enforcement paths (the OpenAI adapter, the MCP adapter, the FastAPI service) for any import of the explainability module. Zero matches. What a human reviewer actually saw in every live path was a raw score and a templated reason string — the explainability output existed only in an earlier phase's offline spot-check script, never wired into anything live. Rather than unilaterally patching this mid-audit, flagged it as a scoped gap for the project owner's explicit go-ahead, then — once approved — designed the fix to attach explainability at one shared point (inside the enforcement engine, where the decision object is constructed) so all three existing adapters inherited it without three separate integrations.

**Result:** Closed a real gap between documented architecture and shipped behavior, verified live across all three adapters (including a real `docker compose up` stack and actual HTTP responses, which caught a packaging bug — a dependency scoped to training-only that the live path now needed) rather than just unit-tested. The audit discipline itself — checking code against documentation claims instead of trusting either in isolation — is the transferable habit, independent of this specific finding.

### 5. Iterating an LLM red-teamer against its own failure mode, twice

**Situation:** The project's synthetic adversarial evaluation set is generated by a local, quantized open-weight LLM instructed to produce diverse prompt-injection attacks. An early batch showed a real problem: attacks kept converging on the same "system recovery / priority override" pretext regardless of the intended phrasing strategy.

**Task:** Fix the generator's homogeneity without just re-prompting once and hoping — verify the fix actually worked at scale, not on a lucky small sample.

**Action:** Swapped to a different base model expecting better instruction-following to reduce the drift — and on the first pass, the problem got *worse* (37% pretext clustering, up from 17%), the opposite of the hypothesis. Root-caused it directly: the prompt itself had listed the exact pretext family as an illustrative example, and the more instruction-faithful model was taking that example as a default template. Fixed by banning the family explicitly; a follow-up smoke test then caught a second, self-inflicted version of the same bug — a single example phrase given for a different technique got reused verbatim across unrelated categories. Fixed again by removing the reusable example entirely and requiring the model to invent its own wording, verified with another smoke test before committing to a full 300-example run.

**Result:** Pretext clustering dropped from 37% to 5%, and literal "ignore all previous instructions"-style phrasing dropped to 3% — but the final report still discloses, rather than hides, that every improvement was found by a human manually reading samples and prompting the specific pattern away, so the low diversity numbers describe a curated set, not evidence the underlying generation approach is naturally this diverse. The lesson generalized into a stated rule for future prompt iteration on this model: never give a single copyable example phrase, or it becomes the new template.

## Notes on presenting the honest-limitations framing in an interview

This project's defining trait is publishing numbers that make it look worse (27.8% precision on real held-out data, an ensemble that misses attacks a simpler model alone would catch, an adapter that's built but not live-verified) instead of hiding them. That is the pitch, not a caveat to get through quickly — a hiring manager evaluating security or ML judgment specifically wants to see that the discipline to disclose a bad number exists, because it's the same discipline that prevents a false sense of security in production. Lead with what was measured and why it was measured that way before getting to the mitigation; don't rush past the bad number to the fix.
