# Known Limitations

Honest, explicit tradeoffs — not gaps to hide, gaps to disclose. Add to this file whenever a design decision trades one property for another; don't let it live only in a commit message.

## Classifier-timeout handling degrades availability, not just under attack (decided 2026-08-11)

**Decision:** when the classifier times out or errors, the enforcement engine fails **closed into HOLD** — the same bucket a genuine mid-confidence score lands in — not fail-open (let the tool call through) and not a hard deny (block outright). See `src/toolwarden/enforcement/policy.py::decide_on_timeout` and `docs/threat-model.md`'s Denial of Service section, where this was originally flagged as an open question rather than decided by default.

**Why:** fail-open would defeat the project's core enforcement thesis — an attacker who can reliably trigger classifier timeouts (e.g. via a pathologically expensive tool-output payload) would get a free pass past detection entirely. A hard deny-on-timeout is the opposite failure: it would block the agent on ordinary infrastructure latency, not just attacks, making the system unusable whenever the classifier is merely slow rather than actually facing something dangerous. Routing timeouts into the existing HOLD/approval-queue path reuses a decision path that's already reviewed and audited, rather than inventing a fourth failure mode.

**The tradeoff this creates:** every classifier timeout now produces an approval-queue item, identical in shape to a real mid-confidence detection. Under load or during a genuine outage, this means:
- Approval-queue depth grows even when nothing malicious is happening — a latency problem becomes a human-review burden.
- Legitimate tool calls slow down (wait for human sign-off) exactly when the system is already under stress.
- Detection *accuracy* (the classifier's own scores, when it does respond) is unaffected by this — the tradeoff is entirely on *availability*, not the honesty of the degradation-curve numbers reported in Phase 8.

**Not measured yet:** how queue depth and hold-latency behave under sustained timeout pressure (e.g. classifier under load, or a deliberate resource-exhaustion attempt targeting the classifier itself rather than the detection logic). This is a real gap — worth a dedicated load test before any claim of production-readiness, not before. Tracked here rather than assumed away.

**Now grounded in measured data, not just a hypothetical (Phase 8, 2026-08-11):** the queue-flooding concern above isn't only a timeout/load scenario. On AgentDojo — a real, structurally novel benchmark, not the synthetic set — the classifier's precision is only 0.27-0.30 (`docs/degradation_curve_report.md`), meaning roughly 70-73% of everything it flags on realistic novel content is a false positive. That's under completely normal operation, no timeout or attack required: an approval queue processing AgentDojo-like traffic through today's thresholds would already be dominated by human review of false positives before a single classifier timeout ever occurs. The timeout-driven flooding described above compounds an existing baseline load, it doesn't introduce a new failure mode from a clean starting point.
