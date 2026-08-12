# Architecture

## Interception point

ToolWarden sits between the agent's tool-calling loop and the actual tool execution. It inspects traffic in both directions: the outbound tool-call request (function name + arguments the agent wants to invoke) and the inbound tool-call result (what the tool actually returns, which may carry attacker-controlled content). See [threat-model.md](threat-model.md) for what each direction defends against.

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

- **Outbound path** (agent → tool): the tool-call request is classified for scope drift before it's allowed to reach the tool.
- **Inbound path** (tool → agent): the tool result is classified for indirect prompt injection before it's handed back to the agent as context.
- Both paths route through the same classifier + enforcement engine — one detection/policy core, two directions of traffic, not two separate systems.
- A **block** stops the action; a **quarantine** isolates the flagged content (e.g. strips/neutralizes it) rather than passing it through unmodified; a **hold-for-approval** pauses execution until a human explicitly signs off via the approval queue. Nothing self-certifies as "confirmed safe" — see [threat-model.md](threat-model.md) repudiation section for why approvals must be attributed and logged.
- Explainability (SHAP for the LightGBM layer, attention visualization for DeBERTa) was built (Phase 5), spot-checked against real flagged examples (`docs/explainability_report.md`), and — after a real gap was found here during the Phase 13 audit (it was built but never actually wired into any live decision path) — **is now attached to every flagged decision as the diagram shows.** `Classifier.score_and_explain()` computes it alongside the score; `EnforcementEngine` attaches the result to the shared decision object at one point (`src/toolwarden/enforcement/engine.py`), so `guarded_loop.py`, `mcp_proxy/server.py`, and `service/app.py` all inherit it without three separate integrations. A human reviewing a HOLD — via the demo/MCP console output or the Phase 12 dashboard's approvals table and resolve dialog — now sees the top DeBERTa tokens and LightGBM features behind the score, not just the raw number and a templated reason string. Full before/after and what got verified live: `docs/known_limitations.md`.

## Integration sequencing: API-first, MCP second

The interception layer is built first against direct API function-calling (OpenAI/Anthropic `tool_use`), with MCP added later as a second adapter reusing the same core (classifier, enforcement engine, approval queue) — not rebuilt for MCP.

**Why this order, not the reverse:** direct tool-calling APIs expose a stable, versioned, structured JSON contract for both the call and the result — the interception layer can rely on a fixed schema. This deliberately mirrors a lesson from a prior project (ContextGuard AI), where an integration built on scraping/DOM-dependent structure turned out to be fragile: the underlying structure it depended on could shift without warning, breaking the integration in ways that had nothing to do with the actual detection logic. MCP is a real, structured protocol (not DOM scraping) and is the right second target, but it's newer and faster-evolving than the direct-API tool-calling contracts, which have been stable for longer. Proving out the classifier and enforcement engine against the more stable contract first means the core detection logic gets validated against a fixed target before absorbing the added variable of a less mature protocol. Phase 10 then adds MCP as an adapter on top of an already-proven core, rather than building core detection logic against a moving target from day one.

## Component boundaries

| Component | Responsibility | Does NOT do |
|---|---|---|
| Interception layer | Capture outbound tool-call requests and inbound tool-call results, in passthrough mode first (Phase 2) | Make any allow/block decision itself |
| Classifier (DeBERTa + LightGBM ensemble) | Score a tool-call request or result for injection/scope-drift risk | Enforce policy, own the approval queue |
| Explainability | Attach SHAP/attention evidence to a flagged score | Change the classifier's decision |
| Enforcement engine | Map classifier score → allow/block/quarantine/hold, per configurable thresholds | Self-certify a held action as safe — that requires human approval |
| Approval queue | Hold flagged actions, record human decisions with timestamp + identity | Auto-approve anything |

This table will be revisited once Phase 2 (interception layer implementation) starts, in case real implementation constraints change a boundary drawn here on paper.
