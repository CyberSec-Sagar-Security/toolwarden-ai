// ToolWarden AI dashboard. Talks only to /api/* (nginx reverse-proxies that
// to the backend container — see docker/nginx.conf), so this file never
// needs a backend hostname/port baked in.
//
// Payload/reason/content values rendered here can contain attacker-supplied
// text (that's the whole point of the dashboard) -- every untrusted value
// is set via textContent, never innerHTML, so a malicious tool-call payload
// can't inject markup into the page that's reviewing it.

const API = "/api/v1";
const POLL_MS = 4000;

function $(sel) { return document.querySelector(sel); }

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `${name}-tab`));
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function cell(tag, text, className) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = text;
  return el;
}

async function fetchJSON(path, options) {
  const res = await fetch(`${API}${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return body;
}

let currentResolveId = null;

function openResolveDialog(pending) {
  currentResolveId = pending.id;
  $("#resolve-summary").textContent = `${pending.direction} — score ${pending.score != null ? pending.score.toFixed(3) : "n/a"} — ${pending.reason}`;
  $("#resolve-decided-by").value = "";
  $("#resolve-notes").value = "";
  $("#resolve-error").textContent = "";
  $("#resolve-dialog").hidden = false;
}

function closeResolveDialog() {
  $("#resolve-dialog").hidden = true;
  currentResolveId = null;
}

$("#resolve-cancel").addEventListener("click", closeResolveDialog);

$("#resolve-confirm").addEventListener("click", async () => {
  const decidedBy = $("#resolve-decided-by").value.trim();
  if (!decidedBy) {
    $("#resolve-error").textContent = "decided_by is required — approvals must be attributed.";
    return;
  }
  try {
    await fetchJSON(`/approvals/${currentResolveId}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: $("#resolve-decision").value,
        decided_by: decidedBy,
        notes: $("#resolve-notes").value.trim(),
      }),
    });
    closeResolveDialog();
    await refreshApprovals();
  } catch (err) {
    $("#resolve-error").textContent = err.message;
  }
});

async function refreshApprovals() {
  let pending;
  try {
    pending = await fetchJSON("/approvals/pending");
  } catch {
    return; // backend not reachable yet -- next poll will retry
  }

  $("#pending-count").textContent = String(pending.length);
  const tbody = $("#approvals-table tbody");
  tbody.innerHTML = "";
  $("#approvals-empty").hidden = pending.length > 0;

  for (const p of pending) {
    const row = document.createElement("tr");
    row.appendChild(cell("td", p.direction));
    row.appendChild(cell("td", p.score != null ? p.score.toFixed(3) : "n/a"));
    row.appendChild(cell("td", p.reason));
    row.appendChild(cell("td", JSON.stringify(p.payload), "payload"));

    const actionCell = document.createElement("td");
    const btn = document.createElement("button");
    btn.className = "resolve-btn";
    btn.textContent = "Review";
    btn.addEventListener("click", () => openResolveDialog(p));
    actionCell.appendChild(btn);
    row.appendChild(actionCell);

    tbody.appendChild(row);
  }
}

async function refreshTraffic() {
  let events;
  try {
    events = await fetchJSON("/traffic?limit=50");
  } catch {
    return;
  }

  const tbody = $("#traffic-table tbody");
  tbody.innerHTML = "";
  for (const e of events) {
    const row = document.createElement("tr");
    row.appendChild(cell("td", e.event || ""));
    row.appendChild(cell("td", e.tool_name || ""));
    const detail = e.event === "tool_call_request" ? e.arguments : e.content;
    row.appendChild(cell("td", typeof detail === "string" ? detail : JSON.stringify(detail), "detail"));
    tbody.appendChild(row);
  }
}

function pollLoop() {
  refreshApprovals();
  refreshTraffic();
  setTimeout(pollLoop, POLL_MS);
}

pollLoop();
