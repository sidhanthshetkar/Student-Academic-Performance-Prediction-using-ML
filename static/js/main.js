// Shared helpers and frontend validation for all pages.

const COLORS = { Low: "#dc2626", Medium: "#d97706", High: "#059669", Unpredicted: "#9ca3af" };

// Bootstrap-style client-side validation for any form with .needs-validation
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form.needs-validation").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
      }
      form.classList.add("was-validated");
    });
  });

  // Confirmation dialogs for destructive actions
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (!confirm(form.dataset.confirm)) e.preventDefault();
    });
  });
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Render recommendation list (used by the prediction page; mirrors the server-side macro)
function renderRecommendations(recs) {
  return recs.map((r, i) => `
    <div class="rec rec-${r.priority}">
      <div class="d-flex justify-content-between align-items-start flex-wrap gap-1">
        <h6 class="mb-1"><span class="rec-num">${i + 1}</span> ${escapeHtml(r.title)}</h6>
        <span class="prio prio-${r.priority}">${r.priority} priority</span>
      </div>
      <div class="small text-muted mb-2">Current: <strong>${escapeHtml(r.current)}</strong>
        ${r.target ? ` · Suggested target: <strong>${escapeHtml(r.target)}</strong>` : ""}</div>
      <ul class="mb-2 small">${r.actions.map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>
      <div class="why"><i class="fa-regular fa-lightbulb me-1"></i><strong>Why:</strong> ${escapeHtml(r.reason)}</div>
    </div>`).join("");
}

function renderFactorBars(rows) {
  return rows.map((b) => {
    const cls = b.status.split(" ")[0];
    const unit = b.factor === "attendance" || b.factor === "previous_semester" ? "%" :
                 b.factor === "study_hours" ? " h" : ` / ${b.max}`;
    return `
    <div class="mb-3">
      <div class="d-flex justify-content-between small">
        <span class="fw-semibold">${escapeHtml(b.label)}</span>
        <span>${b.value}${unit} · <span class="status status-${cls}">${b.status}</span></span>
      </div>
      <div class="progress" style="height:10px"><div class="progress-bar bar-${cls}" style="width:${b.percent}%"></div></div>
      ${b.note ? `<div class="text-muted tiny">${b.note}</div>` : ""}
    </div>`;
  }).join("");
}
