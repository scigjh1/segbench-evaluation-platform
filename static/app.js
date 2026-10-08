const state = { result: null, rows: [], runNumber: Number(localStorage.getItem("segbench-run") || "0") };
const byId = (id) => document.getElementById(id);
const elements = {
  tolerance: byId("tolerance"), toleranceValue: byId("toleranceValue"), runButton: byId("runButton"),
  refreshButton: byId("refreshButton"), exportButton: byId("exportButton"), resultBody: byId("resultBody"),
  metricBars: byId("metricBars"), decisionBadge: byId("decisionBadge"), decisionText: byId("decisionText"),
  runId: byId("runId"), runStatus: byId("runStatus"), searchInput: byId("searchInput"), tableCount: byId("tableCount"),
  toast: byId("toast"), engine: byId("engine"),
};

function showToast(message) { elements.toast.textContent = message; elements.toast.classList.add("show"); clearTimeout(showToast.timer); showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 2300); }
function formatMetric(value) { return Number(value).toFixed(3); }
function formatDelta(value, inverse = false) { const signed = Number(value); const good = inverse ? signed >= 0 : signed >= 0; return `${signed >= 0 ? "+" : ""}${signed.toFixed(3)}`; }

function renderSummary(result) {
  const summary = result.candidate.summary;
  const delta = result.delta;
  byId("dice").textContent = formatMetric(summary.dice); byId("iou").textContent = formatMetric(summary.iou);
  byId("boundaryF1").textContent = formatMetric(summary.boundaryF1); byId("hd95").textContent = summary.hd95.toFixed(2);
  byId("throughput").textContent = summary.throughputFps.toFixed(1); byId("p95").textContent = summary.p95LatencyMs.toFixed(2);
  const deltas = [["diceDelta", delta.dice, false], ["iouDelta", delta.iou, false], ["boundaryDelta", delta.boundaryF1, false], ["hd95Delta", delta.hd95, true]];
  for (const [id, value] of deltas) { const item = byId(id); item.textContent = `${formatDelta(value)} vs baseline`; item.className = value >= 0 ? "positive" : "negative"; }
  byId("sampleCount").textContent = `${summary.samples} 对`;
  byId("precision").textContent = formatMetric(summary.precision); byId("recall").textContent = formatMetric(summary.recall);
  byId("meanLatency").textContent = `${summary.meanLatencyMs.toFixed(2)} ms`;
  const passed = delta.dice >= 0 && delta.iou >= 0 && delta.boundaryF1 >= 0 && delta.hd95 >= 0;
  elements.decisionBadge.textContent = passed ? "通过回归" : "需要复核"; elements.decisionBadge.className = `badge ${passed ? "pass" : "fail"}`;
  elements.decisionText.textContent = passed
    ? `Candidate 在 ${summary.samples} 组样例上未出现核心指标回退，Dice 提升 ${delta.dice.toFixed(3)}，Boundary F1 提升 ${delta.boundaryF1.toFixed(3)}，HD95 改善 ${delta.hd95.toFixed(2)} px。`
    : "Candidate 存在核心指标回退，建议定位失败样例后再进入发布流程。";
}

function renderBars(result) {
  const labels = { dice: "Dice", iou: "IoU", precision: "Precision", recall: "Recall", boundaryF1: "Boundary F1" };
  elements.metricBars.innerHTML = Object.entries(labels).map(([key, label]) => {
    const baseline = result.baseline.summary[key]; const candidate = result.candidate.summary[key];
    return `<div class="bar-row"><span>${label}</span><div class="bar-stack"><div class="bar-track"><div class="bar-fill" style="width:${baseline * 100}%"></div></div><div class="bar-track"><div class="bar-fill candidate" style="width:${candidate * 100}%"></div></div></div><div class="bar-values">${baseline.toFixed(3)}<br>${candidate.toFixed(3)}</div></div>`;
  }).join("");
}

function mergedRows(result) {
  const baseline = new Map(result.baseline.rows.map((row) => [row.name, row]));
  return result.candidate.rows.map((candidate) => ({ candidate, baseline: baseline.get(candidate.name) }));
}

function renderRows() {
  const query = elements.searchInput.value.trim().toLowerCase();
  const visible = state.rows.filter(({ candidate }) => candidate.name.toLowerCase().includes(query));
  elements.tableCount.textContent = `${visible.length} 条`;
  if (!visible.length) { elements.resultBody.innerHTML = '<tr><td class="empty-row" colspan="7">没有匹配结果</td></tr>'; return; }
  elements.resultBody.innerHTML = visible.map(({ baseline, candidate }) => {
    const delta = candidate.dice - baseline.dice;
    return `<tr><td>${candidate.name}</td><td>${baseline.dice.toFixed(4)}</td><td>${candidate.dice.toFixed(4)}</td><td class="${delta >= 0 ? "delta-positive" : ""}">${delta >= 0 ? "+" : ""}${delta.toFixed(4)}</td><td>${candidate.iou.toFixed(4)}</td><td>${candidate.boundaryF1.toFixed(4)}</td><td>${candidate.hd95.toFixed(2)}</td></tr>`;
  }).join("");
}

async function runEvaluation() {
  elements.runButton.disabled = true; elements.runStatus.textContent = "运行中";
  try {
    const response = await fetch(`/api/sample-evaluation?tolerance=${elements.tolerance.value}`);
    const payload = await response.json(); if (!payload.ok) throw new Error(payload.error);
    state.result = payload.result; state.rows = mergedRows(payload.result); state.runNumber += 1; localStorage.setItem("segbench-run", state.runNumber);
    elements.runId.textContent = `EVAL-${String(state.runNumber).padStart(4, "0")}`; elements.runStatus.textContent = "已完成";
    renderSummary(payload.result); renderBars(payload.result); renderRows(); elements.exportButton.disabled = false;
    showToast(`已评测 ${payload.result.candidate.summary.samples} 组掩膜`);
  } catch (error) { elements.runStatus.textContent = "失败"; showToast(error.message); }
  finally { elements.runButton.disabled = false; }
}

function exportCsv() {
  if (!state.rows.length) return;
  const header = ["sample", "baseline_dice", "candidate_dice", "delta_dice", "candidate_iou", "candidate_precision", "candidate_recall", "candidate_boundary_f1", "candidate_hd95"];
  const rows = state.rows.map(({ baseline, candidate }) => [candidate.name, baseline.dice, candidate.dice, candidate.dice - baseline.dice, candidate.iou, candidate.precision, candidate.recall, candidate.boundaryF1, candidate.hd95]);
  const csv = [header, ...rows].map((row) => row.join(",")).join("\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" })); link.download = `${elements.runId.textContent}-segbench.csv`; link.click(); URL.revokeObjectURL(link.href);
}

elements.tolerance.addEventListener("input", () => { elements.toleranceValue.textContent = `${elements.tolerance.value} px`; });
elements.runButton.addEventListener("click", runEvaluation); elements.refreshButton.addEventListener("click", runEvaluation); elements.exportButton.addEventListener("click", exportCsv); elements.searchInput.addEventListener("input", renderRows);
async function initialize() { if (window.lucide) window.lucide.createIcons(); try { const health = await (await fetch("/api/health")).json(); elements.engine.classList.add("online"); elements.engine.lastChild.textContent = health.engine; } catch { elements.engine.lastChild.textContent = "引擎离线"; } await runEvaluation(); }
initialize();
