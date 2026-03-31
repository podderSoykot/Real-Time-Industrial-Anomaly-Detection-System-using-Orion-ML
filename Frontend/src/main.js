import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

const MAX_POINTS = 400;

/** @type {'second' | '30s' | 'minute'} */
let signalMode = "second";

/** Display series for tooltips / anomaly markers (aligned with chart datasets). */
let displayFlags = [];
let displayMeta = [];
let displayScores = [];

const el = (id) => document.getElementById(id);

function parseEpoch(ts) {
  if (ts == null) return Date.now();
  const t = typeof ts === "string" ? new Date(ts) : new Date(ts);
  const ms = t.getTime();
  return Number.isNaN(ms) ? Date.now() : ms;
}

/**
 * Bucket raw points into second / 30s / minute windows (mean value & score; anomaly if any in bucket).
 */
function aggregateSignal(mode, epochMs, vals, scs, flags, meta) {
  const n = epochMs.length;
  if (n === 0) {
    return { labels: [], values: [], scores: [], flags: [], meta: [] };
  }
  const size =
    mode === "second" ? 1000 : mode === "30s" ? 30000 : 60000;
  const maxBuckets = mode === "second" ? 90 : mode === "30s" ? 40 : 60;

  const buckets = new Map();
  for (let i = 0; i < n; i++) {
    const k = Math.floor(epochMs[i] / size) * size;
    if (!buckets.has(k)) {
      buckets.set(k, { sumV: 0, sumS: 0, count: 0, anyA: false, lastMeta: meta[i] });
    }
    const b = buckets.get(k);
    b.sumV += vals[i];
    b.sumS += scs[i];
    b.count += 1;
    b.anyA = b.anyA || flags[i];
    b.lastMeta = meta[i];
  }
  const keys = [...buckets.keys()].sort((a, b) => a - b);
  const take = keys.slice(-maxBuckets);
  const labels = [];
  const outV = [];
  const outS = [];
  const outF = [];
  const outM = [];
  for (const k of take) {
    const b = buckets.get(k);
    labels.push(formatBucketLabel(k, mode));
    outV.push(b.sumV / b.count);
    outS.push(b.sumS / b.count);
    outF.push(b.anyA);
    outM.push(b.lastMeta);
  }
  return { labels, values: outV, scores: outS, flags: outF, meta: outM };
}

function formatBucketLabel(ms, mode) {
  const d = new Date(ms);
  if (mode === "minute") {
    return d.toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function httpToWs(url) {
  const u = new URL(url);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = u.pathname.replace(/\/$/, "") + "/stream/ws";
  u.search = "";
  u.hash = "";
  return u.toString();
}

function shortTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(11, 19);
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function disp(v) {
  if (v == null || v === "") return "—";
  return String(v);
}

function setFooter(msg) {
  if (el("footerMsg")) el("footerMsg").textContent = msg;
}

function setKpi(id, text, statusClass) {
  const node = el(id);
  if (!node) return;
  node.textContent = text;
  node.className = "kpi-value" + (id === "kpiLastVal" ? " mono" : "");
  if (statusClass) node.classList.add(statusClass);
}

let ws = null;
let wsPingTimer = null;
let healthTimer = null;
let anomalySession = 0;

let currentDetectMode = null;
let currentDbscanScoreThreshold = null;

const epochMs = [];
const values = [];
const scores = [];
const anomalyFlags = [];
/** @type {Array<{ machine_name?: string, place?: string, line?: string, sensor_id?: string, zone?: string, shift?: string, notes?: string }>} */
const pointMeta = [];

let totalPointsSeen = 0;

const filterMachine = el("filterMachine");
const filterSensor = el("filterSensor");
const btnFilterClear = el("btnFilterClear");
let lastFilterOptionsAt = 0;

function getFilterState() {
  const sensor = filterSensor?.value ?? "";
  const machine = filterMachine?.value ?? "";
  return { sensor, machine };
}

function matchesFilter(meta) {
  if (!meta) return true;
  const { sensor, machine } = getFilterState();
  if (sensor) return meta.sensor_id === sensor;
  if (machine) return meta.machine_name === machine;
  return true;
}

function getFilteredSeries() {
  const { sensor, machine } = getFilterState();
  const fe = [];
  const fv = [];
  const fs = [];
  const ff = [];
  const fm = [];
  for (let i = 0; i < epochMs.length; i++) {
    const m = pointMeta[i] || {};
    const ok = sensor ? m.sensor_id === sensor : machine ? m.machine_name === machine : true;
    if (!ok) continue;
    fe.push(epochMs[i]);
    fv.push(values[i]);
    fs.push(scores[i]);
    ff.push(anomalyFlags[i]);
    fm.push(m);
  }
  return { epochMs: fe, values: fv, scores: fs, anomalyFlags: ff, meta: fm };
}

function repopulateSelect(selectEl, defaultLabel, values, keepValue) {
  if (!selectEl) return;
  const keepOk = keepValue && values.includes(keepValue);
  selectEl.innerHTML = "";
  const optAll = document.createElement("option");
  optAll.value = "";
  optAll.textContent = defaultLabel;
  selectEl.appendChild(optAll);
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }
  selectEl.value = keepOk ? keepValue : "";
}

function updateFilterOptions() {
  if (!filterMachine || !filterSensor) return;
  const machines = new Set();
  const sensors = new Set();
  for (const m of pointMeta) {
    if (!m) continue;
    if (m.machine_name) machines.add(m.machine_name);
    if (m.sensor_id) sensors.add(m.sensor_id);
  }

  const machineArr = [...machines].sort();
  const sensorArr = [...sensors].sort();
  const { sensor, machine } = getFilterState();

  repopulateSelect(filterMachine, "All machines", machineArr, machine);
  repopulateSelect(filterSensor, "All sensors", sensorArr, sensor);
}

const chartFont = "'DM Sans', system-ui, sans-serif";

const chart = new Chart(el("chartLive"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Value",
        data: [],
        borderColor: "rgba(34, 211, 238, 0.95)",
        backgroundColor: (ctx) => {
          const c = ctx.chart.ctx;
          const { chartArea } = ctx.chart;
          if (!chartArea) return "rgba(34, 211, 238, 0.06)";
          const g = c.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
          g.addColorStop(0, "rgba(34, 211, 238, 0.02)");
          g.addColorStop(1, "rgba(34, 211, 238, 0.14)");
          return g;
        },
        fill: true,
        tension: 0.28,
        borderWidth: 2.5,
        pointRadius: (ctx) => (displayFlags[ctx.dataIndex] ? 6 : 0),
        pointHoverRadius: 7,
        pointBackgroundColor: (ctx) => {
          const i = ctx.dataIndex;
          if (!displayFlags[i]) return "rgba(0, 0, 0, 0)";
          return severityToColor(scoreToSeverity(displayScores[i]));
        },
        pointBorderColor: "#fff",
        pointBorderWidth: 1,
      },
      {
        label: "Score",
        data: [],
        borderColor: "rgba(251, 146, 60, 0.92)",
        backgroundColor: "transparent",
        borderWidth: 1.75,
        tension: 0.25,
        yAxisID: "y1",
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        position: "top",
        align: "end",
        labels: {
          color: "#cbd5e1",
          font: { family: chartFont, size: 12, weight: "700" },
          boxWidth: 9,
          boxHeight: 9,
          padding: 14,
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: "rgba(15, 23, 42, 0.94)",
        titleColor: "#f1f5f9",
        bodyColor: "#cbd5e1",
        borderColor: "rgba(148, 163, 184, 0.25)",
        borderWidth: 1,
        padding: 12,
        cornerRadius: 10,
        titleFont: { family: chartFont, size: 12, weight: "600" },
        bodyFont: { family: chartFont, size: 12 },
        callbacks: {
          afterBody: (items) => {
            const it = items[0];
            if (!it) return [];
            const m = displayMeta[it.dataIndex];
            if (!m) return [];
            const lines = [
              "",
              `Machine: ${disp(m.machine_name)}`,
              `Place: ${disp(m.place)}`,
              `Line: ${disp(m.line)}`,
              `Sensor: ${disp(m.sensor_id)}`,
              `Zone: ${disp(m.zone)}`,
              `Shift: ${disp(m.shift)}`,
            ];
            if (m.notes) lines.push(`Notes: ${m.notes}`);
            return lines;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: "#64748b",
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 9,
          font: { family: chartFont, size: 11, weight: "600" },
        },
        grid: { color: "rgba(71, 85, 105, 0.26)" },
        border: { display: false },
      },
      y: {
        position: "left",
        ticks: { color: "#94a3b8", font: { family: chartFont, size: 11, weight: "600" } },
        grid: { color: "rgba(71, 85, 105, 0.26)" },
        border: { display: false },
        title: {
          display: true,
          text: "Value",
          color: "#22d3ee",
          font: { family: chartFont, size: 11, weight: "600" },
        },
      },
      y1: {
        position: "right",
        grid: { drawOnChartArea: false },
        ticks: { color: "#94a3b8", font: { family: chartFont, size: 11, weight: "600" } },
        border: { display: false },
        title: {
          display: true,
          text: "Score",
          color: "#fb923c",
          font: { family: chartFont, size: 11, weight: "600" },
        },
      },
    },
  },
});

function refreshLiveChart() {
  const f = getFilteredSeries();
  const agg = aggregateSignal(signalMode, f.epochMs, f.values, f.scores, f.anomalyFlags, f.meta);
  chart.data.labels = agg.labels;
  chart.data.datasets[0].data = agg.values;
  chart.data.datasets[1].data = agg.scores;
  displayFlags = agg.flags;
  displayMeta = agg.meta;
  displayScores = agg.scores;
  chart.update("none");
}

function setSignalMode(mode) {
  if (mode !== "second" && mode !== "30s" && mode !== "minute") return;
  signalMode = mode;
  document.querySelectorAll(".signal-tab").forEach((btn) => {
    const m = btn.getAttribute("data-mode");
    const on = m === mode;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
  const hint = el("chartResolutionHint");
  if (hint) {
    const copy = {
      second: "1s: one bucket per second (~90 s window). Cyan = value · Amber = score",
      "30s": "30s: one bucket per 30 seconds (~20 min window). Cyan = value · Amber = score",
      minute: "1m: one bucket per minute (~60 min window). Cyan = value · Amber = score",
    };
    hint.textContent = copy[mode];
  }
  refreshLiveChart();
}

function trimBuffers() {
  while (epochMs.length > MAX_POINTS) {
    epochMs.shift();
    values.shift();
    scores.shift();
    anomalyFlags.shift();
    pointMeta.shift();
  }
}

function isAnomalyFlag(row) {
  const v = row.is_anomaly;
  return v === true || v === 1 || v === "true";
}

function scoreToSeverity(score) {
  const s = Number(score);
  if (!Number.isFinite(s)) return "low";

  if (currentDetectMode === "dbscan_cluster" && Number.isFinite(currentDbscanScoreThreshold)) {
    const t = Number(currentDbscanScoreThreshold);
    // DBSCAN score is distance-like and can vary by dataset/model.
    // Use adaptive thresholds over recent scores above baseline threshold.
    const pool = scores
      .slice(-240)
      .filter((v) => Number.isFinite(v) && v >= t)
      .sort((a, b) => a - b);
    if (pool.length >= 10) {
      const p = (q) => {
        const idx = Math.min(pool.length - 1, Math.max(0, Math.floor(q * (pool.length - 1))));
        return pool[idx];
      };
      const medT = Math.max(t, p(0.50));
      const highT = Math.max(medT, p(0.85));
      if (s >= highT) return "high";
      if (s >= medT) return "med";
      return "low";
    }
    // Early fallback before enough samples accumulate.
    if (s >= t * 2.2) return "high";
    if (s >= t * 1.35) return "med";
    return "low";
  }

  // Adaptive UI thresholds for rolling / unknown score scales.
  // Prevents "all red" when score magnitudes are higher than expected.
  const recent = scores.slice(-120).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (recent.length >= 12) {
    const p = (q) => {
      const idx = Math.min(recent.length - 1, Math.max(0, Math.floor(q * (recent.length - 1))));
      return recent[idx];
    };
    const medT = p(0.55);
    const highT = p(0.82);
    if (s >= highT) return "high";
    if (s >= medT) return "med";
    return "low";
  }

  // Fallback defaults until enough points exist.
  if (s >= 3) return "high";
  if (s >= 1.5) return "med";
  return "low";
}

function severityToColor(sev) {
  if (sev === "high") return "rgba(248, 113, 113, 0.95)"; // red
  if (sev === "med") return "rgba(250, 204, 21, 0.95)"; // yellow
  return "rgba(34, 197, 94, 0.95)"; // green
}

function severityDisplay(sev) {
  if (sev === "high") return "RED";
  if (sev === "med") return "YELLOW";
  return "GREEN";
}

function updateAnomalySummary(rows) {
  let low = 0;
  let med = 0;
  let high = 0;
  for (const r of rows) {
    const sev = scoreToSeverity(r.score);
    if (sev === "high") high += 1;
    else if (sev === "med") med += 1;
    else low += 1;
  }
  if (el("sumShown")) el("sumShown").textContent = String(rows.length);
  if (el("sumLow")) el("sumLow").textContent = String(low);
  if (el("sumMed")) el("sumMed").textContent = String(med);
  if (el("sumHigh")) el("sumHigh").textContent = String(high);
}

function pushPoint(row, { silent = false, skipChart = false } = {}) {
  const ts = row.timestamp || row.t;
  epochMs.push(parseEpoch(ts));
  values.push(Number(row.value));
  scores.push(Number(row.score ?? 0));
  pointMeta.push({
    machine_name: row.machine_name,
    place: row.place,
    line: row.line,
    sensor_id: row.sensor_id,
    zone: row.zone,
    shift: row.shift,
    notes: row.notes,
  });
  const flag = isAnomalyFlag(row);
  anomalyFlags.push(flag);

  totalPointsSeen += 1;
  if (!silent && totalPointsSeen - lastFilterOptionsAt >= 40) {
    updateFilterOptions();
    lastFilterOptionsAt = totalPointsSeen;
  }

  if (flag && !silent) {
    anomalySession += 1;
    prependAnomaly({ ...row, timestamp: ts });
  }
  trimBuffers();
  setKpi("kpiLastVal", values[values.length - 1]?.toFixed(2) ?? "—");
  if (!silent) setKpi("kpiAnomalies", String(anomalySession));
  if (!skipChart) refreshLiveChart();
}

function ingestHistory(points) {
  epochMs.length = 0;
  values.length = 0;
  scores.length = 0;
  anomalyFlags.length = 0;
  pointMeta.length = 0;
  anomalySession = 0;

  for (const p of points) {
    pushPoint(p, { silent: true, skipChart: true });
  }
  anomalySession = points.filter((p) => isAnomalyFlag(p)).length;
  updateFilterOptions();
  const flagged = points
    .filter((p) => isAnomalyFlag(p))
    .slice(-12)
    .reverse();
  anomalyRows.length = 0;
  for (const p of flagged) {
    anomalyRows.push({
      ts: p.timestamp,
      value: p.value,
      score: p.score ?? 0,
      machine_name: p.machine_name,
      place: p.place,
      line: p.line,
      sensor_id: p.sensor_id,
      zone: p.zone,
      shift: p.shift,
      notes: p.notes,
    });
  }
  renderAnomalyList();
  setKpi("kpiLastVal", values[values.length - 1]?.toFixed(2) ?? "—");
  setKpi("kpiAnomalies", String(anomalySession));
  refreshLiveChart();
  setFooter(`Loaded ${points.length} buffered points.`);
}

const anomalyRows = [];

function prependAnomaly(row) {
  const ts = row.timestamp || row.t;
  anomalyRows.unshift({
    ts: ts || "",
    value: row.value,
    score: row.score ?? 0,
    machine_name: row.machine_name,
    place: row.place,
    line: row.line,
    sensor_id: row.sensor_id,
    zone: row.zone,
    shift: row.shift,
    notes: row.notes,
  });
  while (anomalyRows.length > 12) anomalyRows.pop();
  renderAnomalyList();
}

function updateSpotlight(row) {
  const box = el("anomalySpotlight");
  if (!box) return;
  box.hidden = false;
  if (el("spotMachine")) el("spotMachine").textContent = disp(row.machine_name);
  if (el("spotPlace")) el("spotPlace").textContent = disp(row.place);
  if (el("spotLine")) el("spotLine").textContent = disp(row.line);
  if (el("spotSensor")) el("spotSensor").textContent = disp(row.sensor_id);
  if (el("spotZone")) el("spotZone").textContent = disp(row.zone);
  if (el("spotShift")) el("spotShift").textContent = disp(row.shift);
  const ts = row.timestamp || row.t;
  const met = `${shortTime(ts)} · value ${Number(row.value).toFixed(3)} · score ${Number(row.score ?? 0).toFixed(3)}`;
  if (el("spotMetrics")) el("spotMetrics").textContent = met;
  if (el("spotNotes")) el("spotNotes").textContent = disp(row.notes);

  const sev = scoreToSeverity(row.score ?? 0);
  const badge = box.querySelector(".spotlight-badge");
  if (badge) {
    badge.classList.remove("spotlight-badge--low", "spotlight-badge--med", "spotlight-badge--high");
    badge.classList.add(`spotlight-badge--${sev}`);
    badge.textContent = `Latest anomaly · ${severityDisplay(sev)}`;
  }
}

function renderAnomalyList() {
  const ul = el("anomalyList");
  if (!ul) return;
  ul.innerHTML = "";
  const displayed = anomalyRows.filter((r) => matchesFilter(r));
  updateAnomalySummary(displayed);
  if (anomalyRows.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No anomalies in this session yet.";
    ul.appendChild(li);
    const box = el("anomalySpotlight");
    if (box) box.hidden = true;
    return;
  }
  if (displayed.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No anomalies match the current filter.";
    ul.appendChild(li);
    const box = el("anomalySpotlight");
    if (box) box.hidden = true;
    return;
  }

  for (const r of displayed) {
    const sev = scoreToSeverity(r.score);
    const li = document.createElement("li");
    li.className = `anomaly-card anomaly-card--${sev}`;

    const top = document.createElement("div");
    top.className = "ac-top";
    const machine = document.createElement("span");
    machine.className = "ac-machine";
    machine.textContent = disp(r.machine_name);
    const sevPill = document.createElement("span");
    sevPill.className = `ac-sev ac-sev--${sev}`;
    sevPill.textContent = severityDisplay(sev);
    const time = document.createElement("span");
    time.className = "ac-time";
    time.textContent = shortTime(r.ts);
    top.appendChild(machine);
    top.appendChild(sevPill);
    top.appendChild(time);

    const place = document.createElement("div");
    place.className = "ac-place";
    place.textContent = disp(r.place);

    const row2 = document.createElement("div");
    row2.className = "ac-row";
    const line = document.createElement("span");
    line.className = "ac-meta";
    line.textContent = `Line: ${disp(r.line)}`;
    const sensor = document.createElement("span");
    sensor.className = "ac-meta mono";
    sensor.textContent = `Sensor: ${disp(r.sensor_id)}`;
    row2.appendChild(line);
    row2.appendChild(sensor);

    const row3 = document.createElement("div");
    row3.className = "ac-row";
    const zone = document.createElement("span");
    zone.className = "ac-meta";
    zone.textContent = `Zone: ${disp(r.zone)}`;
    const shift = document.createElement("span");
    shift.className = "ac-meta";
    shift.textContent = `Shift: ${disp(r.shift)}`;
    row3.appendChild(zone);
    row3.appendChild(shift);

    const metrics = document.createElement("div");
    metrics.className = "ac-metrics";
    metrics.textContent = `Value ${Number(r.value).toFixed(3)} · score ${Number(r.score).toFixed(3)}`;

    li.appendChild(top);
    li.appendChild(place);
    li.appendChild(row2);
    li.appendChild(row3);
    li.appendChild(metrics);

    if (r.notes) {
      const note = document.createElement("div");
      note.className = "ac-notes";
      note.textContent = String(r.notes);
      li.appendChild(note);
    }

    ul.appendChild(li);
  }

  // Keep spotlight consistent with the current filter.
  updateSpotlight(displayed[0]);
}

renderAnomalyList();

function onWsMessage(ev) {
  let msg;
  try {
    msg = JSON.parse(ev.data);
  } catch {
    return;
  }
  if (msg.type === "history" && Array.isArray(msg.points)) {
    ingestHistory(msg.points);
    return;
  }
  if (msg.type === "point") {
    pushPoint(msg);
    return;
  }
  if (msg.type === "ack") {
    // Optional: could update a "last ack" KPI; keep it quiet to avoid UI noise.
    return;
  }
  if (msg.type === "ping") {
    return;
  }
}

function connectWs() {
  const base = el("apiBase").value.trim();
  if (!base) {
    setFooter("Set API base URL first.");
    return;
  }
  const url = httpToWs(base);
  disconnectWs();
  setKpi("kpiWs", "…", "status-warn");
  setFooter(`Connecting to ${url} …`);

  ws = new WebSocket(url);
  ws.onopen = () => {
    setKpi("kpiWs", "Live", "status-ok");
    el("btnWs").disabled = true;
    el("btnWsStop").disabled = false;
    setFooter("Stream connected. Waiting for data…");
    pollStreamHealth();
  };
  ws.onmessage = onWsMessage;
  ws.onerror = () => {
    setKpi("kpiWs", "Error", "status-bad");
    setFooter("WebSocket error. Is the API running?");
  };
  ws.onclose = () => {
    setKpi("kpiWs", "Off", "");
    el("btnWs").disabled = false;
    el("btnWsStop").disabled = true;
    if (wsPingTimer) {
      clearInterval(wsPingTimer);
      wsPingTimer = null;
    }
  };
}

function disconnectWs() {
  if (ws) {
    ws.close();
    ws = null;
  }
  setKpi("kpiWs", "Off", "");
  el("btnWs").disabled = false;
  el("btnWsStop").disabled = true;
}

async function pollStreamHealth() {
  const base = el("apiBase").value.trim();
  if (!base) return;
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/stream/health`);
    if (!r.ok) throw new Error(String(r.status));
    const j = await r.json();
    setKpi("kpiBuffered", String(j.buffered_points ?? "—"));
    setKpi("kpiClients", String(j.clients ?? "—"));
    setKpi("kpiDetectMode", disp(j.detect_mode ?? "—"));

    currentDetectMode = j.detect_mode ?? null;
    currentDbscanScoreThreshold = j.dbscan_score_threshold ?? null;

    if (j.model_loaded === true) {
      const src = j.model_source ? ` (${j.model_source})` : "";
      setKpi("kpiModelLoaded", `Loaded${src}`, "status-ok");
    } else {
      setKpi("kpiModelLoaded", j.model_source ? `Not loaded (${j.model_source})` : "Not loaded", "status-warn");
    }
    const wsLive = el("kpiWs")?.textContent === "Live";
    if (
      wsLive &&
      Number(j.buffered_points) === 0 &&
      j.synthetic === false
    ) {
      setFooter(
        "Stream connected but buffer is empty. Run: python -m app.Sent_data_over_stream — or unset STREAM_SYNTHETIC=0 on the API."
      );
    }
  } catch {
    setKpi("kpiBuffered", "—");
    setKpi("kpiClients", "—");
    setKpi("kpiDetectMode", "—");
    setKpi("kpiModelLoaded", "—");
    currentDetectMode = null;
    currentDbscanScoreThreshold = null;
  }
}

async function checkApiHealth() {
  const base = el("apiBase").value.trim();
  if (!base) {
    setFooter("Set API base URL first.");
    return false;
  }
  setFooter("Checking API…");
  try {
    const r = await fetch(`${base.replace(/\/$/, "")}/health`);
    if (!r.ok) throw new Error(String(r.status));
    const j = await r.json();
    const ok = j.status === "ok";
    setKpi("kpiApi", ok ? "OK" : "?", ok ? "status-ok" : "status-warn");
    setFooter(
      `API OK · pretrained Orion: ${j.pretrained_orion_loaded ? "yes" : "no"}`
    );
    return true;
  } catch {
    setKpi("kpiApi", "Down", "status-bad");
    setFooter("Cannot reach /health. Start: uvicorn app.api:app --reload");
    return false;
  }
}

async function bootstrap() {
  const ok = await checkApiHealth();
  if (ok) {
    connectWs();
  }
}

el("btnWs").addEventListener("click", connectWs);
el("btnWsStop").addEventListener("click", disconnectWs);
el("btnHealth").addEventListener("click", async () => {
  const ok = await checkApiHealth();
  if (ok) connectWs();
});

if (filterMachine) {
  filterMachine.addEventListener("change", () => {
    // If machine is selected, clear sensor selection for unambiguous filtering.
    if (filterMachine.value) {
      if (filterSensor) filterSensor.value = "";
    }
    refreshLiveChart();
    renderAnomalyList();
  });
}

if (filterSensor) {
  filterSensor.addEventListener("change", () => {
    if (filterSensor.value) {
      if (filterMachine) filterMachine.value = "";
    }
    refreshLiveChart();
    renderAnomalyList();
  });
}

if (btnFilterClear) {
  btnFilterClear.addEventListener("click", () => {
    if (filterMachine) filterMachine.value = "";
    if (filterSensor) filterSensor.value = "";
    refreshLiveChart();
    renderAnomalyList();
  });
}

if (healthTimer) clearInterval(healthTimer);
healthTimer = setInterval(pollStreamHealth, 4000);

document.querySelectorAll(".signal-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const m = btn.getAttribute("data-mode");
    if (m) setSignalMode(m);
  });
});
setSignalMode("second");

bootstrap();
