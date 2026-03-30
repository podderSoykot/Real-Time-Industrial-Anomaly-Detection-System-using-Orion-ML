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

const epochMs = [];
const values = [];
const scores = [];
const anomalyFlags = [];
/** @type {Array<{ machine_name?: string, place?: string, line?: string, sensor_id?: string, zone?: string, shift?: string, notes?: string }>} */
const pointMeta = [];

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
        pointBackgroundColor: "rgba(248, 113, 113, 0.95)",
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
          color: "#94a3b8",
          font: { family: chartFont, size: 11, weight: "600" },
          boxWidth: 10,
          boxHeight: 10,
          padding: 16,
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
          maxTicksLimit: 10,
          font: { family: chartFont, size: 10 },
        },
        grid: { color: "rgba(51, 65, 85, 0.35)" },
        border: { display: false },
      },
      y: {
        position: "left",
        ticks: { color: "#64748b", font: { family: chartFont, size: 10 } },
        grid: { color: "rgba(51, 65, 85, 0.35)" },
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
        ticks: { color: "#64748b", font: { family: chartFont, size: 10 } },
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
  const agg = aggregateSignal(signalMode, epochMs, values, scores, anomalyFlags, pointMeta);
  chart.data.labels = agg.labels;
  chart.data.datasets[0].data = agg.values;
  chart.data.datasets[1].data = agg.scores;
  displayFlags = agg.flags;
  displayMeta = agg.meta;
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
  if (flag && !silent) {
    anomalySession += 1;
    prependAnomaly({ ...row, timestamp: ts });
    updateSpotlight({ ...row, timestamp: ts });
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
  if (flagged.length > 0) {
    updateSpotlight(flagged[0]);
  } else if (el("anomalySpotlight")) {
    el("anomalySpotlight").hidden = true;
  }
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
}

function renderAnomalyList() {
  const ul = el("anomalyList");
  if (!ul) return;
  ul.innerHTML = "";
  if (anomalyRows.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No anomalies in this session yet.";
    ul.appendChild(li);
    return;
  }
  for (const r of anomalyRows) {
    const li = document.createElement("li");
    li.className = "anomaly-card";

    const top = document.createElement("div");
    top.className = "ac-top";
    const machine = document.createElement("span");
    machine.className = "ac-machine";
    machine.textContent = disp(r.machine_name);
    const time = document.createElement("span");
    time.className = "ac-time";
    time.textContent = shortTime(r.ts);
    top.appendChild(machine);
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
