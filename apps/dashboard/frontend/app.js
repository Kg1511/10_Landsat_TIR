const API_BASE =
  window.DASHBOARD_API_BASE ||
  (window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "");

const requiredDownloads = [
  ["tir100m_npy", "*_pred_tir100m_512.npy"],
  ["rgb_original_npy", "*_pred_rgb_chw_original_scale.npy"],
  ["bgr_tif", "*_pred_bgr_chw.tif"],
  ["preview_png", "*_preview.png"],
  ["manifest_json", "inference_manifest.json"],
];

let selectedFile = null;
let lastInspect = null;
let currentLogs = ["Waiting for upload"];

const el = (id) => document.getElementById(id);

function apiUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const num = Number(value);
  if (Math.abs(num) >= 1000 || Math.abs(num) < 0.01) return num.toExponential(3);
  return num.toFixed(4);
}

function formatShape(shape) {
  if (!shape || !shape.length) return "-";
  return shape.join(" x ");
}

function formatRange(range) {
  if (!range || range.length !== 2) return "-";
  return `${formatNumber(range[0])} to ${formatNumber(range[1])}`;
}

function setStatus(text, state) {
  const badge = el("statusBadge");
  badge.textContent = text;
  badge.className = `status-badge ${state}`;
}

function setLogs(lines) {
  currentLogs = lines.length ? lines : currentLogs;
  el("logsPanel").textContent = currentLogs.join("\n");
}

function appendLog(line) {
  setLogs([...currentLogs, line]);
}

function setBusy(isBusy) {
  const button = el("runButton");
  button.disabled = isBusy || !selectedFile || !lastInspect?.valid;
  button.textContent = isBusy ? "Running..." : "Run Final Pipeline";
}

function updateSummary(data) {
  lastInspect = data;
  el("fileName").textContent = data.filename || selectedFile?.name || "No file selected";
  el("shapeValue").textContent = formatShape(data.shape);
  el("dtypeValue").textContent = data.dtype || "-";
  el("minValue").textContent = formatNumber(data.min);
  el("maxValue").textContent = formatNumber(data.max);
  el("meanValue").textContent = formatNumber(data.mean);
  el("stdValue").textContent = formatNumber(data.std);

  const validation = el("validationMessage");
  validation.textContent = data.message || "Waiting for upload";
  validation.className = `validation ${data.valid ? "valid" : "invalid"}`;
  setBusy(false);
}

function resetOutputs() {
  setImage("rawPreview", "rawEmpty", null);
  setImage("srPreview", "srEmpty", null);
  setImage("rgbPreview", "rgbEmpty", null);
  el("rawRange").textContent = "-";
  el("srRange").textContent = "-";
  el("rgbRange").textContent = "-";
  el("rawShape").textContent = "256 x 256";
  el("srShape").textContent = "512 x 512";
  el("rgbShape").textContent = "3 x 512 x 512";
  renderDownloads({});
}

function setImage(imgId, emptyId, record) {
  const img = el(imgId);
  const empty = el(emptyId);
  if (record?.exists && record.url) {
    img.src = `${apiUrl(record.url)}?t=${Date.now()}`;
    img.hidden = false;
    empty.hidden = true;
  } else {
    img.removeAttribute("src");
    img.hidden = true;
    empty.hidden = false;
  }
}

function renderDownloads(outputs) {
  const list = el("downloadList");
  list.innerHTML = "";
  requiredDownloads.forEach(([key, label]) => {
    const record = outputs[key] || {};
    const row = document.createElement("div");
    row.className = "download-row";

    const details = document.createElement("div");
    details.className = "download-name";
    details.textContent = record.filename || label;

    const reason = document.createElement("span");
    reason.className = "download-reason";
    reason.textContent = record.exists ? "Ready" : record.reason || "Not generated";
    details.appendChild(reason);

    const link = document.createElement("a");
    link.className = `download-button ${record.exists ? "" : "disabled"}`;
    link.textContent = "Download";
    link.href = record.exists ? apiUrl(record.url) : "#";
    if (!record.exists) link.setAttribute("aria-disabled", "true");

    row.append(details, link);
    list.appendChild(row);
  });
}

async function readJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.detail || `Request failed with status ${response.status}`;
    const logs = payload.logs || [];
    const err = new Error(message);
    err.logs = logs;
    throw err;
  }
  return payload;
}

async function refreshHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const health = await readJsonResponse(response);
    if (health.demo_mode) {
      setStatus("Demo Mode", "loading");
    } else if (health.model_ready) {
      setStatus("Model Ready", "ready");
    } else {
      setStatus("Model Error", "error");
      const missing = (health.missing || []).join("\n");
      setLogs(["Backend is running", "Missing runtime files:", missing].filter(Boolean));
    }
  } catch (error) {
    setStatus("API Offline", "error");
    setLogs(["Dashboard API is not reachable", error.message]);
  }
}

async function inspectFile(file) {
  selectedFile = file;
  lastInspect = null;
  resetOutputs();
  el("fileName").textContent = file.name;
  setLogs(["File uploaded", "Inspecting array"]);
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch(`${API_BASE}/api/inspect`, {
      method: "POST",
      body: formData,
    });
    const data = await readJsonResponse(response);
    updateSummary(data);
    appendLog(data.valid ? "Shape validated" : data.message);
  } catch (error) {
    updateSummary({
      valid: false,
      filename: file.name,
      shape: [],
      dtype: "",
      message: error.message,
    });
    setLogs([...(error.logs || []), error.message]);
  }
}

async function runPipeline() {
  if (!selectedFile || !lastInspect?.valid) return;
  setBusy(true);
  setStatus("Loading", "loading");
  setLogs(["File uploaded", "Shape validated", "Starting final pipeline"]);

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("color_choice", el("modelSelect").value);
  formData.append("save_preview", el("savePreview").checked ? "true" : "false");
  formData.append("include_npy", el("includeNpy").checked ? "true" : "false");

  try {
    const response = await fetch(`${API_BASE}/api/infer`, {
      method: "POST",
      body: formData,
    });
    const data = await readJsonResponse(response);
    const stats = data.metrics_or_stats || {};
    const outputs = data.outputs || {};

    setImage("rawPreview", "rawEmpty", outputs.raw_preview_png);
    setImage("srPreview", "srEmpty", outputs.sr_preview_png);
    setImage("rgbPreview", "rgbEmpty", outputs.preview_png);
    el("rawShape").textContent = formatShape(stats.input_shape);
    el("srShape").textContent = formatShape(stats.sr_shape);
    el("rgbShape").textContent = formatShape(stats.rgb_shape);
    el("rawRange").textContent = formatRange(stats.input_range);
    el("srRange").textContent = formatRange(stats.sr_range);
    el("rgbRange").textContent = formatRange(stats.rgb_range);
    el("runIdText").textContent = `Run ${data.run_id}`;
    renderDownloads(outputs);
    setLogs(data.logs || ["Run complete"]);
    setStatus(data.demo_mode ? "Demo Mode" : "Model Ready", data.demo_mode ? "loading" : "ready");
  } catch (error) {
    const logs = error.logs?.length ? error.logs : ["Run failed"];
    setLogs([...logs, error.message]);
    setStatus("Error", "error");
  } finally {
    setBusy(false);
  }
}

function attachUploadEvents() {
  const dropzone = el("dropzone");
  const input = el("fileInput");

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) inspectFile(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) inspectFile(file);
  });
}

function init() {
  attachUploadEvents();
  el("runButton").addEventListener("click", runPipeline);
  renderDownloads({});
  refreshHealth();
}

init();
