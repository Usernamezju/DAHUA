/* Standalone demo web: sample browsing, three result panels, video import. */
"use strict";

const $ = (selector) => document.querySelector(selector);
const GROUPS = [];
let currentSampleId = null;
let pollTimer = null;

const LABEL_ZH = {
  normal_walk: "正常行走",
  normal_run: "正常奔跑",
  playful_chase: "嬉戏追逐",
  playful_push: "嬉戏推搡",
  conflict_chase: "冲突追逐",
  conflict_push: "冲突推搡",
};

function toast(message, kind = "ok") {
  const root = $("#toast-root");
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  root.appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* keep */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ---------- sidebar samples ---------- */
function renderGroups() {
  const container = $("#sample-groups");
  container.textContent = "";
  GROUPS.forEach((group) => {
    const title = document.createElement("div");
    title.className = "group-title";
    title.innerHTML = `<span>${group.title}</span><b>${group.rows.length}</b>`;
    container.appendChild(title);
    group.rows.forEach((row) => {
      const button = document.createElement("button");
      button.className = "sample-item";
      button.dataset.id = row.id;
      const id = document.createElement("span");
      id.className = "sample-id";
      id.textContent = row.id;
      const chip = document.createElement("span");
      chip.className = "label-chip";
      chip.textContent = row.label;
      button.append(id, chip);
      button.addEventListener("click", () => selectSample(row.id));
      container.appendChild(button);
    });
  });
  $("#sample-total").textContent = GROUPS.reduce((n, g) => n + g.rows.length, 0);
}

async function loadSamples(selectFirst = true) {
  const { groups } = await fetchJson("/api/samples");
  GROUPS.length = 0;
  GROUPS.push(...groups);
  renderGroups();
  if (selectFirst) {
    const first = groups.flatMap((g) => g.rows)[0];
    if (first) selectSample(first.id);
  }
}

/* ---------- sample display ---------- */
function setMedia(kind, url) {
  const video = $(kind === "rgb" ? "#rgb-video" : "#pose-video");
  const frame = video.closest(".video-frame");
  const tag = $(kind === "rgb" ? "#rgb-status" : "#pose-status");
  if (!url) {
    frame.classList.remove("ready");
    video.removeAttribute("src");
    tag.textContent = "未生成";
    tag.classList.remove("ready");
    return;
  }
  video.src = url;
  frame.classList.add("ready");
  tag.textContent = "就绪";
  tag.classList.add("ready");
}

function renderPrediction(prediction) {
  const topk = prediction.topk || [];
  if (!topk.length) return;
  const top1 = topk[0];
  const finalCard = document.createElement("div");
  finalCard.className = "final-card";
  finalCard.innerHTML = `
    <div class="final-rank">1</div>
    <div>
      <div class="final-label">${top1.label}</div>
      <div class="final-meta">${LABEL_ZH[top1.label] || ""} · M1KD QAT INT8 校准输出</div>
    </div>
    <div class="final-score">${(top1.score * 100).toFixed(1)}%</div>`;
  const finalResult = $("#final-result");
  finalResult.classList.remove("empty-state");
  finalResult.textContent = "";
  finalResult.appendChild(finalCard);

  const list = $("#prediction-list");
  list.classList.remove("empty-state");
  list.textContent = "";
  topk.forEach((entry, index) => {
    const row = document.createElement("div");
    row.className = "prediction-row" + (index === 0 ? " top1" : "");
    const percentage = Math.max(0, Math.min(100, entry.score * 100));
    row.innerHTML = `
      <span class="pred-rank">#${index + 1}</span>
      <span class="pred-label">${entry.label}</span>
      <span class="score-track"><span style="width:${percentage.toFixed(1)}%"></span></span>
      <span class="pred-score">${percentage.toFixed(1)}%</span>`;
    list.appendChild(row);
  });
}

async function selectSample(sampleId) {
  currentSampleId = sampleId;
  document.querySelectorAll(".sample-item").forEach((node) => {
    node.classList.toggle("active", node.dataset.id === sampleId);
  });
  const row = GROUPS.flatMap((g) => g.rows).find((r) => r.id === sampleId);
  $("#current-sample").textContent = sampleId;
  const pill = $("#sample-pill");
  pill.className = "status-pill busy";
  pill.textContent = row ? row.label : "—";
  setMedia("rgb", null);
  setMedia("pose", null);
  $("#final-result").textContent = "推理中…";
  $("#final-result").classList.add("empty-state");
  $("#prediction-list").textContent = "加载中…";
  $("#prediction-list").classList.add("empty-state");
  try {
    const sample = await fetchJson(`/api/samples/${sampleId}`);
    setMedia("rgb", sample.media.rgb);
    setMedia("pose", sample.media.pose);
    renderPrediction(sample.prediction);
    pill.className = "status-pill online";
    pill.textContent = row ? row.label : "—";
  } catch (error) {
    toast(`加载失败：${error.message}`, "error");
    pill.className = "status-pill error";
  }
}

/* ---------- import ---------- */
function bindDropZone() {
  const zone = $("#drop-zone");
  const input = $("#import-file");
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragover");
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      startImport(input.files[0]);
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length) startImport(input.files[0]);
  });
}

async function startImport(file) {
  const status = $("#import-status");
  status.className = "import-status busy";
  status.textContent = `正在上传 ${file.name} …`;
  try {
    const { job_id } = await fetchJson(
      `/api/import?name=${encodeURIComponent(file.name)}`,
      { method: "POST", body: file },
    );
    pollImport(job_id);
  } catch (error) {
    status.className = "import-status failed";
    status.textContent = `上传失败：${error.message}`;
  }
}

function pollImport(jobId) {
  clearTimeout(pollTimer);
  fetchJson(`/api/import/${jobId}`)
    .then((job) => {
      const status = $("#import-status");
      if (job.status === "completed") {
        status.className = "import-status done";
        status.textContent = `导入完成：${job.sample.label}（${job.sample.id}）`;
        toast(`导入完成，最终结果：${job.sample.label}`);
        loadSamples(false).then(() => selectSample(job.sample.id));
      } else if (job.status === "failed") {
        status.className = "import-status failed";
        status.textContent = `导入失败：${job.message}`;
      } else {
        status.textContent = `处理中：${job.status === "queued" ? "排队中" : "骨架提取与推理中"} …`;
        pollTimer = setTimeout(() => pollImport(jobId), 2000);
      }
    })
    .catch((error) => {
      $("#import-status").className = "import-status failed";
      $("#import-status").textContent = `查询失败：${error.message}`;
    });
}

/* ---------- navigation & init ---------- */
function bindNavigation() {
  const titles = { samples: "演示样本", import: "导入视频" };
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
      $(`#view-${button.dataset.view}`).classList.add("active");
      $("#page-title").textContent = titles[button.dataset.view] || "";
    });
  });
}

async function init() {
  bindNavigation();
  bindDropZone();
  try {
    await fetchJson("/api/health");
    $("#health-pill").className = "status-pill online";
    $("#health-pill").textContent = "服务在线";
    $("#server-dot").classList.add("online");
    $("#server-label").textContent = "演示服务已连接";
    await loadSamples();
  } catch (error) {
    $("#health-pill").className = "status-pill error";
    $("#health-pill").textContent = "服务不可用";
    $("#server-label").textContent = "连接失败";
    toast(`无法连接演示服务：${error.message}`, "error");
  }
}

document.addEventListener("DOMContentLoaded", init);
