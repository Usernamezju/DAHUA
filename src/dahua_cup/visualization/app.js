const $ = (selector) => document.querySelector(selector);
let samples = [];
let capabilities = {};

async function request(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function card(name, value) {
  const state = value.enabled ? "可用" : "不可用";
  const reason = value.reason || "配置正确";
  return `<article class="panel compact"><h3>${name}</h3><p><strong>${state}</strong></p><p class="muted">${reason}</p></article>`;
}

function renderGpus(state, teacher) {
  const admission = teacher?.gpu_admission;
  const gpus = state.gpus || [];
  $("#gpus").innerHTML = gpus.length ? gpus.map((gpu) => {
    const free = gpu.memory_total_mb - gpu.memory_used_mb;
    const selected = admission?.gpu_ids?.includes(gpu.index) ? "（Qwen 已选）" : "";
    return `<article class="panel compact"><h3>GPU ${gpu.index} ${selected}</h3><p>${gpu.name}</p><p class="muted">空闲显存 ${free}/${gpu.memory_total_mb} MiB · 利用率 ${gpu.utilization_gpu_percent}%</p></article>`;
  }).join("") : "<p class='muted'>未检测到 NVIDIA GPU；本机仅能查看界面，不能执行服务器 Qwen。</p>";
  if (admission) {
    $("#gpu-summary").textContent = admission.enabled
      ? `Qwen 准入通过：GPU ${admission.gpu_ids.join(", ")}`
      : `Qwen 等待空闲 GPU：需要 ${admission.required_gpu_count} 张，当前候选 ${admission.idle_gpu_ids.join(", ") || "无"}`;
  }
}

async function refresh() {
  const connection = $("#connection");
  connection.textContent = "正在读取服务状态…";
  try {
    const [system, nextCapabilities, gpus] = await Promise.all([
      request("/api/system"), request("/api/capabilities"), request("/api/gpus"),
    ]);
    capabilities = nextCapabilities;
    connection.textContent = "服务已连接；Campus6 配置已加载。";
    $("#system").textContent = JSON.stringify(system, null, 2);
    $("#capabilities").innerHTML = Object.entries(capabilities)
      .map(([name, value]) => card(name, value)).join("");
    renderGpus(gpus, capabilities.qwen_teacher);
    if (selectedId()) await showSample();
  } catch (error) {
    connection.textContent = `无法读取服务：${error.message}`;
    $("#capabilities").innerHTML = "";
    $("#gpus").innerHTML = "";
  }
}

function selectedId() {
  return $("#sample-select").value;
}

function media(sampleId, kind) {
  return `/api/samples/${encodeURIComponent(sampleId)}/media/${kind}`;
}

async function loadSamples(selectId = "") {
  const result = await request("/api/samples?limit=200");
  samples = result.items || [];
  const select = $("#sample-select");
  select.innerHTML = samples.length ? "" : "<option value=''>暂无样本；请上传视频或导入清单。</option>";
  samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.sample_id;
    option.textContent = `${sample.sample_id} · ${sample.status || "pending"}`;
    select.append(option);
  });
  if (selectId && samples.some((sample) => sample.sample_id === selectId)) select.value = selectId;
  $("#sample-status").textContent = `共 ${samples.length} 个服务器样本`;
  await showSample();
}

async function showSample() {
  const sampleId = selectedId();
  if (!sampleId) return;
  const source = $("#source-video");
  source.src = media(sampleId, "preview");
  const pose = $("#pose-video");
  pose.src = media(sampleId, "pose");
  try {
    const prediction = await request(`/api/samples/${encodeURIComponent(sampleId)}/prediction`);
    $("#prediction").textContent = JSON.stringify(prediction, null, 2);
  } catch (_) {
    $("#prediction").textContent = "尚无预测结果";
  }
  const teacher = capabilities.qwen_teacher || {};
  $("#teacher-button").disabled = !teacher.enabled;
  $("#teacher-button").title = teacher.reason || "按需调用服务器 Qwen";
}

async function pollJob(job) {
  const progress = $("#job-status");
  progress.classList.remove("hidden");
  const bar = $("#job-status span");
  const text = $("#job-status p");
  let current = job;
  while (["queued", "running"].includes(current.status)) {
    bar.style.width = `${Math.round((current.progress || 0) * 100)}%`;
    text.textContent = current.message || current.status;
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    current = await request(`/api/jobs/${encodeURIComponent(current.job_id)}`);
  }
  bar.style.width = "100%";
  text.textContent = current.status === "completed" ? "任务完成" : `任务失败：${current.message || "未知错误"}`;
  await refresh();
  await showSample();
}

async function submitAction(action) {
  const sampleId = selectedId();
  if (!sampleId) throw new Error("请先选择或上传视频样本");
  const response = await fetch(`/api/samples/${encodeURIComponent(sampleId)}/jobs`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({action}),
  });
  if (!response.ok) throw new Error((await response.json()).detail || `HTTP ${response.status}`);
  await pollJob(await response.json());
}

async function upload(file) {
  if (!file) return;
  $("#sample-status").textContent = `正在上传 ${file.name}…`;
  const response = await fetch(`/api/videos?filename=${encodeURIComponent(file.name)}`, {method: "POST", body: file});
  if (!response.ok) throw new Error((await response.json()).detail || `HTTP ${response.status}`);
  const sample = await response.json();
  await loadSamples(sample.sample_id);
}

document.addEventListener("DOMContentLoaded", () => {
  $("#refresh").addEventListener("click", refresh);
  $("#sample-select").addEventListener("change", showSample);
  document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => submitAction(button.dataset.action).catch((error) => { $("#sample-status").textContent = error.message; })));
  $("#upload").addEventListener("change", (event) => upload(event.target.files[0]).catch((error) => { $("#sample-status").textContent = error.message; }));
  refresh();
  loadSamples().catch((error) => { $("#sample-status").textContent = error.message; });
  window.setInterval(refresh, 10_000);
});
