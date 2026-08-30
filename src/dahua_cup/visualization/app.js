const $ = (selector) => document.querySelector(selector);

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
    const [system, capabilities, gpus] = await Promise.all([
      request("/api/system"), request("/api/capabilities"), request("/api/gpus"),
    ]);
    connection.textContent = "服务已连接；Campus6 配置已加载。";
    $("#system").textContent = JSON.stringify(system, null, 2);
    $("#capabilities").innerHTML = Object.entries(capabilities)
      .map(([name, value]) => card(name, value)).join("");
    renderGpus(gpus, capabilities.qwen_teacher);
  } catch (error) {
    connection.textContent = `无法读取服务：${error.message}`;
    $("#capabilities").innerHTML = "";
    $("#gpus").innerHTML = "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#refresh").addEventListener("click", refresh);
  refresh();
  window.setInterval(refresh, 10_000);
});
