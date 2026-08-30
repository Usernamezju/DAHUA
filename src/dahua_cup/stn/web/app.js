const state = {
  samples: [],
  selected: null,
  config: null,
  polling: null,
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

let toastTimer = null;
function toast(message, error = false) {
  const root = $("#toast");
  root.textContent = message;
  root.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { root.className = "toast"; }, 4200);
}

function percent(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function decimal(value, digits = 3) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function setMetric(selector, value, level) {
  const node = $(selector);
  node.textContent = value;
  const card = node.closest(".metric-card");
  card.classList.remove("good", "warn", "bad");
  if (level) card.classList.add(level);
}

function classify(value, good, warning, inverse = false) {
  if (value == null) return "";
  if (inverse) {
    if (value <= good) return "good";
    if (value <= warning) return "warn";
    return "bad";
  }
  if (value >= good) return "good";
  if (value >= warning) return "warn";
  return "bad";
}

function renderMetrics(metrics, capacity = null) {
  if (!metrics) {
    ["#metric-iou", "#metric-containment", "#metric-presence",
      "#metric-jitter", "#metric-zoom", "#metric-crowd"].forEach((selector) => {
      setMetric(selector, "—", "");
    });
    $("#diagnosis").className = "diagnosis";
    $("#diagnosis").textContent =
      "选择样本并运行后，将根据定位、包含率、人数判断和抖动给出诊断。";
    $("#worst-list").innerHTML = '<span class="muted">尚无评估结果</span>';
    return;
  }
  setMetric(
    "#metric-iou",
    decimal(metrics.mean_group_iou),
    classify(metrics.mean_group_iou, .60, .45),
  );
  setMetric(
    "#metric-containment",
    percent(metrics.containment_recall),
    classify(metrics.containment_recall, .95, .85),
  );
  setMetric(
    "#metric-presence",
    percent(metrics.presence_accuracy),
    classify(metrics.presence_accuracy, .90, .75),
  );
  setMetric(
    "#metric-jitter",
    decimal(metrics.temporal_jitter, 4),
    classify(metrics.temporal_jitter, .03, .055, true),
  );
  setMetric("#metric-zoom", `${decimal(metrics.mean_zoom_factor, 2)}×`, "");
  const crowdRatio = Number(capacity?.crowd_frame_ratio || 0);
  setMetric(
    "#metric-crowd",
    percent(crowdRatio),
    crowdRatio === 0 ? "good" : crowdRatio <= .10 ? "warn" : "bad",
  );

  const failures = [];
  if (metrics.mean_group_iou < .45) failures.push("联合框定位偏差明显");
  if (metrics.containment_recall < .85) failures.push("人物经常没有被完整包含");
  if (metrics.presence_accuracy < .75) failures.push("单人/双人人数判断不稳定");
  if (metrics.temporal_jitter > .055) failures.push("裁剪框时序抖动较大");
  const diagnosis = $("#diagnosis");
  if (crowdRatio > 0) {
    diagnosis.className = crowdRatio > .10 ? "diagnosis bad" : "diagnosis warn";
    diagnosis.textContent =
      `${percent(crowdRatio)} 的帧出现了3人以上，已经超出当前两查询STN和下游两人骨架链路的容量。页面中的IoU、包含率和人数准确率只评价教师选中的top-2，不能证明模型处理了其他人物；该样本应单独标记为超范围，而不是通过增加同类数据强行学习。`;
  } else if (!failures.length && metrics.mean_group_iou >= .60 &&
      metrics.containment_recall >= .95 && metrics.presence_accuracy >= .90) {
    diagnosis.className = "diagnosis good";
    diagnosis.textContent =
      "该样本上的定位、完整包含和人数判断均达到建议线。继续检查不同数据来源、远距离和遮挡样本；如果多数样本表现一致，暂时不需要仅为增加数量而扩充数据。";
  } else if (failures.length) {
    diagnosis.className = "diagnosis bad";
    diagnosis.textContent =
      `${failures.join("；")}。先观察视频确认失败集中在小人物、双人分离、遮挡还是特定数据来源，再针对该场景补充数据，不能只看总样本数。`;
  } else {
    diagnosis.className = "diagnosis warn";
    diagnosis.textContent =
      "模型已经学会基本裁剪，但部分指标仍处于过渡区。建议先批量检查各来源的代表样本，再决定补充单人、双人或远距离数据。";
  }

  const worst = metrics.worst_frames || [];
  $("#worst-list").innerHTML = worst.length
    ? worst.map((item) =>
      `<span class="worst-frame">F${item.frame_index + 1} · IoU ${Number(item.group_iou).toFixed(2)}</span>`
    ).join("")
    : '<span class="muted">没有带教师框的可评估帧</span>';
}

function renderSamples() {
  const root = $("#sample-list");
  $("#sample-count").textContent = state.samples.length;
  if (!state.samples.length) {
    root.innerHTML = '<div class="list-placeholder">没有匹配的样本</div>';
    return;
  }
  root.innerHTML = state.samples.map((sample) => `
    <button class="sample-item ${state.selected?.sample_id === sample.sample_id ? "active" : ""}"
      data-sample-id="${escapeHtml(sample.sample_id)}">
      <strong>${escapeHtml(sample.sample_id)}</strong>
      <span class="sample-meta">
        <span>${escapeHtml(sample.source_dataset)}</span>
        <span>${sample.selected_people}人</span>
        <span>${sample.frames}帧</span>
        ${sample.capacity_exceeded ? '<span class="sample-capacity">● 超容量</span>' : ""}
        ${sample.visualization_ready ? '<span class="sample-ready">● 已生成</span>' : ""}
      </span>
    </button>
  `).join("");
  root.querySelectorAll("[data-sample-id]").forEach((button) => {
    button.onclick = () => selectSample(button.dataset.sampleId);
  });
}

function setVideo(selector, emptySelector, url) {
  const video = $(selector);
  const empty = $(emptySelector);
  video.pause();
  if (!url) {
    video.removeAttribute("src");
    video.classList.remove("ready");
    empty.hidden = false;
    return;
  }
  video.src = `${url}?v=${Date.now()}`;
  video.classList.add("ready");
  empty.hidden = true;
  video.load();
}

async function loadResult(sampleId) {
  try {
    const result = await api(`/api/samples/${encodeURIComponent(sampleId)}/result`);
    setVideo("#overlay-video", "#overlay-empty",
      `/api/samples/${encodeURIComponent(sampleId)}/media/overlay`);
    setVideo("#crop-video", "#crop-empty",
      `/api/samples/${encodeURIComponent(sampleId)}/media/crop`);
    $("#overlay-status").textContent = "已生成";
    $("#crop-status").textContent = "已生成";
    renderMetrics(result.metrics, result.capacity);
    return true;
  } catch (_) {
    setVideo("#overlay-video", "#overlay-empty", null);
    setVideo("#crop-video", "#crop-empty", null);
    $("#overlay-status").textContent = "待生成";
    $("#crop-status").textContent = "待生成";
    renderMetrics(null);
    return false;
  }
}

async function selectSample(sampleId) {
  const sample = await api(`/api/samples/${encodeURIComponent(sampleId)}`);
  state.selected = sample;
  renderSamples();
  $("#sample-title").textContent = sample.sample_id;
  $("#sample-subtitle").textContent =
    `${sample.source_dataset} · ${sample.selected_people}人轨迹 · ` +
    `${sample.frames}帧 · 教师置信度 ${sample.mean_teacher_confidence.toFixed(3)} · ` +
    `3人以上帧 ${percent(sample.crowd_frame_ratio)}`;
  $("#run-button").disabled = !sample.video_exists;
  if (!sample.video_exists) {
    toast(`视频文件不存在：${sample.video_path}`, true);
  }
  await loadResult(sample.sample_id);
}

async function refreshSamples() {
  const params = new URLSearchParams();
  const query = $("#sample-query").value.trim();
  const dataset = $("#dataset-filter").value;
  const people = $("#people-filter").value;
  if (query) params.set("query", query);
  if (dataset) params.set("dataset", dataset);
  if (people) params.set("people", people);
  params.set("limit", "500");
  const payload = await api(`/api/samples?${params}`);
  state.samples = payload.items;
  renderSamples();
}

async function pollJob(jobId) {
  clearInterval(state.polling);
  $("#job-notice").hidden = false;
  $("#run-button").disabled = true;
  const check = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      $("#job-title").textContent =
        job.status === "queued" ? "任务已排队" : "正在运行STN";
      $("#job-message").textContent = job.message;
      if (job.status === "completed") {
        clearInterval(state.polling);
        $("#job-notice").hidden = true;
        $("#run-button").disabled = false;
        await loadResult(job.sample_id);
        await refreshSamples();
        toast("STN可视化已生成");
        return true;
      } else if (job.status === "failed") {
        clearInterval(state.polling);
        $("#job-notice").hidden = true;
        $("#run-button").disabled = false;
        toast(job.error || "STN可视化失败", true);
        return true;
      }
    } catch (error) {
      clearInterval(state.polling);
      $("#job-notice").hidden = true;
      $("#run-button").disabled = false;
      toast(error.message, true);
      return true;
    }
    return false;
  };
  const terminal = await check();
  if (!terminal) state.polling = setInterval(check, 1200);
}

async function runSelected() {
  if (!state.selected) return;
  try {
    const force = $("#force-run").checked;
    const job = await api(
      `/api/samples/${encodeURIComponent(state.selected.sample_id)}/run?force=${force}`,
      { method: "POST" },
    );
    await pollJob(job.job_id);
  } catch (error) {
    toast(error.message, true);
  }
}

function synchronizeVideos() {
  const first = $("#overlay-video");
  const second = $("#crop-video");
  let syncing = false;
  const mirror = (action) => {
    if (syncing || !second.src) return;
    syncing = true;
    action();
    syncing = false;
  };
  first.addEventListener("play", () => mirror(() => {
    second.currentTime = first.currentTime;
    second.play().catch(() => {});
  }));
  first.addEventListener("pause", () => mirror(() => second.pause()));
  first.addEventListener("seeked", () => mirror(() => {
    second.currentTime = first.currentTime;
  }));
  second.addEventListener("play", () => {
    if (syncing || !first.src) return;
    syncing = true;
    first.currentTime = second.currentTime;
    first.play().catch(() => {});
    syncing = false;
  });
  second.addEventListener("pause", () => {
    if (syncing) return;
    syncing = true;
    first.pause();
    syncing = false;
  });
}

async function boot() {
  try {
    const [health, config, samples] = await Promise.all([
      api("/api/health"),
      api("/api/config"),
      api("/api/samples?limit=500"),
    ]);
    state.config = config;
    state.samples = samples.items;
    $("#health-text").textContent = `服务在线 · ${health.samples}个样本`;
    $("#device-pill").textContent = config.device;
    $("#checkpoint-text").textContent =
      `Checkpoint ${config.checkpoint} · ${config.checkpoint_fingerprint}`;
    const datasets = [...new Set(state.samples.map((item) => item.source_dataset))].sort();
    $("#dataset-filter").innerHTML =
      '<option value="">全部数据集</option>' +
      datasets.map((value) =>
        `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`
      ).join("");
    renderSamples();
  } catch (error) {
    $("#health-text").textContent = "服务连接失败";
    toast(error.message, true);
  }
}

let queryTimer = null;
$("#sample-query").addEventListener("input", () => {
  clearTimeout(queryTimer);
  queryTimer = setTimeout(() => refreshSamples().catch(
    (error) => toast(error.message, true)
  ), 220);
});
$("#dataset-filter").onchange = () => refreshSamples().catch(
  (error) => toast(error.message, true)
);
$("#people-filter").onchange = () => refreshSamples().catch(
  (error) => toast(error.message, true)
);
$("#run-button").onclick = runSelected;
synchronizeVideos();
boot();
