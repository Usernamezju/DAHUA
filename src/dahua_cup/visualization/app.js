const LABELS = [
  ["normal_walk", "正常行走", "速度适中，轨迹独立"],
  ["normal_run", "正常奔跑", "持续跑动，无肢体接触"],
  ["playful_chase", "嬉戏追逐", "环形折返，姿态放松"],
  ["playful_push", "嬉戏推搡", "轻接触，双方主动参与"],
  ["conflict_chase", "冲突追逐", "直线逃离，防御性后退"],
  ["conflict_push", "冲突推搡", "强接触，明显攻击或防御"],
];
const LABEL_NAME = Object.fromEntries(LABELS.map(([id, name]) => [id, name]));
Object.assign(LABEL_NAME, { unknown: "无法判断", damaged: "视频损坏", out_of_scope: "无关样本" });
const SAMPLE_LABEL_GROUPS = LABELS.map(([id, name]) => [id, name]);

const state = {
  capabilities: {}, system: {}, gpus: {}, macroParameters: {}, dashboard: {}, modelStatus: {}, trainingStatus: {}, samples: [], reviewSamples: [],
  sampleTotal: 0, sampleOffset: 0, sampleWindowSize: 500, sampleCursor: 0, reviewTotal: 0,
  hardSamples: [], hardTotal: 0, hardSample: null, hardLoaded: false,
  inferenceSample: null, reviewSample: null, reviewIndex: -1,
  activeJobId: null, jobSubmissionPending: false,
  datasetPage: 0, datasetPageSize: 50, datasetTotal: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value = "") => String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

function toast(message, error = false) {
  const element = document.createElement("div");
  element.className = `toast${error ? " error" : ""}`;
  element.textContent = message;
  $("#toast-root").append(element);
  setTimeout(() => element.remove(), 3600);
}

function formatTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString("zh-CN", { year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false });
}

function titleForStatus(status) {
  return ({waiting_teacher:"等待大模型推理",waiting_human:"等待人工审核",complete:"完成",pending:"待审核",reviewed:"已标注",unknown:"无法判断",damaged:"损坏",out_of_scope:"无关",not_run:"等待分析",queued:"排队中",running:"处理中",completed:"完成",failed:"失败",skipped:"已跳过",blocked_quality:"骨架质量不足"})[status] || status;
}

function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const button = $("#sidebar-toggle");
  if (!button) return;
  button.textContent = collapsed ? "›" : "‹";
  button.setAttribute("aria-label", collapsed ? "展开侧栏" : "收起侧栏");
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  localStorage.setItem("dahua-sidebar-collapsed", collapsed ? "1" : "0");
}

function setPage(page) {
  if (page === "review") page = "inference";
  const titles = {dashboard:"运行总览",inference:"行为识别结果",hard:"待人工审核难例",dataset:"审计记录",training:"增量训练",models:"模型管理",settings:"系统设置"};
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.page === page));
  $$(".page").forEach(item => item.classList.toggle("active", item.id === `page-${page}`));
  $("#page-title").textContent = titles[page];
  history.replaceState(null, "", `#${page}`);
  if (page === "dataset") loadDatasetPage();
  if (page === "hard" && !state.hardLoaded) loadHardSamples();
  else if (page === "hard" && !state.hardSample && state.hardSamples.length) selectHardSample(state.hardSamples[0].sample_id);
}

function reviewer() {
  const value = $("#reviewer").value.trim();
  if (!value) throw new Error("请先填写审核员名称");
  localStorage.setItem("dahua-reviewer", value);
  return value;
}

async function fetchSampleSummaries(params = {}) {
  const query = new URLSearchParams({summary:"true", offset:0, limit:50, ...params});
  return api(`/api/samples?${query}`);
}

async function loadInferenceSamples(query = "", offset = 0) {
  const result = await fetchSampleSummaries({query, offset, limit:state.sampleWindowSize, include_teacher:false});
  state.samples = result.items;
  state.sampleTotal = result.total;
  state.sampleOffset = result.offset;
  renderSampleOptions();
}

async function refreshAll(preserveSelection = true) {
  try {
    const [capabilities, system, gpus, macroParameters, dashboard, modelStatus, trainingStatus, samples, reviewSamples] = await Promise.all([
      api("/api/capabilities"), api("/api/system"), api("/api/gpus"), api("/api/macro-parameters"), api("/api/dashboard"), api("/api/models/status"), api("/api/training/status"),
      fetchSampleSummaries({limit:state.sampleWindowSize,include_teacher:false}), fetchSampleSummaries({status:"pending",limit:50,include_teacher:false}),
    ]);
    Object.assign(state, {
      capabilities, system, gpus, macroParameters, dashboard, modelStatus, trainingStatus,
      samples: samples.items, sampleTotal: samples.total, sampleOffset: samples.offset, sampleCursor: 0,
      reviewSamples: reviewSamples.items, reviewTotal: reviewSamples.total,
      hardSamples: [], hardTotal: 0, hardSample: null, hardLoaded: false,
    });
    $("#server-dot").classList.add("online");
    $("#server-label").textContent = "服务器在线";
    renderDashboard(); renderCapabilities(); renderSampleOptions(); renderHardOptions(); renderModels(); renderTrainingStatus(); renderSettings();
    if (!preserveSelection || !state.inferenceSample) {
      if (state.samples.length) await selectInference(state.samples[0].sample_id, false);
    } else {
      const current = state.samples.find(item => item.sample_id === state.inferenceSample.sample_id);
      if (current) await selectInference(current.sample_id, false);
      else if (state.samples.length) await selectInference(state.samples[0].sample_id, false);
    }
  } catch (error) {
    $("#server-label").textContent = "连接失败";
    toast(error.message, true);
  }
}

function renderDashboard() {
  const d = state.dashboard;
  const reviewed = d.statuses?.reviewed || 0;
  const pending = d.statuses?.pending || 0;
  const excluded = (d.statuses?.unknown || 0) + (d.statuses?.damaged || 0) + (d.statuses?.out_of_scope || 0);
  const completion = d.total ? Math.round((d.total - pending) / d.total * 100) : 0;
  const metrics = [
    ["行为片段", d.total || 0, "Skeleton-only 识别记录", "#35d399"],
    ["待人工审核", pending, "按优先级进入队列", "#fbbf24"],
    ["有效六分类", reviewed, "可导出用于训练", "#60a5fa"],
    ["审核完成率", `${completion}%`, `已排除 ${excluded} 条`, "#a78bfa"],
  ];
  $("#metrics").innerHTML = metrics.map(([name,value,sub,color]) => `<div class="metric-card" style="--accent:${color}"><span>${name}</span><strong>${value}</strong><small>${sub}</small></div>`).join("");
  $("#pending-badge").textContent = pending;
  const max = Math.max(1, ...LABELS.map(([id]) => d.labels?.[id] || 0));
  $("#label-chart").innerHTML = LABELS.map(([id,name]) => { const count=d.labels?.[id]||0; return `<div class="bar-row"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${count/max*100}%"></div></div><strong>${count}</strong></div>`; }).join("");
  const jobs = d.recent_jobs || [];
  $("#recent-jobs").className = jobs.length ? "job-list" : "job-list empty-state";
  $("#recent-jobs").innerHTML = jobs.length ? jobs.map(job => `<div class="job-row"><code>${escapeHtml(job.action)}</code><span>${escapeHtml(job.sample_id)}</span><span>${escapeHtml(job.message || "—")}</span><span class="job-status ${job.status}">${titleForStatus(job.status)}</span></div>`).join("") : "暂无运行任务";
}

function renderCapabilities() {
  const names = {pose_extraction:"RTMPose COCO-17 骨架",campus6_inference:"M1KD INT8 Campus6 推理",manual_review:"人工审核",dataset_export:"数据导出",qwen_teacher:"Qwen3-VL-8B 服务器教师",live_camera:"实时摄像头"};
  $("#capability-list").innerHTML = Object.entries(state.capabilities).map(([id,item]) => `<div class="capability ${item.enabled ? "enabled":""}" title="${escapeHtml(item.reason)}"><span></span><div><strong>${names[id]||id}</strong><small>${item.enabled ? "已启用" : item.reason}</small></div></div>`).join("");
}

function renderSampleOptions() {
  const selected = state.inferenceSample?.sample_id;
  const query = ($("#inference-search")?.value || "").trim().toLowerCase();
  const root = $("#sample-tree-groups");
  if (root) {
    const openLabels = new Set($$("#sample-tree-groups details[open]").map(item => item.dataset.label));
    const selectedSample = state.samples.find(item => item.sample_id === selected);
    root.innerHTML = SAMPLE_LABEL_GROUPS.map(([label, name]) => {
      const allItems = state.samples.filter(item => item.student_label === label);
      const items = allItems.filter(item => !query || `${item.sample_id} ${item.source_dataset} ${item.student_label}`.toLowerCase().includes(query));
      const opened = Boolean(query) || openLabels.has(label) || selectedSample?.student_label === label;
      const count = query ? `${items.length}/${allItems.length}` : allItems.length;
      const children = items.length ? items.map(item => `<button class="sample-tree-item ${item.sample_id===selected?"active":""}" data-sample-id="${escapeHtml(item.sample_id)}"><strong>${escapeHtml(item.sample_id)}</strong><small>${escapeHtml(item.source_dataset||"未知来源")} · ${escapeHtml(LABEL_NAME[item.student_label]||item.student_label||"未标注")}</small></button>`).join("") : '<div class="sample-tree-empty">没有匹配样本</div>';
      return `<details class="sample-status-group" data-label="${label}" ${opened?"open":""}><summary><strong>${name}</strong><b>${count}</b></summary><div class="sample-tree-items">${children}</div></details>`;
    }).join("");
    $$("[data-sample-id]").forEach(button => button.onclick = async () => {
      setSampleTreeOpen(false);
      await selectInference(button.dataset.sampleId);
    });
  }
  const current = $("#sample-tree-current");
  if (current) {
    const sample = state.samples.find(item => item.sample_id === selected) || state.inferenceSample;
    current.textContent = sample ? `${sample.sample_id} · ${sample.source_dataset || "Campus6"}` : "请选择行为片段";
  }
  const samplePrev = $("#sample-prev"), sampleNext = $("#sample-next");
  if (samplePrev) samplePrev.disabled = state.sampleCursor <= 0;
  if (sampleNext) sampleNext.disabled = state.sampleCursor + 1 >= state.sampleTotal;
  const datasets = [...new Set(state.samples.map(item=>item.source_dataset).filter(Boolean))];
  ["#dataset-source"].forEach(selector => {
    const element=$(selector); if(!element)return; const value=element.value;
    element.innerHTML='<option value="all">全部来源</option>'+datasets.map(d=>`<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");
    if ([...element.options].some(o=>o.value===value)) element.value=value;
  });
}

function setSampleTreeOpen(open) {
  const menu = $("#sample-tree-menu"), button = $("#sample-tree-toggle");
  if (!menu || !button) return;
  menu.classList.toggle("hidden", !open);
  button.setAttribute("aria-expanded", open ? "true" : "false");
}

function setVideo(element, url) {
  if (!element) return;
  element.pause(); element.removeAttribute("src"); element.load();
  if (url) { element.src = `${url}?v=${Date.now()}`; element.load(); }
}

async function selectInference(sampleId) {
  try {
    const sample = await api(`/api/samples/${encodeURIComponent(sampleId)}`);
    const localIndex = state.samples.findIndex(item => item.sample_id === sampleId);
    if (localIndex >= 0) state.sampleCursor = state.sampleOffset + localIndex;
    state.inferenceSample = sample;
    state.reviewSample = sample;
    setSampleTreeOpen(false);
    renderSampleOptions();
    setVideo($("#pose-video"), sample.artifacts.pose_video ? sample.media.pose : "");
    $("#pose-status").textContent = sample.artifacts.pose_video ? "已生成" : "未生成";
    renderSemanticGraph(sample.semantic_graph || sample.prediction?.student_evidence?.semantic_graph);
    renderPrediction(sample.prediction);
    renderTeacher(sample.teacher, sample.pseudo_record, sample.prediction, sample.difficulty_checks);
    $("#review-note").value = sample.note || "";
    const pendingItems = filteredReviewSamples();
    const queueIndex = pendingItems.findIndex(item => item.sample_id === sample.sample_id);
    if (queueIndex >= 0) state.reviewIndex = queueIndex;
  } catch(error){ toast(error.message,true); }
}

async function moveInference(delta) {
  const target = state.sampleCursor + delta;
  if (target < 0 || target >= state.sampleTotal) return;
  const localIndex = target - state.sampleOffset;
  if (localIndex >= 0 && localIndex < state.samples.length) {
    await selectInference(state.samples[localIndex].sample_id);
    return;
  }
  const query = $("#inference-search").value.trim();
  const maximumOffset = Math.max(0, state.sampleTotal - state.sampleWindowSize);
  const offset = Math.min(maximumOffset, Math.max(0, target - 10));
  await loadInferenceSamples(query, offset);
  const nextItem = state.samples[target - state.sampleOffset];
  if (nextItem) await selectInference(nextItem.sample_id);
}

function percentage(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "—";
}

function decimal(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "—";
}

function renderSemanticGraph(graph) {
  const root = $("#semantic-graph-summary");
  if (!root) return;
  const persons = Array.isArray(graph?.persons) ? graph.persons : [];
  const relation = (Array.isArray(graph?.relations) ? graph.relations : []).find(
    item => item?.type === "measured_distance"
  );
  const segment = Array.isArray(graph?.segments) ? graph.segments[0] : null;
  if (!persons.length) {
    root.className = "semantic-graph-summary empty-state";
    root.textContent = "尚无可用骨架时空摘要";
    return;
  }
  const personRows = persons.map((person, index) => {
    const number = Number(person?.id);
    const name = Number.isFinite(number) ? `人物 ${number + 1}` : `人物 ${index + 1}`;
    return `<li><strong>${name}</strong><span>关键点覆盖 ${percentage(person?.pose_coverage)}</span><span>出现帧 ${percentage(person?.frame_coverage)}</span><span>平均速度 ${decimal(person?.mean_normalized_speed)}</span></li>`;
  }).join("");
  const relationText = relation
    ? `最小距离 ${decimal(relation.minimum_normalized_distance)} · 平均距离 ${decimal(relation.mean_normalized_distance)} · 共同可见 ${percentage(relation.frame_coverage)}`
    : "未同时观测到两人";
  const timing = segment
    ? `${segment.id || "s0"} · ${(Number(segment.start_ms || 0) / 1000).toFixed(1)}–${(Number(segment.end_ms || 0) / 1000).toFixed(1)} 秒`
    : "—";
  root.className = "semantic-graph-summary";
  root.innerHTML = `<div class="semantic-graph-heading"><strong>骨架时空摘要</strong><span>时段 ${escapeHtml(timing)}</span></div><ul>${personRows}</ul><div class="semantic-relation"><strong>双人关系</strong><span>${escapeHtml(relationText)}</span></div>`;
}

function distributionValues(value) {
  if (Array.isArray(value)) return Object.fromEntries(value.map(item=>[item.label, Number(item.probability ?? item.score)||0]));
  return value || {};
}

function renderSixDistribution(root, value, emptyText="尚无六类分布") {
  if (!root) return;
  const values=distributionValues(value);
  if(!Object.keys(values).length){root.className="prediction-list empty-state";root.textContent=emptyText;return;}
  root.className="prediction-list";
  const ranked=LABELS.map(([id,name])=>({id,name,score:Number(values[id])||0})).sort((left,right)=>right.score-left.score);
  root.innerHTML=ranked.map((item,index)=>`<div class="prediction-row"><span class="rank">${index+1}</span><span>${item.name}</span><div class="score-track"><span style="width:${Math.max(1,item.score*100)}%"></span></div><strong>${(item.score*100).toFixed(1)}%</strong></div>`).join("");
}

function renderDifficultyChecks(checks) {
  const marks={yes:["✓","满足难例条件"],no:["×","不满足"],pending:["?","尚未获得所需结果"]};
  return (checks||[]).map(check=>{const [mark,hint]=marks[check.status]||marks.pending;return `<li class="difficulty-check ${check.status}"><span>${escapeHtml(check.label)}</span><b title="${hint}">${mark}</b></li>`}).join("");
}

function renderTeacher(teacher, pseudoRecord = null, prediction = null, difficultyChecks = []) {
  const status = $("#teacher-status"), root = $("#teacher-result"), distribution=$("#teacher-distribution"), collaboration=$("#collaboration-evidence");
  if(collaboration){
    const skipped = teacher?.status === "skipped";
    collaboration.classList.toggle("hidden", skipped);
    collaboration.innerHTML=skipped ? "" : `<section><strong>四项协同判定</strong><ul class="difficulty-check-list">${renderDifficultyChecks(difficultyChecks)}</ul></section>`;
  }
  if (teacher?.status === "completed" && teacher.result) {
    status.className = "status-pill online";
    status.textContent = "分析完成";
    root.className = "teacher-result";
    const result = teacher.result;
    const label = result.label || result.suggested_label;
    const evidence = (result.evidence || []).map(item => `<li>${escapeHtml(item.description || item.type || "未提供描述")}${item.segment_id ? ` <code>${escapeHtml(item.segment_id)}</code>` : ""}</li>`).join("");
    const counterEvidence = (result.counter_evidence || []).map(item => `<li>${escapeHtml(item)}</li>`).join("");
    root.innerHTML = `<div><span>建议类别</span><strong>${escapeHtml(LABEL_NAME[label] || label || "—")}</strong></div><div><span>置信度</span><strong>${result.confidence == null ? "—" : `${(result.confidence * 100).toFixed(1)}%`}</strong></div><p>${escapeHtml(result.reason || result.reasoning_summary || "暂无判定依据")}</p>${evidence ? `<section class="teacher-evidence"><strong>教师依据</strong><ul>${evidence}</ul></section>` : ""}${counterEvidence ? `<section class="teacher-evidence counter"><strong>反向依据／限制</strong><ul>${counterEvidence}</ul></section>` : ""}`;
    renderSixDistribution(distribution,result.distribution,"尚无教师六类分布");
    return;
  }
  if (pseudoRecord?.sample_id) {
    status.className = "status-pill online";
    status.textContent = "伪标签待复核";
    root.className = "teacher-result";
    const conflicts = (pseudoRecord.conflicts || []).join("、") || "质量分处于人工复核区间";
    root.innerHTML = `<div><span>建议类别</span><strong>${escapeHtml(LABEL_NAME[pseudoRecord.label] || pseudoRecord.label || "—")}</strong></div><div><span>质量分</span><strong>${pseudoRecord.quality_score == null ? "—" : `${(pseudoRecord.quality_score * 100).toFixed(1)}%`}</strong></div><p>${escapeHtml(conflicts)}</p>`;
    renderSixDistribution(distribution,pseudoRecord.distribution,"尚无教师六类分布");
    return;
  }
  const running = ["queued","running"].includes(teacher?.status);
  const skipped = teacher?.status === "skipped";
  const blockedQuality = teacher?.status === "blocked_quality";
  status.className = `status-pill ${running ? "online" : "disabled"}`;
  status.textContent = running ? titleForStatus(teacher.status) : blockedQuality ? "骨架质量不足" : skipped ? "已跳过" : teacher?.status === "failed" ? "分析失败" : teacher?.status === "not_run" ? "尚未运行" : "尚未启用";
  root.className = "teacher-placeholder";
  root.innerHTML = `<div class="teacher-icon">◇</div><div><strong>${running ? "Qwen 多模态教师正在分析" : blockedQuality ? "骨架质量门控未调用 Qwen" : skipped ? "学生高置信，未进入难例判定与 Qwen" : "Qwen 多模态教师"}</strong>${skipped ? "" : `<p>${escapeHtml(teacher?.reason || "读取骨架视频，并在当前模型对应的标签空间内给出独立判断。")}</p>`}</div>`;
  renderSixDistribution(distribution,null,"尚无教师六类分布");
}

function renderPrediction(prediction) {
  const root=$("#prediction-list");
  if(!prediction?.topk?.length){root.className="prediction-list empty-state";root.textContent="尚无预测结果";return;}
  root.className="prediction-list";
  root.innerHTML=prediction.topk.map((item,index)=>`<div class="prediction-row"><span class="rank">${index+1}</span><span>${escapeHtml(LABEL_NAME[item.label]||item.label)}</span><div class="score-track"><span style="width:${Math.max(1,item.score*100)}%"></span></div><strong>${(item.score*100).toFixed(1)}%</strong></div>`).join("");
}

function renderHardOptions() {
  const select=$("#hard-sample-select");if(!select)return;
  $("#hard-total").textContent=state.hardTotal||0;
  $("#hard-badge").textContent=state.hardTotal||0;
  const current=state.hardSample?.sample_id||select.value;
  select.innerHTML=state.hardSamples.length
    ? state.hardSamples.map((item,index)=>`<option value="${escapeHtml(item.sample_id)}">#${index+1} · ${escapeHtml(item.sample_id)} · 命中 ${item.matched_condition_count||0} 项</option>`).join("")
    : '<option value="">当前没有命中条件的难例</option>';
  if(state.hardSamples.some(item=>item.sample_id===current))select.value=current;
  const index=state.hardSamples.findIndex(item=>item.sample_id===current);
  const previous=$("#hard-sample-prev"),next=$("#hard-sample-next");
  if(previous) previous.disabled=index<=0;
  if(next) next.disabled=index<0 || index>=state.hardSamples.length-1;
}

function renderHardTeacher(teacher) {
  const status=$("#hard-teacher-status"),root=$("#hard-teacher-result"),distribution=$("#hard-teacher-distribution");
  const result=teacher?.result;
  if(result){
    status.className="status-pill online";status.textContent="分析完成";
    const evidence=(result.evidence||[]).map(item=>`<li>${escapeHtml(item.description||item.type)}${item.segment_id?` <code>${escapeHtml(item.segment_id)}</code>`:""}</li>`).join("");
    const counter=(result.counter_evidence||[]).map(item=>`<li>${escapeHtml(item)}</li>`).join("");
    root.className="teacher-result";
    root.innerHTML=`<div><span>教师建议</span><strong>${escapeHtml(LABEL_NAME[result.label]||result.label)}</strong></div><div><span>教师自报置信度</span><strong>${result.confidence==null?"—":`${(result.confidence*100).toFixed(1)}%`}</strong></div>${evidence?`<section class="teacher-evidence"><strong>关键证据</strong><ul>${evidence}</ul></section>`:""}${counter?`<section class="teacher-evidence counter"><strong>反向证据／限制</strong><ul>${counter}</ul></section>`:""}`;
    renderSixDistribution(distribution,result.distribution,"尚无教师六类分布");
    return;
  }
  status.className="status-pill disabled";status.textContent=titleForStatus(teacher?.status||"not_run");
  root.className="teacher-placeholder";root.textContent=teacher?.reason||"难例已入队，等待服务器 GPU 准入";
  renderSixDistribution(distribution,null,"尚无教师六类分布");
}

async function loadHardSamples() {
  try {
    const hard = await api("/api/hard-samples?offset=0&limit=50");
    state.hardSamples = hard.items;
    state.hardTotal = hard.total;
    state.hardLoaded = true;
    renderHardOptions();
    if (!state.hardSample && hard.items.length) await selectHardSample(hard.items[0].sample_id);
  } catch (error) {
    toast(error.message, true);
  }
}

async function selectHardSample(sampleId) {
  const item=state.hardSamples.find(value=>value.sample_id===sampleId);if(!item)return;
  state.hardSample=item;renderHardOptions();
  const hardVideo = $("#hard-pose-video");
  setVideo(hardVideo,item.media?.pose||"");
  $("#hard-pose-status").textContent=item.artifacts?.pose_video ? "已生成" : "按需生成中";
  hardVideo.onloadeddata = () => { $("#hard-pose-status").textContent = "已生成"; };
  hardVideo.onerror = () => { $("#hard-pose-status").textContent = "生成失败"; };
  $("#hard-score").className="status-pill online";
  $("#hard-score").textContent=`命中 ${item.matched_condition_count||0} 项`;
  const predictionRoot=$("#hard-prediction-list");
  predictionRoot.className="prediction-list";
  predictionRoot.innerHTML=(item.prediction?.topk||[]).map((entry,index)=>`<div class="prediction-row"><span class="rank">${index+1}</span><span>${escapeHtml(LABEL_NAME[entry.label]||entry.label)}</span><div class="score-track"><span style="width:${Math.max(1,entry.score*100)}%"></span></div><strong>${(entry.score*100).toFixed(1)}%</strong></div>`).join("");
  const checks=renderDifficultyChecks(item.difficulty_checks);
  $("#hard-evidence").innerHTML=`<section><strong>四项难例判定</strong><ul class="difficulty-check-list">${checks}</ul></section><section class="hard-label-section"><strong>人工选择标注</strong><div class="hard-label-grid">${LABELS.map(([id,name])=>`<button type="button" class="hard-label-button" data-hard-label="${id}">${name}</button>`).join("")}</div></section>`;
  $$('[data-hard-label]').forEach(button=>button.onclick=()=>submitHardReview(button.dataset.hardLabel));
  renderHardTeacher(item.teacher);
}

async function moveHardSample(delta) {
  const current=state.hardSample?.sample_id;
  const index=state.hardSamples.findIndex(item=>item.sample_id===current);
  const next=state.hardSamples[index+delta];
  if(next) await selectHardSample(next.sample_id);
}

async function submitHardReview(finalLabel) {
  const sample=state.hardSample;
  if(!sample){toast("请先选择待审核难例",true);return;}
  try {
    const reviewed=await api(`/api/samples/${encodeURIComponent(sample.sample_id)}/review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reviewer:reviewer(),final_label:finalLabel,reason_code:"human_reviewed_hard_sample",note:""})});
    if(!reviewed.incremental_pool)throw new Error("难例未能进入增量学习难例池");
    toast(`已标注为${LABEL_NAME[finalLabel]}，已移入增量学习难例池`);
    state.hardSample=null;state.hardLoaded=false;
    await refreshAll(true);
    await loadHardSamples();
  } catch(error){toast(error.message,true);}
}

async function startJob(action, sampleId = state.inferenceSample?.sample_id) {
  if(!sampleId){toast("请先选择样本",true);return;}
  if(state.activeJobId || state.jobSubmissionPending){toast("已有推理任务正在运行，请等待完成");return;}
  state.jobSubmissionPending=true;renderCapabilities();
  try {
    const job=await api(`/api/samples/${encodeURIComponent(sampleId)}/jobs`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action})});
    state.jobSubmissionPending=false;state.activeJobId=job.job_id;renderCapabilities();
    if(job.reused)toast("该样本已有相同任务，继续跟踪现有进度");
    showJob(job); pollJob(job.job_id,sampleId);
  } catch(error){state.jobSubmissionPending=false;state.activeJobId=null;renderCapabilities();toast(error.message,true);}
}

function showJob(job) {
  const root=$("#inference-job");root.classList.remove("hidden");
  root.querySelector("span").style.width=`${Math.max(2,job.progress*100)}%`;
  root.querySelector("p").textContent=`${job.message||titleForStatus(job.status)} · ${job.sample_id}`;
}

async function pollJob(jobId,sampleId) {
  try {
    if(state.activeJobId!==jobId)return;
    const job=await api(`/api/jobs/${jobId}`);showJob(job);
    if(["queued","running"].includes(job.status)){setTimeout(()=>pollJob(jobId,sampleId),900);return;}
    state.activeJobId=null;renderCapabilities();
    if(job.status==="failed") toast(job.message||"任务失败",true); else toast("处理完成");
    await refreshAll(true);
    if(state.inferenceSample?.sample_id===sampleId) await selectInference(sampleId,false);
  } catch(error){state.activeJobId=null;state.jobSubmissionPending=false;renderCapabilities();toast(error.message,true);}
}

function filteredReviewSamples() {
  return state.reviewSamples;
}

async function selectReview(index, autoPreview = true) {
  const items=filteredReviewSamples();if(!items.length)return;
  index=Math.max(0,Math.min(index,items.length-1));state.reviewIndex=index;
  await selectInference(items[index].sample_id, autoPreview);
}

async function submitReview(finalLabel) {
  if(!state.inferenceSample){toast("请先选择待审核样本",true);return;}
  try {
    const currentId=state.inferenceSample.sample_id;
    const currentIndex=Math.max(0,filteredReviewSamples().findIndex(item=>item.sample_id===currentId));
    const suggested=state.inferenceSample.suggested_label;
    const reason = finalLabel==="damaged"?"damaged":finalLabel==="out_of_scope"?"out_of_scope":finalLabel==="unknown"?"insufficient_evidence":finalLabel===suggested?"accept_suggestion":finalLabel.startsWith("playful")?"intent_playful":finalLabel.startsWith("conflict")?"intent_conflict":"manual_correction";
    const reviewed = await api(`/api/samples/${encodeURIComponent(currentId)}/review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reviewer:reviewer(),final_label:finalLabel,reason_code:reason,note:$("#review-note").value})});
    toast(reviewed.incremental_pool ? "人工标签与 GCN 不一致，已进入增量学习难例池" : `已保存人工结果与模型快照：${LABEL_NAME[finalLabel]}`);state.reviewSample=null;await refreshAll(true);
    if(state.reviewSamples.length) await selectReview(Math.min(currentIndex,state.reviewSamples.length-1));
  } catch(error){toast(error.message,true);}
}

function renderLabelButtons() {
  $("#label-buttons").innerHTML=LABELS.map(([id,name,desc],index)=>`<button class="label-button" data-label="${id}"><span class="hotkey">${index+1}</span><strong>${name}</strong><small>${desc}</small></button>`).join("");
  $$("[data-label]").forEach(button=>button.onclick=()=>submitReview(button.dataset.label));
  $$("[data-special]").forEach(button=>button.onclick=()=>submitReview(button.dataset.special));
}

async function loadDatasetPage() {
  const status=$("#dataset-status").value,dataset=$("#dataset-source").value,query=$("#dataset-search").value.trim();
  try{
    const result=await fetchSampleSummaries({workflow_status:status,dataset,query,offset:state.datasetPage*state.datasetPageSize,limit:state.datasetPageSize});
    const pages=Math.max(1,Math.ceil(result.total/state.datasetPageSize));if(state.datasetPage>=pages){state.datasetPage=pages-1;return loadDatasetPage();}
    const items=result.items;
    state.datasetTotal=result.total;$("#dataset-total").textContent=result.total;
    $("#dataset-table").innerHTML=items.map(item=>`<tr><td><strong>${escapeHtml(item.sample_id)}</strong></td><td>${escapeHtml(item.source_dataset)}</td><td>${escapeHtml(LABEL_NAME[item.student_label]||item.student_label||"未标注")}</td><td>${escapeHtml(LABEL_NAME[item.teacher_label]||item.teacher_label||"未标注")}</td><td>${escapeHtml(LABEL_NAME[item.manual_label]||item.manual_label||"—")}</td><td>${formatTime(item.produced_at)}</td><td><span class="state-badge ${item.workflow_status}">${titleForStatus(item.workflow_status)}</span></td><td><button class="table-action" data-open-review="${escapeHtml(item.sample_id)}">查看</button></td></tr>`).join("");
    $("#page-info").textContent=`第 ${state.datasetPage+1} / ${pages} 页`;$("#page-prev").disabled=state.datasetPage===0;$("#page-next").disabled=state.datasetPage+1>=pages;
    $$("[data-open-review]").forEach(button=>button.onclick=async()=>{setPage("inference");await selectInference(button.dataset.openReview);});
  }catch(error){toast(error.message,true);}
}

function renderModels() {
  const models=[
    ["骨架提取模型","RTMDet-S + RTMPose-S","双人 COCO-17 骨架提取与匿名化跟踪",state.capabilities.pose_extraction?.enabled],
    ["GCN 模型","M1KD QAT INT8 + Logits KD","5.31 MB 量化蒸馏六分类模型",state.capabilities.campus6_inference?.enabled],
    ["多模态大模型","Qwen3-VL-8B","基于骨架视频与结构化语义分析难例",state.capabilities.qwen_teacher?.enabled],
  ];
  $("#model-grid").innerHTML=models.map(([type,name,desc,enabled])=>`<article class="panel model-card compact"><div class="panel-head"><div class="model-icon">◈</div><span class="status-pill ${enabled?"online":"disabled"}">${enabled?"可运行":"不可用"}</span></div><label>${type}</label><select class="model-select" aria-label="${type}"><option selected>${name}</option></select><p>${desc}</p></article>`).join("");
}

function modelReleaseMarkup() {
  const value=state.modelStatus||{}, production=value.production||{}, metrics=production.metrics||{}, candidate=value.candidate;
  const perClass=metrics.per_class||{};
  const classRows=LABELS.map(([id,name])=>`<div><span>${name}</span><strong>${perClass[id]?.accuracy==null?"—":`${(perClass[id].accuracy*100).toFixed(1)}%`}</strong></div>`).join("");
  const stages=["candidate","validated","production"].map(stage=>`<span class="release-stage ${stage==="production"?"active":""}">${({candidate:"Candidate",validated:"Validated",production:"Production"})[stage]}</span>`).join('<i>→</i>');
  return `<div class="model-release"><article class="panel production-card"><div class="panel-head"><div><span class="panel-kicker">CURRENT MODEL</span><h3>${escapeHtml(production.name||"M1KD QAT INT8 + Logits KD")}</h3></div><span class="status-pill online">Production</span></div><div class="release-facts"><div><span>生成时间</span><strong>${formatTime(production.generated_at)}</strong></div><div><span>发布时间</span><strong>${formatTime(production.deployed_at)}</strong></div><div><span>模型大小</span><strong>${production.size_mb==null?"—":`${production.size_mb.toFixed(2)} MB`}</strong></div><div><span>六分类总体准确率</span><strong>${metrics.overall_accuracy==null?"—":`${(metrics.overall_accuracy*100).toFixed(1)}%`}</strong></div><div><span>平均类别准确率</span><strong>${metrics.mean_class_accuracy==null?"—":`${(metrics.mean_class_accuracy*100).toFixed(1)}%`}</strong></div></div><div class="per-class-metrics">${classRows}</div></article><article class="panel release-card"><div class="panel-head"><h3>模型评估与发布</h3><span class="status-pill ${candidate?"online":"disabled"}">${candidate?escapeHtml(candidate.status):"无候选模型"}</span></div><div class="release-flow">${stages}</div><p>${candidate?`最近候选模型 ${escapeHtml(candidate.model_id||"—")} 当前状态：${escapeHtml(candidate.status||"candidate")}。`:"当前没有待评估 Candidate；新模型通过总体、平均类别与逐类指标门槛后方可进入 Production。"}</p><div class="rollback-state"><span>回滚</span><strong>${value.rollback?.available?"可恢复上一 Production":"暂无上一 Production"}</strong></div></article></div>`;
}

function durationText(seconds) {
  if(seconds==null)return "等待训练进程估算";
  const value=Math.max(0,Math.round(seconds));
  const hours=Math.floor(value/3600),minutes=Math.floor(value%3600/60),secs=value%60;
  return [hours?`${hours} 小时`:"",minutes?`${minutes} 分钟`:"",(!hours&&secs)?`${secs} 秒`:""].filter(Boolean).join(" ")||"即将完成";
}

function renderTrainingStatus() {
  const root=$("#training-status");if(!root)return;
  const value=state.trainingStatus||{},running=Boolean(value.running);
  root.innerHTML=`<div class="training-metrics"><article class="panel"><span>当前状态</span><strong>${running?"正在训练":value.status==="failed"?"训练异常":"未训练"}</strong></article><article class="panel"><span>上次训练后新增数据</span><strong>${value.new_sample_count||0} 条</strong></article><article class="panel"><span>当前阶段</span><strong>${({waiting_for_data:"等待新增数据",train:"模型训练",validate:"模型评估",completed:"训练完成"})[value.stage]||value.stage||"—"}</strong></article><article class="panel"><span>预计剩余</span><strong>${running?durationText(value.estimated_remaining_seconds):"—"}</strong></article></div><article class="panel training-progress-card"><div class="panel-head"><div><h3>${running?"增量训练进行中":"增量训练状态"}</h3><p>最近训练：${formatTime(value.last_training_at)}</p></div><span class="status-pill ${running?"online":"disabled"}">${running?`${Math.round((value.progress||0)*100)}%`:"IDLE"}</span></div><div class="training-progress"><span style="width:${Math.max(0,Math.min(100,(value.progress||0)*100))}%"></span></div>${value.message?`<p>${escapeHtml(value.message)}</p>`:""}</article>${modelReleaseMarkup()}`;
}

function renderSettings() {
  renderMacroParameters();
  renderGpuSettings();
}

function macroInputAttrs(type) {
  if (type === "unit_interval") return 'step="0.01" min="0" max="1"';
  if (type === "positive_float") return 'step="0.1" min="0.0001"';
  return 'step="1000" min="0"';
}

function renderMacroParameters() {
  const root = $("#macro-grid");
  if (!root || !state.macroParameters?.parameters) return;
  // Preserve in-progress edits across re-renders (refresh, apply).
  const dirty = {};
  $$("#macro-grid input[data-macro-key]").forEach(input => {
    if (input.dataset.dirty === "1") dirty[input.dataset.macroKey] = input.value;
  });
  const groups = state.macroParameters.groups || {};
  const parameters = state.macroParameters.parameters || [];
  root.innerHTML = ["gate"].map(groupKey => {
    const items = parameters.filter(item => item.group === groupKey);
    if (!items.length) return "";
    const cards = items.map(item => {
      const disabled = item.source === "environment" ? "disabled" : "";
      return `<label class="macro-item">
        <span class="macro-item-head"><strong>${escapeHtml(item.name)}</strong></span>
        <span class="macro-input-row"><input type="number" data-macro-key="${escapeHtml(item.key)}" value="${item.value}" ${macroInputAttrs(item.type)} ${disabled}></span>
      </label>`;
    }).join("");
    return `<div class="macro-group"><div class="macro-group-title">${escapeHtml(groups[groupKey] || groupKey)}</div><div class="macro-group-grid">${cards}</div></div>`;
  }).join("");
  Object.entries(dirty).forEach(([key, value]) => {
    const input = root.querySelector(`input[data-macro-key="${key}"]`);
    if (input && !input.disabled) { input.value = value; input.dataset.dirty = "1"; }
  });
}

async function applyMacroParameters() {
  const values = {};
  $$("#macro-grid input[data-macro-key]").forEach(input => {
    if (input.disabled) return;
    values[input.dataset.macroKey] = Number(input.value);
  });
  if (!Object.keys(values).length) { toast("没有可应用的宏参数", true); return; }
  try {
    state.macroParameters = await api("/api/macro-parameters", {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({values})});
    $$("#macro-grid input[data-macro-key]").forEach(input => { delete input.dataset.dirty; });
    renderMacroParameters();
    toast("宏参数已应用并保存，后续新任务立即生效");
  } catch (error) { toast(error.message, true); }
}

function resetMacroParameters() {
  const list = state.macroParameters?.parameters || [];
  $$("#macro-grid input[data-macro-key]").forEach(input => {
    if (input.disabled) return;
    const spec = list.find(item => item.key === input.dataset.macroKey);
    if (spec) { input.value = spec.default; input.dataset.dirty = "1"; }
  });
  toast("输入已恢复为默认值，点击「应用宏参数」生效");
}

function renderGpuSettings() {
  const value=state.gpus||{}, gpus=value.gpus||[];
  const status=$("#gpu-status"), save=$("#save-gpu-settings");
  if(!status||!save||!$("#gpu-grid"))return;
  status.textContent=value.available?`${gpus.length} 张 ProtoGCN GPU 可用`:"未检测到 GPU";
  status.className=`status-pill ${value.available?"online":"disabled"}`;
  save.disabled=!value.available;
  if(!gpus.length){
    $("#gpu-grid").innerHTML='<div class="empty-state">nvidia-smi 未返回可用 GPU。请确认容器已挂载 NVIDIA 设备与驱动。</div>';
    return;
  }
  const pose=new Set(value.pose_gpu_ids||[]), student=new Set(value.student_gpu_ids||[]), teacher=new Set(value.teacher_gpu_ids||[]), teacherAuto=value.teacher_mode!=="manual";
  $("#teacher-gpu-auto").checked=teacherAuto;
  $("#gpu-grid").innerHTML=gpus.map(gpu=>{
    const ratio=gpu.memory_total_mb?Math.min(100,Math.round(gpu.memory_used_mb/gpu.memory_total_mb*100)):0;
    const serviceState=(gpu.service_roles||[]).length?`本服务占用：${gpu.service_roles.join("、")}`:"本服务空闲";
    return `<div class="gpu-card">
      <div class="gpu-card-head"><span>GPU ${gpu.index}</span><strong>${escapeHtml(gpu.name)}</strong></div>
      <div class="gpu-service-state ${gpu.service_busy?"busy":"idle"}">${escapeHtml(serviceState)}</div>
      <div class="gpu-live"><span>利用率 ${gpu.utilization_gpu_percent}%</span><span>${gpu.temperature_c}°C</span></div>
      <div class="gpu-memory"><span style="width:${ratio}%"></span></div>
      <small>${gpu.memory_used_mb} / ${gpu.memory_total_mb} MiB</small>
      <label><input type="checkbox" data-gpu-kind="pose" value="${gpu.index}" ${pose.has(gpu.index)?"checked":""}> RTMDet/RTMPose</label>
      <label><input type="checkbox" data-gpu-kind="student" value="${gpu.index}" ${student.has(gpu.index)?"checked":""}> M1KD Campus6</label>
      <label><input type="checkbox" data-gpu-kind="teacher" value="${gpu.index}" ${teacher.has(gpu.index)?"checked":""} ${teacherAuto?"disabled":""}> Qwen3-VL-8B</label>
    </div>`;
  }).join("");
}

async function refreshGpus() {
  try{state.gpus=await api("/api/gpus");renderGpuSettings();toast("GPU 状态已刷新");}
  catch(error){toast(error.message,true)}
}

async function saveGpuSettings() {
  const selected=kind=>$$(`input[data-gpu-kind="${kind}"]:checked`).map(input=>Number(input.value));
  try{
    state.gpus=await api("/api/gpus",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({pose_gpu_ids:selected("pose"),student_gpu_ids:selected("student"),teacher_gpu_ids:selected("teacher"),teacher_auto:$("#teacher-gpu-auto").checked})});
    renderGpuSettings();toast("GPU 设置已保存，后续新任务立即生效");
  }catch(error){toast(error.message,true)}
}

function bindEvents() {
  $$(".nav-item").forEach(button=>button.onclick=()=>setPage(button.dataset.page));
  $$("[data-goto]").forEach(button=>button.onclick=()=>setPage(button.dataset.goto));
  $("#sidebar-toggle").onclick=()=>setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
  $("#refresh-button").onclick=()=>refreshAll(true);
  $("#sample-tree-toggle").onclick=event=>{event.stopPropagation();setSampleTreeOpen($("#sample-tree-menu").classList.contains("hidden"))};
  $("#sample-tree-menu").onclick=event=>event.stopPropagation();
  $("#inference-search").onclick=event=>{event.stopPropagation();setSampleTreeOpen(true)};
  let inferenceSearchTimer;
  $("#inference-search").oninput=event=>{clearTimeout(inferenceSearchTimer);inferenceSearchTimer=setTimeout(()=>loadInferenceSamples(event.target.value.trim()),250);setSampleTreeOpen(true)};
  $("#sample-prev").onclick=()=>moveInference(-1);
  $("#sample-next").onclick=()=>moveInference(1);
  document.addEventListener("click",()=>setSampleTreeOpen(false));
  if($("#refresh-gpus"))$("#refresh-gpus").onclick=refreshGpus;
  if($("#save-gpu-settings"))$("#save-gpu-settings").onclick=saveGpuSettings;
  if($("#teacher-gpu-auto"))$("#teacher-gpu-auto").onchange=event=>$$('input[data-gpu-kind="teacher"]').forEach(input=>input.disabled=event.target.checked);
  if($("#apply-macro-parameters"))$("#apply-macro-parameters").onclick=applyMacroParameters;
  if($("#reset-macro-parameters"))$("#reset-macro-parameters").onclick=resetMacroParameters;
  if($("#macro-grid"))$("#macro-grid").addEventListener("input",event=>{if(event.target.dataset.macroKey)event.target.dataset.dirty="1";});
  if($("#hard-sample-select"))$("#hard-sample-select").onchange=event=>selectHardSample(event.target.value);
  if($("#hard-sample-prev"))$("#hard-sample-prev").onclick=()=>moveHardSample(-1);
  if($("#hard-sample-next"))$("#hard-sample-next").onclick=()=>moveHardSample(1);
  $("#previous-review").onclick=()=>selectReview(state.reviewIndex-1);$("#next-review").onclick=()=>selectReview(state.reviewIndex+1);
  $("#undo-review").onclick=async()=>{try{const sample=await api("/api/reviews/undo-last",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({actor:reviewer()})});toast(`已撤销 ${sample.sample_id} 的最近审核`);await refreshAll(true);}catch(error){toast(error.message,true)}};
  ["#dataset-source","#dataset-status"].forEach(s=>$(s).onchange=()=>{state.datasetPage=0;loadDatasetPage()});let searchTimer;$("#dataset-search").oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.datasetPage=0;loadDatasetPage()},250)};
  $("#page-prev").onclick=()=>{state.datasetPage=Math.max(0,state.datasetPage-1);loadDatasetPage()};$("#page-next").onclick=()=>{state.datasetPage++;loadDatasetPage()};
  $("#export-dataset").onclick=async()=>{try{const value=await api("/api/datasets/export",{method:"POST"});toast(`已导出到服务器：${value.path}`);location.href=value.download_url}catch(error){toast(error.message,true)}};
  $("#reviewer").value=localStorage.getItem("dahua-reviewer")||"reviewer";$("#reviewer").onchange=()=>localStorage.setItem("dahua-reviewer",$("#reviewer").value.trim());
  document.addEventListener("keydown",event=>{if(event.key==="Escape")setSampleTreeOpen(false);const page=$("#page-inference");if(!page||!page.classList.contains("active")||["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return;if(/^[1-6]$/.test(event.key))submitReview(LABELS[Number(event.key)-1][0]);else if(event.key==="ArrowLeft")selectReview(state.reviewIndex-1);else if(event.key==="ArrowRight")selectReview(state.reviewIndex+1);});
}

async function init() {
  setSidebarCollapsed(localStorage.getItem("dahua-sidebar-collapsed") === "1");bindEvents();renderLabelButtons();setPage(location.hash.slice(1)||"dashboard");await refreshAll(false);
}
document.addEventListener("DOMContentLoaded",init);
