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
const SAMPLE_STATUS_GROUPS = [
  ["pending", "待审核"], ["reviewed", "已标注"], ["unknown", "无法判断"],
  ["damaged", "视频损坏"], ["out_of_scope", "无关样本"],
];

const state = {
  capabilities: {}, system: {}, gpus: {}, dashboard: {}, samples: [], reviewSamples: [],
  evolution: {},
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
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function projectPath(value) {
  if (!value) return "—";
  const root = String(state.system.repository_root || "").replace(/\/+$/, "");
  const path = String(value);
  if (root && (path === root || path.startsWith(`${root}/`))) {
    return path === root ? "." : `./${path.slice(root.length + 1)}`;
  }
  const dataRoot = String(state.system.data_root || "").replace(/\/+$/, "");
  if (dataRoot && (path === dataRoot || path.startsWith(`${dataRoot}/`))) {
    return path === dataRoot ? "DAHUA_DATA_ROOT" : `DAHUA_DATA_ROOT/${path.slice(dataRoot.length + 1)}`;
  }
  return path;
}

function titleForStatus(status) {
  return ({pending:"待审核",reviewed:"已标注",unknown:"无法判断",damaged:"损坏",out_of_scope:"无关",queued:"排队中",running:"处理中",completed:"完成",failed:"失败",skipped:"已跳过",blocked_quality:"骨架质量不足"})[status] || status;
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
  const titles = {dashboard:"运行总览",inference:"行为识别结果",reasoning:"语义理解与推理依据",privacy:"Skeleton 隐私模式",dataset:"审计记录",evolution:"闭环中心",models:"模型管理",settings:"系统设置"};
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.page === page));
  $$(".page").forEach(item => item.classList.toggle("active", item.id === `page-${page}`));
  $("#page-title").textContent = titles[page];
  history.replaceState(null, "", `#${page}`);
  if (page === "dataset") loadDatasetPage();
}

function reviewer() {
  const value = $("#reviewer").value.trim();
  if (!value) throw new Error("请先填写审核员名称");
  localStorage.setItem("dahua-reviewer", value);
  return value;
}

async function fetchAllSamples(params = {}) {
  const query = new URLSearchParams({...params, offset: 0, limit: 200});
  const first = await api(`/api/samples?${query}`);
  const items = [...first.items];
  for (let offset = 200; offset < first.total; offset += 200) {
    query.set("offset", offset);
    items.push(...(await api(`/api/samples?${query}`)).items);
  }
  return items;
}

async function refreshAll(preserveSelection = true) {
  try {
    const [capabilities, system, gpus, dashboard, evolution, samples, reviewSamples] = await Promise.all([
      api("/api/capabilities"), api("/api/system"), api("/api/gpus"), api("/api/dashboard"),
      Promise.resolve({enabled:false,status:"manual",reason:"Campus6 闭环由审核、伪标签和训练脚本按批次执行"}),
      fetchAllSamples(), fetchAllSamples({status:"pending"}),
    ]);
    Object.assign(state, {capabilities, system, gpus, dashboard, evolution, samples, reviewSamples});
    $("#server-dot").classList.add("online");
    $("#server-label").textContent = "服务器在线";
    renderDashboard(); renderCapabilities(); renderSampleOptions(); renderModels(); renderSettings(); renderPipeline(); renderEvolution();
    if (!preserveSelection || !state.inferenceSample) {
      if (samples.length) await selectInference(samples[0].sample_id, false);
    } else {
      const current = samples.find(item => item.sample_id === state.inferenceSample.sample_id);
      if (current) await selectInference(current.sample_id, false);
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
  const names = {pose_extraction:"RTMPose COCO-17 骨架",campus6_inference:"M1FKD INT8 Campus6 推理",manual_review:"人工审核",dataset_export:"数据导出",qwen_teacher:"Qwen3-VL-8B 服务器教师",live_camera:"实时摄像头"};
  $("#capability-list").innerHTML = Object.entries(state.capabilities).map(([id,item]) => `<div class="capability ${item.enabled ? "enabled":""}" title="${escapeHtml(item.reason)}"><span></span><div><strong>${names[id]||id}</strong><small>${item.enabled ? "已启用" : item.reason}</small></div></div>`).join("");
}

function renderSampleOptions() {
  const selected = state.inferenceSample?.sample_id;
  const query = ($("#inference-search")?.value || "").trim().toLowerCase();
  const root = $("#sample-tree-groups");
  if (root) {
    const openStatuses = new Set($$("#sample-tree-groups details[open]").map(item => item.dataset.status));
    const selectedSample = state.samples.find(item => item.sample_id === selected);
    root.innerHTML = SAMPLE_STATUS_GROUPS.map(([status, name]) => {
      const allItems = state.samples.filter(item => item.status === status);
      const items = allItems.filter(item => !query || `${item.sample_id} ${item.source_dataset} ${item.source_label} ${item.suggested_coarse_label}`.toLowerCase().includes(query));
      const opened = Boolean(query) || openStatuses.has(status) || selectedSample?.status === status;
      const count = query ? `${items.length}/${allItems.length}` : allItems.length;
      const children = items.length ? items.map(item => `<button class="sample-tree-item ${item.sample_id===selected?"active":""}" data-sample-id="${escapeHtml(item.sample_id)}"><strong>${escapeHtml(item.sample_id)}</strong><small>${escapeHtml(item.source_dataset||"未知来源")} · ${escapeHtml(item.source_label||item.suggested_coarse_label||"无类别")}</small></button>`).join("") : '<div class="sample-tree-empty">没有匹配样本</div>';
      return `<details class="sample-status-group" data-status="${status}" ${opened?"open":""}><summary><strong>${name}</strong><b>${count}</b></summary><div class="sample-tree-items">${children}</div></details>`;
    }).join("");
    $$("[data-sample-id]").forEach(button => button.onclick = async () => {
      setSampleTreeOpen(false);
      await selectInference(button.dataset.sampleId);
    });
  }
  const current = $("#sample-tree-current");
  if (current) {
    const sample = state.samples.find(item => item.sample_id === selected);
    current.textContent = sample ? `${sample.sample_id} · ${sample.source_dataset || "Campus6"}` : "请选择行为片段";
  }
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
    state.inferenceSample = sample;
    state.reviewSample = sample;
    setSampleTreeOpen(false);
    renderSampleOptions();
    setVideo($("#pose-video"), sample.artifacts.pose_video ? sample.media.pose : "");
    $("#pose-status").textContent = sample.artifacts.pose_video ? "已生成" : "未生成";
    renderPrediction(sample.prediction);
    renderTeacher(sample.teacher, sample.pseudo_record);
    renderReasoning(sample.teacher, sample.prediction);
    $("#review-note").value = sample.note || "";
    const pendingItems = filteredReviewSamples();
    const queueIndex = pendingItems.findIndex(item => item.sample_id === sample.sample_id);
    if (queueIndex >= 0) state.reviewIndex = queueIndex;
  } catch(error){ toast(error.message,true); }
}

function renderReasoning(teacher, prediction) {
  const status=$("#reasoning-status"), summary=$("#reasoning-summary"), distribution=$("#reasoning-distribution"), reason=$("#reasoning-reason"), trigger=$("#reasoning-trigger");
  if(!status||!summary||!distribution||!reason||!trigger)return;
  const gate=prediction?.teacher_gate||{};
  const result=teacher?.result;
  if(result){
    status.className="status-pill online";status.textContent="教师分析完成";
    summary.className="teacher-result";
    summary.innerHTML=`<div><span>最终建议</span><strong>${escapeHtml(LABEL_NAME[result.label]||result.label||"—")}</strong></div><div><span>置信度</span><strong>${result.confidence==null?"—":`${(result.confidence*100).toFixed(1)}%`}</strong></div>`;
    const values=result.distribution||{};
    distribution.className="prediction-list";
    distribution.innerHTML=LABELS.map(([id,name],index)=>`<div class="prediction-row"><span class="rank">${index+1}</span><span>${name}</span><div class="score-track"><span style="width:${Math.max(1,(values[id]||0)*100)}%"></span></div><strong>${((values[id]||0)*100).toFixed(1)}%</strong></div>`).join("");
    reason.textContent=result.reason||result.reasoning_summary||"教师未返回文字依据";
  }else{
    const skipped=teacher?.status==="skipped";
    status.className="status-pill disabled";status.textContent=skipped?"学生结果已接受":titleForStatus(teacher?.status||"pending");
    summary.className="teacher-placeholder";summary.textContent=teacher?.reason||"后台尚未产生教师结果";
    distribution.className="prediction-list empty-state";distribution.textContent="尚无教师分布";
    reason.textContent=skipped?"学生置信度与类别间隔达到自动接受阈值，无需调用教师。":"尚无语义判断依据";
  }
  trigger.textContent=JSON.stringify({route:gate.route||"pending",reasons:gate.reasons||[],student_confidence:gate.student_confidence??null,top1_top2_margin:gate.top1_top2_margin??null,pose_quality:gate.pose_metrics?.quality??null},null,2);
}

function renderTeacher(teacher, pseudoRecord = null) {
  const status = $("#teacher-status"), root = $("#teacher-result");
  if (teacher?.status === "completed" && teacher.result) {
    status.className = "status-pill online";
    status.textContent = "分析完成";
    root.className = "teacher-result";
    const result = teacher.result;
    const label = result.label || result.suggested_label;
    const evidence = (result.evidence || []).map(item => `<li>${escapeHtml(item.description || item.type || "未提供描述")}${item.segment_id ? ` <code>${escapeHtml(item.segment_id)}</code>` : ""}</li>`).join("");
    const counterEvidence = (result.counter_evidence || []).map(item => `<li>${escapeHtml(item)}</li>`).join("");
    root.innerHTML = `<div><span>建议类别</span><strong>${escapeHtml(LABEL_NAME[label] || label || "—")}</strong></div><div><span>置信度</span><strong>${result.confidence == null ? "—" : `${(result.confidence * 100).toFixed(1)}%`}</strong></div><p>${escapeHtml(result.reason || result.reasoning_summary || "暂无判定依据")}</p>${evidence ? `<section class="teacher-evidence"><strong>教师依据</strong><ul>${evidence}</ul></section>` : ""}${counterEvidence ? `<section class="teacher-evidence counter"><strong>反向依据／限制</strong><ul>${counterEvidence}</ul></section>` : ""}`;
    return;
  }
  if (pseudoRecord?.sample_id) {
    status.className = "status-pill online";
    status.textContent = "伪标签待复核";
    root.className = "teacher-result";
    const conflicts = (pseudoRecord.conflicts || []).join("、") || "质量分处于人工复核区间";
    root.innerHTML = `<div><span>建议类别</span><strong>${escapeHtml(LABEL_NAME[pseudoRecord.label] || pseudoRecord.label || "—")}</strong></div><div><span>质量分</span><strong>${pseudoRecord.quality_score == null ? "—" : `${(pseudoRecord.quality_score * 100).toFixed(1)}%`}</strong></div><p>${escapeHtml(conflicts)}</p>`;
    return;
  }
  const running = ["queued","running"].includes(teacher?.status);
  const skipped = teacher?.status === "skipped";
  const blockedQuality = teacher?.status === "blocked_quality";
  status.className = `status-pill ${running ? "online" : "disabled"}`;
  status.textContent = running ? titleForStatus(teacher.status) : blockedQuality ? "骨架质量不足" : skipped ? "学生高置信，已跳过" : teacher?.status === "failed" ? "分析失败" : teacher?.status === "not_run" ? "尚未运行" : "尚未启用";
  root.className = "teacher-placeholder";
  root.innerHTML = `<div class="teacher-icon">◇</div><div><strong>${running ? "Qwen 多模态教师正在分析" : blockedQuality ? "骨架质量门控未调用 Qwen" : skipped ? "难例门控未调用 Qwen" : "Qwen 多模态教师"}</strong><p>${escapeHtml(teacher?.reason || "读取骨架视频，并在当前模型对应的标签空间内给出独立判断。")}</p></div>`;
}

function renderPrediction(prediction) {
  const root=$("#prediction-list"), evidenceRoot=$("#student-evidence");
  if(!prediction?.topk?.length){root.className="prediction-list empty-state";root.textContent="尚无预测结果";evidenceRoot.classList.add("hidden");evidenceRoot.innerHTML="";return;}
  root.className="prediction-list";
  root.innerHTML=prediction.topk.map((item,index)=>`<div class="prediction-row"><span class="rank">${index+1}</span><span>${escapeHtml(item.label)}</span><div class="score-track"><span style="width:${Math.max(1,item.score*100)}%"></span></div><strong>${(item.score*100).toFixed(1)}%</strong></div>`).join("");
  const studentEvidence = prediction.student_evidence;
  if(!studentEvidence?.evidence?.length){evidenceRoot.classList.add("hidden");evidenceRoot.innerHTML="";return;}
  const measured = studentEvidence.evidence.map(item => `<li>${escapeHtml(item.description || item.type || "未提供依据")}</li>`).join("");
  const limitations = (studentEvidence.limitations || []).map(item => `<li>${escapeHtml(item.description || item.code || "未提供限制")}</li>`).join("");
  evidenceRoot.classList.remove("hidden");
  evidenceRoot.innerHTML = `<section><strong>小模型决策与骨架测量依据</strong><ul>${measured}</ul></section>${limitations ? `<section class="evidence-limitations"><strong>质量限制</strong><ul>${limitations}</ul></section>` : ""}`;
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
    await api(`/api/samples/${encodeURIComponent(currentId)}/review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reviewer:reviewer(),final_label:finalLabel,reason_code:reason,note:$("#review-note").value})});
    toast(`已保存人工结果与模型快照：${LABEL_NAME[finalLabel]}`);state.reviewSample=null;await refreshAll(true);
    if(state.reviewSamples.length) await selectReview(Math.min(currentIndex,state.reviewSamples.length-1));
  } catch(error){toast(error.message,true);}
}

function renderLabelButtons() {
  $("#label-buttons").innerHTML=LABELS.map(([id,name,desc],index)=>`<button class="label-button" data-label="${id}"><span class="hotkey">${index+1}</span><strong>${name}</strong><small>${desc}</small></button>`).join("");
  $$("[data-label]").forEach(button=>button.onclick=()=>submitReview(button.dataset.label));
  $$("[data-special]").forEach(button=>button.onclick=()=>submitReview(button.dataset.special));
}

async function loadDatasetPage() {
  const params=new URLSearchParams({offset:state.datasetPage*state.datasetPageSize,limit:state.datasetPageSize});
  const status=$("#dataset-status").value,dataset=$("#dataset-source").value,query=$("#dataset-search").value.trim();
  if(status!=="all")params.set("status",status);if(dataset!=="all")params.set("dataset",dataset);if(query)params.set("query",query);
  try{
    const result=await api(`/api/samples?${params}`);state.datasetTotal=result.total;$("#dataset-total").textContent=result.total;
    $("#dataset-table").innerHTML=result.items.map(item=>`<tr><td><strong>${escapeHtml(item.sample_id)}</strong></td><td>${escapeHtml(item.source_dataset)}</td><td>${escapeHtml(LABEL_NAME[item.source_label]||item.source_label||"—")}</td><td>${escapeHtml(LABEL_NAME[item.prediction?.topk?.[0]?.label]||item.prediction?.topk?.[0]?.label||"处理中")}</td><td>${escapeHtml(LABEL_NAME[item.manual_label]||"—")}</td><td><span class="state-badge ${item.status}">${titleForStatus(item.status)}</span></td><td><button class="table-action" data-open-review="${escapeHtml(item.sample_id)}">查看 / 处置</button></td></tr>`).join("");
    const pages=Math.max(1,Math.ceil(result.total/state.datasetPageSize));$("#page-info").textContent=`第 ${state.datasetPage+1} / ${pages} 页`;$("#page-prev").disabled=state.datasetPage===0;$("#page-next").disabled=state.datasetPage+1>=pages;
    $$("[data-open-review]").forEach(button=>button.onclick=async()=>{setPage("inference");await selectInference(button.dataset.openReview);});
  }catch(error){toast(error.message,true);}
}

function renderPipeline() {
  const steps=[["数据准备","Campus6 最终清单",true],["骨架提取","RTMDet/RTMPose COCO-17",state.capabilities.pose_extraction?.enabled],["学生预测","M1FKD INT8 Campus6",state.capabilities.campus6_inference?.enabled],["难例筛选","置信度、间隔与姿态质量",true],["大模型教师","按需 Qwen3-VL-8B",state.capabilities.qwen_teacher?.enabled],["人工复核","六分类与审计",true],["批次训练","回放、蒸馏与 QAT",true],["评测发布","通过门槛后人工发布",true]];
  $("#pipeline-flow").innerHTML=steps.map(([name,desc,enabled],i)=>`<div class="flow-step ${enabled?"enabled":""}"><span class="num">0${i+1} · ${enabled?"READY":"LOCKED"}</span><strong>${name}</strong><small>${desc}</small></div>`).join("");
}

function renderEvolution() {
  const value=state.evolution||{}, enabled=Boolean(value.enabled), running=value.status==="running";
  const badge=$("#evolution-status"), button=$("#run-evolution");
  if(!badge||!button)return;
  const statusName={manual:"按批次执行",starting:"启动中",watching:"自动监控中",running:"执行中",error:"运行异常",disabled:"未启用"}[value.status]||value.status||"未知";
  badge.textContent=statusName;
  badge.className=`status-pill ${enabled&&value.status!=="error"?"online":"disabled"}`;
  button.disabled=true;
  button.title=value.reason||"训练闭环通过命令行批次执行";
  const pseudo=value.pseudo||{}, hard=value.hard_mining||{};
  $("#evolution-counts").innerHTML=[
    ["已接受伪标签",pseudo.accepted||0],
    ["待人工复核",pseudo.review||0],
    ["已拒绝",pseudo.rejected||0],
    ["难例池",hard.selected||0],
  ].map(([name,count])=>`<div class="setting-item"><span>${name}</span><code>${count}</code></div>`).join("");
  const training=value.training||{}, report=training.last_report;
  const reasonName={
    training_disabled:"训练已在配置中关闭",
    insufficient_accepted_pseudo_labels:"可训练伪标签尚未达到总量门槛",
    insufficient_new_pseudo_labels:"新增伪标签尚未达到触发门槛",
    missing_training_python:"未找到可运行 ProtoGCN 的训练 Python",
    insufficient_selected_training_gpus:"已选训练 GPU 数量不足或当前不可用",
    missing_production_checkpoint:"未找到当前线上 ProtoGCN 权重",
    training_cooldown:"仍在训练冷却期",
  };
  const lines=[
    `阶段: ${value.stage||"idle"}`,
    `最近扫描: ${formatTime(value.last_scan_at)}`,
    `当前权重: ${value.active_checkpoint||"未找到"}`,
    `可训练伪标签: ${training.trainable_count||0}（新增 ${training.new_count||0}）`,
  ];
  if((training.reasons||[]).length)lines.push(`等待条件: ${training.reasons.map(item=>reasonName[item]||item).join("；")}`);
  if(training.outcome)lines.push(`最近训练结果: ${training.outcome==="promoted"?"验证通过，已切换权重":"验证未通过，保留原权重"}`);
  if(report)lines.push(`验证: baseline Top-1=${(report.baseline?.top1_acc??0).toFixed(4)}, candidate Top-1=${(report.candidate?.top1_acc??0).toFixed(4)}, gate=${report.passed?"PASS":"REJECT"}`);
  if(value.last_error)lines.push(`错误: ${value.last_error}`);
  $("#evolution-training").textContent=lines.join("\n");
}

async function triggerEvolution() {
  toast("Campus6 闭环采用可审计的批次训练；请使用 train_campus6.py 和 experiments/campus6_gap。", true);
}

function renderModels() {
  const models=[
    ["RTMDet + RTMPose","双人 COCO-17 骨架","姿态模型",state.capabilities.pose_extraction?.enabled,"DAHUA_RTMPOSE_PYTHON"],
    ["M1FKD INT8 Campus6","4.56 MB QAT + logits/feature KD","正式推理模型",state.capabilities.campus6_inference?.enabled,state.system.student_runtime?.checkpoint],
    ["Campus6 GAP ProtoGCN","42.50 MB FP32","训练与回归基线",true,state.system.student_runtime?.training_baseline],
    ["Qwen3-VL-8B","仅在服务器空闲 GPU 足够时按需加载","语义教师",state.capabilities.qwen_teacher?.enabled,state.capabilities.qwen_teacher?.reason||"GPU 准入检查中"],
  ];
  $("#model-grid").innerHTML=models.map(([name,desc,type,enabled,config])=>`<article class="panel model-card"><div class="panel-head"><div class="model-icon">◈</div><span class="status-pill ${enabled?"online":"disabled"}">${enabled?"已配置":"未启用"}</span></div><h3>${name}</h3><p>${desc}</p><div class="model-meta"><div><span>用途</span><strong>${type}</strong></div><div><span>配置</span><strong>${config}</strong></div><div><span>状态</span><strong>${enabled?"可运行":"等待接入"}</strong></div></div></article>`).join("");
}

function renderSettings() {
  const fields=[["代码根目录",state.system.repository_root],["服务器数据根目录",state.system.data_root],["运行数据目录",state.system.runtime_root],["Campus6 数据目录",state.system.source_root],["Campus6 清单",state.system.manifest_path],["审核数据库",state.system.database_path],["推理产物目录",state.system.artifact_root],["Qwen GPU 准入",state.capabilities.qwen_teacher?.reason||"已通过"],["当前 INT8 学生权重",state.system.active_student_checkpoint]];
  $("#system-paths").innerHTML=fields.map(([name,value])=>`<div class="setting-item"><span>${name}</span><code>${escapeHtml(projectPath(value))}</code></div>`).join("");
  renderGpuSettings();
}

function renderGpuSettings() {
  const value=state.gpus||{}, gpus=value.gpus||[];
  const status=$("#gpu-status"), save=$("#save-gpu-settings");
  if(!status||!save||!$("#gpu-grid")||!$("#gpu-help"))return;
  status.textContent=value.available?`${gpus.length} 张 ProtoGCN GPU 可用`:"未检测到 GPU";
  status.className=`status-pill ${value.available?"online":"disabled"}`;
  save.disabled=!value.available;
  if(!gpus.length){
    $("#gpu-grid").innerHTML='<div class="empty-state">nvidia-smi 未返回可用 GPU。请确认容器已挂载 NVIDIA 设备与驱动。</div>';
    $("#gpu-help").textContent="GPU 配置不会静默回退到 CPU；设备不可用时推理任务会明确失败。";
    return;
  }
  const pose=new Set(value.pose_gpu_ids||[]), student=new Set(value.student_gpu_ids||[]);
  $("#gpu-grid").innerHTML=gpus.map(gpu=>{
    const ratio=gpu.memory_total_mb?Math.min(100,Math.round(gpu.memory_used_mb/gpu.memory_total_mb*100)):0;
    return `<div class="gpu-card">
      <div class="gpu-card-head"><span>GPU ${gpu.index}</span><strong>${escapeHtml(gpu.name)}</strong></div>
      <div class="gpu-live"><span>利用率 ${gpu.utilization_gpu_percent}%</span><span>${gpu.temperature_c}°C</span></div>
      <div class="gpu-memory"><span style="width:${ratio}%"></span></div>
      <small>${gpu.memory_used_mb} / ${gpu.memory_total_mb} MiB</small>
      <label><input type="checkbox" data-gpu-kind="pose" value="${gpu.index}" ${pose.has(gpu.index)?"checked":""}> RTMDet/RTMPose</label>
      <label><input type="checkbox" data-gpu-kind="student" value="${gpu.index}" ${student.has(gpu.index)?"checked":""}> M1FKD Campus6</label>
    </div>`;
  }).join("");
  $("#gpu-help").textContent=`RTMDet/RTMPose 与 M1FKD Campus6 都使用勾选的 CUDA GPU 池；每个新任务按服务内占用、实时利用率和显存占用自动选择最佳卡。Qwen3-VL-8B 仅在满足空闲显存门槛时独占一张 GPU。`;
}

async function refreshGpus() {
  try{state.gpus=await api("/api/gpus");renderGpuSettings();toast("GPU 状态已刷新");}
  catch(error){toast(error.message,true)}
}

async function saveGpuSettings() {
  const selected=kind=>$$(`input[data-gpu-kind="${kind}"]:checked`).map(input=>Number(input.value));
  try{
    state.gpus=await api("/api/gpus",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({pose_gpu_ids:selected("pose"),student_gpu_ids:selected("student")})});
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
  $("#inference-search").oninput=()=>{renderSampleOptions();setSampleTreeOpen(true)};
  document.addEventListener("click",()=>setSampleTreeOpen(false));
  if($("#refresh-gpus"))$("#refresh-gpus").onclick=refreshGpus;
  if($("#save-gpu-settings"))$("#save-gpu-settings").onclick=saveGpuSettings;
  if($("#run-evolution"))$("#run-evolution").onclick=triggerEvolution;
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
