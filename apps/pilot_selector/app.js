"use strict";

const $ = (id) => document.getElementById(id);
const GROUPS = { tuning: "Chỉnh tham số", evaluation: "Đánh giá" };
const BANDS = {
  low_missing: "Ít thiếu",
  medium_missing: "Vừa",
  high_missing: "Nhiều",
};
const ACTIVITIES = {
  arm_swing_as: "Vung tay",
  body_swing_bs: "Đung đưa thân",
  chest_expansion_ce: "Mở rộng ngực",
  drumming_dr: "Gõ trống",
  maracas_forward_shaking_mfs: "Lắc maracas phía trước",
  maracas_shaking_ms: "Lắc maracas",
  sing_and_clap_sac: "Hát và vỗ tay",
  squat_sq: "Squat",
  tree_pose_tr: "Tư thế cái cây",
  twist_pose_tw: "Vặn mình",
};
let catalog,
  rows = new Map(),
  draft,
  revision,
  currentId = null,
  scope = "selected",
  page = 0;
let currentPreview = null,
  previewToken = 0,
  frame = 0,
  playing = false,
  lastTick = 0;
let mutation = 0,
  savedMutation = 0,
  saving = null,
  saveTimer = null,
  pendingProposal = null;
let saveFailure = false,
  activeView = "2d",
  toastTimer;
const PAGE_SIZE = 40;

function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function toast(text, error = false) {
  $("toast").textContent = text;
  $("toast").className = "toast" + (error ? " error" : "");
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => {
      $("toast").hidden = true;
    },
    error ? 8000 : 3500,
  );
}
async function api(path, body, options = {}) {
  const response = await fetch(
    path,
    body === undefined
      ? options
      : {
          ...options,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error || "Không thực hiện được yêu cầu.");
  return value;
}
const picked = (id) => draft.selected.find((item) => item.clip_id === id);
const excluded = (id) => draft.excluded.find((item) => item.clip_id === id);
const percent = (value) =>
  Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "Chưa đo";

function summary() {
  const groups = { tuning: 0, evaluation: 0 },
    subjects = { tuning: new Set(), evaluation: new Set() };
  const activities = { tuning: new Map(), evaluation: new Map() },
    bands = new Set();
  let reviewed = 0,
    missingReasons = 0,
    sourceGaps = 0;
  for (const item of draft.selected) {
    const row = rows.get(item.clip_id);
    groups[item.group]++;
    subjects[item.group].add(row.subject_id);
    activities[item.group].set(
      row.activity,
      (activities[item.group].get(row.activity) || 0) + 1,
    );
    bands.add(row.missingness_band);
    reviewed += Number(item.reviewed);
    missingReasons += Number(!item.reason.trim());
    sourceGaps += Number(row.source_gap_count > 0);
  }
  const overlap = [...subjects.tuning].filter((id) =>
    subjects.evaluation.has(id),
  );
  const allActivities = [...new Set(catalog.rows.map((row) => row.activity))];
  const errors = [],
    warnings = [];
  if (draft.selected.length !== draft.target)
    errors.push(
      `Chọn đủ ${draft.target} clip; hiện có ${draft.selected.length}.`,
    );
  if (overlap.length)
    errors.push("Trùng mã trẻ giữa hai nhóm: " + overlap.join(", "));
  if (!groups.tuning || !groups.evaluation)
    errors.push("Cần clip trong cả hai nhóm.");
  if (missingReasons) errors.push(`${missingReasons} clip chưa có lý do chọn.`);
  if (draft.excluded.some((item) => !item.reason.trim()))
    errors.push("Có clip bị loại chưa có lý do.");
  if (reviewed < draft.selected.length)
    errors.push(
      `${draft.selected.length - reviewed} clip đang chờ bạn xem trực quan.`,
    );
  for (const group of Object.keys(GROUPS)) {
    const missing = allActivities.filter(
      (activity) => !activities[group].has(activity),
    );
    if (missing.length)
      warnings.push(`${GROUPS[group]} chưa phủ ${missing.length} hoạt động.`);
  }
  if (bands.size < 3)
    warnings.push(
      "Chưa phủ đủ ba tầng thiếu khớp; xem lại độ đa dạng của mẫu.",
    );
  if (sourceGaps)
    warnings.push(
      `${sourceGaps} clip khuyết file frame nguồn; giữ gap khi phân tích.`,
    );
  if (!catalog.complete_population)
    warnings.push("Catalogue hiện chỉ đo một phần dataset.");
  return {
    groups,
    subjects,
    activities,
    bands,
    reviewed,
    overlap,
    errors,
    warnings,
    allActivities,
  };
}

function updateSaveStatus(text = null) {
  $("saveStatus").textContent =
    text ||
    (saving
      ? "Đang lưu trên máy…"
      : saveFailure
        ? "Chưa lưu được · xuất JSON để giữ bản nháp"
        : mutation > savedMutation
          ? "Có thay đổi chưa lưu…"
          : "Đã lưu bản nháp trên máy");
}
async function saveNow() {
  clearTimeout(saveTimer);
  if (saving) {
    await saving;
    return mutation > savedMutation ? saveNow() : undefined;
  }
  if (mutation === savedMutation) return;
  const version = mutation,
    snapshot = JSON.parse(JSON.stringify(draft));
  saving = api("/api/draft", { draft: snapshot, expected_revision: revision });
  updateSaveStatus();
  try {
    const result = await saving;
    revision = result.revision;
    savedMutation = version;
    saveFailure = false;
  } catch (error) {
    saveFailure = true;
    toast(error.message, true);
    throw error;
  } finally {
    saving = null;
    updateSaveStatus();
    renderSummary();
  }
  if (mutation > savedMutation) return saveNow();
}
function changed(renderDetail = true) {
  mutation++;
  saveFailure = false;
  updateSaveStatus();
  renderList();
  renderSummary();
  if (renderDetail) renderSelection();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveNow().catch(() => {});
  }, 550);
}

function renderSummary() {
  if (!draft) return;
  const info = summary(),
    total = draft.selected.length;
  $("selectedStat").textContent = `${total} / ${draft.target}`;
  $("targetStat").textContent =
    `${draft.excluded.length} clip được loại có lưu lý do`;
  $("groupsStat").textContent =
    `${info.groups.tuning} / ${info.groups.evaluation}`;
  $("subjectsStat").textContent =
    `${info.subjects.tuning.size} / ${info.subjects.evaluation.size} mã trẻ · ${info.overlap.length} trùng`;
  const covered = new Set([
    ...info.activities.tuning.keys(),
    ...info.activities.evaluation.keys(),
  ]);
  $("coverageStat").textContent =
    `${covered.size} / ${info.allActivities.length}`;
  $("bandsStat").textContent = `${info.bands.size}/3 tầng thiếu khớp trong mẫu`;
  $("reviewedStat").textContent = `${info.reviewed} / ${total}`;
  $("selectedProgress").style.width =
    Math.min(100, (total / draft.target) * 100) + "%";
  $("reviewedProgress").style.width =
    (total ? (info.reviewed / total) * 100 : 0) + "%";
  $("selectedCount").textContent = total;
  $("finalizeButton").disabled = Boolean(
    info.errors.length || saving || saveFailure,
  );
  $("activityCoverage").replaceChildren();
  for (const activity of info.allActivities) {
    const row = element("div", "", "coverage-line");
    row.append(
      element("span", ACTIVITIES[activity] || activity),
      element(
        "b",
        `${info.activities.tuning.get(activity) || 0} / ${info.activities.evaluation.get(activity) || 0}`,
      ),
    );
    $("activityCoverage").append(row);
  }
  $("readinessItems").replaceChildren();
  if (!info.errors.length)
    $("readinessItems").append(
      element("p", "✓ Đủ điều kiện chốt. Hai nhóm không trùng mã trẻ.", "pass"),
    );
  const list = element("ul");
  for (const text of [...info.errors, ...info.warnings])
    list.append(element("li", text));
  $("readinessItems").append(list);
  $("readinessText").textContent = info.overlap.length
    ? "Cần xử lý trùng mã trẻ trước khi chốt."
    : "Đang tách nhóm theo mã trẻ trong metadata; chưa xác minh danh tính từ skeleton.";
}

function filteredRows() {
  const query = $("search").value.trim().toLowerCase();
  let result = catalog.rows.filter((row) => {
    const item = picked(row.clip_id),
      removed = excluded(row.clip_id);
    if (
      (scope === "selected" && !item) ||
      (scope === "unreviewed" && (!item || item.reviewed)) ||
      (scope === "excluded" && !removed)
    )
      return false;
    if (
      query &&
      ![row.clip_id, row.activity, ACTIVITIES[row.activity], row.subject_id]
        .join(" ")
        .toLowerCase()
        .includes(query)
    )
      return false;
    for (const [id, key] of [
      ["activityFilter", "activity"],
      ["subjectFilter", "subject_id"],
      ["sessionFilter", "session_id"],
      ["bandFilter", "missingness_band"],
    ]) {
      if ($(id).value && row[key] !== $(id).value) return false;
    }
    return !$("groupFilter").value || item?.group === $("groupFilter").value;
  });
  const sort = $("sort").value;
  result.sort((a, b) => {
    if (sort === "missing_desc")
      return (
        (b.missing_joint_ratio ?? -1) - (a.missing_joint_ratio ?? -1) ||
        a.clip_id.localeCompare(b.clip_id)
      );
    if (sort === "missing_asc")
      return (
        (a.missing_joint_ratio ?? 2) - (b.missing_joint_ratio ?? 2) ||
        a.clip_id.localeCompare(b.clip_id)
      );
    if (sort === "gap")
      return (
        (b.max_missing_run ?? -1) - (a.max_missing_run ?? -1) ||
        a.clip_id.localeCompare(b.clip_id)
      );
    return (
      a.activity.localeCompare(b.activity) || a.clip_id.localeCompare(b.clip_id)
    );
  });
  return result;
}

function toggleClip(id, selected) {
  const row = rows.get(id);
  if (row.status !== "ready") return;
  if (selected && !picked(id)) {
    const sameSubject = draft.selected.find(
      (item) => rows.get(item.clip_id).subject_id === row.subject_id,
    );
    const counts = summary().groups;
    const group =
      sameSubject?.group ||
      (counts.tuning <= counts.evaluation ? "tuning" : "evaluation");
    draft.excluded = draft.excluded.filter((item) => item.clip_id !== id);
    draft.selected.push({
      clip_id: id,
      group,
      reviewed: false,
      reason: `Chọn thủ công: ${ACTIVITIES[row.activity] || row.activity}, ${BANDS[row.missingness_band].toLowerCase()}; cần xem và bổ sung lý do nếu cần.`,
    });
  } else if (!selected)
    draft.selected = draft.selected.filter((item) => item.clip_id !== id);
  changed();
}
function changeGroup(id, group) {
  const subject = rows.get(id).subject_id;
  const affected = draft.selected.filter(
    (item) => rows.get(item.clip_id).subject_id === subject,
  );
  for (const item of affected) item.group = group;
  if (affected.length > 1)
    toast(
      `Đã chuyển ${affected.length} clip của mã trẻ ${subject} sang nhóm ${GROUPS[group].toLowerCase()} để giữ tách nhóm.`,
    );
  changed();
}

function renderList() {
  if (!catalog) return;
  const result = filteredRows(),
    pages = Math.max(1, Math.ceil(result.length / PAGE_SIZE));
  page = Math.max(0, Math.min(page, pages - 1));
  $("listCount").textContent =
    `${result.length} clip phù hợp · ${catalog.rows.length} ứng viên`;
  $("pageInfo").textContent = `Trang ${page + 1}/${pages}`;
  $("prevPage").disabled = page === 0;
  $("nextPage").disabled = page >= pages - 1;
  $("emptyList").hidden = result.length > 0;
  const fragment = document.createDocumentFragment();
  for (const row of result.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)) {
    const item = picked(row.clip_id),
      removed = excluded(row.clip_id);
    const tr = element("tr");
    tr.dataset.clipId = row.clip_id;
    if (row.clip_id === currentId) tr.classList.add("current");
    tr.addEventListener("click", () => selectClip(row.clip_id));
    const checkCell = element("td"),
      checkbox = element("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(item);
    checkbox.disabled = row.status !== "ready";
    checkbox.setAttribute("aria-label", `Chọn ${row.clip_id}`);
    checkbox.onclick = (event) => event.stopPropagation();
    checkbox.onchange = () => toggleClip(row.clip_id, checkbox.checked);
    checkCell.append(checkbox);
    const name = element("td"),
      title = element("span", row.clip_id, "clip-name");
    title.title = row.clip_id;
    if (item?.reviewed) title.append(element("span", "✓", "review-dot"));
    name.append(
      title,
      element(
        "span",
        ACTIVITIES[row.activity] || row.activity,
        "activity-name",
      ),
    );
    if (removed) name.append(element("span", "Đã loại", "excluded-note"));
    const subject = element("td", row.subject_id);
    subject.append(element("span", row.session_id, "activity-name"));
    const quality = element("td"),
      band = element(
        "span",
        row.status === "ready" ? percent(row.missing_joint_ratio) : "Cách ly",
        `band ${row.missingness_band || "quarantined"}`,
      );
    band.title =
      row.status === "ready"
        ? `${BANDS[row.missingness_band]} · khoảng thiếu dài nhất ${row.max_missing_run} frame`
        : row.error;
    quality.append(band);
    const groupCell = element("td");
    if (item) {
      const group = element("select", "", "group-select");
      group.setAttribute("aria-label", `Nhóm ${row.clip_id}`);
      for (const [value, label] of Object.entries(GROUPS)) {
        const option = element("option", label);
        option.value = value;
        group.append(option);
      }
      group.value = item.group;
      group.onclick = (event) => event.stopPropagation();
      group.onchange = () => changeGroup(row.clip_id, group.value);
      groupCell.append(group);
    } else groupCell.append(element("span", "—", "muted"));
    tr.append(checkCell, name, subject, quality, groupCell);
    fragment.append(tr);
  }
  $("candidates").replaceChildren(fragment);
}

function renderSelection() {
  if (!currentId) return;
  const item = picked(currentId),
    removed = excluded(currentId),
    row = rows.get(currentId);
  $("chooseClip").checked = Boolean(item);
  $("chooseClip").disabled = row.status !== "ready";
  $("detailGroup").disabled = !item;
  $("detailGroup").value = item?.group || "tuning";
  $("reason").disabled = !item;
  if (document.activeElement !== $("reason"))
    $("reason").value = item?.reason || removed?.reason || "";
  $("reviewed").disabled = !item || currentPreview?.clip_id !== currentId;
  $("reviewed").checked = Boolean(item?.reviewed);
  $("excludeButton").textContent = removed ? "Bỏ loại" : "Loại clip";
}

function pause() {
  playing = false;
  $("playButton").textContent = "▶ Phát";
}
async function selectClip(id) {
  currentId = id;
  currentPreview = null;
  frame = 0;
  pause();
  const token = ++previewToken,
    row = rows.get(id);
  $("clipTitle").textContent = id;
  $("clipMeta").textContent =
    `${ACTIVITIES[row.activity] || row.activity} · mã trẻ ${row.subject_id} · ${row.session_id} · ${row.num_frames} frame`;
  $("previewStatus").className = "preview-status";
  $("previewStatus").textContent =
    row.status === "ready" ? "Đang đọc skeleton với Tracklet V2…" : row.error;
  $("previewStatus").hidden = false;
  $("playButton").disabled = true;
  $("timeline").disabled = true;
  $("eventSelect").replaceChildren(element("option", "Đi tới chỉ báo…"));
  $("jointTable").replaceChildren();
  $("detailStats").replaceChildren();
  for (const [label, value] of [
    ["Thiếu khớp", percent(row.missing_joint_ratio)],
    [
      "Khoảng thiếu dài nhất",
      Number.isFinite(row.max_missing_run)
        ? row.max_missing_run + " frame"
        : "—",
    ],
    [
      "Confidence TB",
      Number.isFinite(row.mean_confidence)
        ? row.mean_confidence.toFixed(2)
        : "—",
    ],
    ["Số người tối đa", row.max_people ?? "—"],
    ["Khuyết frame nguồn", row.source_gap_count ?? "—"],
    ["Bước tọa độ lớn", row.large_step_joint_events ?? "—"],
  ]) {
    const card = element("div");
    card.append(element("span", label), element("strong", String(value)));
    $("detailStats").append(card);
  }
  renderList();
  renderSelection();
  draw();
  if (activeView === "3d")
    $("viewer3d").src = "/viewer/" + encodeURIComponent(id);
  if (row.status !== "ready") return;
  try {
    const payload = await api("/api/preview/" + encodeURIComponent(id));
    if (token !== previewToken) return;
    currentPreview = payload;
    const bounds = [Infinity, -Infinity, Infinity, -Infinity];
    for (const candidates of payload.frames)
      for (const person of candidates)
        for (const point of person)
          if (validPoint(point)) {
            bounds[0] = Math.min(bounds[0], point[0]);
            bounds[1] = Math.max(bounds[1], point[0]);
            bounds[2] = Math.min(bounds[2], point[1]);
            bounds[3] = Math.max(bounds[3], point[1]);
          }
    payload.bounds = Number.isFinite(bounds[0]) ? bounds : [0, 640, 0, 480];
    $("timeline").max = payload.frames.length - 1;
    $("timeline").value = 0;
    $("timeline").disabled = false;
    $("playButton").disabled = false;
    const names = {
      large_step: "Bước tọa độ lớn",
      target_missing: "Track vắng",
      source_gap: "Khuyết frame nguồn",
    };
    for (const event of payload.events) {
      const option = element(
        "option",
        `${names[event.kind]} · frame ${payload.source_frame_ids[event.frame]}`,
      );
      option.value = event.frame;
      $("eventSelect").append(option);
    }
    renderSelection();
    draw();
  } catch (error) {
    if (token !== previewToken) return;
    $("previewStatus").className = "preview-status error";
    $("previewStatus").textContent = error.message;
    toast(error.message, true);
  }
}

function validPoint(point) {
  return (
    point &&
    point.length >= 3 &&
    point.every(Number.isFinite) &&
    point[2] >= (catalog?.confidence_threshold ?? 0.1) &&
    (point[0] !== 0 || point[1] !== 0)
  );
}
function draw() {
  const canvas = $("skeleton"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f7f9fd";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e9eef6";
  ctx.lineWidth = 1;
  for (let x = 30; x < canvas.width; x += 45) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 30; y < canvas.height; y += 45) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  if (!currentPreview) return;
  const payload = currentPreview,
    selected = payload.selected_indices[frame];
  const [minX, maxX, minY, maxY] = payload.bounds;
  const scale = Math.min(
    (canvas.width - 90) / Math.max(maxX - minX, 1),
    (canvas.height - 110) / Math.max(maxY - minY, 1),
  );
  const X = (x) => canvas.width / 2 + (x - (minX + maxX) / 2) * scale;
  const Y = (y) => canvas.height / 2 + (y - (minY + maxY) / 2) * scale;
  for (const [index, person] of payload.frames[frame].entries()) {
    const chosen = index === selected;
    ctx.strokeStyle = chosen ? "#de8131" : "#abb7c8";
    ctx.fillStyle = chosen ? "#e88f44" : "#aebacb";
    ctx.lineWidth = chosen ? 4 : 2;
    for (const [a, b] of payload.edges)
      if (validPoint(person[a]) && validPoint(person[b])) {
        ctx.beginPath();
        ctx.moveTo(X(person[a][0]), Y(person[a][1]));
        ctx.lineTo(X(person[b][0]), Y(person[b][1]));
        ctx.stroke();
      }
    for (const point of person)
      if (validPoint(point)) {
        ctx.beginPath();
        ctx.arc(X(point[0]), Y(point[1]), chosen ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();
      }
  }
  $("frameInfo").textContent =
    `Nguồn ${payload.source_frame_ids[frame]} · ${frame + 1}/${payload.frames.length}`;
  $("timeline").value = frame;
  $("previewStatus").hidden = selected >= 0;
  $("previewStatus").textContent =
    "Track được chọn vắng ở frame này; các ứng viên khác vẫn được hiển thị.";
  $("jointTable").replaceChildren();
  if (selected >= 0)
    for (const [index, point] of payload.frames[frame][selected].entries()) {
      const tr = element("tr");
      tr.append(element("td", payload.joint_names[index]));
      for (const value of point)
        tr.append(
          element("td", Number.isFinite(value) ? value.toFixed(2) : "Thiếu"),
        );
      if (!validPoint(point)) tr.classList.add("muted");
      $("jointTable").append(tr);
    }
}

function tick(time) {
  if (
    playing &&
    currentPreview &&
    time - lastTick >= 1000 / Number($("speed").value)
  ) {
    lastTick = time;
    frame = (frame + 1) % currentPreview.frames.length;
    draw();
  }
  requestAnimationFrame(tick);
}
function setView(view) {
  activeView = view;
  $("canvasWrap").hidden = view !== "2d";
  $("controls").hidden = view !== "2d";
  $("viewer3d").hidden = view !== "3d";
  $("tab2d").classList.toggle("active", view === "2d");
  $("tab3d").classList.toggle("active", view === "3d");
  if (view === "3d" && currentId) {
    pause();
    $("viewer3d").src = "/viewer/" + encodeURIComponent(currentId);
  }
}
function nextClip() {
  const list = filteredRows();
  if (!list.length) return;
  const index =
    (list.findIndex((row) => row.clip_id === currentId) + 1) % list.length;
  page = Math.floor(index / PAGE_SIZE);
  selectClip(list[index].clip_id);
}
function listMessages(container, messages) {
  if (!messages.length) return;
  const list = element("ul");
  for (const message of messages) list.append(element("li", message));
  container.append(list);
}

function bindEvents() {
  for (const id of [
    "search",
    "activityFilter",
    "subjectFilter",
    "sessionFilter",
    "bandFilter",
    "groupFilter",
    "sort",
  ])
    $(id).addEventListener(id === "search" ? "input" : "change", () => {
      page = 0;
      renderList();
    });
  document.querySelectorAll("[data-scope]").forEach((button) => {
    button.onclick = () => {
      scope = button.dataset.scope;
      page = 0;
      document
        .querySelectorAll("[data-scope]")
        .forEach((item) => item.classList.toggle("active", item === button));
      renderList();
    };
  });
  document.querySelectorAll("[data-close]").forEach((button) => {
    button.onclick = () => $(button.dataset.close).close();
  });
  $("prevPage").onclick = () => {
    page--;
    renderList();
  };
  $("nextPage").onclick = () => {
    page++;
    renderList();
  };
  $("resetFilters").onclick = () => {
    for (const id of [
      "search",
      "activityFilter",
      "subjectFilter",
      "sessionFilter",
      "bandFilter",
      "groupFilter",
    ])
      $(id).value = "";
    page = 0;
    renderList();
  };
  $("helpButton").onclick = () => $("helpDialog").showModal();
  $("chooseClip").onchange = () =>
    toggleClip(currentId, $("chooseClip").checked);
  $("detailGroup").onchange = () =>
    changeGroup(currentId, $("detailGroup").value);
  $("reviewed").onchange = () => {
    const item = picked(currentId);
    if (item) {
      item.reviewed = $("reviewed").checked;
      changed();
    }
  };
  $("reason").oninput = () => {
    const item = picked(currentId);
    if (item) {
      item.reason = $("reason").value;
      changed(false);
    }
  };
  $("excludeButton").onclick = () => {
    if (!currentId) return;
    if (excluded(currentId)) {
      draft.excluded = draft.excluded.filter(
        (item) => item.clip_id !== currentId,
      );
      changed();
      return;
    }
    $("excludeClipName").textContent = currentId;
    $("excludeReason").value = "";
    $("excludeDialog").showModal();
  };
  $("confirmExclude").onclick = () => {
    const reason = $("excludeReason").value.trim();
    if (!reason) {
      toast("Hãy ghi lý do loại clip.", true);
      return;
    }
    draft.selected = draft.selected.filter(
      (item) => item.clip_id !== currentId,
    );
    draft.excluded = draft.excluded.filter(
      (item) => item.clip_id !== currentId,
    );
    draft.excluded.push({ clip_id: currentId, reason });
    $("excludeDialog").close();
    changed();
  };
  for (const id of ["target", "seed"])
    $(id).onchange = () => {
      const value = Number($(id).value);
      if (
        !Number.isInteger(value) ||
        (id === "target" && (value < 20 || value > 50))
      ) {
        $(id).value = draft[id];
        toast("Số clip từ 20–50; seed là số nguyên.", true);
        return;
      }
      draft[id] = value;
      changed();
    };
  $("proposeButton").onclick = async () => {
    $("proposeButton").disabled = true;
    try {
      pendingProposal = await api("/api/propose", {
        target: draft.target,
        seed: draft.seed,
        excluded: draft.excluded,
      });
      const info = pendingProposal.summary;
      $("proposalSummary").replaceChildren(
        element(
          "p",
          `${info.selected}/${info.target} clip · ${info.groups.tuning} chỉnh tham số / ${info.groups.evaluation} đánh giá · ${info.subject_overlap.length} mã trẻ trùng.`,
        ),
      );
      listMessages($("proposalSummary"), info.warnings);
      $("proposalDialog").showModal();
    } catch (error) {
      toast(error.message, true);
    } finally {
      $("proposeButton").disabled = false;
    }
  };
  $("applyProposal").onclick = () => {
    draft = pendingProposal.draft;
    $("proposalDialog").close();
    scope = "selected";
    page = 0;
    document
      .querySelectorAll("[data-scope]")
      .forEach((button) =>
        button.classList.toggle("active", button.dataset.scope === scope),
      );
    changed();
    if (draft.selected.length) selectClip(draft.selected[0].clip_id);
  };
  $("nextClip").onclick = nextClip;
  $("playButton").onclick = () => {
    playing = !playing;
    $("playButton").textContent = playing ? "Ⅱ Dừng" : "▶ Phát";
    lastTick = performance.now();
  };
  $("timeline").oninput = () => {
    pause();
    frame = Number($("timeline").value);
    draw();
  };
  $("eventSelect").onchange = () => {
    if ($("eventSelect").value !== "") {
      pause();
      frame = Number($("eventSelect").value);
      draw();
    }
  };
  $("tab2d").onclick = () => setView("2d");
  $("tab3d").onclick = () => setView("3d");
  document.addEventListener("keydown", (event) => {
    if (
      !currentPreview ||
      ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(
        event.target.tagName,
      ) ||
      document.querySelector("dialog[open]")
    )
      return;
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      pause();
      frame = Math.max(
        0,
        Math.min(
          currentPreview.frames.length - 1,
          frame + (event.key === "ArrowRight" ? 1 : -1),
        ),
      );
      draw();
    }
    if (event.code === "Space") {
      event.preventDefault();
      $("playButton").click();
    }
  });
  $("downloadDraft").onclick = () => {
    const link = element("a");
    link.href = URL.createObjectURL(
      new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }),
    );
    link.download = "pilot_draft.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  $("importButton").onclick = () => $("importFile").click();
  $("importFile").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const value = JSON.parse(await file.text());
      if (
        !window.confirm(
          "Nhập bản nháp này sẽ thay lựa chọn hiện tại. Tiếp tục?",
        )
      )
        return;
      await saveNow();
      const result = await api("/api/draft", {
        draft: value,
        expected_revision: revision,
      });
      draft = result.draft;
      revision = result.revision;
      mutation++;
      savedMutation = mutation;
      $("target").value = draft.target;
      $("seed").value = draft.seed;
      renderList();
      renderSummary();
      renderSelection();
      updateSaveStatus();
      toast("Đã nhập và lưu bản nháp.");
    } catch (error) {
      toast(error.message, true);
    } finally {
      event.target.value = "";
    }
  };
  $("finalizeButton").onclick = async () => {
    try {
      await saveNow();
      const info = summary();
      if (info.errors.length) throw new Error(info.errors.join(" "));
      $("finalSummary").replaceChildren(
        element(
          "p",
          `${draft.selected.length} clip · ${info.groups.tuning} chỉnh tham số / ${info.groups.evaluation} đánh giá · ${info.reviewed} đã xem · không trùng mã trẻ.`,
        ),
      );
      listMessages($("finalSummary"), info.warnings);
      $("downloadLinks").replaceChildren();
      $("confirmFinalize").disabled = false;
      $("confirmFinalize").textContent = "Xác nhận chốt & tạo file";
      $("finalDialog").showModal();
    } catch (error) {
      toast(error.message, true);
    }
  };
  $("confirmFinalize").onclick = async () => {
    $("confirmFinalize").disabled = true;
    $("confirmFinalize").textContent = "Đang kiểm tra lại dữ liệu nguồn…";
    try {
      const result = await api("/api/finalize", {
        expected_revision: revision,
      });
      for (const [name, url] of Object.entries(result.files)) {
        const link = element("a", "↓ " + name);
        link.href = url;
        link.download = name;
        $("downloadLinks").append(link);
      }
      $("confirmFinalize").textContent = "✓ Đã lưu phiên bản xuất";
      toast("Đã chốt và lưu manifest cùng hồ sơ lựa chọn trên máy.");
    } catch (error) {
      $("confirmFinalize").disabled = false;
      $("confirmFinalize").textContent = "Thử chốt lại";
      toast(error.message, true);
    }
  };
  window.addEventListener("beforeunload", (event) => {
    if (mutation > savedMutation) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
}

async function start() {
  try {
    const [data, state] = await Promise.all([
      api("/api/catalog"),
      api("/api/state"),
    ]);
    catalog = data;
    rows = new Map(data.rows.map((row) => [row.clip_id, row]));
    draft = state.draft;
    revision = state.revision;
    $("target").value = draft.target;
    $("seed").value = draft.seed;
    const subjectCount = new Set(data.rows.map((row) => row.subject_id)).size;
    $("catalogInfo").textContent =
      `${data.measured_size.toLocaleString("vi-VN")} clip · ${subjectCount} mã trẻ · Tracklet V2`;
    $("catalogVersion").textContent =
      `Catalogue ${data.catalog_sha256.slice(0, 10)} · chỉ số trước tiền xử lý`;
    for (const [id, key] of [
      ["activityFilter", "activity"],
      ["subjectFilter", "subject_id"],
      ["sessionFilter", "session_id"],
    ])
      for (const value of [
        ...new Set(data.rows.map((row) => row[key])),
      ].sort()) {
        const option = element(
          "option",
          key === "activity" ? ACTIVITIES[value] || value : value,
        );
        option.value = value;
        $(id).append(option);
      }
    bindEvents();
    renderList();
    renderSummary();
    updateSaveStatus();
    requestAnimationFrame(tick);
    if (draft.selected.length) selectClip(draft.selected[0].clip_id);
  } catch (error) {
    $("fatal").textContent = "Không mở được công cụ: " + error.message;
    $("fatal").hidden = false;
  }
}
start();
