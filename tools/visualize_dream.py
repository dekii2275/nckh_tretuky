#!/usr/bin/env python3
"""Create a self-contained, interactive DREAM record viewer (no extra packages)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (ROOT / "data_Dream/DREAMdataset/User 33"
                 / "User 33_12_intervention 2_20170623_093149.724000.json")
JOINTS = (
    "head", "sholder_center", "sholder_left", "elbow_left", "wrist_left",
    "hand_left", "sholder_right", "elbow_right", "wrist_right", "hand_right",
)
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
         (1, 6), (6, 7), (7, 8), (8, 9))


def number(value):
    """JSON null and invalid numbers stay missing rather than becoming zero."""
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return round(float(value), 4)
    return None


def quantile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * p)]


def bounds(joints, axes):
    """Fixed, robust view bounds prevent the pose from jumping between frames."""
    result = []
    for axis in axes:
        values = [point[axis] for frames in joints.values() for point in frames
                  if point[axis] is not None and point[3] is not None and point[3] > 0]
        lo, hi = quantile(values, 0.01), quantile(values, 0.99)
        margin = max((hi - lo) * 0.08, 1.0)
        result.append([lo - margin, hi + margin])
    return result


def build_payload(record, start, frames):
    skeleton = record["skeleton"]
    lengths = [len(skeleton[name]["x"]) for name in JOINTS]
    lengths += [len(record[stream][axis]) for stream in ("eye_gaze", "head_gaze")
                for axis in ("rx", "ry", "rz")]
    lengths += [len(skeleton[name][axis]) for name in JOINTS
                for axis in ("y", "z", "confidence")]
    available = min(lengths)
    if start < 0 or start >= available:
        raise ValueError(f"start-frame must be in 0..{available - 1}")
    stop = min(available, start + frames) if frames else available
    if frames is not None and frames <= 0:
        raise ValueError("frames must be positive")

    joints = {
        name: [[number(skeleton[name][axis][i]) for axis in ("x", "y", "z", "confidence")]
               for i in range(start, stop)]
        for name in JOINTS
    }
    gaze = {
        stream: [[number(record[stream][axis][i]) for axis in ("rx", "ry", "rz")]
                 for i in range(start, stop)]
        for stream in ("eye_gaze", "head_gaze")
    }
    fps = number(record.get("frame_rate"))
    if not fps or fps <= 0:
        raise ValueError("frame_rate must be positive")
    initial = next((i for i in range(stop - start)
                    if sum(1 for samples in joints.values()
                           if samples[i][3] is not None and samples[i][3] >= 0.2
                           and all(value is not None for value in samples[i][:3])) >= 5), 0)
    task = record.get("task", {})
    participant = record.get("participant", {})
    return {
        "meta": {
            "participant": participant.get("id", "?"),
            "condition": record.get("condition", "?"),
            "ability": task.get("ability") or "unknown",
            "difficulty": task.get("difficultyLevel", "?"),
            "source": record.get("$id", "record"),
            "warning": "Độ dài tín hiệu không đồng nhất; chỉ hiển thị phần căn hàng chung."
                       if len(set(lengths)) > 1 else "",
        },
        "fps": fps,
        "start": start,
        "count": stop - start,
        "initial": initial,
        "joints": joints,
        "gaze": gaze,
        "front": bounds(joints, (0, 1)),
        "side": bounds(joints, (2, 1)),
    }


HTML = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DREAM • Trình xem tín hiệu</title>
<style>
  :root { color-scheme: light; font-family: system-ui, sans-serif; background: #f3f6fb; color: #17243a; }
  body { max-width: 1360px; margin: auto; padding: 24px; }
  h1 { margin: 0 0 6px; font-size: 28px; }
  p { margin: 5px 0 16px; color: #4a5b73; }
  .meta, .controls, .cards { display: flex; gap: 12px; flex-wrap: wrap; }
  .pill, .panel { background: white; border: 1px solid #dae2ec; border-radius: 12px; }
  .pill { padding: 9px 13px; }
  .panel { padding: 16px; }
  .controls { align-items: center; margin: 16px 0; }
  button, select { font: inherit; padding: 7px 12px; border: 1px solid #a9b9cd; border-radius: 8px; background: white; }
  button { cursor: pointer; background: #233b72; color: white; }
  input[type=range] { accent-color: #3156a4; }
  #scrub { flex: 1; min-width: 250px; }
  .cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .panel h2 { font-size: 18px; margin: 0 0 10px; }
  canvas { display: block; width: 100%; height: auto; }
  .chart { margin-top: 12px; }
  .legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; color: #52647a; }
  .dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 5px; }
  .note { font-size: 13px; line-height: 1.45; }
  @media (max-width: 800px) { body { padding: 12px; } .cards { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>DREAM • Trình xem tín hiệu</h1>
<p id="source"></p>
<div class="meta" id="meta"></div>
<div class="controls">
  <button id="play" type="button">▶ Phát</button>
  <input id="scrub" type="range" min="0" value="0" aria-label="Chọn frame">
  <strong id="clock"></strong>
  <label>Ngưỡng confidence <input id="confidence" type="range" min="0" max="1" step="0.05" value="0.2"></label>
  <span id="threshold"></span>
</div>
<div class="cards">
  <section class="panel"><h2>Nhìn từ trước • X/Y</h2><canvas id="front" width="620" height="490"></canvas></section>
  <section class="panel"><h2>Nhìn từ bên • Z/Y</h2><canvas id="side" width="620" height="490"></canvas></section>
</div>
<section class="panel chart">
  <h2>Hướng nhìn theo thời gian</h2>
  <label>Thành phần <select id="axis"><option value="0">rx</option><option value="1">ry</option><option value="2">rz</option></select></label>
  <canvas id="chart" width="1240" height="240"></canvas>
  <div class="legend"><span><i class="dot" style="background:#187eaa"></i>Eye gaze</span><span><i class="dot" style="background:#dc8343"></i>Head gaze</span><span>Khớp hiện: <strong id="visible"></strong>/10</span></div>
</section>
<p class="note" id="warning"></p>
<p class="note">Trục và tỉ lệ cố định trong toàn record. Mũi tên chỉ hướng vector tại vị trí đầu, chiều dài được phóng đại để dễ xem; không xác định đối tượng đang nhìn. Khoảng trống là giá trị null hoặc khớp dưới ngưỡng confidence. Đây là tín hiệu công khai, không có video gốc hay nhãn đúng/sai cho từng lượt.</p>
<script>
const D = __PAYLOAD__;
const names = ["head", "sholder_center", "sholder_left", "elbow_left", "wrist_left", "hand_left", "sholder_right", "elbow_right", "wrist_right", "hand_right"];
const bones = [[0,1],[1,2],[2,3],[3,4],[4,5],[1,6],[6,7],[7,8],[8,9]];
const $ = id => document.getElementById(id);
let frame = D.initial, playing = false, lastTime = 0, accumulated = 0;
$("source").textContent = D.meta.source;
for (const [key, value] of [["Người tham gia", D.meta.participant], ["Điều kiện", D.meta.condition], ["Task", D.meta.ability], ["Mức", D.meta.difficulty], ["Frame", D.count], ["FPS", D.fps]]) {
  const el = document.createElement("span"); el.className = "pill"; el.textContent = `${key}: ${value}`; $("meta").appendChild(el);
}
$("warning").textContent = D.meta.warning;
$("scrub").max = D.count - 1;

function valid(p) { return p && p.slice(0,3).every(Number.isFinite) && Number.isFinite(p[3]) && p[3] > 0 && p[3] >= +$("confidence").value; }
function project(p, a, b, ranges, w, h) {
  const [xa, ya] = ranges;
  const sx = (w - 80) / (xa[1] - xa[0]), sy = (h - 80) / (ya[1] - ya[0]);
  const scale = Math.min(sx, sy);
  const cx = (xa[0] + xa[1]) / 2, cy = (ya[0] + ya[1]) / 2;
  return [w / 2 + (p[a] - cx) * scale, h / 2 - (p[b] - cy) * scale];
}
function arrow(ctx, from, dir, color, scale) {
  if (!dir || !dir.every(Number.isFinite)) return;
  const x = from[0] + dir[0] * scale, y = from[1] - dir[1] * scale;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(x,y); ctx.stroke();
  const angle = Math.atan2(y - from[1], x - from[0]);
  ctx.beginPath(); ctx.moveTo(x,y);
  ctx.lineTo(x - 11*Math.cos(angle-0.45), y - 11*Math.sin(angle-0.45));
  ctx.lineTo(x - 11*Math.cos(angle+0.45), y - 11*Math.sin(angle+0.45));
  ctx.closePath(); ctx.fill();
}
function drawPose(canvas, axes, ranges) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#fbfcfe"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#e8eef5"; ctx.lineWidth = 1;
  for (let i=1; i<5; i++) { const x=w*i/5, y=h*i/5; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
  const raw = names.map(name => D.joints[name][frame]);
  const pts = raw.map(p => valid(p) ? project(p, axes[0], axes[1], ranges, w, h) : null);
  ctx.strokeStyle = "#7755b8"; ctx.lineWidth = 5; ctx.lineCap = "round";
  for (const [a,b] of bones) if (pts[a] && pts[b]) { ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke(); }
  for (let i=0; i<pts.length; i++) if (pts[i]) {
    ctx.beginPath(); ctx.arc(...pts[i], i===0 ? 6 : 5, 0, 2*Math.PI);
    ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#56389a"; ctx.lineWidth = 2; ctx.stroke();
  }
  if (pts[0]) {
    const dirs = [[D.gaze.eye_gaze[frame], "#187eaa"], [D.gaze.head_gaze[frame], "#dc8343"]];
    for (const [vector,color] of dirs) arrow(ctx, pts[0], [vector[axes[0]], vector[axes[1]]], color, 70);
  }
  ctx.fillStyle = "#53647a"; ctx.font = "15px system-ui";
  ctx.fillText(axes[0]===0 ? "X →" : "Z →", w-54, h-16); ctx.fillText("Y ↑", 14, 25);
  return pts.filter(Boolean).length;
}
function drawChart() {
  const canvas=$("chart"), ctx=canvas.getContext("2d"), w=canvas.width, h=canvas.height;
  const left=50, right=w-18, top=15, bottom=h-30, axis=+$("axis").value;
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#fbfcfe"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#e1e9f1"; ctx.fillStyle="#61718a"; ctx.font="14px system-ui";
  for (const v of [-1,-0.5,0,0.5,1]) {
    const y=bottom-(v+1)/2*(bottom-top); ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(right,y); ctx.stroke();
    ctx.fillText(String(v), 6, y+4);
  }
  const step=Math.max(1,Math.floor(D.count/(right-left)));
  for (const [stream,color] of [["eye_gaze","#187eaa"],["head_gaze","#dc8343"]]) {
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); let connected=false;
    for (let i=0; i<D.count; i+=step) {
      const v=D.gaze[stream][i][axis];
      if (!Number.isFinite(v)) { connected=false; continue; }
      const x=left+i/Math.max(1,D.count-1)*(right-left);
      const y=bottom-(Math.max(-1,Math.min(1,v))+1)/2*(bottom-top);
      if (connected) ctx.lineTo(x,y); else ctx.moveTo(x,y);
      connected=true;
    }
    ctx.stroke();
  }
  const x=left+frame/Math.max(1,D.count-1)*(right-left);
  ctx.strokeStyle="#17243a"; ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(x,top); ctx.lineTo(x,bottom); ctx.stroke();
  ctx.fillStyle="#61718a"; ctx.fillText("Thời gian →", right-110,h-6);
}
function render() {
  $("scrub").value=frame;
  $("clock").textContent=`${((D.start+frame)/D.fps).toFixed(1)} s · frame ${D.start+frame}`;
  $("threshold").textContent=(+$("confidence").value).toFixed(2);
  $("visible").textContent=drawPose($("front"),[0,1],D.front);
  drawPose($("side"),[2,1],D.side);
  drawChart();
}
function tick(now) {
  if (!playing) return;
  if (lastTime) accumulated += now-lastTime;
  lastTime=now;
  const advance=Math.floor(accumulated*D.fps/1000);
  if (advance) {
    accumulated -= advance*1000/D.fps;
    frame=Math.min(D.count-1,frame+advance);
    render();
    if (frame===D.count-1) { playing=false; $("play").textContent="▶ Phát"; return; }
  }
  requestAnimationFrame(tick);
}
$("play").onclick=()=>{ playing=!playing; $("play").textContent=playing?"⏸ Tạm dừng":"▶ Phát"; lastTime=0; accumulated=0; if (playing) { if(frame===D.count-1) frame=0; requestAnimationFrame(tick); } };
$("scrub").oninput=e=>{ frame=+e.target.value; render(); };
$("confidence").oninput=render;
$("axis").onchange=render;
render();
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="One DREAM JSON record")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/visualizations/dream_viewer.html")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None,
                        help="Limit output to this many frames from --start-frame")
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    payload = build_payload(record, args.start_frame, args.frames)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    print(f"Wrote {args.output} ({payload['count']} frames, {payload['fps']} fps)")
    if payload["meta"]["warning"]:
        print(payload["meta"]["warning"])


if __name__ == "__main__":
    main()
