#!/usr/bin/env python3
"""Export multimodal (2D OpenPose + 3D ROMP) interactive review pages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.openpose_parser import parse_clip
from src.data.child_tracking import select_child_track
from src.utils import ensure_dir, load_config

SMPL_24_BONES = [
    [0, 1], [0, 2], [0, 3],
    [1, 4], [4, 7], [7, 10],
    [2, 5], [5, 8], [8, 11],
    [3, 6], [6, 9], [9, 12], [12, 15],
    [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
    [9, 14], [14, 17], [17, 19], [19, 21], [21, 23],
]

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

OPENPOSE_25_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [0, 15], [15, 17], [0, 16], [16, 18], [14, 19], [19, 20],
    [14, 21], [11, 22], [22, 23], [11, 24]
]


def load_romp_3d_clip(romp_dir: Path) -> list[list[list[float]]]:
    """Load per-frame ROMP 3D coordinates from npz files."""
    files = sorted(romp_dir.glob("*.npz"))
    frames_3d = []
    for f in files:
        try:
            data = np.load(f)
            coords = data["coordinates"]  # [N_persons, 24, 3]
            frame_persons = []
            for person in coords:
                valid = np.isfinite(person).all() and (person != 0).any()
                if valid:
                    frame_persons.append(np.round(person.astype(float), 2).tolist())
            frames_3d.append(frame_persons)
        except Exception:
            frames_3d.append([])
    return frames_3d


def generate_multimodal_html(payload: dict) -> str:
    """Generate self-contained HTML page with synchronized 2D OpenPose and 3D Skeleton Viewer."""
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kiểm toán 2D & 3D: {payload['clip_id']} ({payload['activity']})</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #f97316;
      --border: #334155;
      --success: #22c55e;
      --warning: #eab308;
      --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 16px;
      line-height: 1.5;
    }}
    a {{ color: var(--primary); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: #0284c7;
      color: white;
    }}
    .badge.orange {{ background: var(--accent); }}
    .grid-view {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    @media (max-width: 1024px) {{
      .grid-view {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      position: relative;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 14px;
      color: var(--text-muted);
      font-weight: 500;
    }}
    canvas {{
      width: 100%;
      height: 480px;
      background: #090d16;
      border-radius: 6px;
      border: 1px solid #1e293b;
      cursor: crosshair;
      display: block;
    }}
    #canvas3d {{ cursor: grab; }}
    #canvas3d:active {{ cursor: grabbing; }}
    .controls-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .playback-bar {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    button {{
      background: #334155;
      color: white;
      border: 1px solid #475569;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.15s;
    }}
    button:hover {{ background: #475569; }}
    button.btn-primary {{ background: #0284c7; border-color: #0369a1; }}
    button.btn-primary:hover {{ background: #0369a1; }}
    button.btn-success {{ background: #16a34a; border-color: #15803d; }}
    button.btn-success:hover {{ background: #15803d; }}
    .slider-track {{
      flex: 1;
      min-width: 200px;
      accent-color: var(--primary);
    }}
    .review-section {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }}
    @media (max-width: 900px) {{
      .review-section {{ grid-template-columns: 1fr; }}
    }}
    .form-group {{
      margin-bottom: 12px;
    }}
    .form-group label {{
      display: block;
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 4px;
      font-weight: 500;
    }}
    select, input[type="text"], textarea {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      color: white;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 13px;
      font-family: inherit;
    }}
    select:focus, input[type="text"]:focus, textarea:focus {{
      outline: none;
      border-color: var(--primary);
    }}
    .flags-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 8px;
      margin-top: 6px;
    }}
    .flag-item {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      background: #0f172a;
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid #1e293b;
    }}
    .flag-item input {{ accent-color: var(--primary); }}
    .legend-box {{
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .legend-dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 4px;
    }}
    .info-footer {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <a href="index.html">← Quay lại danh mục kiểm toán</a>
      <h2 style="margin-top: 4px;">{payload['clip_id']}</h2>
    </div>
    <div style="text-align: right;">
      <span class="badge">{payload['activity']}</span>
      <span class="badge orange">Subject {payload.get('subject_id', 'N/A')}</span>
      <span class="badge">Session {payload.get('session_id', 'N/A')}</span>
    </div>
  </div>

  <!-- Dual 2D & 3D Display -->
  <div class="grid-view">
    <!-- 2D OpenPose Viewer -->
    <div class="card">
      <div class="card-header">
        <span>Khung xương 2D OpenPose (BODY_25)</span>
        <div class="legend-box">
          <span><span class="legend-dot" style="background: #ea580c;"></span>Người được tracker chọn</span>
          <span><span class="legend-dot" style="background: #38bdf8;"></span>Ứng viên khác</span>
        </div>
      </div>
      <canvas id="canvas2d" width="600" height="480"></canvas>
    </div>

    <!-- 3D ROMP Viewer -->
    <div class="card">
      <div class="card-header">
        <span>Mô hình người 3D (ROMP 24 khớp) — Kéo chuột để xoay 360°, cuộn để phóng to/thu nhỏ</span>
        <div class="legend-box">
          <button id="btnViewFront" style="padding: 2px 8px; font-size: 11px;">Mặt trước</button>
          <button id="btnViewSide" style="padding: 2px 8px; font-size: 11px;">Mặt bên</button>
          <button id="btnViewTop" style="padding: 2px 8px; font-size: 11px;">Từ trên</button>
          <button id="btnViewReset" style="padding: 2px 8px; font-size: 11px;">Mặc định</button>
        </div>
      </div>
      <canvas id="canvas3d" width="600" height="480"></canvas>
    </div>
  </div>

  <!-- Playback & Frame scrubbing -->
  <div class="controls-panel">
    <div class="playback-bar">
      <button id="btnPlay" class="btn-primary">▶ Phát</button>
      <button id="btnPrev">⏮ -1 frame</button>
      <button id="btnNext">+1 frame ⏭</button>
      <input type="range" id="frameSlider" class="slider-track" min="0" max="{payload['num_frames'] - 1}" value="0">
      <span id="frameNumber" style="font-weight: 600; font-size: 13px; min-width: 110px;">Frame: 0 / {payload['num_frames'] - 1}</span>
      <label style="font-size: 12px; color: var(--text-muted); display: flex; align-items: center; gap: 4px;">
        Tốc độ:
        <select id="playbackSpeed" style="width: auto; padding: 3px 6px;">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1.0x</option>
          <option value="2">2.0x</option>
        </select>
      </label>
    </div>
    <div id="frameDetails" style="font-size: 12px; color: var(--text-muted); font-family: monospace;"></div>
  </div>

  <!-- Review and Annotation Form -->
  <div class="card">
    <h3 style="margin-bottom: 12px; font-size: 16px;">Biểu mẫu Kiểm toán & Gán nhãn Đa phương thức (2D + 3D)</h3>
    <div class="review-section">
      <!-- Cột 1: Đánh giá Tracking & Action -->
      <div>
        <div class="form-group">
          <label>1. Đánh giá chất lượng bám vết người (Tracking):</label>
          <select id="review_tracking">
            <option value="">-- Chưa đánh giá --</option>
            <option value="A">A — Rõ ràng theo dõi đúng trẻ (Correct Child)</option>
            <option value="B">B — Theo dõi nhầm chuyên gia / người lớn (Wrong Person)</option>
            <option value="C">C — Ban đầu đúng, sau đó bị nhảy sang người khác (Identity Switch)</option>
            <option value="D">D — Không chắc chắn / Không thể xác định (Uncertain)</option>
          </select>
        </div>

        <div class="form-group">
          <label>2. Đánh giá tư thế hành động trên mô hình 3D ({payload['activity']}):</label>
          <select id="review_action">
            <option value="">-- Chưa đánh giá --</option>
            <option value="valid_action">Đúng hành động: Khung xương 3D thể hiện rõ động tác {payload['activity']}</option>
            <option value="wrong_action">Sai hành động: Khung xương 3D thực hiện động tác khác (hoặc đứng yên)</option>
            <option value="incomplete_action">Hành động chưa hoàn thành / Không rõ ràng</option>
          </select>
        </div>

        <div class="form-group">
          <label>Ghi chú quan sát chi tiết:</label>
          <textarea id="review_notes" rows="3" placeholder="Nhập nhận xét về tư thế, góc nhìn, hoặc lý do nhận diện nhầm..."></textarea>
        </div>
      </div>

      <!-- Cột 2: Các cờ hiện tượng bất thường -->
      <div>
        <label style="font-size: 13px; color: var(--text-muted); font-weight: 500;">3. Các hiện tượng quan sát được trong clip:</label>
        <div class="flags-grid">
          <label class="flag-item"><input type="checkbox" id="flag_multiple_people"> Có nhiều người trong phòng</label>
          <label class="flag-item"><input type="checkbox" id="flag_occlusion"> Che khuất một phần</label>
          <label class="flag-item"><input type="checkbox" id="flag_identity_switch"> Nhìn thấy đổi người rõ rệt</label>
          <label class="flag-item"><input type="checkbox" id="flag_child_leaves"> Trẻ rời khung hình</label>
          <label class="flag-item"><input type="checkbox" id="flag_cross_path"> Chuyên gia đi ngang qua trẻ</label>
          <label class="flag-item"><input type="checkbox" id="flag_pose_failure"> Lỗi vỡ/méo khớp 3D</label>
          <label class="flag-item"><input type="checkbox" id="flag_camera_motion"> Camera rung giật mạnh</label>
        </div>

        <div style="margin-top: 16px; display: flex; gap: 10px; align-items: center;">
          <button id="btnSave" class="btn-success">💾 Lưu đánh giá vào máy</button>
          <button id="btnExportCSV">📥 Xuất file CSV nhãn clip này</button>
          <span id="saveStatus" style="font-size: 12px; color: var(--success);"></span>
        </div>
      </div>
    </div>

    <div class="info-footer">
      <span>Clip: {payload['clip_id']} | Số frame: {payload['num_frames']} | Số ứng viên 3D tối đa: {max((len(f) for f in payload['frames_3d']), default=0)}</span>
      <span>Dữ liệu tự chứa (offline 100%)</span>
    </div>
  </div>

  <script id="dataPayload" type="application/json">{data_json}</script>

  <script>
    const D = JSON.parse(document.getElementById('dataPayload').textContent);
    const canvas2d = document.getElementById('canvas2d');
    const ctx2d = canvas2d.getContext('2d');
    const canvas3d = document.getElementById('canvas3d');
    const ctx3d = canvas3d.getContext('2d');
    const slider = document.getElementById('frameSlider');
    const frameNumber = document.getElementById('frameNumber');
    const frameDetails = document.getElementById('frameDetails');

    // 2D Skeleton Edges (BODY_25)
    const EDGES_2D = {json.dumps(OPENPOSE_25_EDGES)};
    const PALETTE_2D = ['#38bdf8', '#34d399', '#a855f7', '#fbbf24', '#f43f5e'];

    // 3D Skeleton Bones (SMPL 24)
    const BONES_3D = {json.dumps(SMPL_24_BONES)};
    const PALETTE_3D = ['#38bdf8', '#34d399', '#f97316', '#a855f7', '#fbbf24'];

    // 3D Orbit Camera State
    let camYaw = 0.4;     // Azimuth (rad)
    let camPitch = 0.2;   // Elevation (rad)
    let camDist = 600;    // Distance from target
    let camPanX = 0;
    let camPanY = 0;
    let isDragging = false;
    let dragStartX = 0, dragStartY = 0;

    // Bounds calculation for 2D
    let xmin2d = Infinity, xmax2d = -Infinity, ymin2d = Infinity, ymax2d = -Infinity;
    D.frames_2d.forEach(people => {{
      people.forEach(person => {{
        person.forEach(p => {{
          if (p[2] >= 0.05 && (p[0] !== 0 || p[1] !== 0)) {{
            xmin2d = Math.min(xmin2d, p[0]); xmax2d = Math.max(xmax2d, p[0]);
            ymin2d = Math.min(ymin2d, p[1]); ymax2d = Math.max(ymax2d, p[1]);
          }}
        }});
      }});
    }});
    if (!isFinite(xmin2d)) {{ xmin2d = 0; xmax2d = 1000; ymin2d = 0; ymax2d = 800; }}
    const margin2d = 40;
    const scale2d = Math.min((canvas2d.width - margin2d * 2) / Math.max(xmax2d - xmin2d, 1),
                             (canvas2d.height - margin2d * 2) / Math.max(ymax2d - ymin2d, 1));
    const toScreenX2d = x => margin2d + (x - xmin2d) * scale2d;
    const toScreenY2d = y => margin2d + (y - ymin2d) * scale2d;

    // Center of 3D world
    let center3D = [0, 0, 0];
    let ptCount3D = 0;
    D.frames_3d.forEach(people => {{
      people.forEach(person => {{
        person.forEach(p => {{
          center3D[0] += p[0]; center3D[1] += p[1]; center3D[2] += p[2];
          ptCount3D++;
        }});
      }});
    }});
    if (ptCount3D > 0) {{
      center3D = [center3D[0] / ptCount3D, center3D[1] / ptCount3D, center3D[2] / ptCount3D];
    }}

    // Render 2D Frame
    function render2D(frameIdx) {{
      ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
      const people = D.frames_2d[frameIdx] || [];
      const selIdx = D.selected_indices ? D.selected_indices[frameIdx] : 0;

      people.forEach((person, pIdx) => {{
        const isSelected = (pIdx === selIdx);
        const color = isSelected ? '#ea580c' : PALETTE_2D[pIdx % PALETTE_2D.length];
        ctx2d.strokeStyle = color;
        ctx2d.fillStyle = color;
        ctx2d.lineWidth = isSelected ? 3.5 : 1.8;

        const valid = p => p && p[2] >= 0.05 && (p[0] !== 0 || p[1] !== 0);

        // Bones
        for (const [a, b] of EDGES_2D) {{
          if (valid(person[a]) && valid(person[b])) {{
            ctx2d.beginPath();
            ctx2d.moveTo(toScreenX2d(person[a][0]), toScreenY2d(person[a][1]));
            ctx2d.lineTo(toScreenX2d(person[b][0]), toScreenY2d(person[b][1]));
            ctx2d.stroke();
          }}
        }}

        // Joints
        person.forEach(p => {{
          if (valid(p)) {{
            ctx2d.beginPath();
            ctx2d.arc(toScreenX2d(p[0]), toScreenY2d(p[1]), isSelected ? 4 : 2.5, 0, Math.PI * 2);
            ctx2d.fill();
          }}
        }});

        // Bounding box & label
        const validPts = person.filter(valid);
        if (validPts.length > 0) {{
          const xs = validPts.map(p => p[0]);
          const ys = validPts.map(p => p[1]);
          const minX = Math.min(...xs), maxX = Math.max(...xs);
          const minY = Math.min(...ys), maxY = Math.max(...ys);

          ctx2d.setLineDash([4, 3]);
          ctx2d.strokeRect(toScreenX2d(minX), toScreenY2d(minY), (maxX - minX) * scale2d, (maxY - minY) * scale2d);
          ctx2d.setLineDash([]);
          ctx2d.font = '12px system-ui';
          ctx2d.fillText((isSelected ? '★ ' : '') + 'Person ' + pIdx, toScreenX2d(minX), toScreenY2d(minY) - 5);
        }}
      }});
    }}

    // 3D Projection
    function project3D(p) {{
      // Coordinates: X: right, Y: down, Z: forward into room
      // World centered:
      const wx = p[0] - center3D[0];
      const wy = -(p[1] - center3D[1]); // flip Y so up is positive
      const wz = p[2] - center3D[2];

      // Rotate Yaw around Y
      const cosY = Math.cos(camYaw), sinY = Math.sin(camYaw);
      const x1 = wx * cosY - wz * sinY;
      const z1 = wx * sinY + wz * cosY;

      // Rotate Pitch around X
      const cosP = Math.cos(camPitch), sinP = Math.sin(camPitch);
      const y2 = wy * cosP - z1 * sinP;
      const z2 = wy * sinP + z1 * cosP;

      const depth = z2 + camDist;
      if (depth < 10) return null; // behind camera

      const f = 450;
      const scrX = canvas3d.width / 2 + (x1 / depth) * f + camPanX;
      const scrY = canvas3d.height / 2 - (y2 / depth) * f + camPanY;
      return {{ x: scrX, y: scrY, depth: depth }};
    }}

    // Render 3D Frame
    function render3D(frameIdx) {{
      ctx3d.clearRect(0, 0, canvas3d.width, canvas3d.height);

      // Floor grid
      ctx3d.strokeStyle = '#1e293b';
      ctx3d.lineWidth = 1;
      const floorY = 280 - center3D[1]; // approximate floor level
      const gridSize = 350;
      for (let g = -gridSize; g <= gridSize; g += 70) {{
        const p1 = project3D([center3D[0] + g, center3D[1] - floorY, center3D[2] - gridSize]);
        const p2 = project3D([center3D[0] + g, center3D[1] - floorY, center3D[2] + gridSize]);
        if (p1 && p2) {{
          ctx3d.beginPath(); ctx3d.moveTo(p1.x, p1.y); ctx3d.lineTo(p2.x, p2.y); ctx3d.stroke();
        }}
        const q1 = project3D([center3D[0] - gridSize, center3D[1] - floorY, center3D[2] + g]);
        const q2 = project3D([center3D[0] + gridSize, center3D[1] - floorY, center3D[2] + g]);
        if (q1 && q2) {{
          ctx3d.beginPath(); ctx3d.moveTo(q1.x, q1.y); ctx3d.lineTo(q2.x, q2.y); ctx3d.stroke();
        }}
      }}

      const people3D = D.frames_3d[frameIdx] || [];
      const selIdx = D.selected_indices ? D.selected_indices[frameIdx] : 0;

      // Sort people by camera depth
      people3D.forEach((person, pIdx) => {{
        const isSelected = (pIdx === selIdx);
        const color = isSelected ? '#ea580c' : PALETTE_3D[pIdx % PALETTE_3D.length];

        // Project joints
        const projected = person.map(p => project3D(p));

        // Draw bones
        ctx3d.strokeStyle = color;
        ctx3d.lineWidth = isSelected ? 3.5 : 2.0;
        for (const [a, b] of BONES_3D) {{
          const pa = projected[a], pb = projected[b];
          if (pa && pb) {{
            ctx3d.beginPath();
            ctx3d.moveTo(pa.x, pa.y);
            ctx3d.lineTo(pb.x, pb.y);
            ctx3d.stroke();
          }}
        }}

        // Draw joint spheres
        ctx3d.fillStyle = color;
        projected.forEach((p, jIdx) => {{
          if (p) {{
            const rad = Math.max(1.5, Math.min(6, (isSelected ? 4 : 2.5) * (400 / p.depth)));
            ctx3d.beginPath();
            ctx3d.arc(p.x, p.y, rad, 0, Math.PI * 2);
            ctx3d.fill();
          }}
        }});

        // Head label & height calculation
        const headPt = projected[15];
        const footPt = projected[10] || projected[11];
        if (headPt) {{
          ctx3d.font = '12px system-ui';
          ctx3d.fillStyle = '#f8fafc';
          let heightText = '';
          if (person[15] && footPt && person[10]) {{
            const h = Math.abs(person[10][1] - person[15][1]);
            heightText = ' (h≈' + Math.round(h) + 'px)';
          }}
          ctx3d.fillText('Person ' + pIdx + (isSelected ? ' [TARGET]' : '') + heightText, headPt.x - 20, headPt.y - 12);
        }}
      }});
    }}

    function renderCurrentFrame() {{
      const t = Number(slider.value);
      frameNumber.textContent = 'Frame: ' + t + ' / ' + (D.num_frames - 1);
      render2D(t);
      render3D(t);

      const p2d = D.frames_2d[t] ? D.frames_2d[t].length : 0;
      const p3d = D.frames_3d[t] ? D.frames_3d[t].length : 0;
      const sel = D.selected_indices ? D.selected_indices[t] : 0;
      frameDetails.textContent = 'Số người phát hiện: 2D=' + p2d + ' người, 3D=' + p3d + ' người | Đang chọn theo dõi: Person ' + sel;
    }}

    // Playback loop
    let isPlaying = false;
    let timer = null;
    const btnPlay = document.getElementById('btnPlay');
    function togglePlay() {{
      isPlaying = !isPlaying;
      btnPlay.textContent = isPlaying ? '⏸ Tạm dừng' : '▶ Phát';
      if (isPlaying) {{
        const speed = parseFloat(document.getElementById('playbackSpeed').value);
        const interval = Math.round(100 / speed);
        timer = setInterval(() => {{
          let next = Number(slider.value) + 1;
          if (next >= D.num_frames) next = 0;
          slider.value = next;
          renderCurrentFrame();
        }}, interval);
      }} else {{
        clearInterval(timer);
      }}
    }}
    btnPlay.onclick = togglePlay;
    document.getElementById('playbackSpeed').onchange = () => {{
      if (isPlaying) {{ togglePlay(); togglePlay(); }}
    }};

    slider.oninput = renderCurrentFrame;
    document.getElementById('btnPrev').onclick = () => {{
      slider.value = Math.max(0, Number(slider.value) - 1);
      renderCurrentFrame();
    }};
    document.getElementById('btnNext').onclick = () => {{
      slider.value = Math.min(D.num_frames - 1, Number(slider.value) + 1);
      renderCurrentFrame();
    }};

    // 3D Canvas Mouse Orbit Controls
    canvas3d.onmousedown = e => {{
      isDragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
    }};
    window.onmousemove = e => {{
      if (!isDragging) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      camYaw += dx * 0.008;
      camPitch = Math.max(-1.4, Math.min(1.4, camPitch + dy * 0.008));
      renderCurrentFrame();
    }};
    window.onmouseup = () => {{ isDragging = false; }};
    canvas3d.onwheel = e => {{
      e.preventDefault();
      camDist = Math.max(150, Math.min(2500, camDist + e.deltaY * 0.8));
      renderCurrentFrame();
    }};

    // Preset View Buttons
    document.getElementById('btnViewFront').onclick = () => {{
      camYaw = 0; camPitch = 0; camDist = 600; camPanX = 0; camPanY = 0;
      renderCurrentFrame();
    }};
    document.getElementById('btnViewSide').onclick = () => {{
      camYaw = Math.PI / 2; camPitch = 0; camDist = 600; camPanX = 0; camPanY = 0;
      renderCurrentFrame();
    }};
    document.getElementById('btnViewTop').onclick = () => {{
      camYaw = 0; camPitch = 1.35; camDist = 600; camPanX = 0; camPanY = 0;
      renderCurrentFrame();
    }};
    document.getElementById('btnViewReset').onclick = () => {{
      camYaw = 0.4; camPitch = 0.2; camDist = 600; camPanX = 0; camPanY = 0;
      renderCurrentFrame();
    }};

    // Persistence & Review LocalStorage
    const storageKey = 'review_3d_' + D.clip_id;
    function loadSavedReview() {{
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;
        const obj = JSON.parse(raw);
        document.getElementById('review_tracking').value = obj.review_tracking || '';
        document.getElementById('review_action').value = obj.review_action || '';
        document.getElementById('review_notes').value = obj.review_notes || '';
        ['flag_multiple_people', 'flag_occlusion', 'flag_identity_switch', 'flag_child_leaves',
         'flag_cross_path', 'flag_pose_failure', 'flag_camera_motion'].forEach(f => {{
           document.getElementById(f).checked = !!obj[f];
         }});
      }} catch (e) {{}}
    }}

    function getReviewObject() {{
      const obj = {{
        clip_id: D.clip_id,
        activity: D.activity,
        subject_id: D.subject_id,
        session_id: D.session_id,
        num_frames: D.num_frames,
        review_tracking: document.getElementById('review_tracking').value,
        review_action: document.getElementById('review_action').value,
        review_notes: document.getElementById('review_notes').value,
        updated_at: new Date().toISOString()
      }};
      ['flag_multiple_people', 'flag_occlusion', 'flag_identity_switch', 'flag_child_leaves',
       'flag_cross_path', 'flag_pose_failure', 'flag_camera_motion'].forEach(f => {{
         obj[f] = document.getElementById(f).checked ? 1 : 0;
       }});
      return obj;
    }}

    function saveReview() {{
      const obj = getReviewObject();
      localStorage.setItem(storageKey, JSON.stringify(obj));
      const s = document.getElementById('saveStatus');
      s.textContent = '✓ Đã lưu nháp ' + new Date().toLocaleTimeString();
      setTimeout(() => {{ s.textContent = ''; }}, 2500);
    }}

    document.getElementById('btnSave').onclick = saveReview;
    document.querySelectorAll('#review_tracking, #review_action, #review_notes, input[type="checkbox"]').forEach(el => {{
      el.onchange = saveReview;
    }});

    // Export CSV
    document.getElementById('btnExportCSV').onclick = () => {{
      const obj = getReviewObject();
      const headers = Object.keys(obj);
      const row = headers.map(k => '"' + String(obj[k]).replaceAll('"', '""') + '"').join(',');
      const csvContent = '\\uFEFF' + headers.join(',') + '\\n' + row + '\\n';
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = D.clip_id + '_review_3d.csv';
      a.click();
      URL.revokeObjectURL(url);
    }};

    // Initial render
    loadSavedReview();
    renderCurrentFrame();
  </script>
</body>
</html>
"""
    return html


def build_index_html(records: list[dict], out_dir: Path) -> None:
    """Build the index dashboard listing all multimodal review clips."""
    rows_html = []
    for r in records:
        clip_id = r["clip_id"]
        rows_html.append(f"""
        <tr>
          <td><a href="{clip_id}.html" style="font-weight:600;">{clip_id}</a></td>
          <td><span class="badge">{r.get('activity', '')}</span></td>
          <td>{r.get('subject_id', '')}</td>
          <td>{r.get('session_id', '')}</td>
          <td>{r.get('num_frames', '')}</td>
          <td id="status_{clip_id}">—</td>
        </tr>
        """)
    
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Danh mục Kiểm toán Khung xương Đa phương thức (2D + 3D)</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      padding: 24px;
      line-height: 1.5;
    }}
    a {{ color: #38bdf8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 9999px;
      font-size: 12px;
      background: #0284c7;
      color: white;
    }}
    .card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{ color: #94a3b8; font-weight: 600; }}
    tr:hover {{ background: #243247; }}
    input {{
      background: #0f172a;
      border: 1px solid #334155;
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      width: 300px;
    }}
    button {{
      background: #16a34a;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{ background: #15803d; }}
  </style>
</head>
<body>
  <h1>Kiểm toán Khung xương Đa phương thức (2D OpenPose + 3D ROMP)</h1>
  <p style="color: #94a3b8; margin-top: 4px;">Xem trực quan mô hình người 3D (xoay 360 độ) đối chiếu cùng khung xương 2D để xác thực độ chính xác của nhãn hành động và tracking.</p>

  <div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <input type="text" id="search" placeholder="🔍 Tìm kiếm clip_id, activity, subject...">
      <div>
        <button id="btnExportAll">📥 Xuất toàn bộ CSV nhãn hợp nhất</button>
      </div>
    </div>
    <div id="statSummary" style="margin-top: 10px; font-size: 13px; color: #38bdf8;"></div>

    <table>
      <thead>
        <tr>
          <th>Clip ID</th>
          <th>Hoạt động (Activity)</th>
          <th>Subject</th>
          <th>Session</th>
          <th>Số frame</th>
          <th>Trạng thái Review</th>
        </tr>
      </thead>
      <tbody id="clipTable">
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>

  <script>
    const clips = {json.dumps([r['clip_id'] for r in records])};
    function updateStatus() {{
      let reviewed = 0;
      clips.forEach(id => {{
        const cell = document.getElementById('status_' + id);
        if (!cell) return;
        try {{
          const item = localStorage.getItem('review_3d_' + id);
          if (item) {{
            const obj = JSON.parse(item);
            if (obj.review_tracking) {{
              cell.innerHTML = '<span style="color:#22c55e; font-weight:600;">✓ ' + obj.review_tracking + ' (' + (obj.review_action || 'Chưa xét pose') + ')</span>';
              reviewed++;
              return;
            }}
          }}
          cell.innerHTML = '<span style="color:#94a3b8;">Chưa review</span>';
        }} catch(e) {{}}
      }});
      document.getElementById('statSummary').textContent = 'Tiến độ: ' + reviewed + ' / ' + clips.length + ' clip đã được đánh giá';
    }}

    document.getElementById('search').oninput = e => {{
      const q = e.target.value.toLowerCase();
      document.querySelectorAll('#clipTable tr').forEach(tr => {{
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }};

    document.getElementById('btnExportAll').onclick = () => {{
      const allRows = [];
      clips.forEach(id => {{
        try {{
          const item = localStorage.getItem('review_3d_' + id);
          if (item) allRows.push(JSON.parse(item));
        }} catch(e) {{}}
      }});
      if (allRows.length === 0) {{
        alert('Chưa có clip nào được lưu đánh giá trong trình duyệt này.');
        return;
      }}
      const keys = Object.keys(allRows[0]);
      const csvLines = [keys.join(',')];
      allRows.forEach(row => {{
        csvLines.push(keys.map(k => '"' + String(row[k] || '').replaceAll('"', '""') + '"').join(','));
      }});
      const blob = new Blob(['\\uFEFF' + csvLines.join('\\n')], {{ type: 'text/csv;charset=utf-8' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'manual_labels_multimodal_3d.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    }};

    updateStatus();
    window.onfocus = updateStatus;
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of clips to export (default: 10, set 0 for all 120)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/tracking_review_3d")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / "artifacts/tracking_review/review_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing review manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if args.limit > 0:
        records = records[:args.limit]

    out_dir = ensure_dir(args.output_dir)
    config = load_config(PROJECT_ROOT / "configs/paper_repro.yaml")
    threshold = float(config.get("data", {}).get("preprocessing", {}).get("confidence_threshold", 0.1))

    print(f"Exporting {len(records)} multimodal 2D + 3D review clips to {out_dir}...")

    exported = []
    for idx, r in enumerate(records):
        clip_id = r["clip_id"]
        activity = r["activity"]
        
        # 2D path
        openpose_dir = REPO_ROOT / "data_MMASD/2D_skeleton/2D_openpose/output" / activity / clip_id
        if not openpose_dir.is_dir():
            print(f"[{idx+1}/{len(records)}] Skipping {clip_id}: missing 2D dir {openpose_dir}")
            continue

        # 3D path
        romp_dir = REPO_ROOT / "data_MMASD/3D_skeleton/ROMP_3D_Coordinates" / activity / clip_id
        if not romp_dir.is_dir():
            print(f"[{idx+1}/{len(records)}] Skipping {clip_id}: missing 3D dir {romp_dir}")
            continue

        frames_2d, _ = parse_clip(openpose_dir, strict=True)
        frames_3d = load_romp_3d_clip(romp_dir)

        # Ensure frame counts align
        num_frames = min(len(frames_2d), len(frames_3d))
        frames_2d = frames_2d[:num_frames]
        frames_3d = frames_3d[:num_frames]

        tracking = select_child_track(
            frames_2d,
            confidence_threshold=threshold,
            strategy=config.get("data", {}).get("tracking_strategy", "torso_interpolation"),
            prefer_smaller=config.get("data", {}).get("prefer_smaller_person", True),
        )

        payload = {
            "clip_id": clip_id,
            "activity": activity,
            "subject_id": r.get("subject_id", ""),
            "session_id": r.get("session_id", ""),
            "num_frames": num_frames,
            "selected_indices": tracking.selected_indices[:num_frames],
            "frames_2d": [[np.round(p.astype(float), 2).tolist() for p in people] for people in frames_2d],
            "frames_3d": frames_3d,
        }

        page_html = generate_multimodal_html(payload)
        (out_dir / f"{clip_id}.html").write_text(page_html, encoding="utf-8")
        exported.append(r)
        print(f"[{idx+1}/{len(records)}] Exported {clip_id} ({num_frames} frames, 2D + 3D)")

    build_index_html(exported, out_dir)
    print(f"\nDone! Exported {len(exported)} multimodal clips.")
    print(f"Open in browser: file://{out_dir.resolve()}/index.html")


if __name__ == "__main__":
    main()
