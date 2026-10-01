#!/usr/bin/env python3
"""Export stratified raw-skeleton review pages; never assign child ground truth."""
import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.child_tracking import person_features, select_child_track
from src.data.dataset import read_manifest
from src.data.openpose_parser import parse_clip
from src.utils import json_dump, load_config
from src.window_study import verify_preserved, write_csv


def reordering_frames(frames, threshold):
    """Mutual nearest torso matches changing raw index: a review proxy only.

    Restrict to equal candidate counts >=2 and normalized torso distance <0.25.
    This can still confuse crossings or detection noise with raw-index reordering.
    """
    events = []
    previous = []
    for time, people in enumerate(frames):
        current = [person_features(p, threshold) for p in people]
        if len(previous) == len(current) and len(current) >= 2 and all(f is not None for f in previous + current):
            distances = np.asarray([[np.linalg.norm(a.torso - b.torso) / max(a.diagonal, b.diagonal, 1)
                                     for b in current] for a in previous])
            forward, backward = distances.argmin(1), distances.argmin(0)
            if any(j != i and backward[j] == i and distances[i, j] < 0.25 for i, j in enumerate(forward)):
                events.append(time)
        previous = current
    return events


def choose_stratified(rows, per_class):
    chosen = []
    for activity in sorted({r["activity"] for r in rows}):
        pool = [r for r in rows if r["activity"] == activity]
        selected = []
        def take(candidates, key):
            candidates = [r for r in candidates if r not in selected]
            if candidates:
                selected.append(max(candidates, key=key))
        # Explicit session and people strata, then high reordering proxies.
        for day in ("D1", "D8", "D16"):
            take([r for r in pool if r["session_id"] == day], lambda r: (r["reordering_proxy_rate"], r["clip_id"]))
        take([r for r in pool if r["max_people"] == 1], lambda r: (r["num_frames"], r["clip_id"]))
        take([r for r in pool if r["frames_one_person"] > 0], lambda r: (r["frames_one_person"], r["clip_id"]))
        take([r for r in pool if r["max_people"] >= 2], lambda r: (r["frames_multiple_people"], r["clip_id"]))
        take(pool, lambda r: (r["reordering_proxy_rate"], r["reordering_proxy_events"], r["clip_id"]))
        while len(selected) < min(per_class, len(pool)):
            subjects = {r["subject_id"] for r in selected}
            sessions = Counter(r["session_id"] for r in selected)
            take(pool, lambda r: (r["subject_id"] not in subjects, -sessions[r["session_id"]], r["reordering_proxy_rate"], r["clip_id"]))
        chosen.extend(selected)
    if len(chosen) < 100:
        raise RuntimeError("At least 100 review clips are required")
    return chosen


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Tracked candidate review</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#18232e}canvas{background:white;border:1px solid #ccd3da;width:min(100%,1000px)}button,select,input{font:inherit;margin:6px}#info{white-space:pre-wrap;overflow-wrap:anywhere}label{display:block}</style>
<a href="index.html">All review clips</a><h1 id="title"></h1>
<p>Manual identity review. Orange = selected tracked candidate; other colors = raw candidates; black square = MidHip. This is a raw skeleton frame view, without RGB video. Selection is not child ground truth.</p>
<button id="play">Play / pause</button><input id="frame" type="range" min="0" value="0" style="width:50%"><span id="number"></span>
<button id="event">Next reordering proxy</button><br><canvas id="canvas" width="1100" height="680"></canvas><pre id="info"></pre>
<label>Human review: <select id="decision"><option value="unreviewed">Unreviewed</option><option value="child_consistent">Appears to track child consistently</option><option value="adult">Appears to track adult</option><option value="switch">Apparent identity switch</option><option value="uncertain">Uncertain / requires RGB</option></select></label>
<label>Reviewer notes: <input id="notes" size="85" placeholder="Include frame numbers and uncertainty"></label>
<button id="download">Download this review JSON</button><p>Draft responses are stored in this browser when local storage is available. Download JSON to preserve them.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent),c=document.getElementById('canvas'),ctx=c.getContext('2d'),slider=document.getElementById('frame');
const edges=[[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14],[0,15],[15,17],[0,16],[16,18],[14,19],[19,20],[14,21],[11,22],[22,23],[11,24]];
const colors=['#2563eb','#15803d','#9333ea','#0891b2','#be185d'];
document.getElementById('title').textContent=d.clip_id+' | '+d.activity+' | '+d.subject_id+' '+d.session_id;
slider.max=d.frames.length-1;
const points=d.frames.flat(2).filter(p=>p[2]>=d.threshold&&(p[0]!==0||p[1]!==0));
let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
for(const p of points){xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1]);}
if(!points.length){xmin=0;xmax=1;ymin=0;ymax=1;}
const scale=Math.min((c.width-100)/Math.max(xmax-xmin,1),(c.height-100)/Math.max(ymax-ymin,1));
const X=x=>50+(x-xmin)*scale,Y=y=>50+(y-ymin)*scale;
function draw(){const t=+slider.value,selected=d.selected_indices[t];ctx.clearRect(0,0,c.width,c.height);let lines=['Source frame: '+d.source_files[t],'Selected index: '+selected+'; frame '+t+' / '+(d.frames.length-1)];
 d.frames[t].forEach((person,i)=>{const color=i===selected?'#ea580c':colors[i%colors.length];ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=i===selected?3.5:1.5;
 const valid=p=>p[2]>=d.threshold&&(p[0]!==0||p[1]!==0);const good=person.filter(valid);
 for(const [a,b] of edges){if(valid(person[a])&&valid(person[b])){ctx.beginPath();ctx.moveTo(X(person[a][0]),Y(person[a][1]));ctx.lineTo(X(person[b][0]),Y(person[b][1]));ctx.stroke();}}
 for(const p of person){if(valid(p)){ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),i===selected?4:2.5,0,2*Math.PI);ctx.fill();}}
 if(good.length){const xx=good.map(p=>p[0]),yy=good.map(p=>p[1]);const lo=[Math.min(...xx),Math.min(...yy)],hi=[Math.max(...xx),Math.max(...yy)];ctx.setLineDash([5,3]);ctx.strokeRect(X(lo[0]),Y(lo[1]),(hi[0]-lo[0])*scale,(hi[1]-lo[1])*scale);ctx.setLineDash([]);ctx.font='16px system-ui';ctx.fillText('candidate '+i+(i===selected?' SELECTED':''),X(lo[0]),Y(lo[1])-8);const conf=good.reduce((s,p)=>s+p[2],0)/good.length;lines.push('Candidate '+i+': bbox '+JSON.stringify([lo,hi])+' | mean valid-joint confidence '+conf.toFixed(3));}
 else lines.push('Candidate '+i+': no joints above threshold; bbox unavailable');
 const hip=person[8];if(valid(hip)){ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.strokeRect(X(hip[0])-6,Y(hip[1])-6,12,12);}lines.push('  MidHip x/y/confidence: '+JSON.stringify(hip));
 });document.getElementById('number').textContent='Frame '+t;document.getElementById('info').textContent=lines.join('\n');}
slider.oninput=draw;let playing=false;document.getElementById('play').onclick=()=>playing=!playing;
setInterval(()=>{if(playing){slider.value=(+slider.value+1)%d.frames.length;draw();}},100);
document.getElementById('event').onclick=()=>{slider.value=d.reordering_proxy_frames.find(t=>t>+slider.value)??d.reordering_proxy_frames[0]??0;draw();};
const decision=document.getElementById('decision'),notes=document.getElementById('notes'),key='mmasd_review_'+d.clip_id;
try{const old=JSON.parse(localStorage.getItem(key)||'{}');decision.value=old.decision||'unreviewed';notes.value=old.notes||'';}catch(e){}
const response=()=>({clip_id:d.clip_id,decision:decision.value,notes:notes.value,review_type:'human_visual_review',selected_track_is_ground_truth:false});
decision.onchange=notes.oninput=()=>{try{localStorage.setItem(key,JSON.stringify(response()));}catch(e){}};
document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(response(),null,2)],{type:'application/json'}));a.download=d.clip_id+'_review.json';a.click();URL.revokeObjectURL(a.href);};draw();
</script>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tracking_review")
    args = parser.parse_args()
    if args.per_class < 10:
        raise ValueError("Use at least 10 clips per class")
    verify_preserved()
    config = load_config(ROOT / "configs/paper_repro.yaml")
    threshold = config["data"]["preprocessing"]["confidence_threshold"]
    records = [dict(r, split=split) for split in ("train", "val")
               for r in read_manifest(ROOT / f"artifacts/splits/paper_repro/{split}.csv")]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = []
    for i, record in enumerate(records):
        frames, _ = parse_clip(record["path"], strict=True)
        events = reordering_frames(frames, threshold)
        counts = [len(f) for f in frames]
        stats.append({**record, "num_frames": len(frames), "max_people": max(counts, default=0),
                      "frames_one_person": counts.count(1), "frames_multiple_people": sum(n >= 2 for n in counts),
                      "reordering_proxy_events": len(events), "reordering_proxy_rate": len(events) / max(len(frames)-1, 1)})
        if (i+1) % 100 == 0:
            print(f"Scanned {i+1}/{len(records)} raw clips", flush=True)
    write_csv(output / "candidate_population.csv", stats)
    selected = choose_stratified(stats, args.per_class)
    write_csv(output / "review_manifest.csv", selected)
    write_csv(output / "manual_review_template.csv", [{"clip_id": r["clip_id"], "decision": "unreviewed", "frame_numbers": "", "notes": ""} for r in selected])
    index_rows = []
    for i, record in enumerate(selected):
        frames, paths = parse_clip(record["path"], strict=True)
        tracking = select_child_track(frames, confidence_threshold=threshold,
                                      strategy=config["data"]["tracking_strategy"],
                                      prefer_smaller=config["data"]["prefer_smaller_person"])
        events = reordering_frames(frames, threshold)
        payload = {**record, "threshold": threshold, "selected_indices": tracking.selected_indices,
                   "source_files": [str(p) for p in paths], "reordering_proxy_frames": events,
                   "frames": [[np.round(person.astype(float), 3).tolist() for person in people] for people in frames]}
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        page = record["clip_id"] + ".html"
        (output / page).write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
        index_rows.append(f'<tr><td><a href="{html.escape(page)}">{html.escape(record["clip_id"])}</a></td><td>{record["activity"]}</td><td>{record["session_id"]}</td><td>{record["subject_id"]}</td><td>{record["max_people"]}</td><td>{record["reordering_proxy_events"]}</td></tr>')
        if (i+1) % 20 == 0:
            print(f"Exported {i+1}/{len(selected)} review pages", flush=True)
    intro = '<!doctype html><meta charset="utf-8"><title>MMASD manual identity review</title><style>body{font:16px system-ui;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}input{font:inherit}</style><h1>Manual tracked-person identity review</h1><p>Skeleton-only review; use RGB where child/adult identity is ambiguous. No automatic child ground truth. Reordering counts are a mutual-nearest-torso proxy, not verified identity switches. Each page has every raw frame, all candidates, selected index, boxes, MidHip and confidence.</p><input id="q" placeholder="Filter class, day or subject"><table><thead><tr><th>Clip</th><th>Class</th><th>Day</th><th>Subject</th><th>Max people</th><th>Reordering proxy events</th></tr></thead><tbody>'
    ending = '</tbody></table><script>document.getElementById("q").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));</script>'
    (output / "index.html").write_text(intro + "\n".join(index_rows) + ending)
    json_dump({"review_clips": len(selected), "per_class": dict(Counter(r["activity"] for r in selected)),
               "population_clips": len(stats),
               "population_single_person_only_clips": sum(r["max_people"] == 1 for r in stats),
               "population_clips_with_one_person_frames": sum(r["frames_one_person"] > 0 for r in stats),
               "per_day": dict(Counter(r["session_id"] for r in selected)), "subjects": len({r["subject_id"] for r in selected}),
               "single_person_only_clips": sum(r["max_people"] == 1 for r in selected),
               "clips_with_one_person_frames": sum(r["frames_one_person"] > 0 for r in selected),
               "clips_with_multiple_people": sum(r["max_people"] >= 2 for r in selected),
               "clips_with_reordering_proxy": sum(r["reordering_proxy_events"] > 0 for r in selected),
               "max_reordering_proxy_rate": max(r["reordering_proxy_rate"] for r in selected),
               "selection": "Deterministic class/day/people strata, high reordering proxies, then subject diversity. No accuracy-based selection.",
               "reordering_definition": "Mutual nearest torso matches with changed raw index; equal counts >=2; normalized distance <0.25. Diagnostic proxy only.",
               "review_status": "unreviewed; no child identity ground truth asserted", "tracker_changed": False},
              output / "review_summary.json")
    verify_preserved()
    print(f"Review index: {output / 'index.html'}", flush=True)
    if output == ROOT / "artifacts/tracking_review":
        # Refresh the human-label UI while retaining any existing CSV labels.
        from prepare_manual_review import main as prepare_human_review
        prepare_human_review()


if __name__ == "__main__":
    main()
