#!/usr/bin/env python3
"""Compare frozen legacy selection with tracklet V2 on Protocol A, without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.child_tracking import person_features, select_child_track
from src.data.dataset import read_manifest
from src.data.openpose_parser import parse_clip
from src.data.tracklet_tracking import _quality_config, candidate_from_person, select_tracklet_v2
from src.utils import json_dump, load_config


EDGES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
         (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(14,19),(19,20),
         (14,21),(11,22),(22,23),(11,24)]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def feature_sequence(result, threshold):
    return [person_features(person, threshold) if person is not None else None for person in result.selected_people]


def metrics_for(result, frames, threshold, center_threshold, scale_threshold, quality):
    features = feature_sequence(result, threshold)
    valid_counts = [sum(person_features(person, threshold) is not None for person in people) for people in frames]
    center_events=[]; scale_events=[]; switch_events=[]; proxy_events=[]
    valid_transitions=0
    center_values=[]; scale_values=[]
    for time in range(1,len(frames)):
        previous,current=features[time-1],features[time]
        if previous is None or current is None:
            continue
        valid_transitions += 1
        reference=max(previous.diagonal,current.diagonal,1.0)
        center=float(np.linalg.norm(previous.center-current.center)/reference)
        scale=abs(math.log(max(current.diagonal,1.0)/max(previous.diagonal,1.0)))
        changed=result.selected_indices[time]!=result.selected_indices[time-1]
        center_values.append(center); scale_values.append(scale)
        if center>center_threshold: center_events.append(time)
        if scale>scale_threshold: scale_events.append(time)
        if changed: switch_events.append(time)
        flags=sum((center>center_threshold,scale>scale_threshold,changed))
        if (valid_counts[time-1]>=2 or valid_counts[time]>=2) and flags>=2:
            proxy_events.append(time)
    fragments=0; selected_frames=0
    for people,index in zip(frames,result.selected_indices,strict=True):
        if index<0: continue
        selected_frames += 1
        accepted,_=candidate_from_person(people[index],index,threshold,quality)
        fragments += accepted is None
    missing=len(frames)-selected_frames
    return {
        "num_frames":len(frames),"selected_frames":selected_frames,"missing_frames":missing,
        "fragment_frames":fragments,"fragment_rate":fragments/max(len(frames),1),
        "centroid_jump_count":len(center_events),"scale_jump_count":len(scale_events),
        "candidate_index_change_count":len(switch_events),"continuity_proxy_count":len(proxy_events),
        "centroid_jump_frames":";".join(map(str,center_events)),
        "scale_jump_frames":";".join(map(str,scale_events)),
        "candidate_index_change_frames":";".join(map(str,switch_events)),
        "continuity_proxy_frames":";".join(map(str,proxy_events)),
        "valid_transition_count":valid_transitions,
        "centroid_jump_rate":len(center_events)/max(valid_transitions,1),
        "scale_jump_rate":len(scale_events)/max(valid_transitions,1),
        "continuity_proxy_rate":len(proxy_events)/max(valid_transitions,1),
        "missing_rate":missing/max(len(frames),1),
        "selected_confidence_mean":float(np.mean([f.mean_confidence for f in features if f is not None])) if selected_frames else None,
        "center_displacement_mean":float(np.mean(center_values)) if center_values else None,
        "scale_log_jump_mean":float(np.mean(scale_values)) if scale_values else None,
        "high_risk":bool(proxy_events or missing/max(len(frames),1)>=0.05),
    }


def draw(ax, people, selected, threshold, color):
    for index,person in enumerate(people):
        f=person_features(person,threshold)
        if f is None: continue
        valid=(np.isfinite(person).all(1)&(person[:,2]>=threshold)&np.any(person[:,:2]!=0,axis=1))
        chosen=index==selected
        line_color=color if chosen else "#91a4b7"; width=3 if chosen else 1
        for a,b in EDGES:
            if valid[a] and valid[b]: ax.plot(person[[a,b],0],person[[a,b],1],c=line_color,lw=width)
        ax.scatter(person[valid,0],person[valid,1],s=16 if chosen else 7,c=line_color)
        low=person[valid,:2].min(0);high=person[valid,:2].max(0)
        ax.add_patch(plt.Rectangle(low,*(high-low),fill=False,ec=line_color,lw=width,ls="--"))
        ax.text(low[0],low[1],f"#{index} j={f.valid_count} d={f.diagonal:.0f}",fontsize=6,color=line_color,
                bbox=dict(fc="white",ec="none",alpha=.65,pad=.3))
    if selected<0:
        ax.text(.5,.5,"TARGET MISSING",transform=ax.transAxes,ha="center",va="center",color="#b2182b",fontsize=12)
    ax.invert_yaxis();ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.12)


def render_comparison(record, frames, v1, v2, indices, path, threshold):
    figure,axes=plt.subplots(2,len(indices),figsize=(4*len(indices),8),squeeze=False)
    for column,time in enumerate(indices):
        draw(axes[0,column],frames[time],v1.selected_indices[time],threshold,"#e66101")
        draw(axes[1,column],frames[time],v2.selected_indices[time],threshold,"#1b9e77")
        axes[0,column].set_title(f"V1 t={time} selected={v1.selected_indices[time]}",fontsize=8)
        axes[1,column].set_title(f"V2 t={time} selected={v2.selected_indices[time]}",fontsize=8)
    figure.suptitle(f"{record['clip_id']} | {record['activity']}\nrows: legacy V1 (orange) vs tracklet V2 (green); gray=other raw candidates",fontsize=11)
    figure.tight_layout(rect=(0,0,1,.93));path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=130);plt.close(figure)


def aggregate(rows, prefix):
    frames=sum(row[f"{prefix}_num_frames"] for row in rows)
    transitions=sum(row[f"{prefix}_valid_transition_count"] for row in rows)
    return {
        "fragment_selection_frames":sum(row[f"{prefix}_fragment_frames"] for row in rows),
        "fragment_selection_rate":sum(row[f"{prefix}_fragment_frames"] for row in rows)/frames,
        "high_risk_clips":sum(row[f"{prefix}_high_risk"] for row in rows),
        "centroid_jump_events":sum(row[f"{prefix}_centroid_jump_count"] for row in rows),
        "centroid_jump_rate":sum(row[f"{prefix}_centroid_jump_count"] for row in rows)/max(transitions,1),
        "scale_jump_events":sum(row[f"{prefix}_scale_jump_count"] for row in rows),
        "scale_jump_rate":sum(row[f"{prefix}_scale_jump_count"] for row in rows)/max(transitions,1),
        "continuity_proxy_events":sum(row[f"{prefix}_continuity_proxy_count"] for row in rows),
        "continuity_proxy_rate":sum(row[f"{prefix}_continuity_proxy_count"] for row in rows)/max(transitions,1),
        "clips_with_continuity_proxy":sum(row[f"{prefix}_continuity_proxy_count"]>0 for row in rows),
        "missing_target_frames":sum(row[f"{prefix}_missing_frames"] for row in rows),
        "missing_target_rate":sum(row[f"{prefix}_missing_frames"] for row in rows)/frames,
        "mean_selected_confidence":float(np.mean([row[f"{prefix}_selected_confidence_mean"] for row in rows if row[f"{prefix}_selected_confidence_mean"] is not None])),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--v2-config",type=Path,default=ROOT/"configs/paper_repro_tracker_v2.yaml")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/tracker_v2_audit")
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=load_config(args.v2_config);tracking=config["tracking"]
    threshold=float(config["data"]["preprocessing"]["confidence_threshold"])
    old_rules=json.loads((ROOT/"artifacts/person_selection_audit/thresholds_and_rules.json").read_text())
    center_threshold=float(old_rules["centroid_normalized_threshold"])
    scale_threshold=float(old_rules["scale_abs_log_ratio_threshold"])
    quality=_quality_config(tracking)
    records=read_manifest(ROOT/"artifacts/paper_repro/manifest.csv")
    previous_top={row["clip_id"]:row for row in read_manifest(ROOT/"artifacts/person_selection_audit/top20_visual_manifest.csv")}
    rows=[];visual_cache={}
    for number,record in enumerate(records,1):
        frames,_=parse_clip(record["path"],strict=True)
        v1=select_child_track(frames,threshold,"temporal_nearest",True)
        v2=select_tracklet_v2(frames,threshold,tracking)
        m1=metrics_for(v1,frames,threshold,center_threshold,scale_threshold,quality)
        m2=metrics_for(v2,frames,threshold,center_threshold,scale_threshold,quality)
        row={"clip_id":record["clip_id"],"activity":record["activity"],"subject":record["subject_id"],"session":record["session_id"]}
        row.update({f"v1_{key}":value for key,value in m1.items()})
        row.update({f"v2_{key}":value for key,value in m2.items()})
        row.update({
            "fragment_rate_delta":m2["fragment_rate"]-m1["fragment_rate"],
            "centroid_jump_count_delta":m2["centroid_jump_count"]-m1["centroid_jump_count"],
            "scale_jump_count_delta":m2["scale_jump_count"]-m1["scale_jump_count"],
            "continuity_proxy_count_delta":m2["continuity_proxy_count"]-m1["continuity_proxy_count"],
            "missing_frames_delta":m2["missing_frames"]-m1["missing_frames"],
            "v2_track_count":len(v2.tracks),"v2_target_track_id":v2.target_track_id,
            "v2_rejected_candidate_count":sum(v2.rejected_candidates_per_frame),
        })
        rows.append(row)
        if record["clip_id"] in previous_top:
            visual_cache[record["clip_id"]]=(record,frames,v1,v2)
        if number%100==0:print(f"audited {number}/{len(records)}",flush=True)
    write_csv(out/"per_clip_v1_v2.csv",rows)
    v1_total=aggregate(rows,"v1");v2_total=aggregate(rows,"v2")
    comparisons=[]
    for metric in v1_total:
        before=v1_total[metric];after=v2_total[metric]
        comparisons.append({"metric":metric,"v1":before,"v2":after,"delta":after-before,
            "relative_change":(after-before)/before if isinstance(before,(int,float)) and before else None})
    write_csv(out/"aggregate_comparison.csv",comparisons)
    per_activity=[]
    for activity in sorted({row["activity"] for row in rows}):
        group=[row for row in rows if row["activity"]==activity]
        a1,a2=aggregate(group,"v1"),aggregate(group,"v2")
        per_activity.append({"activity":activity,"clips":len(group),
            **{f"v1_{k}":v for k,v in a1.items()},**{f"v2_{k}":v for k,v in a2.items()}})
    write_csv(out/"per_activity_v1_v2.csv",per_activity)

    visual_rows=[]
    for clip_id,visual in previous_top.items():
        record,frames,v1,v2=visual_cache[clip_id]
        center=int(visual["center_frame"]);indices=list(range(max(0,center-2),min(len(frames),center+3)))
        path=out/"visual_comparisons"/f"{clip_id}_around_{center}.png"
        render_comparison(record,frames,v1,v2,indices,path,threshold)
        visual_rows.append({"clip_id":clip_id,"center_frame":center,"frame_indices":";".join(map(str,indices)),
            "v1_selected_indices":";".join(str(v1.selected_indices[x]) for x in indices),
            "v2_selected_indices":";".join(str(v2.selected_indices[x]) for x in indices),
            "artifact":str(path.relative_to(ROOT))})
    write_csv(out/"visual_comparison_manifest.csv",visual_rows)

    def reduction(metric):
        before=v1_total[metric];after=v2_total[metric]
        return 1-after/before if before else (1.0 if not after else -math.inf)
    gates={
        "fragment_reduction_at_least_80pct":reduction("fragment_selection_rate")>=.80,
        "high_risk_clip_reduction_at_least_30pct":reduction("high_risk_clips")>=.30,
        "centroid_jump_rate_reduction_at_least_30pct":reduction("centroid_jump_rate")>=.30,
        "scale_jump_rate_reduction_at_least_30pct":reduction("scale_jump_rate")>=.30,
        "missing_target_rate_at_most_5pct":v2_total["missing_target_rate"]<=.05,
    }
    summary={
        "status":"PASS" if all(gates.values()) else "FAIL",
        "clips":len(rows),"frames":sum(row["v1_num_frames"] for row in rows),
        "same_manifest":True,"same_proxy_thresholds":True,
        "centroid_threshold":center_threshold,"scale_log_threshold":scale_threshold,
        "fragment_definition":f"selected detection fails V2 quality rule: valid_joints>={quality.min_valid_joints}, torso_anchors>={quality.min_torso_anchors}, diagonal>={quality.min_diagonal_px}px",
        "v1":v1_total,"v2":v2_total,
        "relative_reductions":{metric:reduction(metric) for metric in ("fragment_selection_rate","high_risk_clips","centroid_jump_rate","scale_jump_rate","continuity_proxy_rate")},
        "pass_gates":gates,"training_performed":False,
        "config":{key:value for key,value in config.items() if not key.startswith("_")},
    }
    json_dump(summary,out/"summary.json")
    print(json.dumps({key:value for key,value in summary.items() if key!="config"},indent=2))


if __name__=="__main__":
    main()
