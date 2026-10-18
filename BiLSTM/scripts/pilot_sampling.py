"""Deterministic pilot proposals and validation independent of the browser."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

GROUPS = ("tuning", "evaluation")
BANDS = ("low_missing", "medium_missing", "high_missing")
ACTIVITY_LABELS = {
    "arm_swing_as": "vung tay",
    "body_swing_bs": "đung đưa thân",
    "chest_expansion_ce": "mở rộng ngực",
    "drumming_dr": "gõ trống",
    "maracas_forward_shaking_mfs": "lắc maracas phía trước",
    "maracas_shaking_ms": "lắc maracas",
    "sing_and_clap_sac": "hát và vỗ tay",
    "squat_sq": "squat",
    "tree_pose_tr": "tư thế cái cây",
    "twist_pose_tw": "vặn mình",
}
BAND_LABELS = {
    "low_missing": "ít thiếu khớp",
    "medium_missing": "thiếu khớp vừa",
    "high_missing": "thiếu khớp nhiều",
}
SCHEMA = "mmasd_pilot_selection_v1"


def quotas(total: int, activities: list[str], rng: random.Random) -> dict[str, int]:
    order = activities.copy()
    rng.shuffle(order)
    return {
        activity: total // len(order) + (index < total % len(order))
        for index, activity in enumerate(order)
    }


def propose(
    catalog: dict[str, Any],
    target: int = 40,
    seed: int = 42,
    excluded: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if type(target) is not int or not 20 <= target <= 50 or type(seed) is not int:
        raise ValueError("Pilot target must be between 20 and 50")
    excluded = excluded or []
    if not isinstance(excluded, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("clip_id"), str) for item in excluded
    ):
        raise ValueError("Invalid exclusions")
    excluded_ids = {row["clip_id"] for row in excluded}
    rows = [
        row
        for row in catalog["rows"]
        if row["status"] == "ready" and row["clip_id"] not in excluded_ids
    ]
    if not rows:
        raise ValueError("No eligible candidate clips")
    activities = sorted({row["activity"] for row in catalog["rows"]})
    subjects = sorted({row["subject_id"] for row in rows})
    if len(subjects) < 2:
        raise ValueError("At least two subject IDs are required for disjoint groups")
    rng = random.Random(seed)
    group_quotas = {
        "tuning": quotas(target // 2, activities, rng),
        "evaluation": quotas(target - target // 2, activities, rng),
    }
    # Split subjects before selecting clips. Search uses availability, never model accuracy.
    best = None
    for _ in range(256):
        order = subjects.copy()
        rng.shuffle(order)
        partition = set(order[: len(order) // 2])
        capacity = Counter(
            ("tuning" if row["subject_id"] in partition else "evaluation", row["activity"])
            for row in rows
        )
        shortage = sum(
            max(0, count - capacity[group, activity])
            for group, group_counts in group_quotas.items()
            for activity, count in group_counts.items()
        )
        if best is None or shortage < best[0]:
            best = (shortage, partition)
        if shortage == 0:
            break
    partition = best[1]
    selected = []
    band_counts: Counter = Counter()
    subject_counts: Counter = Counter()
    session_counts: Counter = Counter()
    for group in GROUPS:
        for activity in activities:
            pool = [
                row
                for row in rows
                if row["activity"] == activity
                and ((row["subject_id"] in partition) == (group == "tuning"))
            ]
            pool.sort(key=lambda row: row["clip_id"])
            rng.shuffle(pool)
            for _ in range(group_quotas[group][activity]):
                if not pool:
                    break
                bands = {row["missingness_band"] for row in pool}
                band = min(
                    bands, key=lambda value: (band_counts[activity, value], BANDS.index(value))
                )
                candidates = [row for row in pool if row["missingness_band"] == band]
                chosen = min(
                    candidates,
                    key=lambda row: (
                        subject_counts[group, row["subject_id"]],
                        session_counts[group, row["session_id"]],
                    ),
                )
                selected.append(
                    {
                        "clip_id": chosen["clip_id"],
                        "group": group,
                        "reviewed": False,
                        "reason": f"Đề xuất seed {seed}: hoạt động {ACTIVITY_LABELS.get(activity, activity)}, {BAND_LABELS[band]}; bổ sung đa dạng mã trẻ và phiên.",
                    }
                )
                band_counts[activity, band] += 1
                subject_counts[group, chosen["subject_id"]] += 1
                session_counts[group, chosen["session_id"]] += 1
                pool.remove(chosen)
    return {
        "schema_version": SCHEMA,
        "catalog_sha256": catalog["catalog_sha256"],
        "target": target,
        "seed": seed,
        "selected": selected,
        "excluded": excluded,
        "method": "Subject partition first; activity quotas; missingness strata; subject/session diversity; fixed seed. Exploratory sample, not population prevalence.",
    }


def validate_draft(draft: dict[str, Any], catalog: dict[str, Any]) -> None:
    if (
        not isinstance(draft, dict)
        or not isinstance(draft.get("selected"), list)
        or not isinstance(draft.get("excluded"), list)
    ):
        raise ValueError("Bản nháp phải có danh sách clip chọn và clip loại hợp lệ.")
    if (
        draft.get("schema_version") != SCHEMA
        or draft.get("catalog_sha256") != catalog["catalog_sha256"]
    ):
        raise ValueError(
            "Bản chọn thuộc phiên bản dữ liệu khác; cần xuất lại hoặc chọn đúng catalogue."
        )
    if type(draft.get("target")) is not int or not 20 <= draft["target"] <= 50:
        raise ValueError("Số clip mục tiêu phải nằm trong khoảng 20–50.")
    if type(draft.get("seed")) is not int:
        raise ValueError("Seed phải là số nguyên.")
    known = {row["clip_id"]: row for row in catalog["rows"]}
    seen = set()
    for selection in draft.get("selected", []):
        if not isinstance(selection, dict) or not isinstance(selection.get("clip_id"), str):
            raise ValueError("Clip chọn không hợp lệ.")
        clip_id = selection.get("clip_id")
        if clip_id not in known or clip_id in seen or known[clip_id]["status"] != "ready":
            raise ValueError("Clip không hợp lệ, lặp hoặc bị cách ly: " + str(clip_id))
        seen.add(clip_id)
        if selection.get("group") not in GROUPS:
            raise ValueError("Nhóm clip không hợp lệ.")
        if type(selection.get("reviewed")) is not bool or not isinstance(
            selection.get("reason"), str
        ):
            raise ValueError("Trạng thái review hoặc lý do không hợp lệ.")
        if len(selection["reason"]) > 4000:
            raise ValueError("Lý do tối đa 4.000 ký tự.")
    excluded_seen = set()
    for exclusion in draft.get("excluded", []):
        if not isinstance(exclusion, dict) or not isinstance(exclusion.get("clip_id"), str):
            raise ValueError("Clip loại không hợp lệ.")
        clip_id = exclusion.get("clip_id")
        if clip_id not in known or clip_id in seen or clip_id in excluded_seen:
            raise ValueError("Danh sách loại có clip lặp, không tồn tại hoặc đang được chọn.")
        if not isinstance(exclusion.get("reason"), str) or len(exclusion["reason"]) > 4000:
            raise ValueError("Lý do loại không hợp lệ.")
        excluded_seen.add(clip_id)


def selection_summary(draft: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    validate_draft(draft, catalog)
    rows = {row["clip_id"]: row for row in catalog["rows"]}
    activities = sorted({row["activity"] for row in catalog["rows"]})
    selected = draft["selected"]
    subjects = {
        group: {rows[item["clip_id"]]["subject_id"] for item in selected if item["group"] == group}
        for group in GROUPS
    }
    overlap = sorted(subjects["tuning"] & subjects["evaluation"])
    counts = Counter(item["group"] for item in selected)
    coverage = {
        group: dict(
            Counter(
                rows[item["clip_id"]]["activity"] for item in selected if item["group"] == group
            )
        )
        for group in GROUPS
    }
    bands = dict(Counter(rows[item["clip_id"]]["missingness_band"] for item in selected))
    reviewed = sum(item["reviewed"] for item in selected)
    errors, warnings = [], []
    if len(selected) != draft["target"]:
        errors.append(f"Đang chọn {len(selected)}/{draft['target']} clip; cần đủ số mục tiêu.")
    if overlap:
        errors.append("Trùng mã trẻ giữa hai nhóm: " + ", ".join(overlap))
    if not all(counts[group] for group in GROUPS):
        errors.append("Cần có clip trong cả nhóm chỉnh tham số và nhóm đánh giá.")
    if any(not item["reason"].strip() for item in selected):
        errors.append("Mỗi clip được chọn cần có lý do.")
    if any(not item["reason"].strip() for item in draft["excluded"]):
        errors.append("Mỗi clip bị loại cần có lý do.")
    if reviewed != len(selected):
        errors.append(f"Còn {len(selected) - reviewed} clip chưa được đánh dấu đã xem.")
    for group in GROUPS:
        missing = [activity for activity in activities if not coverage[group].get(activity)]
        if missing:
            warnings.append(f"{group}: chưa phủ {len(missing)} hoạt động: {', '.join(missing)}")
    absent_bands = [band for band in BANDS if not bands.get(band)]
    if absent_bands:
        warnings.append("Chưa phủ đủ ba tầng thiếu khớp: " + ", ".join(absent_bands))
    if not catalog.get("complete_population"):
        warnings.append("Catalogue mới khảo sát một phần dữ liệu; ghi rõ phạm vi khi báo cáo.")
    if any(rows[item["clip_id"]]["source_gap_count"] for item in selected):
        warnings.append("Có clip khuyết file frame nguồn; cần giữ thông tin gap khi phân tích.")
    return {
        "selected": len(selected),
        "target": draft["target"],
        "reviewed": reviewed,
        "groups": {group: counts[group] for group in GROUPS},
        "subjects": {group: len(subjects[group]) for group in GROUPS},
        "subject_overlap": overlap,
        "activity_coverage": coverage,
        "missingness_bands": bands,
        "errors": errors,
        "warnings": warnings,
        "can_finalize": not errors,
    }
