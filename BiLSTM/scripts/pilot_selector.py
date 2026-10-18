"""Run the local visual MMASD pilot selector with durable, versioned exports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
from export_tracking_review_3d import OPENPOSE_25_EDGES, SMPL_JOINT_NAMES, generate_multimodal_html
from pilot_catalog import (
    DEFAULT_OUTPUT,
    REPO,
    load_catalog,
    read_observation,
    source_digest,
    verify_pipeline,
)
from pilot_sampling import propose, selection_summary, validate_draft
from romp_review_data import match_selected_person, read_romp_frames
from src.data.openpose_parser import BODY_25_NAMES
from src.reproducibility import sha256_file
from src.utils import cfg_path, load_config

ASSETS = REPO / "apps/pilot_selector"


class ConflictError(ValueError):
    """A saved draft or source input changed since the client loaded it."""


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["clip_id", "group"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        safe = {
            key: "'" + value
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
            else value
            for key, value in row.items()
        }
        writer.writerow(safe)
    return "\ufeff" + stream.getvalue()


class PilotStore:
    def __init__(self, directory: Path, check_pipeline: bool = True) -> None:
        self.directory = directory
        self.catalog = load_catalog(directory / "catalog.json")
        if check_pipeline:
            verify_pipeline(self.catalog)
        self.config = load_config(REPO / self.catalog["config_path"])
        self.rows = {row["clip_id"]: row for row in self.catalog["rows"]}
        self.state_dir = directory / "selections" / self.catalog["catalog_sha256"][:16]
        self.draft_path = self.state_dir / "draft.json"
        self.lock = threading.RLock()
        if self.draft_path.exists():
            envelope = json.loads(self.draft_path.read_text())
            validate_draft(envelope["draft"], self.catalog)
        else:
            draft = propose(self.catalog)
            write_json_atomic(self.draft_path, {"revision": 0, "saved_at": now(), "draft": draft})

    def state(self) -> dict[str, Any]:
        with self.lock:
            envelope = json.loads(self.draft_path.read_text())
            envelope["summary"] = selection_summary(envelope["draft"], self.catalog)
            return envelope

    def save(self, draft: dict[str, Any], expected_revision: int) -> dict[str, Any]:
        validate_draft(draft, self.catalog)
        with self.lock:
            current = self.state()
            if type(expected_revision) is not int or expected_revision != current["revision"]:
                raise ConflictError(
                    "Bản nháp đã được thay đổi ở cửa sổ khác. Tải lại trang trước khi lưu."
                )
            write_json_atomic(
                self.draft_path,
                {"revision": expected_revision + 1, "saved_at": now(), "draft": draft},
            )
            return self.state()

    def observation(self, clip_id: str) -> tuple:
        if clip_id not in self.rows or self.rows[clip_id]["status"] != "ready":
            raise ValueError("Clip không có trong catalogue khả dụng.")
        verify_pipeline(self.catalog)
        observed = read_observation(self.rows[clip_id], self.config)
        if source_digest(observed[1]) != self.rows[clip_id]["source_sha256"]:
            raise ConflictError(
                "Dữ liệu clip đã thay đổi; cần tạo lại catalogue trước khi sử dụng."
            )
        return observed

    def preview(self, clip_id: str) -> dict[str, Any]:
        # Always validate source content; the UI requests a preview once per clip.
        frames, _, tracking, raw, valid, ids = self.observation(clip_id)
        threshold = self.catalog["confidence_threshold"]
        scale = self.rows[clip_id]["reference_body_diagonal"]
        events = []
        for index in range(len(frames)):
            if tracking.selected_indices[index] < 0:
                events.append({"frame": index, "kind": "target_missing", "joints": []})
            if index and ids[index] != ids[index - 1] + 1:
                events.append({"frame": index, "kind": "source_gap", "joints": []})
            if index and scale:
                pair = valid[index] & valid[index - 1]
                if ids[index] != ids[index - 1] + 1:
                    continue
                step = np.linalg.norm(raw[index, :, :2] - raw[index - 1, :, :2], axis=1) / scale
                joints = np.flatnonzero(pair & (step > 0.35)).tolist()
                if joints:
                    events.append({"frame": index, "kind": "large_step", "joints": joints})
        people = [
            [
                [
                    [float(value) if np.isfinite(value) else None for value in point]
                    for point in person
                ]
                for person in frame
            ]
            for frame in frames
        ]
        payload = {
            "clip_id": clip_id,
            "frames": people,
            "source_frame_ids": ids,
            "selected_indices": tracking.selected_indices,
            "events": events,
            "edges": OPENPOSE_25_EDGES,
            "joint_names": BODY_25_NAMES,
            "threshold": threshold,
            "source_sha256": self.rows[clip_id]["source_sha256"],
        }
        return payload

    def viewer(self, clip_id: str) -> str:
        frames, _, tracking, _, _, ids = self.observation(clip_id)
        row = self.rows[clip_id]
        root = cfg_path(self.config, "data", "root", default="../data_MMASD")
        directory = root / "3D_skeleton/ROMP_3D_Coordinates" / row["activity"] / clip_id
        romp, _ = read_romp_frames(directory)
        frames_3d = [romp.get(index, []) for index in ids]
        matches = [
            match_selected_person(person, candidates, self.catalog["confidence_threshold"])
            for person, candidates in zip(tracking.selected_people, frames_3d)
        ]
        frames_2d = [
            [
                [
                    point.astype(float).tolist() if np.isfinite(point).all() else None
                    for point in person
                ]
                for person in frame
            ]
            for frame in frames
        ]
        return generate_multimodal_html(
            {
                **row,
                "num_frames": len(frames),
                "frames_2d": frames_2d,
                "frames_3d": frames_3d,
                "selected_indices": tracking.selected_indices,
                "selected_indices_3d": [match["index"] for match in matches],
                "person_matches": matches,
                "source_frame_ids": ids,
                "joint_names": SMPL_JOINT_NAMES,
                "threshold": self.catalog["confidence_threshold"],
                "schema_version": "mmasd_3d_review_v3",
            }
        )

    def finalize(self, expected_revision: int) -> dict[str, Any]:
        with self.lock:
            envelope = self.state()
            if expected_revision != envelope["revision"]:
                raise ConflictError("Bản nháp đã đổi; tải lại trước khi chốt.")
            draft, summary = envelope["draft"], envelope["summary"]
            if not summary["can_finalize"]:
                raise ValueError("Chưa thể chốt: " + " ".join(summary["errors"]))
            verify_pipeline(self.catalog)
            for item in draft["selected"]:
                self.observation(item["clip_id"])
            export_id = (
                datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S_")
                + uuid.uuid4().hex[:8]
            )
            directory = self.state_dir / "exports" / export_id
            rows = []
            for item in sorted(
                draft["selected"], key=lambda value: (value["group"], value["clip_id"])
            ):
                row = self.rows[item["clip_id"]]
                source_path = (
                    cfg_path(self.config, "data", "root", default="../data_MMASD")
                    / self.config["data"]["skeleton_dir"]
                    / row["activity"]
                    / row["clip_id"]
                ).resolve()
                manifest_path = (
                    str(source_path.relative_to(REPO))
                    if source_path.is_relative_to(REPO)
                    else str(source_path)
                )
                rows.append(
                    {
                        "clip_id": row["clip_id"],
                        "group": item["group"],
                        "subject_id": row["subject_id"],
                        "session_id": row["session_id"],
                        "session_group": row["subject_id"] + "::" + row["session_id"],
                        "activity": row["activity"],
                        "activity_id": row["activity_id"],
                        "num_frames": row["num_frames"],
                        "path": manifest_path,
                        "missing_joint_ratio": row["missing_joint_ratio"],
                        "mean_confidence": row["mean_confidence"],
                        "max_missing_run": row["max_missing_run"],
                        "source_gap_count": row["source_gap_count"],
                        "reason": item["reason"],
                        "human_reviewed": item["reviewed"],
                        "source_sha256": row["source_sha256"],
                        "config_sha256": self.catalog["config_sha256"],
                        "catalog_sha256": self.catalog["catalog_sha256"],
                    }
                )
            export = {
                "schema_version": "mmasd_pilot_export_v1",
                "created_at": now(),
                "draft_revision": envelope["revision"],
                "selection": draft,
                "summary": summary,
                "config": {
                    key: value for key, value in self.config.items() if not key.startswith("_")
                },
                "catalog_sha256": self.catalog["catalog_sha256"],
                "tool_sha256": {
                    str(path.relative_to(REPO)): sha256_file(path)
                    for path in [
                        Path(__file__),
                        Path(__file__).with_name("pilot_sampling.py"),
                        *sorted(ASSETS.glob("*")),
                    ]
                    if path.is_file()
                },
                "interpretation": "Visual review is not verified child identity or pose ground truth. Pilot is exploratory; no classifier test result is implied.",
                "manifest": rows,
            }
            write_json_atomic(directory / "selection.json", export)
            for name, subset in [
                ("manifest.csv", rows),
                ("tuning.csv", [row for row in rows if row["group"] == "tuning"]),
                ("evaluation.csv", [row for row in rows if row["group"] == "evaluation"]),
            ]:
                (directory / name).write_text(csv_text(subset), encoding="utf-8", newline="")
            return {
                "export_id": export_id,
                "summary": summary,
                "files": {
                    name: f"/api/exports/{export_id}/{name}"
                    for name in ("selection.json", "manifest.csv", "tuning.csv", "evaluation.csv")
                },
            }


def make_handler(store: PilotStore):
    class Handler(BaseHTTPRequestHandler):
        def respond(
            self,
            status: int,
            body: Any,
            mime: str = "application/json; charset=utf-8",
            download: str | None = None,
        ) -> None:
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False)
            content = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path in {"/", "/app.js", "/style.css"}:
                    filename = "index.html" if path == "/" else path[1:]
                    mime = {
                        "index.html": "text/html",
                        "app.js": "application/javascript",
                        "style.css": "text/css",
                    }[filename]
                    self.respond(200, (ASSETS / filename).read_bytes(), mime + "; charset=utf-8")
                elif path == "/api/catalog":
                    self.respond(200, store.catalog)
                elif path == "/api/state":
                    self.respond(200, store.state())
                elif path.startswith("/api/preview/"):
                    self.respond(200, store.preview(path[len("/api/preview/") :]))
                elif path.startswith("/viewer/"):
                    self.respond(
                        200, store.viewer(path[len("/viewer/") :]), "text/html; charset=utf-8"
                    )
                elif path.startswith("/api/exports/"):
                    parts = path.split("/")
                    if (
                        len(parts) != 5
                        or not re.fullmatch(r"[0-9a-f_]+", parts[3])
                        or parts[4]
                        not in {"selection.json", "manifest.csv", "tuning.csv", "evaluation.csv"}
                    ):
                        raise ValueError("Tệp xuất không hợp lệ.")
                    target = store.state_dir / "exports" / parts[3] / parts[4]
                    self.respond(200, target.read_bytes(), "application/octet-stream", parts[4])
                else:
                    self.respond(404, {"error": "Không tìm thấy trang."})
            except ConflictError as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, OSError, KeyError) as error:
                self.respond(400, {"error": str(error)})

        def do_POST(self) -> None:
            try:
                origin = self.headers.get("Origin")
                allowed = {
                    f"http://127.0.0.1:{self.server.server_port}",
                    f"http://localhost:{self.server.server_port}",
                }
                if origin and origin not in allowed:
                    self.respond(403, {"error": "Yêu cầu phải đến từ giao diện cục bộ."})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000 or not self.headers.get(
                    "Content-Type", ""
                ).startswith("application/json"):
                    raise ValueError("Yêu cầu JSON không hợp lệ.")
                payload = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/draft":
                    self.respond(200, store.save(payload["draft"], payload["expected_revision"]))
                elif path == "/api/propose":
                    draft = propose(
                        store.catalog,
                        payload["target"],
                        payload["seed"],
                        payload.get("excluded", []),
                    )
                    self.respond(
                        200, {"draft": draft, "summary": selection_summary(draft, store.catalog)}
                    )
                elif path == "/api/finalize":
                    self.respond(200, store.finalize(payload["expected_revision"]))
                else:
                    self.respond(404, {"error": "Không tìm thấy thao tác."})
            except ConflictError as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"Pilot HTTP: {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if not (args.data_dir / "catalog.json").exists():
        parser.error("Chưa có catalogue. Chạy make pilot-catalog trước.")
    store = PilotStore(args.data_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Pilot selector: http://127.0.0.1:{server.server_port}", flush=True)
    print("Bản nháp được lưu trên ổ đĩa; Ctrl+C để dừng.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
