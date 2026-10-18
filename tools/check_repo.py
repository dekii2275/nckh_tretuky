#!/usr/bin/env python3
"""Check Python syntax and documentation links in a fresh checkout."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = (
    "BiLSTM/src",
    "BiLSTM/scripts",
    "BiLSTM/tests",
    "tools",
    "tests",
    "autism",
)
# These locations may be absent from a fresh clone: local inputs/checkpoints
# are ignored, and the MMASD submodule is optional until initialized.
LOCAL_ONLY_DIRS = tuple(Path(name) for name in (
    "data_MMASD", "Data_DRYAD", "data_Dream",
    "checkpoints", "runs", "BiLSTM/checkpoints", "BiLSTM/runs",
    "BiLSTM/artifacts/tracking_review_3d", "artifacts/visualizations",
    "artifacts/dryad_msg3d/experiments",
    "MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis",
))
LINK = re.compile(r"\]\(([^)]+)\)")


def main() -> None:
    errors: list[str] = []
    python_files = sorted({path for name in CODE_DIRS for path in (ROOT / name).rglob("*.py")})
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Python: {path.relative_to(ROOT)}: {exc}")

    docs = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for section in ("plans", "reports"):
        docs.extend(sorted((ROOT / "docs" / section).glob("*.md")))
    docs = list(dict.fromkeys(docs))
    for path in docs:
        if not path.is_file():
            errors.append(f"Thiếu tài liệu: {path.relative_to(ROOT)}")
            continue
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            target = unquote(match.group(1).split("#", 1)[0].strip("<> "))
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            try:
                relative = resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"Link ngoài repo: {path.relative_to(ROOT)} -> {target}")
                continue
            if any(relative == local or local in relative.parents for local in LOCAL_ONLY_DIRS):
                continue
            if not resolved.exists():
                errors.append(f"Link hỏng: {path.relative_to(ROOT)} -> {target}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(python_files)} file Python, {len(docs)} README/tài liệu, link cục bộ hợp lệ")


if __name__ == "__main__":
    main()
