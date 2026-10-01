#!/usr/bin/env python3
"""Read-only syntax and local documentation-link check for this repository."""

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
LINK = re.compile(r"\]\(([^)]+)\)")


def main() -> None:
    errors: list[str] = []
    python_files = sorted({path for name in CODE_DIRS for path in (ROOT / name).rglob("*.py")})
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Python: {path.relative_to(ROOT)}: {exc}")

    docs = [ROOT / "README.md", ROOT / "docs/README.md"]
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
            if not (path.parent / target).exists():
                errors.append(f"Link hỏng: {path.relative_to(ROOT)} -> {target}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(python_files)} file Python, {len(docs)} README/tài liệu, link cục bộ hợp lệ")


if __name__ == "__main__":
    main()
