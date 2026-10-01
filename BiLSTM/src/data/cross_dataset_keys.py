"""Deterministic identifiers shared by the local MMASD modalities.

The local OpenPose tree uses the original clip basename directly, while the
MMASD+ exports prepend ``processed_`` and append person/output tokens.  This
module deliberately parses only the stable original-clip component.  It does
not assign a child/therapist role and it does not treat an output suffix as a
verified person identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ACTIVITY_PREFIXES = ("as", "bs", "ce", "dr", "fg", "mfs", "ms", "sac", "sq", "tr", "tw")

ACTIVITY_NAMES = {
    "as": "arm_swing_as",
    "bs": "body_swing_bs",
    "ce": "chest_expansion_ce",
    "dr": "drumming_dr",
    "fg": "frog_pose_fg",
    "mfs": "maracas_forward_shaking_mfs",
    "ms": "maracas_shaking_ms",
    "sac": "sing_and_clap_sac",
    "sq": "squat_sq",
    "tr": "tree_pose_tr",
    "tw": "twist_pose_tw",
}

# This is intentionally a search expression rather than an OpenPose-only
# anchored expression: MMASD+ has e.g. ``processed_Arm_swingas_...``.
CLIP_KEY_RE = re.compile(
    r"(?P<prefix>as|bs|ce|dr|fg|mfs|ms|sac|sq|tr|tw)_"
    r"(?P<subject_id>\d+)_"
    r"(?P<session_id>D(?:1|8|16))_"
    r"(?P<clip_index>\d{3})_"
    r"(?P<state>[iny])"
)

PLUS_OUTPUT_TAIL_RE = re.compile(r"^_(?P<person_token>\d+)_(?P<label_token>\d+)$")


@dataclass(frozen=True)
class CanonicalClip:
    """The stable original-clip component parsed from a path/name."""

    key: str
    prefix: str
    activity: str
    subject_id: str
    session_id: str
    clip_index: str
    state: str
    match_start: int
    match_end: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "key": self.key,
            "prefix": self.prefix,
            "activity": self.activity,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "clip_index": self.clip_index,
            "state": self.state,
            "match_start": self.match_start,
            "match_end": self.match_end,
        }


@dataclass(frozen=True)
class PlusNameParse:
    """Canonical clip plus conservatively parsed output-name metadata."""

    clip: CanonicalClip | None
    person_token: str
    label_token: str
    tail: str
    variant: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip.to_dict() if self.clip else None,
            "person_token": self.person_token,
            "label_token": self.label_token,
            "tail": self.tail,
            "variant": self.variant,
            "reason": self.reason,
        }


def _canonical_from_match(match: re.Match[str]) -> CanonicalClip:
    groups = match.groupdict()
    prefix = groups["prefix"]
    key = match.group(0)
    return CanonicalClip(
        key=key,
        prefix=prefix,
        activity=ACTIVITY_NAMES[prefix],
        subject_id=groups["subject_id"],
        session_id=groups["session_id"],
        clip_index=groups["clip_index"],
        state=groups["state"],
        match_start=match.start(),
        match_end=match.end(),
    )


def parse_clip_key(value: str | Path, *, search: bool = False) -> CanonicalClip | None:
    """Parse a clip key from a directory/file name.

    ``search=False`` is used for OpenPose directory names and requires the
    complete name to be exactly one key.  ``search=True`` is used for MMASD+
    processed outputs, where the key is embedded in a longer name.
    """

    text = Path(value).name if isinstance(value, Path) else str(value)
    match = CLIP_KEY_RE.search(text) if search else CLIP_KEY_RE.fullmatch(text)
    return _canonical_from_match(match) if match else None


def parse_plus_output_name(value: str | Path) -> PlusNameParse:
    """Parse a MMASD+ output name without interpreting role semantics.

    The parser records formatting anomalies instead of normalising the raw
    path.  A trailing ``_<integer>_<integer>`` is exposed as output metadata
    only when it is the complete tail after the canonical key.  Additional
    tokens, copied-file suffixes, and whitespace make the output metadata
    ``UNKNOWN`` while retaining the canonical clip key when available.
    """

    text = Path(value).stem if isinstance(value, Path) else Path(str(value)).stem
    clip = parse_clip_key(text, search=True)
    if clip is None:
        return PlusNameParse(None, "", "", "", "", "NO_CANONICAL_CLIP_KEY")

    tail = text[clip.match_end :]
    compact_tail = re.sub(r"\s+", "", tail)
    variant_parts: list[str] = []
    if tail != compact_tail:
        variant_parts.append("WHITESPACE_IN_TAIL")

    without_copy_suffix = compact_tail
    if without_copy_suffix.endswith("(1)"):
        without_copy_suffix = without_copy_suffix[: -len("(1)")]
        variant_parts.append("COPY_SUFFIX_(1)")

    match = PLUS_OUTPUT_TAIL_RE.fullmatch(without_copy_suffix)
    if match:
        variant = "+".join(variant_parts) if variant_parts else "NORMAL"
        return PlusNameParse(
            clip,
            match.group("person_token"),
            match.group("label_token"),
            tail,
            variant,
            "PARSED_OUTPUT_TAIL",
        )

    if compact_tail == "":
        return PlusNameParse(clip, "", "", tail, "MISSING_OUTPUT_TAIL", "NO_OUTPUT_TAIL")
    variant_parts.append("NONSTANDARD_OUTPUT_TAIL")
    return PlusNameParse(
        clip,
        "",
        "",
        tail,
        "+".join(variant_parts),
        "CANONICAL_CLIP_FOUND_OUTPUT_TAIL_UNPARSED",
    )


def clip_group_key(clip: CanonicalClip) -> tuple[str, str, str]:
    """Return the safe metadata key used only for candidate discovery."""

    return clip.prefix, clip.subject_id, clip.session_id


def output_basename(value: str | Path) -> str:
    """Return a stable display name for joining CSV/MP4/ROMP outputs."""

    text = Path(value).stem if isinstance(value, Path) else Path(str(value)).stem
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" (1)"):
        text = text[: -len(" (1)")]
    return text
