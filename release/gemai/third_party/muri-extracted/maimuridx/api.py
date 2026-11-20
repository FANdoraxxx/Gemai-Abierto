"""A standard-library-only, process-isolated API around unchanged MaiMuriDX."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

UPSTREAM_COMMIT = "0ebab7ff6853a159ef6af76d9e3c43b23157bb5e"
_ROOT = Path(__file__).resolve().parent
_FIELD = re.compile(r"&([A-Za-z0-9_]+)[ \t]*=")


class DetectionError(RuntimeError):
    """The original detector failed or timed out."""


def looks_like_maidata(text: str) -> bool:
    """Require an actual leading field, ignoring blank lines and Simai comments."""
    leading = re.sub(r"\|\|[^\r\n]*", "", text).lstrip("\ufeff \t\r\n")
    return bool(re.match(r"&[A-Za-z0-9_]+[ \t]*=", leading))


def parse_maidata(text: str) -> dict[str, str]:
    """Return last-occurrence maidata fields, preserving chart-body newlines.

    An ampersand starts a field, matching the upstream maidata convention.
    The return values include all text up to the following field. This retains
    the upstream per-chart line/column convention for a selected inote block.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = text.lstrip("\ufeff")
    matches = list(_FIELD.finditer(text))
    return {
        match.group(1): text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        for i, match in enumerate(matches)
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def detect_simai(
    text: str,
    *,
    first: float = 0.0,
    config: Mapping[str, Any] | None = None,
    hand_radius: float | None = None,
    timeout: float | None = 120.0,
) -> dict[str, Any]:
    """Detect a raw Simai string and return static/dynamic JSON and report text.

    ``first`` is seconds. ``hand_radius`` overrides only hand_radius_normal,
    in the upstream 1080-pixel coordinate system; the default remains 60.
    ``config`` is a partial override of upstream/config.json.
    Each call starts its own interpreter, so upstream module globals and report
    buffers cannot leak between calls or into the caller's Python modules.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    request = {
        "text": text,
        "first": _finite_number(first, "first"),
        "config": dict(config) if config is not None else {},
    }
    if hand_radius is not None:
        request["config"]["hand_radius_normal"] = _finite_number(hand_radius, "hand_radius")
    if timeout is not None:
        timeout = _finite_number(timeout, "timeout")
        if timeout <= 0:
            raise ValueError("timeout must be positive or None")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(_ROOT / "_worker.py")],
            input=json.dumps(request, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        raise DetectionError(f"MaiMuriDX detection exceeded {timeout} seconds") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise DetectionError(f"Cannot run MaiMuriDX: {exc}") from exc
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise DetectionError(f"MaiMuriDX worker returned no valid JSON: {detail}") from exc
    if completed.returncode or "error" in result:
        raise DetectionError(result.get("error") or completed.stderr.strip())
    return result


def detect_file(
    path: str | os.PathLike[str],
    *,
    difficulty: int | str | None = None,
    first: float | None = None,
    format: str = "auto",
    config: Mapping[str, Any] | None = None,
    hand_radius: float | None = None,
    timeout: float | None = 120.0,
) -> dict[str, Any]:
    """Read UTF-8/BOM Simai or maidata.txt without any interactive prompts.

    For maidata, difficulty is the numeric inote suffix (1..7 convention).
    Multiple nonempty difficulties require an explicit selection.
    An explicit first overrides maidata first_<difficulty>, then global first.
    A directory argument selects that directory's maidata.txt.
    """
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source /= "maidata.txt"
    text = source.read_text(encoding="utf-8-sig")
    if format not in {"auto", "simai", "maidata"}:
        raise ValueError("format must be auto, simai, or maidata")
    fields = parse_maidata(text) if format != "simai" else {}
    is_maidata = format == "maidata" or (
        format == "auto"
        and (source.name.lower() == "maidata.txt" or looks_like_maidata(text))
    )
    extra: dict[str, Any] = {"source": str(source), "format": "maidata" if is_maidata else "simai"}
    if is_maidata:
        available = sorted(
            {key[6:] for key, value in fields.items() if re.fullmatch(r"inote_\d+", key) and value.strip()},
            key=int,
        )
        if not available:
            raise ValueError("maidata has no nonempty &inote_<difficulty> chart")
        if difficulty is None:
            if len(available) > 1:
                raise ValueError("Choose --difficulty; available difficulties: " + ", ".join(available))
            selected = available[0]
        else:
            selected = str(difficulty)
            if selected not in available:
                raise ValueError(f"Difficulty {selected!r} is unavailable; choose from: {', '.join(available)}")
        text = fields["inote_" + selected]
        first_key = "first_" + selected if "first_" + selected in fields else "first"
        selected_first = _finite_number(fields.get(first_key, "0").strip(), first_key) if first is None else first
        extra.update(
            difficulty=selected,
            available_difficulties=available,
            first_source=first_key if first is None else "override",
            title=fields.get("title", "").strip(),
        )
    else:
        if difficulty is not None:
            raise ValueError("difficulty applies only to maidata")
        selected_first = 0.0 if first is None else first
        extra["first_source"] = "default" if first is None else "override"
    result = detect_simai(text, first=selected_first, config=config, hand_radius=hand_radius, timeout=timeout)
    result["metadata"].update(extra)
    return result
