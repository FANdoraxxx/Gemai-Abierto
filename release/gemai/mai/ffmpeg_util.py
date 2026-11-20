"""Resolve the FFmpeg executable without relying solely on the process PATH."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def ffmpeg_exe() -> str:
    candidates = [os.environ.get("MAI_FFMPEG"), shutil.which("ffmpeg"),
                  r"C:\ffmpeg\bin\ffmpeg.exe"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "找不到 ffmpeg。请设置 MAI_FFMPEG=C:\\ffmpeg\\bin\\ffmpeg.exe，"
        "或把包含 ffmpeg.exe 的目录加入 PATH。")
