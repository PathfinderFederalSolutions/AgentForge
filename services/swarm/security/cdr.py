from __future__ import annotations
import subprocess
import tempfile
import os
from typing import Tuple

def _run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return p.returncode, out.decode(), err.decode()

def cdr_media(input_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_cdr.mp4")
    rc, _, err = _run(["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264", "-c:a", "aac", "-map_metadata", "-1", out_path])
    if rc != 0:
        raise RuntimeError(f"FFmpeg CDR failed: {err}")
    return out_path

def cdr_image(input_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_cdr.png")
    rc, _, err = _run(["convert", input_path, "-strip", out_path])
    if rc != 0:
        raise RuntimeError(f"ImageMagick CDR failed: {err}")
    return out_path

def cdr_docs(input_path: str) -> str:
    # Apache Tika text extract (read-only sanitization)
    rc, out, err = _run(["tika", "-t", input_path])
    if rc != 0:
        raise RuntimeError(f"Tika extract failed: {err}")
    return out