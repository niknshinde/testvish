"""Real-video ingestion: sample frames from a video file for the VLM path.

Pipeline: video -> frames at ``fps`` (OpenCV if installed, else the ffmpeg CLI)
-> **change-gated sampling** (skip frames that are near-identical to the last
kept frame, so a static gate at night doesn't cost a VLM call per second)
-> :class:`Frame` objects with ``image_path`` and optional telemetry.

Usage::

    frames = video_to_frames("patrol.mp4", "out/frames", fps=1, start=datetime(2026, 11, 26, 0, 0))
    pipeline = SecurityPipeline(analyzer=ClaudeVisionAnalyzer(cfg))
    pipeline.run(frames)
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .models import Frame, Telemetry


def extract_frames(video: str | Path, out_dir: str | Path, fps: float = 1.0) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video))
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(native / fps)))
        paths, i = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                p = out / f"frame_{len(paths) + 1:06d}.jpg"
                cv2.imwrite(str(p), img)
                paths.append(p)
            i += 1
        cap.release()
        return paths
    except ImportError:
        pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install opencv-python or ffmpeg to ingest real video.")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    str(out / "frame_%06d.jpg")], check=True)
    return sorted(out.glob("frame_*.jpg"))


def _thumb(path: Path, size: int = 32) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size)), dtype=np.float32) / 255.0


def change_gate(paths: list[Path], threshold: float = 0.04, max_skip: int = 30) -> list[int]:
    """Indices of frames worth analysing: mean abs pixel change vs last kept frame > threshold,
    plus at least one frame every ``max_skip`` frames as a heartbeat."""
    keep, last, since = [], None, 0
    for i, p in enumerate(paths):
        t = _thumb(p)
        if last is None or float(np.abs(t - last).mean()) > threshold or since >= max_skip:
            keep.append(i)
            last, since = t, 0
        else:
            since += 1
    return keep


def load_telemetry_csv(path: str | Path, start: datetime) -> list[Telemetry]:
    """CSV columns: t_offset_s,lat,lon,altitude_m,battery_pct,zone[,wind_mps,mode]."""
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            rows.append(Telemetry(ts=start + timedelta(seconds=float(r["t_offset_s"])), lat=float(r["lat"]),
                                  lon=float(r["lon"]), altitude_m=float(r["altitude_m"]),
                                  battery_pct=float(r["battery_pct"]), zone=r.get("zone") or None,
                                  wind_mps=float(r.get("wind_mps") or 0), mode=r.get("mode") or "patrol"))
    return rows


def video_to_frames(video: str | Path, out_dir: str | Path, fps: float = 1.0, start: datetime | None = None,
                    telemetry: list[Telemetry] | None = None, gate: bool = True,
                    default_zone: str | None = None) -> list[Frame]:
    start = start or datetime.now().replace(microsecond=0)
    paths = extract_frames(video, out_dir, fps)
    idx = change_gate(paths) if gate else list(range(len(paths)))
    tel = sorted(telemetry or [], key=lambda t: t.ts)
    frames = []
    for n, i in enumerate(idx, 1):
        ts = start + timedelta(seconds=i / fps)
        t = min(tel, key=lambda x: abs((x.ts - ts).total_seconds())) if tel else None
        frames.append(Frame(frame_id=i + 1, ts=ts, description="", zone=(t.zone if t else default_zone),
                            telemetry=t, image_path=str(paths[i])))
    return frames
