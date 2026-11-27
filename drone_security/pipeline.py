"""The streaming pipeline that wires every component together.

    telemetry ─► TelemetryRules ─────────────────────────────┐
                                                             ▼
    frame ─► FrameAnalyzer ─► EntityTracker ─► DetectionRules ─► AlertManager ─► sinks
                 │                 │                                 │
                 └──── detections ─┴─► LogEntry ──► FrameIndex ◄──────┘
                                              └──► SecurityMemory (LLM context)

``SecurityPipeline.process_frame`` is synchronous and O(detections) so it can
run on the dock's edge computer at video frame rate; the expensive parts (VLM,
LLM agent) are pluggable and optional.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .alerts import AlertManager, AlertSink, jsonl_sink
from .config import SiteConfig, load_config
from .index import FrameIndex
from .memory import SecurityMemory
from .models import Alert, Detection, Frame, LogEntry, Telemetry, hhmm
from .perception import FrameAnalyzer, TextFrameAnalyzer
from .rules import DETECTION_RULES, TelemetryRules
from .tracker import EntityTracker, TrackUpdate

_ACTION_PHRASE = {
    "entering": "entering at", "exiting": "leaving from", "parked": "parked at", "climbing": "climbing at",
    "tampering": "tampering at", "delivering": "delivering at", "loitering": "lingering at",
    "running": "running at", "walking": "moving at", "standing": "standing at",
}
_LOGGED_ACTIONS = {"entering", "exiting", "parked", "climbing", "tampering", "delivering", "loitering", "running"}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


@dataclass
class FrameResult:
    frame: Frame
    detections: list[Detection] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)


@dataclass
class RunStats:
    frames: int = 0
    telemetry: int = 0
    detections: int = 0
    logs: int = 0
    alerts: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


class SecurityPipeline:
    def __init__(self, config: SiteConfig | None = None, analyzer: FrameAnalyzer | None = None,
                 index: FrameIndex | None = None, sinks: list[AlertSink] | None = None,
                 db_path: str | Path = ":memory:", log_sink=None):
        self.config = config or load_config()
        self.analyzer = analyzer or TextFrameAnalyzer(self.config)
        self.index = index or FrameIndex(db_path)
        trk = self.config.tracking
        self.tracker = EntityTracker(visit_gap_s=trk.get("visit_gap_s", 900),
                                     person_reid_window_s=trk.get("person_reid_window_s", 600))
        sinks = list(sinks or [])
        if self.config.alerting.get("jsonl_path"):
            sinks.append(jsonl_sink(self.config.alerting["jsonl_path"]))
        self.alerts = AlertManager(cooldown_s=self.config.alerting.get("cooldown_s", 600), sinks=sinks)
        self.rules = [r(self.config) for r in DETECTION_RULES]
        self.rules = [r for r in self.rules if r.enabled]
        self.telemetry_rules = TelemetryRules(self.config)
        self.memory = SecurityMemory(self.tracker)
        self.logs: list[LogEntry] = []
        self.log_sink = log_sink
        self._last_action: dict[str, str | None] = {}
        self.stats = RunStats()

    # ------------------------------------------------------------------
    def process_telemetry(self, t: Telemetry) -> list[Alert]:
        self.stats.telemetry += 1
        self.index.add_telemetry(t)
        out = [a for c in self.telemetry_rules.evaluate(t) if (a := self.alerts.submit(c))]
        for a in out:
            self.memory.add_alert(a)
        self.index.add_alerts(out)
        self.stats.alerts += len(out)
        return out

    def _log_line(self, det: Detection, upd: TrackUpdate) -> LogEntry | None:
        e = upd.entity
        prev_action = self._last_action.get(e.entity_id)
        self._last_action[e.entity_id] = det.action
        noteworthy_action = det.action in _LOGGED_ACTIONS and det.action != prev_action
        if not (upd.new_visit or upd.zone_changed or noteworthy_action):
            return None
        phrase = _ACTION_PHRASE.get(det.action or "", "spotted at") if noteworthy_action or upd.new_visit \
            else "spotted at"
        if det.obj_type == "package":
            phrase = "left at" if upd.new_visit else "still at"
        msg = f"{det.label} {phrase} {det.zone or 'unknown zone'}, {hhmm(det.ts)}"
        if det.obj_type == "vehicle" and upd.visits_today > 1 and upd.new_visit:
            msg += f" ({_ordinal(upd.visits_today)} visit today)"
        if e.authorized:
            msg += " [authorised]"
        return LogEntry(ts=det.ts, frame_id=det.frame_id, message=msg + ".", obj_type=det.obj_type,
                        entity_id=e.entity_id)

    def process_frame(self, frame: Frame) -> FrameResult:
        if frame.telemetry and frame.zone is None:
            frame.zone = frame.telemetry.zone
        res = FrameResult(frame=frame)
        res.detections = self.analyzer.analyze(frame)
        for det in res.detections:
            authorized = det.obj_type == "vehicle" and self.config.is_authorized_vehicle(
                plate=det.plate, color=det.color, make=det.make, model=det.model, category=det.category)
            upd = self.tracker.update(det, authorized=authorized)
            if (line := self._log_line(det, upd)) is not None:
                res.logs.append(line)
                if self.log_sink:
                    self.log_sink(line)
            for rule in self.rules:
                for cand in rule.evaluate(det, upd, frame):
                    if (a := self.alerts.submit(cand)) is not None:
                        res.alerts.append(a)
        self.index.add_frame(frame, res.detections)
        self.index.add_logs(res.logs)
        self.index.add_alerts(res.alerts)
        self.memory.observe_frame(frame.ts, res.detections, res.logs, res.alerts)
        self.logs.extend(res.logs)
        s = self.stats
        s.frames += 1
        s.detections += len(res.detections)
        s.logs += len(res.logs)
        s.alerts += len(res.alerts)
        return res

    def run(self, records: Iterable[Frame | Telemetry]) -> RunStats:
        t0 = _time.perf_counter()
        for rec in records:
            if isinstance(rec, Telemetry):
                self.process_telemetry(rec)
            else:
                self.process_frame(rec)
        self.index.flush()
        self.stats.seconds += _time.perf_counter() - t0
        return self.stats


# ---------------------------------------------------------------------------
# JSONL stream IO
# ---------------------------------------------------------------------------

def write_stream(records: Iterable[Frame | Telemetry], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for r in records:
            kind = "telemetry" if isinstance(r, Telemetry) else "frame"
            f.write(json.dumps({"type": kind, **r.to_dict()}) + "\n")
            n += 1
    return n


def read_stream(path: str | Path) -> Iterator[Frame | Telemetry]:
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            kind = d.pop("type", "frame")
            yield Telemetry.from_dict(d) if kind == "telemetry" else Frame.from_dict(d)
