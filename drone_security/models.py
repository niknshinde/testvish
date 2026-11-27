"""Core data model shared by every stage of the pipeline.

Everything is a plain dataclass so it serialises cleanly to JSON (for logs,
the frame index and the LLM context) and is trivial to construct in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    """Alert severity. IntEnum so severities can be compared and sorted."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: "str | int | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).strip().upper()]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


def hhmm(ts: datetime) -> str:
    return ts.strftime("%H:%M")


@dataclass
class Telemetry:
    """One telemetry sample from the docked drone."""

    ts: datetime
    drone_id: str = "drone-1"
    lat: float = 0.0
    lon: float = 0.0
    altitude_m: float = 0.0
    battery_pct: float = 100.0
    zone: str | None = None          # zone the camera is pointing at / hovering over
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    wind_mps: float = 0.0
    mode: str = "patrol"             # patrol | docked | returning | investigating

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Telemetry":
        d = dict(d)
        d["ts"] = _parse_ts(d["ts"])
        return cls(**d)


@dataclass
class Frame:
    """A video frame. In simulation the pixels are replaced by a text description.

    ``image_path`` is set when the frame comes from a real video; a VLM then
    produces ``caption`` (and detections) from the pixels.
    """

    frame_id: int
    ts: datetime
    description: str
    zone: str | None = None
    telemetry: Telemetry | None = None
    image_path: str | None = None
    caption: str | None = None       # VLM caption (real video) or == description (simulation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ts": self.ts.isoformat(),
            "description": self.description,
            "zone": self.zone,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "image_path": self.image_path,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Frame":
        tel = d.get("telemetry")
        return cls(
            frame_id=int(d["frame_id"]),
            ts=_parse_ts(d["ts"]),
            description=d["description"],
            zone=d.get("zone"),
            telemetry=Telemetry.from_dict(tel) if tel else None,
            image_path=d.get("image_path"),
            caption=d.get("caption"),
        )


@dataclass
class Detection:
    """A single object found in a frame, with the attributes we care about."""

    frame_id: int
    ts: datetime
    obj_type: str                    # person | vehicle | animal | package | object
    category: str | None = None      # truck, sedan, van, ... / dog, deer, ...
    color: str | None = None
    make: str | None = None
    model: str | None = None
    plate: str | None = None
    action: str | None = None        # entering, exiting, parked, loitering, climbing, ...
    zone: str | None = None
    attributes: list[str] = field(default_factory=list)   # hoodie, backpack, flashlight, ...
    confidence: float = 0.9
    entity_id: str | None = None     # filled in by the tracker
    source_text: str = ""            # the clause the detection was parsed from

    @property
    def label(self) -> str:
        """Human readable label, e.g. 'Blue Ford F150' or 'Person in dark hoodie'."""
        if self.obj_type == "vehicle":
            # make/model are stored in canonical casing ("Ford", "F150", "BMW").
            parts = [self.color, self.make, self.model]
            if not (self.make or self.model):
                parts.append(self.category or "vehicle")
            text = " ".join(p for p in parts if p).strip()
            return (text[:1].upper() + text[1:]) if text else "Vehicle"
        if self.obj_type == "person":
            desc = ", ".join(self.attributes[:2])
            return f"Person ({desc})" if desc else "Person"
        if self.obj_type == "animal":
            return (self.category or "animal").title()
        if self.obj_type == "package":
            return "Package"
        return (self.category or self.obj_type).title()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["label"] = self.label
        return d


@dataclass
class LogEntry:
    """A contextual log line for future reference ('Blue Ford F150 spotted at garage, 12:00.')."""

    ts: datetime
    frame_id: int
    message: str
    obj_type: str
    entity_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


@dataclass
class Alert:
    alert_id: str
    ts: datetime
    rule_id: str
    severity: Severity
    message: str
    frame_id: int | None = None
    zone: str | None = None
    entity_id: str | None = None
    recommendation: str | None = None
    details: str | None = None
    evidence_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["severity"] = self.severity.name
        return d


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
