"""Entity tracking / re-identification across frames.

The tracker turns a stream of independent detections into *entities* with
history: "the blue Ford F150 has made 2 visits today", "this person has been at
the main gate for 4 minutes". That history is what makes rules like loitering
and repeat-vehicle possible and is the backbone of the agent's long-term context.

Re-identification is attribute based (this is a prototype without appearance
embeddings):

* vehicles: licence plate if read, otherwise (colour, make, model, category);
* people: clothing/carried-item attributes, but only within a time window
  (clothing is a weak identifier across hours);
* animals / packages / objects: (category, zone) within a short window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import Detection, hhmm

# Only clothing is used to re-identify people; carried items (flashlight, crowbar) come and go.
CLOTHING = {"hoodie", "jacket", "vest", "coat", "shirt", "uniform", "hat", "cap", "helmet", "hooded", "hi-vis"}


@dataclass
class Visit:
    start: datetime
    last_seen: datetime
    zones: list[str] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    zone_since: datetime | None = None      # when the entity arrived in its current zone

    @property
    def duration_s(self) -> float:
        return (self.last_seen - self.start).total_seconds()


@dataclass
class Entity:
    entity_id: str
    obj_type: str
    label: str
    signature: str
    first_seen: datetime
    last_seen: datetime
    current_zone: str | None = None
    visits: list[Visit] = field(default_factory=list)
    detections: int = 0
    authorized: bool = False
    attributes: set[str] = field(default_factory=set)

    @property
    def current_visit(self) -> Visit:
        return self.visits[-1]

    def visits_on(self, day: datetime) -> int:
        return sum(1 for v in self.visits if v.start.date() == day.date())

    def dwell_s(self, now: datetime) -> float:
        """Seconds continuously spent in the current zone during the current visit."""
        v = self.current_visit
        return (now - (v.zone_since or v.start)).total_seconds()

    def summary(self) -> str:
        times = ", ".join(hhmm(v.start) for v in self.visits[:6])
        return (f"{self.entity_id}: {self.label} - {len(self.visits)} visit(s) at {times}; "
                f"zones: {', '.join(sorted({z for v in self.visits for z in v.zones}))}")

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id, "obj_type": self.obj_type, "label": self.label,
            "first_seen": self.first_seen.isoformat(), "last_seen": self.last_seen.isoformat(),
            "visits": [{"start": v.start.isoformat(), "end": v.last_seen.isoformat(),
                        "zones": v.zones, "frames": v.frames[:50]} for v in self.visits],
            "detections": self.detections, "authorized": self.authorized,
        }


@dataclass
class TrackUpdate:
    entity: Entity
    new_entity: bool
    new_visit: bool
    zone_changed: bool
    dwell_s: float
    visits_today: int


class EntityTracker:
    def __init__(self, visit_gap_s: float = 900, person_reid_window_s: float = 600,
                 misc_window_s: float = 300):
        self.visit_gap = timedelta(seconds=visit_gap_s)
        self.person_window = timedelta(seconds=person_reid_window_s)
        self.misc_window = timedelta(seconds=misc_window_s)
        self.entities: dict[str, Entity] = {}
        self._by_signature: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def signature(det: Detection) -> str:
        if det.obj_type == "vehicle":
            if det.plate:
                return f"vehicle|plate:{det.plate.upper()}"
            return "vehicle|" + "|".join(str(x or "").lower() for x in (det.color, det.make, det.model, det.category))
        if det.obj_type == "person":
            attrs = sorted(a for a in det.attributes if a.split()[-1] in CLOTHING)
            return "person|" + (",".join(attrs) or f"anon@{det.zone}")
        return f"{det.obj_type}|{det.category}|{det.zone}"

    def _window_for(self, det: Detection) -> timedelta | None:
        if det.obj_type == "vehicle":
            return None                      # vehicles are re-identified all day
        if det.obj_type == "person":
            # clothing-described people are re-identified within the day; anonymous ones briefly
            return self.person_window if "|anon@" in self.signature(det) else timedelta(hours=12)
        if det.obj_type == "package":
            return timedelta(hours=24)       # a package stays where it was left
        return self.misc_window

    def _new_id(self, obj_type: str) -> str:
        self._counters[obj_type] = self._counters.get(obj_type, 0) + 1
        return f"{obj_type}-{self._counters[obj_type]:03d}"

    def _match_plate_less(self, det: Detection) -> Entity | None:
        """A vehicle seen once with a plate and once without should still match."""
        if det.obj_type != "vehicle" or det.plate or not (det.make or det.model):
            return None
        for e in self.entities.values():
            if e.obj_type == "vehicle" and e.signature.startswith("vehicle|plate:") and e.label == det.label:
                return e
        return None

    def update(self, det: Detection, authorized: bool = False) -> TrackUpdate:
        sig = self.signature(det)
        ent = None
        eid = self._by_signature.get(sig)
        if eid:
            cand = self.entities[eid]
            window = self._window_for(det)
            if window is None or det.ts - cand.last_seen <= window:
                ent = cand
        if ent is None and det.obj_type == "person" and sig.startswith("person|anon@"):
            # an undescribed person in the same zone moments later is most likely the same person
            recent = [e for e in self.entities.values() if e.obj_type == "person" and e.current_zone == det.zone
                      and timedelta(0) <= det.ts - e.last_seen <= timedelta(seconds=120)]
            ent = max(recent, key=lambda e: e.last_seen) if recent else None
        if ent is None:
            ent = self._match_plate_less(det)
            if ent is None and det.obj_type == "vehicle" and det.plate:
                # plate read now, earlier sightings may have lacked it
                legacy = self._by_signature.get(
                    "vehicle|" + "|".join(str(x or "").lower() for x in (det.color, det.make, det.model, det.category)))
                ent = self.entities.get(legacy) if legacy else None

        new_entity = ent is None
        if ent is None:
            ent = Entity(entity_id=self._new_id(det.obj_type), obj_type=det.obj_type, label=det.label,
                         signature=sig, first_seen=det.ts, last_seen=det.ts, authorized=authorized)
            self.entities[ent.entity_id] = ent
        self._by_signature[sig] = ent.entity_id

        # stationary objects (packages) don't make "visits"; they persist until they disappear
        new_visit = new_entity or (det.obj_type != "package" and (det.ts - ent.last_seen) > self.visit_gap)
        if new_visit:
            ent.visits.append(Visit(start=det.ts, last_seen=det.ts, zone_since=det.ts))
            ent.current_zone = None
        visit = ent.current_visit
        zone_changed = det.zone != ent.current_zone
        if zone_changed:
            visit.zone_since = det.ts
            ent.current_zone = det.zone
        if det.zone and det.zone not in visit.zones:
            visit.zones.append(det.zone)
        if det.frame_id not in visit.frames:
            visit.frames.append(det.frame_id)
        visit.last_seen = det.ts
        ent.last_seen = det.ts
        ent.detections += 1
        ent.authorized = ent.authorized or authorized
        ent.attributes.update(det.attributes)
        det.entity_id = ent.entity_id
        return TrackUpdate(entity=ent, new_entity=new_entity, new_visit=new_visit, zone_changed=zone_changed,
                           dwell_s=ent.dwell_s(det.ts), visits_today=ent.visits_on(det.ts))

    # ------------------------------------------------------------------
    def find(self, text: str) -> list[Entity]:
        """Fuzzy lookup used by the agent: 'blue ford f150', 'person in hoodie'."""
        words = [w for w in text.lower().replace("-", "").split() if w not in {"the", "a", "an"}]
        out = []
        for e in self.entities.values():
            hay = (e.label + " " + e.signature + " " + " ".join(e.attributes)).lower().replace("-", "")
            if all(w in hay for w in words):
                out.append(e)
        return out
