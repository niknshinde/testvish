"""Declarative security & safety alert rules.

Each rule is a small class with an ``id`` and an ``evaluate`` method returning
zero or more :class:`AlertCandidate`. Thresholds and severities come from the
``rules:`` section of ``config/property.yaml``, so operators tune behaviour
without touching code. Rules are pure functions of (detection, tracker state,
config) which makes them easy to unit-test.

Two families:

* **Security rules** run on every detection, after the tracker has attached
  entity history (dwell time, visit count).
* **Safety rules** run on every telemetry sample (battery, altitude, geofence,
  wind, telemetry gaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import SiteConfig
from .models import Detection, Frame, Severity, Telemetry, hhmm
from .tracker import TrackUpdate

SUSPICIOUS_ITEMS = {"crowbar", "weapon", "gun", "knife", "bolt cutters", "ladder", "spray paint",
                    "gas can", "balaclava", "ski mask", "mask", "masked"}


@dataclass
class AlertCandidate:
    rule_id: str
    severity: Severity
    message: str
    ts: datetime
    dedup_key: str
    frame_id: int | None = None
    zone: str | None = None
    entity_id: str | None = None
    recommendation: str | None = None
    details: str | None = None
    evidence_frames: list[int] = field(default_factory=list)


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Security rules (per detection)
# ---------------------------------------------------------------------------

class DetectionRule:
    id = "base"

    def __init__(self, config: SiteConfig):
        self.config = config
        self.params = config.rule(self.id)

    @property
    def enabled(self) -> bool:
        return bool(self.params.get("enabled", True))

    def severity(self, key: str = "severity", default: str = "medium") -> Severity:
        return Severity.parse(self.params.get(key, default))

    def evaluate(self, det: Detection, upd: TrackUpdate, frame: Frame) -> list[AlertCandidate]:
        raise NotImplementedError


class LoiteringRule(DetectionRule):
    """A person who stays in the same zone longer than ``min_dwell_s``."""

    id = "loitering"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "person" or det.category == "courier":
            return []
        if det.action not in (None, "standing", "loitering"):
            return []   # moving/climbing people are covered by other rules
        min_dwell = float(self.params.get("min_dwell_s", 120))
        if upd.dwell_s < min_dwell:
            return []
        e = upd.entity
        since = e.current_visit.zone_since or e.current_visit.start
        night = " during night hours" if self.config.is_night(det.ts) else ""
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="high"),
            message=f"Person loitering at {det.zone}, {hhmm(det.ts)}.",
            details=f"{e.label} has been at {det.zone} for {_fmt_dur(upd.dwell_s)} since {hhmm(since)}{night}.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=e.entity_id,
            dedup_key=f"{self.id}:{e.entity_id}:{since.isoformat()}",
            recommendation=f"Hold position over {det.zone}, switch on spotlight and two-way speaker warning; "
                           f"notify the owner with the live clip.",
            evidence_frames=e.current_visit.frames[-10:],
        )]


class NightActivityRule(DetectionRule):
    """Any person on the property during night hours; critical inside restricted zones."""

    id = "night_activity"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "person" or not self.config.is_night(det.ts):
            return []
        restricted = self.config.is_restricted(det.zone)
        sev = self.severity("restricted_severity", "critical") if restricted else self.severity(default="high")
        where = f"restricted zone '{det.zone}'" if restricted else str(det.zone)
        e = upd.entity
        return [AlertCandidate(
            rule_id=self.id, severity=sev,
            message=f"Person detected at {det.zone} during night hours, {hhmm(det.ts)}.",
            details=f"{e.label} seen at {where} at {det.ts:%H:%M:%S} (night window "
                    f"{self.config.night_start:%H:%M}-{self.config.night_end:%H:%M}).",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=e.entity_id,
            dedup_key=f"{self.id}:{e.entity_id}:{e.current_visit.start.isoformat()}",
            recommendation="Investigate: fly to subject, record continuously and notify the owner"
                           + ("; escalate to security/police if not identified." if restricted else "."),
            evidence_frames=[det.frame_id],
        )]


class PerimeterBreachRule(DetectionRule):
    id = "perimeter_breach"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "person" or det.action not in {"climbing", "tampering"}:
            return []
        verb = "climbing" if det.action == "climbing" else "tampering"
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="critical"),
            message=f"Perimeter breach: person {verb} at {det.zone}, {hhmm(det.ts)}.",
            details=f"Source: '{det.source_text}'.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=upd.entity.entity_id,
            dedup_key=f"{self.id}:{upd.entity.entity_id}:{det.zone}",
            recommendation="Trigger siren, track subject with the drone, call security/emergency services "
                           "and preserve the clip as evidence.",
            evidence_frames=[det.frame_id],
        )]


class SuspiciousObjectRule(DetectionRule):
    id = "suspicious_object"

    def evaluate(self, det, upd, frame):
        items = sorted(set(det.attributes) & SUSPICIOUS_ITEMS)
        if det.obj_type == "object" and det.category in SUSPICIOUS_ITEMS:
            items = [det.category]
        if not items:
            return []
        who = "Person" if det.obj_type == "person" else "Unattended item"
        what = " and ".join(items)
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="critical"),
            message=f"{who} with {what} at {det.zone}, {hhmm(det.ts)}.",
            details=f"Source: '{det.source_text}'.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=upd.entity.entity_id,
            dedup_key=f"{self.id}:{upd.entity.entity_id}:{what}",
            recommendation="Treat as potential intrusion: notify security immediately and keep the subject in frame.",
            evidence_frames=[det.frame_id],
        )]


class RestrictedZoneRule(DetectionRule):
    """People in restricted zones outside business hours (night is covered by NightActivityRule)."""

    id = "restricted_zone"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "person" or not self.config.is_restricted(det.zone):
            return []
        if self.config.is_business_hours(det.ts) or self.config.is_night(det.ts):
            return []
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="high"),
            message=f"Person in restricted zone {det.zone} after hours, {hhmm(det.ts)}.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=upd.entity.entity_id,
            dedup_key=f"{self.id}:{upd.entity.entity_id}:{det.zone}",
            recommendation="Verify identity via two-way speaker; notify owner.",
            evidence_frames=[det.frame_id],
        )]


class RepeatVehicleRule(DetectionRule):
    """An unrecognised vehicle that returns on the same day ('blue Ford F150 entered twice today')."""

    id = "repeat_vehicle"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "vehicle" or not upd.new_visit or upd.entity.authorized:
            return []
        n = upd.visits_today
        if n < int(self.params.get("min_visits", 2)):
            return []
        e = upd.entity
        times = ", ".join(hhmm(v.start) for v in e.visits if v.start.date() == det.ts.date())
        times_word = {2: "twice", 3: "three times"}.get(n, f"{n} times")
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="medium"),
            message=f"{e.label} entered {times_word} today, {hhmm(det.ts)}.",
            details=f"Visits at {times}. Vehicle is not on the authorised list"
                    + (f" (plate {det.plate})." if det.plate else "; plate not read."),
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=e.entity_id,
            dedup_key=f"{self.id}:{e.entity_id}:{n}",
            recommendation="Check against expected visitors/deliveries; capture the plate on this visit "
                           "and add to the watch-list if unknown.",
            evidence_frames=[f for v in e.visits for f in v.frames[:2]],
        )]


class VehicleDwellRule(DetectionRule):
    id = "vehicle_dwell"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "vehicle" or upd.entity.authorized:
            return []
        min_dwell = float(self.params.get("min_dwell_s", 900))
        if upd.dwell_s < min_dwell:
            return []
        e = upd.entity
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="medium"),
            message=f"Unrecognised {e.label} parked at {det.zone} for {_fmt_dur(upd.dwell_s)}, {hhmm(det.ts)}.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=e.entity_id,
            dedup_key=f"{self.id}:{e.entity_id}:{e.current_visit.start.isoformat()}",
            recommendation="Record plate and occupants; ask owner whether the vehicle is expected.",
            evidence_frames=e.current_visit.frames[-5:],
        )]


class UnattendedPackageRule(DetectionRule):
    id = "unattended_package"

    def evaluate(self, det, upd, frame):
        if det.obj_type != "package":
            return []
        if upd.dwell_s < float(self.params.get("min_dwell_s", 1800)):
            return []
        e = upd.entity
        return [AlertCandidate(
            rule_id=self.id, severity=self.severity(default="low"),
            message=f"Package left unattended at {det.zone} for {_fmt_dur(upd.dwell_s)}, {hhmm(det.ts)}.",
            ts=det.ts, frame_id=det.frame_id, zone=det.zone, entity_id=e.entity_id,
            dedup_key=f"{self.id}:{e.entity_id}",
            recommendation="Remind the owner to collect the package (theft risk).",
            evidence_frames=[e.current_visit.frames[0], det.frame_id],
        )]


DETECTION_RULES = [LoiteringRule, NightActivityRule, PerimeterBreachRule, SuspiciousObjectRule,
                   RestrictedZoneRule, RepeatVehicleRule, VehicleDwellRule, UnattendedPackageRule]


# ---------------------------------------------------------------------------
# Safety rules (per telemetry sample)
# ---------------------------------------------------------------------------

class TelemetryRules:
    """Drone safety checks. Stateful only for the telemetry-gap check."""

    def __init__(self, config: SiteConfig):
        self.config = config
        self._last_ts: datetime | None = None

    def evaluate(self, t: Telemetry) -> list[AlertCandidate]:
        c = self.config
        out: list[AlertCandidate] = []
        r = c.rule("low_battery")
        if r.get("enabled", True) and t.mode != "docked":
            if t.battery_pct <= r.get("critical_pct", 15):
                out.append(AlertCandidate("low_battery", Severity.CRITICAL,
                                          f"Drone battery critical ({t.battery_pct:.0f}%), {hhmm(t.ts)}.", t.ts,
                                          f"low_battery:critical:{t.ts.date()}",
                                          recommendation="Abort patrol and land at dock immediately."))
            elif t.battery_pct <= r.get("warn_pct", 25):
                out.append(AlertCandidate("low_battery", Severity.MEDIUM,
                                          f"Drone battery low ({t.battery_pct:.0f}%), {hhmm(t.ts)}.", t.ts,
                                          f"low_battery:warn:{t.ts.date()}",
                                          recommendation="Return to dock to recharge; resume patrol after."))
        r = c.rule("altitude")
        if r.get("enabled", True) and t.mode not in ("docked", "returning", "landing"):
            if t.altitude_m < r.get("min_m", 5) or t.altitude_m > r.get("max_m", 120):
                out.append(AlertCandidate("altitude", Severity.parse(r.get("severity", "high")),
                                          f"Drone altitude out of bounds ({t.altitude_m:.0f} m), {hhmm(t.ts)}.",
                                          t.ts, "altitude",
                                          recommendation=f"Hold between {r.get('min_m', 5)} and {r.get('max_m', 120)} m "
                                                         "(regulatory ceiling)."))
        r = c.rule("geofence")
        if r.get("enabled", True) and not c.geofence.contains(t.lat, t.lon):
            out.append(AlertCandidate("geofence", Severity.parse(r.get("severity", "high")),
                                      f"Drone outside property geofence ({t.lat:.5f}, {t.lon:.5f}), {hhmm(t.ts)}.",
                                      t.ts, "geofence",
                                      recommendation="Return inside the geofence; check GPS / wind drift."))
        r = c.rule("high_wind")
        if r.get("enabled", True) and t.wind_mps > r.get("max_mps", 12) and t.mode != "docked":
            out.append(AlertCandidate("high_wind", Severity.parse(r.get("severity", "medium")),
                                      f"High wind {t.wind_mps:.0f} m/s, {hhmm(t.ts)}.", t.ts, "high_wind",
                                      recommendation="Reduce patrol radius or dock until wind drops."))
        r = c.rule("telemetry_gap")
        if r.get("enabled", True) and self._last_ts is not None:
            gap = (t.ts - self._last_ts).total_seconds()
            if gap > r.get("max_gap_s", 30) and t.mode != "docked":
                out.append(AlertCandidate("telemetry_gap", Severity.parse(r.get("severity", "medium")),
                                          f"Telemetry gap of {gap:.0f}s ended, {hhmm(t.ts)}.", t.ts,
                                          f"telemetry_gap:{t.ts.isoformat()}",
                                          recommendation="Check datalink; footage during the gap may be missing."))
        self._last_ts = t.ts
        return out
