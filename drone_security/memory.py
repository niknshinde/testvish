"""Context management: a bounded, hierarchical memory of the day.

A drone records ~86k frames a day at 1 fps; an LLM can't (and shouldn't) see
them all. ``SecurityMemory`` keeps three tiers of context whose total size is
bounded no matter how long the video is:

1. **Working memory** - the last ``recent_n`` log lines verbatim.
2. **Episodic memory** - one compressed line per hour ("08:00-09:00: 3 vehicle
   sightings (Blue Ford F150, ...), 1 person; alerts: none").
3. **Semantic memory** - the entity registry from the tracker (who/what was
   seen, how many visits, where) plus every alert, most severe first.

``context_block()`` renders these tiers into a compact text block that is
injected into the agent's system prompt; anything finer-grained is fetched on
demand through the index tools (retrieval rather than stuffing).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

from .models import Alert, Detection, LogEntry, Severity, hhmm
from .tracker import EntityTracker


@dataclass
class _Hour:
    start: datetime
    objects: Counter = field(default_factory=Counter)
    labels: Counter = field(default_factory=Counter)
    alerts: list[str] = field(default_factory=list)
    frames: int = 0


class SecurityMemory:
    def __init__(self, tracker: EntityTracker, recent_n: int = 25, max_alerts: int = 25):
        self.tracker = tracker
        self.recent: deque[LogEntry] = deque(maxlen=recent_n)
        self.alerts: list[Alert] = []
        self.max_alerts = max_alerts
        self.hours: dict[str, _Hour] = {}
        self.frames_seen = 0
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None

    def _hour(self, ts: datetime) -> _Hour:
        key = ts.strftime("%Y-%m-%d %H")
        if key not in self.hours:
            self.hours[key] = _Hour(start=ts.replace(minute=0, second=0, microsecond=0))
        return self.hours[key]

    def observe_frame(self, ts: datetime, detections: list[Detection], logs: list[LogEntry],
                      alerts: list[Alert]) -> None:
        self.frames_seen += 1
        self.first_ts = self.first_ts or ts
        self.last_ts = ts
        h = self._hour(ts)
        h.frames += 1
        for d in detections:
            h.objects[d.category or d.obj_type] += 1
        self.recent.extend(logs)
        for d in detections:
            h.labels[d.label] += 1
        for a in alerts:
            self.add_alert(a)

    def add_alert(self, a: Alert) -> None:
        self.alerts.append(a)
        self._hour(a.ts).alerts.append(f"{a.severity.name}: {a.message}")

    # ---- rendering -------------------------------------------------------
    def episodic_lines(self, max_hours: int = 24) -> list[str]:
        """One line per active hour; hours beyond the most recent ``max_hours`` are rolled up into one line."""
        active = [k for k in sorted(self.hours) if self.hours[k].objects or self.hours[k].alerts]
        older, recent = active[:-max_hours], active[-max_hours:]
        lines = []
        if older:
            n_obj = sum(sum(self.hours[k].objects.values()) for k in older)
            n_al = sum(len(self.hours[k].alerts) for k in older)
            lines.append(f"Earlier ({self.hours[older[0]].start:%Y-%m-%d %H:00} to {self.hours[older[-1]].start:%Y-%m-%d %H:59}): "
                         f"{len(older)} active hours, {n_obj} detections, {n_al} alerts (query tools for detail)")
        for key in recent:
            h = self.hours[key]
            objs = ", ".join(f"{n} {k}" for k, n in h.objects.most_common(4))
            top = ", ".join(l for l, _ in h.labels.most_common(3))
            al = "; ".join(h.alerts[:3]) or "none"
            lines.append(f"{h.start:%H:00}-{h.start:%H:59}: {objs} (top: {top}); alerts: {al}"[:300])
        return lines

    def entity_lines(self, limit: int = 20) -> list[str]:
        ents = sorted(self.tracker.entities.values(), key=lambda e: (-len(e.visits), e.first_seen))
        return [e.summary() + (" [authorised]" if e.authorized else "") for e in ents[:limit]]

    def alert_lines(self) -> list[str]:
        ranked = sorted(self.alerts, key=lambda a: (-int(a.severity), a.ts))[: self.max_alerts]
        return [f"{a.alert_id} {a.severity.name} {hhmm(a.ts)} [{a.rule_id}] {a.message}" for a in
                sorted(ranked, key=lambda a: a.ts)]

    def context_block(self) -> str:
        if not self.frames_seen:
            return "No frames processed yet."
        sev = Counter(a.severity.name for a in self.alerts)
        parts = [
            f"Coverage: {self.frames_seen} frames from {self.first_ts:%Y-%m-%d %H:%M} to {self.last_ts:%H:%M}.",
            f"Alerts: {len(self.alerts)} total ({', '.join(f'{k}={v}' for k, v in sev.most_common()) or 'none'}).",
            "## Alerts (chronological)", *(self.alert_lines() or ["none"]),
            "## Tracked entities (most visits first)", *(self.entity_lines() or ["none"]),
            "## Hourly episodes", *(self.episodic_lines() or ["none"]),
            "## Most recent log lines", *(f"{hhmm(l.ts)} {l.message}" for l in self.recent),
        ]
        return "\n".join(parts)

    def highest_severity(self) -> Severity | None:
        return max((a.severity for a in self.alerts), default=None)
