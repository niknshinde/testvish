"""Alert manager: de-duplication, cool-down and fan-out to sinks."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .models import Alert, Severity
from .rules import AlertCandidate

AlertSink = Callable[[Alert], None]

_COLOURS = {Severity.CRITICAL: "\033[1;41;97m", Severity.HIGH: "\033[1;31m", Severity.MEDIUM: "\033[33m",
            Severity.LOW: "\033[36m", Severity.INFO: "\033[37m"}


def console_sink(stream=sys.stdout, colour: bool = True) -> AlertSink:
    def _sink(alert: Alert) -> None:
        tag = f"[ALERT:{alert.severity.name}]"
        if colour:
            tag = f"{_COLOURS[alert.severity]}{tag}\033[0m"
        stream.write(f"{tag} {alert.message}\n")
    return _sink


def jsonl_sink(path: str | Path) -> AlertSink:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _sink(alert: Alert) -> None:
        with p.open("a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
    return _sink


class AlertManager:
    """Turns rule candidates into alerts.

    * exact duplicates (same ``dedup_key``) are emitted once;
    * the same rule for the same entity/zone is suppressed for ``cooldown_s``
      so a loitering person doesn't produce an alert every frame;
    * ...unless the severity escalates (battery MEDIUM -> CRITICAL), which always goes out.
    """

    def __init__(self, cooldown_s: float = 600, sinks: list[AlertSink] | None = None):
        self.cooldown = timedelta(seconds=cooldown_s)
        self.sinks = list(sinks or [])
        self.alerts: list[Alert] = []
        self._seen_keys: set[str] = set()
        self._last_by_scope: dict[str, tuple[datetime, Severity]] = {}
        self.suppressed = 0

    def submit(self, cand: AlertCandidate) -> Alert | None:
        scope = f"{cand.rule_id}:{cand.entity_id or cand.zone or '-'}"
        last = self._last_by_scope.get(scope)
        escalation = last is not None and cand.severity > last[1]   # e.g. battery LOW -> CRITICAL
        if cand.dedup_key in self._seen_keys or (
                last is not None and cand.ts - last[0] < self.cooldown and not escalation):
            self.suppressed += 1
            return None
        self._seen_keys.add(cand.dedup_key)
        self._last_by_scope[scope] = (cand.ts, cand.severity)
        alert = Alert(
            alert_id=f"A{len(self.alerts) + 1:04d}", ts=cand.ts, rule_id=cand.rule_id, severity=cand.severity,
            message=cand.message, frame_id=cand.frame_id, zone=cand.zone, entity_id=cand.entity_id,
            recommendation=cand.recommendation, details=cand.details, evidence_frames=cand.evidence_frames,
        )
        self.alerts.append(alert)
        for sink in self.sinks:
            sink(alert)
        return alert
