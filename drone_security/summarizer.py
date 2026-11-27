"""Video summarisation and agent recommendations (deterministic, offline).

The LLM agent can produce richer prose, but these functions guarantee a
useful answer with no API key and give the LLM grounded facts to build on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING

from .models import Severity, hhmm

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import SecurityPipeline


_IRREGULAR = {"person": "people"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word if n == 1 else _IRREGULAR.get(word, word + 's')}"


def one_sentence_summary(p: "SecurityPipeline") -> str:
    m = p.memory
    if not m.frames_seen:
        return "No footage has been processed yet."
    ents = list(p.tracker.entities.values())
    kinds = Counter(e.obj_type for e in ents)
    vehicle_visits = sum(len(e.visits) for e in ents if e.obj_type == "vehicle")
    alerts = sorted(p.alerts.alerts, key=lambda a: (-int(a.severity), a.ts))
    n_crit = sum(1 for a in alerts if a.severity == Severity.CRITICAL)
    parts = [f"On {m.first_ts:%d %b %Y} the drone analysed {m.frames_seen:,} frames and saw "
             f"{_plural(vehicle_visits, 'vehicle visit')}, {_plural(kinds.get('person', 0), 'person')} and "
             f"{_plural(kinds.get('animal', 0), 'animal')}"]
    if alerts:
        sec = [a for a in alerts if a.rule_id not in _SAFETY_RULES]
        # the most serious incident = the entity with the highest total alert severity
        score: Counter = Counter()
        for a in sec:
            score[a.entity_id] += int(a.severity) ** 2
        top = alerts[0]
        if score:
            worst = score.most_common(1)[0][0]
            top = next(a for a in sec if a.entity_id == worst and a.severity == max(
                b.severity for b in sec if b.entity_id == worst))
        parts.append(f", raising {_plural(len(alerts), 'alert')} ({n_crit} critical) - most serious: "
                     f"{top.message.rstrip('.').lower()}")
    repeat = [e for e in ents if e.obj_type == "vehicle" and len(e.visits) > 1 and not e.authorized]
    if repeat:
        e = repeat[0]
        parts.append(f"; the unrecognised {e.label} visited {len(e.visits)} times "
                     f"({', '.join(hhmm(v.start) for v in e.visits)})")
    return "".join(parts) + "."


_SAFETY_RULES = {"low_battery", "altitude", "geofence", "high_wind", "telemetry_gap"}


def correlate_incidents(p: "SecurityPipeline", window_min: int = 10) -> list[str]:
    """Cross-event reasoning: link people's alerts to vehicles present at the same time."""
    findings = []
    vehicles = [e for e in p.tracker.entities.values() if e.obj_type == "vehicle" and not e.authorized]
    for a in p.alerts.alerts:
        if a.rule_id not in {"perimeter_breach", "suspicious_object", "loitering", "restricted_zone"}:
            continue
        for v in vehicles:
            for i, visit in enumerate(v.visits, 1):
                if visit.start - timedelta(minutes=window_min) <= a.ts <= visit.last_seen + timedelta(minutes=window_min):
                    findings.append(f"{a.message.rstrip('.')} happened during visit {i} of the {v.label} "
                                    f"({hhmm(visit.start)}-{hhmm(visit.last_seen)}); the vehicle's occupant is the "
                                    f"likely subject.")
    return list(dict.fromkeys(findings))


def recommendations(p: "SecurityPipeline", limit: int = 10) -> list[str]:
    recs: list[tuple[int, str]] = []
    alerts = p.alerts.alerts
    # 1) cross-event correlations are the highest-value insight
    for f in correlate_incidents(p):
        recs.append((5, f"Treat as a suspect vehicle: {f} Share plate/footage with security and add it to the "
                        f"watch-list."))
    # 2) the most severe open alerts, one recommendation per rule
    seen_rules = set()
    for a in sorted(alerts, key=lambda a: (-int(a.severity), a.ts)):
        if a.rule_id in seen_rules or not a.recommendation:
            continue
        seen_rules.add(a.rule_id)
        recs.append((int(a.severity), f"[{a.severity.name}] {hhmm(a.ts)} {a.message} -> {a.recommendation}"))
    # 3) hot-spot zones
    per_zone = Counter(a.zone for a in alerts if a.zone and a.rule_id not in _SAFETY_RULES)
    for zone, n in per_zone.most_common(2):
        if n >= 2:
            recs.append((2, f"{zone.title()} produced {n} security alerts: add motion-activated lighting and "
                            f"increase drone patrol frequency there, especially at night."))
    # 4) drone operations
    by_rule = defaultdict(list)
    for a in alerts:
        by_rule[a.rule_id].append(a)
    if "low_battery" in by_rule:
        worst = min(by_rule["low_battery"], key=lambda a: a.ts)
        recs.append((2, f"Battery ran low at {hhmm(worst.ts)}: shorten evening patrol legs or schedule a "
                        f"recharge before the 22:00-06:00 high-risk window."))
    if "geofence" in by_rule or "high_wind" in by_rule:
        recs.append((1, "Wind/geofence events detected: enable wind-aware patrol speeds and tighten the "
                        "return-to-home margin."))
    recs.sort(key=lambda r: -r[0])
    return [r for _, r in recs[:limit]]
