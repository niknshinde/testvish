"""Site configuration loading (zones, schedule, authorised vehicles, rule params)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "property.yaml"


def _t(value: str) -> time:
    h, m = str(value).split(":")
    return time(int(h), int(m))


@dataclass
class Zone:
    name: str
    aliases: list[str]
    restricted: bool = False
    center: tuple[float, float] | None = None


@dataclass
class Geofence:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


@dataclass
class SiteConfig:
    name: str
    geofence: Geofence
    dock: tuple[float, float]
    zones: list[Zone]
    night_start: time
    night_end: time
    business_start: time
    business_end: time
    authorized_vehicles: list[dict[str, Any]]
    rules: dict[str, dict[str, Any]]
    alerting: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)

    # ---- helpers -----------------------------------------------------
    def zone(self, name: str | None) -> Zone | None:
        if not name:
            return None
        for z in self.zones:
            if z.name == name:
                return z
        return None

    def is_restricted(self, zone_name: str | None) -> bool:
        z = self.zone(zone_name)
        return bool(z and z.restricted)

    def resolve_zone(self, text: str) -> str | None:
        """Map free text to a canonical zone name using the longest matching alias."""
        text_l = text.lower()
        hits: list[tuple[bool, int, str]] = []    # (is_destination, -alias_len, zone)
        for z in self.zones:
            for alias in z.aliases:
                for m in re.finditer(rf"\b{re.escape(alias)}\b", text_l):
                    # "running across backyard toward tool shed" -> the subject is in the backyard
                    dest = bool(re.search(r"\b(toward|towards|heading to|in the direction of)\s+(the\s+)?$",
                                          text_l[:m.start()]))
                    hits.append((dest, -len(alias), z.name))
        return min(hits)[2] if hits else None

    def is_night(self, ts: datetime) -> bool:
        t = ts.time()
        if self.night_start <= self.night_end:
            return self.night_start <= t < self.night_end
        return t >= self.night_start or t < self.night_end

    def is_business_hours(self, ts: datetime) -> bool:
        return self.business_start <= ts.time() < self.business_end

    def rule(self, rule_id: str) -> dict[str, Any]:
        return self.rules.get(rule_id, {})

    def is_authorized_vehicle(self, *, plate: str | None, color: str | None, make: str | None,
                              model: str | None, category: str | None) -> bool:
        for v in self.authorized_vehicles:
            if plate and v.get("plate") and plate.upper() == str(v["plate"]).upper():
                return True
            keys = [k for k in ("color", "make", "model", "category") if v.get(k)]
            if not keys:
                continue
            observed = {"color": color, "make": make, "model": model, "category": category}
            if all(observed[k] and str(observed[k]).lower() == str(v[k]).lower() for k in keys):
                return True
        return False


def load_config(path: str | Path | None = None) -> SiteConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    prop = raw["property"]
    zones = [
        Zone(
            name=z["name"],
            aliases=[a.lower() for a in z.get("aliases", [z["name"]])],
            restricted=bool(z.get("restricted", False)),
            center=(z["center"]["lat"], z["center"]["lon"]) if z.get("center") else None,
        )
        for z in raw.get("zones", [])
    ]
    sched = raw.get("schedule", {})
    return SiteConfig(
        name=prop.get("name", "Property"),
        geofence=Geofence(**prop["geofence"]),
        dock=(prop["dock"]["lat"], prop["dock"]["lon"]),
        zones=zones,
        night_start=_t(sched.get("night_start", "23:00")),
        night_end=_t(sched.get("night_end", "05:00")),
        business_start=_t(sched.get("business_start", "07:00")),
        business_end=_t(sched.get("business_end", "20:00")),
        authorized_vehicles=raw.get("authorized_vehicles", []),
        rules=raw.get("rules", {}),
        alerting=raw.get("alerting", {}),
        tracking=raw.get("tracking", {}),
    )
