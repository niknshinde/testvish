"""Simulated drone day: telemetry + text-described video frames.

``simulate_day`` produces a deterministic 24h stream for one property:

* telemetry every 30 s (position, altitude, battery, wind, mode),
* one patrol frame per minute describing the zone the camera points at,
* a scripted set of security/safety scenarios layered on top (the "ground
  truth" that tests and the demo rely on).

``synthetic_stream`` produces arbitrarily long random streams for scalability
benchmarks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterator

from .config import SiteConfig, load_config
from .models import Frame, Telemetry

DEFAULT_DAY = date(2026, 11, 26)

PATROL_ROUTE = ["main gate", "driveway", "garage", "front porch", "backyard", "perimeter fence",
                "tool shed", "parking lot"]

QUIET = {
    "main gate": ["Main gate closed, no activity", "Clear view of main gate, nothing unusual",
                  "No vehicles at main gate", "Main gate area quiet, street lamp on"],
    "driveway": ["Empty driveway", "Driveway clear, leaves on the ground", "No activity on driveway"],
    "garage": ["Garage door closed, no activity", "Garage exterior clear", "Garage roof and door unchanged"],
    "front porch": ["Front porch clear", "Porch light on, porch empty", "Front door closed, nothing unusual"],
    "backyard": ["Backyard quiet, sprinklers on", "Lawn clear", "Pool cover in place, backyard empty"],
    "perimeter fence": ["Perimeter fence intact", "Trees swaying near perimeter fence",
                        "East wall section clear"],
    "tool shed": ["Tool shed door closed and locked", "Shed area clear"],
    "parking lot": ["Parking lot empty", "Street quiet, parking lot clear", "No vehicles in parking lot"],
}
AMBIENT = ["Birds on the lawn in backyard", "Neighbour's cat walking along perimeter fence",
           "Squirrel near tool shed"]


@dataclass
class ScriptedEvent:
    at: str            # "HH:MM[:SS]"
    zone: str
    description: str
    note: str = ""     # what the system is expected to do (documentation / tests)


# The ground-truth scenario. Expected outcomes are noted for the test-suite and the demo.
SCENARIO: list[ScriptedEvent] = [
    # --- 00:00 midnight loiterer at the main gate -----------------------------------------
    ScriptedEvent("00:00:00", "main gate", "Person in dark hoodie standing near main gate", "night_activity HIGH"),
    ScriptedEvent("00:00:20", "main gate", "Person in dark hoodie at main gate, looking around"),
    ScriptedEvent("00:00:40", "main gate", "Person in dark hoodie still at main gate, peering through the bars"),
    ScriptedEvent("00:01:00", "main gate", "Person in dark hoodie pacing at main gate", "loitering HIGH at 00:01"),
    ScriptedEvent("00:01:40", "main gate", "Person in dark hoodie still near main gate"),
    ScriptedEvent("00:03:00", "parking lot", "Person in dark hoodie walking away along the street"),
    # --- night wildlife --------------------------------------------------------------------
    ScriptedEvent("02:17:00", "backyard", "Fox crossing the backyard", "log only"),
    ScriptedEvent("06:10:00", "parking lot", "Jogger in orange jacket running past on the street", "log only"),
    # --- owner leaves (authorised vehicle) --------------------------------------------------
    ScriptedEvent("07:42:00", "garage", "White Tesla Model 3 (plate OWN-001) exiting garage", "authorised"),
    ScriptedEvent("07:43:00", "main gate", "White Tesla Model 3 leaving through main gate"),
    # --- blue Ford F150, visit 1 ----------------------------------------------------------------
    ScriptedEvent("08:31:00", "main gate", "Blue Ford F150 pickup truck entering through main gate"),
    ScriptedEvent("08:33:00", "driveway", "Blue Ford F150 parked in driveway; man in grey jacket stepping out"),
    ScriptedEvent("08:36:00", "front porch", "Man in grey jacket walking to front porch, rings the doorbell"),
    ScriptedEvent("08:41:00", "main gate", "Blue Ford F150 leaving through main gate"),
    # --- gardener (authorised) ----------------------------------------------------------------
    ScriptedEvent("10:05:00", "driveway",
                  "Green van (plate GRD-452) parked in driveway; gardener in green uniform walking to backyard",
                  "authorised van, no alert"),
    # --- delivery ---------------------------------------------------------------------------------
    ScriptedEvent("11:20:00", "main gate", "White delivery van entering through main gate"),
    ScriptedEvent("11:22:00", "front porch", "Courier in yellow vest carrying a package to the front porch"),
    ScriptedEvent("11:23:00", "front porch", "Package left on front porch"),
    ScriptedEvent("11:24:00", "main gate", "White delivery van exiting through main gate"),
    ScriptedEvent("11:40:00", "front porch", "Package still on front porch"),
    # --- blue Ford F150, visit 2: repeat visit + tampering ----------------------------------------
    ScriptedEvent("11:58:00", "main gate", "Blue Ford F150 entering through main gate", "repeat_vehicle MEDIUM"),
    ScriptedEvent("12:00:00", "garage", "Blue Ford F150 at garage", "log: Blue Ford F150 spotted at garage, 12:00."),
    ScriptedEvent("12:02:00", "front porch", "Package still on front porch", "unattended_package LOW"),
    ScriptedEvent("12:04:00", "garage", "Man in grey jacket trying the garage side door", "perimeter_breach CRITICAL"),
    ScriptedEvent("12:06:00", "main gate", "Blue Ford F150 leaving through main gate"),
    ScriptedEvent("13:30:00", "front porch", "Front porch clear"),
    # --- street life ------------------------------------------------------------------------------
    ScriptedEvent("14:10:00", "parking lot", "Two people walking a dog on the street past the parking lot"),
    # --- unknown car dwelling ---------------------------------------------------------------------
    ScriptedEvent("18:20:00", "parking lot", "Red Toyota Camry sedan (plate KXP-8821) parked in parking lot, engine running"),
    ScriptedEvent("18:27:00", "parking lot", "Red Toyota Camry still parked in parking lot"),
    ScriptedEvent("18:36:00", "parking lot", "Red Toyota Camry parked in parking lot, occupant inside", "vehicle_dwell MEDIUM"),
    ScriptedEvent("18:44:00", "parking lot", "Red Toyota Camry driving away from parking lot"),
    # --- evening intrusion --------------------------------------------------------------------
    ScriptedEvent("21:15:00", "tool shed", "Raccoon near tool shed"),
    ScriptedEvent("22:47:00", "perimeter fence", "Person in black jacket with flashlight walking along perimeter fence",
                  "restricted_zone HIGH"),
    ScriptedEvent("22:49:00", "perimeter fence", "Person in black jacket climbing perimeter fence carrying a crowbar",
                  "perimeter_breach + suspicious_object CRITICAL"),
    ScriptedEvent("22:51:00", "backyard", "Person in black jacket running across backyard toward tool shed"),
    ScriptedEvent("23:05:00", "tool shed", "Person in black jacket prying the tool shed door with a crowbar",
                  "night_activity CRITICAL"),
    ScriptedEvent("23:08:00", "perimeter fence", "Person in black jacket climbing over the east wall, fleeing"),
    ScriptedEvent("23:40:00", "backyard", "Deer in backyard"),
]

# Safety scenarios applied to telemetry (start, end, field overrides)
TELEMETRY_SCENARIOS = [
    ("13:00:00", "13:01:30", {"_drop": True}),                        # datalink gap
    ("15:20:00", "15:25:00", {"wind_mps": 13.5}),                    # gusts
    ("16:45:00", "16:46:00", {"lat_offset": 0.0009}),                # drift outside geofence
    ("19:38:00", "19:45:00", {"battery_pct": 22.0}),                 # low battery
    ("19:46:00", "19:48:00", {"battery_pct": 13.0, "mode": "returning"}),
]


def _at(day: date, hms: str) -> datetime:
    parts = [int(p) for p in hms.split(":")] + [0]
    return datetime(day.year, day.month, day.day, parts[0], parts[1], parts[2])


def _battery(ts: datetime) -> tuple[float, str]:
    """40-min flight / 20-min dock cycle."""
    minute = (ts.hour * 60 + ts.minute) % 60 + ts.second / 60
    if minute < 40:
        return 100.0 - minute * 1.5, "patrol"
    return 40.0 + (minute - 40) * 3.0, "docked"


def simulate_day(day: date = DEFAULT_DAY, config: SiteConfig | None = None, seed: int = 7,
                 frame_interval_s: int = 60, telemetry_interval_s: int = 30) -> list[Frame | Telemetry]:
    cfg = config or load_config()
    rng = random.Random(seed)
    start = datetime(day.year, day.month, day.day)
    events = sorted(((_at(day, e.at), e) for e in SCENARIO), key=lambda x: x[0])
    event_times = [t for t, _ in events]
    zone_center = {z.name: z.center for z in cfg.zones}

    def active_event(ts: datetime) -> ScriptedEvent | None:
        best = None
        for t, e in events:
            if abs((ts - t).total_seconds()) <= 45:
                best = e
        return best

    def telemetry_at(ts: datetime, zone: str) -> Telemetry | None:
        battery, mode = _battery(ts)
        ev = active_event(ts)
        if ev:
            zone, mode = ev.zone, "investigating"
        lat, lon = zone_center.get(zone) or cfg.dock
        alt = 35.0 + 5 * math.sin(ts.minute / 9.0)
        wind = 3.0 + 2.0 * math.sin(ts.hour / 3.0) + rng.random()
        if mode == "docked":
            lat, lon, alt = cfg.dock[0], cfg.dock[1], 0.0
        for s, e, over in TELEMETRY_SCENARIOS:
            if _at(day, s) <= ts <= _at(day, e):
                if over.get("_drop"):
                    return None
                wind = over.get("wind_mps", wind)
                lat += over.get("lat_offset", 0.0)
                battery = over.get("battery_pct", battery)
                mode = over.get("mode", "patrol" if mode == "docked" else mode)
                if mode != "docked" and alt == 0.0:
                    lat, lon = zone_center.get(zone) or cfg.dock
                    lat += over.get("lat_offset", 0.0)
                    alt = 35.0
        return Telemetry(ts=ts, lat=round(lat + rng.uniform(-2e-5, 2e-5), 6),
                         lon=round(lon + rng.uniform(-2e-5, 2e-5), 6), altitude_m=round(alt, 1),
                         battery_pct=round(battery, 1), zone=zone, heading_deg=round(rng.uniform(0, 359), 0),
                         speed_mps=0.0 if mode == "docked" else round(rng.uniform(2, 6), 1),
                         wind_mps=round(wind, 1), mode=mode)

    records: list[tuple[datetime, int, Frame | Telemetry]] = []
    last_tel: Telemetry | None = None
    for i in range(0, 24 * 3600, telemetry_interval_s):
        ts = start + timedelta(seconds=i)
        zone = PATROL_ROUTE[(i // 60) % len(PATROL_ROUTE)]
        t = telemetry_at(ts, zone)
        if t:
            records.append((ts, 0, t))

    def nearest_telemetry(ts: datetime, zone: str) -> Telemetry | None:
        return telemetry_at(ts, zone)

    frame_id = 0
    frames: list[tuple[datetime, Frame]] = []
    for i in range(0, 24 * 3600, frame_interval_s):
        ts = start + timedelta(seconds=i)
        if any(abs((ts - t).total_seconds()) < frame_interval_s for t in event_times):
            continue  # the scripted frame replaces the routine patrol frame
        zone = PATROL_ROUTE[(i // 60) % len(PATROL_ROUTE)]
        desc = rng.choice(QUIET[zone])
        if rng.random() < 0.01:
            desc = rng.choice(AMBIENT)
        frames.append((ts, Frame(0, ts, desc, zone=zone)))
    for ts, ev in events:
        frames.append((ts, Frame(0, ts, ev.description, zone=ev.zone)))
    frames.sort(key=lambda x: x[0])
    for ts, f in frames:
        frame_id += 1
        f.frame_id = frame_id
        f.telemetry = nearest_telemetry(ts, f.zone or "main gate")
        records.append((ts, 1, f))

    records.sort(key=lambda r: (r[0], r[1]))
    return [r for _, _, r in records]


# ---------------------------------------------------------------------------
# Random long streams for scalability tests
# ---------------------------------------------------------------------------

_COLORS = ["blue", "red", "white", "black", "silver", "grey", "green"]
_CARS = ["Ford F150", "Toyota Camry", "Honda Civic", "Tesla Model Y", "Chevrolet Silverado", "Ford Transit",
         "Nissan Altima", "Jeep Wrangler"]
_PEOPLE = ["Person in {c} hoodie", "Man in {c} jacket", "Woman in {c} coat", "Courier in yellow vest"]
_ACTS_V = ["entering through main gate", "parked in driveway", "leaving through main gate", "at garage",
           "parked in parking lot"]
_ACTS_P = ["walking on driveway", "standing at front porch", "walking in backyard", "at main gate, looking around",
           "climbing perimeter fence", "walking past parking lot"]


def synthetic_stream(n_frames: int, seed: int = 1, start: datetime | None = None,
                     frame_interval_s: int = 5, event_rate: float = 0.3) -> Iterator[Frame]:
    rng = random.Random(seed)
    ts = start or datetime(2026, 1, 1)
    for fid in range(1, n_frames + 1):
        ts += timedelta(seconds=frame_interval_s)
        zone = PATROL_ROUTE[fid % len(PATROL_ROUTE)]
        if rng.random() < event_rate:
            if rng.random() < 0.55:
                desc = f"{rng.choice(_COLORS).title()} {rng.choice(_CARS)} {rng.choice(_ACTS_V)}"
            else:
                desc = f"{rng.choice(_PEOPLE).format(c=rng.choice(_COLORS))} {rng.choice(_ACTS_P)}"
        else:
            desc = rng.choice(QUIET[zone])
        yield Frame(fid, ts, desc, zone=zone)
