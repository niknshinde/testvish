"""Alert rules: each security/safety rule in isolation, through the pipeline."""
from datetime import datetime, timedelta

from drone_security.models import Severity, Telemetry
from tests.conftest import at


def rules_fired(pipeline):
    return [a.rule_id for a in pipeline.alerts.alerts]


def test_alert_triggered_at_midnight_for_loitering(pipeline, frame):
    for hms in ("00:00:00", "00:00:20", "00:00:40", "00:01:00"):
        pipeline.process_frame(frame(hms, "Person in dark hoodie standing near main gate"))
    loiter = [a for a in pipeline.alerts.alerts if a.rule_id == "loitering"]
    assert len(loiter) == 1
    assert loiter[0].message == "Person loitering at main gate, 00:01."
    assert loiter[0].severity == Severity.HIGH
    assert "night" in loiter[0].details
    assert "night_activity" in rules_fired(pipeline)


def test_no_loitering_alert_for_brief_presence(pipeline, frame):
    pipeline.process_frame(frame("12:00:00", "Person in red coat standing at front porch"))
    pipeline.process_frame(frame("12:00:30", "Person in red coat standing at front porch"))
    assert "loitering" not in rules_fired(pipeline)


def test_no_night_alert_at_noon(pipeline, frame):
    pipeline.process_frame(frame("12:00:00", "Man in grey jacket walking on driveway"))
    assert "night_activity" not in rules_fired(pipeline)


def test_night_activity_in_restricted_zone_is_critical(pipeline, frame):
    pipeline.process_frame(frame("02:30:00", "Person with flashlight in backyard"))
    [a] = [a for a in pipeline.alerts.alerts if a.rule_id == "night_activity"]
    assert a.severity == Severity.CRITICAL


def test_perimeter_breach_and_suspicious_object(pipeline, frame):
    pipeline.process_frame(frame("14:10:00", "Person climbing perimeter fence carrying a crowbar"))
    fired = rules_fired(pipeline)
    assert "perimeter_breach" in fired and "suspicious_object" in fired
    assert all(a.severity == Severity.CRITICAL for a in pipeline.alerts.alerts)


def test_repeat_vehicle_alert_on_second_visit_only(pipeline, frame):
    pipeline.process_frame(frame("08:31:00", "Blue Ford F150 entering through main gate"))
    pipeline.process_frame(frame("08:41:00", "Blue Ford F150 leaving through main gate"))
    assert "repeat_vehicle" not in rules_fired(pipeline)
    pipeline.process_frame(frame("11:58:00", "Blue Ford F150 entering through main gate"))
    [a] = [a for a in pipeline.alerts.alerts if a.rule_id == "repeat_vehicle"]
    assert a.message == "Blue Ford F150 entered twice today, 11:58."
    assert "08:31, 11:58" in a.details


def test_authorised_vehicle_never_alerts(pipeline, frame):
    for hms in ("07:00:00", "09:00:00", "11:00:00"):
        pipeline.process_frame(frame(hms, "White Tesla Model 3 (plate OWN-001) entering through main gate"))
    pipeline.process_frame(frame("11:20:00", "White Tesla Model 3 parked in driveway"))
    assert pipeline.alerts.alerts == []
    assert pipeline.logs[0].message.endswith("[authorised].")


def test_vehicle_dwell(pipeline, frame):
    for hms in ("18:20:00", "18:28:00", "18:36:00"):
        pipeline.process_frame(frame(hms, "Red Toyota Camry parked in parking lot"))
    assert "vehicle_dwell" in rules_fired(pipeline)


def test_unattended_package(pipeline, frame):
    pipeline.process_frame(frame("11:23:00", "Package left on front porch"))
    pipeline.process_frame(frame("11:40:00", "Package still on front porch"))
    assert "unattended_package" not in rules_fired(pipeline)
    pipeline.process_frame(frame("12:00:00", "Package still on front porch"))
    assert "unattended_package" in rules_fired(pipeline)


def test_restricted_zone_after_hours(pipeline, frame):
    pipeline.process_frame(frame("21:30:00", "Person walking near tool shed"))
    assert "restricted_zone" in rules_fired(pipeline)
    p2 = type(pipeline)(pipeline.config)
    p2.process_frame(frame("10:00:00", "Gardener walking near tool shed"))   # business hours
    assert p2.alerts.alerts == []


def test_cooldown_suppresses_duplicates(pipeline, frame):
    for i in range(10):
        pipeline.process_frame(frame(f"02:{i:02d}:00", "Person in black coat standing in backyard"))
    assert rules_fired(pipeline).count("night_activity") == 1
    assert pipeline.alerts.suppressed > 0


# ---- telemetry / drone safety -------------------------------------------------------------

def tel(hms, **kw):
    base = dict(ts=at(hms), lat=40.0009, lon=-75.00125, altitude_m=35, battery_pct=80, wind_mps=3, mode="patrol")
    base.update(kw)
    return Telemetry(**base)


def test_low_battery_warning_then_critical(pipeline):
    pipeline.process_telemetry(tel("19:00:00", battery_pct=24))
    pipeline.process_telemetry(tel("19:05:00", battery_pct=12))
    sev = [a.severity for a in pipeline.alerts.alerts if a.rule_id == "low_battery"]
    assert sev == [Severity.MEDIUM, Severity.CRITICAL]


def test_docked_drone_low_battery_is_fine(pipeline):
    pipeline.process_telemetry(tel("19:00:00", battery_pct=10, mode="docked", altitude_m=0))
    assert pipeline.alerts.alerts == []


def test_geofence_altitude_wind_and_gap(pipeline):
    pipeline.process_telemetry(tel("10:00:00"))
    pipeline.process_telemetry(tel("10:00:30", lat=40.0030))           # outside geofence
    pipeline.process_telemetry(tel("10:01:00", altitude_m=150))        # above 120 m ceiling
    pipeline.process_telemetry(tel("10:01:30", wind_mps=15))
    pipeline.process_telemetry(tel("10:05:00"))                        # 210 s gap
    fired = set(rules_fired(pipeline))
    assert {"geofence", "altitude", "high_wind", "telemetry_gap"} <= fired
