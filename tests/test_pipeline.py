"""End-to-end scenario over the full simulated day (the ground truth in simulator.SCENARIO)."""
from drone_security.models import Severity


def msgs(p):
    return [l.message for l in p.logs]


def test_truck_logged_correctly(day_pipeline):
    assert "Blue Ford F150 spotted at garage, 12:00." in msgs(day_pipeline)
    assert "Blue Ford F150 entering at main gate, 08:31." in msgs(day_pipeline)
    assert "Blue Ford F150 entering at main gate, 11:58 (2nd visit today)." in msgs(day_pipeline)


def test_midnight_loitering_alert(day_pipeline):
    a = next(a for a in day_pipeline.alerts.alerts if a.rule_id == "loitering")
    assert a.message == "Person loitering at main gate, 00:01."


def test_expected_alert_set(day_pipeline):
    got = {(a.rule_id, a.ts.strftime("%H:%M")) for a in day_pipeline.alerts.alerts}
    expected = {
        ("night_activity", "00:00"), ("loitering", "00:01"), ("repeat_vehicle", "11:58"),
        ("unattended_package", "12:02"), ("perimeter_breach", "12:04"), ("telemetry_gap", "13:02"),
        ("high_wind", "15:20"), ("geofence", "16:45"), ("vehicle_dwell", "18:36"), ("low_battery", "19:38"),
        ("low_battery", "19:46"), ("restricted_zone", "22:47"), ("perimeter_breach", "22:49"),
        ("suspicious_object", "22:49"), ("night_activity", "23:05"),
    }
    assert expected <= got
    # and nothing fired on quiet patrol frames / authorised vehicles / wildlife
    assert len(got) <= len(expected) + 2


def test_no_false_alerts_for_authorised_or_wildlife(day_pipeline):
    ents = {a.entity_id for a in day_pipeline.alerts.alerts if a.entity_id}
    labels = {day_pipeline.tracker.entities[e].label for e in ents}
    assert not any(l.startswith(("White Tesla", "Green van", "Fox", "Deer", "Raccoon")) for l in labels)


def test_entities_and_visits(day_pipeline):
    [truck] = day_pipeline.tracker.find("blue ford f150")
    assert len(truck.visits) == 2
    [intruder] = day_pipeline.tracker.find("black jacket")
    assert {"perimeter fence", "tool shed", "backyard"} <= {z for v in intruder.visits for z in v.zones}


def test_every_frame_indexed(day_pipeline):
    s = day_pipeline.index.stats()
    assert s["frames"] == day_pipeline.stats.frames == 1442
    assert s["alerts"] == len(day_pipeline.alerts.alerts)
    assert s["telemetry"] == day_pipeline.stats.telemetry


def test_memory_context_is_compact_and_informative(day_pipeline):
    ctx = day_pipeline.memory.context_block()
    assert "Blue Ford F150 - 2 visit(s)" in ctx and "CRITICAL" in ctx
    assert len(ctx) < 12_000
    assert day_pipeline.memory.highest_severity() == Severity.CRITICAL


def test_stream_roundtrip(tmp_path):
    from drone_security.pipeline import read_stream, write_stream
    from drone_security.simulator import simulate_day
    recs = simulate_day()
    write_stream(recs, tmp_path / "s.jsonl")
    back = list(read_stream(tmp_path / "s.jsonl"))
    assert len(back) == len(recs)
    assert back[1000].to_dict() == recs[1000].to_dict()


def test_simulation_is_deterministic():
    from drone_security.simulator import simulate_day
    a, b = simulate_day(seed=3), simulate_day(seed=3)
    assert [r.to_dict() for r in a[:300]] == [r.to_dict() for r in b[:300]]
