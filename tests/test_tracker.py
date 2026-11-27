from datetime import datetime, timedelta

from drone_security.models import Detection
from drone_security.tracker import EntityTracker

T0 = datetime(2026, 11, 26, 8, 0)


def veh(minutes, **kw):
    base = dict(frame_id=minutes, ts=T0 + timedelta(minutes=minutes), obj_type="vehicle", color="blue",
                make="Ford", model="F150", category="truck", zone="main gate")
    base.update(kw)
    return Detection(**base)


def test_same_vehicle_same_entity_and_visits_counted():
    t = EntityTracker(visit_gap_s=900)
    u1 = t.update(veh(0))
    u2 = t.update(veh(5, zone="driveway"))
    u3 = t.update(veh(200))          # > 15 min later -> new visit
    assert u1.entity is u2.entity is u3.entity
    assert (u1.new_visit, u2.new_visit, u3.new_visit) == (True, False, True)
    assert u3.visits_today == 2
    assert u2.zone_changed


def test_different_vehicles_are_different_entities():
    t = EntityTracker()
    a = t.update(veh(0)).entity
    b = t.update(veh(1, color="red", make="Toyota", model="Camry", category="sedan")).entity
    assert a is not b


def test_plate_links_sightings_with_and_without_plate():
    t = EntityTracker()
    a = t.update(veh(0, plate="ABC-123")).entity
    b = t.update(veh(2)).entity
    assert a is b


def test_person_dwell_time_and_reid_by_clothing():
    t = EntityTracker()
    p = dict(obj_type="person", zone="main gate", attributes=["dark hoodie"])
    t.update(Detection(1, T0, **p))
    u = t.update(Detection(2, T0 + timedelta(seconds=90), **p))
    assert u.dwell_s == 90 and not u.new_entity
    # carried items change, clothing doesn't -> same person
    u2 = t.update(Detection(3, T0 + timedelta(seconds=120), obj_type="person", zone="main gate",
                            attributes=["dark hoodie", "flashlight"]))
    assert u2.entity is u.entity


def test_find_entities_by_text():
    t = EntityTracker()
    t.update(veh(0))
    assert [e.label for e in t.find("blue ford f150")] == ["Blue Ford F150"]
    assert t.find("green tractor") == []
