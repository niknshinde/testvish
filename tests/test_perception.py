"""Object/event detection from text frame descriptions (and the VLM payload mapping)."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from drone_security.models import Frame
from drone_security.perception import ClaudeVisionAnalyzer, TextFrameAnalyzer


@pytest.fixture
def analyze(cfg):
    a = TextFrameAnalyzer(cfg)
    return lambda text, zone=None: a.analyze(Frame(1, datetime(2026, 11, 26, 12, 0), text, zone=zone))


def test_vehicle_make_model_colour(analyze):
    [d] = analyze("Blue Ford F150 pickup truck entering through main gate")
    assert (d.obj_type, d.color, d.make, d.model, d.category) == ("vehicle", "blue", "Ford", "F150", "truck")
    assert d.action == "entering" and d.zone == "main gate"
    assert d.label == "Blue Ford F150"


def test_model_without_noun_infers_make_and_category(analyze):
    [d] = analyze("Blue Ford F150 at garage")
    assert d.category == "truck" and d.zone == "garage"


def test_plate_extracted(analyze):
    [d] = analyze("Red Toyota Camry sedan (plate KXP-8821) parked in driveway, engine running")
    assert d.plate == "KXP-8821" and d.action == "parked" and d.make == "Toyota"


def test_person_attributes_and_loitering_action(analyze):
    [d] = analyze("Person in dark hoodie at main gate, looking around")
    assert d.obj_type == "person" and "dark hoodie" in d.attributes and d.action == "loitering"


def test_multiple_objects_and_zone_carry_over(analyze):
    dets = analyze("White delivery van parked at front porch; courier in yellow vest carrying a package")
    kinds = [d.obj_type for d in dets]
    assert kinds == ["vehicle", "person", "package"]
    assert all(d.zone == "front porch" for d in dets)          # 2nd clause inherits the zone
    assert dets[1].category == "courier"


def test_suspicious_item_attached_to_person(analyze):
    [d] = analyze("Person climbing perimeter fence carrying a crowbar")
    assert d.action == "climbing" and "crowbar" in d.attributes and "suspicious" in d.attributes


@pytest.mark.parametrize("text", ["No activity at garage", "No vehicles at main gate", "Empty driveway",
                                  "Parking lot empty", "Garage door closed, no activity"])
def test_quiet_frames_have_no_detections(analyze, text):
    assert analyze(text) == []


def test_animals(analyze):
    [d] = analyze("Deer crossing the backyard")
    assert (d.obj_type, d.category, d.zone) == ("animal", "deer", "backyard")


def test_group_of_people(analyze):
    dets = analyze("Two people walking a dog on the street past the parking lot")
    person = next(d for d in dets if d.obj_type == "person")
    assert "group of 2" in person.attributes
    assert any(d.obj_type == "animal" and d.category == "dog" for d in dets)


def test_frame_zone_used_when_text_has_no_landmark(analyze):
    [d] = analyze("Man in grey jacket walking", zone="driveway")
    assert d.zone == "driveway"


# ---- VLM path (Claude) with a mocked client: no network needed ------------------------------

class _FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=self.payload)])


def _client(payload, stop_reason="end_turn"):
    msgs = _FakeMessages(payload, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_vlm_payload_mapped_to_detections(cfg, tmp_path):
    import json
    payload = json.dumps({"caption": "A blue pickup truck at the front gate at night.", "detections": [
        {"obj_type": "vehicle", "category": "truck", "color": "Blue", "make": None, "model": "f150", "plate": None,
         "action": "entering", "zone": "front gate", "attributes": [], "confidence": 0.82}]})
    client, msgs = _client(payload)
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    f = Frame(7, datetime(2026, 11, 26, 0, 1), "", zone="main gate", image_path=str(img))
    [d] = ClaudeVisionAnalyzer(cfg, client=client).analyze(f)
    assert (d.make, d.model, d.color, d.zone) == ("Ford", "F150", "blue", "main gate")   # alias -> canonical zone
    assert f.caption.startswith("A blue pickup")
    req = msgs.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["messages"][0]["content"][0]["type"] == "image"


def test_vlm_falls_back_to_text_parser_on_refusal_or_error(cfg):
    client, _ = _client("{}", stop_reason="refusal")
    f = Frame(1, datetime(2026, 11, 26, 12), "Blue Ford F150 at garage", zone="garage")
    [d] = ClaudeVisionAnalyzer(cfg, client=client).analyze(f)
    assert d.model == "F150"

    class Boom:
        class beta:
            class messages:
                @staticmethod
                def create(**_):
                    raise ConnectionError("offline")
    [d] = ClaudeVisionAnalyzer(cfg, client=Boom).analyze(f)
    assert d.make == "Ford"
