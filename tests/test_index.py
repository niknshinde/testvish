"""Frame-by-frame index: structured, time, full-text, vector and hybrid queries; persistence."""
from drone_security.index import FrameIndex, HashingEmbedder
from drone_security.pipeline import SecurityPipeline


def test_show_all_truck_events(day_pipeline):
    rows = day_pipeline.index.query_detections(category="truck")
    assert len(rows) == 6
    assert {r["label"] for r in rows} == {"Blue Ford F150"}
    assert [r["ts"][11:16] for r in rows] == ["08:31", "08:33", "08:41", "11:58", "12:00", "12:06"]


def test_query_by_time_window(day_pipeline):
    frames = day_pipeline.index.frames_between("00:00", "00:01")
    assert [f["frame_id"] for f in frames] == [1, 2, 3, 4, 5]
    dets = day_pipeline.index.query_detections(start="22:40", end="23:10", obj_type="person")
    assert {d["zone"] for d in dets} >= {"perimeter fence", "tool shed"}


def test_query_by_attributes(day_pipeline):
    idx = day_pipeline.index
    assert {r["label"] for r in idx.query_detections(color="red", obj_type="vehicle")} == {"Red Toyota Camry"}
    assert len(idx.query_detections(make="ford", model="f-150")) == 6           # normalised matching
    assert all(r["action"] == "climbing" for r in idx.query_detections(action="climbing"))


def test_alert_queries(day_pipeline):
    crit = day_pipeline.index.alerts(min_severity=4)
    assert crit and all(a["severity_name"] == "CRITICAL" for a in crit)
    night = day_pipeline.index.alerts(start="22:00", end="23:59")
    assert any(a["rule_id"] == "perimeter_breach" for a in night)


def test_full_text_search_bm25(day_pipeline):
    rows = day_pipeline.index.search_text("crowbar")
    assert {r["frame_id"] for r in rows} == {1372, 1388}
    assert day_pipeline.index.search_text("hoodie gate")[0]["zone"] == "main gate"


def test_semantic_search_handles_synonyms(day_pipeline):
    # 'pickup' never appears in the 12:00 frame text, the embedding links it to F150/truck
    top = day_pipeline.index.semantic_search("pickup", limit=10)
    assert any("F150" in r["description"] for r in top)
    top = day_pipeline.index.semantic_search("intruder breach", limit=5)
    assert any("climbing" in r["description"] for r in top)


def test_hybrid_search(day_pipeline):
    rows = day_pipeline.index.hybrid_search("person climbing the fence with a tool", limit=5)
    assert rows[0]["frame_id"] in {1372, 1391}


def test_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=128)
    a, b = e.embed(["Blue truck at gate"]), e.embed(["Blue truck at gate"])
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5
    assert (a == b).all()


def test_index_persists_to_disk(tmp_path, cfg, frame):
    db = tmp_path / "idx.db"
    p = SecurityPipeline(cfg, db_path=db)
    p.process_frame(frame("12:00:00", "Blue Ford F150 at garage"))
    p.index.close()
    idx = FrameIndex(db)
    assert idx.stats()["frames"] == 1
    assert idx.query_detections(category="truck")[0]["zone"] == "garage"
    assert idx.search_text("garage")
