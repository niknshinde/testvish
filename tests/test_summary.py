from drone_security.summarizer import correlate_incidents, one_sentence_summary, recommendations


def test_one_sentence_summary(day_pipeline):
    s = one_sentence_summary(day_pipeline)
    assert s.count(". ") == 0 and s.endswith(".")            # a single sentence
    assert "1,442 frames" in s and "critical" in s and "Blue Ford F150 visited 2 times" in s


def test_cross_event_correlation_links_tampering_to_truck(day_pipeline):
    [finding] = [f for f in correlate_incidents(day_pipeline) if "garage" in f]
    assert "visit 2 of the Blue Ford F150" in finding


def test_recommendations_are_prioritised(day_pipeline):
    recs = recommendations(day_pipeline)
    assert recs[0].startswith("Treat as a suspect vehicle")
    assert any("Battery ran low" in r for r in recs) or any("battery" in r.lower() for r in recs)


def test_empty_summary(pipeline):
    assert one_sentence_summary(pipeline) == "No footage has been processed yet."
