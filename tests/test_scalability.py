"""Scalability: throughput, query latency and bounded LLM context on long synthetic streams."""
import time

import pytest

from drone_security.pipeline import SecurityPipeline
from drone_security.simulator import synthetic_stream


@pytest.fixture(scope="module")
def big(cfg=None):
    from drone_security.config import load_config
    p = SecurityPipeline(load_config())
    t0 = time.perf_counter()
    p.run(synthetic_stream(20_000, seed=3))
    return p, time.perf_counter() - t0


def test_throughput(big):
    p, seconds = big
    assert p.stats.frames == 20_000
    assert p.stats.frames / seconds > 1_000          # >1k frames/s on a laptop CPU (VLM excluded)


def test_query_latency(big):
    p, _ = big
    for fn in (lambda: p.index.query_detections(category="truck", limit=5000),
               lambda: p.index.search_text("climbing fence"),
               lambda: p.index.hybrid_search("intruder near the fence")):
        fn()
        t = time.perf_counter()
        assert fn()
        assert time.perf_counter() - t < 0.5


def test_context_stays_bounded(big):
    p, _ = big
    assert len(p.memory.context_block()) < 20_000    # ~5k tokens no matter how long the video is


@pytest.mark.slow
def test_100k_frames_on_disk(tmp_path):
    from drone_security.config import load_config
    p = SecurityPipeline(load_config(), db_path=tmp_path / "big.db")
    p.run(synthetic_stream(100_000, seed=5))
    assert p.index.stats()["frames"] == 100_000
