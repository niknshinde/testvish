import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drone_security.config import load_config  # noqa: E402
from drone_security.models import Frame, Telemetry  # noqa: E402
from drone_security.pipeline import SecurityPipeline  # noqa: E402
from drone_security.simulator import DEFAULT_DAY, simulate_day  # noqa: E402

DAY = DEFAULT_DAY


def at(hms: str, day=DAY) -> datetime:
    p = [int(x) for x in hms.split(":")] + [0]
    return datetime(day.year, day.month, day.day, p[0], p[1], p[2])


class FrameFactory:
    def __init__(self):
        self.n = 0

    def __call__(self, hms: str, description: str, zone: str | None = None, telemetry: Telemetry | None = None):
        self.n += 1
        return Frame(self.n, at(hms), description, zone=zone, telemetry=telemetry)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def frame():
    return FrameFactory()


@pytest.fixture
def pipeline(cfg):
    return SecurityPipeline(cfg)


@pytest.fixture(scope="session")
def day_pipeline():
    """The full simulated day, processed once and shared by the scenario tests."""
    p = SecurityPipeline(load_config())
    p.run(simulate_day())
    return p
