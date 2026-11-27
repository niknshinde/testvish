"""Drone Security Analyst Agent - prototype.

Processes drone telemetry and (simulated or real) video frames, detects and
tracks objects, raises real-time security/safety alerts, indexes every frame
for search, and answers follow-up questions through an analyst agent.
"""

from .config import SiteConfig, load_config
from .index import FrameIndex
from .models import Alert, Detection, Frame, LogEntry, Severity, Telemetry
from .pipeline import SecurityPipeline

__all__ = ["SiteConfig", "load_config", "FrameIndex", "Alert", "Detection", "Frame", "LogEntry", "Severity",
           "Telemetry", "SecurityPipeline"]
__version__ = "0.1.0"
