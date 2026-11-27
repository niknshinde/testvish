"""The Security Analyst agent: answers questions and follow-ups about the footage.

Architecture
------------
* :class:`SecurityTools` - the agent's tool surface: thin, typed wrappers over
  the frame index, tracker and summariser. Every answer is grounded in these.
* :class:`LLMSecurityAgent` - a LangChain ``create_agent`` tool-calling agent on
  Claude. The bounded memory block (alerts, entities, hourly episodes) is put in
  the system prompt; details are fetched with tools. Conversation history is
  kept so follow-ups ("when did it leave?") work.
* :class:`OfflineSecurityAgent` - a deterministic intent router over the same
  tools. Used when no API key is configured, in CI, and as a fallback if the
  LLM call fails. It also resolves simple follow-up pronouns.

``make_agent()`` picks the LLM agent when ``ANTHROPIC_API_KEY`` is available.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from .models import Severity
from .perception import COLORS, VEHICLE_MAKES, VEHICLE_MODELS, VEHICLE_NOUNS, _MODEL_CANONICAL
from .summarizer import one_sentence_summary, recommendations

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import SecurityPipeline

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _fmt_det(r: dict) -> str:
    return (f"{r['ts'][11:16]} frame#{r['frame_id']} {r['label']} ({r['obj_type']}"
            f"{'/' + r['category'] if r.get('category') else ''}) {r.get('action') or 'seen'} at {r['zone']}"
            f"{' [' + r['entity_id'] + ']' if r.get('entity_id') else ''}")


def _fmt_frame(r: dict) -> str:
    return f"{r['ts'][11:19]} frame#{r['frame_id']} [{r['zone']}] {r.get('caption') or r['description']}"


class SecurityTools:
    """Grounded read-only tools over a processed run. Return values are plain strings for the LLM."""

    def __init__(self, pipeline: "SecurityPipeline"):
        self.p = pipeline
        self.index = pipeline.index

    def search_frames(self, query: str, limit: int = 10) -> str:
        """Hybrid keyword + semantic search over indexed frame descriptions. Use for free-text questions
        like 'person near the gate with a flashlight' or 'pickup truck'."""
        self.index.flush()
        rows = self.index.hybrid_search(query, limit=limit)
        return "\n".join(_fmt_frame(r) for r in rows) or "No matching frames."

    def find_object_events(self, obj_type: Optional[str] = None, category: Optional[str] = None,
                           color: Optional[str] = None, make: Optional[str] = None, model: Optional[str] = None,
                           zone: Optional[str] = None, action: Optional[str] = None,
                           start: Optional[str] = None, end: Optional[str] = None, limit: int = 50) -> str:
        """Structured lookup of detections. obj_type: person|vehicle|animal|package|object.
        category e.g. truck, van, sedan, suv, courier, deer. start/end as 'HH:MM'. Example: show all truck
        events -> category='truck'."""
        self.index.flush()
        rows = self.index.query_detections(obj_type=obj_type, category=category, color=color, make=make,
                                           model=model, zone=zone, action=action, start=start, end=end, limit=limit)
        return "\n".join(_fmt_det(r) for r in rows) or "No matching detections."

    def activity_between(self, start: str, end: str, zone: Optional[str] = None) -> str:
        """Everything detected between two times ('HH:MM'), optionally in one zone, plus alerts raised."""
        self.index.flush()
        dets = self.index.query_detections(start=start, end=end, zone=zone, limit=200)
        alerts = self.index.alerts(start=start, end=end)
        out = [_fmt_det(r) for r in dets] or ["No detections in that window (routine patrol frames only)."]
        if alerts:
            out.append("Alerts:")
            out += [f"{a['ts'][11:16]} {a['severity_name']} {a['message']}" for a in alerts]
        return "\n".join(out)

    def list_alerts(self, min_severity: str = "INFO", start: Optional[str] = None, end: Optional[str] = None) -> str:
        """Alerts raised, filtered by minimum severity (INFO, LOW, MEDIUM, HIGH, CRITICAL) and time."""
        self.index.flush()
        rows = self.index.alerts(min_severity=int(Severity.parse(min_severity)), start=start, end=end)
        return "\n".join(f"{a['alert_id']} {a['ts'][11:16]} {a['severity_name']} [{a['rule_id']}] {a['message']}"
                         + (f" Details: {a['details']}" if a.get("details") else "")
                         + (f" Recommended: {a['recommendation']}" if a.get("recommendation") else "")
                         for a in rows) or "No alerts."

    def entity_history(self, description: str) -> str:
        """History of a tracked person/vehicle, e.g. 'blue ford f150' or 'black jacket': visits, times, zones."""
        ents = self.p.tracker.find(description)
        if not ents:
            return f"No tracked entity matches '{description}'."
        lines = []
        for e in ents[:5]:
            lines.append(e.summary() + (" [authorised]" if e.authorized else ""))
            for i, v in enumerate(e.visits, 1):
                lines.append(f"  visit {i}: {v.start:%H:%M}-{v.last_seen:%H:%M}, zones {', '.join(v.zones)}, "
                             f"frames {v.frames[:8]}")
        return "\n".join(lines)

    def object_inventory(self) -> str:
        """What kinds of objects appeared in the video, with counts of detections and distinct entities."""
        self.index.flush()
        rows = self.index.object_counts()
        lines = [f"{r['category']} ({r['obj_type']}): {r['entities']} distinct, {r['detections']} detections"
                 for r in rows]
        ents = [f"{e.entity_id}: {e.label}" for e in self.p.tracker.entities.values()]
        return "\n".join(lines + ["Entities: " + "; ".join(ents)]) if rows else "No objects detected."

    def video_summary(self) -> str:
        """One-sentence summary of the whole video plus hourly episodes."""
        return one_sentence_summary(self.p) + "\n" + "\n".join(self.p.memory.episodic_lines())

    def get_recommendations(self) -> str:
        """Prioritised recommended actions based on alerts and cross-event correlations."""
        return "\n".join(f"- {r}" for r in recommendations(self.p)) or "No action needed."

    def get_frame(self, frame_id: int) -> str:
        """Full record of one frame: time, zone, telemetry, description and detections."""
        r = self.index.frame(int(frame_id))
        return json.dumps(r, default=str) if r else f"Frame {frame_id} not found."

    TOOL_NAMES = ["search_frames", "find_object_events", "activity_between", "list_alerts", "entity_history",
                  "object_inventory", "video_summary", "get_recommendations", "get_frame"]


# ---------------------------------------------------------------------------
# LLM agent (LangChain + Claude)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Security Analyst for a docked drone that patrols {site}. You analyse what the
drone saw (frame descriptions / VLM captions, telemetry, rule-based alerts) and answer the property owner's
questions.

How to work:
- Ground every claim in tool results or the context below; cite times (HH:MM) and frame numbers.
- Use the tools for anything not in the context: find_object_events for structured object questions,
  activity_between for time windows, search_frames for free-text, entity_history for "how many times / when did
  X", get_recommendations for what to do next.
- When asked for recommendations, prioritise human safety, then property, then drone operations; be specific
  (who/what/where/when) and concise.
- If the footage does not contain the answer, say so. Never invent plates, identities or events.

=== Current context (bounded memory; fetch details with tools) ===
{context}
"""


class LLMSecurityAgent:
    def __init__(self, pipeline: "SecurityPipeline", model: str = DEFAULT_MODEL, chat_model=None,
                 effort: str = "medium"):
        from langchain.agents import create_agent
        from langchain_core.tools import StructuredTool

        self.p = pipeline
        self.tools = SecurityTools(pipeline)
        lc_tools = [StructuredTool.from_function(getattr(self.tools, n)) for n in SecurityTools.TOOL_NAMES]
        if chat_model is None:
            from langchain_anthropic import ChatAnthropic

            chat_model = ChatAnthropic(model=model, max_tokens=16000, output_config={"effort": effort})
        self.graph = create_agent(chat_model, lc_tools, system_prompt=self._system_prompt())
        self.messages: list[Any] = []
        self.fallback = OfflineSecurityAgent(pipeline)

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(site=self.p.config.name, context=self.p.memory.context_block())

    def ask(self, question: str) -> str:
        from langchain_core.messages import HumanMessage

        self.messages.append(HumanMessage(question))
        try:
            state = self.graph.invoke({"messages": self.messages})
        except Exception as exc:  # API/network failure: answer offline rather than fail the user
            log.warning("LLM agent failed (%s); answering offline", exc)
            self.messages.pop()
            return self.fallback.ask(question)
        self.messages = list(state["messages"])
        final = self.messages[-1]
        content = final.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return str(content).strip()


# ---------------------------------------------------------------------------
# Offline agent (deterministic intent router)
# ---------------------------------------------------------------------------

_TIME = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|midnight|noon)"


def _norm_time(s: str) -> str:
    s = s.strip().lower()
    if s == "midnight":
        return "00:00"
    if s == "noon":
        return "12:00"
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        return s
    h, mi, ap = int(m[1]), int(m[2] or 0), m[3]
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{mi:02d}"


def _shift(hhmm_s: str, minutes: int) -> str:
    h, m = map(int, hhmm_s.split(":"))
    t = max(0, min(23 * 60 + 59, h * 60 + m + minutes))
    return f"{t // 60:02d}:{t % 60:02d}"


OBJ_WORDS = {
    "person": ("person", None), "people": ("person", None), "persons": ("person", None), "man": ("person", None),
    "woman": ("person", None), "intruder": ("person", None), "courier": ("person", "courier"),
    "vehicle": ("vehicle", None), "vehicles": ("vehicle", None), "car": ("vehicle", None), "cars": ("vehicle", None),
    "animal": ("animal", None), "animals": ("animal", None), "wildlife": ("animal", None),
    "package": ("package", None), "packages": ("package", None), "parcel": ("package", None),
    "deer": ("animal", "deer"), "fox": ("animal", "fox"), "dog": ("animal", "dog"), "cat": ("animal", "cat"),
    "raccoon": ("animal", "raccoon"),
}


def parse_object_filters(text: str, zones: list[str]) -> dict[str, str]:
    t = text.lower()
    f: dict[str, str] = {}
    for model, (make, cat) in VEHICLE_MODELS.items():
        if re.search(rf"\b{re.escape(model)}\b", t) or (model == "f150" and "f-150" in t):
            f.update(obj_type="vehicle", model=_MODEL_CANONICAL.get(model, model.title()), make=make)
            break
    for w, canon in VEHICLE_MAKES.items():
        if re.search(rf"\b{w}\b", t):
            f.update(obj_type="vehicle", make=canon)
    for noun, cat in sorted(VEHICLE_NOUNS.items(), key=lambda kv: -len(kv[0])):
        if cat and re.search(rf"\b{re.escape(noun)}s?\b", t) and "model" not in f:
            f.update(obj_type="vehicle", category=cat)
            break
    if "obj_type" not in f:
        for w, (ot, cat) in OBJ_WORDS.items():
            if re.search(rf"\b{w}\b", t):
                f["obj_type"] = ot
                if cat:
                    f["category"] = cat
                break
    for c in COLORS - {"dark"}:
        if re.search(rf"\b{c}\b", t) and f.get("obj_type") == "vehicle":
            f["color"] = "grey" if c == "gray" else c
    for z in sorted(zones, key=len, reverse=True):
        if z in t:
            f["zone"] = z
            break
    return f


_ACTION_WORDS = {"climb": "climbing", "enter": "entering", "leav": "exiting", "exit": "exiting",
                 "park": "parked", "loiter": "loitering", "tamper": "tampering", "deliver": "delivering"}
_STOPWORDS = set("""a an the all any show list find me give get display what which who where when were was is are
there did do does of in on at to by for from with near around today video footage events event sightings seen
spotted detections detected please and or every times""".split())


class OfflineSecurityAgent:
    """Rule-based question answering over the same tools (no LLM required)."""

    def __init__(self, pipeline: "SecurityPipeline"):
        self.p = pipeline
        self.tools = SecurityTools(pipeline)
        self.zones = [z.name for z in pipeline.config.zones]
        self.last_filters: dict[str, str] = {}

    def ask(self, question: str) -> str:
        q = question.lower().strip()
        filters = parse_object_filters(q, self.zones)
        follow_up = bool(re.search(r"\b(it|its|they|them|that (car|truck|vehicle|person)|he|she|this one)\b", q))
        if follow_up and not filters.get("obj_type") and self.last_filters:
            filters = {**self.last_filters, **filters}
        if filters.get("obj_type"):
            self.last_filters = filters

        # time windows: "between 22:00 and 23:30", "after 22:00", "around midnight", "at 12:00"
        m = re.search(rf"between\s+{_TIME}\s+(?:and|to|-)\s+{_TIME}", q) or \
            re.search(rf"from\s+{_TIME}\s+(?:to|until|-)\s+{_TIME}", q)
        window = None
        if m:
            window = (_norm_time(m[1]), _norm_time(m[2]))
        elif (m := re.search(rf"\b(?:after|since)\s+{_TIME}", q)):
            window = (_norm_time(m[1]), "23:59")
        elif (m := re.search(rf"\b(?:before|until)\s+{_TIME}", q)):
            window = ("00:00", _norm_time(m[1]))
        elif (m := re.search(rf"\b(?:at|around|near)\s+{_TIME}", q)) and re.search(r"\d|midnight|noon", m[1]):
            c = _norm_time(m[1])
            window = (_shift(c, -15), _shift(c, 15))
        elif re.search(r"\b(night|overnight)\b", q):
            window = None  # handled below via alerts/night filter

        if re.search(r"\bsummar|\bwhat happened today|\boverview|\btl;?dr", q):
            return self.tools.video_summary()
        if re.search(r"recommend|what should|next step|what to do|action", q):
            return "Recommended actions:\n" + self.tools.get_recommendations()
        if re.search(r"\b(alerts?|incidents?|threats?|suspicious|dangerous|emergenc)", q):
            sev = "INFO"
            for s in ("critical", "high", "medium", "low"):
                if s in q:
                    sev = s.upper()
            start, end = window if window else (None, None)
            return self.tools.list_alerts(min_severity=sev, start=start, end=end)
        if re.search(r"what (objects|things|kinds?)|which objects|inventory|what did (you|the drone) see|"
                     r"objects (were|are) in", q):
            return self.tools.object_inventory()
        if re.search(r"\bwhen did\b.*\b(leave|left|exit|depart|go)\b", q) and filters.get("obj_type"):
            return self.tools.find_object_events(action="exiting", **filters)
        if re.search(r"\bwhen did\b.*\b(arrive|enter|come|get here)\b", q) and filters.get("obj_type"):
            return self.tools.find_object_events(action="entering", **filters)
        if re.search(r"how many times|how often|visits?|twice|come back|return", q) \
                and filters.get("obj_type"):
            desc = " ".join(filter(None, [filters.get("color"), filters.get("make"), filters.get("model"),
                                          filters.get("category") if not filters.get("model") else None]))
            return self.tools.entity_history(desc or filters["obj_type"])
        if window:
            if filters.get("obj_type"):
                return self.tools.find_object_events(start=window[0], end=window[1],
                                                     **{k: v for k, v in filters.items()})
            return self.tools.activity_between(window[0], window[1], zone=filters.get("zone"))
        if (m := re.search(r"frame\s*#?\s*(\d+)", q)):
            return self.tools.get_frame(int(m[1]))
        action = next((a for w, a in _ACTION_WORDS.items() if re.search(rf"\b{w}", q)), None)
        if action:
            filters["action"] = action
        if filters.get("obj_type") and not self._leftover_words(q, filters):
            return self.tools.find_object_events(**filters)
        return self.tools.search_frames(question)

    def _leftover_words(self, q: str, filters: dict[str, str]) -> list[str]:
        """Content words not explained by the structured filters -> needs free-text search instead."""
        explained = {str(v).lower() for v in filters.values()} | set(OBJ_WORDS) | set(VEHICLE_NOUNS) \
            | set(VEHICLE_MAKES) | {w for z in self.zones for w in z.split()} | set(COLORS)
        words = re.findall(r"[a-z0-9-]+", q)
        return [w for w in words if w not in _STOPWORDS and w not in explained and w.rstrip("s") not in explained
                and not any(w.startswith(a) for a in _ACTION_WORDS)]


def make_agent(pipeline: "SecurityPipeline", prefer_llm: bool = True, model: str = DEFAULT_MODEL):
    """LLM agent if credentials are present and LangChain is installed, otherwise the offline agent."""
    if prefer_llm and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        try:
            return LLMSecurityAgent(pipeline, model=model)
        except ImportError as exc:
            log.warning("LangChain/Anthropic not installed (%s); using offline agent", exc)
    return OfflineSecurityAgent(pipeline)
