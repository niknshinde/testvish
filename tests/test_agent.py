"""Agent Q&A: offline router (follow-ups, intents) and the LangChain tool-calling agent."""
import pytest

from drone_security.agent import OfflineSecurityAgent, SecurityTools, parse_object_filters


@pytest.fixture(scope="module")
def agent(day_pipeline):
    return OfflineSecurityAgent(day_pipeline)


def test_what_objects_were_in_the_video(agent):
    a = agent.ask("What objects were in the video?")
    for word in ("truck", "person", "package", "deer"):
        assert word in a


def test_show_all_truck_events(agent):
    a = agent.ask("show all truck events")
    assert a.count("Blue Ford F150") == 6


def test_how_many_visits_and_follow_up_pronoun(agent):
    a = agent.ask("How many times did the blue Ford F150 come today?")
    assert "2 visit(s)" in a
    follow = agent.ask("When did it leave?")
    assert "08:41" in follow and "12:06" in follow and "exiting" in follow


def test_time_questions(agent):
    a = agent.ask("What happened around midnight?")
    assert "loitering" in a.lower() and "00:01" in a
    b = agent.ask("What happened between 22:45 and 23:10?")
    assert "climbing" in b and "crowbar" in b.lower() or "Perimeter breach" in b


def test_alert_and_recommendation_questions(agent):
    a = agent.ask("Any critical alerts after 22:00?")
    assert "CRITICAL" in a and "12:04" not in a
    r = agent.ask("What should I do next?")
    assert "Blue Ford F150" in r and "suspect" in r.lower()


def test_summary_question(agent):
    a = agent.ask("Summarise the video")
    assert a.startswith("On 26 Nov 2026") and "Blue Ford F150" in a


def test_free_text_falls_back_to_hybrid_search(agent):
    a = agent.ask("person with a flashlight")
    assert "flashlight" in a.splitlines()[0]


def test_parse_object_filters(cfg):
    zones = [z.name for z in cfg.zones]
    assert parse_object_filters("blue ford f150 at the garage", zones) == {
        "obj_type": "vehicle", "model": "F150", "make": "Ford", "color": "blue", "zone": "garage"}
    assert parse_object_filters("any trucks?", zones) == {"obj_type": "vehicle", "category": "truck"}
    assert parse_object_filters("show people", zones)["obj_type"] == "person"


def test_langchain_agent_calls_tools(day_pipeline):
    """Drive the real LangChain create_agent loop with a scripted chat model (no API key needed)."""
    pytest.importorskip("langchain")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, ToolMessage

    from drone_security.agent import LLMSecurityAgent

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            self.__dict__["bound_tools"] = [t.name for t in tools]
            return self

    model = ScriptedModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "find_object_events", "args": {"category": "truck"},
                                           "id": "call_1", "type": "tool_call"}]),
        AIMessage(content="The blue Ford F150 appeared in 6 frames across 2 visits (08:31 and 11:58)."),
    ]))
    agent = LLMSecurityAgent(day_pipeline, chat_model=model)
    answer = agent.ask("show all truck events")
    assert "2 visits" in answer
    tool_msgs = [m for m in agent.messages if isinstance(m, ToolMessage)]
    assert tool_msgs and "Blue Ford F150" in tool_msgs[0].content
    assert set(model.__dict__["bound_tools"]) == set(SecurityTools.TOOL_NAMES)
    assert "Blue Ford F150 - 2 visit(s)" in agent._system_prompt()     # bounded memory injected


def test_llm_agent_falls_back_offline_on_error(day_pipeline):
    pytest.importorskip("langchain")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from drone_security.agent import LLMSecurityAgent

    class Broken(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *a, **k):
            raise ConnectionError("no network")

    agent = LLMSecurityAgent(day_pipeline, chat_model=Broken(messages=iter([])))
    assert "Blue Ford F150" in agent.ask("show all truck events")
