import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from gtm_agent import gtm_agent as gtm

os.environ["LANGSMITH_TRACING"] = "false"


class FakeToolChatModel(BaseChatModel):
    responses: list[AIMessage]
    bound_tools: list = Field(default_factory=list)
    received_messages: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-chat-model"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class GtmAgentTests(unittest.TestCase):
    def test_agent_uses_only_gtm_tools_and_system_prompt(self):
        model = FakeToolChatModel(responses=[AIMessage(content="Ready.")])
        agent = gtm._create_gtm_agent(model)

        result = agent.invoke({"messages": [{"role": "user", "content": "Help me."}]})

        self.assertEqual(
            {tool.name for tool in model.bound_tools},
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        self.assertEqual(len(model.bound_tools), 7)
        system_messages = [
            message for message in model.received_messages[0]
            if isinstance(message, SystemMessage)
        ]
        self.assertEqual([message.content for message in system_messages], [gtm.SYSTEM_PROMPT])
        self.assertEqual(result["messages"][-1].content, "Ready.")

    def test_tool_runtime_is_injected_for_current_rep_tool(self):
        model = FakeToolChatModel(responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_current_rep", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(content="Rep found."),
        ])
        agent = gtm._create_gtm_agent(model)

        with patch.object(gtm.data_service, "get_rep", return_value={"name": "Riley"}) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Who am I?"}]},
                config={"metadata": {"user_id": "rep-1"}},
            )

        get_rep.assert_called_once_with("rep-1")
        tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertIn("Riley", tool_result.content)
        self.assertEqual(result["messages"][-1].content, "Rep found.")

    def test_run_agent_passes_messages_and_config_to_agent(self):
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="Completed.")]}

        with patch.object(gtm, "gtm_agent", agent):
            result = gtm.run_agent("Find the offering", user_id="rep-1", thread_id="thread-1")

        call_args = agent.invoke.call_args
        self.assertEqual(
            call_args.args[0],
            {"messages": [{"role": "user", "content": "Find the offering"}]},
        )
        self.assertEqual(call_args.kwargs["config"]["run_name"], "GTM Assistant")
        self.assertEqual(call_args.kwargs["config"]["metadata"]["user_id"], "rep-1")
        self.assertEqual(call_args.kwargs["config"]["metadata"]["thread_id"], "thread-1")
        self.assertEqual(call_args.kwargs["config"]["metadata"]["environment"], "production")
        self.assertEqual(call_args.kwargs["config"]["metadata"]["request_intent"], "lookup_offering")
        self.assertEqual(result["reply"], "Completed.")
        self.assertEqual(result["run_id"], str(call_args.kwargs["config"]["run_id"]))


if __name__ == "__main__":
    unittest.main()
