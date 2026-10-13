import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

agent_module = importlib.import_module("gtm_agent.gtm_agent")
SYSTEM_PROMPT = agent_module.SYSTEM_PROMPT
send_prospect_email = agent_module.send_prospect_email


class SendProspectEmailAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {"name": "Alex", "email": "alex@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.rep = {"name": "Sam", "email": "sam@example.com"}

    def send(self, body):
        return send_prospect_email.func(
            self.prospect,
            "Following up",
            body,
            self.runtime,
            self.rep,
        )

    def test_attachment_language_fails_without_sending(self):
        for body in (
            "Attached is our pricing deck.",
            "The ATTACHMENTS are ready.",
        ):
            with self.subTest(body=body), patch("gtm_agent.gtm_agent.uuid.uuid4") as create_id:
                result = self.send(body)

            self.assertEqual(
                result,
                {
                    "status": "failed",
                    "error": "Attachments are not supported; remove references to attachments or ask the rep for a link.",
                },
            )
            create_id.assert_not_called()

    def test_ordinary_email_still_sends(self):
        result = self.send("Would you be available for a short call next week?")

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], "Would you be available for a short call next week?")

    def test_prompt_requires_document_limitation_and_choice_before_sending(self):
        self.assertIn("attachments are unsupported", SYSTEM_PROMPT)
        self.assertIn("offer a text-only email or to include a link the rep provides", SYSTEM_PROMPT)
        self.assertIn("wait for the rep's choice or link before sending", SYSTEM_PROMPT)

    def test_prompt_requires_accurate_tool_failure_reporting(self):
        self.assertIn("If send_prospect_email returns a failure, report it accurately", SYSTEM_PROMPT)
        self.assertIn("do not claim the email or an attachment was sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
