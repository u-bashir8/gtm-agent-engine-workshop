import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


class SensitiveProspectDataTests(unittest.TestCase):
    def tearDown(self):
        data_service._PROFILES.clear()

    def test_get_prospect_returns_only_allowlisted_fields(self):
        record = {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "disqualified": False,
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 90000000,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.func("LEAD-1")

        self.assertEqual(result["prospect"], {
            "prospect_id": "LEAD-1",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "disqualified": False,
        })

    def test_new_profile_excludes_billing_qualification(self):
        record = {
            "name": "Ada Lovelace",
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 90000000,
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=[]),
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-1")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES["LEAD-1"])

    def test_cached_profile_is_sanitized_before_return(self):
        data_service._PROFILES["LEAD-1"] = {
            "prospect_id": "LEAD-1",
            "billing_qualification": {"tax_id": "sensitive"},
        }

        result = gtm_agent.build_prospect_profile.func("LEAD-1")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES["LEAD-1"])

    def test_scoring_loads_identifiers_and_sends_only_allowed_profile_fields(self):
        data_service._PROFILES["LEAD-1"] = {
            "prospect_id": "LEAD-1",
            "name": "Ada Lovelace",
            "account_details": [{"account": "Analytical Engines"}],
            "annual_revenue": 90000000,
            "tech_stack": ["Python"],
            "engagement_history": [{"channel": "Inbound"}],
            "billing_qualification": {"tax_id": "sensitive"},
            "email": "ada@example.com",
        }
        offering = {
            "offering_id": "OFFER-1",
            "name": "Platform",
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 50000000,
            "description": "A platform",
        }
        llm_result = SimpleNamespace(model_dump=lambda: {"score": 80})
        scoring_llm = Mock(invoke=Mock(return_value=llm_result))

        with (
            patch.object(data_service, "get_offering", return_value=offering) as get_offering,
            patch.object(gtm_agent, "_scoring_llm", scoring_llm),
        ):
            result = gtm_agent.score_prospect.func("LEAD-1", "OFFER-1")

        self.assertEqual(result, {"score": 80})
        get_offering.assert_called_once_with("OFFER-1")
        self.assertEqual(
            set(gtm_agent.score_prospect.args_schema.model_fields),
            {"prospect_id", "offering_id"},
        )
        scoring_request = scoring_llm.invoke.call_args.args[0]
        request_text = scoring_request[1]["content"]
        scoring_profile = json.loads(request_text.split("Prospect profile:\n", 1)[1])
        self.assertEqual(set(scoring_profile), {
            "name", "account", "annual_revenue", "tech_stack", "engagement_history",
        })
        self.assertEqual(scoring_profile["name"], "Ada Lovelace")
        self.assertEqual(scoring_profile["account"], [{"account": "Analytical Engines"}])
        self.assertEqual(scoring_profile["annual_revenue"], 90000000)
        self.assertEqual(scoring_profile["tech_stack"], ["Python"])
        self.assertEqual(scoring_profile["engagement_history"], [{"channel": "Inbound"}])
        for sensitive_field in ("billing_qualification", "tax_id", "date_of_birth", "card_on_file", "credit_check_ref"):
            self.assertNotIn(sensitive_field, request_text)


if __name__ == "__main__":
    unittest.main()
