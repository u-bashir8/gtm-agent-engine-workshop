import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import gtm_agent


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "required_tech_stack": ["Snowflake", "dbt", "Kafka"],
            "min_annual_revenue": 10_000_000,
            "description": "Data platform",
            "target_segment": "Enterprise",
        }
        self.profile = {"prospect_id": "PROSPECT-1"}
        self.assessment = {
            "justification": "Strong revenue and segment fit.",
            "rubric_breakdown": {
                "revenue_fit": 80,
                "tech_stack_match": 100,
                "segment_fit": 60,
                "component_max": 100,
            },
        }

    def score(self, tech_stack, proposed_score=None, offering=None):
        response = dict(self.assessment)
        response["rubric_breakdown"] = dict(self.assessment["rubric_breakdown"])
        if proposed_score is not None:
            response["score"] = proposed_score
        with (
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=tech_stack),
            patch.object(
                gtm_agent,
                "_scoring_llm",
                SimpleNamespace(invoke=lambda messages: SimpleNamespace(model_dump=lambda: response)),
            ),
        ):
            return gtm_agent.score_prospect.func(self.profile, offering or self.offering)

    def test_identical_components_always_produce_same_total(self):
        first = self.score(["Snowflake", "dbt", "Kafka"], proposed_score=1)
        second = self.score(["Snowflake", "dbt", "Kafka"], proposed_score=99)

        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["score"], 84.0)

    def test_lower_required_technology_coverage_lowers_total(self):
        complete = self.score(["Snowflake", "dbt", "Kafka"])
        partial = self.score(["Snowflake", "dbt"])

        self.assertLess(partial["score"], complete["score"])
        self.assertAlmostEqual(complete["score"] - partial["score"], 13.3)

    def test_technology_match_reflects_fetched_stack(self):
        result = self.score(["Snowflake", "other"])

        self.assertAlmostEqual(result["rubric_breakdown"]["tech_stack_match"], 100 / 3)

    def test_llm_overall_score_is_ignored(self):
        result = self.score(["Snowflake", "dbt", "Kafka"], proposed_score=1)

        self.assertEqual(result["score"], 84.0)

    def test_missing_or_empty_required_technologies_are_rejected_without_division(self):
        for required_tech_stack in (None, []):
            with self.subTest(required_tech_stack=required_tech_stack):
                offering = dict(self.offering)
                if required_tech_stack is None:
                    offering.pop("required_tech_stack")
                else:
                    offering["required_tech_stack"] = required_tech_stack

                result = self.score([], offering=offering)

                self.assertIsNone(result["score"])
                self.assertEqual(result["error"], "Cannot score without a valid offering.")

    def test_prompt_and_output_include_every_rubric_component(self):
        result = self.score(["Snowflake", "dbt", "Kafka"])

        for component in ("revenue_fit", "tech_stack_match", "segment_fit"):
            self.assertIn(component, gtm_agent.SCORING_PROMPT)
            self.assertIn(component, result["rubric_breakdown"])
        self.assertIn("target_segment", gtm_agent.SCORING_PROMPT)
        self.assertIn("computes the overall score", gtm_agent.SCORING_PROMPT)
        self.assertNotIn("score", gtm_agent.ProspectAssessment.model_fields)


if __name__ == "__main__":
    unittest.main()
