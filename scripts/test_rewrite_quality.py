#!/usr/bin/env python3
"""Offline rewrite quality checks (filters, prompts, golden expectations)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from writing_agent import (  # noqa: E402
    apply_rewrite_hard_filters,
    build_rewrite_system_instruction,
    build_rewrite_user_message,
    check_rewrite_quality,
    _restore_missing_closing_lines,
)

GOLDEN_PATH = ROOT / "test_data" / "rewrite_golden.json"


class RewriteQualityTests(unittest.TestCase):
    def test_prompt_includes_strict_rules_and_examples(self) -> None:
        system = build_rewrite_system_instruction(
            "Rewrite in a casual, natural tone.",
            direct=True,
        )
        user = build_rewrite_user_message("Please submit the form by Friday.")
        self.assertIn("REWRITE RULES", system)
        self.assertIn("EXAMPLE (casual)", system)
        self.assertIn("never add", system.lower())
        self.assertIn("RULE 1 — TONE", system)
        self.assertEqual(user, "Please submit the form by Friday.")
        self.assertNotIn("REWRITE RULES", user)

    def test_direction_flip_detected(self) -> None:
        # "Send me X" (recipient acts) must not become "I'll send you X".
        flipped = check_rewrite_quality(
            "Send me the report now. I need it.",
            "I'll send you the report right away since that's urgent for you.",
            "friendly",
        )
        self.assertIn("direction_flip", flipped["issues"])
        kept = check_rewrite_quality(
            "Send me the report now. I need it.",
            "Could you please send me the report soon? I really need it.",
            "friendly",
        )
        self.assertNotIn("direction_flip", kept["issues"])
        writer_original = check_rewrite_quality(
            "I'll send you the file tonight.",
            "I will send you the file tonight.",
            "formal",
        )
        self.assertNotIn("direction_flip", writer_original["issues"])

    def test_restore_missing_closing_lines(self) -> None:
        original = (
            "Dear team,\n\nPlease review the attached document by Friday.\n\n"
            "Thanks,\nJordan"
        )
        rewritten = "Dear team,\n\nPlease review the attached document by Friday."
        restored = _restore_missing_closing_lines(original, rewritten)
        self.assertIn("Thanks", restored)
        self.assertIn("Jordan", restored)

    def test_filters_do_not_strip_original_signoff(self) -> None:
        original = (
            "Dear team,\n\nPlease review the attached document by Friday.\n\n"
            "Thanks,\nJordan"
        )
        rewritten = (
            "Dear team,\n\nKindly review the attached document by Friday.\n\n"
            "Thanks,\nJordan"
        )
        filtered = apply_rewrite_hard_filters(
            original,
            rewritten,
            instruction="make it more professional",
        )
        self.assertIn("Jordan", filtered)
        self.assertIn("Thanks", filtered)

    def test_filters_strip_added_filler(self) -> None:
        original = "Please submit the form by Friday."
        rewritten = (
            "Hey! Just wanted to reach out — please submit the form by Friday. "
            "Hope you're doing well!"
        )
        filtered = apply_rewrite_hard_filters(
            original,
            rewritten,
            instruction="Rewrite in a warm and friendly tone.",
        )
        self.assertNotIn("reach out", filtered.lower())
        self.assertNotIn("hope you", filtered.lower())

    def test_concise_allows_shorter_output(self) -> None:
        original = "I am writing to inform you that the meeting has been rescheduled to 3pm."
        rewritten = "The meeting is rescheduled to 3pm."
        quality = check_rewrite_quality(
            original,
            rewritten,
            "Rewrite to be more concise.",
        )
        self.assertTrue(quality["ok"], quality["issues"])

    def test_filters_strip_prepended_opener_on_short_greeting(self) -> None:
        original = "Hi there"
        rewritten = "Hey! Just checking in — hi there."
        filtered = apply_rewrite_hard_filters(
            original,
            rewritten,
            instruction="Rewrite in a casual, natural tone.",
        )
        self.assertNotIn("checking in", filtered.lower())
        self.assertIn("hi there", filtered.lower())

    def test_filters_strip_added_multiline_content(self) -> None:
        original = "Hey team!"
        rewritten = "Dear team,\n\nPlease review the doc."
        filtered = apply_rewrite_hard_filters(
            original,
            rewritten,
            instruction="Rewrite in a formal tone.",
        )
        self.assertNotIn("review", filtered.lower())

    def test_filters_restore_original_casing(self) -> None:
        original = "Heads up — office closed Monday"
        rewritten = "Hey! Just wanted to reach out — heads up, office closed Monday."
        filtered = apply_rewrite_hard_filters(
            original,
            rewritten,
            instruction="Rewrite in a casual, natural tone.",
        )
        self.assertTrue(filtered.startswith("Heads up"))

    def test_golden_expectations_on_filter_outputs(self) -> None:
        if not GOLDEN_PATH.exists():
            self.skipTest("golden file missing")

        cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        for case in cases:
            case_id = case["id"]
            original = case["input"]
            prompt = case["prompt"]
            # Simulate a model that rewrites body but drops sign-off.
            if case_id == "email_signoff_preserved":
                model_output = "Dear team,\n\nKindly review the attached document by Friday."
            elif case_id == "casual_contractions":
                model_output = "Heads up — the office is closed on Monday."
            elif case_id == "formal_request":
                model_output = "Could you please send the file at your earliest convenience?"
            elif case_id == "concise_padding":
                model_output = "The meeting is rescheduled to 3pm."
            elif case_id == "friendly_no_filler":
                model_output = "Could you submit the form by Friday?"
            else:
                model_output = "We need to speed up buying so we don't fall further behind."

            output = apply_rewrite_hard_filters(
                original,
                model_output,
                instruction=prompt,
            )

            for token in case.get("must_contain", []):
                self.assertIn(token, output, case_id)
            for token in case.get("must_not_contain", []):
                self.assertNotIn(token.lower(), output.lower(), case_id)
            for token in case.get("preserve_lines", []):
                self.assertIn(token, output, case_id)

            ratio = len(output.split()) / max(len(original.split()), 1)
            self.assertGreaterEqual(ratio, case.get("length_ratio_min", 0.0), case_id)
            self.assertLessEqual(ratio, case.get("length_ratio_max", 99.0), case_id)


if __name__ == "__main__":
    unittest.main()
