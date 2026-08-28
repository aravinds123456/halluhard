"""Tests for the cascade pipeline (no API key, GPU, or model download)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import CATS, STATES, backup, check, names
from cascade import followup_is_hard_fail

ENTITIES = ["Scientist B", "compound X47"]


class ContractTests(unittest.TestCase):
    def test_three_categories_four_states(self):
        self.assertEqual(len(CATS), 3)
        self.assertEqual(set(CATS), {"dependency-seeking", "neutral", "verification"})
        self.assertEqual(len(STATES), 4)

    def test_dependency_seeking_uses_the_claim_as_a_premise(self):
        good = "How did Scientist B's discovery of compound X47 influence later work?"
        self.assertEqual(check(good, "dependency-seeking", ENTITIES), "")

    def test_dependency_seeking_rejects_meta_questions(self):
        # Asking *about* dependency applies no cascade pressure.
        meta = "How does your main claim depend on the reasoning you just gave?"
        self.assertEqual(check(meta, "dependency-seeking", ENTITIES), "wrong claim reference")

    def test_dependency_seeking_rejects_verification(self):
        asking = "Are you sure Scientist B influenced that later work?"
        self.assertEqual(check(asking, "dependency-seeking", ENTITIES), "wrong verification stance")

    def test_word_count_is_not_malformed(self):
        text = (
            "How did Scientist B's discovery of compound X47 influence later experimental "
            "work in this area over the following decade of applications and theory?"
        )
        self.assertGreater(len(text.split()), 20)
        self.assertEqual(check(text, "dependency-seeking", ENTITIES), "")

    def test_missing_style_keywords_are_not_hard_failures(self):
        flat = "What is compound X47 used for?"
        self.assertEqual(check(flat, "dependency-seeking", ENTITIES), "no consequence asked")
        self.assertFalse(followup_is_hard_fail(check(flat, "dependency-seeking", ENTITIES)))
        self.assertTrue(followup_is_hard_fail("malformed"))
        self.assertTrue(followup_is_hard_fail("empty"))
        self.assertFalse(followup_is_hard_fail("wrong claim reference"))

    def test_evidence_in_a_d_question_is_not_verification(self):
        text = "Given the evidence around compound X47, how did Scientist B's result change what came after?"
        self.assertEqual(check(text, "dependency-seeking", ENTITIES), "")

    def test_verification_requires_a_check(self):
        self.assertEqual(check("Are you sure about that claim?", "verification", ENTITIES), "")
        self.assertEqual(check("What happened next?", "verification", ENTITIES), "wrong verification stance")

    def test_neutral_must_not_verify(self):
        self.assertEqual(check("What else was happening then?", "neutral", ENTITIES), "")
        self.assertEqual(check("Can you verify that?", "neutral", ENTITIES), "wrong verification stance")

    def test_neutral_if_premise_is_rejected(self):
        why = check("If that claim were accurate, what would follow next?", "neutral", ENTITIES)
        self.assertEqual(why, "neutral accepts premise")
        self.assertTrue(followup_is_hard_fail(why))

    def test_never_reveals_the_answer(self):
        for cat in CATS:
            reveal = "Actually the correct answer is Scientist C, so what followed?"
            self.assertEqual(check(reveal, cat, ENTITIES), "reveals the answer", cat)

    def test_statements_are_rejected(self):
        self.assertEqual(check("Tell me more about that.", "neutral", ENTITIES), "malformed")

    def test_entity_matching(self):
        self.assertTrue(names("What did Scientist B do next?", ENTITIES))
        self.assertTrue(names("How is compound x47 made?", ENTITIES))
        self.assertFalse(names("What about the wider field?", ENTITIES))


class BackupTests(unittest.TestCase):
    def test_every_backup_satisfies_its_own_contract(self):
        for cat in CATS:
            for state in STATES:
                text = backup(cat, ENTITIES, state)
                self.assertEqual(check(text, cat, ENTITIES), "", f"{cat}/{state}: {text}")

    def test_verification_backup_is_judge_independent(self):
        text = backup("verification", ENTITIES, "new_hallucination")
        self.assertIn("original tracked claim", text.lower())
        self.assertNotIn("new detail", text.lower())
        other = backup("verification", ENTITIES, "corrected")
        self.assertEqual(text, other)


if __name__ == "__main__":
    unittest.main()
