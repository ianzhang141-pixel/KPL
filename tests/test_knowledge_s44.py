"""S44个人通识资料的来源隔离、降级与术语归一化。"""

from __future__ import annotations

import unittest

from kplab import knowledge_s44


class GeneralKnowledgeIsolation(unittest.TestCase):
    def test_personal_summary_never_becomes_truth_or_model_feature(self):
        data = knowledge_s44.payload()
        self.assertFalse(data["official"])
        self.assertFalse(data["externalVerified"])
        self.assertEqual(data["modelActiveCount"], 0)
        self.assertTrue(all(not item["validationAllowed"] for item in data["entries"]))
        self.assertTrue(all(not item["modelFeatureAllowed"] for item in data["entries"]))

    def test_absolute_role_claims_are_downgraded(self):
        entries = {item["id"]: item for item in knowledge_s44.payload()["entries"]}
        self.assertEqual({key for key, item in entries.items()
                          if item["status"] == "overgeneralized"},
                         {"jungler_role", "mid_role", "roam_role"})

    def test_version_snapshots_are_isolated(self):
        data = knowledge_s44.payload()
        self.assertEqual(data["volatileCount"], 4)
        ranked = next(item for item in data["entries"]
                      if item["id"] == "hero_role_rank_snapshot")
        self.assertEqual(ranked["capturedAt"], "2026-09-07")
        self.assertFalse(ranked["annotationAliasAllowed"])


class GeneralKnowledgeTerms(unittest.TestCase):
    def test_term_aliases_normalize_for_human_annotations_only(self):
        result = knowledge_s44.normalize_term(" 野区 入侵 ")
        self.assertEqual(result["termId"], "jungle_invade")
        self.assertEqual(result["sourceStatus"], "non_official_personal_summary")
        self.assertFalse(result["validationAllowed"])
        self.assertFalse(result["modelFeatureAllowed"])

    def test_unknown_or_non_text_term_is_not_guessed(self):
        self.assertIsNone(knowledge_s44.normalize_term("控双龙"))
        self.assertIsNone(knowledge_s44.normalize_term(None))


if __name__ == "__main__":
    unittest.main()
