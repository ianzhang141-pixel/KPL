"""S44非官方经济参考资料的来源、冲突隔离与估算公式。"""

from __future__ import annotations

import unittest

from kplab import economy_s44


class EconomyProvenance(unittest.TestCase):
    def test_non_official_source_never_becomes_validation_truth(self):
        payload = economy_s44.payload()
        self.assertFalse(payload["official"])
        self.assertFalse(payload["externalVerified"])
        self.assertTrue(all(not entry["validationAllowed"] for entry in payload["entries"]))
        ids = [entry["id"] for entry in payload["entries"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_non_official_conflicts_are_superseded_by_official_rules(self):
        data = economy_s44.payload()
        superseded = {entry["id"]: entry for entry in data["entries"]
                      if entry["status"] == "superseded_by_official"}
        self.assertEqual(set(superseded), {
            "early_mid_wave_conflict", "red_falcon_gold_conflict",
            "tyrant_respawn_conflict", "vision_spirit_respawn_conflict",
        })
        self.assertEqual(data["conflictCount"], 0)
        self.assertEqual(data["supersededCount"], 4)
        self.assertTrue(all(not entry["estimateAllowed"] for entry in superseded.values()))
        self.assertEqual(superseded["tyrant_respawn_conflict"]["values"]
                         ["officialTyrantRespawnSec"], 210)

    def test_official_authority_outranks_non_official_economy_notes(self):
        from kplab import knowledge_s44, season_s44

        official = season_s44.payload()
        reference = economy_s44.payload()
        self.assertTrue(official["official"])
        self.assertFalse(official["externalVerified"])
        self.assertGreater(official["authorityRank"], reference["authorityRank"])
        self.assertGreater(official["authorityRank"], knowledge_s44.payload()["authorityRank"])


class EconomyReferenceCalculations(unittest.TestCase):
    def test_natural_gold_uses_inclusive_tick_at_30_seconds(self):
        self.assertEqual(economy_s44.natural_gold_at(29)["gold"], 0)
        self.assertEqual(economy_s44.natural_gold_at(30)["gold"], 3)
        self.assertEqual(economy_s44.natural_gold_at(210)["gold"], 543)

    def test_two_player_unit_split_matches_document_formula(self):
        result = economy_s44.shared_unit_gold(100, 2, last_hitter=1)
        self.assertEqual(result["perHeroRaw"], [130, 80])
        self.assertEqual(result["teamTotalRaw"], 210)
        self.assertEqual(result["sourceStatus"], "non_official_reference")

    def test_tower_gold_separates_team_base_from_nearby_bonus(self):
        result = economy_s44.tower_gold("highGround", nearby_heroes=4)
        self.assertEqual(result["nearbyHeroRaw"], 90)
        self.assertEqual(result["remoteHeroRaw"], 80)
        self.assertEqual(result["teamTotalRaw"], 440)

    def test_invalid_inputs_are_rejected(self):
        for call in (
            lambda: economy_s44.natural_gold_at(float("nan")),
            lambda: economy_s44.shared_unit_gold(100, 0),
            lambda: economy_s44.tower_gold("crystal", 1),
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
