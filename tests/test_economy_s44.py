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

    def test_all_known_conflicts_are_isolated(self):
        conflicts = {entry["id"]: entry for entry in economy_s44.payload()["entries"]
                     if entry["status"] == "conflict"}
        self.assertEqual(set(conflicts), {
            "early_mid_wave_conflict", "red_falcon_gold_conflict",
            "tyrant_respawn_conflict", "vision_spirit_respawn_conflict",
        })
        self.assertTrue(all(not entry["estimateAllowed"] for entry in conflicts.values()))
        self.assertEqual(conflicts["tyrant_respawn_conflict"]["values"]
                         ["currentConfirmedTyrantRespawnSec"], 210)


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
