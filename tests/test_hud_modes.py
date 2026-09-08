import tempfile
import unittest
from pathlib import Path

from kplab import hud


class HudModeTests(unittest.TestCase):
    def test_player_pov_does_not_require_hidden_team_gold(self):
        regions = {
            "clock": [0.7, 0.08, 0.05, 0.04],
            "blueKills": [0.66, 0.08, 0.03, 0.04],
            "redKills": [0.76, 0.08, 0.03, 0.04],
        }
        report = hud.completeness(regions, hud.PROFILE_PLAYER_POV)
        self.assertTrue(report["complete"])
        self.assertNotIn("blueGold", report["required"])
        self.assertNotIn("blueKills", report["required"])

    def test_player_pov_uses_relative_kill_names(self):
        self.assertIn("allyKills", hud.REGIONS)
        self.assertIn("enemyKills", hud.REGIONS)
        self.assertEqual(hud.REGIONS["allyKills"]["read"], "int")

    def test_full_scoreboard_still_requires_both_team_gold_values(self):
        regions = {
            "clock": [0.7, 0.08, 0.05, 0.04],
            "blueKills": [0.66, 0.08, 0.03, 0.04],
            "redKills": [0.76, 0.08, 0.03, 0.04],
        }
        report = hud.completeness(regions, hud.PROFILE_FULL_SCOREBOARD)
        self.assertFalse(report["complete"])
        self.assertEqual(report["missingRequired"], ["blueGold", "redGold"])

    def test_profile_mode_is_persisted_and_controls_verdict(self):
        regions = {
            "clock": [0.7, 0.08, 0.05, 0.04],
            "blueKills": [0.66, 0.08, 0.03, 0.04],
            "redKills": [0.76, 0.08, 0.03, 0.04],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hud.save_profile(root, "player_pov", {
                "name": "选手单人视角",
                "mode": hud.PROFILE_PLAYER_POV,
                "regions": regions,
            })
            saved = hud.get_profile(root, "player_pov")
            self.assertEqual(saved["mode"], hud.PROFILE_PLAYER_POV)
            self.assertTrue(saved["calibrated"])
            self.assertTrue(hud.describe(saved)["ok"])


if __name__ == "__main__":
    unittest.main()
