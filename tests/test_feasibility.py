"""样本量与因果可行性的测试。

盯的是三件会让人得出错误结论的事：

1. 把 Q(s,a) 的样本单位算成「帧」而不是「动作实例」
   （我第一版就是这么错的，结果把需求低估了 5 倍）
2. 在没有动作定义的情况下声称能评估重叠性
3. 把「结构断裂」当成「平滑漂移」，以为降权就能解决
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import feasibility, model, rules, schema, state  # noqa: E402


def manual(value):
    return schema.field(value, schema.SOURCE_MANUAL)


def make_games(count, blue_wins_every=2):
    rows = [{"atSec": 60 * i,
             "fields": {"blueGold": manual(3000 + i * 2000),
                        "redGold": manual(2900 + i * 1800),
                        "blueKills": manual(i), "redKills": manual(max(0, i - 1))}}
            for i in range(1, 15)]
    return [state.build({"gameId": f"G{g}", "blueWin": g % blue_wins_every == 0},
                        rows, rules.load()) for g in range(count)]


class ActionInstancesAreNotFrames(unittest.TestCase):
    """Q(s,a) 的样本单位是「动作发生了几次」，不是「有多少帧」。

    **这条测试来自我自己犯的一个错。** 第一版把每场的帧数（20）
    直接拿去除格子需求，得出「Q(s,a) 只比 V(s) 贵一倍」。

    实际上像「入侵野区」这种动作一场只发生三五次，
    每场提供的动作样本是个位数。按帧算会把需求低估近一个数量级 ——
    而且这种低估最危险：它会让人以为「几百场就够做因果推断了」。
    """

    def test_sparser_actions_need_more_games(self):
        rare = feasibility.required_games(action_instances_per_game=1.0)
        common = feasibility.required_games(action_instances_per_game=20.0)
        self.assertGreater(rare["qsa"]["games"], common["qsa"]["games"] * 5,
                           "动作越稀疏需要的场次应该显著增加")

    def test_qsa_is_much_more_expensive_than_vs(self):
        need = feasibility.required_games(target_pp=3.0, frames_per_game=20,
                                          action_instances_per_game=4.0)
        self.assertGreater(need["ratio"], 5,
                           "Q(s,a) 只比 V(s) 贵几倍 —— 多半是把动作实例当成帧算了")

    def test_frames_per_game_does_not_affect_qsa(self):
        """帧数只影响 V(s)，不影响 Q(s,a) —— 两者的样本单位不同。"""
        a = feasibility.required_games(frames_per_game=10, action_instances_per_game=4.0)
        b = feasibility.required_games(frames_per_game=40, action_instances_per_game=4.0)
        self.assertEqual(a["qsa"]["games"], b["qsa"]["games"])
        self.assertNotEqual(a["vs"]["games"], b["vs"]["games"])

    def test_finer_action_definitions_cost_more(self):
        coarse = feasibility.required_games(strata=10, actions=3)
        fine = feasibility.required_games(strata=40, actions=6)
        self.assertGreater(fine["qsa"]["games"], coarse["qsa"]["games"] * 4)

    def test_note_states_the_sample_unit_explicitly(self):
        note = feasibility.required_games()["note"]
        self.assertIn("动作实例", note)
        self.assertIn("不是", note)


class OverlapNeedsActionDefinitions(unittest.TestCase):
    """没有动作定义就说不了重叠性 —— 这是因果推断能不能做的关键。"""

    def test_without_actions_overlap_cannot_be_assessed(self):
        occ = feasibility.occupancy(make_games(20))
        report = feasibility.positivity_report(occ)
        self.assertFalse(report["canAssessOverlap"],
                         "没有动作定义却声称评估了重叠性")
        self.assertIn("重叠性", report["verdict"])

    def test_state_coverage_is_still_reported(self):
        occ = feasibility.occupancy(make_games(20))
        self.assertGreater(len(occ["states"]), 0)
        self.assertGreater(occ["usableFrames"], 0)

    def test_with_actions_empty_cells_are_flagged(self):
        """有动作定义时，空格子必须被点出来 —— 那些是答不了的问题。"""
        def always_retreat(frame):
            return "撤退"          # 数据里只有一种动作 → 其他动作全是空格

        occ = feasibility.occupancy(make_games(20), always_retreat)
        report = feasibility.positivity_report(occ)
        self.assertTrue(report["canAssessOverlap"])
        # 只有一种动作时不该报空格（只有一列），但状态格子应该都在
        self.assertEqual(report["actions"], ["撤退"])

    def test_empty_cells_are_called_unanswerable_not_just_thin(self):
        verdict = feasibility._overlap_verdict(["A × 开龙"], [], 4)
        self.assertIn("再多数据也答不了", verdict,
                      "空格子和样本少是两回事，必须说清楚前者加数据没用")

    def test_thin_cells_are_called_fixable(self):
        verdict = feasibility._overlap_verdict([], ["A × 开龙（5 帧）"], 4)
        self.assertIn("加数据能解决", verdict)

    def test_low_confidence_frames_do_not_count_toward_coverage(self):
        """低置信度的帧算不出特征，不该被算进覆盖率里充数。"""
        rows = [{"atSec": 60, "fields": {
            "blueGold": schema.field(3000, schema.SOURCE_OCR, 0.4),
            "redGold": schema.field(2900, schema.SOURCE_OCR, 0.4)}}]
        built = state.build({"gameId": "L", "blueWin": True}, rows, rules.load())
        occ = feasibility.occupancy([built])
        self.assertEqual(occ["usableFrames"], 0)
        self.assertEqual(occ["skippedFrames"], 1)


class SeasonStabilityDistinguishesDriftFromBreak(unittest.TestCase):
    """降权能处理漂移，处理不了结构断裂。"""

    def test_needs_at_least_two_trainable_seasons(self):
        result = feasibility.season_stability({"S1": make_games(2)})
        self.assertFalse(result["comparable"])

    def test_refused_seasons_are_reported_not_silently_dropped(self):
        result = feasibility.season_stability({"S1": make_games(2), "S2": make_games(2)})
        self.assertTrue(result["refused"], "训不出来的赛季被悄悄丢掉了")

    def test_unstable_features_are_called_structural_breaks(self):
        verdict = feasibility._stability_verdict(
            ["goldDiffK"], {"goldDiffK": {"relative": 0.9}})
        self.assertIn("结构断裂", verdict)
        self.assertIn("降权解决不了", verdict,
                      "必须说清楚降权救不了结构断裂，否则会有人以为加个权重就行")

    def test_stable_features_allow_pooling(self):
        verdict = feasibility._stability_verdict([], {})
        self.assertIn("可以合并", verdict)


class TimeDecayIsMonotone(unittest.TestCase):
    def test_older_seasons_weigh_less(self):
        weights = [feasibility.time_decay_weight(i) for i in range(6)]
        self.assertEqual(weights, sorted(weights, reverse=True))
        self.assertEqual(weights[0], 1.0)

    def test_half_life_behaves(self):
        self.assertAlmostEqual(feasibility.time_decay_weight(2, half_life=2.0), 0.5)


class StratumBinning(unittest.TestCase):
    def test_frames_without_features_have_no_stratum(self):
        self.assertIsNone(feasibility.stratum_of({"minute": None, "goldDiffK": 1.0}))

    def test_extreme_values_still_land_in_a_bin(self):
        self.assertIsNotNone(feasibility.stratum_of({"minute": 40.0, "goldDiffK": 60.0}))
        self.assertIsNotNone(feasibility.stratum_of({"minute": 0.0, "goldDiffK": -60.0}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
