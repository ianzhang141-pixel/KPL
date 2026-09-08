"""分数模型的测试 —— 重点是「它能不能被证伪」。

分数模型最危险的地方是它**自洽**：权重说开龙值 5 分，模型就说开龙好，
每一步推理都对得上，看不出哪里错了。错误以系统性偏差的形式存在，
不体现在任何置信区间上。

所以这里盯的是：

1. 校准必须能打脸 —— 权重定反了，校准要报不单调
2. 缺项和劣势必须区分 —— 两者在分数上长得一模一样
3. 拿不到的分项必须标出来，不能悄悄当 0
4. 拟合排除了哪些分项必须明说（遗漏变量偏差）
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import model, rules, schema, scoring, state  # noqa: E402


def manual(value):
    return schema.field(value, schema.SOURCE_MANUAL)


def synthetic_games(count=120, seed=7):
    random.seed(seed)
    games = []
    for index in range(count):
        edge = random.uniform(-1, 1)
        rows = []
        for minute in range(1, 13):
            rows.append({"atSec": 60 * minute, "fields": {
                "blueGold": manual(int(3000 + minute * 1800 + edge * minute * 400)),
                "redGold": manual(int(3000 + minute * 1800 - edge * minute * 400)),
                "blueKills": manual(max(0, int(minute * 0.5 + edge * 2))),
                "redKills": manual(max(0, int(minute * 0.5 - edge * 2)))}, "events": []})
        win = random.random() < 1 / (1 + 2.718 ** (-edge * 3))
        games.append(state.build({"gameId": f"G{index}", "blueWin": win},
                                 rows, rules.load()))
    return games


class CalibrationCanFalsifyTheWeights(unittest.TestCase):
    """这是整个模块存在的理由：让一组假设变得可以被数据打脸。"""

    @classmethod
    def setUpClass(cls):
        cls.games = synthetic_games()

    def test_sensible_weights_are_monotone(self):
        result = scoring.calibrate(self.games)
        self.assertTrue(result["rows"], "校准一行都没产出 —— coverage 分母多半用错了")
        self.assertTrue(result["monotone"])

    def test_reversed_weight_is_caught(self):
        """把经济权重定成负的，校准必须报不单调。

        如果这条过不了，说明校准根本没有鉴别力，
        那这个分数模型就是不可证伪的 —— 和占卜没区别。
        """
        weights = {k: v["weight"] for k, v in scoring.COMPONENTS.items()}
        weights["economy"] = -1.0
        result = scoring.calibrate(self.games, weights)
        self.assertFalse(result["monotone"], "权重定反了却没被校准抓到")
        self.assertIn("不单调", result["verdict"])

    def test_verdict_warns_that_passing_is_not_proof(self):
        verdict = scoring.calibrate(self.games)["verdict"]
        self.assertIn("不说明每一项的权重都对", verdict,
                      "校准通过不等于权重正确，必须说清楚")

    def test_too_few_samples_refuses_to_conclude(self):
        result = scoring.calibrate(self.games[:2])
        self.assertIn("样本太少", result["verdict"])


class MissingIsNotWeak(unittest.TestCase):
    """缺项和劣势在分数上长得一模一样 —— 必须靠 coverage 区分。"""

    def test_coverage_is_measured_against_obtainable_not_total(self):
        """分母用错会让分数永远不可用。

        血量、技能CD 这些**永远拿不到**，占了约一半权重。
        如果 coverage 除以全部权重，它永远到不了阈值，
        `comparable` 永远是 False，这个模型一帧都算不出来。
        """
        built = synthetic_games(1)[0]
        scored = scoring.score_frame(built["frames"][5])
        self.assertGreater(scored["coverage"], 0.6,
                           "coverage 分母把永远拿不到的分项也算进去了")
        self.assertTrue(scored["comparable"])

    def test_ceiling_reports_the_permanent_gap(self):
        scored = scoring.score_frame(synthetic_games(1)[0]["frames"][5])
        self.assertLess(scored["ceiling"], 1.0,
                        "有分项永远拿不到，ceiling 不该是 1")
        self.assertIn("战斗力", scored["ceilingNote"])

    def test_low_coverage_frame_is_marked_incomparable(self):
        rows = [{"atSec": 60, "fields": {"blueGold": manual(3000)}}]   # 红方缺失
        built = state.build({"gameId": "L", "blueWin": True}, rows, rules.load())
        scored = scoring.score_frame(built["frames"][0])
        self.assertFalse(scored["comparable"])
        self.assertIn("不能和其他帧比较", scored["note"])

    def test_incomparable_frames_are_excluded_from_calibration(self):
        rows = [{"atSec": 60, "fields": {"blueGold": manual(3000)}}]
        built = state.build({"gameId": "L", "blueWin": True}, rows, rules.load())
        result = scoring.calibrate([built])
        self.assertGreater(result["skippedFrames"], 0)
        self.assertEqual(result["rows"], [])


class UnobtainableComponentsAreDeclared(unittest.TestCase):
    """拿不到的东西必须标出来，不能悄悄当 0 参与计算。"""

    def test_health_and_cooldowns_are_marked_unobtainable(self):
        for key in ("health", "cooldowns"):
            self.assertFalse(scoring.COMPONENTS[key]["obtainable"],
                             f"{key} 在转播画面上根本没有稳定显示")
            self.assertTrue(scoring.COMPONENTS[key].get("blocker"))

    def test_every_component_states_its_data_requirement(self):
        for key, spec in scoring.COMPONENTS.items():
            self.assertTrue(spec.get("needs"), f"{key} 没写需要什么数据")
            self.assertTrue(spec.get("why"), f"{key} 没写凭什么给这个权重")

    def test_unobtainable_components_never_contribute(self):
        scored = scoring.score_frame(synthetic_games(1)[0]["frames"][5])
        for item in scored["breakdown"]:
            if item["key"] in scoring.blocked_components():
                self.assertFalse(item["available"],
                                 "拿不到的分项竟然参与了计算")

    def test_coverage_note_mentions_the_users_example(self):
        """用户举的例子（技能CD）正好落在拿不到的那一半里，要说清楚。"""
        self.assertIn("技能CD", scoring.coverage_note())


class DisagreementsSurfaceLearning(unittest.TestCase):
    def test_finds_frames_where_score_and_outcome_conflict(self):
        found = scoring.disagreements(synthetic_games(), threshold=8.0)
        self.assertTrue(found, "一个分歧样本都没有 —— 多半是阈值或覆盖率的问题")
        for item in found:
            self.assertNotEqual(item["scoreFavoured"], item["actualWinner"])

    def test_sorted_by_magnitude(self):
        found = scoring.disagreements(synthetic_games(), threshold=5.0)
        sizes = [abs(f["scoreDiff"]) for f in found]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


class FittingIsAnchoredAndHonest(unittest.TestCase):
    def test_refuses_when_data_is_thin(self):
        with self.assertRaises(model.TrainingRefused):
            scoring.fit_weights(synthetic_games(2))

    def test_excluded_components_are_reported(self):
        """排除了哪些分项必须明说 —— 它们的效应会被留下的项吸收。"""
        fit = scoring.fit_weights(synthetic_games())
        self.assertTrue(fit["excludedComponents"])
        self.assertIn("吸收", fit["exclusionWarning"])

    def test_stronger_prior_keeps_weights_closer_to_it(self):
        loose = scoring.fit_weights(synthetic_games(), prior_strength=0.5)
        tight = scoring.fit_weights(synthetic_games(), prior_strength=500.0)
        key = "economy"
        self.assertLessEqual(abs(tight["shift"][key]), abs(loose["shift"][key]) + 1e-6,
                             "先验强度调大之后权重反而离先验更远了")

    def test_note_warns_about_collinearity(self):
        fit = scoring.fit_weights(synthetic_games())
        self.assertIn("共线", fit["note"],
                      "经济和等级高度相关，单看一项的位移会误导，必须提醒")


class ParadigmComparisonIsExplicit(unittest.TestCase):
    """这两种模型的关系必须写在代码里，不能只存在于对话里。"""

    def test_scoring_weakness_is_stated_as_echoing_assumptions(self):
        compare = scoring.compare_paradigms()
        text = " ".join(compare["scoring"]["weaknesses"])
        self.assertIn("回声", text, "必须点明分数模型只是把假设还给你")

    def test_causal_weakness_is_overlap(self):
        compare = scoring.compare_paradigms()
        self.assertIn("重叠性", " ".join(compare["causal"]["weaknesses"]))

    def test_verdict_says_they_are_complements_not_rivals(self):
        verdict = scoring.compare_paradigms()["verdict"]
        self.assertIn("不是二选一", verdict)
        self.assertIn("状态转移模型", verdict,
                      "用户的例子里隐含了转移模型，那是第三样东西，要点出来")


if __name__ == "__main__":
    unittest.main(verbosity=2)
