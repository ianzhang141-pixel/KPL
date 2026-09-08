"""胜率模型的诚实性测试。

这个文件盯的不是「算得准不准」—— 现在没有真实数据，谈不上准。
盯的是**这套东西会不会假装自己很准**。

具体防四件事：

1. 基线曲线被标成模型输出（那就等于骗人）
2. 数据不够时偷偷训练出一个坏模型
3. 只有赢的比赛（或只有输的）也照训不误 —— 那是选择偏差
4. 分支胜率（Q(s,a)）被编出来 —— 那需要因果推断，现在做不了

第 4 条最危险：编一个「如果当时不开团，胜率是 61%」出来，
看起来正是用户想要的东西，而且没有任何一步会报错。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import model, rules, schema, state  # noqa: E402


def manual(value):
    return schema.field(value, schema.SOURCE_MANUAL)


def observation(at_sec, blue_gold, red_gold, blue_kills=0, red_kills=0, events=None):
    return {
        "atSec": at_sec,
        "fields": {
            "blueGold": manual(blue_gold), "redGold": manual(red_gold),
            "blueKills": manual(blue_kills), "redKills": manual(red_kills),
        },
        "events": events or [],
    }


def build_game(blue_win=True, rows=None):
    rows = rows or [
        (60, 3000, 2900), (120, 7000, 6500), (180, 11500, 10200),
        (240, 16000, 14000), (300, 20000, 18000),
    ]
    return state.build({"gameId": "T", "blueWin": blue_win},
                       [observation(*r) for r in rows], rules.load())


class BaselineIsLabelledAsAGuess(unittest.TestCase):
    """基线曲线绝不能被当成模型输出。"""

    def test_curve_without_model_is_marked_baseline(self):
        curve = model.win_curve(build_game())
        self.assertEqual(curve["kind"], "baseline")
        self.assertFalse(curve["trained"], "基线曲线被标成了 trained")

    def test_baseline_reports_no_accuracy(self):
        curve = model.win_curve(build_game())
        self.assertIsNone(curve["accuracy"],
                          "基线没有准确率可言，给出一个数就是编的")

    def test_baseline_carries_a_visible_warning(self):
        curve = model.win_curve(build_game())
        self.assertIn("假设", curve["note"])
        self.assertTrue(len(curve["note"]) > 40, "警告太短，前端显示不出分量")

    def test_priors_each_state_their_reasoning(self):
        """每个系数都要写清楚凭什么这么定，否则没人能核对它。"""
        for key, entry in model.BASELINE_PRIORS.items():
            self.assertIn("weight", entry)
            self.assertTrue(entry.get("why"), f"{key} 没写依据")


class TrainingRefusesBadData(unittest.TestCase):
    """数据不够就拒绝，不产出一个看起来能用的坏模型。"""

    def test_refuses_single_game(self):
        with self.assertRaises(model.TrainingRefused):
            model.train([build_game()])

    def test_refuses_when_all_games_have_the_same_outcome(self):
        """只有赢的比赛 —— 这是选择偏差，再多数据也救不了。"""
        games = [build_game(blue_win=True) for _ in range(60)]
        with self.assertRaises(model.TrainingRefused) as caught:
            model.train(games)
        self.assertIn("选择偏差", str(caught.exception))

    def test_refusal_explains_why_rather_than_just_failing(self):
        with self.assertRaises(model.TrainingRefused) as caught:
            model.train([build_game()])
        message = str(caught.exception)
        self.assertIn("场比赛", message)
        self.assertIn("不是 bug", message, "拒绝要说明这是设计，否则会被当成故障绕过去")

    def test_games_without_outcome_label_are_skipped(self):
        """没记胜负的比赛训不了 —— 标签是唯一的监督信号。"""
        no_label = state.build({"gameId": "N"},
                               [observation(60, 3000, 2900)], rules.load())
        with self.assertRaises(model.TrainingRefused):
            model.train([no_label] * 60)


class BranchAnalysisIsNotFabricated(unittest.TestCase):
    """Q(s,a) 没实现，就必须说没实现。"""

    def test_branch_status_is_explicitly_unimplemented(self):
        status = model.branch_analysis_status()
        self.assertFalse(status["implemented"])
        self.assertTrue(status["why"], "未实现要说明为什么，否则下一个人会随手糊一个")

    def test_explanation_never_invents_branch_probabilities(self):
        curve = model.win_curve(build_game(rows=[
            (60, 3000, 2900), (120, 12000, 6500), (180, 13000, 15000),
        ]))
        for inflection in model.find_inflections(curve):
            explained = model.explain_inflection(inflection)
            self.assertIsNone(explained["branches"],
                              "编出了分支胜率 —— 那需要因果推断，现在做不了")
            self.assertFalse(explained["limitation"]["implemented"])

    def test_why_mentions_the_causal_trap(self):
        """必须点明「拟合 ≠ 因果」这个坑，否则下一个人还会踩。"""
        why = model.branch_analysis_status()["why"]
        self.assertTrue("因果" in why and "相关" in why)


class FeaturesRespectTheDataContract(unittest.TestCase):
    """特征抽取不许碰标签、不许用低置信度的值、不许把缺失当 0。"""

    def test_low_confidence_fields_are_excluded(self):
        low = [{"atSec": 60, "fields": {
            "blueGold": schema.field(3000, schema.SOURCE_OCR, 0.4),
            "redGold": schema.field(2900, schema.SOURCE_OCR, 0.4)}}]
        built = state.build({"gameId": "L"}, low, rules.load())
        self.assertIsNone(model.extract_features(built["frames"][0]),
                          "低置信度的值被当成特征了 —— 模型会学到识别误差的规律")

    def test_missing_gold_yields_no_features_rather_than_zero(self):
        built = state.build({"gameId": "M"},
                            [{"atSec": 60, "fields": {"blueGold": manual(3000)}}],
                            rules.load())
        self.assertIsNone(model.extract_features(built["frames"][0]),
                          "红方经济未知时仍算出了特征 —— 那等于把缺失当成 0")

    def test_label_is_never_a_feature(self):
        built = build_game(blue_win=True)
        features = model.extract_features(built["frames"][0])
        self.assertIsNotNone(features)
        for key in features:
            self.assertNotIn("win", key.lower(),
                             "特征里出现了胜负标签 —— 这是最严重的信息泄漏")

    def test_frames_without_features_are_skipped_not_guessed(self):
        """从头到尾没观测到的字段 → 这一帧算不出特征 → 跳过并计数。"""
        rows = [{"atSec": 60, "fields": {"blueGold": manual(3000)}},   # 红方经济从未出现
                {"atSec": 120, "fields": {"blueGold": manual(7000)}}]
        built = state.build({"gameId": "S"}, rows, rules.load())
        curve = model.win_curve(built)
        self.assertEqual(curve["points"], [])
        self.assertEqual(curve["skippedFrames"], 2,
                         "算不出特征的帧应该被跳过并计数，而不是猜一个胜率")

    def test_a_fresh_carry_is_allowed_but_a_stale_one_is_not(self):
        """沿用值可以进特征，但只在它还够可信的时候。

        这是 state.py 那套置信度衰减的直接后果，值得单独钉住：
        上一帧刚读到的经济，这一帧没读到时沿用是合理的（0.98×0.85=0.83，仍可信）；
        但连续沿用五六帧之后置信度会跌破 0.75，那时这一帧就该被跳过。

        **写这条测试是因为我一开始把期望写反了** ——
        以为「这一帧没读到就必须跳过」，实际上沿用一帧是设计好的行为。
        """
        rows = [{"atSec": 60, "fields": {"blueGold": manual(3000), "redGold": manual(2900)}}]
        rows += [{"atSec": 60 + 30 * i, "fields": {}} for i in range(1, 8)]
        curve = model.win_curve(state.build({"gameId": "C"}, rows, rules.load()))
        self.assertGreater(len(curve["points"]), 1, "紧邻的沿用值应该还能用")
        self.assertGreater(curve["skippedFrames"], 0,
                           "沿用太久的值应该跌破可信线并被跳过")


class ObjectivesStaySeparateInFeatures(unittest.TestCase):
    """中立资源在特征里也不许合并 —— 和 state.py 是同一条规矩。"""

    def test_five_objectives_have_five_weights(self):
        for key in ("tyrants", "darkTyrants", "overlords",
                    "stormDragons", "prophetOverlords"):
            self.assertIn(key, model.OBJECTIVE_WEIGHTS)

    def test_storm_dragon_outweighs_tyrant(self):
        """风暴龙王和暴君价值差一个量级，权重必须不同。

        权重相同就等于又把它们合并了 —— 换个地方犯 LOL 那个 bug。
        """
        self.assertGreater(model.OBJECTIVE_WEIGHTS["stormDragons"],
                           model.OBJECTIVE_WEIGHTS["tyrants"])
        self.assertGreater(model.OBJECTIVE_WEIGHTS["darkTyrants"],
                           model.OBJECTIVE_WEIGHTS["tyrants"])


class InflectionsAnswerWhereNotWho(unittest.TestCase):
    def test_finds_the_big_swing(self):
        curve = model.win_curve(build_game(rows=[
            (60, 5000, 5000), (120, 6000, 5500), (180, 6500, 16000),
        ]))
        inflections = model.find_inflections(curve)
        self.assertTrue(inflections)
        self.assertEqual(inflections[0]["clock"], "03:00")
        self.assertLess(inflections[0]["deltaPP"], 0)

    def test_sorted_by_magnitude(self):
        curve = model.win_curve(build_game(rows=[
            (60, 5000, 5000), (120, 9000, 5000), (180, 9500, 5200), (240, 6000, 20000),
        ]))
        deltas = [abs(f["deltaPP"]) for f in model.find_inflections(curve)]
        self.assertEqual(deltas, sorted(deltas, reverse=True))

    def test_small_drift_is_not_an_inflection(self):
        curve = model.win_curve(build_game(rows=[
            (60, 5000, 5000), (120, 5050, 5000), (180, 5100, 5050),
        ]))
        self.assertEqual(model.find_inflections(curve), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
