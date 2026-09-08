"""锁住三条不许违反的性质。

这个文件存在的理由：本项目最危险的错误**都不会报错**。
把「没读到」写成 0、用后面的帧修前面的帧、把黑暗暴君算进暴君 ——
这三种改动都能让代码正常跑、让报表更好看，只是把数据悄悄弄坏。

所以它们必须被测试钉住。任何一条测试挂了，
都不要「顺手改一下测试让它通过」—— 那说明改动本身是错的。

跑法（不需要装任何东西）：
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import annotate, evaluate, ocr, rules, schema, state, store  # noqa: E402


def ocr_field(value, confidence=0.9):
    return schema.field(value, schema.SOURCE_OCR, confidence)


class NoFutureLeakage(unittest.TestCase):
    """性质一：t 时刻的 State 只能含 t 时刻及之前的信息。"""

    def test_bad_later_read_does_not_rewrite_earlier_frame(self):
        observations = [
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(2000)}},   # 倒退，必是错的
            {"atSec": 180, "fields": {"blueGold": ocr_field(4200)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        first = built["frames"][0]["teams"]["100"]["totalGold"]
        self.assertEqual(schema.get(first), 3000,
                         "第 2 帧的坏读数回头改了第 1 帧 —— 这是未来信息泄漏")

    def test_contradicting_read_is_rejected_not_accepted(self):
        observations = [
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(2000)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        second = built["frames"][1]["teams"]["100"]["totalGold"]
        self.assertEqual(schema.src(second), schema.SOURCE_CARRIED,
                         "与已知的过去矛盾的读数应该被丢弃并沿用旧值")
        self.assertTrue(built["rejections"], "拒绝了读数就该有记录，否则没人知道发生过")

    def test_observations_are_processed_in_time_order(self):
        # 乱序输入也必须按时间处理，否则「上一帧」就不是时间上更早的那一帧
        observations = [
            {"atSec": 180, "fields": {"blueGold": ocr_field(4200)}},
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(3800)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        times = [f["atSec"] for f in built["frames"]]
        self.assertEqual(times, sorted(times))
        golds = [schema.get(f["teams"]["100"]["totalGold"]) for f in built["frames"]]
        self.assertEqual(golds, [3000, 3800, 4200])

    def test_carried_confidence_decays(self):
        """沿用越久越不可信，最终必须跌破可信线，不能一直冒充事实。"""
        observations = [{"atSec": 60, "fields": {"blueGold": ocr_field(3000, 0.9)}}]
        observations += [{"atSec": 60 + 30 * i, "fields": {}} for i in range(1, 6)]
        built = state.build({"gameId": "T"}, observations, rules.load())
        confidences = [schema.conf(f["teams"]["100"]["totalGold"]) for f in built["frames"]]
        self.assertTrue(all(b <= a for a, b in zip(confidences, confidences[1:])),
                        "沿用值的置信度必须单调不增")
        self.assertLess(confidences[-1], schema.TRUST_THRESHOLD,
                        "沿用太久的值还被当作可信 —— 这会让陈旧数据冒充事实")


class ObjectiveCountersStaySeparate(unittest.TestCase):
    """性质二：五种中立资源绝不合并（LOL ③ 号 bug 的教训）。"""

    def test_five_objectives_have_five_counters(self):
        observations = [{
            "atSec": 600,
            "fields": {},
            "events": [
                {"type": "TYRANT_KILL", "teamId": 100},
                {"type": "DARK_TYRANT_KILL", "teamId": 100},
                {"type": "OVERLORD_KILL", "teamId": 100},
                {"type": "STORM_DRAGON_KILL", "teamId": 100},
                {"type": "PROPHET_OVERLORD_KILL", "teamId": 100},
            ],
        }]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["100"]
        for key in ("tyrants", "darkTyrants", "overlords", "stormDragons", "prophetOverlords"):
            self.assertEqual(schema.get(team[key]), 1,
                             f"{key} 应该正好是 1 —— 不是 0（漏了）也不是 2（和别的合并了）")

    def test_dark_tyrant_is_not_a_tyrant(self):
        observations = [{"atSec": 600, "fields": {},
                         "events": [{"type": "DARK_TYRANT_KILL", "teamId": 200}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["200"]
        self.assertEqual(schema.get(team["darkTyrants"]), 1)
        self.assertEqual(schema.get(team["tyrants"]), 0,
                         "黑暗暴君被算进了暴君 —— 正是 LOL 那个 bug 的翻版")

    def test_storm_dragon_is_not_an_overlord(self):
        observations = [{"atSec": 900, "fields": {},
                         "events": [{"type": "STORM_DRAGON_KILL", "teamId": 100}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["100"]
        self.assertEqual(schema.get(team["stormDragons"]), 1)
        self.assertEqual(schema.get(team["overlords"]), 0,
                         "风暴龙王被算进了主宰 —— 两者价值差一个数量级")

    def test_kill_credits_the_right_side(self):
        observations = [{"atSec": 100, "fields": {},
                         "events": [{"type": "HERO_KILL", "teamId": 100}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        frame = built["frames"][0]
        self.assertEqual(schema.get(frame["teams"]["100"]["kills"]), 1)
        self.assertEqual(schema.get(frame["teams"]["200"]["deaths"]), 1)


class MissingIsNotZero(unittest.TestCase):
    """性质三：读不出来是「不知道」，绝不是 0。"""

    def test_unknown_field_stays_none(self):
        built = state.build({"gameId": "T"},
                            [{"atSec": 60, "fields": {"blueGold": ocr_field(3000)}}],
                            rules.load())
        red = built["frames"][0]["teams"]["200"]["totalGold"]
        self.assertIsNone(schema.get(red), "没读到的经济被写成了 0")

    def test_derived_value_is_unknown_when_any_input_is_unknown(self):
        built = state.build({"gameId": "T"},
                            [{"atSec": 60, "fields": {"blueGold": ocr_field(3000)}}],
                            rules.load())
        diff = built["frames"][0]["diff"]["totalGold"]
        self.assertIsNone(schema.get(diff),
                          "红方经济未知时，经济差必须是未知 —— "
                          "否则会凭空造出 +3000 的假优势")

    def test_derive_itself_returns_unknown_when_any_input_is_unknown(self):
        """直接盯住 schema.derive 的契约，不要只靠调用方的前置判断。

        state.py 在调用 derive 之前自己也检查了一遍，所以即使 derive 的守卫被删掉，
        经济差那条测试照样能过 —— 这正是一次变异测试暴露出来的盲点。
        契约要在契约所在的那一层测，否则下一个调用方就会掉进去。
        """
        known = ocr_field(100)
        self.assertIsNone(schema.get(schema.derive(20, known, schema.unknown())),
                          "只要有一个输入未知，派生值就必须是未知")
        self.assertIsNone(schema.get(schema.derive(20, schema.unknown())))
        self.assertEqual(schema.src(schema.derive(20, known, schema.unknown())),
                         schema.SOURCE_UNKNOWN)

    def test_derived_confidence_is_the_weakest_input(self):
        low, high = ocr_field(100, 0.4), schema.field(80, schema.SOURCE_MANUAL, 0.98)
        self.assertAlmostEqual(schema.conf(schema.derive(20, low, high)), 0.4,
                               msg="做一次减法不该让结果变得更确定")

    def test_field_with_none_value_cannot_claim_a_source(self):
        entry = schema.field(None, schema.SOURCE_MANUAL, 1.0)
        self.assertEqual(schema.src(entry), schema.SOURCE_UNKNOWN)
        self.assertEqual(schema.conf(entry), 0.0)

    def test_ocr_returns_none_not_zero_on_failure(self):
        for text in ("", "abc", "———", "99999999"):
            value, confidence = ocr.parse_int(text, lo=0, hi=200000)
            self.assertIsNone(value, f"{text!r} 解析失败时应返回 None 而不是 0")
            self.assertEqual(confidence, 0.0)


class RuleValidation(unittest.TestCase):
    """规则校验：不可能的值必须挡在写入之前，而不是留到报告里。"""

    def test_out_of_range_level_is_rejected_at_annotation(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"players": {"1": {"level": 99}}})

    def test_pixel_coordinates_are_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"players": {"1": {"x": 420}}})

    def test_unknown_event_type_is_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"events": [{"type": "BARON", "teamId": 100}]})

    def test_empty_annotation_is_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {})

    def test_clock_needs_two_digit_seconds(self):
        self.assertEqual(ocr.parse_clock("12:34")[0], 754)
        self.assertEqual(ocr.parse_clock("1 2 : 3 4")[0], 754)   # OCR 塞进来的空格
        self.assertIsNone(ocr.parse_clock("2.3万")[0],
                          "把经济读成了时间 —— 会让整场时间轴错位")

    def test_game_id_blocks_path_traversal(self):
        self.assertFalse(store.valid_game_id("../evil"))
        self.assertFalse(store.valid_game_id("a/b"))
        self.assertTrue(store.valid_game_id("KPL2026_AG_vs_EST_G1"))


class StorageRoundTrip(unittest.TestCase):
    """存储：压缩 + 只追加，读回来要和写进去的一样。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_append_only_jsonl_reads_back_in_order(self):
        path = store.observations_path(self.dir, "G1")
        for i in range(5):
            store.append_jsonl_gz(path, {"i": i})
        self.assertEqual([r["i"] for r in store.read_jsonl_gz(path)], list(range(5)))

    def test_files_are_actually_compressed(self):
        import gzip
        path = store.state_path(self.dir, "G1")
        store.write_json_gz(path, {"x": "重复内容" * 500})
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            self.assertIn("重复内容", handle.read())
        self.assertLess(path.stat().st_size, 1000, "看起来没有真的压缩")

    def test_annotation_is_written_to_both_streams(self):
        store.create_game(self.dir, "G1", {})
        observation = annotate.build_observation(60, {"blueGold": 3000})
        annotate.save(self.dir, "G1", observation)
        # annotations 是纯人工的标尺，observations 是喂给 state 的输入，两边都要有
        self.assertEqual(len(annotate.load_annotations(self.dir, "G1")), 1)
        self.assertEqual(
            len(list(store.read_jsonl_gz(store.observations_path(self.dir, "G1")))), 1)


class EvaluationHonesty(unittest.TestCase):
    """评估：没有标尺时必须说「量不了」，不能给一个准确率。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        store.create_game(self.dir, "G1", {})

    def test_no_annotations_means_no_accuracy_number(self):
        store.append_jsonl_gz(store.observations_path(self.dir, "G1"),
                              {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}})
        report = evaluate.run(self.dir, "G1")
        self.assertFalse(report["verdict"]["ok"])
        self.assertIsNone(report.get("exactRate"),
                          "没有人工标注却给出了准确率 —— 那个数字是编的")

    def test_detects_a_wrong_ocr_read(self):
        annotate.save(self.dir, "G1", annotate.build_observation(60, {"blueGold": 11000}))
        store.append_jsonl_gz(store.observations_path(self.dir, "G1"),
                              {"atSec": 60, "fields": {"blueGold": ocr_field(1100)}})
        report = evaluate.run(self.dir, "G1")
        self.assertEqual(report["exactRate"], 0.0)
        self.assertFalse(report["verdict"]["ok"])


class RulesAreMarkedUnverified(unittest.TestCase):
    """游戏常量默认必须是「待核对」，不能冒充查证过的事实。"""

    def test_defaults_start_unverified(self):
        table = rules.load()
        self.assertEqual(len(rules.unverified(table)), len(table),
                         "凭印象写的默认值不许标成已核对")

    def test_override_marks_verified(self):
        directory = Path(tempfile.mkdtemp())
        rules.save_override(directory, "towersPerLane", 2, "测试")
        table = rules.load(directory)
        self.assertTrue(table["towersPerLane"]["verified"])
        self.assertNotIn("towersPerLane", rules.unverified(table))


if __name__ == "__main__":
    unittest.main(verbosity=2)
