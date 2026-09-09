import tempfile
import unittest
from pathlib import Path

from kplab import annotate, evaluate, ocr, rules, schema, store


class RelativeTeamFieldTests(unittest.TestCase):
    def test_score_ocr_only_trusts_two_preprocessing_paths_that_agree(self):
        self.assertEqual(ocr._choose_score_read("13", "13"), ("13", 0.85))
        self.assertEqual(ocr._choose_score_read("43", "13"), ("13", 0.45))
        self.assertEqual(ocr._choose_score_read("", "7"), ("7", 0.45))
        self.assertEqual(ocr._choose_score_read("", ""), ("", 0.0))

    def test_score_sequence_downgrades_decrease_and_impossible_jump(self):
        good = schema.field(13, schema.SOURCE_OCR, 0.85)
        self.assertIsNone(ocr.guard_score_sequence(good, 13, 20)[1])
        decreased, warning = ocr.guard_score_sequence(
            schema.field(3, schema.SOURCE_OCR, 0.85), 13, 20
        )
        self.assertIn("倒退", warning)
        self.assertLess(schema.conf(decreased), schema.TRUST_THRESHOLD)
        jumped, warning = ocr.guard_score_sequence(
            schema.field(43, schema.SOURCE_OCR, 0.85), 13, 20
        )
        self.assertIn("跳变", warning)
        self.assertLess(schema.conf(jumped), schema.TRUST_THRESHOLD)

    def test_manual_relative_kills_are_kept_separate_from_blue_red(self):
        observation = annotate.build_observation(
            120, {"allyKills": 3, "enemyKills": 5}, rules.load()
        )
        self.assertEqual(schema.get(observation["fields"]["allyKills"]), 3)
        self.assertEqual(schema.get(observation["fields"]["enemyKills"]), 5)
        self.assertNotIn("blueKills", observation["fields"])
        self.assertNotIn("redKills", observation["fields"])

    def test_evaluation_scores_relative_kills_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_id = "RELATIVE_001"
            store.create_game(root, game_id, {})
            truth = annotate.build_observation(
                120, {"allyKills": 3, "enemyKills": 5}, rules.load()
            )
            annotate.save(root, game_id, truth)
            store.append_jsonl_gz(store.observations_path(root, game_id), {
                "atSec": 120,
                "fields": {
                    "clock": schema.field(120, schema.SOURCE_OCR, 0.9),
                    "allyKills": schema.field(3, schema.SOURCE_OCR, 0.9),
                    "enemyKills": schema.field(4, schema.SOURCE_OCR, 0.9),
                },
            })
            report = evaluate.run(root, game_id)
            self.assertEqual(report["compared"], 2)
            self.assertEqual(report["exact"], 1)
            self.assertEqual(report["byField"]["allyKills"]["exactRate"], 1.0)
            self.assertEqual(report["byField"]["enemyKills"]["exactRate"], 0.0)

    def test_low_confidence_wrong_guess_is_not_counted_as_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_id = "LOWCONF_001"
            store.create_game(root, game_id, {})
            truth = annotate.build_observation(120, {"allyKills": 3}, rules.load())
            annotate.save(root, game_id, truth)
            store.append_jsonl_gz(store.observations_path(root, game_id), {
                "atSec": 120,
                "fields": {"allyKills": schema.field(8, schema.SOURCE_OCR, 0.45)},
            })
            report = evaluate.run(root, game_id)
            self.assertEqual(report["compared"], 1)
            self.assertEqual(report["exact"], 0)
            self.assertEqual(report["trustedCompared"], 0)
            self.assertIsNone(report["trustedExactRate"])

    def test_attempted_frame_without_observation_counts_as_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_id = "NO_CLOCK_001"
            frame = "frame_001.jpg"
            store.create_game(root, game_id, {
                "ocrLastRun": {"requested": 1, "saved": 0, "unknownClock": 1,
                               "attemptedFiles": [frame]},
            })
            annotate.save(root, game_id, annotate.build_observation(
                120, {"allyKills": 3, "enemyKills": 5}, rules.load(), frame
            ))
            # 另加一条无关机器记录，让评估进入机器对照分支。
            store.append_jsonl_gz(store.observations_path(root, game_id), {
                "atSec": 999, "frameFile": "other.jpg",
                "fields": {"clock": schema.field(999, schema.SOURCE_OCR, 0.9)},
            })
            report = evaluate.run(root, game_id)
            self.assertEqual(report["missed"], 3)  # 两项比分 + 计时
            self.assertEqual(report["byField"]["clock"]["missed"], 1)

    def test_wrong_clock_is_compared_by_frame_instead_of_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_id = "CLOCK_001"
            store.create_game(root, game_id, {})
            truth = annotate.build_observation(
                120, {"allyKills": 3}, rules.load(), "frame_001.jpg"
            )
            annotate.save(root, game_id, truth)
            store.append_jsonl_gz(store.observations_path(root, game_id), {
                "atSec": 128,
                "frameFile": "frame_001.jpg",
                "fields": {
                    "clock": schema.field(128, schema.SOURCE_OCR, 0.9),
                    "allyKills": schema.field(3, schema.SOURCE_OCR, 0.9),
                },
            })
            report = evaluate.run(root, game_id)
            self.assertEqual(report["byField"]["clock"]["compared"], 1)
            self.assertEqual(report["byField"]["clock"]["exactRate"], 0.0)
            self.assertTrue(any(item["field"] == "clock" for item in report["examples"]))


if __name__ == "__main__":
    unittest.main()
