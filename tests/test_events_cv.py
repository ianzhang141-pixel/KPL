"""图标变化检测的性质测试。

不需要 ffmpeg —— 直接喂签名（一小块区域压缩成的几个数），绕开取像素那一步。

盯的是四件**不会报错但会把数据弄坏**的事：

1. 把防御塔和中立资源用同一套逻辑处理（塔单向、资源周期）
2. 图标抖动被当成多次击杀
3. 只有一个信号就断定英雄阵亡
4. 归属不明的资源击杀被硬塞给某一方
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import events_cv, rules  # noqa: E402
from kplab.rules import BLUE, RED   # noqa: E402

PRESENT = {"r": 60.0, "g": 200.0, "b": 200.0, "saturation": 0.7, "value": 200.0}
GONE = {"r": 18.0, "g": 26.0, "b": 40.0, "saturation": 0.55, "value": 40.0}
COLOURFUL = {"r": 210.0, "g": 70.0, "b": 60.0, "saturation": 0.71, "value": 210.0}
GRAY = {"r": 105.0, "g": 105.0, "b": 105.0, "saturation": 0.0, "value": 105.0}


class TowersAreOneWay(unittest.TestCase):
    """防御塔：消失即摧毁，不会重建。"""

    def test_disappearing_icon_reports_destruction(self):
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        self.assertIsNone(tracker.update(0, PRESENT))
        self.assertIsNone(tracker.update(300, GONE), "单帧变化必须先当成候选遮挡")
        event = tracker.update(330, GONE)
        self.assertIsNotNone(event)
        self.assertEqual(event["type"], "TOWER_DESTROYED")
        self.assertIn("candidateWindow", event)

    def test_credit_goes_to_the_other_team(self):
        """塔属于蓝方，拆掉它的功劳记给红方。"""
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        tracker.update(0, PRESENT)
        tracker.update(300, GONE)
        self.assertEqual(tracker.update(330, GONE)["teamId"], RED)

    def test_reappearing_icon_is_recorded_as_a_contradiction(self):
        """塔不会重建。图标回来了 = 我判错了，必须记下来。"""
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        tracker.update(0, PRESENT)
        tracker.update(300, GONE)
        tracker.update(330, GONE)
        tracker.update(360, PRESENT)
        self.assertTrue(tracker.contradictions,
                        "塔图标回来了却没有记为矛盾 —— 那等于默认它重建了")

    def test_destruction_is_reported_only_once(self):
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        tracker.update(0, PRESENT)
        tracker.update(300, GONE)
        self.assertIsNotNone(tracker.update(330, GONE))
        self.assertIsNone(tracker.update(360, GONE), "同一座塔被报了两次摧毁")

    def test_single_changed_frame_is_recorded_as_occlusion_not_destruction(self):
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        tracker.update(0, PRESENT)
        self.assertIsNone(tracker.update(30, GONE))
        self.assertIsNone(tracker.update(60, PRESENT))
        self.assertEqual(tracker.destroyed_at, None)
        self.assertEqual(len(tracker.occlusions), 1)

    def test_candidate_keeps_game_and_source_time_windows(self):
        tracker = events_cv.TowerTracker("t1", BLUE, "蓝方一塔")
        tracker.update(20, PRESENT, 452)
        tracker.update(25, GONE, 457)
        event = tracker.update(30, GONE, 462)
        self.assertEqual(event["candidateWindow"], {"fromSec": 20, "toSec": 30})
        self.assertEqual(event["sourceCandidateWindow"], {"fromSec": 452, "toSec": 462})


class ObjectivesAreCyclic(unittest.TestCase):
    """中立资源：被击杀后会刷新回来。用塔的逻辑处理就会数错。"""

    def test_respawn_is_normal_not_a_second_kill(self):
        tracker = events_cv.ObjectiveTracker("tyrantPit", "暴君坑")
        tracker.update(0, PRESENT)
        self.assertIsNotNone(tracker.update(180, GONE))
        tracker.update(420, PRESENT)          # 刷新
        self.assertEqual(len(tracker.kills), 1,
                         "刷新被当成了第二次击杀 —— 这正是塔和资源必须分开的原因")

    def test_a_real_second_kill_after_respawn_counts(self):
        tracker = events_cv.ObjectiveTracker("tyrantPit", "暴君坑")
        tracker.update(0, PRESENT)
        tracker.update(180, GONE)
        tracker.update(420, PRESENT)
        self.assertIsNotNone(tracker.update(480, GONE))
        self.assertEqual(len(tracker.kills), 2)

    def test_icon_jitter_is_rejected(self):
        """两次击杀间隔短于刷新时间 —— 那是图标闪烁，不是真的被拿了两次。"""
        tracker = events_cv.ObjectiveTracker("tyrantPit", "暴君坑")
        tracker.update(0, PRESENT)
        tracker.update(180, GONE)
        tracker.update(190, PRESENT)
        tracker.update(200, GONE)
        self.assertEqual(len(tracker.kills), 1)
        self.assertTrue(tracker.rejected, "抖动被丢弃了但没有记录下来")

    def test_tower_and_objective_use_different_classes(self):
        """这条是防止以后有人图省事把两者合并。"""
        self.assertIsNot(events_cv.TowerTracker, events_cv.ObjectiveTracker)


class ObjectiveFormDependsOnTime(unittest.TestCase):
    """同一个坑在不同时间刷的是不同的东西，收益差一个量级。"""

    def setUp(self):
        self.rules = rules.load(Path(tempfile.mkdtemp()))

    def test_tyrant_becomes_dark_tyrant(self):
        early = events_cv.resolve_objective_event("tyrantPit", 120, self.rules)
        switch = float(rules.value(self.rules, "shadowOverlordFromSec"))
        late = events_cv.resolve_objective_event("tyrantPit", switch + 60, self.rules)
        self.assertEqual(early, "TYRANT_KILL")
        self.assertEqual(late, "DARK_TYRANT_KILL")
        self.assertNotEqual(early, late, "两种形态被记成了同一个事件")

    def test_overlord_becomes_shadow_overlord(self):
        switch = float(rules.value(self.rules, "darkTyrantFromSec"))
        self.assertEqual(
            events_cv.resolve_objective_event("overlordPit", switch - 60, self.rules),
            "OVERLORD_KILL")
        self.assertEqual(
            events_cv.resolve_objective_event("overlordPit", switch + 60, self.rules),
            "PROPHET_OVERLORD_KILL")

    def test_storm_dragon_can_be_in_either_pit(self):
        switch = float(rules.value(self.rules, "stormDragonFromSec"))
        for name in ("stormDragonLowerPit", "stormDragonUpperPit"):
            self.assertEqual(events_cv.resolve_objective_event(name, switch + 1, self.rules),
                             "STORM_DRAGON_KILL")


class PortraitsNeedCorroboration(unittest.TestCase):
    """头像变灰只是一个信号。要有第二个信号才敢说「他死了」。"""

    def test_gray_portrait_is_detected(self):
        tracker = events_cv.PortraitTracker(2, "蓝2")
        tracker.update(0, COLOURFUL)
        event = tracker.update(240, GRAY)
        self.assertIsNotNone(event)
        self.assertEqual(event["type"], "PORTRAIT_GRAY")
        self.assertEqual(event["teamId"], BLUE)

    def test_respawn_clears_the_dead_flag(self):
        tracker = events_cv.PortraitTracker(2, "蓝2")
        tracker.update(0, COLOURFUL)
        tracker.update(240, GRAY)
        tracker.update(360, COLOURFUL)
        self.assertIsNotNone(tracker.update(500, GRAY), "复活后再死没有被检测到")
        self.assertEqual(len(tracker.deaths), 2)

    def test_two_signals_agreeing_gives_a_confirmed_kill(self):
        portraits = [{"type": "PORTRAIT_GRAY", "slot": 2, "teamId": BLUE, "atSec": 240}]
        scores = [{"atSec": 245, "teamId": RED, "delta": 1}]
        result = events_cv.corroborate_kills(portraits, scores)
        self.assertEqual(len(result["confirmed"]), 1)
        self.assertEqual(result["confirmed"][0]["teamId"], RED, "击杀方记反了")
        self.assertEqual(result["confirmed"][0]["victimSlot"], 2)

    def test_portrait_alone_is_not_a_kill(self):
        result = events_cv.corroborate_kills(
            [{"type": "PORTRAIT_GRAY", "slot": 2, "teamId": BLUE, "atSec": 240}], [])
        self.assertEqual(result["confirmed"], [],
                         "只凭头像变灰就断定阵亡 —— 特效遮挡也会让它变灰")
        self.assertEqual(len(result["portraitOnly"]), 1)

    def test_score_alone_is_reported_but_not_attributed_to_a_player(self):
        result = events_cv.corroborate_kills([], [{"atSec": 240, "teamId": RED}])
        self.assertEqual(result["confirmed"], [])
        self.assertEqual(len(result["scoreOnly"]), 1,
                         "比分动了说明确实死了人，不该丢掉，只是不知道是谁")

    def test_score_for_the_victims_own_team_is_not_a_match(self):
        """蓝方的人死了，加分的应该是红方。同队加分对不上。"""
        result = events_cv.corroborate_kills(
            [{"type": "PORTRAIT_GRAY", "slot": 2, "teamId": BLUE, "atSec": 240}],
            [{"atSec": 242, "teamId": BLUE}])
        self.assertEqual(result["confirmed"], [])


class AttributionIsHonestlyUnknown(unittest.TestCase):
    """图标消失说不出是谁打的 —— 这个「不知道」必须传下去。"""

    def test_no_nearby_score_means_unknown(self):
        result = events_cv.attribute_objective(
            {"type": "TYRANT_KILL", "atSec": 180, "pit": "tyrantPit"}, [])
        self.assertIsNone(result["teamId"])
        self.assertEqual(result["confidence"], 0.0)

    def test_both_teams_scoring_means_unknown(self):
        result = events_cv.attribute_objective(
            {"type": "TYRANT_KILL", "atSec": 180, "pit": "tyrantPit"},
            [{"atSec": 175, "teamId": BLUE}, {"atSec": 185, "teamId": RED}])
        self.assertIsNone(result["teamId"])

    def test_one_sided_scoring_is_only_a_lean_never_certain(self):
        result = events_cv.attribute_objective(
            {"type": "TYRANT_KILL", "atSec": 180, "pit": "tyrantPit"},
            [{"atSec": 178, "teamId": BLUE}])
        self.assertEqual(result["teamId"], BLUE)
        self.assertLess(result["confidence"], 0.5,
                        "打赢团战不等于拿到资源，置信度不能给高")

    def test_unattributed_events_never_reach_observations(self):
        scan_result = {"events": [
            {"type": "TYRANT_KILL", "atSec": 180, "teamId": None, "confidence": 0.0},
            {"type": "TOWER_DESTROYED", "atSec": 300, "teamId": RED, "confidence": 0.6},
        ]}
        written = events_cv.to_observation_events(scan_result)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["type"], "TOWER_DESTROYED")

    def test_low_confidence_events_are_dropped_not_downgraded(self):
        """低置信度的事件不能「降级写进去」——
        写进观测后计数器就 +1 了，一路错到胜率曲线。"""
        scan_result = {"events": [
            {"type": "OVERLORD_KILL", "atSec": 500, "teamId": BLUE, "confidence": 0.35},
        ]}
        self.assertEqual(events_cv.to_observation_events(scan_result), [])


class TowerAnnouncementsAreSupportingNotRequired(unittest.TestCase):
    BASE = [{"type": "TOWER_DESTROYED", "atSec": 300, "teamId": RED,
             "confidence": 0.62, "evidence": ["固定地标持续缺失"]}]

    def test_missing_announcement_keeps_state_machine_event(self):
        result = events_cv.corroborate_tower_events(self.BASE, [])
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["confidence"], 0.5)

    def test_matching_announcement_increases_confidence(self):
        result = events_cv.corroborate_tower_events(
            self.BASE, [{"type": "TOWER_DESTROYED", "atSec": 311, "teamId": RED}])
        self.assertGreater(result[0]["confidence"], self.BASE[0]["confidence"])
        self.assertIn("推塔播报佐证", result[0]["evidence"])

    def test_conflicting_announcement_forces_review(self):
        result = events_cv.corroborate_tower_events(
            self.BASE, [{"type": "TOWER_DESTROYED", "atSec": 301, "teamId": BLUE}])
        self.assertLess(result[0]["confidence"], 0.5)
        self.assertTrue(result[0]["needsReview"])


class TowerHighRateBacktracking(unittest.TestCase):
    EVENT = {"type": "TOWER_DESTROYED", "atSec": 120, "teamId": RED,
             "confidence": 0.62, "evidence": ["固定地标持续缺失"]}

    def test_sustained_missing_tail_tightens_event_time(self):
        result = events_cv.refine_tower_trace(self.EVENT, [
            {"atSec": 100.0, "distance": 0.0},
            {"atSec": 100.5, "distance": 0.03},
            {"atSec": 101.0, "distance": 0.22},
            {"atSec": 101.5, "distance": 0.24},
            {"atSec": 102.0, "distance": 0.23},
        ])
        self.assertTrue(result["backtrack"]["ok"])
        self.assertEqual(result["atSec"], 101.0)
        self.assertIn("事后高帧率回溯", result["evidence"])

    def test_reappearing_icon_is_rejected_as_occlusion(self):
        result = events_cv.refine_tower_trace(self.EVENT, [
            {"atSec": 100.0, "distance": 0.0},
            {"atSec": 100.5, "distance": 0.23},
            {"atSec": 101.0, "distance": 0.02},
        ])
        self.assertFalse(result["backtrack"]["ok"])
        self.assertLess(result["confidence"], 0.5)


class RefusesWithoutCalibration(unittest.TestCase):
    def test_scan_refuses_when_landmarks_are_not_calibrated(self):
        empty = Path(tempfile.mkdtemp())
        result = events_cv.scan([{"atSec": 0, "path": "/nope.jpg"}], empty,
                                rules.load(empty), [])
        self.assertFalse(result["ok"])
        self.assertEqual(result["events"], [])
        self.assertTrue(result.get("landmarksNeeded"))

    def test_saving_rejects_unknown_kinds_and_bad_boxes(self):
        directory = Path(tempfile.mkdtemp())
        events_cv.save_landmarks(directory, {
            "good": {"kind": "tower", "teamId": BLUE, "box": [0.1, 0.1, 0.05, 0.05]},
            "badKind": {"kind": "dragon", "box": [0.1, 0.1, 0.05, 0.05]},
            "badBox": {"kind": "tower", "box": [2.0, 0.1, 0.05, 0.05]},
        })
        items = events_cv.load_landmarks(directory)["items"]
        self.assertIn("good", items)
        self.assertNotIn("badKind", items)
        self.assertNotIn("badBox", items)

    def test_old_single_high_ground_tower_migrates_to_middle_lane(self):
        import json
        directory = Path(tempfile.mkdtemp())
        target = directory / "kpl" / "landmarks.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"calibrated": True, "items": {
            "blue_hg": {"kind": "tower", "teamId": BLUE,
                        "box": [0.1, 0.1, 0.02, 0.02]}
        }}), encoding="utf-8")
        items = events_cv.load_landmarks(directory)["items"]
        self.assertIn("blue_mid_hg", items)
        self.assertNotIn("blue_hg", items)

    def test_skill_and_storm_boxes_are_valid_landmark_kinds(self):
        directory = Path(tempfile.mkdtemp())
        events_cv.save_landmarks(directory, {
            "u1": {"kind": "ultimate", "slot": 1, "box": [0.1, 0.1, 0.01, 0.01]},
            "s1": {"kind": "summoner", "slot": 1, "box": [0.2, 0.1, 0.02, 0.02]},
            "dragon": {"kind": "storm_objective", "box": [0.3, 0.3, 0.03, 0.03]},
        })
        kinds = {v["kind"] for v in events_cv.load_landmarks(directory)["items"].values()}
        self.assertEqual(kinds, {"ultimate", "summoner", "storm_objective"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
