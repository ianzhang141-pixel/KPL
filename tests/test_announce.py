"""用播报补归属的性质测试。

这一步解决的是「暴君是谁拿的」—— 图标消失答不了的那个问题。
思路：粗扫定位到区间 → 回到区间里精抽帧 → 找播报横幅 → 从横幅判队伍。

盯的是三件会悄悄产出假归属的事：

1. 区间算错（往前找错了帧，窗口就是错的）
2. 一个区间里挤了多个事件时，把横幅硬认给其中一个
3. 横幅没找到或颜色分不清时，猜一个队伍

第 2 条最要命：团战里三个人头 + 一条龙共用一个区间，
那条播报到底属于哪个事件根本无从判断，猜一个就是编。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import announce, events_cv  # noqa: E402
from kplab.rules import BLUE, RED      # noqa: E402


def sig(r, g, b):
    return {"signature": {"r": float(r), "g": float(g), "b": float(b)}}


class BracketsTheEvent(unittest.TestCase):
    """事件必定发生在「上一帧图标还在」和「这一帧没了」之间。"""

    FRAMES = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]

    def test_bracket_is_the_gap_before_the_event(self):
        out = announce.bracket_events([{"type": "X", "atSec": 300.0}], self.FRAMES)
        bracket = out[0]["bracket"]
        self.assertEqual(bracket["fromSec"], 240.0)
        self.assertEqual(bracket["toSec"], 300.0)
        self.assertTrue(bracket["exact"])

    def test_first_frame_event_has_no_earlier_frame(self):
        """第一帧就没图标 —— 没有更早的帧可以定界，必须标出来。"""
        out = announce.bracket_events([{"type": "X", "atSec": 0.0}], self.FRAMES)
        self.assertFalse(out[0]["bracket"]["exact"],
                         "没有更早的帧却声称区间是精确的")

    def test_bracket_width_matches_sampling_interval(self):
        out = announce.bracket_events([{"type": "X", "atSec": 180.0}], self.FRAMES)
        self.assertEqual(out[0]["bracket"]["widthSec"], 60.0)

    def test_events_in_the_same_gap_are_grouped(self):
        bracketed = announce.bracket_events(
            [{"type": "A", "atSec": 300.0}, {"type": "B", "atSec": 300.0},
             {"type": "C", "atSec": 180.0}], self.FRAMES)
        groups = announce.group_by_bracket(bracketed)
        counts = sorted(g["count"] for g in groups)
        self.assertEqual(counts, [1, 2], "同一个区间里的事件没有被归到一组")


class TintOnlyDecidesTeamNotHero(unittest.TestCase):
    """横幅只用来判队伍。读文字要回到 OCR，那条路没这个可靠。"""

    def test_blue_banner_gives_blue(self):
        result = announce.read_tint([sig(40, 90, 220)] * 4)
        self.assertEqual(result["teamId"], BLUE)

    def test_red_banner_gives_red(self):
        result = announce.read_tint([sig(215, 55, 50)] * 4)
        self.assertEqual(result["teamId"], RED)

    def test_grey_banner_is_unknown(self):
        result = announce.read_tint([sig(120, 120, 124)] * 4)
        self.assertIsNone(result["teamId"], "颜色分不清却猜了一方")

    def test_margin_just_under_threshold_is_unknown(self):
        margin = announce.TINT_MARGIN - 1.0
        result = announce.read_tint([sig(100, 100, 100 + margin)] * 3)
        self.assertIsNone(result["teamId"])

    def test_no_samples_is_unknown(self):
        self.assertIsNone(announce.read_tint([])["teamId"])

    def test_confidence_never_claims_certainty(self):
        """颜色本身不是铁证，置信度必须封顶。"""
        result = announce.read_tint([sig(0, 0, 255)] * 5)
        self.assertEqual(result["teamId"], BLUE)
        self.assertLessEqual(result["strength"], 0.75,
                             "颜色判出来的归属不该有接近 1 的置信度")

    def test_never_returns_a_hero_name(self):
        result = announce.read_tint([sig(40, 90, 220)] * 3)
        self.assertNotIn("heroName", result)
        self.assertNotIn("player", result)


class RefusesWithoutCalibration(unittest.TestCase):
    def test_no_banner_region_means_no_attribution(self):
        result = announce.refine(
            Path("/nope.mp4"),
            {"events": [{"type": "TYRANT_KILL", "atSec": 180, "teamId": None}]},
            [120.0, 180.0], None)
        self.assertFalse(result["ok"])
        for event in result["events"]:
            self.assertIsNone(event.get("teamId"),
                              "播报区没标定却给出了归属")

    def test_missing_video_is_reported_not_guessed(self):
        result = announce.refine(
            Path("/nope.mp4"),
            {"events": [{"type": "TYRANT_KILL", "atSec": 180, "teamId": None}]},
            [120.0, 180.0], (0.3, 0.09, 0.34, 0.075))
        self.assertFalse(result["ok"])
        self.assertIn("录像", result["error"])


class AlreadyAttributedEventsAreLeftAlone(unittest.TestCase):
    def test_settled_events_are_not_rescanned(self):
        """已经知道归属的事件不用回去精扫 —— 省掉的是真金白银的抽帧时间。"""
        result = announce.refine(
            Path("/nope.mp4"),
            {"events": [{"type": "TOWER_DESTROYED", "atSec": 300, "teamId": RED}]},
            [240.0, 300.0], (0.3, 0.09, 0.34, 0.075))
        self.assertTrue(result["ok"])
        self.assertEqual(result["refined"], 0)
        self.assertEqual(result["events"][0]["teamId"], RED)


class CostIsBoundedByWindows(unittest.TestCase):
    """整套方法可行的前提：只精抽区间，不精抽整场。"""

    def test_cost_scales_with_windows_not_match_length(self):
        groups = [{"fromSec": 120.0, "toSec": 180.0, "count": 1},
                  {"fromSec": 240.0, "toSec": 300.0, "count": 1}]
        cost = announce.plan_cost(groups, step_sec=1.5)
        self.assertEqual(cost["windows"], 2)
        # 两个 60 秒窗口按 1.5 秒一帧 ≈ 82 帧，远小于整场 20 分钟按同密度的 800 帧
        self.assertLess(cost["framesToExtract"], 120)


class AmbiguousWindowsAreDowngraded(unittest.TestCase):
    """一个区间里有多个事件时，横幅属于哪一个无从判断。"""

    def test_multi_event_window_halves_confidence_below_write_threshold(self):
        """团战区间的归属必须掉到写入阈值以下，不能进观测。

        这是本模块最重要的一条：三个人头加一条龙共用一个区间，
        那条播报属于哪个事件根本判断不了。**减半之后正好过不了 0.5 的写入门槛**，
        于是这类猜测不会污染数据 —— 需要人工复核。
        """
        single = announce.read_tint([sig(40, 90, 220)] * 4)["strength"]
        halved = single * 0.5
        self.assertLess(halved, 0.5,
                        "多事件区间减半后仍然过得了写入阈值 —— 那等于把猜测写进了数据")
        self.assertEqual(
            events_cv.to_observation_events(
                {"events": [{"type": "TYRANT_KILL", "atSec": 180,
                             "teamId": BLUE, "confidence": halved}]}),
            [])

    def test_single_event_window_does_reach_the_threshold(self):
        """反过来也要成立，否则这套方法等于白做。"""
        strength = announce.read_tint([sig(40, 90, 220)] * 4)["strength"]
        self.assertGreaterEqual(strength, 0.5)
        self.assertEqual(len(events_cv.to_observation_events(
            {"events": [{"type": "TYRANT_KILL", "atSec": 180,
                         "teamId": BLUE, "confidence": strength}]})), 1)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 ffmpeg")
class RefineOnARealVideo(unittest.TestCase):
    """真正跑一遍 refine() —— 前面那些测试只碰到了守卫，没碰到主流程。

    **这一组是变异测试逼出来的。** 原来「团战区间置信度减半」那条测试
    是自己把数字乘 0.5 再断言，根本没调用 refine()；
    于是把代码里的 `confidence *= 0.5` 删掉，测试照样全过。
    同样，「没标定播报区要拒绝」那条因为顺带没给录像，
    删掉标定检查后被录像检查兜住了，也没抓到。

    和之前 schema.derive()、小地图守卫那两次是同一个病：
    **契约要在契约所在的那一层测。**
    """

    WIDTH, HEIGHT, FPS = 160, 90, 2
    BANNER_BOX = (0.25, 0.10, 0.50, 0.20)

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        raw = cls.tmp / "raw"
        raw.mkdir()
        # 20 秒：t=8~12 有一条蓝色播报，其余时间播报区是空的
        for index in range(20 * cls.FPS):
            at = index / cls.FPS
            banner = (40, 90, 220) if 8.0 <= at <= 12.0 else (30, 34, 40)
            rows = []
            for y in range(cls.HEIGHT):
                row = []
                for x in range(cls.WIDTH):
                    in_banner = (0.25 <= x / cls.WIDTH <= 0.75
                                 and 0.10 <= y / cls.HEIGHT <= 0.30)
                    row.append(banner if in_banner else (24, 28, 34))
                rows.append(row)
            out = bytearray(b"P6\n%d %d\n255\n" % (cls.WIDTH, cls.HEIGHT))
            for row in rows:
                for pixel in row:
                    out += bytes(pixel)
            (raw / f"{index:05d}.ppm").write_bytes(out)

        cls.video = cls.tmp / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(cls.FPS),
             "-i", str(raw / "%05d.ppm"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-r", str(cls.FPS), str(cls.video)], check=True)

    def test_single_event_window_gets_attributed(self):
        result = announce.refine(
            self.video,
            {"events": [{"type": "TYRANT_KILL", "atSec": 15.0, "teamId": None}]},
            [0.0, 15.0], self.BANNER_BOX, step_sec=1.0)
        self.assertTrue(result["ok"], result.get("error"))
        event = result["events"][0]
        self.assertEqual(event["teamId"], BLUE, "播报是蓝色的，归属应该判给蓝方")
        self.assertGreaterEqual(event["confidence"], 0.5,
                                "单事件区间的归属应该够格写进观测")

    def test_multi_event_window_is_actually_halved_by_the_code(self):
        """直接对比同一段录像下「一个事件」和「三个事件」的置信度。"""
        one = announce.refine(
            self.video,
            {"events": [{"type": "TYRANT_KILL", "atSec": 15.0, "teamId": None}]},
            [0.0, 15.0], self.BANNER_BOX, step_sec=1.0)["events"][0]["confidence"]
        many = announce.refine(
            self.video,
            {"events": [{"type": "TYRANT_KILL", "atSec": 15.0, "teamId": None},
                        {"type": "HERO_KILL", "atSec": 15.0, "teamId": None},
                        {"type": "TOWER_DESTROYED", "atSec": 15.0, "teamId": None}]},
            [0.0, 15.0], self.BANNER_BOX, step_sec=1.0)["events"][0]["confidence"]
        self.assertAlmostEqual(many, one * 0.5, places=3,
                               msg="团战区间的置信度没有被减半")
        self.assertLess(many, 0.5, "减半后仍然过得了写入阈值")
        self.assertEqual(events_cv.to_observation_events({"events": [
            {"type": "TYRANT_KILL", "atSec": 15.0, "teamId": BLUE, "confidence": many}]}), [])

    def test_uncalibrated_banner_refuses_even_with_a_valid_video(self):
        """录像是好的，只是播报区没标定 —— 必须因为标定而拒绝。

        原来这条测试顺带用了个不存在的录像，于是删掉标定检查后
        被「找不到录像」兜住了，变异抓不到。
        """
        result = announce.refine(
            self.video,
            {"events": [{"type": "TYRANT_KILL", "atSec": 15.0, "teamId": None}]},
            [0.0, 15.0], None, step_sec=1.0)
        self.assertFalse(result["ok"])
        self.assertIn("播报区", result["error"])
        for event in result["events"]:
            self.assertIsNone(event.get("teamId"))

    def test_window_without_a_banner_stays_unknown(self):
        """区间落在 t=15~20，那段没有播报 —— 必须仍然是不知道。"""
        result = announce.refine(
            self.video,
            {"events": [{"type": "OVERLORD_KILL", "atSec": 19.0, "teamId": None}]},
            [15.0, 19.0], self.BANNER_BOX, step_sec=1.0)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["events"][0].get("teamId"))


class BannerRegionIsRegistered(unittest.TestCase):
    def test_banner_is_a_calibratable_hud_region(self):
        from kplab import hud
        self.assertIn(announce.BANNER_REGION, hud.REGIONS)

    def test_banner_subregions_are_all_calibratable(self):
        from kplab import hud
        for region in announce.BANNER_FIELD_REGIONS.values():
            self.assertIn(region, hud.REGIONS)


class StructuredBannerFields(unittest.TestCase):
    def test_objective_type_and_assists_are_preserved(self):
        parsed = announce.parse_fields({
            "bannerKiller": "暖阳",
            "bannerTarget": "对方打野",
            "bannerObjective": "击败风暴龙王",
            "bannerAssists": "选手A、选手B",
        })
        self.assertEqual(parsed["objectiveType"], "STORM_DRAGON_KILL")
        self.assertEqual(parsed["killer"], "暖阳")
        self.assertEqual(parsed["assists"], ["选手A", "选手B"])

    def test_unreadable_fields_stay_unknown(self):
        parsed = announce.parse_fields({})
        self.assertIsNone(parsed["killer"])
        self.assertIsNone(parsed["objectiveType"])
        self.assertFalse(parsed["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
