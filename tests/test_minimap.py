"""小地图识别的性质测试。

这些测试**不需要 ffmpeg** —— 它们直接喂原始 RGB 字节，绕开取像素那一步。
这样在没装 ffmpeg 的机器上（比如用户的 Mac 第一次跑）也能验证算法本身。

需要 ffmpeg 的那部分单独放在最后，装了才跑，没装自动跳过。

盯的是同一类东西：**不会报错、只会把数据悄悄弄坏的行为**。
小地图这块尤其危险，因为「找到 7 个蓝点」和「找到 5 个蓝点」在代码里
长得一模一样，只有规则知道前者不可能。
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import minimap, schema  # noqa: E402

# 64 而不是 40：半径 2 的圆点直径 5，圆心间距必须 ≥6 才不会连成一片。
# 一开始用 40×40、半径 3、间距 5，结果点全粘在一起 ——
# 连通域把它们判成一个标记是对的，是夹具摆得太密。
SIZE = 64


def make_canvas(background=(20, 25, 35)):
    return bytearray(bytes(background) * (SIZE * SIZE))


def draw_dot(canvas, cx, cy, colour, radius=2):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                base = (y * SIZE + x) * 3
                canvas[base:base + 3] = bytes(colour)


BLUE = (40, 110, 235)
RED = (230, 60, 50)


def detect_canvas(canvas, thresholds=None):
    """跳过 ffmpeg，直接跑分类 + 连通域 + 评估。"""
    thresholds = thresholds or minimap.default_thresholds()
    labels = minimap.classify_pixels(bytes(canvas), SIZE, SIZE, thresholds)
    lo = 4          # 小画布上把最小尺寸放宽，否则 3 像素半径的点会被滤掉
    hi = int(thresholds["maxBlobPixels"])
    blue, blue_rej = minimap.find_blobs(labels, SIZE, SIZE, 1, lo, hi)
    red, red_rej = minimap.find_blobs(labels, SIZE, SIZE, 2, lo, hi)
    coloured = sum(1 for label in labels if label) / (SIZE * SIZE)
    return {
        "blue": blue, "red": red,
        "blueCount": len(blue), "redCount": len(red),
        "quality": minimap.assess(len(blue), len(red),
                                  bool(thresholds.get("verified")),
                                  blue_rej, red_rej, coloured),
    }


class FindsMarkers(unittest.TestCase):
    def test_finds_five_and_five(self):
        canvas = make_canvas()
        for x, y in ((8, 8), (24, 10), (10, 28), (26, 40), (8, 52)):
            draw_dot(canvas, x, y, BLUE)
        for x, y in ((44, 8), (56, 24), (40, 32), (52, 44), (44, 56)):
            draw_dot(canvas, x, y, RED)
        result = detect_canvas(canvas)
        self.assertEqual(result["blueCount"], 5)
        self.assertEqual(result["redCount"], 5)
        self.assertTrue(result["quality"]["usable"])

    def test_centroid_is_close_to_truth(self):
        canvas = make_canvas()
        draw_dot(canvas, 16, 32, BLUE)
        result = detect_canvas(canvas)
        blob = result["blue"][0]
        self.assertAlmostEqual(blob["x"], 16 / SIZE, delta=0.03)
        self.assertAlmostEqual(blob["y"], 32 / SIZE, delta=0.03)

    def test_fewer_than_five_is_fine_not_an_error(self):
        """有人在草丛/阴影里看不见是正常的，不该判为坏数据。"""
        canvas = make_canvas()
        for x, y in ((8, 8), (24, 10), (10, 28)):
            draw_dot(canvas, x, y, BLUE)
        for x, y in ((44, 8), (56, 24), (40, 32)):
            draw_dot(canvas, x, y, RED)
        result = detect_canvas(canvas)
        self.assertTrue(result["quality"]["usable"], "找到的人少于 5 个不等于识别失败")


class RejectsImpossibleResults(unittest.TestCase):
    """这一组是重点：宁可说「不知道」，也不许挑几个交差。"""

    def test_more_than_five_per_team_is_fatal(self):
        canvas = make_canvas()
        for i in range(7):     # 7 个蓝点 —— 不可能
            draw_dot(canvas, 4 + i * 8, 20, BLUE)
        result = detect_canvas(canvas)
        self.assertEqual(result["blueCount"], 7)
        self.assertFalse(result["quality"]["usable"],
                         "超过每方 5 人仍判为可用 —— 这会把假位置喂进下游")
        self.assertEqual(result["quality"]["confidence"], 0.0)

    def test_never_picks_top_five_to_save_face(self):
        """超限时不许砍成 5 个交差。产出必须是空的。"""
        canvas = make_canvas()
        for i in range(8):
            draw_dot(canvas, 4 + (i % 4) * 14, 16 + (i // 4) * 24, BLUE)
        result = detect_canvas(canvas)
        self.assertEqual(minimap.to_observation_players(result), {},
                         "识别不可用时仍产出了位置字段")

    def test_oversize_blob_is_reported_not_swallowed(self):
        """整片底色被判成队伍色时，超大团块被丢弃 —— 那是最强的告警信号。

        这条测试来自一个真 bug：超大团块被静默丢弃后，
        「蓝方 0 个」看起来像是正常的「没找到」，于是整帧被判为可用。
        """
        canvas = make_canvas(background=BLUE)   # 整张图都是蓝的
        for x, y in ((44, 8), (56, 24), (40, 32)):
            draw_dot(canvas, x, y, RED)
        result = detect_canvas(canvas)
        self.assertFalse(result["quality"]["usable"])
        self.assertTrue(
            any("超大色块" in p or "阈值太松" in p for p in result["quality"]["problems"]),
            "超大团块被丢弃却没有报出来")

    def test_one_team_missing_entirely_is_suspicious(self):
        """一方全无、另一方好几个 —— 五个人同时消失的可能性极低。"""
        canvas = make_canvas()
        for x, y in ((44, 8), (56, 24), (40, 32), (52, 44)):
            draw_dot(canvas, x, y, RED)
        result = detect_canvas(canvas)
        self.assertEqual(result["blueCount"], 0)
        self.assertFalse(result["quality"]["usable"],
                         "一方完全找不到却判为可用 —— 多半是阈值坏了")

    def test_empty_image_finds_nothing_and_says_so(self):
        result = detect_canvas(make_canvas())
        self.assertEqual(result["blueCount"], 0)
        self.assertEqual(result["redCount"], 0)
        self.assertFalse(result["quality"]["usable"])


class EachGuardPinnedOnItsOwn(unittest.TestCase):
    """逐条钉住 assess() 里的每一道守卫，互不依赖。

    **为什么需要这一组：** 上面那些用画布的测试是「端到端」的，
    一次会同时触发好几道守卫。变异测试发现：把「超过 5 人」那条守卫整个删掉，
    测试竟然还全过 —— 因为那个夹具同时也触发了「一方全无」那条守卫。

    这和立项时在 schema.derive() 上踩的是同一个坑：
    **有别的守卫兜着，不等于这道守卫被测到了。**
    所以这里直接调 assess()，每次只让一个条件成立。
    """

    CLEAN = {"tooSmall": 0, "tooLarge": 0, "largestRejectedPixels": 0}

    def test_baseline_is_usable(self):
        """先确认基准是可用的，否则下面每一条都可能是假阳性。"""
        verdict = minimap.assess(3, 3, True, self.CLEAN, self.CLEAN, 0.02)
        self.assertTrue(verdict["usable"])
        self.assertGreater(verdict["confidence"], 0)

    def test_guard_more_than_five(self):
        # 只有「蓝方 7 人」这一条不对：红方 3 人正常、无丢弃、染色比例正常
        verdict = minimap.assess(7, 3, True, self.CLEAN, self.CLEAN, 0.02)
        self.assertFalse(verdict["usable"], "「每方最多 5 人」这道守卫没起作用")
        self.assertEqual(verdict["confidence"], 0.0)

    def test_guard_oversize_blob(self):
        # 只有「有超大团块被丢弃」这一条不对
        oversize = {"tooSmall": 0, "tooLarge": 1, "largestRejectedPixels": 20000}
        verdict = minimap.assess(3, 3, True, oversize, self.CLEAN, 0.02)
        self.assertFalse(verdict["usable"], "「超大团块」这道守卫没起作用")

    def test_guard_coloured_ratio(self):
        # 只有「染色面积过大」这一条不对
        verdict = minimap.assess(3, 3, True, self.CLEAN, self.CLEAN, 0.6)
        self.assertFalse(verdict["usable"], "「染色面积」这道守卫没起作用")

    def test_guard_one_team_missing(self):
        # 只有「一方全无」这一条不对
        verdict = minimap.assess(0, 4, True, self.CLEAN, self.CLEAN, 0.02)
        self.assertFalse(verdict["usable"], "「一方全无」这道守卫没起作用")

    def test_guard_nothing_found(self):
        verdict = minimap.assess(0, 0, True, self.CLEAN, self.CLEAN, 0.0)
        self.assertFalse(verdict["usable"])

    def test_low_count_is_not_fatal(self):
        """守卫不能过度触发：只找到 2 个人是信息不全，不是数据错误。"""
        verdict = minimap.assess(1, 1, True, self.CLEAN, self.CLEAN, 0.01)
        self.assertTrue(verdict["usable"], "找到的人少不等于识别坏了")


class HonestAboutWhatItKnows(unittest.TestCase):
    """本模块认不出英雄，也不许假装认得出。"""

    def test_never_produces_a_hero_name(self):
        canvas = make_canvas()
        for x, y in ((8, 8), (24, 10)):
            draw_dot(canvas, x, y, BLUE)
        result = detect_canvas(canvas)
        for blob in result["blue"] + result["red"]:
            self.assertNotIn("heroName", blob, "小地图识别不该产出英雄名 —— 它认不出来")

    def test_positions_carry_cv_source_and_low_confidence(self):
        canvas = make_canvas()
        for x, y in ((8, 8), (24, 10), (10, 28)):
            draw_dot(canvas, x, y, BLUE)
        for x, y in ((44, 8), (56, 24), (40, 32)):
            draw_dot(canvas, x, y, RED)
        players = minimap.to_observation_players(detect_canvas(canvas))
        self.assertTrue(players)
        for entry in players.values():
            self.assertEqual(schema.src(entry["x"]), schema.SOURCE_CV)
            self.assertFalse(
                schema.trusted(entry["x"]),
                "小地图猜出来的位置不该达到可信线 —— 「这是几号位」基本是猜的")

    def test_unverified_thresholds_lower_confidence(self):
        canvas = make_canvas()
        for x, y in ((8, 8), (24, 10), (10, 28)):
            draw_dot(canvas, x, y, BLUE)
        for x, y in ((44, 8), (56, 24), (40, 32)):
            draw_dot(canvas, x, y, RED)
        unverified = detect_canvas(canvas)["quality"]["confidence"]
        table = minimap.default_thresholds()
        table["verified"] = True
        verified = detect_canvas(canvas, table)["quality"]["confidence"]
        self.assertLess(unverified, verified, "没标定过的阈值应该给更低的置信度")


class PortraitTemplateAssociation(unittest.TestCase):
    @staticmethod
    def patterned(size):
        colours = ((220, 50, 40), (40, 210, 70), (45, 75, 220), (220, 190, 45))
        raw = bytearray()
        for y in range(size):
            for x in range(size):
                quadrant = (1 if y >= size // 2 else 0) * 2 + (1 if x >= size // 2 else 0)
                raw.extend(colours[quadrant])
        return bytes(raw)

    def test_same_appearance_scores_higher_than_different(self):
        same = minimap.appearance_descriptor(self.patterned(24), 24, 24)
        flat = minimap.appearance_descriptor(bytes((90, 90, 90)) * 24 * 24, 24, 24)
        self.assertGreater(minimap.descriptor_similarity(same, same),
                           minimap.descriptor_similarity(same, flat))

    def test_slot_is_matched_to_minimap_icon_not_position_sorted(self):
        canvas = bytearray(bytes((20, 25, 35)) * (SIZE * SIZE))
        icon = self.patterned(10)
        for y in range(10):
            start = ((27 + y) * SIZE + 27) * 3
            canvas[start:start + 30] = icon[y * 30:(y + 1) * 30]
        matches = minimap.match_portraits(
            bytes(canvas), SIZE, SIZE, {4: self.patterned(24)}, min_similarity=0.55)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["slot"], 4)
        self.assertAlmostEqual(matches[0]["x"], 0.5, delta=0.08)
        self.assertAlmostEqual(matches[0]["y"], 0.5, delta=0.08)

    def test_observation_keeps_template_matched_slot(self):
        result = {"method": "portrait-template", "quality": {"usable": True},
                  "matches": [{"slot": 8, "x": 0.2, "y": 0.7, "confidence": 0.64}]}
        players = minimap.to_observation_players(result)
        self.assertEqual(set(players), {"8"})


@unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg")
class PixelReadingNeedsFfmpeg(unittest.TestCase):
    def test_missing_file_raises_clean_error(self):
        from kplab import video
        with self.assertRaises(video.VideoError):
            minimap.read_rgb(Path("/不存在的图片.jpg"))

    def test_box_out_of_range_is_rejected(self):
        from kplab import video
        with self.assertRaises(video.VideoError):
            minimap.read_rgb(Path("/不存在的图片.jpg"), (0.5, 0.5, 0.9, 0.9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
