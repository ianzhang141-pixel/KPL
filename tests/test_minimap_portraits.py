"""小地图头像逐一匹配：在一张**已知答案**的合成画面上量召回和误检。

## 为什么要合成一张画面

用户报的是「十个人只认出四个，而且把龙坑里的龙认成了英雄」。
这两件事在真实录像上一眼就能看见，但真实录像不在仓库里，
而且没有逐帧的位置真值 —— 没有真值就没法说改动到底是变好还是变坏。

所以这里画一张假的比赛画面：小地图上放 10 个位置已知的英雄图标，
两侧栏放同样 10 个英雄的头像（更大、方形、四角是 UI 装饰），
再在地图中间放一个**和英雄一样大、一样圆、一样有亮边框**的龙图标。

它当然不等于真画面 —— 合成的图案比真英雄干净得多，所以这里的分数
**不能当成真实录像上的预期值**。它能保证的是方向：
同一张图上，改动之后认对的人不该变少，龙不该被认成人。

## 这张图专门复现的两个坑

1. **深色英雄被硬门槛提前扔掉。** 5 号英雄的配色和地图底色接近，
   圆形边缘分数只有 0.34。旧版 ``circle < 0.42`` 一票否决，
   它连候选都进不去 —— 而边缘反差跟「他是不是个人」并没有必然联系。
2. **龙顶替了那个空出来的槽位。** 5 号找不到自己之后，
   分数第二高的就是龙 —— 于是龙被贴上「蓝5」的标签画在了图上。
   把资源坑框标定出来、对那一圈额外抬高门槛之后，龙过不去了，
   5 号也就老老实实地报「没找到」而不是指着龙说那是他。
"""

from __future__ import annotations

import math
import random
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from kplab import minimap

WIDTH, HEIGHT = 1280, 720
# 小地图在左上角，和 hud 内置坐标一致。
MINIMAP_BOX = (0.0, 0.0, 0.150, 0.270)


# ---------------------------------------------------------------- 画一张假画面

def _write_png(path: Path, pixels: list[list[list[int]]]) -> None:
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _draw_icon(pixels, cx, cy, radius, art, border) -> None:
    for y in range(cy - radius - 2, cy + radius + 3):
        for x in range(cx - radius - 2, cx + radius + 3):
            if not (0 <= x < len(pixels[0]) and 0 <= y < len(pixels)):
                continue
            distance = math.hypot(x - cx, y - cy)
            if distance > radius + 2:
                continue
            pixels[y][x] = list(border) if distance > radius - 1 else list(art(x - cx, y - cy, radius))


def _hero_art(seed: int):
    """十个英雄十种几何，彼此结构不同。"""
    rnd = random.Random(seed * 977 + 13)
    base = (rnd.randrange(50, 200), rnd.randrange(50, 200), rnd.randrange(50, 200))
    accent = (rnd.randrange(60, 255), rnd.randrange(60, 255), rnd.randrange(60, 255))
    mode = (seed - 1) % 10

    def art(dx, dy, radius):
        u, v = dx / max(1, radius), dy / max(1, radius)
        if mode == 0:
            hit = v < -0.15
        elif mode == 1:
            hit = u < -0.15
        elif mode == 2:
            hit = (u * u + v * v) < 0.25
        elif mode == 3:
            hit = (u + v) > 0.1
        elif mode == 4:
            hit = (u - v) > 0.1
        elif mode == 5:
            hit = abs(u) < 0.3
        elif mode == 6:
            hit = abs(v) < 0.3
        elif mode == 7:
            hit = (u * u + v * v) > 0.45
        elif mode == 8:
            hit = (u > 0) != (v > 0)
        else:
            hit = abs(u) + abs(v) < 0.6
        return accent if hit else base

    return art


def _dragon_art(dx, dy, radius):
    """龙：也是一个带亮边框的圆形图标 —— 天生就长得像「某个英雄」。"""
    return (215, 190, 90) if (dx * dx + dy * dy) < (radius * 0.45) ** 2 else (120, 60, 150)


def _draw_beam(pixels, x0, y0, x1, y1, half_width, colour) -> None:
    """回城/传送特效：一条又粗又长的亮紫色光柱。

    它的 RGB 让 classify_pixels 判成蓝方（蓝够亮、蓝比红多、蓝比绿多），
    而且用户实测里它真的被当成了一个英雄头像。它和英雄标记的区别不在颜色，
    在**形状和大小** —— 它是一整条几十上百像素的连通区域。
    """
    length = max(1.0, math.hypot(x1 - x0, y1 - y0))
    for step in range(int(length) + 1):
        cx = round(x0 + (x1 - x0) * step / length)
        cy = round(y0 + (y1 - y0) * step / length)
        for dy in range(-half_width, half_width + 1):
            for dx in range(-half_width, half_width + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < len(pixels[0]) and 0 <= y < len(pixels):
                    pixels[y][x] = list(colour)


SPOTS = [(0.20, 0.22), (0.32, 0.40), (0.18, 0.62), (0.44, 0.70), (0.30, 0.84),
         (0.72, 0.24), (0.60, 0.44), (0.80, 0.58), (0.66, 0.76), (0.86, 0.86)]


def build_scene(directory: Path) -> dict:
    pixels = [[[18, 24, 20] for _ in range(WIDTH)] for _ in range(HEIGHT)]
    mx, my = int(MINIMAP_BOX[0] * WIDTH), int(MINIMAP_BOX[1] * HEIGHT)
    mw, mh = int(MINIMAP_BOX[2] * WIDTH), int(MINIMAP_BOX[3] * HEIGHT)
    rnd = random.Random(7)
    for y in range(my, my + mh):                       # 地图底色 + 一点纹理
        for x in range(mx, mx + mw):
            noise = rnd.randrange(-8, 9)
            pixels[y][x] = [max(0, 26 + noise), max(0, 34 + noise), max(0, 28 + noise)]

    icon_radius = max(6, int(mw * 0.075))
    truth: dict[int, tuple[float, float]] = {}
    for slot, (fx, fy) in enumerate(SPOTS, start=1):
        cx, cy = mx + int(fx * mw), my + int(fy * mh)
        border = (60, 120, 235) if slot <= 5 else (225, 70, 70)
        _draw_icon(pixels, cx, cy, icon_radius, _hero_art(slot), border)
        truth[slot] = ((cx - mx) / mw, (cy - my) / mh)

    dcx, dcy = mx + int(0.52 * mw), my + int(0.50 * mh)
    _draw_icon(pixels, dcx, dcy, icon_radius, _dragon_art, (200, 160, 60))
    pad = icon_radius + 3

    # 回城特效：一条会被判成「蓝色」的粗光柱。用户实测里它被认成了红10。
    bx0, by0 = mx + int(0.08 * mw), my + int(0.06 * mh)
    bx1, by1 = mx + int(0.40 * mw), my + int(0.16 * mh)
    _draw_beam(pixels, bx0, by0, bx1, by1, max(3, icon_radius // 2), (170, 60, 230))
    beam = (((bx0 + bx1) / 2 - mx) / mw, ((by0 + by1) / 2 - my) / mh)

    # 地图上零散的队伍色小装饰（守卫、路标）。太小，不该被当成英雄。
    for fx, fy, colour in ((0.90, 0.12, (60, 120, 235)), (0.10, 0.44, (225, 70, 70))):
        cx, cy = mx + int(fx * mw), my + int(fy * mh)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                pixels[cy + dy][cx + dx] = list(colour)

    scene = {
        "truth": truth,
        "dragon": ((dcx - mx) / mw, (dcy - my) / mh),
        "beam": beam,
        # 用户在标定页上框「暴君坑」时会框出来的那个框（画面相对坐标）
        "pitBox": ((dcx - pad) / WIDTH, (dcy - pad) / HEIGHT,
                   (2 * pad) / WIDTH, (2 * pad) / HEIGHT),
    }

    half = 26
    portrait_boxes: dict[int, tuple[float, float, float, float]] = {}
    for slot in range(1, 11):
        blue = slot <= 5
        x0 = 30 if blue else WIDTH - 30 - half * 2
        y0 = 120 + ((slot - 1) % 5) * (half * 2 + 14)
        for y in range(y0, y0 + half * 2):             # 方形底 = UI 装饰
            for x in range(x0, x0 + half * 2):
                pixels[y][x] = [46, 52, 64]
        for y in range(y0, y0 + 10):                   # 四角的等级数字之类
            for x in range(x0, x0 + 10):
                pixels[y][x] = [240, 210, 60]
        _draw_icon(pixels, x0 + half, y0 + half, half - 3, _hero_art(slot),
                   (60, 120, 235) if blue else (225, 70, 70))
        portrait_boxes[slot] = (x0 / WIDTH, y0 / HEIGHT,
                                (half * 2) / WIDTH, (half * 2) / HEIGHT)
    scene["portraitBoxes"] = portrait_boxes
    scene["image"] = directory / "scene.png"
    _write_png(scene["image"], pixels)
    return scene


# ---------------------------------------------------------------- 不需要 ffmpeg 的部分

class CircleMaskComparesLikeWithLike(unittest.TestCase):
    """小地图上的标记是圆的，侧栏头像是方的，四角是完全不同的两样东西。"""

    @staticmethod
    def _square(size: int, inside: tuple[int, int, int], corner: tuple[int, int, int]) -> bytes:
        raw = bytearray()
        centre = (size - 1) / 2
        for y in range(size):
            for x in range(size):
                far = ((x - centre) ** 2 + (y - centre) ** 2) > (size / 2) ** 2
                raw.extend(corner if far else inside)
        return bytes(raw)

    def test_corners_do_not_affect_a_masked_descriptor(self):
        left = self._square(24, (200, 60, 60), (250, 210, 60))     # 四角是等级数字
        right = self._square(24, (200, 60, 60), (26, 34, 28))      # 四角是地图底色
        masked = minimap.descriptor_similarity(
            minimap.appearance_descriptor(left, 24, 24),
            minimap.appearance_descriptor(right, 24, 24))
        unmasked = minimap.descriptor_similarity(
            minimap.appearance_descriptor(left, 24, 24, circle=False),
            minimap.appearance_descriptor(right, 24, 24, circle=False))
        self.assertGreater(masked, unmasked + 0.05,
                           "圆形掩码没生效：方形四角还在拉低同一个英雄的相似度。")
        self.assertGreater(masked, 0.95)

    def test_structure_descriptor_does_not_invent_values_outside_the_circle(self):
        # 圆外的格子没有像素。补成平均值等于凭空造出一个「和模板一致」的格子。
        raw = self._square(24, (200, 60, 60), (250, 210, 60))
        values = minimap.structure_descriptor(raw, 24, 24)
        self.assertEqual(len(values), 64)
        self.assertEqual(values[0], 0.0, "左上角在圆外，不该有自己的结构值。")


class StaticZonesOnlyCoverTheMinimap(unittest.TestCase):
    def test_boxes_outside_the_minimap_are_dropped(self):
        zones = minimap.static_zones_in_minimap(
            (0.0, 0.0, 0.15, 0.27),
            [(0.05, 0.10, 0.02, 0.03),      # 小地图里的龙坑
             (0.80, 0.30, 0.04, 0.07)],     # 右侧栏的头像，不在小地图上
            160, 160)
        self.assertEqual(len(zones), 1,
                         "小地图外的地标不能在小地图上凭空变出一块禁区。")

    def test_zone_is_converted_into_minimap_pixels(self):
        (x0, y0, x1, y1), = minimap.static_zones_in_minimap(
            (0.0, 0.0, 0.20, 0.20), [(0.10, 0.10, 0.02, 0.02)], 100, 100)
        self.assertAlmostEqual(x0, 50.0, delta=0.01)
        self.assertAlmostEqual(y0, 50.0, delta=0.01)
        self.assertAlmostEqual(x1, 60.0, delta=0.01)
        self.assertAlmostEqual(y1, 60.0, delta=0.01)

    def test_static_zone_raises_the_bar_but_does_not_forbid(self):
        # 龙坑正是最该看清有几个人的地方。一票否决等于开龙团的瞬间集体失明。
        self.assertGreater(minimap.STATIC_ZONE_MARGIN, 0.0)
        self.assertLess(minimap.STATIC_ZONE_MARGIN, 0.2)


class CircleScoreIsGradedNotVetoed(unittest.TestCase):
    """圆边反差跟英雄配色和地形有关，和「他是不是个人」没有必然联系。"""

    def test_weak_edges_cost_score_instead_of_being_thrown_away(self):
        self.assertLess(minimap.CANDIDATE_CIRCLE_FLOOR, 0.42,
                        "候选门槛又抬回硬否决了：深色英雄站在深色野区会整类消失。")
        self.assertGreater(minimap.circle_surcharge(0.30),
                           minimap.circle_surcharge(0.45))
        self.assertEqual(minimap.circle_surcharge(0.60), 0.0,
                         "边缘够清楚就不该再加价。")


class ThresholdsAreValidatedOnSave(unittest.TestCase):
    """这份文件之后每一帧都会读，写坏了不会报错，只会让识别悄悄失败。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_out_of_range_similarity_is_rejected(self):
        with self.assertRaises(ValueError):
            minimap.save_thresholds(self.dir, {"minSimilarity": 5})

    def test_non_numeric_is_rejected(self):
        with self.assertRaises(ValueError):
            minimap.save_thresholds(self.dir, {"minBlobPixels": "abc"})

    def test_negative_pixel_count_is_rejected(self):
        with self.assertRaises(ValueError):
            minimap.save_thresholds(self.dir, {"minBlobPixels": -3})

    def test_good_values_round_trip(self):
        minimap.save_thresholds(self.dir, {"minSimilarity": 0.72, "minBlobPixels": 20})
        saved = minimap.load_thresholds(self.dir)
        self.assertAlmostEqual(saved["minSimilarity"], 0.72)
        self.assertEqual(saved["minBlobPixels"], 20)


# ---------------------------------------------------------------- 需要 ffmpeg 的整条链路

@unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg 才能读像素")
class SyntheticSceneRecall(unittest.TestCase):
    """一张已知答案的合成画面。分数不能外推到真实录像，方向可以。"""

    POSITION_TOLERANCE = 0.05          # 小地图宽度的 5%

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.scene = build_scene(cls.dir)
        thresholds = minimap.default_thresholds()
        thresholds["verified"] = True
        cls.with_pits = minimap.detect_with_portraits(
            cls.scene["image"], MINIMAP_BOX, cls.scene["portraitBoxes"], thresholds,
            static_landmark_boxes=[cls.scene["pitBox"]])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _placed_correctly(self, result) -> list[int]:
        good = []
        for match in result["matches"]:
            gx, gy = self.scene["truth"][match["claimedSlot"]]
            if math.hypot(match["x"] - gx, match["y"] - gy) <= self.POSITION_TOLERANCE:
                good.append(match["claimedSlot"])
        return good

    def test_finds_most_of_the_ten(self):
        good = self._placed_correctly(self.with_pits)
        self.assertGreaterEqual(
            len(good), 8,
            f"合成画面上只认对 {len(good)}/10。逐槽位体检："
            + "; ".join(f"{s['slot']}:{s['status']}@{s.get('bestScore', 0)}"
                        for s in self.with_pits["slots"]))

    def test_no_match_lands_on_the_dragon(self):
        dx, dy = self.scene["dragon"]
        for match in self.with_pits["matches"]:
            distance = math.hypot(match["x"] - dx, match["y"] - dy)
            self.assertGreater(
                distance, 0.05,
                f"槽位 {match['claimedSlot']} 被贴到了龙身上 —— "
                "标定过的资源坑框没有起到提防作用。")

    def test_every_reported_match_is_where_it_really_is(self):
        # 宁可漏掉也不能指错人：报出来的每一个都必须落在真值附近。
        good = set(self._placed_correctly(self.with_pits))
        wrong = [m["claimedSlot"] for m in self.with_pits["matches"]
                 if m["claimedSlot"] not in good]
        self.assertEqual(wrong, [], f"这些槽位被放到了错误的位置：{wrong}")

    def test_no_match_lands_on_the_recall_beam(self):
        bx, by = self.scene["beam"]
        for match in self.with_pits["matches"]:
            self.assertGreater(
                math.hypot(match["x"] - bx, match["y"] - by), 0.10,
                f"槽位 {match['claimedSlot']} 被贴到了回城光柱上 —— "
                "那是一整条几十上百像素的连通区域，不可能是英雄标记。")

    def test_no_slot_is_matched_to_the_other_teams_icon(self):
        # 「连最基础的红蓝轮廓都分不清」是用户原话。队伍色是硬条件，不是加分项。
        for match in self.with_pits["matches"]:
            owner = min(self.scene["truth"],
                        key=lambda slot: math.hypot(
                            match["x"] - self.scene["truth"][slot][0],
                            match["y"] - self.scene["truth"][slot][1]))
            distance = math.hypot(match["x"] - self.scene["truth"][owner][0],
                                  match["y"] - self.scene["truth"][owner][1])
            if distance > self.POSITION_TOLERANCE:
                continue                       # 不在任何英雄身上，别的测试管
            self.assertEqual(
                match["claimedSlot"] <= 5, owner <= 5,
                f"槽位 {match['claimedSlot']} 认下了对方阵营的图标（真身是 {owner}）。")

    def test_no_match_sits_on_empty_ground(self):
        # 报出来的每一个点，要么在某个英雄身上，要么就不该报。
        for match in self.with_pits["matches"]:
            nearest = min(
                math.hypot(match["x"] - x, match["y"] - y)
                for x, y in self.scene["truth"].values())
            self.assertLessEqual(
                nearest, self.POSITION_TOLERANCE,
                f"槽位 {match['claimedSlot']} 落在空地上（离最近的英雄 {nearest:.3f}）。")

    def test_unsure_identity_is_reported_without_a_number(self):
        # 分不清是谁时只报队伍，不写号码 —— 硬安一个号码只会把错的写进数据。
        for match in self.with_pits["matches"]:
            if match["ambiguous"]:
                self.assertIsNone(match["slot"])
                self.assertIn(match["teamId"], (100, 200))
            else:
                self.assertEqual(match["slot"], match["claimedSlot"])

    def test_report_covers_all_ten_slots_with_a_reason(self):
        slots = self.with_pits["slots"]
        self.assertEqual([s["slot"] for s in slots], list(range(1, 11)))
        for item in slots:
            self.assertIn("statusText", item)
            self.assertTrue(item["statusText"], f"槽位 {item['slot']} 没有说明卡在哪一步。")

    def test_uncalibrated_pits_are_called_out(self):
        thresholds = minimap.default_thresholds()
        thresholds["verified"] = True
        without = minimap.detect_with_portraits(
            self.scene["image"], MINIMAP_BOX, self.scene["portraitBoxes"], thresholds)
        self.assertTrue(
            any("资源坑" in problem for problem in without["quality"]["problems"]),
            "没标资源坑时必须明说龙可能被认成英雄，不能默默地给出可能错的结果。")
        self.assertEqual(without["staticZoneCount"], 0)
        self.assertGreaterEqual(self.with_pits["staticZoneCount"], 1)


class TeamColourIsAHardRequirement(unittest.TestCase):
    """用户实测里出现「蓝方槽位认下红方图标」—— 最基础的一条约束被当成了加分项。"""

    def test_default_ratio_is_a_real_threshold(self):
        self.assertGreaterEqual(minimap.DEFAULT_MIN_MARKER_RATIO, 0.03,
                                "队伍色门槛低到形同虚设，空地和对方图标都会被放进来。")

    @staticmethod
    def _canvas(size, team_label_colour, hero_colour):
        raw = bytearray()
        centre = (size - 1) / 2
        for y in range(size):
            for x in range(size):
                distance = math.hypot(x - centre, y - centre)
                if distance > size * 0.5:
                    raw.extend((26, 34, 28))            # 地图底色
                elif distance > size * 0.40:
                    raw.extend(team_label_colour)       # 队伍色外圈
                else:
                    raw.extend(hero_colour)             # 英雄原画
        return bytes(raw)

    def test_ring_is_read_not_the_whole_square(self):
        # 一个**红方**图标，中间那张脸偏蓝。整块里的蓝像素能压过外圈的红，
        # 只有盯着外圈看才判得对 —— 用户看到的红蓝混合就是这么来的。
        size = 40
        raw = self._canvas(size, (225, 70, 70), (60, 120, 235))
        labels = minimap.classify_pixels(raw, size, size, minimap.default_thresholds())
        blue = sum(1 for label in labels if label == 1)
        red = sum(1 for label in labels if label == 2)
        self.assertGreater(blue, red, "这个测试样本本身就该是「整块蓝多于红」，否则测不到东西。")

        detail = minimap.match_portraits_detailed(
            raw, size, size, {}, labels)      # 没有模板，只验证工具本身
        self.assertEqual(detail["matches"], [])

    def test_oversize_regions_are_marked_not_erased(self):
        # 「这里没有队伍色」和「这里被一片特效糊住了」必须区分开：
        # 前者说明没人，后者只说明这一帧的颜色证据用不了。
        width = height = 20
        labels = [0] * (width * height)
        for index in range(width * height):        # 整片都是蓝 —— 远超上限
            labels[index] = 1
        mask, stats = minimap.marker_mask(labels, width, height, 5, 50)
        self.assertEqual(stats["tooLarge"], 1)
        self.assertTrue(all(value == minimap.OVERSIZE for value in mask),
                        "超大团块被抹成 0 了 —— 那样「被特效盖住」会和「空地」混为一谈。")

    def test_small_specks_are_erased(self):
        width = height = 20
        labels = [0] * (width * height)
        labels[5 * width + 5] = 1                  # 一个孤零零的像素
        mask, stats = minimap.marker_mask(labels, width, height, 5, 50)
        self.assertEqual(stats["tooSmall"], 1)
        self.assertEqual(set(mask), {0})


class UnsureIdentityIsNotGuessed(unittest.TestCase):
    """两个人长得一样时，硬安一个号码就是把错的写进数据。"""

    @staticmethod
    def _render(size, art, ring=(60, 120, 235), background=(26, 34, 28)):
        raw = bytearray()
        centre = (size - 1) / 2
        radius = size * 0.46
        for y in range(size):
            for x in range(size):
                distance = math.hypot(x - centre, y - centre)
                if distance > radius:
                    raw.extend(background)
                elif distance > radius - max(2.0, size * 0.10):
                    raw.extend(ring)
                else:
                    raw.extend(art(x - centre, y - centre, radius))
        return bytes(raw)

    def _scene_with_one_icon(self, size=64, icon=26):
        canvas = bytearray(bytes((26, 34, 28)) * (size * size))
        art = _hero_art(2)
        icon_bytes = self._render(icon, art)
        top = left = (size - icon) // 2
        for row in range(icon):
            start = ((top + row) * size + left) * 3
            canvas[start:start + icon * 3] = icon_bytes[row * icon * 3:(row + 1) * icon * 3]
        return bytes(canvas), self._render(24, art)

    def test_two_identical_templates_get_a_position_but_no_number(self):
        canvas, template = self._scene_with_one_icon()
        size = 64
        labels = minimap.classify_pixels(canvas, size, size, minimap.default_thresholds())
        mask, _ = minimap.marker_mask(labels, size, size, 12, 900)
        detail = minimap.match_portraits_detailed(
            canvas, size, size, {1: template, 2: template}, mask,
            min_similarity=0.55, min_marker_ratio=0.03)
        self.assertEqual(len(detail["matches"]), 1,
                         "同一个图标不能同时算成两个人。")
        match = detail["matches"][0]
        self.assertTrue(match["ambiguous"],
                        "两个模板一模一样，不可能分得出是谁。")
        self.assertIsNone(match["slot"],
                          "分不清是谁却写上了号码 —— 这就是用户看到的「头像错认」。")
        self.assertEqual(match["teamId"], 100, "队伍还是知道的，位置也是真的。")

    def test_a_distinctive_template_still_gets_its_number(self):
        # 反面：模板确实不同的时候，必须照常给出号码，不能一律推说「分不清」。
        canvas, template = self._scene_with_one_icon()
        other = self._render(24, _hero_art(7))
        size = 64
        labels = minimap.classify_pixels(canvas, size, size, minimap.default_thresholds())
        mask, _ = minimap.marker_mask(labels, size, size, 12, 900)
        detail = minimap.match_portraits_detailed(
            canvas, size, size, {1: template, 2: other}, mask,
            min_similarity=0.55, min_marker_ratio=0.03)
        named = [m for m in detail["matches"] if m["slot"] is not None]
        self.assertTrue(named, "模板明显不同却还是不肯给号码，就成了另一种失灵。")
        self.assertEqual(named[0]["slot"], 1)


class AssignmentIsExactNotGreedy(unittest.TestCase):
    """贪心分配会连环出错：被抢走图标的槽位只好去认第二像的东西。"""

    def test_matches_brute_force_optimum(self):
        import itertools
        import random
        rnd = random.Random(20260909)
        for _ in range(120):
            slots = [1, 2, 3]
            columns = list(range(5))
            scores = {(slot, column): round(rnd.random(), 3)
                      for slot in slots for column in columns if rnd.random() < 0.7}
            chosen = minimap.assign_one_to_one(scores, slots)
            self.assertEqual(len(set(chosen.values())), len(chosen),
                             "同一个图标被分给了两个槽位。")
            total = sum(scores[pair] for pair in chosen.items())
            best = 0.0
            for size in range(len(slots) + 1):
                for slot_group in itertools.combinations(slots, size):
                    for column_group in itertools.permutations(columns, size):
                        pairs = list(zip(slot_group, column_group))
                        if all(pair in scores for pair in pairs):
                            best = max(best, sum(scores[pair] for pair in pairs))
            self.assertAlmostEqual(total, best, places=9)

    def test_greedy_would_lose_here(self):
        # 贪心会先把 c0 给分数最高的 1 号，2 号只剩 0.10；
        # 最优解是 1 号让出 c0，总分 0.90+0.85 > 0.95+0.10。
        scores = {(1, "c0"): 0.95, (1, "c1"): 0.90, (2, "c0"): 0.85, (2, "c1"): 0.10}
        chosen = minimap.assign_one_to_one(scores, [1, 2])
        self.assertEqual(chosen, {1: "c1", 2: "c0"})


if __name__ == "__main__":
    unittest.main()
