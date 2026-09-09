"""小地图 → 十个英雄的位置。这是路线图里标的最大技术缺口。

## 为什么它值得单独一个模块

记分板上的经济和比分是**文字**，OCR 能读。
小地图上的英雄是**彩色圆点**，OCR 完全读不了，需要另一条技术路线。

而位置信息不是锦上添花。LOL 项目的路线图里写着：

> 外部模型只吃粗特征，**不知道敌方打野在哪**，
> 而这正是「入侵值不值」所必需的。

没有位置，这个项目想回答的那类问题就一个都答不了。

## 这一版做什么、不做什么

第一阶段只用颜色找蓝红色块；真实 KPL 帧证明这种办法会把道路、塔和装饰当人。
当前改为两阶段：颜色只生成弱候选，随后把左右十个已标定头像逐一与小地图
多尺度圆形候选比较。相似度、圆形边界或人工核验不过关就不输出，宁可缺人。

输出的是 1~10 号槽位，不根据上下位置硬猜，也不凭图像编造英雄名称。

## 怎么在不装任何 Python 包的前提下处理图像

标准库没有 JPEG 解码器，而这个项目不许用 pip 装东西。
解法：让 ffmpeg 把图像区域直接吐成**原始 RGB 字节流**，
Python 那边就只是读 bytes 了，一个包都不用装。

    ffmpeg -i 帧.jpg -vf "crop=...,scale=..." -pix_fmt rgb24 -f rawvideo -

ffmpeg 本来就是抽帧要用的，不引入新东西。

## 阈值全部是待标定的

判断「多蓝算蓝方」的那几个数字，和 rules.py 里的游戏常量一样，
**是凭经验写的起点，不是查证过的事实**。不同录像源的色调差别很大
（虎牙和官方的画质、编码、色彩空间都不一样）。
所以阈值可以被 data/kpl/minimap_profile.json 覆盖，
并且在没有被人核对过之前，产出的置信度会被压低。
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from . import schema, video

# 分析时把小地图缩放到这个边长。
# 取 160 是权衡：太小则两个贴在一起的英雄会粘成一团，
# 太大则纯 Python 的连通域标记会慢（160×160 = 25600 像素，尚可）。
ANALYSIS_SIZE = 160


def default_thresholds() -> dict[str, Any]:
    """判断一个像素属于哪一方的规则。全部是待标定的起点。"""
    return {
        "verified": False,
        "note": "凭经验写的起点，不同录像源色调差别很大，必须用真实画面标定。",
        # 蓝方：蓝色通道明显压过红色
        "blueMinB": 90,
        "blueBOverR": 40,      # B - R 至少这么多
        "blueBOverG": 15,      # B - G 至少这么多
        # 红方：红色通道明显压过蓝色
        "redMinR": 95,
        "redROverB": 40,
        "redROverG": 30,
        # 一个标记至少要有这么多像素才算数，用来滤掉噪点和地图上的装饰
        "minBlobPixels": 12,
        "maxBlobPixels": 900,
    }


def profile_path(data_dir: Path) -> Path:
    return data_dir / "kpl" / "minimap_profile.json"


def load_thresholds(data_dir: Path | None = None) -> dict[str, Any]:
    table = default_thresholds()
    if data_dir is None:
        return table
    path = profile_path(data_dir)
    if not path.is_file():
        return table
    try:
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return table
    if isinstance(saved, dict):
        for key, value in saved.items():
            if key in table:
                table[key] = value
    return table


def save_thresholds(data_dir: Path, values: dict[str, Any], verified: bool = True) -> Path:
    table = load_thresholds(data_dir)
    for key, value in values.items():
        if key in table and key != "verified":
            table[key] = value
    table["verified"] = bool(verified)
    path = profile_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------- 取像素

def read_rgb(
    image: Path,
    box: tuple[float, float, float, float] | None = None,
    size: int = ANALYSIS_SIZE,
) -> tuple[bytes, int, int]:
    """把图像（的一块区域）读成原始 RGB 字节。返回 (数据, 宽, 高)。

    box 是相对坐标 (x, y, w, h)，和 hud.py 用的是同一套坐标系。
    """
    exe = video.ffmpeg_path()
    if not exe:
        raise video.VideoError(video.INSTALL_HINT)
    if not image.is_file():
        raise video.VideoError(f"找不到图片：{image}")

    filters = []
    if box is not None:
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise video.VideoError(f"区域超出画面范围：{box}")
        filters.append(f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}")
    filters.append(f"scale={size}:{size}")

    result = subprocess.run(
        [exe, "-nostdin", "-loglevel", "error", "-i", str(image),
         "-vf", ",".join(filters), "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"],
        capture_output=True, timeout=120, check=False,
    )
    if result.returncode != 0 or not result.stdout:
        raise video.VideoError(
            f"读取像素失败：{result.stderr.decode('utf-8', 'ignore').strip()[:200]}"
        )
    expected = size * size * 3
    if len(result.stdout) < expected:
        raise video.VideoError(
            f"像素数据不完整：拿到 {len(result.stdout)} 字节，期望 {expected}"
        )
    return result.stdout[:expected], size, size


# ---------------------------------------------------------------- 分类与聚类

def classify_pixels(
    data: bytes, width: int, height: int, thresholds: dict[str, Any]
) -> list[int]:
    """把每个像素标成 0=背景 / 1=蓝方 / 2=红方。"""
    b_min = int(thresholds["blueMinB"])
    b_over_r = int(thresholds["blueBOverR"])
    b_over_g = int(thresholds["blueBOverG"])
    r_min = int(thresholds["redMinR"])
    r_over_b = int(thresholds["redROverB"])
    r_over_g = int(thresholds["redROverG"])

    labels = [0] * (width * height)
    for index in range(width * height):
        base = index * 3
        red, green, blue = data[base], data[base + 1], data[base + 2]
        if blue >= b_min and blue - red >= b_over_r and blue - green >= b_over_g:
            labels[index] = 1
        elif red >= r_min and red - blue >= r_over_b and red - green >= r_over_g:
            labels[index] = 2
    return labels


def find_blobs(
    labels: list[int], width: int, height: int, team: int,
    min_pixels: int, max_pixels: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """连通域标记：把相邻的同色像素聚成一个个标记。

    返回 (合格的标记, 被丢弃的统计)。

    **被丢弃的必须报出来，不能默默吞掉。**
    这是测试时发现的一个真 bug：阈值调松之后，整个小地图底色连成了
    一个巨大的蓝色区域，它因为超过尺寸上限被丢弃，于是「蓝方 0 个标记」，
    而 assess() 看到红方有 5 个、没有超过 5 个的违规，就判定「可用」。

    一个超大团块被丢弃，恰恰是阈值错得离谱的最强信号 ——
    把它当成「什么都没找到」是把最有价值的诊断信息扔了。

    纯 Python 的四邻域 BFS。25600 个像素这个规模完全够用，
    不值得为它引入 numpy —— 那会破坏「不用装任何东西」这条底线。
    """
    seen = [False] * (width * height)
    blobs: list[dict[str, Any]] = []
    rejected = {"tooSmall": 0, "tooLarge": 0, "largestRejectedPixels": 0}

    for start in range(width * height):
        if seen[start] or labels[start] != team:
            continue
        queue = deque([start])
        seen[start] = True
        pixels: list[int] = []
        while queue:
            index = queue.popleft()
            pixels.append(index)
            row, col = divmod(index, width)
            for nrow, ncol in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= nrow < height and 0 <= ncol < width):
                    continue
                neighbour = nrow * width + ncol
                if not seen[neighbour] and labels[neighbour] == team:
                    seen[neighbour] = True
                    queue.append(neighbour)

        if len(pixels) < min_pixels:
            rejected["tooSmall"] += 1
            continue
        if len(pixels) > max_pixels:
            rejected["tooLarge"] += 1
            rejected["largestRejectedPixels"] = max(
                rejected["largestRejectedPixels"], len(pixels))
            continue
        rows = [p // width for p in pixels]
        cols = [p % width for p in pixels]
        blobs.append({
            "teamId": 100 if team == 1 else 200,
            "pixels": len(pixels),
            "x": round(sum(cols) / len(cols) / width, 4),
            "y": round(sum(rows) / len(rows) / height, 4),
        })

    blobs.sort(key=lambda b: -b["pixels"])
    return blobs, rejected


# ---------------------------------------------------------------- 对外接口

def detect(
    image: Path,
    box: tuple[float, float, float, float] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在一张画面上找出小地图标记。

    兼容的颜色候选接口。正式标定页在有十个侧边头像时使用
    ``detect_with_portraits()``，本函数不负责槽位关联。
    """
    thresholds = thresholds or default_thresholds()
    data, width, height = read_rgb(image, box)
    labels = classify_pixels(data, width, height, thresholds)

    min_px = int(thresholds["minBlobPixels"])
    max_px = int(thresholds["maxBlobPixels"])
    blue, blue_rejected = find_blobs(labels, width, height, 1, min_px, max_px)
    red, red_rejected = find_blobs(labels, width, height, 2, min_px, max_px)

    total_pixels = width * height
    coloured = sum(1 for label in labels if label)

    return {
        "blue": blue,
        "red": red,
        "blueCount": len(blue),
        "redCount": len(red),
        "rejected": {"blue": blue_rejected, "red": red_rejected},
        "colouredRatio": round(coloured / total_pixels, 4),
        "thresholdsVerified": bool(thresholds.get("verified")),
        "quality": assess(
            len(blue), len(red), bool(thresholds.get("verified")),
            blue_rejected, red_rejected, coloured / total_pixels,
        ),
    }


# ---------------------------------------------------------------- 头像模板关联

def appearance_descriptor(data: bytes, width: int, height: int,
                          grid: int = 4) -> list[float]:
    """把头像压成分区色彩描述子，用于跨尺寸比较。

    每格保存归一化 RGB 与亮度。归一化色彩降低了转播压缩和明暗变化的影响，
    分区又保留了发色、肤色、盔甲等大致空间关系。它不是人脸识别模型；低于门槛
    的匹配会被拒绝，不能为了凑齐十个人而硬分配。
    """
    if not data or width <= 0 or height <= 0:
        return []
    out: list[float] = []
    for gy in range(grid):
        y0, y1 = gy * height // grid, max(gy * height // grid + 1, (gy + 1) * height // grid)
        for gx in range(grid):
            x0, x1 = gx * width // grid, max(gx * width // grid + 1, (gx + 1) * width // grid)
            tr = tg = tb = count = 0
            for y in range(y0, min(height, y1)):
                for x in range(x0, min(width, x1)):
                    base = (y * width + x) * 3
                    tr += data[base]
                    tg += data[base + 1]
                    tb += data[base + 2]
                    count += 1
            if not count:
                out.extend((0.0, 0.0, 0.0, 0.0))
                continue
            r, g, b = tr / count, tg / count, tb / count
            total = max(1.0, r + g + b)
            out.extend((r / total, g / total, b / total, max(r, g, b) / 255.0))
    return out


def descriptor_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    distance = (sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)) ** 0.5
    return round(max(0.0, 1.0 - distance / 0.42), 4)


def structure_descriptor(data: bytes, width: int, height: int,
                         grid: int = 8) -> list[float]:
    """归一化灰度结构，区分英雄脸部轮廓与普通地图纹理。"""
    if not data or width <= 0 or height <= 0:
        return []
    values: list[float] = []
    for gy in range(grid):
        y0, y1 = gy * height // grid, max(gy * height // grid + 1, (gy + 1) * height // grid)
        for gx in range(grid):
            x0, x1 = gx * width // grid, max(gx * width // grid + 1, (gx + 1) * width // grid)
            total = count = 0.0
            for y in range(y0, min(height, y1)):
                for x in range(x0, min(width, x1)):
                    base = (y * width + x) * 3
                    total += data[base] * 0.299 + data[base + 1] * 0.587 + data[base + 2] * 0.114
                    count += 1
            values.append(total / max(1.0, count))
    mean = sum(values) / len(values)
    deviation = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if deviation < 1.0:
        return [0.0] * len(values)
    return [(v - mean) / deviation for v in values]


def structure_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    correlation = sum(a * b for a, b in zip(left, right)) / len(left)
    return round(max(0.0, min(1.0, (correlation + 1.0) / 2.0)), 4)


def _crop_bytes(data: bytes, width: int, height: int,
                cx: int, cy: int, size: int) -> tuple[bytes, int, int]:
    half = max(2, size // 2)
    x0, x1 = max(0, cx - half), min(width, cx + half)
    y0, y1 = max(0, cy - half), min(height, cy + half)
    out = bytearray()
    for y in range(y0, y1):
        start = (y * width + x0) * 3
        out.extend(data[start:start + (x1 - x0) * 3])
    return bytes(out), x1 - x0, y1 - y0


def radial_edge_score(data: bytes, width: int, height: int,
                      cx: int, cy: int, size: int) -> float:
    """头像是有闭合边框的小圆标；道路/草丛通常没有一整圈径向边缘。"""
    inner = max(2.0, size * 0.27)
    outer = max(inner + 1.0, size * 0.50)
    distances: list[float] = []
    for index in range(20):
        angle = 2.0 * math.pi * index / 20.0
        points = []
        for radius in (inner, outer):
            x = min(width - 1, max(0, round(cx + math.cos(angle) * radius)))
            y = min(height - 1, max(0, round(cy + math.sin(angle) * radius)))
            base = (y * width + x) * 3
            points.append((data[base], data[base + 1], data[base + 2]))
        distance = sum((points[0][i] - points[1][i]) ** 2 for i in range(3)) ** 0.5 / 441.67
        distances.append(distance)
    strong = sum(1 for d in distances if d >= 0.08) / len(distances)
    return round((sum(distances) / len(distances)) * 0.55 + strong * 0.45, 4)


def match_portraits(
    map_data: bytes,
    width: int,
    height: int,
    portrait_templates: dict[int, bytes],
    labels: list[int] | None = None,
    min_similarity: float = 0.80,
) -> list[dict[str, Any]]:
    """逐个把两侧英雄头像与小地图候选图标比较。

    候选点来自整张小地图的多尺度滑窗，不再把红蓝色块中心直接当英雄位置。
    队伍颜色只作为弱佐证；最终必须通过对应槽位头像的外观匹配。
    """
    if not portrait_templates:
        return []
    template_desc = {
        int(slot): (appearance_descriptor(raw, 24, 24),
                    structure_descriptor(raw, 24, 24))
        for slot, raw in portrait_templates.items()
        if raw and len(raw) >= 24 * 24 * 3
    }
    if not template_desc:
        return []

    candidates: list[dict[str, Any]] = []
    step = max(3, width // 40)
    # KPL 720p 转播的小地图头像缩到 160×160 后通常仍有约 17~29px。
    # 旧版只搜到 10~17px，实际等于在头像内部找一小块，极易匹配到地图纹理。
    sizes = sorted({max(14, width // 9), max(18, width // 8),
                    max(22, width // 6), max(26, round(width / 5.5))})
    for size in sizes:
        margin = size // 2
        for cy in range(margin, height - margin, step):
            for cx in range(margin, width - margin, step):
                crop, cw, ch = _crop_bytes(map_data, width, height, cx, cy, size)
                desc = appearance_descriptor(crop, cw, ch)
                # 平坦地图底纹不可能是头像；先用描述子的亮度分区变化做廉价筛选。
                values = desc[3::4]
                texture = max(values, default=0.0) - min(values, default=0.0)
                if texture < 0.055:
                    continue
                circle = radial_edge_score(map_data, width, height, cx, cy, size)
                if circle < 0.42:
                    continue
                candidates.append({"cx": cx, "cy": cy, "size": size,
                                   "desc": desc,
                                   "structure": structure_descriptor(crop, cw, ch),
                                   "texture": texture, "circle": circle})

    pairs: list[dict[str, Any]] = []
    for slot, wanted_pair in template_desc.items():
        wanted, wanted_structure = wanted_pair
        team_label = 1 if slot <= 5 else 2
        best_for_slot: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            colour_similarity = descriptor_similarity(wanted, candidate["desc"])
            shape_similarity = structure_similarity(wanted_structure, candidate["structure"])
            similarity = 0.35 * colour_similarity + 0.65 * shape_similarity
            if similarity < min_similarity or candidate["circle"] < 0.48:
                continue
            # 候选框内确实有本方颜色时最多加 0.06；没有也不直接拒绝，避免头像边框
            # 被压缩或技能特效改变颜色后整个人消失。
            team_ratio = 0.0
            marker_ratio = 0.0
            if labels:
                cx, cy, size = candidate["cx"], candidate["cy"], candidate["size"]
                half = size // 2
                hit = any_hit = total = 0
                for y in range(max(0, cy-half), min(height, cy+half), 2):
                    for x in range(max(0, cx-half), min(width, cx+half), 2):
                        total += 1
                        hit += labels[y * width + x] == team_label
                        any_hit += labels[y * width + x] in (1, 2)
                team_ratio = hit / max(1, total)
                marker_ratio = any_hit / max(1, total)
            # 真实 KPL 帧里阵营边框会被英雄原画、选中框和转播调色覆盖，不能把
            # 红蓝阈值设成硬门槛。它只作很小的同队加分，主体仍是头像结构比较。
            if labels is not None and marker_ratio < 0.012:
                continue
            score = min(1.0, similarity + min(0.025, team_ratio * 0.15)
                        + min(0.04, candidate["circle"] * 0.12))
            if score >= min_similarity:
                best_for_slot.append({"slot": slot, "candidate": index,
                                      "score": score, "similarity": similarity,
                                      "shapeSimilarity": shape_similarity,
                                      "colourSimilarity": colour_similarity,
                                      "circleScore": candidate["circle"],
                                      "markerColourRatio": marker_ratio,
                                      "teamColourRatio": team_ratio})
        pairs.extend(sorted(best_for_slot, key=lambda p: -p["score"])[:6])

    # 全局贪心一对一分配。同一个小地图头像不能同时冒充两名选手。
    used_slots: set[int] = set()
    used_candidates: list[int] = []
    matches: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda p: -p["score"]):
        if pair["slot"] in used_slots:
            continue
        candidate = candidates[pair["candidate"]]
        if any((candidate["cx"] - candidates[i]["cx"]) ** 2
               + (candidate["cy"] - candidates[i]["cy"]) ** 2
               < (min(candidate["size"], candidates[i]["size"]) * 0.65) ** 2
               for i in used_candidates):
            continue
        used_slots.add(pair["slot"])
        used_candidates.append(pair["candidate"])
        matches.append({
            "slot": pair["slot"],
            "teamId": 100 if pair["slot"] <= 5 else 200,
            "x": round(candidate["cx"] / width, 4),
            "y": round(candidate["cy"] / height, 4),
            "confidence": round(pair["score"], 3),
            "appearanceSimilarity": round(pair["similarity"], 3),
            "shapeSimilarity": round(pair["shapeSimilarity"], 3),
            "colourSimilarity": round(pair["colourSimilarity"], 3),
            "circleScore": round(pair["circleScore"], 3),
            "markerColourRatio": round(pair["markerColourRatio"], 3),
            "teamColourRatio": round(pair["teamColourRatio"], 3),
            "method": "portrait-template",
        })
    return sorted(matches, key=lambda m: m["slot"])


def detect_with_portraits(
    image: Path,
    box: tuple[float, float, float, float],
    portrait_boxes: dict[int, tuple[float, float, float, float]],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """颜色只提团队先验；位置与槽位由两侧头像逐一模板匹配。"""
    thresholds = thresholds or default_thresholds()
    data, width, height = read_rgb(image, box)
    labels = classify_pixels(data, width, height, thresholds)
    templates: dict[int, bytes] = {}
    for slot, portrait_box in portrait_boxes.items():
        try:
            raw, _, _ = read_rgb(image, portrait_box, size=24)
        except Exception:  # 单个头像模板失败不能毁掉其他九个
            continue
        templates[int(slot)] = raw
    matches = match_portraits(data, width, height, templates, labels)
    blue = [m for m in matches if m["teamId"] == 100]
    red = [m for m in matches if m["teamId"] == 200]
    problems: list[str] = []
    if len(templates) < 10:
        problems.append(f"只标定了 {len(templates)}/10 个两侧英雄头像模板。")
    if len(matches) < 6:
        problems.append(f"只有 {len(matches)} 个头像通过逐一匹配门槛；其余拒绝输出，避免假点。")
    confidence = (sum(m["confidence"] for m in matches) / len(matches)) if matches else 0.0
    if not thresholds.get("verified"):
        confidence = min(confidence, 0.45)
        problems.append("阵营颜色阈值尚未人工核验，整体置信度已封顶为 0.45。")
    return {
        "blue": blue, "red": red, "blueCount": len(blue), "redCount": len(red),
        "matches": matches, "portraitTemplateCount": len(templates),
        "thresholdsVerified": bool(thresholds.get("verified")),
        "quality": {"usable": bool(matches), "confidence": round(confidence, 3),
                    "problems": problems},
        "method": "portrait-template",
    }


def assess(
    blue_count: int, red_count: int, verified: bool,
    blue_rejected: dict[str, int] | None = None,
    red_rejected: dict[str, int] | None = None,
    coloured_ratio: float = 0.0,
) -> dict[str, Any]:
    """这次识别有多可信。

    每方最多 5 个英雄，这是硬约束。找出 7 个蓝色标记说明阈值把地图上
    别的东西也算进来了 —— 这种情况必须把置信度打到 0，
    而不是挑最大的 5 个交差。**挑 5 个交差是在编造。**

    找出的比 5 少是正常的：有人在草丛里、在阴影区、或者死了没显示。
    少不代表错，只代表这一帧的信息不全。

    但「一方一个都没有、另一方有好几个」不正常 —— 十有八九是阈值坏了，
    而不是那一队五个人同时消失。这一条是测试时发现漏judgment后补的。
    """
    blue_rejected = blue_rejected or {}
    red_rejected = red_rejected or {}
    problems: list[str] = []
    confidence = 0.55 if verified else 0.3
    fatal = False

    # ① 超过每方 5 人：阈值一定错了
    if blue_count > 5 or red_count > 5:
        problems.append(
            f"找到蓝 {blue_count}、红 {red_count} 个标记，超过每方 5 个的上限。"
            "阈值把地图上别的东西也算进来了，这一帧的位置全部不可用。"
        )
        fatal = True

    # ② 有超大团块被丢弃：底色被当成了队伍颜色
    oversize = blue_rejected.get("tooLarge", 0) + red_rejected.get("tooLarge", 0)
    if oversize:
        largest = max(blue_rejected.get("largestRejectedPixels", 0),
                      red_rejected.get("largestRejectedPixels", 0))
        problems.append(
            f"有 {oversize} 个超大色块被丢弃（最大 {largest} 像素）。"
            "这说明阈值把地图底色也判成了队伍颜色，识别结果不可信。"
        )
        fatal = True

    # ③ 染色面积过大：同上，另一个角度的同一个信号
    if coloured_ratio > 0.35:
        problems.append(
            f"小地图上有 {coloured_ratio * 100:.0f}% 的像素被判成了队伍颜色，"
            "远超正常范围（十个小圆点通常不到 5%）。阈值太松。"
        )
        fatal = True

    # ④ 一方全无、另一方好几个：不正常的不对称
    if not fatal and blue_count == 0 and red_count >= 3:
        problems.append(
            f"蓝方一个标记都没找到，红方却有 {red_count} 个。"
            "五个人同时看不见的可能性很低，更像是蓝色阈值坏了。"
        )
        fatal = True
    if not fatal and red_count == 0 and blue_count >= 3:
        problems.append(
            f"红方一个标记都没找到，蓝方却有 {blue_count} 个。"
            "五个人同时看不见的可能性很低，更像是红色阈值坏了。"
        )
        fatal = True

    # ⑤ 什么都没找到
    if not fatal and blue_count == 0 and red_count == 0:
        problems.append("一个标记都没找到。可能是小地图区域框错了，或者阈值太严。")
        fatal = True

    if fatal:
        confidence = 0.0
    else:
        found = blue_count + red_count
        confidence *= 0.6 + 0.4 * (found / 10.0)
        if found < 4:
            problems.append(f"只找到 {found} 个标记，偏少。位置信息很不完整。")

    if not verified:
        problems.append(
            "颜色阈值还没有用真实画面标定过，这个结果只能当参考。"
            "标定办法：kplab minimap --calibrate"
        )

    return {
        "confidence": round(min(1.0, max(0.0, confidence)), 3),
        "usable": not fatal,
        "problems": problems,
    }


def to_observation_players(result: dict[str, Any]) -> dict[str, Any]:
    """把识别结果转成观测记录里的 players 结构。

    **关键的一步，也是最容易做错的一步。**

    新结果已经通过侧边头像关联槽位，直接保留槽位但在真实样本人工验收前封顶。
    旧颜色结果仍按位置临时排序并砍半，兼容已有数据但不会冒充可信事实。
    """
    confidence = float((result.get("quality") or {}).get("confidence", 0.0))
    if not result["quality"]["usable"]:
        return {}

    players: dict[str, Any] = {}
    if result.get("method") == "portrait-template":
        for match in result.get("matches", []):
            slot = match.get("slot")
            if not isinstance(slot, int) or not 1 <= slot <= 10:
                continue
            # 已经逐头像关联，不再砍半；但轻量模板法在真实样本人工核验前封顶 0.68。
            confidence = min(0.68, float(match.get("confidence", 0.0)))
            players[str(slot)] = {
                "x": schema.field(match["x"], schema.SOURCE_CV, confidence),
                "y": schema.field(match["y"], schema.SOURCE_CV, confidence),
            }
        return players

    for team_key, slots in (("blue", range(1, 6)), ("red", range(6, 11))):
        blobs = sorted(result[team_key], key=lambda b: (b["y"], b["x"]))
        for slot, blob in zip(slots, blobs):
            players[str(slot)] = {
                # 置信度砍半：位置本身可能是对的，但「这是几号位」是猜的
                "x": schema.field(blob["x"], schema.SOURCE_CV, confidence * 0.5),
                "y": schema.field(blob["y"], schema.SOURCE_CV, confidence * 0.5),
            }
    return players


def summary(result: dict[str, Any]) -> str:
    lines = [
        f"蓝方标记 {result['blueCount']} 个    红方标记 {result['redCount']} 个",
        f"置信度 {result['quality']['confidence']}    "
        f"{'可用' if result['quality']['usable'] else '不可用'}",
    ]
    for problem in result["quality"]["problems"]:
        lines.append(f"  ⚠️  {problem}")
    if result["blue"] or result["red"]:
        lines.append("")
        lines.append("  标记位置（相对小地图，左上 0,0）：")
        for blob in result["blue"]:
            lines.append(f"    蓝  ({blob['x']:.3f}, {blob['y']:.3f})   {blob['pixels']} 像素")
        for blob in result["red"]:
            lines.append(f"    红  ({blob['x']:.3f}, {blob['y']:.3f})   {blob['pixels']} 像素")
    lines.append("")
    lines.append("  有头像模板时按 1~10 号槽位关联；低相似度候选会被拒绝，不凑人数。")
    return "\n".join(lines)
