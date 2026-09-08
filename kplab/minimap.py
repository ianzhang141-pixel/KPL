"""小地图 → 十个英雄的位置。这是路线图里标的最大技术缺口。

## 为什么它值得单独一个模块

记分板上的经济和比分是**文字**，OCR 能读。
小地图上的英雄是**彩色圆点**，OCR 完全读不了，需要另一条技术路线。

而位置信息不是锦上添花。LOL 项目的路线图里写着：

> 外部模型只吃粗特征，**不知道敌方打野在哪**，
> 而这正是「入侵值不值」所必需的。

没有位置，这个项目想回答的那类问题就一个都答不了。

## 这一版做什么、不做什么

**做：** 认出小地图上有几个蓝色标记、几个红色标记，各自在哪。
**不做：** 认出那个蓝点具体是哪个英雄。

这个区分是刻意的，也是诚实的。
区分英雄需要头像模板库、要处理遮挡和缩放，是另一个量级的工作。
而「蓝方有 3 个人在上半区」这种信息，光靠颜色和位置就能得出，
已经能回答一部分问题了 —— 先把能做的做扎实。

**所以本模块永远不会给出 heroName。** 它只说「这里有一个蓝方标记」。

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

    返回结果里**没有 heroName** —— 本模块认不出是哪个英雄，
    也不会假装认得出。它只说「这里有一个蓝方标记」。
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

    识别出来的是「一个蓝方标记在 (0.3, 0.4)」，
    但观测结构要求的是「1 号位在 (0.3, 0.4)」——
    中间隔着「哪个标记是哪个人」，而本模块**答不了这个问题**。

    所以这里按位置排序后依次填进该方的槽位，并把置信度压到很低。
    这是一个**明确标注为不可靠的猜测**，不是识别结果。
    下游看到 0.2 的置信度就知道不能当事实用。

    等以后做出英雄区分，这个函数才会真正有意义。在那之前，
    位置信息的正经来源仍然是人工标注。
    """
    confidence = result["quality"]["confidence"]
    if not result["quality"]["usable"]:
        return {}

    players: dict[str, Any] = {}
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
    lines.append("  注意：本模块认不出是哪个英雄，只知道「这里有一个某方的标记」。")
    return "\n".join(lines)
