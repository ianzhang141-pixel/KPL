"""从屏幕上方的**播报**里读出「是谁拿的」。

## 这一步补的是哪个洞

`events_cv.py` 能从图标消失判断出「暴君死了」「塔被拆了」，
但**说不出是谁拿的** —— 图标消失本身不含这个信息。
上一版对此的处理是老实返回「不知道」。

用户指出了补法：**推塔、击杀、开龙都会在屏幕中间上方弹一条播报，而且会停留几秒。**

所以流程变成：

    粗扫（每 30~60 秒一帧）
      → 发现「t=300 那帧图标没了，t=240 那帧还在」
      → 事件必定发生在 (240, 300] 这个区间里
      → **只在这 60 秒里精抽帧**（每 1~2 秒一张）
      → 在这些帧的播报区找那条横幅
      → 从横幅判断归属

关键在于**只精抽这一小段**。整场 20 分钟按 1 秒抽是 1200 帧，
但每个事件只需要在 60 秒的窗口里抽 30~40 帧，
一场十几个事件也就几百帧，代价完全可以接受。

## 归属判到什么粒度

横幅上有文字（谁击杀了谁），但读文字又回到 OCR 那条不可靠的路上了。
**这里只判队伍**：横幅的配色是偏蓝还是偏红。

判队伍就够了 —— State 里的中立资源计数器、推塔数都是按队伍记的。
「具体是哪个英雄拿的」是更细的粒度，本模块不做，也不假装做得了。

## 三个必须诚实处理的情况

1. **区间里没找到播报** → 仍然是「不知道」。
   可能是播报区框歪了、可能是精抽的间隔跨过了播报、可能那一版 UI 不一样。

2. **一个区间里有多个事件**（团战：连续三个人头 + 一条龙）→
   找到的播报未必属于**这一个**事件。必须降置信度并标出来。

3. **横幅颜色两边都不明显** → 「不知道」。
   宁可漏，不可错 —— 一条错的归属会一路错进胜率曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import events_cv, video
from .rules import BLUE, RED

# 播报区在 HUD 标定里的键名。屏幕中间上方那一条。
BANNER_REGION = "banner"
BANNER_FIELD_REGIONS = {
    "killer": "bannerKiller",
    "target": "bannerTarget",
    "objective": "bannerObjective",
    "assists": "bannerAssists",
}

OBJECTIVE_WORDS = (
    ("风暴龙王", "STORM_DRAGON_KILL"),
    ("暗影暴君", "DARK_TYRANT_KILL"),
    ("暗影主宰", "PROPHET_OVERLORD_KILL"),
    ("暴君", "TYRANT_KILL"),
    ("主宰", "OVERLORD_KILL"),
)

# 播报区和「什么都没有」比，变化超过这个值就认为横幅出现了。
# 凭经验定的起点，需要用真实画面标定。
BANNER_THRESHOLD = 0.10

# 蓝红两个通道差多少才算「明显偏向一方」。
# 定得保守一点：宁可判不出来，也不要判错。
TINT_MARGIN = 12.0


def parse_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """把四个已标定播报子区域整理成结构化佐证。

    OCR/头像识别读不清时字段保持 ``None``。尤其不能把一条很长的团战播报
    强行拆给某个事件；原始文字始终保留，便于人工复核。
    """
    texts = {key: str(raw.get(region) or "").strip()
             for key, region in BANNER_FIELD_REGIONS.items()}
    objective_type = None
    objective_text = texts["objective"]
    for word, event_type in OBJECTIVE_WORDS:
        if word in objective_text:
            objective_type = event_type
            break
    assists = [part.strip() for part in
               texts["assists"].replace("、", ",").replace("/", ",").split(",")
               if part.strip()]
    return {
        "killer": texts["killer"] or None,
        "target": texts["target"] or None,
        "objective": objective_text or None,
        "objectiveType": objective_type,
        "assists": assists,
        "raw": texts,
        "complete": bool(texts["killer"] and (texts["target"] or objective_type)),
    }


def bracket_events(
    events: list[dict[str, Any]],
    frame_times: list[float],
) -> list[dict[str, Any]]:
    """给每个事件框出「它一定发生在这段时间里」的区间。

    粗扫是在离散的帧上做的：t=240 那帧图标还在，t=300 那帧没了，
    所以事件发生在 (240, 300]。区间的长度就是抽帧间隔。
    """
    ordered = sorted(set(float(t) for t in frame_times))
    out: list[dict[str, Any]] = []
    for event in events:
        at = float(event.get("atSec", 0.0))
        before = None
        for t in ordered:
            if t < at:
                before = t
            else:
                break
        out.append({
            **event,
            "bracket": {
                # 没有更早的帧时，只能从 0 开始找
                "fromSec": before if before is not None else max(0.0, at - 60.0),
                "toSec": at,
                "widthSec": round(at - (before if before is not None else max(0.0, at - 60.0)), 2),
                "exact": before is not None,
            },
        })
    return out


def group_by_bracket(bracketed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把落在同一个区间里的事件归成一组。

    团战里可能同时有三个人头 + 一条龙，它们共用一个区间。
    这时候区间里的播报**不能**认定属于其中某一个 ——
    必须知道这一组有几个事件，才能诚实地降置信度。
    """
    buckets: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for event in bracketed:
        key = (event["bracket"]["fromSec"], event["bracket"]["toSec"])
        buckets.setdefault(key, []).append(event)
    return [
        {"fromSec": key[0], "toSec": key[1], "events": items, "count": len(items)}
        for key, items in sorted(buckets.items())
    ]


# ---------------------------------------------------------------- 找横幅

def scan_window(
    source: Path,
    box: tuple[float, float, float, float],
    from_sec: float,
    to_sec: float,
    baseline: dict[str, float],
    step_sec: float = 1.5,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """在一小段录像里精抽帧，找播报横幅。

    **只处理这一段**，不碰整个录像 —— 这是整套方法可行的前提。
    """
    if to_sec <= from_sec:
        return {"found": False, "reason": "区间宽度为 0，没法找。", "samples": []}

    work_dir = work_dir or (source.parent / ".banner_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    at = from_sec
    index = 0
    try:
        while at <= to_sec + 1e-6:
            shot = work_dir / f"b_{int(at * 1000)}.jpg"
            try:
                video.crop_region(source, shot, at, box, scale_width=240)
                signature = events_cv.read_patch(shot, (0.0, 0.0, 1.0, 1.0), size=32)
            except Exception as err:      # noqa: BLE001 - 单帧失败不该毁掉整段
                samples.append({"atSec": round(at, 2), "error": str(err)[:120]})
                at += step_sec
                index += 1
                continue
            finally:
                shot.unlink(missing_ok=True)

            distance = events_cv.signature_distance(baseline, signature)
            samples.append({
                "atSec": round(at, 2),
                "distance": round(distance, 4),
                "present": distance > BANNER_THRESHOLD,
                "signature": {k: round(v, 2) for k, v in signature.items()},
            })
            at += step_sec
            index += 1
    finally:
        try:
            work_dir.rmdir()
        except OSError:
            pass

    hits = [s for s in samples if s.get("present")]
    if not hits:
        return {
            "found": False,
            "reason": "这段区间里没找到播报横幅。可能是播报区框歪了、"
                      "精抽间隔跨过了横幅、或者这一版 UI 的播报不在这个位置。",
            "samples": samples,
        }
    return {"found": True, "hits": hits, "samples": samples,
            "spanSec": round(hits[-1]["atSec"] - hits[0]["atSec"], 2)}


def read_tint(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """看横幅整体偏蓝还是偏红。

    只判队伍，不读文字 —— 读文字要回到 OCR，那条路没这个可靠。
    两边都不明显时返回「不知道」，**不猜**。
    """
    if not hits:
        return {"teamId": None, "reason": "没有横幅样本。"}

    blues = [h["signature"]["b"] for h in hits if "signature" in h]
    reds = [h["signature"]["r"] for h in hits if "signature" in h]
    if not blues or not reds:
        return {"teamId": None, "reason": "样本里没有可用的颜色数据。"}

    blue = sum(blues) / len(blues)
    red = sum(reds) / len(reds)
    margin = abs(blue - red)

    if margin < TINT_MARGIN:
        return {"teamId": None, "blue": round(blue, 1), "red": round(red, 1),
                "margin": round(margin, 1),
                "reason": f"蓝红两个通道只差 {margin:.1f}（需要 {TINT_MARGIN}），"
                          "颜色上分不出偏向哪一方。"}

    return {
        "teamId": BLUE if blue > red else RED,
        "blue": round(blue, 1), "red": round(red, 1), "margin": round(margin, 1),
        # 差得越明显越可信，但封顶 0.75 —— 颜色本身就不是铁证
        "strength": round(min(0.75, 0.35 + margin / 120.0), 3),
    }


# ---------------------------------------------------------------- 主流程

def refine(
    source: Path,
    scan_result: dict[str, Any],
    frame_times: list[float],
    banner_box: tuple[float, float, float, float] | None,
    quiet_baseline: dict[str, float] | None = None,
    step_sec: float = 1.5,
    source_offset_sec: float = 0.0,
) -> dict[str, Any]:
    """对粗扫出来的事件逐个回到区间里找播报，补上归属。

    `quiet_baseline` 是「播报区什么都没有时长什么样」的基准。
    没给就从第一帧取 —— 但第一帧未必是干净的，所以会标出来。
    """
    if banner_box is None:
        return {
            "ok": False,
            "error": "播报区还没标定。需要在标定页上把屏幕中间上方那条播报框出来。\n"
                     "**没标定就没法找横幅，归属只能继续是「不知道」。**",
            "events": scan_result.get("events", []),
        }
    # 先看有没有活要干。全都已经有归属时，连录像都不用碰 ——
    # 守卫的顺序有讲究：先检查录像存不存在，会让「本来就不需要精扫」的场景
    # 白白失败一次。
    needs_attribution = [
        e for e in scan_result.get("events", [])
        if e.get("teamId") not in (BLUE, RED)
    ]
    settled = [
        e for e in scan_result.get("events", [])
        if e.get("teamId") in (BLUE, RED)
    ]
    if not needs_attribution:
        return {"ok": True, "events": settled, "refined": 0,
                "note": "所有事件都已经有归属，不需要回去找播报。"}

    if not source.is_file():
        return {"ok": False, "error": f"找不到录像文件：{source}",
                "events": scan_result.get("events", [])}

    bracketed = bracket_events(needs_attribution, frame_times)
    groups = group_by_bracket(bracketed)

    baseline = quiet_baseline
    baseline_note = ""
    if baseline is None:
        baseline_note = ("播报区的「空白基准」是从区间起点那一帧取的。"
                         "如果那一帧刚好也有播报，这一段的判断都会失效。")

    refined: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for group in groups:
        window_baseline = baseline
        if window_baseline is None:
            # 拿区间起点那一帧当空白基准
            try:
                shot = source.parent / f".baseline_{int(group['fromSec'])}.jpg"
                video.crop_region(source, shot, group["fromSec"] + source_offset_sec,
                                  banner_box, scale_width=240)
                window_baseline = events_cv.read_patch(shot, (0.0, 0.0, 1.0, 1.0), size=32)
                shot.unlink(missing_ok=True)
            except Exception as err:      # noqa: BLE001
                reports.append({"fromSec": group["fromSec"],
                                "error": f"取基准失败：{type(err).__name__}: {err}"})
                refined.extend(group["events"])
                continue

        found = scan_window(source, banner_box,
                            group["fromSec"] + source_offset_sec,
                            group["toSec"] + source_offset_sec,
                            window_baseline, step_sec)
        if not found["found"]:
            for event in group["events"]:
                refined.append({**event, "attribution": found["reason"]})
            reports.append({"fromSec": group["fromSec"], "toSec": group["toSec"],
                            "found": False, "reason": found["reason"]})
            continue

        tint = read_tint(found["hits"])
        # 一个区间里有多个事件时，横幅未必属于其中哪一个
        ambiguous = group["count"] > 1
        for event in group["events"]:
            if tint["teamId"] is None:
                refined.append({**event,
                                "attribution": tint.get("reason", "颜色上判断不出归属。")})
                continue
            confidence = tint["strength"]
            note = f"区间 {group['fromSec']:.0f}~{group['toSec']:.0f}s 内找到播报，"
            if ambiguous:
                confidence *= 0.5
                note += (f"但这段区间里有 {group['count']} 个事件，"
                         "**没法确定这条播报属于哪一个** —— 置信度已减半。")
            else:
                note += "且区间内只有这一个事件。"
            refined.append({
                **event,
                "teamId": tint["teamId"],
                "confidence": round(confidence, 3),
                "attribution": note,
                "bannerEvidence": {"blue": tint["blue"], "red": tint["red"],
                                   "margin": tint["margin"],
                                   "hits": len(found["hits"])},
            })
        reports.append({"fromSec": group["fromSec"], "toSec": group["toSec"],
                        "found": True, "eventsInWindow": group["count"],
                        "tint": tint, "spanSec": found.get("spanSec")})

    return {
        "ok": True,
        "events": settled + refined,
        "refined": sum(1 for e in refined if e.get("teamId") in (BLUE, RED)),
        "stillUnknown": sum(1 for e in refined if e.get("teamId") not in (BLUE, RED)),
        "windows": reports,
        "baselineNote": baseline_note,
        "summary": _summary(refined, groups, baseline_note),
    }


def _summary(refined, groups, baseline_note) -> list[str]:
    lines: list[str] = []
    got = sum(1 for e in refined if e.get("teamId") in (BLUE, RED))
    lost = len(refined) - got
    multi = sum(1 for g in groups if g["count"] > 1)

    if got:
        lines.append(f"{got} 个事件通过播报补上了归属。")
    if lost:
        lines.append(
            f"{lost} 个仍然不知道是谁拿的。"
            "没找到横幅、或者横幅颜色分不出偏向 —— 这类只能人工补。"
        )
    if multi:
        lines.append(
            f"有 {multi} 个区间里挤了多个事件（多半是团战）。"
            "**那段里的播报没法确定属于哪一个事件**，置信度已经减半，"
            "这类结果要人工复核。"
        )
    if baseline_note:
        lines.append(baseline_note)
    if not lines:
        lines.append("没有需要补归属的事件。")
    return lines


def plan_cost(groups: list[dict[str, Any]], step_sec: float = 1.5) -> dict[str, Any]:
    """算一下精抽要多少帧。用来让人对代价有数。"""
    total = sum(max(1, int((g["toSec"] - g["fromSec"]) / step_sec) + 1) for g in groups)
    return {
        "windows": len(groups),
        "framesToExtract": total,
        "note": (f"只在 {len(groups)} 个窗口里精抽，共约 {total} 帧。"
                 "整场按同样密度抽会是它的几十倍 —— 这就是为什么先粗扫再精扫。"),
    }
