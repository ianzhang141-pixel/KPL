"""人工标注 —— 不需要装任何东西就能开始产出可信数据。

## 为什么这个模块排在 OCR 前面

LOL 项目卡了很久，卡点写在项目状态里：「用户还没有用真实数据跑过一遍核对」。
代码全都写好了，链路也通了，但因为需要用户在自己电脑上完成一系列操作，
整个项目就停在那儿，后面的每一步都动不了。

这边如果把「先装 ffmpeg、再装 tesseract、再标定 HUD、再跑识别」
设成开始工作的前提，会卡得更死 —— 那是四道坎，每一道都可能卡住好几天。

所以人工标注这条路是**零依赖**的：有一张截图（甚至只是对着录像暂停看一眼），
就能填一帧数据。它同时解决三件事：

  1. 第一天就能产出**完全可信**的数据（source=manual）
  2. 产出 ground truth —— 没有它，OCR 准不准根本无从验证（见 evaluate.py）
  3. 逼着人真的看一遍画面，从而发现标定错位、版本差异这类问题

## 标注的粒度

不要求每帧填满。只填看得清的：
经济和比分在记分板上一直有，等级要看头像旁边，位置要看小地图。
**填不出来就留空**，留空会被记成 unknown，这是正确的行为；
瞎填一个数才是灾难，因为它带着 manual 的高置信度进入下游。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import rules, schema, store

# 标注页上允许填的字段，以及各自的合法范围。
# 范围校验在写入那一刻就做 —— 让人在填错的当场看到，而不是几周后在报告里看到。
NUMERIC_FIELDS: dict[str, dict[str, Any]] = {
    "blueGold": {"label": "蓝方总经济", "lo": 0, "hi": 300000, "scope": "team"},
    "redGold": {"label": "红方总经济", "lo": 0, "hi": 300000, "scope": "team"},
    "blueKills": {"label": "蓝方击杀", "lo": 0, "hi": 200, "scope": "team"},
    "redKills": {"label": "红方击杀", "lo": 0, "hi": 200, "scope": "team"},
    # 单人直播视角通常只显示“我方/敌方”，不能在不知道出生阵营时硬映射成蓝/红。
    "allyKills": {"label": "己方击杀", "lo": 0, "hi": 200, "scope": "relative_team"},
    "enemyKills": {"label": "敌方击杀", "lo": 0, "hi": 200, "scope": "relative_team"},
    "blueTowers": {"label": "蓝方已拆塔数", "lo": 0, "hi": 20, "scope": "team"},
    "redTowers": {"label": "红方已拆塔数", "lo": 0, "hi": 20, "scope": "team"},
}


# 在示意图上目测点出来的位置能有多准。压在可信线（0.75）以下是有意的：
# 那是「大概在中路河道」，不是「在 (0.42, 0.61)」。
APPROX_POSITION_CONFIDENCE = 0.55


class AnnotateError(ValueError):
    pass


def _clean_int(raw: Any, lo: int, hi: int, label: str) -> int | None:
    """空值返回 None（合法：表示「看不清」）；填了但不合法就报错。"""
    if raw is None or raw == "":
        return None
    try:
        value = int(str(raw).strip().replace(",", ""))
    except (TypeError, ValueError):
        raise AnnotateError(f"{label} 填的不是整数：{raw!r}") from None
    if not (lo <= value <= hi):
        raise AnnotateError(f"{label} 填的是 {value}，超出合理范围 {lo}~{hi}。请再看一眼画面。")
    return value


def _clean_coord(raw: Any, label: str) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise AnnotateError(f"{label} 不是数字：{raw!r}") from None
    if not (0.0 <= value <= 1.0):
        raise AnnotateError(
            f"{label} 是 {value}，超出 0~1。"
            "小地图坐标用的是相对位置（左上角 0,0，右下角 1,1），不是像素。"
        )
    return round(value, 4)


def build_observation(
    at_sec: float,
    payload: dict[str, Any],
    rules_table: dict[str, Any] | None = None,
    frame_file: str | None = None,
) -> dict[str, Any]:
    """把标注页提交上来的一份表单，变成一条标准观测记录。

    payload 形如：
        {"blueGold": 12300, "redGold": 11800,
         "players": {"3": {"level": 8, "x": 0.42, "y": 0.61, "alive": true}},
         "events": [{"type": "TYRANT_KILL", "teamId": 100}]}
    """
    rules_table = rules_table or rules.load()
    max_level = int(rules.value(rules_table, "maxLevel"))
    max_slots = int(rules.value(rules_table, "itemSlots"))

    if at_sec is None:
        raise AnnotateError("必须填这一帧对应的比赛时间。没有时间的观测没法排进时间轴。")
    try:
        at_sec = float(at_sec)
    except (TypeError, ValueError):
        raise AnnotateError(f"比赛时间不是数字：{at_sec!r}") from None
    if at_sec < 0:
        raise AnnotateError("比赛时间不能是负数。")

    fields: dict[str, Any] = {}
    for key, spec in NUMERIC_FIELDS.items():
        value = _clean_int(payload.get(key), spec["lo"], spec["hi"], spec["label"])
        if value is not None:
            fields[key] = schema.field(value, schema.SOURCE_MANUAL)

    players: dict[str, Any] = {}
    for slot_raw, raw in (payload.get("players") or {}).items():
        try:
            slot = int(slot_raw)
        except (TypeError, ValueError):
            continue
        if not 1 <= slot <= 10 or not isinstance(raw, dict):
            continue
        entry: dict[str, Any] = {}
        for name, hi, label in (
            ("level", max_level, f"{slot} 号位等级"),
            ("kills", 100, f"{slot} 号位击杀"),
            ("deaths", 100, f"{slot} 号位死亡"),
            ("assists", 100, f"{slot} 号位助攻"),
            ("itemCount", max_slots, f"{slot} 号位装备数"),
        ):
            value = _clean_int(raw.get(name), 0, hi, label)
            if value is not None:
                entry[name] = schema.field(value, schema.SOURCE_MANUAL)

        # 位置有两种来源，精度差一个量级，**绝不能给同一个置信度**：
        #
        #   exact  — 在录像自己的小地图上点的（按标定的小地图区域换算）。
        #            这是精确的，可以当事实用。
        #   approx — 在示意图上点的。示意图和真实小地图不是等比的，
        #            没法把录像小地图上的一点对应到示意图上的同一点，
        #            所以这只是「大概在这一带」，人眼目测的结果。
        #
        # approx 的置信度压到可信线以下，它能用来看「谁在哪一带」，
        # 但不会进入模型特征 —— 否则等于拿人的目测误差当训练信号。
        precision = str(raw.get("posPrecision") or "approx")
        pos_confidence = (schema.DEFAULT_CONFIDENCE[schema.SOURCE_MANUAL]
                          if precision == "exact" else APPROX_POSITION_CONFIDENCE)
        for axis in ("x", "y"):
            value = _clean_coord(raw.get(axis), f"{slot} 号位小地图 {axis}")
            if value is not None:
                entry[axis] = schema.field(value, schema.SOURCE_MANUAL, pos_confidence)
        if "x" in entry or "y" in entry:
            entry["posPrecision"] = precision

        if raw.get("alive") is not None:
            entry["alive"] = schema.field(bool(raw["alive"]), schema.SOURCE_MANUAL)
        for text_key in ("heroName", "role", "playerName"):
            if raw.get(text_key):
                entry[text_key] = str(raw[text_key]).strip()[:40]
        if entry:
            players[str(slot)] = entry

    events: list[dict[str, Any]] = []
    for raw in payload.get("events") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or "").strip().upper()
        if kind not in schema.EVENT_TYPES:
            raise AnnotateError(
                f"不认识的事件类型：{kind or '(空)'}。"
                f"可选：{'、'.join(schema.EVENT_TYPES)}"
            )
        team_id = raw.get("teamId")
        try:
            team_id = int(team_id)
        except (TypeError, ValueError):
            raise AnnotateError(f"事件 {kind} 没写清楚是哪一方（teamId 要 100 或 200）。") from None
        if team_id not in (rules.BLUE, rules.RED):
            raise AnnotateError(f"事件 {kind} 的 teamId 只能是 100（蓝）或 200（红），收到 {team_id}。")
        events.append({
            "type": kind,
            "teamId": team_id,
            "atSec": float(raw.get("atSec") or at_sec),
            "note": str(raw.get("note") or "")[:120],
        })

    if not fields and not players and not events:
        raise AnnotateError(
            "这一帧什么都没填。空标注不会被保存 —— "
            "如果这一帧确实什么都看不清，跳过它就好，不用记一条空的。"
        )

    return {
        "atSec": round(at_sec, 2),
        "frameFile": frame_file,
        "fields": fields,
        "players": players,
        "events": events,
        "annotatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "by": str(payload.get("by") or "")[:40],
    }


def save(
    data_dir: Path, game_id: str, observation: dict[str, Any]
) -> dict[str, Any]:
    """把一条标注同时写进 annotations 和 observations。

    两处都写是有意的：
      · annotations 是**纯人工**记录，是评估 OCR 的标尺，不能混进机器结果
      · observations 是喂给 state.build 的输入，人工和机器的都在里面
    """
    store.append_jsonl_gz(store.annotations_path(data_dir, game_id), observation)
    store.append_jsonl_gz(store.observations_path(data_dir, game_id), observation)
    return observation


def load_annotations(data_dir: Path, game_id: str) -> list[dict[str, Any]]:
    return list(store.read_jsonl_gz(store.annotations_path(data_dir, game_id)))


def progress(data_dir: Path, game_id: str) -> dict[str, Any]:
    """标注进度，给网页显示。"""
    records = load_annotations(data_dir, game_id)
    times = sorted(float(r.get("atSec") or 0) for r in records)
    filled = sum(
        len(r.get("fields") or {}) + sum(len(v) for v in (r.get("players") or {}).values())
        for r in records
    )
    return {
        "count": len(records),
        "fieldsFilled": filled,
        "events": sum(len(r.get("events") or []) for r in records),
        "firstSec": times[0] if times else None,
        "lastSec": times[-1] if times else None,
        "annotatedSeconds": times,
    }
