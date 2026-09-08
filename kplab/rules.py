"""王者荣耀的游戏常量 —— 以及一份「这些常量还没被人核对过」的声明。

## 为什么这个文件要单独存在

LOL 项目里有一个明确的 bug：先锋（RIFTHERALD）和虚空幼虫（HORDE）被累加进了
`barons`（大龙）。开局吃 6 个幼虫，界面上显示「大龙 6 条」，用这份数据训出来的
胜率模型是坏的，而且坏得很安静 —— 没有任何一步会报错。

那个 bug 的根源不是手滑，是**把语义不同的资源合并进了同一个计数器**。
王者荣耀的中立资源比 LOL 更容易犯这个错：暴君和主宰是两条河道的两个东西，
黑暗暴君（暗影暴君）是暴君的强化版但收益完全不同，风暴龙王在游戏后期替换主宰。
它们必须是四个独立的计数器，任何时候都不许合并。

## 第二个原因：版本会变，而我不能假装记得

暴君几分几秒刷新、主宰什么时候变成风暴龙王、每路有几座塔 —— 这些数字
**每个赛季都可能调整**，写代码的人（包括 AI）凭记忆写下来的很可能是过时的。

所以这里的每一条都带 `verified` 标记。用户已提供的S44明确数值标为 `True`，
没有被本次信息覆盖的旧默认值仍为 `False`，意思是
**「这是待核对的默认值，不是已确认的事实」**。
`kplab check` 会在报告顶部提醒还有多少条没核对。
用户核对后可以写进 data/kpl/rules_override.json 覆盖，不用改代码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BLUE, RED = 100, 200

TEAM_NAMES = {BLUE: "蓝方", RED: "红方"}

# ---------------------------------------------------------------- 位置

# 王者荣耀的分路命名和 LOL 不一样，不要用 TOP/MID/BOT/JUNGLE/SUPPORT 硬套。
ROLES = ("对抗路", "中路", "发育路", "游走", "打野")

ROLE_ALIASES = {
    "对抗路": "对抗路", "上单": "对抗路", "单人路": "对抗路", "TOP": "对抗路",
    "中路": "中路", "中单": "中路", "法师": "中路", "MID": "中路",
    "发育路": "发育路", "射手": "发育路", "ADC": "发育路", "BOT": "发育路",
    "游走": "游走", "辅助": "游走", "SUP": "游走", "SUPPORT": "游走",
    "打野": "打野", "JUG": "打野", "JUNGLE": "打野",
}


def normalize_role(text: str | None) -> str | None:
    if not text:
        return None
    return ROLE_ALIASES.get(text.strip().upper()) or ROLE_ALIASES.get(text.strip())


# ---------------------------------------------------------------- 中立资源

# 关键：这五种是**五个独立的计数器**，任何情况下都不许合并成一个「大龙数」。
# 这正是 LOL 项目 ③ 号 bug 的教训。
NEUTRAL_OBJECTIVES = {
    "TYRANT": {
        "key": "tyrants",
        "label": "暴君",
        "lane": "下路河道",
        "note": "击杀给全队经济。是前期资源，和主宰不是一回事。",
    },
    "DARK_TYRANT": {
        "key": "darkTyrants",
        "label": "暗影暴君",
        "lane": "下路河道",
        "note": "S44于10分钟出现，收益与暴君不同，必须分开计。",
    },
    "OVERLORD": {
        "key": "overlords",
        "label": "主宰",
        "lane": "上路河道",
        "note": "击杀后推出主宰先锋（强化兵线），是推塔资源，不是经济资源。",
    },
    "STORM_DRAGON": {
        "key": "stormDragons",
        "label": "风暴龙王",
        "lane": "上路河道",
        "note": "游戏后期替换主宰。收益远大于主宰，绝不能并进 overlords。",
    },
    "PROPHET_OVERLORD": {
        "key": "prophetOverlords",
        "label": "暗影主宰",
        "lane": "上路河道",
        "note": "S44于10分钟出现；字段名保留旧称以兼容已有数据。",
    },
}

BUFFS = {
    "RED_BUFF": {"key": "redBuffs", "label": "红BUFF"},
    "BLUE_BUFF": {"key": "blueBuffs", "label": "蓝BUFF"},
}

# ---------------------------------------------------------------- 可计算核心常量

CURRENT_SEASON = "S44"
USER_RULE_SOURCE = "用户提供于 2026-09-08"


def _pending(value: Any, what: str, why: str = "") -> dict[str, Any]:
    """一条还没被人核对过的常量。"""
    return {"value": value, "verified": False, "what": what, "why": why}


def _confirmed(value: Any, what: str, why: str = "") -> dict[str, Any]:
    """用户明确提供的当前赛季常量；不等同于外部官方来源复核。"""
    return {"value": value, "verified": True, "what": what, "why": why,
            "season": CURRENT_SEASON, "by": USER_RULE_SOURCE}


# S44中由用户明确提供的整数已标记为用户核对；其余仍是待核对默认值。
# 网页再次核对会写进 rules_override.json，不用修改源码。
DEFAULTS: dict[str, dict[str, Any]] = {
    "towersPerLane": _pending(
        2, "每条路的外塔数量（一塔 + 二塔）",
        "check 会用它判断「塔数」的上限，写错了就检测不出 OCR 把 6 看成 8",
    ),
    "highGroundTowers": _pending(3, "高地塔数量", "同上，用于上限校验"),
    "laneCount": _confirmed(3, "分路数量", "三路兵线组成已由用户提供"),
    "tyrantFirstSpawnSec": _confirmed(
        240, "暴君首次刷新时间（秒）",
        "用于判断「0 分 30 秒就出现暴君击杀」这种明显不可能的识别结果",
    ),
    "tyrantRespawnSec": _confirmed(210, "暴君重生间隔（秒）"),
    "overlordFirstSpawnSec": _confirmed(240, "主宰首次刷新时间（秒）", "同上"),
    "overlordRespawnSec": _confirmed(240, "主宰重生间隔（秒）"),
    "darkTyrantFromSec": _confirmed(600, "暗影暴君首次出现时间（秒）", "同上"),
    "darkTyrantRespawnSec": _confirmed(210, "暗影暴君重生间隔（秒）"),
    "shadowOverlordFromSec": _confirmed(600, "暗影主宰首次出现时间（秒）"),
    "shadowOverlordRespawnSec": _confirmed(210, "暗影主宰重生间隔（秒）"),
    "stormDragonFromSec": _confirmed(
        1200, "风暴龙王首次出现时间（秒）",
        "S44为20分钟，不能沿用旧版本的15分钟值",
    ),
    "stormDragonRespawnSec": _confirmed(180, "风暴龙王重生间隔（秒）"),
    "primalBondDurationSec": _confirmed(90, "原初羁绊持续时间（秒）"),
    "primalBondObjectiveDamageReductionPct": _confirmed(50, "原初羁绊对龙伤害降低（%）"),
    "minionFirstSpawnSec": _confirmed(10, "首波兵线登场时间（秒）"),
    "minionWaveRespawnSec": _confirmed(33, "每波兵线刷新间隔（秒）"),
    "crossbowMinionFromSec": _confirmed(240, "弩车加入兵线时间（秒）"),
    "cannonMinionFromSec": _confirmed(600, "炮车加入兵线时间（秒）"),
    "minionSpeedupFromSec": _confirmed(600, "兵线开始加速时间（秒）"),
    "jungleProtectionEndSec": _confirmed(240, "野区保护结束时间（秒）"),
    "jungleProtectionDamageReductionPct": _confirmed(15, "野区保护伤害降低（%）"),
    "towerEarlyProtectionEndSec": _confirmed(240, "一塔前期保护结束时间（秒）"),
    "towerEarlyDamageReductionPct": _confirmed(40, "一塔前期受到伤害额外降低（%）"),
    "buffFirstSpawnSec": _confirmed(30, "红蓝石像首次出现时间（秒）"),
    "buffRespawnSec": _confirmed(90, "红蓝石像重生间隔（秒）"),
    "buffDurationSec": _confirmed(70, "红蓝石像增益持续时间（秒）"),
    "primordialSpiritFirstSpawnSec": _confirmed(60, "空间之灵首次出现时间（秒）"),
    "primordialSpiritRespawnSec": _confirmed(60, "空间之灵刷新间隔（秒）"),
    "primordialSpiritEndSec": _confirmed(240, "空间之灵停止刷新时间（秒）"),
    "primordialPortalEndSec": _confirmed(600, "原初法阵消失时间（秒）"),
    "typicalGameSec": _pending(
        1200, "一局的典型时长（秒）",
        "只用于进度条和异常提示，不参与任何计算",
    ),
    "maxLevel": _pending(15, "英雄最高等级", "用于识别结果的上限校验"),
    "itemSlots": _pending(6, "装备栏数量", "用于识别结果的上限校验"),
}


def load(data_dir: Path | None = None) -> dict[str, Any]:
    """读取常量表，允许用 data/kpl/rules_override.json 覆盖。

    覆盖文件的格式：
        {"towersPerLane": {"value": 2, "verified": true, "by": "用户核对于 2026-09-08"}}
    只写 value 也可以，但那样 verified 仍然是 False —— **核对过要明确写出来**。
    """
    table = {k: dict(v) for k, v in DEFAULTS.items()}
    if data_dir is None:
        return table

    path = override_path(data_dir)
    if not path.is_file():
        return table
    try:
        with path.open("r", encoding="utf-8") as handle:
            override = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return table
    if not isinstance(override, dict):
        return table

    for key, raw in override.items():
        if key not in table:
            continue
        if isinstance(raw, dict):
            if "value" in raw:
                table[key]["value"] = raw["value"]
            table[key]["verified"] = bool(raw.get("verified"))
            if raw.get("by"):
                table[key]["by"] = raw["by"]
        else:
            table[key]["value"] = raw
    return table


def override_path(data_dir: Path) -> Path:
    return data_dir / "kpl" / "rules_override.json"


def value(table: dict[str, Any], key: str) -> Any:
    entry = table.get(key) or DEFAULTS.get(key) or {}
    return entry.get("value")


def unverified(table: dict[str, Any]) -> list[str]:
    """返回还没被核对过的常量名。"""
    return sorted(k for k, v in table.items() if not v.get("verified"))


def save_override(data_dir: Path, key: str, val: Any, by: str = "") -> Path:
    """把用户核对过的一条常量写进覆盖文件，并标记 verified。"""
    path = override_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, ValueError, UnicodeDecodeError):
            current = {}
    entry: dict[str, Any] = {"value": val, "verified": True,
                             "season": CURRENT_SEASON}
    if by:
        entry["by"] = by
    current[key] = entry
    with path.open("w", encoding="utf-8") as handle:
        json.dump(current, handle, ensure_ascii=False, indent=2)
    return path


def max_towers(table: dict[str, Any]) -> int:
    """一方最多能被拆多少座塔（含高地塔），用于识别结果的上限校验。"""
    return (
        int(value(table, "towersPerLane")) * int(value(table, "laneCount"))
        + int(value(table, "highGroundTowers"))
    )
