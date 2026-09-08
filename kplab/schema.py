"""State 的数据结构，以及「每个数字从哪来」的记账方式。

## 这个文件解决的是 LOL 项目不存在的问题

lolab 的每一个数字都来自 Riot 官方 API。`totalGold = 8421` 就是事实，
没有第二种可能。所以 lolab 的 State 里只需要存值。

王者荣耀没有官方 API。每一个数字都来自「对录像画面的识别」：
经济是从记分板上 OCR 出来的，位置是从小地图上找出来的，
等级是从头像旁边那个小数字读出来的。**这些都是推断，都可能错。**

如果照抄 lolab 的结构只存一个 `8421`，那么下游的所有人 —— 包括几个月后
接手的 AI —— 都会把它当成事实。等到胜率模型训出来发现不对劲，
已经没有任何办法回头区分「哪些数字是可信的、哪些是 OCR 猜的」。

所以这里的每个可观测字段都是一个三元组：

    value        值，识别不出来就是 None（**不是 0**）
    source       来源：manual / ocr / carried / derived / unknown
    confidence   0.0 ~ 1.0

## 为什么识别不出来必须是 None 而不是 0

「这一帧没读出经济」和「这一帧经济是 0」是完全不同的两件事。
写成 0，下游算经济差就会得到一个巨大的假差值，而且看起来完全正常。
LOL 项目的教训是「不编造数字」；在 OCR 场景下，把缺失写成 0 就是编造。
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------- 来源

SOURCE_MANUAL = "manual"    # 人工在标注页上填的，最可信
SOURCE_OCR = "ocr"          # 机器从画面识别的，可能错
SOURCE_CARRIED = "carried"  # 上一帧沿用下来的（这一帧没读到，但值不该凭空消失）
SOURCE_DERIVED = "derived"  # 由其他字段算出来的（例如经济差）
SOURCE_UNKNOWN = "unknown"  # 不知道，值必须是 None

SOURCES = (SOURCE_MANUAL, SOURCE_OCR, SOURCE_CARRIED, SOURCE_DERIVED, SOURCE_UNKNOWN)

# 各来源的默认置信度。manual 也不给 1.0 —— 人也会看错、也会填错行。
DEFAULT_CONFIDENCE = {
    SOURCE_MANUAL: 0.98,
    SOURCE_OCR: 0.5,
    SOURCE_CARRIED: 0.4,
    SOURCE_DERIVED: 0.0,   # 由参与计算的字段里最低的那个决定，见 derive()
    SOURCE_UNKNOWN: 0.0,
}

# 低于这个置信度的字段，前端必须显示成「存疑」而不是直接显示数字。
TRUST_THRESHOLD = 0.75


def field(
    value: Any = None,
    source: str = SOURCE_UNKNOWN,
    confidence: float | None = None,
) -> dict[str, Any]:
    """构造一个带来源的字段。

    值为 None 时强制 source=unknown、confidence=0 ——
    不允许出现「没有值但声称是人工填的」这种自相矛盾的记录。
    """
    if value is None:
        return {"value": None, "source": SOURCE_UNKNOWN, "confidence": 0.0}
    if source not in SOURCES:
        source = SOURCE_UNKNOWN
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE.get(source, 0.0)
    return {
        "value": value,
        "source": source,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
    }


def unknown() -> dict[str, Any]:
    return {"value": None, "source": SOURCE_UNKNOWN, "confidence": 0.0}


def get(entry: Any, default: Any = None) -> Any:
    """取出字段的值。传进来的如果已经是裸值，就原样返回（兼容手写数据）。"""
    if isinstance(entry, dict) and "value" in entry and "source" in entry:
        val = entry["value"]
        return default if val is None else val
    return default if entry is None else entry


def conf(entry: Any) -> float:
    if isinstance(entry, dict) and "confidence" in entry:
        try:
            return float(entry["confidence"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def src(entry: Any) -> str:
    if isinstance(entry, dict) and "source" in entry:
        return str(entry["source"])
    return SOURCE_UNKNOWN


def is_known(entry: Any) -> bool:
    return get(entry) is not None


def trusted(entry: Any) -> bool:
    """够不够格当作事实用。前端显示、模型训练都应该按这个判断。"""
    return is_known(entry) and conf(entry) >= TRUST_THRESHOLD


def carry(entry: Any, decay: float = 0.85) -> dict[str, Any]:
    """把上一帧的值沿用到这一帧，置信度按 decay 衰减。

    沿用是合理的（经济不会因为这一帧没读到就归零），但沿用得越久越不可信，
    所以置信度必须随帧数递减，不能原样带过来。
    """
    if not is_known(entry):
        return unknown()
    return field(get(entry), SOURCE_CARRIED, conf(entry) * decay)


def derive(value: Any, *parts: Any) -> dict[str, Any]:
    """由若干字段算出来的值，置信度取参与计算的字段里最低的那个。

    经济差 = 蓝方经济 − 红方经济。如果红方经济是 OCR 猜的（0.5），
    那么经济差最多也只能有 0.5 的可信度 —— 不能因为做了一次减法就变得更确定。
    """
    if value is None or not parts:
        return unknown()
    known = [p for p in parts if is_known(p)]
    if len(known) != len(parts):
        return unknown()   # 只要有一个是未知的，结果就是未知，不能当 0 处理
    return field(value, SOURCE_DERIVED, min(conf(p) for p in known))


# ---------------------------------------------------------------- 结构定义

# 每名选手每帧要记录的可观测字段
PLAYER_FIELDS = (
    "level",       # 等级
    "totalGold",   # 总经济
    "kills",
    "deaths",
    "assists",
    "x",           # 小地图坐标，0~1 的相对值，不是像素
    "y",
    "alive",       # 是否存活（死亡倒计时可见时为 False）
    "itemCount",   # 已成型装备数
)

# 每队每帧要记录的字段。中立资源的键来自 rules.NEUTRAL_OBJECTIVES，
# 五种资源五个计数器，绝不合并。
TEAM_COUNTERS = (
    "kills",
    "deaths",
    "towers",
    "highGroundTowers",
    "crystal",
    "tyrants",
    "darkTyrants",
    "overlords",
    "stormDragons",
    "prophetOverlords",
    "redBuffs",
    "blueBuffs",
)

# 这些字段只增不减，check 会据此抓 OCR 错误
MONOTONIC_TEAM = (
    "kills", "deaths", "towers", "highGroundTowers", "crystal",
    "tyrants", "darkTyrants", "overlords", "stormDragons", "prophetOverlords",
)
MONOTONIC_PLAYER = ("level", "totalGold", "kills", "deaths", "assists")

EVENT_TYPES = (
    "HERO_KILL",
    "TOWER_DESTROYED",
    "HIGHGROUND_DESTROYED",
    "CRYSTAL_DESTROYED",
    "TYRANT_KILL",
    "DARK_TYRANT_KILL",
    "OVERLORD_KILL",
    "STORM_DRAGON_KILL",
    "PROPHET_OVERLORD_KILL",
    "RED_BUFF_KILL",
    "BLUE_BUFF_KILL",
    "RECALL",
    "GAME_END",
)


def new_player(slot: int, team_id: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "slot": slot,
        "teamId": team_id,
        "heroName": None,
        "role": None,
        "playerName": None,
    }
    for name in PLAYER_FIELDS:
        entry[name] = unknown()
    return entry


def new_team() -> dict[str, Any]:
    entry: dict[str, Any] = {}
    for name in TEAM_COUNTERS:
        entry[name] = field(0, SOURCE_DERIVED, 1.0)
    entry["totalGold"] = unknown()
    entry["totalLevel"] = unknown()
    return entry


def coverage(frame: dict[str, Any]) -> dict[str, Any]:
    """这一帧到底有多少字段是真的知道的。

    这是核对页面上最该显示的一个数字：如果一帧里 40 个字段只认出来 6 个，
    那这一帧的「局势」根本谈不上可信，不该拿去训练任何东西。
    """
    known = total = trust = 0
    for player in frame.get("players", []) or []:
        for name in PLAYER_FIELDS:
            total += 1
            if is_known(player.get(name)):
                known += 1
            if trusted(player.get(name)):
                trust += 1
    for team in (frame.get("teams") or {}).values():
        for name in ("totalGold", "totalLevel"):
            total += 1
            if is_known(team.get(name)):
                known += 1
            if trusted(team.get(name)):
                trust += 1
    return {
        "knownFields": known,
        "trustedFields": trust,
        "totalFields": total,
        "knownRatio": round(known / total, 3) if total else 0.0,
        "trustedRatio": round(trust / total, 3) if total else 0.0,
    }


def strip_to_values(obj: Any) -> Any:
    """把带来源的结构压成裸值，只用于给人看的简表 / 导出。

    **不要**用它生成训练特征 —— 那会把置信度信息丢掉，
    正是这个文件想避免的事。
    """
    if isinstance(obj, dict):
        if "value" in obj and "source" in obj and "confidence" in obj:
            return obj["value"]
        return {k: strip_to_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_to_values(v) for v in obj]
    return obj


def iter_fields(frame: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """遍历一帧里所有带来源的字段，产出 (路径, 字段)。"""
    for player in frame.get("players", []) or []:
        for name in PLAYER_FIELDS:
            yield f"players[{player.get('slot')}].{name}", player.get(name) or unknown()
    for team_id, team in (frame.get("teams") or {}).items():
        for name, entry in team.items():
            if isinstance(entry, dict) and "source" in entry:
                yield f"teams[{team_id}].{name}", entry
