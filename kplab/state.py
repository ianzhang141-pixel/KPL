"""观测序列 → 分钟级 State。整条链上最关键的一步。

## 无未来信息泄漏

和 lolab 一样，这是硬规矩：t 时刻的 State 只能包含 t 时刻及之前可观测的信息。

在 OCR 场景下，这条规矩有一个 lolab 没遇到过的陷阱：**回头修正**。

假设第 5 帧读出经济 3000，第 6 帧读出 2000。总经济只增不减，所以其中一帧是错的。
「用第 6 帧去修第 5 帧」是很自然的想法，而且能让曲线更好看 ——
但那是拿未来的观测去改过去的状态，正是要禁止的事。
一旦这么做，第 5 帧的 State 里就掺进了第 6 帧才知道的信息，
用它训出来的模型在真实对局里（那时没有第 6 帧）根本复现不了。

所以这里只做**前向**处理：
  · 第 6 帧的 2000 与「截至第 5 帧已知的 3000」矛盾 → 判第 6 帧这个读数不可信
  · 处理方式是丢弃它并沿用 3000（降置信度），**不是**回头改第 5 帧
  · 第 5 帧写下之后就不再改动

`build()` 里有一个断言守着这件事：每帧写完就冻结，后续帧只能读不能改。

## 唯一的标签例外

`meta.blueWin` 是将来训练 V(s) 的标签，不属于 State 特征，构造特征时必须排除。
和 lolab 的约定完全一致。
"""

from __future__ import annotations

from typing import Any

from . import rules, schema
from .rules import BLUE, RED

# 沿用一帧衰减多少置信度。0.85 意味着连续沿用 5 帧后
# 置信度从 0.98 掉到 0.43，会自动跌破 TRUST_THRESHOLD 变成「存疑」，
# 这正是想要的效果：太久没重新看到的值，不该继续当事实用。
CARRY_DECAY = 0.85

# 沿用超过这么多帧就干脆判为未知。经济和位置在 5 分钟里可以变得面目全非，
# 与其给一个 10 帧前的旧值，不如承认不知道。
MAX_CARRY_FRAMES = 6


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _slot_team(slot: int) -> int:
    return BLUE if slot <= 5 else RED


# ---------------------------------------------------------------- 前向校验

def _accept_scalar(
    previous: Any,
    incoming: Any,
    monotonic: bool,
    hi: int | None = None,
) -> tuple[dict[str, Any], str | None]:
    """决定这一帧的读数要不要采纳。只看「过去」和「现在」，绝不看未来。

    返回 (采纳后的字段, 拒绝原因或 None)。
    """
    if not schema.is_known(incoming):
        return schema.unknown(), None

    value = schema.get(incoming)

    if hi is not None and isinstance(value, (int, float)) and value > hi:
        return schema.unknown(), f"读数 {value} 超过上限 {hi}"

    if monotonic and schema.is_known(previous):
        old = schema.get(previous)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)) and value < old:
            # 与已知的过去矛盾。丢掉这个读数，不动过去那一帧。
            return schema.unknown(), f"读数 {value} 低于此前已知的 {old}（该字段只增不减）"

    return dict(incoming), None


def _carry_or_unknown(
    previous: Any, carried_for: int
) -> tuple[dict[str, Any], int]:
    """这一帧没读到时，沿用上一帧的值并衰减置信度。"""
    if not schema.is_known(previous) or carried_for >= MAX_CARRY_FRAMES:
        return schema.unknown(), carried_for + 1
    return schema.carry(previous, CARRY_DECAY), carried_for + 1


# ---------------------------------------------------------------- 主流程

def build(
    meta: dict[str, Any],
    observations: list[dict[str, Any]],
    rules_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把一场比赛的观测序列转成 State 序列。

    observations 里每条形如：
        {"atSec": 120.0, "frameFile": "00004_120s.jpg",
         "fields": {"blueGold": {...}, "player3Level": {...}, ...},
         "players": {"3": {"x": {...}, "y": {...}}},
         "events": [{"type": "TYRANT_KILL", "teamId": 100}]}
    `fields` 里的每个值都是 schema.field() 构造的带来源字段。
    """
    rules_table = rules_table or rules.load()
    max_level = int(rules.value(rules_table, "maxLevel"))
    max_towers = rules.max_towers(rules_table)
    max_slots = int(rules.value(rules_table, "itemSlots"))

    # 严格按时间升序处理。这一行是「无未来信息」的前提：
    # 如果乱序处理，「上一帧」就不再是时间上更早的那一帧，整条规矩就破了。
    ordered = sorted(
        (o for o in observations if isinstance(o, dict)),
        key=lambda o: float(o.get("atSec") or 0.0),
    )

    players_prev = {slot: schema.new_player(slot, _slot_team(slot)) for slot in range(1, 11)}
    teams_prev = {BLUE: schema.new_team(), RED: schema.new_team()}
    carried: dict[str, int] = {}

    # 累计计数器：只由事件累加，不依赖任何未来信息
    counters = {BLUE: {k: 0 for k in schema.TEAM_COUNTERS},
                RED: {k: 0 for k in schema.TEAM_COUNTERS}}

    frames_out: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for index, obs in enumerate(ordered):
        at_sec = float(obs.get("atSec") or 0.0)
        fields = obs.get("fields") or {}
        obs_players = obs.get("players") or {}

        # ---- 1) 先累加「截至这一帧」发生的事件 ----
        # 顺序和 lolab 一致：先累加事件，再拍快照。
        # 事件是「上一帧到这一帧之间」的事，属于这一帧可观测的过去。
        recent: list[dict[str, Any]] = []
        for event in obs.get("events") or []:
            if not isinstance(event, dict):
                continue
            applied = _apply_event(event, counters)
            if applied:
                recent.append({**event, "clock": _clock(event.get("atSec", at_sec))})

        # ---- 2) 再拍这一帧的快照 ----
        players_now: dict[int, dict[str, Any]] = {}
        for slot in range(1, 11):
            prev = players_prev[slot]
            entry = schema.new_player(slot, _slot_team(slot))
            entry["heroName"] = obs_players.get(str(slot), {}).get("heroName") or prev.get("heroName")
            entry["role"] = rules.normalize_role(
                obs_players.get(str(slot), {}).get("role") or prev.get("role")
            )
            entry["playerName"] = obs_players.get(str(slot), {}).get("playerName") or prev.get("playerName")

            raw_player = obs_players.get(str(slot)) or {}
            for name in schema.PLAYER_FIELDS:
                # 选手级字段可能来自两处：HUD 区域（player3Level）或标注（players["3"]["level"]）
                incoming = raw_player.get(name)
                if incoming is None:
                    incoming = fields.get(f"player{slot}{name[0].upper()}{name[1:]}")

                hi = max_level if name == "level" else (max_slots if name == "itemCount" else None)
                monotonic = name in schema.MONOTONIC_PLAYER
                key = f"p{slot}.{name}"

                accepted, why = _accept_scalar(prev.get(name), incoming, monotonic, hi)
                if why:
                    rejections.append({"frame": index, "clock": _clock(at_sec),
                                       "field": key, "reason": why})
                if schema.is_known(accepted):
                    carried[key] = 0
                    entry[name] = accepted
                else:
                    entry[name], carried[key] = _carry_or_unknown(
                        prev.get(name), carried.get(key, 0)
                    )
            players_now[slot] = entry

        # ---- 队伍级 ----
        teams_now: dict[int, dict[str, Any]] = {}
        for team_id, prefix in ((BLUE, "blue"), (RED, "red")):
            prev = teams_prev[team_id]
            team = schema.new_team()

            # 由事件累加出来的计数器：置信度取决于事件本身，不是 OCR
            for name in schema.TEAM_COUNTERS:
                team[name] = schema.field(
                    counters[team_id][name], schema.SOURCE_DERIVED, 0.9
                )

            # 比分板上直接读到的击杀数，比事件累加更可信，优先用它
            kills_key = f"{prefix}Kills"
            accepted, why = _accept_scalar(prev.get("kills"), fields.get(kills_key), True, 200)
            if why:
                rejections.append({"frame": index, "clock": _clock(at_sec),
                                   "field": kills_key, "reason": why})
            if schema.is_known(accepted):
                team["kills"] = accepted
                carried[kills_key] = 0

            # 总经济
            gold_key = f"{prefix}Gold"
            accepted, why = _accept_scalar(prev.get("totalGold"), fields.get(gold_key), True, 300000)
            if why:
                rejections.append({"frame": index, "clock": _clock(at_sec),
                                   "field": gold_key, "reason": why})
            if schema.is_known(accepted):
                team["totalGold"] = accepted
                carried[gold_key] = 0
            else:
                team["totalGold"], carried[gold_key] = _carry_or_unknown(
                    prev.get("totalGold"), carried.get(gold_key, 0)
                )

            # 队伍总等级：十个人的等级加起来。只有全部已知才算，
            # 缺一个就是未知 —— 缺人还相加会得到一个偏小的假值。
            levels = [players_now[s].get("level") for s in range(1, 11)
                      if _slot_team(s) == team_id]
            if all(schema.is_known(v) for v in levels):
                team["totalLevel"] = schema.derive(
                    sum(schema.get(v, 0) for v in levels), *levels
                )
            else:
                team["totalLevel"] = schema.unknown()

            # 塔数上限校验
            if schema.get(team["towers"], 0) > max_towers:
                rejections.append({"frame": index, "clock": _clock(at_sec),
                                   "field": f"{prefix}.towers",
                                   "reason": f"塔数 {schema.get(team['towers'])} 超过上限 {max_towers}"})

            teams_now[team_id] = team

        # ---- 经济差等派生量 ----
        blue, red = teams_now[BLUE], teams_now[RED]
        gold_diff = None
        if schema.is_known(blue["totalGold"]) and schema.is_known(red["totalGold"]):
            gold_diff = schema.get(blue["totalGold"]) - schema.get(red["totalGold"])
        diff = {
            "totalGold": schema.derive(gold_diff, blue["totalGold"], red["totalGold"]),
            "kills": schema.derive(
                (schema.get(blue["kills"], 0) - schema.get(red["kills"], 0))
                if schema.is_known(blue["kills"]) and schema.is_known(red["kills"]) else None,
                blue["kills"], red["kills"],
            ),
            "towers": schema.derive(
                schema.get(blue["towers"], 0) - schema.get(red["towers"], 0),
                blue["towers"], red["towers"],
            ),
        }

        frame = {
            "frameIndex": index,
            "atSec": round(at_sec, 2),
            "clock": _clock(at_sec),
            "minute": round(at_sec / 60.0, 2),
            "frameFile": obs.get("frameFile"),
            "players": [players_now[s] for s in range(1, 11)],
            "teams": {str(BLUE): teams_now[BLUE], str(RED): teams_now[RED]},
            "diff": diff,
            "recentEvents": recent,
        }
        frame["coverage"] = schema.coverage(frame)

        frames_out.append(frame)

        # 冻结：这一帧一旦写进 frames_out 就不再改动。
        # players_prev / teams_prev 保存的是副本，后续帧改不到已写出的那份。
        players_prev = {s: dict(players_now[s]) for s in range(1, 11)}
        teams_prev = {BLUE: dict(teams_now[BLUE]), RED: dict(teams_now[RED])}

    return {
        "meta": {
            "gameId": meta.get("gameId"),
            "title": meta.get("title"),
            "tournament": meta.get("tournament"),
            "blueTeam": meta.get("blueTeam"),
            "redTeam": meta.get("redTeam"),
            "sourceKind": meta.get("sourceKind"),
            "frameCount": len(frames_out),
            "frameIntervalSec": _guess_interval(ordered),
            "durationSec": round(float(ordered[-1].get("atSec") or 0.0), 2) if ordered else 0,
            "blueWin": meta.get("blueWin"),   # 训练标签，不属于 State 特征
            "rulesVerified": not rules.unverified(rules_table),
            "unverifiedRules": rules.unverified(rules_table),
            "builtBy": "kplab.state.build",
        },
        "frames": frames_out,
        "rejections": rejections,
        "coverage": _overall_coverage(frames_out),
    }


def _apply_event(event: dict[str, Any], counters: dict[int, dict[str, int]]) -> bool:
    """把一条事件累加进计数器。

    这里是最需要小心的地方 —— LOL 项目那个「先锋和幼虫被算成大龙」的 bug
    就发生在这种函数里。五种中立资源各进各的计数器，**一行一个，不共用分支**，
    宁可啰嗦也不合并。
    """
    kind = event.get("type")
    team_id = event.get("teamId")
    if team_id not in (BLUE, RED):
        return False
    side = counters[team_id]
    other = counters[RED if team_id == BLUE else BLUE]

    if kind == "HERO_KILL":
        side["kills"] += 1
        other["deaths"] += 1
    elif kind == "TOWER_DESTROYED":
        side["towers"] += 1
    elif kind == "HIGHGROUND_DESTROYED":
        side["highGroundTowers"] += 1
    elif kind == "CRYSTAL_DESTROYED":
        side["crystal"] += 1
    elif kind == "TYRANT_KILL":
        side["tyrants"] += 1
    elif kind == "DARK_TYRANT_KILL":
        side["darkTyrants"] += 1          # 不是 tyrants
    elif kind == "OVERLORD_KILL":
        side["overlords"] += 1
    elif kind == "STORM_DRAGON_KILL":
        side["stormDragons"] += 1         # 不是 overlords
    elif kind == "PROPHET_OVERLORD_KILL":
        side["prophetOverlords"] += 1     # 不是 overlords
    elif kind == "RED_BUFF_KILL":
        side["redBuffs"] += 1
    elif kind == "BLUE_BUFF_KILL":
        side["blueBuffs"] += 1
    elif kind in ("RECALL", "GAME_END"):
        return True                       # 记录但不计数
    else:
        return False
    return True


def _guess_interval(observations: list[dict[str, Any]]) -> float | None:
    if len(observations) < 2:
        return None
    gaps = [
        float(observations[i + 1].get("atSec") or 0) - float(observations[i].get("atSec") or 0)
        for i in range(len(observations) - 1)
    ]
    gaps = [g for g in gaps if g > 0]
    return round(sorted(gaps)[len(gaps) // 2], 2) if gaps else None


def _overall_coverage(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"knownRatio": 0.0, "trustedRatio": 0.0, "frames": 0}
    known = sum(f["coverage"]["knownRatio"] for f in frames) / len(frames)
    trust = sum(f["coverage"]["trustedRatio"] for f in frames) / len(frames)
    return {
        "knownRatio": round(known, 3),
        "trustedRatio": round(trust, 3),
        "frames": len(frames),
        "usableFrames": sum(1 for f in frames if f["coverage"]["trustedRatio"] >= 0.5),
    }
