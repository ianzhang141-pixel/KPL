"""样本够不够？—— 用算的，不用猜的。

## 这个文件回答什么

用户问了三件事：

1. 没有导播的个人视角录像看不到经济面板，是不是就废了？
2. 只用职业比赛，样本量够不够？要多少？
3. 跨赛季的老录像能不能用？越远权重越低行不行？

下面每一条都有一个明确的答案，而且**关键的那条不是「要多少场」**。

---

## 一、个人视角录像：不是废了，是分层用

用户说得对：没有导播的对局几乎看不到经济面板，只在选手偶尔调出来的那几帧能看到。
所以这类录像**做不了经济曲线**。

但 `events_cv.py` 那条路**根本不需要经济面板** ——
推塔看小地图图标、打龙看图标、击杀看头像变灰。这些在个人视角里一样看得见。

所以正确的做法不是丢掉，是**按能提供什么分层**：

| 录像类型 | 经济曲线 | 事件（塔/龙/人头） | 位置 |
| --- | --- | --- | --- |
| 职业转播 | ✅ 长时间显示 | ✅ | ✅ |
| 个人视角 | ❌ 基本没有 | ✅ | ✅ 小地图一直在 |

个人视角录像仍然能贡献**事件序列**和**位置**，
只是它的帧在训练 V(s) 时会因为缺经济特征而被跳过（`model.extract_features` 已经这么做了）。
**不用为它们额外写代码，现有的置信度机制已经处理了。**

---

## 二、样本量：卡住因果推断的不是「多少场」

这是最重要的一节。

训 V(s)（胜率曲线）确实是「越多越好」，几百场就能有个像样的东西。
但 **Q(s,a)（换个打法会怎样）卡住的不是总量，是重叠性（positivity）。**

要估计「在局面 s 下做动作 a 会怎样」，数据里必须**真的有人在 s 下做过 a**。
如果职业选手在某个局面下从来不做某个选择 —— 比如落后 5000 经济时从不开龙 ——
那么「落后时开龙会怎样」这个格子是空的，
**再加十万场职业比赛，它还是空的。**

    数据量解决方差，不解决偏差。
    重叠性缺失比偏差更狠：它让那个问题在数学上就无法回答。

### 这里有个反直觉的结论

**只用职业比赛，会让测量变好，但让因果识别变差。**

职业选手打得规范，意味着他们的选择高度集中在一个很窄的带里。
恰恰是「不规范的选择」才提供了反事实所需的对照。

所以路人局 / 个人视角录像在因果推断上**反而有独特价值** ——
他们会做职业选手不会做的选择，那些正是空格子的填充物。

`occupancy()` 和 `positivity_report()` 就是用来看这些格子空不空的。

---

## 三、跨赛季：能用，但要分清「漂移」和「断裂」

用户的直觉（越远的赛季权重越低）方向是对的，是标准做法。
但有一个区分必须做：

| 类型 | 例子 | 降权能不能解决 |
| --- | --- | --- |
| **平滑漂移** | 版本微调、英雄强度小幅变动 | ✅ 能 |
| **结构断裂** | 暴君收益改动、地图改版、新机制 | ❌ **不能** |

降权处理的是「同一个规律，噪音大一点」。
但如果某个补丁把暴君的收益改了，那是**因果结构本身变了** ——
老数据里「拿暴君 → 赢」的强度和新版本不是同一个量，
降权只是让错误的信号小声一点，它仍然在往错的方向拉。

### 正确做法：先测，再决定合不合并

`season_stability()` 做的就是这件事：按赛季分别拟合，比较系数。

- 系数在各赛季间稳定 → 可以合并，用时间衰减权重
- 某个系数在某赛季突变 → 那一项发生了结构断裂，
  老赛季的数据**只能用于其他项**，不能用于这一项

**哪些东西大概率跨赛季稳定**（可以放心用老数据）：
地图拓扑、「经济领先 → 胜率高」这个单调关系、
「推塔是不可逆的」、「时间越晚领先越难翻盘」。

**哪些大概率不稳定**：各个中立资源的具体价值、英雄相关的一切。
"""

from __future__ import annotations

import math
from typing import Any, Callable

from . import model, schema

# ---------------------------------------------------------------- 分层

# 状态分层。格子太细每格没样本，太粗又看不出重叠性问题。
# 这个粒度是起点，真实数据到位后要调。
MINUTE_BINS = ((0, 5), (5, 10), (10, 15), (15, 99))
GOLD_BINS = ((-999, -5), (-5, -2), (-2, 2), (2, 5), (5, 999))   # 单位：千


def stratum_of(features: dict[str, float]) -> tuple[str, str] | None:
    """把一帧的特征映射到一个状态格子。"""
    minute = features.get("minute")
    gold = features.get("goldDiffK")
    if minute is None or gold is None:
        return None
    minute_label = next((f"{lo}-{hi}分" for lo, hi in MINUTE_BINS if lo <= minute < hi), None)
    gold_label = next((f"经济差{lo}~{hi}k" for lo, hi in GOLD_BINS if lo <= gold < hi), None)
    if minute_label is None or gold_label is None:
        return None
    return (minute_label, gold_label)


def occupancy(
    games: list[dict[str, Any]],
    action_of: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """统计每个（状态格子 × 动作）里有多少样本。

    `action_of` 把一帧映射成一个动作名。**现在传不了** ——
    行为抽象还没做（那是路线图里的「你2」，必须由懂游戏的人定义）。
    不传就只统计状态覆盖，不统计动作重叠。
    """
    cells: dict[tuple[str, str, str], int] = {}
    states: dict[tuple[str, str], int] = {}
    usable = skipped = 0

    for built in games:
        for frame in built.get("frames") or []:
            features = model.extract_features(frame)
            if features is None:
                skipped += 1
                continue
            stratum = stratum_of(features)
            if stratum is None:
                skipped += 1
                continue
            usable += 1
            states[stratum] = states.get(stratum, 0) + 1
            if action_of is None:
                continue
            action = action_of(frame)
            if action:
                key = (stratum[0], stratum[1], action)
                cells[key] = cells.get(key, 0) + 1

    return {
        "games": len(games),
        "usableFrames": usable,
        "skippedFrames": skipped,
        "states": {f"{a}|{b}": n for (a, b), n in sorted(states.items())},
        "cells": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(cells.items())},
        "hasActions": action_of is not None,
        "actionNote": None if action_of else (
            "没有传入动作定义，所以只统计了状态覆盖。\n"
            "**重叠性（决定因果推断能不能做）需要动作定义才能算** ——\n"
            "而行为抽象还没做，那一步必须由懂游戏的人来定义。"
        ),
    }


# 一个格子里少于这么多样本，就不足以支撑该格子的估计。
# 30 是个常见的经验起点，不是定理。
MIN_PER_CELL = 30


def positivity_report(occ: dict[str, Any]) -> dict[str, Any]:
    """看有哪些格子是空的或太薄 —— 那些就是答不了的问题。"""
    if not occ["hasActions"]:
        thin_states = {k: v for k, v in occ["states"].items() if v < MIN_PER_CELL}
        return {
            "canAssessOverlap": False,
            "thinStates": thin_states,
            "verdict": (
                f"状态覆盖：{len(occ['states'])} 个格子有样本，"
                f"其中 {len(thin_states)} 个不足 {MIN_PER_CELL} 帧。\n"
                "**但这还不是重叠性。** 重叠性问的是「在这个局面下，"
                "不同的选择各有多少样本」，需要先定义动作。"
            ),
        }

    by_state: dict[str, dict[str, int]] = {}
    for key, count in occ["cells"].items():
        minute, gold, action = key.split("|")
        by_state.setdefault(f"{minute}|{gold}", {})[action] = count

    actions = sorted({k.split("|")[2] for k in occ["cells"]})
    empty: list[str] = []
    thin: list[str] = []
    for state, per_action in by_state.items():
        for action in actions:
            count = per_action.get(action, 0)
            if count == 0:
                empty.append(f"{state} × {action}")
            elif count < MIN_PER_CELL:
                thin.append(f"{state} × {action}（{count} 帧）")

    return {
        "canAssessOverlap": True,
        "actions": actions,
        "emptyCells": empty,
        "thinCells": thin,
        "verdict": _overlap_verdict(empty, thin, len(by_state) * len(actions)),
    }


def _overlap_verdict(empty: list[str], thin: list[str], total: int) -> str:
    if not empty and not thin:
        return "所有（状态 × 动作）格子都有足够样本，重叠性没问题。"
    lines = []
    if empty:
        lines.append(
            f"{len(empty)}/{total} 个格子是**空的** —— 这些局面下从来没人做过那个选择。\n"
            "  这类问题**再多数据也答不了**，因为反事实在数据里根本不存在。\n"
            "  唯一的出路是引入会做那些选择的样本（比如路人局），"
            "或者承认这些问题超出范围。"
        )
    if thin:
        lines.append(
            f"{len(thin)} 个格子样本太少（<{MIN_PER_CELL}）。\n"
            "  这类是**加数据能解决**的，属于方差问题。"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 需要多少

def required_games(
    target_pp: float = 3.0,
    frames_per_game: int = 20,
    strata: int = 20,
    actions: int = 4,
    per_cell: int = MIN_PER_CELL,
    action_instances_per_game: float = 4.0,
) -> dict[str, Any]:
    """粗算需要多少场。**这是数量级估计，不是精确要求。**

    两个层次差别巨大，而且差得比直觉大得多：

    - 训 V(s)：每一帧都是一个样本，几百场就够
    - 估 Q(s,a)：只有**真的发生了那个动作**的时刻才算一个样本

    第二点是最容易算错的地方，我第一版就算错了：
    默认拿「每场 20 帧」去除格子需求，得出「120 场就够」。
    **那是错的，而且错得很危险** —— 它会让人以为 Q(s,a) 比 V(s) 只贵一倍。

    真实情况是：像「入侵野区」这种动作，一场也就发生三五次。
    每场提供的**动作样本**是个位数，不是 20 个。
    所以 `action_instances_per_game` 必须单独给，默认按 4 次/场算。
    """
    # V(s)：按二项分布的置信区间宽度粗算
    # 半宽 ≈ 1.96 × sqrt(0.25/n)，解出 n
    half = target_pp / 100.0
    frames_for_v = math.ceil(0.25 * (1.96 / half) ** 2)
    games_for_v = math.ceil(frames_for_v / max(1, frames_per_game))

    # Q(s,a)：格子数 × 每格样本 = 需要多少个「动作实例」，
    # 再除以**每场能提供多少个动作实例**（不是每场多少帧！）
    cells = strata * actions
    instances_needed = cells * per_cell
    games_for_q = math.ceil(instances_needed / max(0.1, action_instances_per_game))

    return {
        "targetPP": target_pp,
        "vs": {"frames": frames_for_v, "games": games_for_v},
        "qsa": {
            "cells": cells,
            "actionInstancesNeeded": instances_needed,
            "instancesPerGame": action_instances_per_game,
            "games": games_for_q,
        },
        "ratio": round(games_for_q / max(1, games_for_v), 1),
        "note": (
            f"训 V(s) 到 ±{target_pp}pp 大约需要 {games_for_v} 场。\n"
            f"估 Q(s,a) 需要 {cells} 个（状态×动作）格子各 {per_cell} 个样本，"
            f"共 {instances_needed} 个**动作实例**；\n"
            f"按一场能观察到 {action_instances_per_game} 次该动作算，"
            f"大约需要 {games_for_q} 场 —— 是 V(s) 的 {games_for_q / max(1, games_for_v):.0f} 倍。\n"
            "关键在于：Q(s,a) 的样本单位是「动作发生了几次」，不是「有多少帧」。\n"
            "**而且这个数字的前提是每个格子都有人做过那个选择。**\n"
            "只用职业比赛时，很多格子是结构性空的，那部分不在这个数里。"
        ),
    }


# ---------------------------------------------------------------- 跨赛季

def season_stability(
    games_by_season: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """按赛季分别拟合，比较系数 —— 看老数据能不能合并进来。

    系数稳定 = 平滑漂移，可以合并 + 时间衰减权重。
    某个系数突变 = 结构断裂（比如暴君收益改了），
    **那一项的老数据不能用**，降权也救不了 —— 降权只是让错误的信号小声一点。
    """
    fitted: dict[str, Any] = {}
    refused: dict[str, str] = {}
    for season, games in games_by_season.items():
        try:
            fitted[season] = model.train(games)
        except model.TrainingRefused as err:
            refused[season] = str(err).split("\n")[0]

    if len(fitted) < 2:
        return {
            "comparable": False,
            "fitted": list(fitted),
            "refused": refused,
            "verdict": "至少要有两个赛季各自训得出来，才谈得上比较系数。",
        }

    keys = model.FEATURE_ORDER
    spread: dict[str, Any] = {}
    for key in keys:
        values = [m["weights"][key] for m in fitted.values()]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        deviation = math.sqrt(variance)
        # 相对离散度：标准差 / 平均绝对值
        scale = max(1e-6, sum(abs(v) for v in values) / len(values))
        spread[key] = {
            "mean": round(mean, 4),
            "sd": round(deviation, 4),
            "relative": round(deviation / scale, 3),
            "bySeason": {s: round(m["weights"][key], 4) for s, m in fitted.items()},
        }

    unstable = [k for k, v in spread.items() if v["relative"] > 0.5]
    return {
        "comparable": True,
        "seasons": list(fitted),
        "refused": refused,
        "spread": spread,
        "unstableFeatures": unstable,
        "verdict": _stability_verdict(unstable, spread),
    }


def _stability_verdict(unstable: list[str], spread: dict[str, Any]) -> str:
    if not unstable:
        return (
            "各赛季的系数都稳定，看起来是平滑漂移。\n"
            "可以合并，建议给老赛季一个时间衰减权重。"
        )
    names = "、".join(model.FEATURE_LABELS.get(k, k) for k in unstable)
    return (
        f"这几项在赛季之间变化很大：{names}。\n"
        "**这更像结构断裂而不是漂移** —— 可能是某个补丁改了对应机制。\n"
        "降权解决不了结构断裂：它只是让错误的信号小声一点，方向仍然是错的。\n"
        "建议：老赛季的数据只用于稳定的那些项，这几项只用当前赛季。"
    )


def time_decay_weight(season_index: int, half_life: float = 2.0) -> float:
    """时间衰减权重。season_index=0 是当前赛季。

    半衰期 2 个赛季意味着：两个赛季前的数据权重减半。
    **这个半衰期是拍脑袋定的**，应该用 season_stability() 的结果来调 ——
    系数越稳定，半衰期就可以越长。
    """
    return 0.5 ** (max(0, season_index) / max(0.1, half_life))


def summarize(occ: dict[str, Any], report: dict[str, Any],
              need: dict[str, Any]) -> list[str]:
    lines = [
        f"现有数据：{occ['games']} 场，{occ['usableFrames']} 帧可用"
        f"（{occ['skippedFrames']} 帧因为缺可信字段被跳过）。",
    ]
    lines.extend(report["verdict"].split("\n"))
    lines.extend(need["note"].split("\n"))
    if occ.get("actionNote"):
        lines.extend(occ["actionNote"].split("\n"))
    return lines
