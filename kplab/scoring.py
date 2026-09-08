"""把局势量化成分数 —— 以及它和因果推断是什么关系。

## 先回答那个问题：这两个不是同一类模型

用户提的思路是：把英雄强度、血量、技能CD、兵线、经济、装备各自量化成分数，
加起来得到「蓝方 80、红方 70」，再由分数推出分支胜率。

这叫**状态评估函数**，本质上是一个**手工设计的 V(s)**。
它和 Q(s,a) 因果推断回答的是**不同的问题**：

| | 分数模型 | 因果推断 |
| --- | --- | --- |
| 回答 | 「现在这个局面值多少」 | 「换个打法结果会怎样」 |
| 正确性来自 | 你定的规则对不对 | 数据里真实发生过什么 |
| 需要数据 | 很少，主要靠领域知识 | 很多，而且要重叠性 |
| 冷启动 | ✅ 今天就能跑 | ❌ 要几百上千场 |

如果再加上「打一架会变成什么样」的推演规则，那就变成
**评估函数 + 状态转移模型 = 前向推演**（象棋引擎、AlphaGo 的结构）。
那是第三种东西，不是因果推断。

## 两者的失败方式正好相反 —— 这是最要紧的一点

**分数模型的致命弱点：它只能反映你已有的认知。**

如果你的权重里写着「开龙值 5 分」，那模型算出来「开龙好」——
这不是证据，这是**把你的假设原样回声给你**。而且它是**自洽的**：
每一步推理都对得上，看不出哪里错了。
错误以系统性偏差的形式存在，不会体现在任何置信区间上。

**因果推断的致命弱点：重叠性。**
没人做过的选择答不了，再多数据也没用（见 `feasibility.py`）。

**但正因为如此，它们互补：**

分数模型能推演**数据里从没发生过的动作** —— 那正是因果推断的死穴。
因果推断能**推翻你的先验** —— 那正是分数模型做不到的。

## 所以正确的做法是合用，而且有现成的合法

1. 分数模型当**先验**，冷启动就能出结果
2. 用数据**校准**它：分数差 10 分，实际胜率真的更高吗？
3. **分歧的地方就是你学到东西的地方** —— 分数说 A 赢、数据说 B 赢，
   要么权重错了，要么有你没考虑到的因素

第 2 步是这个文件最重要的功能：**它让分数模型变成可证伪的**。
一个不能被证伪的分数模型，和占卜没有区别。

这也是统计上的标准做法（model-assisted / doubly robust）：
用结构化的先验降低方差，用数据纠正偏差。

## 一个必须先说清楚的现实问题

用户列的那些量，**大部分现在拿不到**。见 `COMPONENTS` 里的 `obtainable` 字段：
血量和技能CD在转播画面上根本没有稳定显示，兵线状态要另做一套识别。
在这些拿不到之前，分数模型只能用现有的那几项 ——
**用一个缺了一半输入的分数模型去做判断，比没有模型更危险。**
"""

from __future__ import annotations

import math
from typing import Any, Callable

from . import model, schema
from .rules import BLUE, RED

# ---------------------------------------------------------------- 分项

# 每一项都要说清楚三件事：权重多少、凭什么、以及**现在拿不拿得到**。
#
# `obtainable` 是这张表里最重要的一列。用户提的很多量在转播画面上
# 根本没有稳定显示，写进模型只会得到一个大面积缺失的分数 ——
# 而缺失的分数和低分长得一模一样。
COMPONENTS: dict[str, dict[str, Any]] = {
    "economy": {
        "label": "经济优势",
        "weight": 1.0,
        "why": "最直接的综合国力指标，也是唯一在转播里长期可见的。",
        "needs": "双方总经济",
        "obtainable": True,
        "how": "记分板 OCR / 人工标注",
    },
    "objectives": {
        "label": "中立资源",
        "weight": 0.8,
        "why": "暴君/主宰/风暴龙王各自加权，五种分开算（见 model.OBJECTIVE_WEIGHTS）。",
        "needs": "各类资源的击杀计数",
        "obtainable": True,
        "how": "events_cv 看图标消失 + announce 看播报定归属",
    },
    "towers": {
        "label": "推塔进度",
        "weight": 0.9,
        "why": "不可逆的地图控制，比等量经济更值钱。",
        "needs": "双方已拆塔数",
        "obtainable": True,
        "how": "events_cv 看小地图图标消失",
    },
    "levels": {
        "label": "等级差",
        "weight": 0.4,
        "why": "通常已反映在经济里，权重给小以免重复计算。",
        "needs": "十个人的等级",
        "obtainable": True,
        "how": "头像旁的小数字 OCR / 人工标注",
    },
    "aliveCount": {
        "label": "存活人数差",
        "weight": 1.2,
        "why": "**当下战斗力最强的单一指标。** 五打三和五打五是完全不同的局面。",
        "needs": "谁活着（头像是否变灰）",
        "obtainable": True,
        "how": "events_cv 的头像变灰检测",
    },
    # ---- 下面这些是用户提到但现在拿不到的 ----
    "heroPower": {
        "label": "英雄强度（阵容期望）",
        "weight": 0.7,
        "why": "阵容本身有强弱和时间曲线（前期强势 vs 后期强势）。",
        "needs": "十个英雄的身份 + 一份英雄强度表",
        "obtainable": False,
        "how": "需要英雄识别（小地图认不出是谁）+ 一份随版本更新的强度表",
        "blocker": "两样都没有。英雄识别要头像模板库，强度表要人工维护并随版本更新。",
    },
    "health": {
        "label": "血量状态",
        "weight": 1.0,
        "why": "残血的五个人和满血的五个人战斗力差得远。",
        "needs": "十个人的当前血量",
        "obtainable": False,
        "how": "转播画面上没有稳定显示十个人血量的地方",
        "blocker": "**根本看不到。** 只有被导播选中的那个人的血条可见。",
    },
    "cooldowns": {
        "label": "技能 / 召唤师技能 CD",
        "weight": 0.9,
        "why": "闪现和大招在不在，直接决定敢不敢开团。用户的例子里正是这一项。",
        "needs": "十个人的技能冷却状态",
        "obtainable": False,
        "how": "转播画面上没有",
        "blocker": "**根本看不到。** 理论上可以从「上次使用时间 + CD 长度」推算，"
                   "但那需要先能检测技能释放，工作量极大且误差会累积。",
    },
    "waveState": {
        "label": "兵线状态",
        "weight": 0.6,
        "why": "兵线在谁的半区、有没有攻城车，决定了推进窗口。",
        "needs": "三路兵线位置",
        "obtainable": False,
        "how": "小地图上兵线是很小的点，和英雄标记混在一起",
        "blocker": "需要单独的识别方案，本项目没做。",
    },
    "keyItems": {
        "label": "关键装备",
        "weight": 0.8,
        "why": "某几件装备的成型是质变点，不是线性的经济。",
        "needs": "十个人的装备栏",
        "obtainable": False,
        "how": "转播偶尔展示，不连续",
        "blocker": "只在导播切到装备面板时可见，做不了逐帧。",
    },
}


def obtainable_components() -> list[str]:
    return [k for k, v in COMPONENTS.items() if v["obtainable"]]


def blocked_components() -> list[str]:
    return [k for k, v in COMPONENTS.items() if not v["obtainable"]]


def coverage_note() -> str:
    got, missing = obtainable_components(), blocked_components()
    total_weight = sum(COMPONENTS[k]["weight"] for k in COMPONENTS)
    got_weight = sum(COMPONENTS[k]["weight"] for k in got)
    return (
        f"分项共 {len(COMPONENTS)} 个，现在拿得到 {len(got)} 个"
        f"（按权重算占 {got_weight / total_weight:.0%}）。\n"
        f"拿不到的：{'、'.join(COMPONENTS[k]['label'] for k in missing)}。\n"
        "**其中血量和技能CD在转播画面上根本没有稳定显示** —— "
        "这不是识别难度问题，是信息本身不存在。\n"
        "用户举的例子（「减去技能CD不健康」）正好落在拿不到的那一半里。"
    )


# ---------------------------------------------------------------- 打分

def _component_values(frame: dict[str, Any]) -> dict[str, tuple[float, float] | None]:
    """算出每个分项的原始值（蓝、红）。拿不到的返回 None。"""
    teams = frame.get("teams") or {}
    blue, red = teams.get(str(BLUE)) or {}, teams.get(str(RED)) or {}
    out: dict[str, tuple[float, float] | None] = {}

    def pair(key):
        b, r = blue.get(key), red.get(key)
        if not schema.trusted(b) or not schema.trusted(r):
            return None
        return float(schema.get(b)), float(schema.get(r))

    gold = pair("totalGold")
    out["economy"] = (gold[0] / 1000.0, gold[1] / 1000.0) if gold else None

    towers = pair("towers")
    out["towers"] = towers

    levels = pair("totalLevel")
    out["levels"] = levels

    objective_blue = objective_red = 0.0
    complete = True
    for key, weight in model.OBJECTIVE_WEIGHTS.items():
        got = pair(key)
        if got is None:
            complete = False
            break
        objective_blue += got[0] * weight
        objective_red += got[1] * weight
    out["objectives"] = (objective_blue, objective_red) if complete else None

    alive_blue = alive_red = 0
    known = 0
    for player in frame.get("players") or []:
        entry = player.get("alive")
        if not schema.trusted(entry):
            continue
        known += 1
        if schema.get(entry):
            if player.get("teamId") == BLUE:
                alive_blue += 1
            else:
                alive_red += 1
    # 十个人的存活状态必须全知道，否则「3 人存活」可能只是没读到
    out["aliveCount"] = (float(alive_blue), float(alive_red)) if known == 10 else None

    for key in blocked_components():
        out[key] = None
    return out


def score_frame(frame: dict[str, Any],
                weights: dict[str, float] | None = None) -> dict[str, Any]:
    """给一帧打分。返回双方总分 + 每一项的拆解。

    **必须看 `coverage`。** 只有 3 项可用时算出来的 80 分，
    和 8 项都可用时的 80 分不是一回事 —— 直接比较两者是错的。
    """
    weights = weights or {k: v["weight"] for k, v in COMPONENTS.items()}
    values = _component_values(frame)

    breakdown: list[dict[str, Any]] = []
    blue_total = red_total = 0.0
    used_weight = 0.0

    # 两个分母，含义完全不同，混用会让分数永远不可用：
    #
    #   obtainable_weight —— 「现在这条管线**能拿到**的那些分项」的总权重
    #   full_weight       —— 包含血量、技能CD 这些**永远拿不到**的分项
    #
    # coverage 必须除以前者。除以后者的话，因为有 48% 的权重结构性缺失，
    # coverage 永远到不了 60%，`comparable` 永远是 False，
    # **这个模型一帧都算不出来**。第一版就是这么写的，跑校准时零行输出才发现。
    obtainable_weight = sum(abs(weights.get(k, 0.0)) for k in obtainable_components())
    full_weight = sum(abs(weights.get(k, 0.0)) for k in COMPONENTS)

    for key, spec in COMPONENTS.items():
        weight = weights.get(key, 0.0)
        pair = values.get(key)
        if pair is None:
            breakdown.append({"key": key, "label": spec["label"], "available": False,
                              "structural": not spec["obtainable"],
                              "reason": spec.get("blocker") or "这一帧没有可信数据"})
            continue
        blue_value, red_value = pair
        blue_total += weight * blue_value
        red_total += weight * red_value
        used_weight += abs(weight)
        breakdown.append({
            "key": key, "label": spec["label"], "available": True,
            "weight": weight, "blue": round(blue_value, 3), "red": round(red_value, 3),
            "contribution": round(weight * (blue_value - red_value), 3),
        })

    coverage = used_weight / obtainable_weight if obtainable_weight else 0.0
    ceiling = obtainable_weight / full_weight if full_weight else 0.0
    return {
        "blue": round(blue_total, 3),
        "red": round(red_total, 3),
        "diff": round(blue_total - red_total, 3),
        "breakdown": breakdown,
        # 这一帧拿到了「能拿到的部分」里的多少
        "coverage": round(coverage, 3),
        # 整个模型里有多少是**永远**拿得到的（血量、技能CD 拿不到，所以不到 1）
        "ceiling": round(ceiling, 3),
        "comparable": coverage >= 0.6,
        "note": None if coverage >= 0.6 else (
            f"能拿到的分项里只有 {coverage:.0%} 有数据，这一帧**不能和其他帧比较** —— "
            "缺项和劣势在分数上长得一模一样。"
        ),
        "ceilingNote": (
            f"即使全部读到，这个分数也只覆盖了完整模型的 {ceiling:.0%} —— "
            "血量、技能CD 这些在转播画面上根本没有。"
            "**所以它反映的是「资源和进度」，不是「当下战斗力」。**"
        ),
    }


# ---------------------------------------------------------------- 校准

# 分数差分桶，用来画可靠性曲线
DIFF_BINS = ((-999, -20), (-20, -10), (-10, -3), (-3, 3), (3, 10), (10, 20), (20, 999))


def calibrate(games: list[dict[str, Any]],
              weights: dict[str, float] | None = None) -> dict[str, Any]:
    """**让分数模型变成可证伪的。**

    问一个很简单的问题：分数差落在某一档时，蓝方实际赢了百分之多少？

    如果分数差 +20 的局面实际只赢一半，那这套权重就是错的 ——
    不管它在纸面上多么有道理。

    这是分数模型和占卜的区别：**能被数据打脸。**
    而且这个检验很便宜，几十场就能看出权重是不是离谱，
    远比做完整的因果推断省。
    """
    buckets: dict[tuple[int, int], dict[str, int]] = {
        b: {"n": 0, "blueWins": 0} for b in DIFF_BINS
    }
    skipped = 0

    for built in games:
        label = (built.get("meta") or {}).get("blueWin")
        if label is None:
            continue
        for frame in built.get("frames") or []:
            scored = score_frame(frame, weights)
            if not scored["comparable"]:
                skipped += 1
                continue
            diff = scored["diff"]
            bucket = next((b for b in DIFF_BINS if b[0] <= diff < b[1]), None)
            if bucket is None:
                continue
            buckets[bucket]["n"] += 1
            if label:
                buckets[bucket]["blueWins"] += 1

    rows = []
    for (lo, hi), stats in buckets.items():
        if not stats["n"]:
            continue
        rate = stats["blueWins"] / stats["n"]
        rows.append({
            "range": f"{lo}~{hi}" if abs(lo) < 900 and abs(hi) < 900
                     else (f"<{hi}" if abs(lo) >= 900 else f">{lo}"),
            "low": lo, "high": hi,
            "frames": stats["n"],
            "blueWinRate": round(rate, 3),
        })
    rows.sort(key=lambda r: r["low"])

    return {
        "rows": rows,
        "skippedFrames": skipped,
        "monotone": _is_monotone(rows),
        "verdict": _calibration_verdict(rows),
    }


def _is_monotone(rows: list[dict[str, Any]]) -> bool:
    """分数差越大，胜率应该越高。不单调就说明权重有问题。"""
    rates = [r["blueWinRate"] for r in rows if r["frames"] >= 20]
    return all(a <= b + 1e-9 for a, b in zip(rates, rates[1:]))


def _calibration_verdict(rows: list[dict[str, Any]]) -> str:
    solid = [r for r in rows if r["frames"] >= 20]
    if len(solid) < 3:
        return (
            "样本太少，校准结果不可信。至少要有 3 个分数档各 20 帧以上。\n"
            "**在校准之前，这套权重只是一组假设。**"
        )
    if not _is_monotone(rows):
        return (
            "**分数差和胜率不单调** —— 分数更高的局面反而赢得更少。\n"
            "这套权重有问题：某一项的方向或大小定反了。\n"
            "看 disagreements() 找出具体是哪些局面对不上。"
        )
    lowest, highest = solid[0]["blueWinRate"], solid[-1]["blueWinRate"]
    if highest - lowest < 0.25:
        return (
            f"单调，但区分度很弱（最低档 {lowest:.0%} → 最高档 {highest:.0%}）。\n"
            "分数在动，胜率没怎么动 —— 说明权重抓的不是决定胜负的东西。"
        )
    return (
        f"单调且有区分度（{lowest:.0%} → {highest:.0%}）。\n"
        "这套权重通过了基本检验。**但通过校准只说明它不离谱，"
        "不说明每一项的权重都对** —— 几组不同的权重可能给出同样好的校准曲线。"
    )


def disagreements(games: list[dict[str, Any]],
                  weights: dict[str, float] | None = None,
                  threshold: float = 15.0,
                  limit: int = 30) -> list[dict[str, Any]]:
    """找出「分数说这边稳赢、结果那边赢了」的局面。

    **这些是最有价值的样本。** 分数和数据一致的地方你什么也学不到；
    分歧的地方要么是权重错了，要么是有你没考虑进去的因素。
    """
    out: list[dict[str, Any]] = []
    for built in games:
        meta = built.get("meta") or {}
        label = meta.get("blueWin")
        if label is None:
            continue
        for frame in built.get("frames") or []:
            scored = score_frame(frame, weights)
            if not scored["comparable"]:
                continue
            diff = scored["diff"]
            # 分数强烈倾向一方，结果却相反
            if diff >= threshold and not label:
                loser, winner = "蓝方", "红方"
            elif diff <= -threshold and label:
                loser, winner = "红方", "蓝方"
            else:
                continue
            top = sorted((b for b in scored["breakdown"] if b.get("available")),
                         key=lambda b: -abs(b["contribution"]))[:3]
            out.append({
                "gameId": meta.get("gameId"),
                "clock": frame.get("clock"),
                "scoreDiff": diff,
                "scoreFavoured": loser,
                "actualWinner": winner,
                "topContributors": top,
                "coverage": scored["coverage"],
            })
    out.sort(key=lambda d: -abs(d["scoreDiff"]))
    return out[:limit]


# ---------------------------------------------------------------- 混合

def fit_weights(games: list[dict[str, Any]],
                prior: dict[str, float] | None = None,
                prior_strength: float = 5.0) -> dict[str, Any]:
    """从数据里学权重，但**以你的先验为起点并被它拉住**。

    这是分数模型和数据驱动的结合点，统计上叫「带先验的正则化」：

    - 数据少 → 结果接近你定的权重（不会学出一堆噪音）
    - 数据多 → 数据可以把权重拉离先验（能推翻你的假设）

    `prior_strength` 越大，越相信你的先验。
    **这个参数本身没有正确值** —— 它表达的是「你有多信自己」。
    """
    prior = prior or {k: v["weight"] for k, v in COMPONENTS.items()}

    # 先看每个分项在数据里到底出现了多少次。
    #
    # 第一版要求「所有能拿到的分项都齐全」才算一帧 —— 结果一个分项缺失
    # 就把全部 1800 帧废掉了。太严了。
    # 但反过来随便丢分项也不行：丢掉一项，它的效应会被剩下的项**吸收**，
    # 剩下那些权重就偏了（遗漏变量偏差）。所以丢了什么必须明说。
    candidates = obtainable_components()
    present: dict[str, int] = {k: 0 for k in candidates}
    total_frames = 0
    per_frame: list[dict[str, tuple[float, float] | None]] = []
    labels: list[int] = []

    for built in games:
        label = (built.get("meta") or {}).get("blueWin")
        if label is None:
            continue
        for frame in built.get("frames") or []:
            values = _component_values(frame)
            total_frames += 1
            per_frame.append(values)
            labels.append(1 if label else 0)
            for key in candidates:
                if values.get(key) is not None:
                    present[key] += 1

    if not total_frames:
        raise model.TrainingRefused("没有带胜负标签的帧，拟合不了。")

    # 出现率太低的分项直接排除 —— 留着它会把大部分帧废掉
    MIN_PRESENCE = 0.5
    usable = [k for k in candidates if present[k] / total_frames >= MIN_PRESENCE]
    excluded = [k for k in candidates if k not in usable]

    if not usable:
        raise model.TrainingRefused(
            "没有任何分项在半数以上的帧里有数据，拟合不了。\n"
            "**在数据够之前，就用你定的先验权重，别去拟合。**"
        )

    rows: list[tuple[dict[str, float], int]] = []
    for values, label in zip(per_frame, labels):
        if any(values.get(k) is None for k in usable):
            continue
        rows.append(({k: values[k][0] - values[k][1] for k in usable}, label))

    if len(rows) < model.MIN_FRAMES:
        raise model.TrainingRefused(
            f"只有 {len(rows)} 帧的 {len(usable)} 个分项齐全，少于 {model.MIN_FRAMES} 帧。\n"
            "**在数据够之前，就用你定的先验权重，别去拟合。**"
        )

    weights = {k: prior.get(k, 0.0) for k in usable}
    bias = 0.0
    rate = 0.02
    n = len(rows)
    for _ in range(2000):
        grad = {k: 0.0 for k in usable}
        grad_bias = 0.0
        for features, label in rows:
            z = bias + sum(weights[k] * features[k] for k in usable)
            error = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z)))) - label
            for k in usable:
                grad[k] += error * features[k]
            grad_bias += error
        for k in usable:
            # 往先验拉回去的一项 —— 这就是「先验强度」起作用的地方
            pull = prior_strength * (weights[k] - prior.get(k, 0.0)) / n
            weights[k] -= rate * (grad[k] / n + pull)
        bias -= rate * grad_bias / n

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "prior": {k: round(prior.get(k, 0.0), 4) for k in usable},
        "shift": {k: round(weights[k] - prior.get(k, 0.0), 4) for k in usable},
        "frames": n,
        "fittedComponents": usable,
        "excludedComponents": excluded,
        "presence": {k: round(present[k] / total_frames, 3) for k in candidates},
        "priorStrength": prior_strength,
        "exclusionWarning": (
            f"这些分项因为出现率不足 {MIN_PRESENCE:.0%} 被排除："
            f"{'、'.join(COMPONENTS[k]['label'] for k in excluded)}。\n"
            "**它们的效应会被留下的项吸收** —— 剩下这些权重因此是偏的，"
            "别把它们当成「该项的真实价值」。"
        ) if excluded else "",
        "note": (
            "shift 是数据把权重拉离先验的幅度。\n"
            "某一项 shift 很大 → 你的直觉和数据不一致，值得回头想想为什么。\n"
            "**但也可能是那一项和别的项共线** —— 经济和等级高度相关，"
            "拟合时权重会在两者之间乱跑，单看一项的 shift 会误导。"
        ),
    }


def compare_paradigms() -> dict[str, Any]:
    """分数模型 vs 因果推断：各自答什么、各自怎么错。"""
    return {
        "scoring": {
            "answers": "现在这个局面值多少分",
            "type": "状态评估函数（手工设计的 V(s)）",
            "strengths": [
                "冷启动 —— 今天就能跑，不需要数据",
                "**能推演数据里从没发生过的动作** —— 这正是因果推断的死穴",
                "完全可解释，每一分从哪来都能拆开",
                "数据少时方差小",
            ],
            "weaknesses": [
                "**只能反映你已有的认知** —— 权重说开龙值 5 分，"
                "模型就说开龙好，那是假设的回声不是证据",
                "错误是系统性的，而且**自洽**，不体现在任何置信区间上",
                "发现不了你没想到的因素",
            ],
        },
        "causal": {
            "answers": "换个打法，结果会怎么变",
            "type": "因果效应估计（Q(s,a)）",
            "strengths": [
                "**能推翻你的先验** —— 这是分数模型做不到的",
                "不确定性会诚实地体现在置信区间上",
            ],
            "weaknesses": [
                "**重叠性** —— 没人做过的选择答不了，再多数据也没用",
                "需要大量样本，且要有动作定义",
                "选择偏差不处理好就是相关性冒充因果",
            ],
        },
        "verdict": (
            "不是二选一。两者的失败方式正好相反，所以互补：\n"
            "  · 分数模型当先验，冷启动就有输出\n"
            "  · 用数据校准它（calibrate）—— 这一步让它变成可证伪的\n"
            "  · **分歧的地方就是学到东西的地方**（disagreements）\n"
            "  · 数据够了再用带先验的拟合（fit_weights）慢慢把权重交给数据\n"
            "\n"
            "用户举的例子里还隐含了第三样东西：\n"
            "「红方可以凭更高战斗力通过打架得分」—— 这是**状态转移模型**。\n"
            "评估函数 + 转移模型 = 前向推演（象棋引擎的结构），\n"
            "那是另一套假设，每一条都可能悄悄出错，需要单独验证。"
        ),
    }
