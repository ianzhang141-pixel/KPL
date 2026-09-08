"""胜率模型 V(s) —— 以及一条不许越过的界线。

## 先说清楚这个文件里有什么、没有什么

**有：** 特征抽取、一个纯标准库写的逻辑回归、训练时的拒绝条件、
以及一条**明确标注为「假设值」的基线曲线**。

**没有：** 任何在真实王者荣耀数据上训练出来的模型。一个都没有。

## 为什么会有「基线曲线」这种东西

用户想看到产品长什么样：上传录像 → 出胜率曲线 → 点拐点 → 看分析。
但真实数据是零，训不出模型，于是有两条路：

  A. 什么都不显示，等几个月后数据够了再说
  B. 先用一条**写死的、公开承认是假设的**曲线，把整条链路跑通

选 B，理由和 LOL 项目路线图里的 AI2 一样：
**在花几个月采数据之前，先看到产品长什么样，是最降风险的一步。**

但 B 有一个巨大的风险：**基线曲线会被当成模型输出。**
一旦有人拿着这条曲线去做决策、去跟战队汇报，这个项目就变成了骗人的东西。

所以这里的防线是硬的：

1. 基线的每个系数都写在 `BASELINE_PRIORS` 里，附带「凭什么这么定」
2. 任何一条曲线都带 `kind` 字段：`"baseline"`（假设）或 `"trained"`（学出来的）
3. `kind="baseline"` 时 `trained=False`、`accuracy=None` ——
   **不给准确率，因为没有准确率**
4. 前端必须把两者画成不同的样子（虚线 vs 实线 + 警告横幅）
5. `train()` 在数据不够时**直接拒绝并报错**，不会悄悄退回基线

## 训练的拒绝条件（抄 LOL 项目做对的地方）

LOL 项目的 `train.py` 在数据不足（<500 行 / <20 场）时直接拒绝训练并报错。
那是那个仓库里少数几个明确做对的设计之一，这里照抄并加严：

  · 场次太少 → 拒绝
  · 可用帧太少 → 拒绝
  · **只有赢的比赛或只有输的比赛 → 拒绝**（这是选择偏差，再多数据也救不了）
  · 可信字段比例太低 → 拒绝（拿一堆 OCR 猜出来的数训练，学到的是噪音）
"""

from __future__ import annotations

import math
from typing import Any

from . import schema
from .rules import BLUE, RED

# ---------------------------------------------------------------- 特征

# 特征名 → 人话。前端和报告都用这个。
FEATURE_LABELS = {
    "minute": "比赛时间（分钟）",
    "goldDiffK": "经济差（千）",
    "killDiff": "击杀差",
    "towerDiff": "推塔差",
    "levelDiff": "总等级差",
    "objectiveScore": "中立资源分",
}

# 中立资源的权重。**五种资源分开算，权重不同** ——
# 这正是 LOL 那个「先锋和幼虫被算成大龙」的 bug 想避免的事。
# 这些权重是**凭对游戏的理解定的，不是学出来的**，所以列在这里供人核对。
OBJECTIVE_WEIGHTS = {
    "tyrants": 1.0,           # 暴君：前期经济
    "darkTyrants": 2.0,       # 黑暗暴君：后期强化，收益明显更大
    "overlords": 1.5,         # 主宰：推进资源
    "stormDragons": 3.0,      # 风暴龙王：通常直接决定一波
    "prophetOverlords": 2.5,  # 先知主宰
}


def _diff(frame: dict[str, Any], key: str) -> float | None:
    """取蓝方减红方的差。**任一方未知就返回 None，绝不当 0 处理。**"""
    blue = frame["teams"][str(BLUE)].get(key)
    red = frame["teams"][str(RED)].get(key)
    if not schema.trusted(blue) or not schema.trusted(red):
        return None
    return float(schema.get(blue)) - float(schema.get(red))


def _objective_score(frame: dict[str, Any]) -> float | None:
    """加权的中立资源差。五种分开取，权重见 OBJECTIVE_WEIGHTS。"""
    total = 0.0
    for key, weight in OBJECTIVE_WEIGHTS.items():
        value = _diff(frame, key)
        if value is None:
            return None
        total += value * weight
    return total


def extract_features(frame: dict[str, Any]) -> dict[str, float] | None:
    """把一帧 State 变成特征。

    **只用 `schema.trusted()` 的字段。** 低置信度的值是猜的，
    喂进模型学到的是识别误差的规律，不是游戏的规律。

    **绝不读 `meta.blueWin`** —— 那是标签。
    **绝不读后面的帧** —— 那是未来信息。

    可信字段不够时返回 None，表示「这一帧没法算特征」，
    而不是用 0 填上凑一个数。
    """
    gold = _diff(frame, "totalGold")
    kills = _diff(frame, "kills")
    towers = _diff(frame, "towers")
    levels = _diff(frame, "totalLevel")
    objectives = _objective_score(frame)

    # 经济差是最核心的一个，没有它这一帧就不要了
    if gold is None:
        return None

    return {
        "minute": float(frame.get("minute") or 0.0),
        "goldDiffK": gold / 1000.0,
        "killDiff": kills if kills is not None else 0.0,
        "towerDiff": towers if towers is not None else 0.0,
        "levelDiff": levels if levels is not None else 0.0,
        "objectiveScore": objectives if objectives is not None else 0.0,
    }


# ---------------------------------------------------------------- 基线

# 每一条都必须写清楚「凭什么定这个数」。
# 这些是**假设，不是从数据里学出来的**，任何时候都不许说成模型参数。
BASELINE_PRIORS: dict[str, dict[str, Any]] = {
    "goldDiffK": {
        "weight": 0.42,
        "why": "领先 1000 经济大约对应胜率 +10 个点。凭对 MOBA 的一般理解定的，"
               "王者荣耀节奏比 LOL 快、经济差扩大得更猛，这个数很可能偏小。",
    },
    "killDiff": {
        "weight": 0.10,
        "why": "击杀和经济高度相关，权重给小以免重复计算同一件事。",
    },
    "towerDiff": {
        "weight": 0.22,
        "why": "推塔是不可逆的地图控制，比等量经济更值钱。",
    },
    "objectiveScore": {
        "weight": 0.15,
        "why": "中立资源的加权分，权重见 OBJECTIVE_WEIGHTS。",
    },
    "levelDiff": {
        "weight": 0.05,
        "why": "等级差通常已经反映在经济差里，给很小的权重。",
    },
    "minuteScale": {
        "weight": 0.035,
        "why": "**时间越晚，同样的领先越难被翻盘。**"
               "所以上面所有权重都会乘上 (1 + 0.035 × 分钟)。"
               "这是这条基线里最像样的一个结构，也最可能是对的。",
    },
}

BASELINE_NOTE = (
    "这是一条**假设出来的**曲线，不是从数据里学出来的。\n"
    "系数写在 model.BASELINE_PRIORS 里，每条都注明了凭什么这么定。\n"
    "它可以用来看产品长什么样、看链路通不通，"
    "**不能用来判断任何一场比赛谁优谁劣，更不能拿去跟人汇报。**"
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def baseline_probability(features: dict[str, float]) -> float:
    """基线胜率。**假设值，不是预测值。**"""
    time_scale = 1.0 + BASELINE_PRIORS["minuteScale"]["weight"] * features.get("minute", 0.0)
    total = 0.0
    for key in ("goldDiffK", "killDiff", "towerDiff", "objectiveScore", "levelDiff"):
        total += BASELINE_PRIORS[key]["weight"] * features.get(key, 0.0)
    return _sigmoid(total * time_scale)


# ---------------------------------------------------------------- 训练

MIN_GAMES = 30
MIN_FRAMES = 400
MIN_TRUSTED_RATIO = 0.5


class TrainingRefused(RuntimeError):
    """数据不够就拒绝训练，不产出一个看起来能用的坏模型。"""


def _check_trainable(rows: list[tuple[dict[str, float], int]], games: int,
                     trusted_ratio: float) -> None:
    problems: list[str] = []
    if games < MIN_GAMES:
        problems.append(f"只有 {games} 场比赛，少于 {MIN_GAMES} 场。")
    if len(rows) < MIN_FRAMES:
        problems.append(f"只有 {len(rows)} 帧可用，少于 {MIN_FRAMES} 帧。")

    wins = sum(1 for _, label in rows if label == 1)
    losses = len(rows) - wins
    if rows and (wins == 0 or losses == 0):
        problems.append(
            f"样本里{'全是赢的' if losses == 0 else '全是输的'}比赛。"
            "这是严重的选择偏差 —— 模型会学到「所有局面都会赢」，"
            "**再多数据也救不了，必须补另一类比赛。**"
        )
    if trusted_ratio < MIN_TRUSTED_RATIO:
        problems.append(
            f"可信字段只占 {trusted_ratio * 100:.0f}%，低于 {MIN_TRUSTED_RATIO * 100:.0f}%。"
            "拿一堆识别猜出来的数训练，学到的是识别误差的规律，不是游戏的规律。"
        )

    if problems:
        raise TrainingRefused(
            "数据不够，拒绝训练。\n" + "\n".join(f"  · {p}" for p in problems)
            + "\n\n这不是 bug。训一个数据不足的模型出来，"
              "它会给出看起来很像那么回事的曲线，而那条曲线是没有意义的 ——"
              "\n那比没有曲线更糟糕。"
        )


FEATURE_ORDER = ("goldDiffK", "killDiff", "towerDiff", "levelDiff", "objectiveScore", "minute")


def train(
    games: list[dict[str, Any]],
    steps: int = 3000,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """在若干场 State 上训练逻辑回归。纯标准库的梯度下降。

    games 是 `state.build()` 的输出列表。
    数据不够会抛 `TrainingRefused` —— 这是设计，不是缺陷。
    """
    rows: list[tuple[dict[str, float], int]] = []
    trusted_sum = 0.0
    trusted_count = 0

    for built in games:
        label = (built.get("meta") or {}).get("blueWin")
        if label is None:
            continue                      # 没有胜负标签的比赛训不了
        for frame in built.get("frames") or []:
            features = extract_features(frame)
            if features is None:
                continue
            rows.append((features, 1 if label else 0))
            trusted_sum += (frame.get("coverage") or {}).get("trustedRatio", 0.0)
            trusted_count += 1

    ratio = trusted_sum / trusted_count if trusted_count else 0.0
    _check_trainable(rows, len(games), ratio)

    # 特征标准化：经济差以千为单位、时间以分钟为单位，量纲差得远，
    # 不归一化梯度下降会歪得厉害
    means = {k: sum(f[k] for f, _ in rows) / len(rows) for k in FEATURE_ORDER}
    stds = {}
    for k in FEATURE_ORDER:
        var = sum((f[k] - means[k]) ** 2 for f, _ in rows) / len(rows)
        stds[k] = math.sqrt(var) or 1.0

    weights = {k: 0.0 for k in FEATURE_ORDER}
    bias = 0.0
    n = len(rows)
    for _ in range(steps):
        grad = {k: 0.0 for k in FEATURE_ORDER}
        grad_bias = 0.0
        for features, label in rows:
            z = bias + sum(weights[k] * (features[k] - means[k]) / stds[k] for k in FEATURE_ORDER)
            error = _sigmoid(z) - label
            for k in FEATURE_ORDER:
                grad[k] += error * (features[k] - means[k]) / stds[k]
            grad_bias += error
        for k in FEATURE_ORDER:
            weights[k] -= learning_rate * grad[k] / n
        bias -= learning_rate * grad_bias / n

    correct = 0
    for features, label in rows:
        z = bias + sum(weights[k] * (features[k] - means[k]) / stds[k] for k in FEATURE_ORDER)
        if (1 if _sigmoid(z) >= 0.5 else 0) == label:
            correct += 1

    return {
        "kind": "trained",
        "weights": weights,
        "bias": bias,
        "means": means,
        "stds": stds,
        "games": len(games),
        "frames": n,
        "trainAccuracy": round(correct / n, 4),
        "trustedRatio": round(ratio, 3),
        "warning": (
            "训练集上的准确率**不是**真实准确率。"
            "没有做留出验证，这个数只能说明模型有没有学进去，不能说明它准不准。"
        ),
    }


def trained_probability(model: dict[str, Any], features: dict[str, float]) -> float:
    z = model["bias"] + sum(
        model["weights"][k] * (features[k] - model["means"][k]) / model["stds"][k]
        for k in FEATURE_ORDER
    )
    return _sigmoid(z)


# ---------------------------------------------------------------- 曲线

def win_curve(built: dict[str, Any], model: dict[str, Any] | None = None) -> dict[str, Any]:
    """算出一场比赛的胜率曲线。

    没有训练好的模型时，用基线，并把 `kind` 标成 `"baseline"`。
    **前端必须据此把两者画成不同的样子。**
    """
    kind = "trained" if model else "baseline"
    points: list[dict[str, Any]] = []
    skipped = 0

    for frame in built.get("frames") or []:
        features = extract_features(frame)
        if features is None:
            skipped += 1
            continue        # 可信字段不够，这一帧算不出胜率 —— 不猜
        probability = (trained_probability(model, features) if model
                       else baseline_probability(features))
        points.append({
            "frameIndex": frame["frameIndex"],
            "atSec": frame["atSec"],
            "clock": frame["clock"],
            "minute": frame["minute"],
            "blueWinProbability": round(probability, 4),
            "features": {k: round(v, 3) for k, v in features.items()},
            "coverage": (frame.get("coverage") or {}).get("trustedRatio", 0.0),
            "events": frame.get("recentEvents") or [],
        })

    return {
        "kind": kind,
        "trained": kind == "trained",
        "accuracy": (model or {}).get("trainAccuracy") if model else None,
        "points": points,
        "skippedFrames": skipped,
        "note": BASELINE_NOTE if kind == "baseline" else (model or {}).get("warning", ""),
        "meta": built.get("meta") or {},
    }


# ---------------------------------------------------------------- 拐点

# 胜率变化超过这么多个百分点，才算一个拐点。
# 3 个点是个起点，不是查证过的阈值 —— 真实数据到位后要重调。
INFLECTION_THRESHOLD = 0.03


def find_inflections(curve: dict[str, Any],
                     threshold: float = INFLECTION_THRESHOLD) -> list[dict[str, Any]]:
    """找出胜率曲线上变化最大的那些点。

    **这一步只回答「哪里出事了」，不回答「谁做错了」。**

    LOL 项目的路线图里专门强调过这个区分：
    14:32 胜率掉 6 个点是事实，但原因可能是你入侵被抓、
    也可能是下路同时送了两个头、还可能是对面中单一个越塔操作。
    **曲线本身分不清。** 从「哪里出事」到「哪里做错」，
    中间隔着 Q(s,a) 和归因，那两样都还没做。
    """
    points = curve.get("points") or []
    out: list[dict[str, Any]] = []

    for i in range(1, len(points)):
        previous, current = points[i - 1], points[i]
        delta = current["blueWinProbability"] - previous["blueWinProbability"]
        if abs(delta) < threshold:
            continue
        out.append({
            "index": i,
            "frameIndex": current["frameIndex"],
            "atSec": current["atSec"],
            "clock": current["clock"],
            "fromProbability": previous["blueWinProbability"],
            "toProbability": current["blueWinProbability"],
            "deltaPP": round(delta * 100, 1),
            "direction": "蓝方" if delta > 0 else "红方",
            "events": current.get("events") or [],
            "featureChanges": _feature_changes(previous, current),
        })

    out.sort(key=lambda item: -abs(item["deltaPP"]))
    return out


def _feature_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """这两帧之间哪些特征动了。这是「哪里出事了」的直接证据。"""
    changes = []
    for key, label in FEATURE_LABELS.items():
        if key == "minute":
            continue
        before = previous["features"].get(key, 0.0)
        after = current["features"].get(key, 0.0)
        if abs(after - before) < 1e-9:
            continue
        changes.append({
            "key": key, "label": label,
            "before": before, "after": after,
            "delta": round(after - before, 3),
        })
    changes.sort(key=lambda c: -abs(c["delta"]))
    return changes


def explain_inflection(inflection: dict[str, Any]) -> dict[str, Any]:
    """把一个拐点解释成人话，并**明确说出哪些问题现在答不了**。"""
    events = inflection.get("events") or []
    changes = inflection.get("featureChanges") or []

    lines: list[str] = []
    if events:
        readable = {
            "HERO_KILL": "击杀英雄", "TOWER_DESTROYED": "推掉防御塔",
            "HIGHGROUND_DESTROYED": "推掉高地塔", "CRYSTAL_DESTROYED": "推掉水晶",
            "TYRANT_KILL": "拿到暴君", "DARK_TYRANT_KILL": "拿到黑暗暴君",
            "OVERLORD_KILL": "拿到主宰", "STORM_DRAGON_KILL": "拿到风暴龙王",
            "PROPHET_OVERLORD_KILL": "拿到先知主宰",
            "RED_BUFF_KILL": "拿到红BUFF", "BLUE_BUFF_KILL": "拿到蓝BUFF",
        }
        for event in events:
            side = "蓝方" if event.get("teamId") == BLUE else "红方"
            lines.append(f"{side}{readable.get(event.get('type'), event.get('type'))}")
    for change in changes[:3]:
        lines.append(f"{change['label']} {change['before']:+.1f} → {change['after']:+.1f}")

    return {
        "clock": inflection["clock"],
        "deltaPP": inflection["deltaPP"],
        "beneficiary": inflection["direction"],
        "whatHappened": lines or ["这一段没有记录到事件，胜率变化来自数值的连续漂移。"],
        # 下面这块是刻意留空的，见 branch_analysis_status()
        "branches": None,
        "limitation": branch_analysis_status(),
    }


def branch_analysis_status() -> dict[str, Any]:
    """「如果当时换个打法，胜率会是多少」—— 这个问题现在答不了。

    这就是 Q(s,a)，而且必须是**因果的** Q(s,a)。

    最容易犯的错是拿历史数据拟合「做了动作 A 的人后来赢了多少」。
    那学到的是相关性：**优势方才有余裕回城、才敢开龙**，
    于是「回城」和「赢」高度相关，模型就会告诉你回城能赢。
    LOL 项目里 GPT 那版就是这么做的，用户当时明确警告过要避免。

    要做对需要倾向性加权 / 双重稳健 / 离线策略评估这类方法把选择偏差扣掉，
    而且**数据量只解决方差，不解决偏差** —— 再多录像也绕不过这一步。

    所以这里返回一个明确的「未实现」，而不是一个编出来的分支胜率。
    """
    return {
        "implemented": False,
        "title": "分支胜率（如果换个打法会怎样）",
        "why": (
            "这需要 Q(s,a)，而且必须是因果估计，不是拟合。\n"
            "简单拟合「做了 A 的人赢了多少」会学到「优势方才有余裕做 A」——\n"
            "那是相关性不是因果，给出来的建议会系统性地错。\n"
            "需要倾向性加权 / 离线策略评估这类方法，是研究性质的工作。"
        ),
        "blockedBy": "真实数据（现在是零）+ 行为抽象（还没定义）",
    }
