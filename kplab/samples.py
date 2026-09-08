"""录像样本的分层标签与训练权重元数据。

职业比赛、职业选手排位和普通高端局不是同一种分布。这里先把来源边界
写进每场 meta.json；权重只是将来训练时可调整的起点，现在不会触发训练。
"""

from __future__ import annotations

import math
from typing import Any


SAMPLE_TYPES: dict[str, dict[str, Any]] = {
    "professional_match": {
        "label": "职业正式比赛",
        "defaultWeight": 1.5,
        "scope": "战队体系 + 选手决策",
    },
    "pro_player_ranked": {
        "label": "职业选手单排/巅峰赛",
        "defaultWeight": 1.2,
        "scope": "选手个人决策",
    },
    "top_player_ranked": {
        "label": "其他顶尖选手对局",
        "defaultWeight": 1.0,
        "scope": "高分段个人决策",
    },
    "scrim": {
        "label": "训练赛/约战",
        "defaultWeight": 1.2,
        "scope": "协同决策（需单独评估）",
    },
    "other": {
        "label": "其他录像",
        "defaultWeight": 0.6,
        "scope": "待人工核对",
    },
}


class SampleError(ValueError):
    pass


def _short(value: Any, label: str, limit: int = 100) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise SampleError(f"{label}不能超过{limit}个字。")
    return text


def _tags(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("，", ",").split(",")
    elif isinstance(value, list):
        parts = value
    else:
        parts = []
    result: list[str] = []
    for raw in parts:
        tag = str(raw or "").strip()
        if not tag or tag in result:
            continue
        if len(tag) > 30:
            raise SampleError("每个标签不能超过30个字。")
        result.append(tag)
    if len(result) > 20:
        raise SampleError("一场录像最多20个标签。")
    return result


def normalize_metadata(body: dict[str, Any]) -> dict[str, Any]:
    """校验并标准化一场样本的公共元数据。"""
    sample_type = str(body.get("sampleType") or "other")
    if sample_type not in SAMPLE_TYPES:
        raise SampleError("样本类型不支持。")
    default_weight = float(SAMPLE_TYPES[sample_type]["defaultWeight"])
    raw_weight = body.get("trainingWeight")
    try:
        weight = default_weight if raw_weight in (None, "") else float(raw_weight)
    except (TypeError, ValueError):
        raise SampleError("训练权重必须是数字。") from None
    if not math.isfinite(weight) or not 0.1 <= weight <= 5:
        raise SampleError("训练权重必须在0.1到5之间。")

    result: dict[str, Any] = {
        "sampleType": sample_type,
        "sampleTypeLabel": SAMPLE_TYPES[sample_type]["label"],
        "analysisScope": SAMPLE_TYPES[sample_type]["scope"],
        "trainingWeight": round(weight, 3),
        "weightStatus": "initial-unvalidated",
        "gamePatch": _short(body.get("gamePatch"), "游戏版本", 40),
        "focalPlayer": _short(body.get("focalPlayer"), "关注选手", 60),
        "focalTeam": _short(body.get("focalTeam"), "关注战队", 60),
        "rankTier": _short(body.get("rankTier"), "分段", 60),
        "tournament": _short(body.get("tournament"), "赛事", 100),
        "tags": _tags(body.get("tags")),
    }
    # 职业正式比赛保留独立、稳定的标签，后续可以只筛这一层分析战队习惯。
    result["isProfessionalMatch"] = sample_type == "professional_match"
    return result


def public_types() -> list[dict[str, Any]]:
    return [{"value": key, **value} for key, value in SAMPLE_TYPES.items()]
