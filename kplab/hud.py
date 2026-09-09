"""比赛画面上「哪个数字在哪个位置」—— HUD 区域标定。

## 这是整条链路上最脆弱的一环，所以要单独拿出来讲

LOL 那边 `totalGold` 是 JSON 里的一个键，永远在那儿。
王者荣耀这边 `totalGold` 是画面顶部记分板上的一串像素，而这串像素的位置取决于：

  · 录像来源（虎牙 / 斗鱼 / B站 / 官方，各家的水印和裁剪都不一样）
  · 是不是 KPL 职业比赛（职业转播有额外的选手信息条，位置和路人局不同）
  · 分辨率与画面比例（21:9 的转播流两侧会被裁）
  · 版本更新（UI 会改）

**所以这里不存在一套「正确的」坐标。** 任何写死的坐标都只是一次猜测。

## 怎么处理这件事

1. 坐标一律用相对值（0~1），不用像素 —— 这样 720p 和 1080p 通用。
2. 内置的 `KPL_BROADCAST` 只是**起点**，明确标为未标定。
3. 用户在网页标定页上，对着一张真实截图把框拖到正确位置，
   存成 data/kpl/hud_profiles.json。**标定过的才允许拿去跑 OCR。**
4. 没标定就跑 OCR，`ocr.py` 会拒绝，而不是给一堆错数字。

这条规矩的来源同样是 LOL 项目：那边宁可在没有模型时返回 0 并显示
「尚无模型曲线」，也不编一条曲线出来。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 一个区域：(x, y, w, h)，全部是相对整幅画面的比例
Box = tuple[float, float, float, float]

PROFILE_FULL_SCOREBOARD = "full_scoreboard"
PROFILE_PLAYER_POV = "player_pov"
PROFILE_MODES = (PROFILE_FULL_SCOREBOARD, PROFILE_PLAYER_POV)


def valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001


# ---------------------------------------------------------------- 区域清单

# 每个区域要读什么、期望是什么类型。read_kind 决定 OCR 用哪种解析方式。
REGIONS: dict[str, dict[str, Any]] = {
    "clock": {
        "label": "比赛计时",
        "read": "clock",
        "why": "整条时间轴的锚点。这个读错，所有帧都会错位到别的时刻。",
    },
    "blueKills": {"label": "蓝方击杀数", "read": "int", "why": "比分板左侧"},
    "redKills": {"label": "红方击杀数", "read": "int", "why": "比分板右侧"},
    "allyKills": {
        "label": "己方击杀数",
        "read": "int",
        "why": "单人视角左侧比分；在阵营映射确认前不写成蓝方。",
    },
    "enemyKills": {
        "label": "敌方击杀数",
        "read": "int",
        "why": "单人视角右侧比分；在阵营映射确认前不写成红方。",
    },
    "blueGold": {
        "label": "蓝方总经济", "read": "int",
        "why": "可能显示成 2.3万 这种缩写，解析要处理「万」",
    },
    "redGold": {"label": "红方总经济", "read": "int", "why": "同上"},
    "banner": {
        "label": "播报（屏幕中间上方）",
        "read": "banner",
        "why": "推塔 / 击杀 / 开龙都会在这里弹一条横幅并停留几秒。"
               "**这是判断「是谁拿的」唯一可靠的来源** —— "
               "小地图上图标消失只说明资源死了，说不出归属。",
    },
    "minimap": {
        "label": "小地图",
        "read": "minimap",
        "why": "十个英雄的位置。**没有它就做不了「入侵野区值不值」这类判断**",
    },
}

# 十个选手各自的头像条区域（等级、存活状态）。
for _slot in range(1, 11):
    _side = "蓝" if _slot <= 5 else "红"
    REGIONS[f"player{_slot}Level"] = {
        "label": f"{_side}方 {(_slot - 1) % 5 + 1} 号位 等级",
        "read": "int",
        "why": "头像旁边的小数字，字号最小，也最容易识别错",
    }


# ---------------------------------------------------------------- 内置起点

# 明确说明：这不是标定结果，是一个待修正的起点。所有值都可能是错的。
KPL_BROADCAST: dict[str, Any] = {
    "name": "KPL 职业转播（未标定的起点）",
    "mode": PROFILE_FULL_SCOREBOARD,
    "calibrated": False,
    "note": "内置的**猜测值**，必须在标定页上对着真实画面改过才能用。\n"
                "已知的方位（用户确认过）：小地图在左上角、经济面板在下方中间、"
                "计时和比分在上方中间。但**具体的框还是猜的** —— "
                "不同录像源的裁剪和分辨率都不一样。",
    "aspect": None,
    "regions": {
        "clock": [0.470, 0.010, 0.060, 0.045],
        "blueKills": [0.415, 0.010, 0.045, 0.045],
        "redKills": [0.540, 0.010, 0.045, 0.045],
        # 经济面板在**屏幕下方中间**（用户原话：正式比赛会在下方中间较长时间展示）。
        # 第一版放在了顶部，同样是错的。
        "blueGold": [0.330, 0.900, 0.080, 0.050],
        "redGold": [0.590, 0.900, 0.080, 0.050],
        # 小地图在**左上角**。
        # 第一版按 LOL 的习惯放在了左下（LOL 的小地图在右下角），
        # 用户对着真实画面一眼看出不对 —— 裁出来的是赞助商横幅。
        # **这正是「不能照抄 LOL」那份文档警告的事，我却在坐标上犯了。**
        "minimap": [0.000, 0.000, 0.150, 0.270],
        "banner": [0.330, 0.090, 0.340, 0.075],
    },
}


def builtin_profiles() -> dict[str, dict[str, Any]]:
    return {"kpl_broadcast": json.loads(json.dumps(KPL_BROADCAST))}


# ---------------------------------------------------------------- 读写

def profiles_path(data_dir: Path) -> Path:
    return data_dir / "kpl" / "hud_profiles.json"


def load_profiles(data_dir: Path) -> dict[str, dict[str, Any]]:
    """内置起点 + 用户标定过的。同名时用户的覆盖内置的。"""
    out = builtin_profiles()
    path = profiles_path(data_dir)
    if not path.is_file():
        return out
    try:
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return out
    if isinstance(saved, dict):
        for key, profile in saved.items():
            if isinstance(profile, dict):
                out[key] = profile
    return out


def get_profile(data_dir: Path, name: str) -> dict[str, Any] | None:
    return load_profiles(data_dir).get(name)


def save_profile(data_dir: Path, name: str, profile: dict[str, Any]) -> Path:
    """保存一份标定。

    只有当**所有必需区域**都被明确设置过，才允许 calibrated=True。
    这是防止「拖了两个框就以为标定完了」。
    """
    path = profiles_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                saved = loaded
        except (OSError, ValueError, UnicodeDecodeError):
            saved = {}

    regions = {
        key: [round(float(v), 5) for v in box]
        for key, box in (profile.get("regions") or {}).items()
        if key in REGIONS and valid_box(box)
    }
    mode = str(profile.get("mode") or PROFILE_FULL_SCOREBOARD)
    if mode not in PROFILE_MODES:
        mode = PROFILE_FULL_SCOREBOARD
    report = completeness(regions, mode)
    saved[name] = {
        "name": profile.get("name") or name,
        "mode": mode,
        "calibrated": report["complete"],
        "note": profile.get("note") or "",
        "aspect": profile.get("aspect"),
        "calibratedAt": profile.get("calibratedAt") or "",
        "regions": regions,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2)
    return path


# 没有这几个区域，State 里就没有任何一个球队级别的数字，整场没有意义
REQUIRED_BY_MODE = {
    # 赛事转播的固定记分板应该同时提供时间、比分和团队经济。
    PROFILE_FULL_SCOREBOARD: ("clock", "blueKills", "redKills", "blueGold", "redGold"),
    # 单人视角通常不常驻显示双方总经济。经济必须留为 unknown，
    # 不应因此拒绝读取确实可见的时间和比分。
    PROFILE_PLAYER_POV: ("clock",),
}

# 保留原名供旧代码/外部调用者使用。
REQUIRED = REQUIRED_BY_MODE[PROFILE_FULL_SCOREBOARD]


def completeness(
    regions: dict[str, Any], mode: str = PROFILE_FULL_SCOREBOARD
) -> dict[str, Any]:
    if mode not in PROFILE_MODES:
        mode = PROFILE_FULL_SCOREBOARD
    required = REQUIRED_BY_MODE[mode]
    have = {k for k, v in regions.items() if valid_box(v)}
    missing_required = [k for k in required if k not in have]
    optional = [k for k in REGIONS if k not in required]
    return {
        "mode": mode,
        "required": list(required),
        "complete": not missing_required,
        "missingRequired": missing_required,
        "haveCount": len(have),
        "totalCount": len(REGIONS),
        "missingOptional": [k for k in optional if k not in have],
        "hasMinimap": "minimap" in have,
        "hasLevels": sum(1 for k in have if k.endswith("Level")),
    }


def describe(profile: dict[str, Any] | None) -> dict[str, Any]:
    """给网页 / 命令行看的一句话状态。"""
    if not profile:
        return {"ok": False, "message": "没有这个标定档案。"}
    mode = str(profile.get("mode") or PROFILE_FULL_SCOREBOARD)
    report = completeness(profile.get("regions") or {}, mode)
    if not profile.get("calibrated"):
        return {
            "ok": False,
            "message": (
                "这份标定还没完成，不能用来跑识别。"
                + (f"缺少必需区域：{'、'.join(report['missingRequired'])}。"
                   if report["missingRequired"] else "缺少人工确认。")
            ),
            **report,
        }
    extra = ""
    if not report["hasMinimap"]:
        extra = "（小地图没标定，位置信息会全部缺失，「入侵值不值」这类问题做不了）"
    return {"ok": True, "message": f"标定完成，{report['haveCount']}/{report['totalCount']} 个区域可用。{extra}", **report}


def to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        int(round(x * width)), int(round(y * height)),
        int(round(w * width)), int(round(h * height)),
    )
