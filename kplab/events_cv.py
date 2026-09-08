"""从小地图和头像栏的**图标变化**推断事件。

## 这条路子为什么比 OCR 好

原来的思路是 OCR：把记分板上的数字读出来，比对前后帧的差。
问题是 OCR 认错一个数字就全错，而且游戏字体、缩放、压缩噪点都会让它出错。

用户指出了一条好得多的路：**盯图标的出现与消失**。

| 事件 | 画面上的表现 |
| --- | --- |
| 推掉防御塔 | 小地图上那座塔的图标**消失** |
| 拿到暴君 / 主宰 | 小地图上那个坑的图标**消失** |
| 英雄阵亡 | 头像栏里那个头像**变灰**，并出现复活倒计时 |
| （佐证）击杀 | 右上角比分**加一** |

「某个固定位置的图标还在不在」比「这串像素是几」容易判断一个数量级 ——
前者只要比较一小块区域和它自己之前的样子，后者要认字。

## 但有一个致命的区分，弄错了数据就是坏的

**防御塔是单向的，中立资源是周期的。**

- 塔的图标消失了就永远不会回来。所以「图标回来了」= 我的判断错了。
- 暴君会刷新。图标消失又出现是**完全正常的**，
  而且「消失 → 再出现」恰恰是一次完整的击杀 + 刷新周期。

如果照抄塔的逻辑去处理暴君，会得到「暴君被拆了 6 座」这种东西 ——
换个地方重犯 LOL 那个「先锋算成大龙」的错。
所以这里两类地标走**两条完全不同的状态机**，代码里刻意不合并。

## 第二条原则：两个独立信号对上了才算数

头像变灰说明有人死了，比分加一也说明有人死了。
这两个信号来自画面上**完全不同的两个区域**，出错的原因也不同。

- 两个都动了 → 高置信度，而且知道是谁死的
- 只有比分动 → 确实死了人，但不知道是谁（记团队事件，不记到具体人头上）
- 只有头像变灰 → **可疑**。可能是阈值不对、可能是特效遮挡。低置信度并标记出来

这比任何单一信号都可靠，而且失败时是**明着失败**，不是悄悄给个错答案。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import minimap, schema
from .rules import BLUE, RED

# ---------------------------------------------------------------- 地标

# 地标类型决定用哪条状态机。**这个区分是本模块的核心。**
KIND_TOWER = "tower"            # 单向：消失即摧毁，不会回来
KIND_OBJECTIVE = "objective"    # 周期：消失是被击杀，之后会刷新回来
KIND_PORTRAIT = "portrait"      # 变灰即阵亡，复活后变回彩色

LANDMARK_KINDS = (KIND_TOWER, KIND_OBJECTIVE, KIND_PORTRAIT)

# 中立资源坑 → 事件类型。哪个坑刷出什么由时间决定（暴君 10 分钟后变黑暗暴君），
# 所以这里只记「哪个坑被拿了」，具体是哪种形态交给 resolve_objective_event()。
OBJECTIVE_PITS = {
    "tyrantPit": {"label": "暴君坑（下路河道）", "early": "TYRANT_KILL",
                  "late": "DARK_TYRANT_KILL", "switchRule": "darkTyrantFromSec"},
    "overlordPit": {"label": "主宰坑（上路河道）", "early": "OVERLORD_KILL",
                    "late": "STORM_DRAGON_KILL", "switchRule": "stormDragonFromSec"},
}


def landmarks_path(data_dir: Path) -> Path:
    return data_dir / "kpl" / "landmarks.json"


def load_landmarks(data_dir: Path) -> dict[str, Any]:
    """读取标定过的地标。没标定过就返回空 —— **不猜位置。**

    塔和坑在小地图上的位置是固定的，但具体在哪取决于录像的裁剪和分辨率，
    只能靠人在标定页上点出来。没标定就什么都检测不了，这是正确的行为。
    """
    path = landmarks_path(data_dir)
    if not path.is_file():
        return {"calibrated": False, "items": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return {"calibrated": False, "items": {}}
    if not isinstance(saved, dict):
        return {"calibrated": False, "items": {}}
    items = saved.get("items") if isinstance(saved.get("items"), dict) else {}
    return {"calibrated": bool(saved.get("calibrated")) and bool(items), "items": items}


def save_landmarks(data_dir: Path, items: dict[str, Any], calibrated: bool = True) -> Path:
    cleaned: dict[str, Any] = {}
    for name, item in (items or {}).items():
        if not isinstance(item, dict):
            continue
        box = item.get("box")
        kind = item.get("kind")
        if kind not in LANDMARK_KINDS or not _valid_box(box):
            continue
        entry = {"kind": kind, "box": [round(float(v), 5) for v in box],
                 "label": str(item.get("label") or name)[:40]}
        if item.get("teamId") in (BLUE, RED):
            entry["teamId"] = int(item["teamId"])
        if item.get("slot") is not None:
            try:
                entry["slot"] = int(item["slot"])
            except (TypeError, ValueError):
                pass
        cleaned[name] = entry

    path = landmarks_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"calibrated": bool(calibrated) and bool(cleaned),
                   "items": cleaned}, handle, ensure_ascii=False, indent=2)
    return path


def _valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1


# ---------------------------------------------------------------- 取特征

def patch_signature(data: bytes, width: int, height: int) -> dict[str, float]:
    """把一小块区域压成几个数：平均色 + 饱和度 + 亮度。

    图标在不在，就看这几个数和基准比变了多少。
    不做模板匹配 —— 那需要图标素材库，而且版本一更新素材就过时。
    """
    if not data:
        return {"r": 0.0, "g": 0.0, "b": 0.0, "saturation": 0.0, "value": 0.0}
    count = width * height
    total_r = total_g = total_b = 0
    total_sat = total_val = 0.0
    for i in range(count):
        base = i * 3
        red, green, blue = data[base], data[base + 1], data[base + 2]
        total_r += red
        total_g += green
        total_b += blue
        high, low = max(red, green, blue), min(red, green, blue)
        total_val += high
        total_sat += 0.0 if high == 0 else (high - low) / high
    return {
        "r": total_r / count, "g": total_g / count, "b": total_b / count,
        "saturation": total_sat / count, "value": total_val / count,
    }


def read_patch(image: Path, box: tuple[float, float, float, float],
               size: int = 24) -> dict[str, float]:
    data, width, height = minimap.read_rgb(image, box, size)
    return patch_signature(data, width, height)


def signature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """两个签名差多远。0 = 完全一样，越大差别越大。"""
    colour = ((a["r"] - b["r"]) ** 2 + (a["g"] - b["g"]) ** 2
              + (a["b"] - b["b"]) ** 2) ** 0.5
    return colour / 441.67          # 除以 RGB 空间最大距离，归一到 0~1


# 超过这个距离就认为「这块地方变了」。凭经验定的起点，需要用真实画面标定。
CHANGE_THRESHOLD = 0.14
# 饱和度掉到基准的这个比例以下，就认为头像变灰了。同样是待标定的起点。
GRAY_RATIO = 0.45


# ---------------------------------------------------------------- 状态机

class TowerTracker:
    """防御塔：**单向**。消失即摧毁，不会回来。

    「图标又出现了」不是塔重建了 —— 塔不会重建。
    那只说明我判错了一次，必须记下来而不是当没发生。
    """

    def __init__(self, name: str, team_id: int, label: str) -> None:
        self.name, self.team_id, self.label = name, team_id, label
        self.baseline: dict[str, float] | None = None
        self.destroyed_at: float | None = None
        self.contradictions: list[dict[str, Any]] = []

    def update(self, at_sec: float, signature: dict[str, float]) -> dict[str, Any] | None:
        if self.baseline is None:
            self.baseline = signature
            return None
        changed = signature_distance(self.baseline, signature) > CHANGE_THRESHOLD

        if self.destroyed_at is None:
            if changed:
                self.destroyed_at = at_sec
                # 塔属于哪一方，功劳就记给对面
                winner = RED if self.team_id == BLUE else BLUE
                return {"type": "TOWER_DESTROYED", "teamId": winner,
                        "atSec": at_sec, "landmark": self.name,
                        "note": f"{self.label} 的图标消失"}
            return None

        if not changed:
            # 已经判为摧毁，图标却回来了 —— 塔不会重建，所以是我错了
            self.contradictions.append({
                "atSec": at_sec, "landmark": self.name,
                "why": "判为摧毁之后图标又出现了。防御塔不会重建，"
                       "说明之前那次判定是错的（可能是特效或小地图遮挡）。",
            })
        return None


class ObjectiveTracker:
    """中立资源：**周期**。消失是被击杀，之后会刷新回来。

    和塔用同一套逻辑会得到「暴君被拆了 6 座」——
    换个地方重犯 LOL 那个「先锋算成大龙」的错。所以单独一个类。

    还有一条：刷新是需要时间的。两次击杀之间间隔短得离谱，
    说明是图标闪烁或遮挡造成的抖动，不是真的被拿了两次。
    """

    MIN_SECONDS_BETWEEN_KILLS = 60.0

    def __init__(self, name: str, label: str) -> None:
        self.name, self.label = name, label
        self.baseline: dict[str, float] | None = None
        self.present = True
        self.last_kill_at: float | None = None
        self.kills: list[float] = []
        self.rejected: list[dict[str, Any]] = []

    def update(self, at_sec: float, signature: dict[str, float]) -> dict[str, Any] | None:
        if self.baseline is None:
            self.baseline = signature
            return None
        changed = signature_distance(self.baseline, signature) > CHANGE_THRESHOLD

        if self.present and changed:
            if (self.last_kill_at is not None
                    and at_sec - self.last_kill_at < self.MIN_SECONDS_BETWEEN_KILLS):
                self.rejected.append({
                    "atSec": at_sec, "landmark": self.name,
                    "why": f"距上次击杀只有 {at_sec - self.last_kill_at:.0f} 秒，"
                           f"短于刷新所需的最短时间。判为图标抖动，不计。",
                })
                return None
            self.present = False
            self.last_kill_at = at_sec
            self.kills.append(at_sec)
            # 归属哪一方，图标本身说不出来 —— 要靠别的信号，见 attribute_objective()
            return {"type": "OBJECTIVE_TAKEN", "pit": self.name,
                    "atSec": at_sec, "teamId": None,
                    "note": f"{self.label} 的图标消失"}

        if not self.present and not changed:
            self.present = True     # 刷新回来了，完全正常
        return None


class PortraitTracker:
    """英雄头像：变灰 = 阵亡，恢复彩色 = 复活。

    判据是**饱和度**而不是亮度：灰度头像的饱和度接近 0，
    而亮度会随特效和技能光效大幅变化，不可靠。
    """

    def __init__(self, slot: int, label: str) -> None:
        self.slot, self.label = slot, label
        self.alive_saturation: float | None = None
        self.dead = False
        self.deaths: list[float] = []

    def update(self, at_sec: float, signature: dict[str, float]) -> dict[str, Any] | None:
        saturation = signature["saturation"]
        if self.alive_saturation is None:
            self.alive_saturation = saturation
            return None

        looks_gray = saturation < self.alive_saturation * GRAY_RATIO

        if not self.dead and looks_gray:
            self.dead = True
            self.deaths.append(at_sec)
            team = BLUE if self.slot <= 5 else RED
            return {"type": "PORTRAIT_GRAY", "slot": self.slot, "teamId": team,
                    "atSec": at_sec,
                    "note": f"{self.label} 头像变灰（饱和度 "
                            f"{saturation:.2f} < 基准 {self.alive_saturation:.2f} 的 "
                            f"{GRAY_RATIO:.0%}）"}

        if self.dead and not looks_gray:
            self.dead = False
            # 头像基准随版本/皮肤会漂移，复活时用新值缓慢校正
            self.alive_saturation = 0.7 * self.alive_saturation + 0.3 * saturation
        return None


# ---------------------------------------------------------------- 佐证

def corroborate_kills(
    portrait_events: list[dict[str, Any]],
    score_changes: list[dict[str, Any]],
    window_sec: float = 45.0,
) -> dict[str, Any]:
    """把「头像变灰」和「比分加一」两个信号对起来。

    两个信号来自画面上完全不同的两块区域，出错原因也不同 ——
    对得上就可信，对不上就明着说对不上，而不是挑一个信。
    """
    confirmed: list[dict[str, Any]] = []
    portrait_only: list[dict[str, Any]] = []
    score_only: list[dict[str, Any]] = []
    used: set[int] = set()

    for portrait in portrait_events:
        match_index = None
        for index, change in enumerate(score_changes):
            if index in used:
                continue
            # 比分加的应该是死者的对面
            victim_team = portrait.get("teamId")
            scorer = change.get("teamId")
            if victim_team is not None and scorer == victim_team:
                continue
            if abs(change["atSec"] - portrait["atSec"]) <= window_sec:
                match_index = index
                break

        if match_index is None:
            portrait_only.append(portrait)
            continue
        used.add(match_index)
        change = score_changes[match_index]
        victim_team = portrait.get("teamId")
        confirmed.append({
            "type": "HERO_KILL",
            "teamId": RED if victim_team == BLUE else BLUE,   # 击杀方
            "victimSlot": portrait.get("slot"),
            "atSec": portrait["atSec"],
            "confidence": 0.8,
            "evidence": ["头像变灰", "比分变化"],
            "note": f"{portrait.get('note', '')}；同时比分变化，两个信号对上",
        })

    for index, change in enumerate(score_changes):
        if index not in used:
            score_only.append(change)

    return {
        "confirmed": confirmed,
        "portraitOnly": portrait_only,
        "scoreOnly": score_only,
        "summary": _corroboration_summary(confirmed, portrait_only, score_only),
    }


def _corroboration_summary(confirmed, portrait_only, score_only) -> list[str]:
    lines: list[str] = []
    if confirmed:
        lines.append(f"{len(confirmed)} 次击杀有两个独立信号佐证，可信。")
    if portrait_only:
        lines.append(
            f"{len(portrait_only)} 次头像变灰但比分没动。"
            "**这是最需要怀疑的一类** —— 可能是变灰阈值不对，"
            "也可能是技能特效把头像盖住了。不要直接当成击杀。"
        )
    if score_only:
        lines.append(
            f"{len(score_only)} 次比分变化但没找到对应的头像变灰。"
            "确实死了人，但不知道是谁 —— 只记团队击杀数，不记到具体人头上。"
        )
    if not lines:
        lines.append("这一段没有检测到击杀。")
    return lines


def attribute_objective(
    objective_event: dict[str, Any],
    score_changes: list[dict[str, Any]],
    gold_lead: int | None = None,
    window_sec: float = 30.0,
) -> dict[str, Any]:
    """中立资源被谁拿了 —— 图标消失本身回答不了这个问题。

    小地图上暴君的图标没了，只说明它死了，**说不出是谁打的**。
    这里只做一件事：如果附近有比分变化，说明有团战，
    但那**仍然不能证明**是赢的那方拿到了资源。

    所以这个函数的正确输出往往是「不知道」。
    强行猜一方，会在下游变成一条看起来很确定的假数据。
    """
    nearby = [c for c in score_changes
              if abs(c["atSec"] - objective_event["atSec"]) <= window_sec]
    if not nearby:
        return {**objective_event, "teamId": None, "confidence": 0.0,
                "attribution": "不知道是哪一方拿的。图标消失只说明资源死了。"}

    sides = {c.get("teamId") for c in nearby if c.get("teamId") in (BLUE, RED)}
    if len(sides) == 1:
        side = sides.pop()
        return {**objective_event, "teamId": side, "confidence": 0.35,
                "attribution": f"附近只有{'蓝' if side == BLUE else '红'}方比分在涨，"
                               "**倾向**是这一方拿的 —— 但这只是倾向，"
                               "打赢团战和拿到资源不是一回事。"}
    return {**objective_event, "teamId": None, "confidence": 0.0,
            "attribution": "附近双方比分都有变化，判断不了归属。"}


def resolve_objective_event(pit_name: str, at_sec: float,
                            rules_table: dict[str, Any]) -> str:
    """同一个坑在不同时间刷的是不同的东西。

    暴君坑：10 分钟前是暴君，之后是黑暗暴君。
    主宰坑：后期变风暴龙王。
    **这两种收益完全不同，绝不能记成同一个事件。**
    """
    from . import rules
    spec = OBJECTIVE_PITS.get(pit_name)
    if not spec:
        return "OBJECTIVE_TAKEN"
    switch_at = rules.value(rules_table, spec["switchRule"])
    try:
        switch_at = float(switch_at)
    except (TypeError, ValueError):
        return spec["early"]
    return spec["late"] if at_sec >= switch_at else spec["early"]


# ---------------------------------------------------------------- 主流程

def scan(
    frames: list[dict[str, Any]],
    data_dir: Path,
    rules_table: dict[str, Any] | None = None,
    score_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按时间顺序扫一遍帧图，产出事件。

    frames: [{"atSec": float, "path": Path}]，必须已按时间排好。
    score_changes: 比分变化（来自 OCR 或人工），用于佐证。没有就只出低置信度结果。
    """
    from . import rules
    rules_table = rules_table or rules.load(data_dir)
    landmarks = load_landmarks(data_dir)
    if not landmarks["calibrated"]:
        return {
            "ok": False,
            "error": "地标还没有标定过，检测不了。\n"
                     "需要在标定页上把每座塔、两个资源坑、十个头像的位置框出来。\n"
                     "**没有标定就猜位置，产出的每一个事件都是错的。**",
            "events": [], "landmarksNeeded": True,
        }

    items = landmarks["items"]
    towers = {n: TowerTracker(n, i.get("teamId", BLUE), i.get("label", n))
              for n, i in items.items() if i["kind"] == KIND_TOWER}
    objectives = {n: ObjectiveTracker(n, i.get("label", n))
                  for n, i in items.items() if i["kind"] == KIND_OBJECTIVE}
    portraits = {n: PortraitTracker(i.get("slot", 1), i.get("label", n))
                 for n, i in items.items() if i["kind"] == KIND_PORTRAIT}

    raw_events: list[dict[str, Any]] = []
    portrait_events: list[dict[str, Any]] = []
    errors: list[str] = []

    for frame in frames:
        at_sec = float(frame["atSec"])
        path = Path(frame["path"])
        for name, item in items.items():
            try:
                signature = read_patch(path, tuple(item["box"]))
            except Exception as err:            # noqa: BLE001 - 单块失败不该毁掉整场
                errors.append(f"{path.name} 的 {name}: {type(err).__name__}: {err}")
                continue
            kind = item["kind"]
            if kind == KIND_TOWER:
                event = towers[name].update(at_sec, signature)
            elif kind == KIND_OBJECTIVE:
                event = objectives[name].update(at_sec, signature)
            else:
                event = portraits[name].update(at_sec, signature)
            if event is None:
                continue
            if event["type"] == "PORTRAIT_GRAY":
                portrait_events.append(event)
            else:
                raw_events.append(event)

    # 中立资源：先定形态（暴君 / 黑暗暴君），再试着定归属
    resolved: list[dict[str, Any]] = []
    for event in raw_events:
        if event["type"] != "OBJECTIVE_TAKEN":
            resolved.append({**event, "confidence": 0.6})
            continue
        kind = resolve_objective_event(event["pit"], event["atSec"], rules_table)
        resolved.append(attribute_objective({**event, "type": kind},
                                            score_changes or []))

    kills = corroborate_kills(portrait_events, score_changes or [])

    contradictions = [c for t in towers.values() for c in t.contradictions]
    jitter = [r for o in objectives.values() for r in o.rejected]

    return {
        "ok": True,
        "events": resolved + kills["confirmed"],
        "kills": kills,
        "contradictions": contradictions,
        "objectiveJitter": jitter,
        "readErrors": errors[:20],
        "quality": _quality(resolved, kills, contradictions, errors),
    }


def _quality(resolved, kills, contradictions, errors) -> dict[str, Any]:
    problems: list[str] = []
    if contradictions:
        problems.append(
            f"{len(contradictions)} 次「塔判为摧毁后图标又回来了」。"
            "塔不会重建，说明变化阈值偏松或者被特效干扰，塔的判定不可信。"
        )
    if kills["portraitOnly"]:
        problems.append(
            f"{len(kills['portraitOnly'])} 次头像变灰没有比分佐证。"
        )
    if errors:
        problems.append(f"{len(errors)} 块区域读取失败。")

    unattributed = sum(1 for e in resolved
                       if e.get("type", "").endswith("_KILL") and e.get("teamId") is None)
    if unattributed:
        problems.append(
            f"{unattributed} 次中立资源击杀判断不出归属。"
            "这是正常的 —— 图标消失说不出是谁打的，需要人工补。"
        )
    return {
        "confirmedKills": len(kills["confirmed"]),
        "problems": problems,
        "trustworthy": not contradictions and not errors,
    }


def to_observation_events(scan_result: dict[str, Any],
                          min_confidence: float = 0.5) -> list[dict[str, Any]]:
    """把扫描结果转成可以写进观测的事件。

    **低于阈值的一律丢掉，而不是降级写进去。**
    一条置信度 0.2 的「红方拿到主宰」进了观测，下游看到的就是一条事件，
    中立资源计数器会 +1，然后一路错到胜率曲线里。
    """
    out: list[dict[str, Any]] = []
    for event in scan_result.get("events", []):
        if event.get("teamId") not in (BLUE, RED):
            continue                       # 归属不明的不写，宁可漏
        if float(event.get("confidence", 0.0)) < min_confidence:
            continue
        out.append({"type": event["type"], "teamId": event["teamId"],
                    "atSec": event["atSec"], "note": event.get("note", "")[:120]})
    return out
