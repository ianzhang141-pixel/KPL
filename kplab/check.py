"""数据完整性与一致性检查，输出中文报告。

回答一个很具体的问题：**这场比赛的数据，够不够格拿去做后面的事？**

分三层，从便宜到贵：

  1. 有没有 —— 观测存在吗、帧数够吗、时间轴连续吗
  2. 对不对 —— 有没有违反游戏规则的值（等级 27、经济倒退、塔数 12）
  3. 可不可信 —— 有多少字段是真读到的、多少是沿用的、多少纯属未知

第 3 层是 LOL 项目没有的。那边的数字来自官方 API，只有「有」和「没有」；
这边的数字来自识别，「有」还要再分「可信的有」和「猜的有」。
一份 95% 字段齐全但全是低置信度 OCR 的数据，比一份 40% 齐全但都是人工核对过的
更危险 —— 因为它看起来很完整。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import rules, schema, store


def run(data_dir: Path, game_id: str) -> dict[str, Any]:
    meta = store.load_meta(data_dir, game_id)
    rules_table = rules.load(data_dir)

    report: dict[str, Any] = {
        "gameId": game_id,
        "found": meta is not None,
        "meta": meta or {},
        "problems": [],
        "warnings": [],
        "notes": [],
    }
    if meta is None:
        report["verdict"] = {"ok": False, "summary": f"data 目录里没有这场比赛：{game_id}"}
        return report

    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))
    annotations = list(store.read_jsonl_gz(store.annotations_path(data_dir, game_id)))
    report["observationCount"] = len(observations)
    report["annotationCount"] = len(annotations)

    # ---- 待核对的游戏常量 ----
    pending = rules.unverified(rules_table)
    if pending:
        report["warnings"].append(
            f"有 {len(pending)} 条游戏常量还没被人核对过："
            f"{'、'.join(rules.DEFAULTS[k]['what'] for k in pending[:4])}"
            + ("等。" if len(pending) > 4 else "。")
            + "这些是凭印象写的默认值，版本改动后可能不对，"
            "下面的上限校验会跟着不准。核对办法见 kplab rules。"
        )

    if not observations:
        report["problems"].append(
            "这场一条观测都没有。先去标注（kplab annotate）或抽帧后跑识别。"
        )
        report["verdict"] = {"ok": False, "summary": "没有观测数据，后面的检查无从谈起。"}
        return report

    # ---- 第 1 层：时间轴 ----
    times = sorted(float(o.get("atSec") or 0.0) for o in observations)
    report["timeline"] = _check_timeline(times, report)

    # ---- 第 2、3 层：跑一遍 State ----
    from . import state as state_mod

    built = state_mod.build(meta, observations, rules_table)
    report["frameCount"] = built["meta"]["frameCount"]
    report["coverage"] = built["coverage"]
    report["rejections"] = built["rejections"][:50]
    report["rejectionCount"] = len(built["rejections"])

    _check_rules(built, report, rules_table)
    _check_trust(built, report)
    _check_sources(built, report)

    if meta.get("blueWin") is None:
        report["warnings"].append(
            "这场没有记录胜负（blueWin）。胜负是将来训练 V(s) 的唯一标签，"
            "没有它这场数据训不了模型 —— 现在补最省事。"
        )

    ok = not report["problems"]
    report["verdict"] = {
        "ok": ok,
        "summary": _summarize(report, ok),
    }
    return report


def _check_timeline(times: list[float], report: dict[str, Any]) -> dict[str, Any]:
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    duplicates = sum(1 for g in gaps if g == 0)
    if duplicates:
        report["warnings"].append(
            f"有 {duplicates} 对观测的时间戳完全相同。同一时刻被记了两次，"
            "后一条会覆盖前一条，可能不是你想要的。"
        )
    big = [g for g in gaps if gaps and g > 0]
    median = sorted(big)[len(big) // 2] if big else 0.0
    holes = [
        {"fromSec": round(times[i], 1), "toSec": round(times[i + 1], 1),
         "gapSec": round(gaps[i], 1)}
        for i in range(len(gaps)) if median and gaps[i] > median * 2.5
    ]
    if holes:
        report["warnings"].append(
            f"时间轴上有 {len(holes)} 处明显的空档（最长 "
            f"{max(h['gapSec'] for h in holes):.0f} 秒）。"
            "空档期间发生的团战和资源争夺完全看不到，"
            "这段时间的胜率变化将来会解释不了。"
        )
    return {
        "startSec": round(times[0], 1),
        "endSec": round(times[-1], 1),
        "spanSec": round(times[-1] - times[0], 1),
        "medianGapSec": round(median, 1),
        "duplicateTimestamps": duplicates,
        "holes": holes[:20],
    }


def _check_rules(built: dict[str, Any], report: dict[str, Any], rules_table: dict[str, Any]) -> None:
    """违反游戏规则的值。这些是识别错误最可靠的信号。"""
    max_towers = rules.max_towers(rules_table)
    max_level = int(rules.value(rules_table, "maxLevel"))
    tyrant_at = float(rules.value(rules_table, "tyrantFirstSpawnSec"))
    overlord_at = float(rules.value(rules_table, "overlordFirstSpawnSec"))

    violations: list[dict[str, Any]] = []
    for frame in built["frames"]:
        at = frame["atSec"]
        for side, team in frame["teams"].items():
            towers = schema.get(team["towers"])
            if towers is not None and towers > max_towers:
                violations.append({"clock": frame["clock"], "what": f"{side} 塔数 {towers} > 上限 {max_towers}"})
            # 资源不可能在刷新之前被击杀 —— 这条能一次抓出「时间读错」和「资源认错」
            if schema.get(team["tyrants"], 0) > 0 and at < tyrant_at:
                violations.append({"clock": frame["clock"], "what": f"{side} 在暴君刷新（{tyrant_at:.0f}s）前就有暴君击杀"})
            if schema.get(team["overlords"], 0) > 0 and at < overlord_at:
                violations.append({"clock": frame["clock"], "what": f"{side} 在主宰刷新（{overlord_at:.0f}s）前就有主宰击杀"})
        for player in frame["players"]:
            level = schema.get(player["level"])
            if level is not None and not (1 <= level <= max_level):
                violations.append({"clock": frame["clock"], "what": f"{player['slot']} 号位等级 {level} 不在 1~{max_level}"})

    report["ruleViolations"] = violations[:40]
    report["ruleViolationCount"] = len(violations)
    if violations:
        report["problems"].append(
            f"有 {len(violations)} 处违反游戏规则的值。"
            "这类值不可能是真的，说明识别或标定出了问题，必须先查清楚 —— "
            "带着它们往下走，训出来的模型是坏的。"
        )


def _check_trust(built: dict[str, Any], report: dict[str, Any]) -> None:
    cov = built["coverage"]
    frames = built["meta"]["frameCount"]
    usable = cov.get("usableFrames", 0)

    if cov["knownRatio"] < 0.2:
        report["problems"].append(
            f"平均每帧只有 {cov['knownRatio'] * 100:.0f}% 的字段读到了值。"
            "这个覆盖率下的「局势」谈不上可信，先解决标定或标注的问题。"
        )
    elif cov["knownRatio"] < 0.5:
        report["warnings"].append(
            f"平均每帧只有 {cov['knownRatio'] * 100:.0f}% 的字段有值，偏低。"
        )

    if frames and usable / frames < 0.5:
        report["warnings"].append(
            f"{frames} 帧里只有 {usable} 帧的可信字段过半。"
            "可以用来核对，但还不足以拿去训练任何东西。"
        )

    if cov["knownRatio"] - cov["trustedRatio"] > 0.25:
        report["notes"].append(
            f"有值的字段占 {cov['knownRatio'] * 100:.0f}%，"
            f"但其中只有 {cov['trustedRatio'] * 100:.0f}% 够得上「可信」。"
            "差额主要是低置信度的识别结果和沿用值 —— "
            "**这类数据最危险，因为表格看起来是满的。**"
        )


def _check_sources(built: dict[str, Any], report: dict[str, Any]) -> None:
    tally: dict[str, int] = {}
    for frame in built["frames"]:
        for _, entry in schema.iter_fields(frame):
            tally[schema.src(entry)] = tally.get(schema.src(entry), 0) + 1
    report["sourceTally"] = tally

    manual = tally.get(schema.SOURCE_MANUAL, 0)
    machine = sum(tally.get(s, 0) for s in schema.MACHINE_SOURCES)
    if machine and not manual:
        report["warnings"].append(
            "这场全部是机器识别、没有一条人工标注。"
            "没有人工标注就没有 ground truth，识别准不准无从验证 —— "
            "建议至少人工标 5~10 帧，用 kplab evaluate 看看识别的准确率。"
        )


def _summarize(report: dict[str, Any], ok: bool) -> str:
    cov = report.get("coverage") or {}
    head = "通过" if ok else "不通过"
    return (
        f"{head}：{report.get('frameCount', 0)} 帧，"
        f"字段覆盖 {cov.get('knownRatio', 0) * 100:.0f}%、"
        f"可信 {cov.get('trustedRatio', 0) * 100:.0f}%，"
        f"{report.get('ruleViolationCount', 0)} 处违规、"
        f"{report.get('rejectionCount', 0)} 个读数被拒。"
    )


# ---------------------------------------------------------------- 文本报告

def format_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 72)
    add(f"比赛：{report['gameId']}")
    meta = report.get("meta") or {}
    if meta.get("blueTeam") or meta.get("redTeam"):
        add(f"对阵：{meta.get('blueTeam') or '?'}（蓝） vs {meta.get('redTeam') or '?'}（红）")
    if meta.get("tournament"):
        add(f"赛事：{meta['tournament']}")
    add("=" * 72)

    if not report["found"]:
        add(f"❌ {report['verdict']['summary']}")
        return "\n".join(lines)

    add(f"观测 {report.get('observationCount', 0)} 条    "
        f"人工标注 {report.get('annotationCount', 0)} 条    "
        f"State {report.get('frameCount', 0)} 帧")

    timeline = report.get("timeline") or {}
    if timeline:
        add(f"时间轴：{timeline['startSec']:.0f}s → {timeline['endSec']:.0f}s"
            f"（跨度 {timeline['spanSec'] / 60:.1f} 分钟，间隔中位数 {timeline['medianGapSec']:.0f}s）")

    cov = report.get("coverage") or {}
    if cov:
        add(f"字段覆盖：读到 {cov.get('knownRatio', 0) * 100:.0f}%    "
            f"可信 {cov.get('trustedRatio', 0) * 100:.0f}%    "
            f"可用帧 {cov.get('usableFrames', 0)}/{cov.get('frames', 0)}")

    tally = report.get("sourceTally") or {}
    if tally:
        readable = {
            schema.SOURCE_MANUAL: "人工", schema.SOURCE_OCR: "识别",
            schema.SOURCE_CV: "小地图", schema.SOURCE_CARRIED: "沿用",
            schema.SOURCE_DERIVED: "推算",
            schema.SOURCE_UNKNOWN: "未知",
        }
        parts = [f"{readable.get(k, k)} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])]
        add("字段来源：" + "    ".join(parts))

    for title, key, mark in (
        ("必须先解决的问题", "problems", "❌"),
        ("需要注意", "warnings", "⚠️"),
        ("说明", "notes", "·"),
    ):
        items = report.get(key) or []
        if not items:
            continue
        add("")
        add(f"── {title} ──")
        for item in items:
            body = str(item).replace("\n", "\n     ")
            add(f"  {mark} {body}")

    violations = report.get("ruleViolations") or []
    if violations:
        add("")
        add(f"── 违反游戏规则的值（前 {len(violations)} 条）──")
        for item in violations[:15]:
            add(f"  {item['clock']}  {item['what']}")

    rejections = report.get("rejections") or []
    if rejections:
        add("")
        add(f"── 被拒绝的读数（共 {report.get('rejectionCount', 0)} 个，前 15 条）──")
        add("  （拒绝是好事：说明校验在干活。数量突然变多通常意味着标定漂了。）")
        for item in rejections[:15]:
            add(f"  {item['clock']}  {item['field']}：{item['reason']}")

    add("")
    add("=" * 72)
    add(("✅ " if report["verdict"]["ok"] else "❌ ") + report["verdict"]["summary"])
    add("=" * 72)
    return "\n".join(lines)
