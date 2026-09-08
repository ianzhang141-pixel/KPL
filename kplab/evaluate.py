"""用人工标注当标尺，评估自动识别到底准不准。

## 为什么必须有这一步

如果没有这个模块，「OCR 识别」这件事就是不可证伪的：
识别出来一堆数字，看起来都挺像那么回事，没人知道其中有多少是错的。
等到几个月后训练出的胜率模型不对劲，已经没办法回头区分
「是模型的问题」还是「输入的数字本来就有 30% 是错的」。

LOL 项目里对应的教训写在项目状态第五节：**不要夸大验证程度**。
那边明确写着「用真实 Riot 数据跑通：❌ 从未验证」。
这边要做的是同一件事，只不过验证的对象从「接口通不通」变成「数字对不对」。

## 怎么比

只在**人工和机器都给出了值**的字段上比较。

  · 人工填了、机器没读出来 → 记为「漏读」，不算错
  · 机器读了、人工没填     → 无法评判，跳过（人工是标尺，标尺没刻度就量不了）
  · 两边都有               → 比较，得出准确率

分开报「完全一致率」和「接近率」：经济读成 12300 vs 12800 是小错，
读成 12300 vs 3800 是大错，两者对下游的伤害完全不同。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import schema, store

# 各字段允许的「接近」范围。经济数字大，允许相对误差；等级和比分必须完全对。
TOLERANCE: dict[str, dict[str, Any]] = {
    "blueGold": {"rel": 0.03, "label": "蓝方经济"},
    "redGold": {"rel": 0.03, "label": "红方经济"},
    "blueKills": {"abs": 0, "label": "蓝方击杀"},
    "redKills": {"abs": 0, "label": "红方击杀"},
    "level": {"abs": 0, "label": "等级"},
    "kills": {"abs": 0, "label": "击杀"},
    "deaths": {"abs": 0, "label": "死亡"},
    "assists": {"abs": 0, "label": "助攻"},
    "itemCount": {"abs": 0, "label": "装备数"},
    "x": {"abs": 0.05, "label": "小地图 x"},
    "y": {"abs": 0.05, "label": "小地图 y"},
}


def _close(field_name: str, truth: Any, guess: Any) -> bool:
    spec = TOLERANCE.get(field_name) or {"abs": 0}
    if not isinstance(truth, (int, float)) or not isinstance(guess, (int, float)):
        return truth == guess
    if "rel" in spec:
        return abs(truth - guess) <= max(1.0, abs(truth) * float(spec["rel"]))
    return abs(truth - guess) <= float(spec.get("abs", 0))


def _index_by_time(records: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    return {round(float(r.get("atSec") or 0.0), 1): r for r in records}


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    """把一条观测摊平成 {字段路径: schema 字段}。"""
    out: dict[str, Any] = {}
    for key, entry in (record.get("fields") or {}).items():
        out[key] = entry
    for slot, raw in (record.get("players") or {}).items():
        for name, entry in (raw or {}).items():
            if isinstance(entry, dict) and "source" in entry:
                out[f"p{slot}.{name}"] = entry
    return out


def _base_name(path: str) -> str:
    return path.split(".", 1)[1] if path.startswith("p") and "." in path else path


def run(data_dir: Path, game_id: str) -> dict[str, Any]:
    """比较人工标注与机器识别，输出准确率报告。"""
    annotations = list(store.read_jsonl_gz(store.annotations_path(data_dir, game_id)))
    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))

    # observations 里既有人工也有机器的，机器的部分是被评估对象
    machine = [
        o for o in observations
        if any(
            schema.src(e) == schema.SOURCE_OCR
            for e in _flatten(o).values()
        )
    ]

    report: dict[str, Any] = {
        "gameId": game_id,
        "annotationCount": len(annotations),
        "machineObservationCount": len(machine),
        "byField": {},
        "notes": [],
    }

    if not annotations:
        report["verdict"] = {
            "ok": False,
            "summary": "这场没有人工标注，无法评估识别准确率。"
                       "先标 5~10 帧（kplab annotate），才有标尺可用。",
        }
        return report
    if not machine:
        report["verdict"] = {
            "ok": False,
            "summary": "这场没有机器识别的观测，没有可评估的对象。"
                       "全部是人工标注的数据本身就是可信的，不需要评估。",
        }
        return report

    truth_by_time = _index_by_time(annotations)
    compared = exact = close = 0
    misses = 0
    per_field: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []

    for record in machine:
        at = round(float(record.get("atSec") or 0.0), 1)
        truth_record = truth_by_time.get(at)
        if truth_record is None:
            continue      # 这一帧没有人工标注，无从比较
        truth_fields = _flatten(truth_record)
        guess_fields = _flatten(record)

        for path, truth_entry in truth_fields.items():
            if not schema.is_known(truth_entry):
                continue
            guess_entry = guess_fields.get(path)
            name = _base_name(path)
            bucket = per_field.setdefault(
                name,
                {"label": (TOLERANCE.get(name) or {}).get("label", name),
                 "compared": 0, "exact": 0, "close": 0, "missed": 0},
            )
            if not schema.is_known(guess_entry):
                misses += 1
                bucket["missed"] += 1
                continue

            truth_value = schema.get(truth_entry)
            guess_value = schema.get(guess_entry)
            compared += 1
            bucket["compared"] += 1
            if truth_value == guess_value:
                exact += 1
                close += 1
                bucket["exact"] += 1
                bucket["close"] += 1
            elif _close(name, truth_value, guess_value):
                close += 1
                bucket["close"] += 1
                if len(examples) < 30:
                    examples.append({"clock": _clock(at), "field": path,
                                     "truth": truth_value, "guess": guess_value, "level": "接近"})
            else:
                if len(examples) < 30:
                    examples.append({"clock": _clock(at), "field": path,
                                     "truth": truth_value, "guess": guess_value, "level": "错"})

    for bucket in per_field.values():
        n = bucket["compared"]
        bucket["exactRate"] = round(bucket["exact"] / n, 3) if n else None
        bucket["closeRate"] = round(bucket["close"] / n, 3) if n else None

    report["byField"] = per_field
    report["compared"] = compared
    report["exact"] = exact
    report["close"] = close
    report["missed"] = misses
    report["exactRate"] = round(exact / compared, 3) if compared else None
    report["closeRate"] = round(close / compared, 3) if compared else None
    report["examples"] = examples

    if compared == 0:
        report["verdict"] = {
            "ok": False,
            "summary": "人工标注和机器识别没有落在同一个时刻，比不了。"
                       "标注时请对准抽帧的时间点（标注页上会列出可选时刻）。",
        }
        return report

    if compared < 30:
        report["notes"].append(
            f"只比较了 {compared} 个字段，样本太少，这个准确率不稳定。"
            "至少标到 100 个字段（大约 10~15 帧）再看结论。"
        )

    rate = report["exactRate"] or 0.0
    if rate >= 0.95:
        verdict = "识别准确率够高，可以考虑用它批量处理，但仍要定期抽样复核。"
        ok = True
    elif rate >= 0.8:
        verdict = ("识别大体可用，但每 5 个字段就有 1 个不对。"
                   "适合当「人工标注的草稿」用（先跑识别，再人工改），不适合直接进模型。")
        ok = False
    else:
        verdict = ("识别错得太多，不能用。先检查 HUD 标定是不是错位了 —— "
                   "覆盖率高但准确率低，最常见的原因就是框的位置不对。")
        ok = False

    report["verdict"] = {
        "ok": ok,
        "summary": f"完全一致 {rate * 100:.0f}%、可接受 {(report['closeRate'] or 0) * 100:.0f}%"
                   f"（比较 {compared} 个字段，另有 {misses} 个人工填了但机器没读出来）。{verdict}",
    }
    return report


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def format_text(report: dict[str, Any]) -> str:
    lines = ["=" * 72, f"识别准确率评估：{report['gameId']}", "=" * 72,
             f"人工标注 {report['annotationCount']} 条    "
             f"机器观测 {report['machineObservationCount']} 条"]
    by_field = report.get("byField") or {}
    if by_field:
        lines.append("")
        lines.append(f"{'字段':<14}{'比较':>6}{'完全一致':>10}{'可接受':>9}{'漏读':>7}")
        lines.append("-" * 50)
        for name, bucket in sorted(by_field.items(), key=lambda kv: -kv[1]["compared"]):
            exact = f"{bucket['exactRate'] * 100:.0f}%" if bucket["exactRate"] is not None else "—"
            near = f"{bucket['closeRate'] * 100:.0f}%" if bucket["closeRate"] is not None else "—"
            lines.append(f"{bucket['label']:<14}{bucket['compared']:>6}{exact:>10}{near:>9}{bucket['missed']:>7}")
    examples = report.get("examples") or []
    if examples:
        lines.append("")
        lines.append("── 不一致的例子 ──")
        for item in examples[:15]:
            lines.append(f"  {item['clock']}  {item['field']:<16}"
                         f"人工 {item['truth']}  机器 {item['guess']}  [{item['level']}]")
    for note in report.get("notes") or []:
        lines.append("")
        lines.append(f"  ⚠️  {note}")
    lines.append("")
    lines.append("=" * 72)
    lines.append(("✅ " if report["verdict"]["ok"] else "⚠️  ") + report["verdict"]["summary"])
    lines.append("=" * 72)
    return "\n".join(lines)
