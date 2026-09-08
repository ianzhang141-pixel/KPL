"""从画面区域读出数字 —— 可插拔的识别后端。

## 这个文件的核心立场

**没有可用的识别后端时，返回「不知道」，不返回数字。**

这听起来像废话，但它是这个项目最容易被破坏的一条规矩。
写代码的人（尤其是 AI）在补全逻辑时，很容易顺手写出
`gold = parsed or 0` 或者 `level = int(text) if text.isdigit() else 1`，
于是 State 里就多了一堆看起来正常、实际是凭空捏造的数字。
LOL 项目的教训是「没有模型时返回 0，不编造数字」；
在这里，编造的成本更高，因为下游根本看不出来。

## 后端

按可用性依次尝试，都不可用就整体不可用：

  · `manual`      —— 不是识别，是人工标注。永远可用，也永远最准（见 annotate.py）
  · `tesseract`   —— 外部程序，`brew install tesseract`
  · `paddleocr`   —— pip 包，中文数字识别效果好，但要装一堆依赖

装不上任何一个也不影响主流程：人工标注那条路完全不依赖 OCR，
而且**人工标注本来就是第一步要做的事**（要先有 ground truth 才能评估 OCR）。

## 为什么识别出来的值还要再过一遍规则校验

OCR 把 `8` 读成 `9` 是最常见的错误，读出来的还是一个合法数字，
光看值本身完全看不出问题。所以每个值都要过一遍
「这个位置的数字合不合理」的检查（等级不能是 27、经济不能倒退），
不合理的直接降置信度或判为未知。这一步在 `parse_int()` 和 `state.py` 里。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import hud, schema


class OcrError(RuntimeError):
    pass


# ---------------------------------------------------------------- 后端探测

def _which(name: str) -> str | None:
    return shutil.which(name)


def _paddle_available() -> bool:
    try:
        import paddleocr  # type: ignore # noqa: F401
    except Exception:      # noqa: BLE001 - 装一半 / 版本不对都算不可用
        return False
    return True


def backends() -> list[dict[str, Any]]:
    tess = _which("tesseract")
    return [
        {
            "name": "manual",
            "available": True,
            "label": "人工标注",
            "note": "不需要装任何东西。准确率最高，也是评估其他后端的标尺。",
            "install": "",
        },
        {
            "name": "tesseract",
            "available": bool(tess),
            "label": "Tesseract",
            "path": tess or "",
            "note": "外部程序。对记分板上的等宽数字够用，对艺术字体一般。",
            "install": "brew install tesseract",
        },
        {
            "name": "paddleocr",
            "available": _paddle_available(),
            "label": "PaddleOCR",
            "note": "识别中文和游戏字体更好，但依赖较重，装起来麻烦。",
            "install": "pip3 install paddleocr paddlepaddle",
        },
    ]


def best_backend() -> str | None:
    """返回可用的自动识别后端。只有人工可用时返回 None（不是错误）。"""
    for entry in backends():
        if entry["name"] != "manual" and entry["available"]:
            return str(entry["name"])
    return None


def status() -> dict[str, Any]:
    auto = best_backend()
    return {
        "backends": backends(),
        "autoBackend": auto,
        "autoAvailable": bool(auto),
        "message": (
            f"自动识别可用，使用 {auto}。识别结果仍需抽样核对。"
            if auto else
            "本机没有可用的自动识别后端。\n"
            "这**不影响**你现在就开始工作 —— 人工标注那条路不需要它，\n"
            "而且必须先有人工标注的 ground truth，才有办法评估自动识别准不准。"
        ),
    }


# ---------------------------------------------------------------- 文本解析

_CN_DIGITS = str.maketrans("〇零一二三四五六七八九", "00123456789")

# OCR 在游戏字体上的常见混淆。只在「整体解析失败」时才尝试，不无脑替换。
_CONFUSIONS = (("O", "0"), ("o", "0"), ("D", "0"), ("l", "1"), ("I", "1"),
               ("|", "1"), ("S", "5"), ("B", "8"), ("Z", "2"), ("g", "9"))


def parse_int(
    text: str,
    lo: int | None = None,
    hi: int | None = None,
) -> tuple[int | None, float]:
    """把 OCR 出来的一段文字解析成整数，返回 (值, 置信度)。

    解析不出来返回 (None, 0.0) —— **不返回 0**。
    超出 [lo, hi] 合理范围的也返回 None：与其给一个错的数，不如说不知道。
    """
    if not text:
        return None, 0.0
    raw = text.strip().translate(_CN_DIGITS)

    # 「2.3万」「1.8w」这种缩写：记分板上经济常这么显示
    wan = re.search(r"(\d+(?:\.\d+)?)\s*[万wW]", raw)
    if wan:
        try:
            value = int(round(float(wan.group(1)) * 10000))
        except ValueError:
            return None, 0.0
        # 缩写本身就丢了精度（2.3万 可能是 22500~23499），置信度要低
        return _bounded(value, lo, hi, 0.55)

    digits = re.sub(r"[^\d]", "", raw)
    if digits:
        try:
            value = int(digits)
        except ValueError:
            return None, 0.0
        # 原文里非数字字符越多，说明识别得越乱，置信度越低
        noise = 1.0 - min(len(raw) - len(digits), 4) * 0.12
        return _bounded(value, lo, hi, 0.75 * noise)

    # 一个数字都没解析出来，才试字符混淆替换，并把置信度压得很低
    fixed = raw
    for wrong, right in _CONFUSIONS:
        fixed = fixed.replace(wrong, right)
    digits = re.sub(r"[^\d]", "", fixed)
    if digits:
        try:
            return _bounded(int(digits), lo, hi, 0.3)
        except ValueError:
            return None, 0.0
    return None, 0.0


def _bounded(
    value: int, lo: int | None, hi: int | None, confidence: float
) -> tuple[int | None, float]:
    if lo is not None and value < lo:
        return None, 0.0
    if hi is not None and value > hi:
        return None, 0.0
    return value, round(max(0.0, min(1.0, confidence)), 3)


def parse_clock(text: str) -> tuple[int | None, float]:
    """把 "12:34" 解析成秒数。计时读错会让整场时间轴错位，所以格式要求最严。"""
    if not text:
        return None, 0.0
    # 先去掉所有空白：OCR 常在数字之间塞进空格，"1 2 : 3 4" 其实是 12:34
    raw = re.sub(r"\s+", "", text).translate(_CN_DIGITS)
    if re.search(r"[万wW]", raw):
        return None, 0.0        # 带「万」的是经济不是时间，别把 2.3万 读成 2:03
    # 秒数必须正好两位：计时器永远显示 12:34 / 2:05，不会是 2:3。
    # 放宽成 1~2 位会让 "2.3" 之类的东西也匹配上，那是最难发现的一类错。
    hit = re.search(r"(\d{1,2})[:：.。](\d{2})(?!\d)", raw)
    if not hit:
        return None, 0.0
    minutes, seconds = int(hit.group(1)), int(hit.group(2))
    if seconds >= 60 or minutes > 90:
        return None, 0.0
    # 分隔符被识别成 "." 而不是 ":" 时，本身就说明这一格识别得不干净
    return minutes * 60 + seconds, 0.85 if ":" in raw or "：" in raw else 0.6


# ---------------------------------------------------------------- 识别入口

def read_regions(
    image: Path,
    profile: dict[str, Any],
    rules_table: dict[str, Any],
    backend: str | None = None,
) -> dict[str, Any]:
    """对一张画面跑识别，返回 {区域名: schema 字段}。

    标定没完成就直接拒绝 —— 用错位置的框跑出来的数字，
    比没有数字更糟糕，因为它看起来是有效的。
    """
    verdict = hud.describe(profile)
    if not verdict["ok"]:
        raise OcrError(f"不能识别：{verdict['message']}")

    backend = backend or best_backend()
    if not backend:
        raise OcrError(
            "本机没有可用的自动识别后端，不能跑识别。\n"
            "现在就能做的是人工标注：kplab annotate <比赛编号>"
        )
    if backend == "tesseract":
        reader = _tesseract_read
    elif backend == "paddleocr":
        reader = _paddle_read
    else:
        raise OcrError(f"未知的识别后端：{backend}")

    from . import rules, video

    max_level = int(rules.value(rules_table, "maxLevel"))
    out: dict[str, Any] = {}
    tmp_dir = image.parent / ".ocr_tmp"

    for key, box in (profile.get("regions") or {}).items():
        spec = hud.REGIONS.get(key)
        if not spec or spec["read"] == "minimap":
            continue        # 小地图不是 OCR 的事，见 minimap_note()
        crop = tmp_dir / f"{image.stem}__{key}.png"
        try:
            video.crop_region(image, crop, 0.0, tuple(box), scale_width=320)
            text = reader(crop)
        except Exception as err:   # noqa: BLE001 - 单个区域失败不该毁掉整帧
            out[key] = schema.unknown()
            out.setdefault("_errors", {})[key] = str(err)[:160]  # type: ignore[index]
            continue
        finally:
            crop.unlink(missing_ok=True)

        if spec["read"] == "clock":
            value, confidence = parse_clock(text)
        elif key.endswith("Level"):
            value, confidence = parse_int(text, lo=1, hi=max_level)
        elif key.endswith("Gold"):
            value, confidence = parse_int(text, lo=0, hi=200000)
        elif key.endswith("Kills"):
            value, confidence = parse_int(text, lo=0, hi=200)
        else:
            value, confidence = parse_int(text)

        out[key] = (
            schema.field(value, schema.SOURCE_OCR, confidence)
            if value is not None else schema.unknown()
        )
        out.setdefault("_raw", {})[key] = text[:80]   # type: ignore[index]

    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return out


def _tesseract_read(image: Path) -> str:
    exe = _which("tesseract")
    if not exe:
        raise OcrError("tesseract 不在 PATH 里。")
    result = subprocess.run(
        [exe, str(image), "stdout", "--psm", "7",
         "-c", "tessedit_char_whitelist=0123456789:./万wW"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:
        raise OcrError(result.stderr.strip()[:200] or "tesseract 执行失败")
    return result.stdout.strip()


def _paddle_read(image: Path) -> str:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as err:  # noqa: BLE001
        raise OcrError(f"PaddleOCR 不可用：{err}") from err
    global _PADDLE
    if _PADDLE is None:
        _PADDLE = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
    result = _PADDLE.ocr(str(image), cls=False)
    parts: list[str] = []
    for block in result or []:
        for line in block or []:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)) and line[1]:
                parts.append(str(line[1][0]))
    return " ".join(parts)


_PADDLE: Any = None


def minimap_note() -> str:
    """小地图为什么不走 OCR。"""
    return (
        "小地图上的十个英雄不是文字，是彩色圆形头像，OCR 读不了。\n"
        "要得到位置需要的是模板匹配 / 颜色分割那类图像处理，属于另一条技术路线。\n"
        "本版本**没有实现**它，所以位置字段目前只能靠人工标注（在标注页上点小地图）。\n"
        "在它实现之前，「入侵野区值不值」这类依赖敌方位置的问题做不了 ——\n"
        "这一点和 LOL 项目里「外部模型不知道敌方打野在哪」是同一个限制。"
    )
