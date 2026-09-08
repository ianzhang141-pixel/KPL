"""录像 → 画面帧。依赖外部 ffmpeg，没装就明说，不假装能跑。

## 为什么这一步必须存在

LOL 那边是 `riot.fetch_timeline()` 一个 HTTPS 请求就拿到全部数据。
王者荣耀没有这个接口，唯一的公开信息载体是**比赛录像的画面**：
记分板上的经济和比分、头像旁边的等级、小地图上的十个点。

所以这边的「数据源」是一个视频文件，第一步是把它按固定间隔切成图片。

## 为什么用 ffmpeg 而不是 pip 装一个库

用户没有编程基础，装 Python 包这件事本身就是一道坎（LOL 项目的
`lolab` 因此立下「只用标准库」的规矩）。ffmpeg 是一个独立程序，
`brew install ffmpeg` 一条命令，而且 LOL 项目的 huya.py 下载录像时已经在用它了，
不引入新的东西。

## 没装 ffmpeg 会怎样

`probe()` 和 `extract_frames()` 会抛 `VideoError`，说清楚缺什么、怎么装。
**不会**退回到某种「估算」模式 —— 没有画面就是没有数据，
这时候产生任何数字都是编造。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoError(RuntimeError):
    pass


BROWSER_FRAME_MAX_BYTES = 8 * 1024 * 1024
MAX_BROWSER_FRAMES = 2000
MAX_BROWSER_SCAN_POINTS = 12000


SAMPLING_MODES = {
    "smart": "智能战术采样",
    "dense": "全程3秒精细采样",
    "fixed": "自定义固定间隔",
}


INSTALL_HINT = (
    "本机没有找到 ffmpeg。它是一个独立的程序，不是 Python 包。\n"
    "  在 Mac 上安装：先装 Homebrew（brew.sh），然后在终端里执行\n"
    "      brew install ffmpeg\n"
    "  装完后把这个窗口关掉重开，再试一次。"
)


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def available() -> dict[str, Any]:
    """给网页控制台看的自检结果。"""
    ffmpeg, ffprobe = ffmpeg_path(), ffprobe_path()
    return {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "ok": bool(ffmpeg and ffprobe),
        "hint": "" if (ffmpeg and ffprobe) else INSTALL_HINT,
    }


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as err:
        raise VideoError(INSTALL_HINT) from err
    except subprocess.TimeoutExpired as err:
        raise VideoError(f"ffmpeg 执行超过 {timeout} 秒还没结束，已中断。") from err


@dataclass
class VideoInfo:
    path: str
    durationSec: float
    width: int
    height: int
    fps: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "durationSec": round(self.durationSec, 2),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "aspect": round(self.width / self.height, 4) if self.height else None,
        }


def probe(video: Path) -> VideoInfo:
    """读出录像的时长和分辨率。分辨率决定 HUD 区域怎么换算。"""
    if not video.is_file():
        raise VideoError(f"找不到录像文件：{video}")
    exe = ffprobe_path()
    if not exe:
        raise VideoError(INSTALL_HINT)

    result = _run([
        exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(video),
    ], timeout=120)
    if result.returncode != 0:
        raise VideoError(f"ffprobe 读不了这个文件：{result.stderr.strip()[:300]}")

    try:
        parsed = json.loads(result.stdout)
        stream = (parsed.get("streams") or [{}])[0]
        duration = float((parsed.get("format") or {}).get("duration") or 0.0)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        rate = str(stream.get("avg_frame_rate") or "0/1")
        num, _, den = rate.partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as err:
        raise VideoError(f"ffprobe 的输出看不懂：{err}") from err

    if width <= 0 or height <= 0:
        raise VideoError("这个文件里没有视频流（宽高读出来是 0）。")
    return VideoInfo(str(video), duration, width, height, fps)


def extract_frames(
    video: Path,
    out_dir: Path,
    every_sec: float = 30.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    width: int = 1280,
    quality: int = 3,
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    """按固定间隔抽帧，返回 [{index, atSec, file}]。

    every_sec 默认 30 秒。选 30 而不是 60 的理由：
    王者荣耀一局通常 15~20 分钟，比 LOL 短得多，一分钟一帧太粗，
    很多团战和资源争夺会整个落在两帧之间看不见。

    抽出来的是 jpg 不是 png：核对画面用不上无损，png 会大 5~10 倍，
    存储那本账（见 store.py 开头）一样要算。
    """
    if every_sec <= 0:
        raise VideoError("抽帧间隔必须大于 0 秒。")
    exe = ffmpeg_path()
    if not exe:
        raise VideoError(INSTALL_HINT)

    info = probe(video)
    end = info.durationSec if end_sec is None else min(end_sec, info.durationSec)
    if end <= start_sec:
        raise VideoError(
            f"结束时间（{end:.0f}s）不晚于开始时间（{start_sec:.0f}s），没有可抽的区间。"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    # 逐帧 seek 而不是用 fps 滤镜一次抽完：
    # 滤镜方式要顺序解码整个文件，一小时的录像要跑很久；
    # 逐帧用 -ss 精确跳转，几十帧总共只要几秒，而且中途可以报进度、可以断点续。
    index = 0
    at = start_sec
    while at < end:
        target = out_dir / f"{index:05d}_{int(round(at))}s.jpg"
        if not target.is_file():
            result = _run([
                exe, "-nostdin", "-loglevel", "error",
                "-ss", f"{at:.3f}", "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-q:v", str(quality),
                "-y", str(target),
            ], timeout=120)
            if result.returncode != 0 or not target.is_file():
                # 单帧失败不该毁掉整场，记下来继续
                records.append({
                    "index": index, "atSec": round(at, 3), "file": None,
                    "error": result.stderr.strip()[:200] or "ffmpeg 没有输出文件",
                })
                index += 1
                at += every_sec
                continue
        records.append({"index": index, "atSec": round(at, 3), "file": target.name})
        if on_progress:
            on_progress(index + 1, int((end - start_sec) / every_sec) + 1)
        index += 1
        at += every_sec

    return records


def sampling_plan(
    duration_sec: float,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    mode: str = "smart",
    every_sec: float = 5.0,
) -> dict[str, Any]:
    """生成浏览器抽帧时间表。

    智能模式不冒充“理解了团战”：它保证越到后期基础帧越密，同时让浏览器
    每2秒做一次低清画面变化扫描，变化明显时保存该帧。0~4分钟基础帧最多
    隔8秒、4~10分钟最多隔5秒、10分钟后最多隔3秒。
    """
    try:
        duration = float(duration_sec)
        start = max(0.0, float(start_sec))
        end = duration if end_sec is None else min(duration, float(end_sec))
        custom = float(every_sec)
    except (TypeError, ValueError):
        raise VideoError("录像时长和抽帧时间必须是数字。") from None
    if not all(math.isfinite(value) for value in (duration, start, end, custom)):
        raise VideoError("录像时长和抽帧时间必须是有限数字。")
    if duration <= 0 or end <= start:
        raise VideoError("结束时间必须晚于开始时间。")
    if mode not in SAMPLING_MODES:
        raise VideoError("不支持这种抽帧模式。")
    if custom <= 0:
        raise VideoError("固定抽帧间隔必须大于0秒。")

    times: list[float] = []
    at = start
    while at < end:
        times.append(round(at, 3))
        if len(times) > MAX_BROWSER_FRAMES:
            raise VideoError("这段录像会超过2000张基础帧，请缩短范围或放大间隔。")
        if mode == "smart":
            interval = 8.0 if at < 240 else 5.0 if at < 600 else 3.0
        elif mode == "dense":
            interval = 3.0
        else:
            interval = custom
        at += interval
    if mode == "smart":
        scan = {round(value, 3) for value in times}
        scan_at = start
        while scan_at < end:
            scan.add(round(scan_at, 3))
            if len(scan) > MAX_BROWSER_SCAN_POINTS:
                raise VideoError("这段录像的智能扫描点过多，请缩短处理范围。")
            scan_at += 2.0
        scan_times = sorted(scan)
    else:
        scan_times = times
    return {
        "mode": mode,
        "modeLabel": SAMPLING_MODES[mode],
        "times": times,
        "baseFrames": len(times),
        "scanTimes": scan_times,
        "scanFrames": len(scan_times),
        "changeScanEverySec": 2 if mode == "smart" else None,
        "adaptive": mode == "smart",
        "maximumGapSec": 8 if mode == "smart" else (3 if mode == "dense" else custom),
    }


def save_browser_frame(
    out_dir: Path, at_sec: float, body: bytes, content_type: str
) -> dict[str, Any]:
    """保存浏览器在本机解码出来的一帧，不接收整段录像。

    这个入口只允许 JPEG/PNG，并同时检查文件头，避免网页把任意内容写进
    frames 目录。相同时间点重复提取会原子替换旧图，不产生重复垃圾文件。
    """
    try:
        at_sec = float(at_sec)
    except (TypeError, ValueError):
        raise VideoError("帧时间必须是数字。") from None
    if not math.isfinite(at_sec) or not 0 <= at_sec <= 24 * 60 * 60:
        raise VideoError("帧时间必须在 0 秒到 24 小时之间。")
    if not body:
        raise VideoError("收到的是空图片。")
    if len(body) > BROWSER_FRAME_MAX_BYTES:
        raise VideoError("单张帧图超过 8 MB，请降低截图宽度或画质。")

    mime = content_type.partition(";")[0].strip().lower()
    if mime == "image/jpeg" and body.startswith(b"\xff\xd8\xff") and body.endswith(b"\xff\xd9"):
        suffix = "jpg"
    elif mime == "image/png" and body.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = "png"
    else:
        raise VideoError("只接受浏览器生成的 JPEG 或 PNG 帧图。")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"browser_{at_sec:010.3f}s.{suffix}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(target)
    return {"ok": True, "file": target.name, "atSec": round(at_sec, 3),
            "bytes": len(body)}


def crop_region(
    video: Path, out_file: Path, at_sec: float,
    box: tuple[float, float, float, float],
    scale_width: int = 0,
) -> Path:
    """截出画面里的一块区域（相对坐标 0~1），用于标定 HUD 位置。

    box = (x, y, w, h)，都是相对整幅画面的比例，不是像素 ——
    这样同一套标定可以用在 1080p 和 720p 的录像上。
    """
    exe = ffmpeg_path()
    if not exe:
        raise VideoError(INSTALL_HINT)
    x, y, w, h = box
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise VideoError(f"区域超出画面范围：{box}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    crop = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"
    if scale_width > 0:
        crop += f",scale={scale_width}:-2"
    result = _run([
        exe, "-nostdin", "-loglevel", "error",
        "-ss", f"{at_sec:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", crop, "-q:v", "2",
        "-y", str(out_file),
    ], timeout=120)
    if result.returncode != 0 or not out_file.is_file():
        raise VideoError(f"截图失败：{result.stderr.strip()[:300]}")
    return out_file
