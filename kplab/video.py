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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


class VideoError(RuntimeError):
    pass


class FrameDispatcher:
    """把同一条解码流按各模块所需频率分发，避免重复解码。"""

    def __init__(self) -> None:
        self._consumers: dict[str, dict[str, Any]] = {}

    def subscribe(self, name: str, every_sec: float, callback: Any) -> None:
        interval = float(every_sec)
        if not name or interval < 0 or not math.isfinite(interval) or not callable(callback):
            raise ValueError("帧消费者名称、间隔或回调不合法。")
        self._consumers[name] = {"everySec": interval, "lastAt": None,
                                 "callback": callback, "delivered": 0}

    def dispatch(self, at_sec: float, frame: bytes) -> list[str]:
        delivered: list[str] = []
        for name, consumer in self._consumers.items():
            last = consumer["lastAt"]
            if last is not None and at_sec - last + 1e-9 < consumer["everySec"]:
                continue
            consumer["callback"](at_sec, frame)
            consumer["lastAt"] = at_sec
            consumer["delivered"] += 1
            delivered.append(name)
        return delivered

    def counts(self) -> dict[str, int]:
        return {name: int(value["delivered"]) for name, value in self._consumers.items()}


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


def _run_cancelable(cmd: list[str], timeout: int, should_cancel: Any = None) -> subprocess.CompletedProcess[str]:
    """运行短ffmpeg任务；服务停止时立刻终止子进程，而不是等线程池退出。"""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as err:
        raise VideoError(INSTALL_HINT) from err
    started = time.monotonic()
    while True:
        if should_cancel and should_cancel():
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        if time.monotonic() - started > timeout:
            process.kill()
            stdout, stderr = process.communicate()
            raise VideoError(f"ffmpeg 执行超过 {timeout} 秒还没结束，已中断。")
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            continue


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


def _remote(value: str) -> bool:
    return urlsplit(value).scheme.lower() in {"http", "https"}


def _header_argument(headers: dict[str, str] | None) -> str:
    """生成 ffmpeg 请求头；认证字段只在内存中传递，调用方不得记录。"""
    if not headers:
        return ""
    lines: list[str] = []
    for key, value in headers.items():
        clean_key = str(key).replace("\r", "").replace("\n", "").strip()
        clean_value = str(value).replace("\r", "").replace("\n", "").strip()
        if clean_key and clean_value:
            lines.append(f"{clean_key}: {clean_value}")
    return "\r\n".join(lines) + ("\r\n" if lines else "")


def _input_options(source: str, headers: dict[str, str] | None = None) -> list[str]:
    options: list[str] = []
    if _remote(source):
        options += ["-reconnect", "1", "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "8", "-rw_timeout", "30000000"]
    header_text = _header_argument(headers)
    if header_text:
        options += ["-headers", header_text]
    return options


def probe_input(source: str | Path, headers: dict[str, str] | None = None) -> VideoInfo:
    """读取本地或远程媒体信息，不把远程媒体保存到磁盘。"""
    raw = str(source)
    if not _remote(raw):
        return probe(Path(raw))
    exe = ffprobe_path()
    if not exe:
        raise VideoError(INSTALL_HINT)
    result = _run([
        exe, "-v", "error", *_input_options(raw, headers),
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,codec_name:format=duration",
        "-of", "json", raw,
    ], timeout=60)
    if result.returncode != 0:
        detail = result.stderr.strip()[-300:] or "无法访问远程媒体"
        raise VideoError(f"读取远程录像失败：{detail}")
    try:
        parsed = json.loads(result.stdout)
        stream = (parsed.get("streams") or [{}])[0]
        duration = float((parsed.get("format") or {}).get("duration") or 0.0)
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        numerator, _, denominator = str(stream.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(numerator) / float(denominator) if denominator and float(denominator) else 0.0
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as err:
        raise VideoError(f"远程录像信息无法识别：{err}") from err
    if width <= 0 or height <= 0:
        raise VideoError("远程地址中没有可用的视频画面。")
    return VideoInfo(raw, duration, width, height, fps)


def _jpeg_stream(pipe: Any, maximum_frame_bytes: int = BROWSER_FRAME_MAX_BYTES):
    buffer = b""
    while True:
        chunk = pipe.read(64 * 1024)
        if not chunk:
            break
        buffer += chunk
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                buffer = buffer[-1:]
                break
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    buffer = buffer[start:]
                if len(buffer) > maximum_frame_bytes:
                    raise VideoError("远程帧图超过8 MB，已停止以避免异常占用内存。")
                break
            frame = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            if len(frame) <= maximum_frame_bytes:
                yield frame


def stream_extract_frames(
    source: str,
    out_dir: Path,
    duration_sec: float,
    mode: str = "smart",
    every_sec: float = 5.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    width: int = 1280,
    quality: int = 4,
    headers: dict[str, str] | None = None,
    on_progress: Any = None,
    should_cancel: Any = None,
) -> list[dict[str, Any]]:
    """单次连续读取媒体流并保存稀疏证据帧，不生成完整录像副本。

    ffmpeg 只启动一次。未来的小地图、OCR、血量和事件模块应订阅同一条解码流，
    不得为每个模块重新下载或解码整场录像。
    """
    exe = ffmpeg_path()
    if not exe:
        raise VideoError(INSTALL_HINT)
    plan = sampling_plan(duration_sec, start_sec, end_sec, mode, every_sec)
    start = max(0.0, float(start_sec))
    end = min(float(duration_sec), float(end_sec)) if end_sec is not None else float(duration_sec)
    if mode == "smart":
        # prev_selected_t 让不同阶段使用不同最大间隔，同时保持一次顺序解码。
        select = (
            "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,"
            f"if(lt(t+{start:.3f}\\,240)\\,8\\,"
            f"if(lt(t+{start:.3f}\\,600)\\,5\\,3))),scale={width}:-2"
        )
    else:
        interval = 3.0 if mode == "dense" else float(every_sec)
        select = f"fps=1/{interval:.6f},scale={width}:-2"

    command = [exe, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    command += _input_options(source, headers)
    command += ["-i", source, "-t", f"{end - start:.3f}", "-an", "-vf", select,
                "-fps_mode", "vfr", "-c:v", "mjpeg", "-q:v", str(quality),
                "-f", "image2pipe", "pipe:1"]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as err:
        raise VideoError(INSTALL_HINT) from err
    assert process.stdout is not None and process.stderr is not None
    error_chunks: list[bytes] = []

    def drain_errors() -> None:
        while True:
            chunk = process.stderr.read(4096)
            if not chunk:
                return
            error_chunks.append(chunk)
            if sum(len(value) for value in error_chunks) > 64 * 1024:
                del error_chunks[:-4]

    error_thread = threading.Thread(target=drain_errors, daemon=True)
    error_thread.start()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def save_evidence(at: float, frame: bytes) -> None:
        index = len(records)
        target = out_dir / f"stream_{at:010.3f}s.jpg"
        temporary = target.with_suffix(".jpg.tmp")
        temporary.write_bytes(frame)
        temporary.replace(target)
        records.append({"index": index, "atSec": at, "file": target.name,
                        "bytes": len(frame)})

    dispatcher = FrameDispatcher()
    dispatcher.subscribe("evidence-writer", 0, save_evidence)
    try:
        for index, frame in enumerate(_jpeg_stream(process.stdout)):
            if should_cancel and should_cancel():
                process.terminate()
                break
            if index >= len(plan["times"]):
                process.terminate()
                break
            at = float(plan["times"][index])
            dispatcher.dispatch(at, frame)
            if on_progress:
                on_progress(index + 1, len(plan["times"]), at)
            if len(records) >= MAX_BROWSER_FRAMES:
                process.terminate()
                break
    except Exception:
        process.terminate()
        raise
    finally:
        process.stdout.close()
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        return_code = process.wait(timeout=5)
    error_thread.join(timeout=2)
    if return_code not in (0, -15):
        detail = b"".join(error_chunks).decode("utf-8", errors="replace").strip()[-400:]
        raise VideoError(f"远程录像流式读取失败：{detail or 'ffmpeg 没有输出画面'}")
    if not records:
        raise VideoError("远程录像没有产生可用画面。")
    return records


def sparse_seek_frames(
    source: str,
    out_dir: Path,
    duration_sec: float,
    every_sec: float = 60.0,
    width: int = 640,
    quality: int = 5,
    headers: dict[str, str] | None = None,
    on_progress: Any = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    mode: str = "fixed",
    relative_timestamps: bool = False,
    file_prefix: str = "scout",
    should_cancel: Any = None,
) -> list[dict[str, Any]]:
    """对超长回放做稀疏随机定位，不顺序下载或解码整段录像。

    侦察阶段只需要判断哪些区间是对局。逐点定位比从头连续解码两三个小时
    更省网络、磁盘和算力；连续失败三次就回到上层重新解析媒体地址。
    """
    exe = ffmpeg_path()
    if not exe:
        raise VideoError(INSTALL_HINT)
    start = max(0.0, float(start_sec))
    end = min(float(duration_sec), float(end_sec)) if end_sec is not None else float(duration_sec)
    if end <= start:
        raise VideoError("分段结束时间必须晚于开始时间。")
    plan_mode = "smart" if mode == "smart" else "fixed"
    relative_plan = sampling_plan(end - start, 0, None, plan_mode, every_sec)
    absolute_times = [round(start + float(at), 3) for at in relative_plan["times"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    # HLS 的部分平台清单不支持在整条 m3u8 上跨很远 seek。先读文本清单，
    # 把时间点映射到对应的几秒媒体段；这只下载清单和目标小段，不下载整段回放。
    targets = [(index, float(at), source, float(at))
               for index, at in enumerate(absolute_times)]
    if Path(urlsplit(source).path).suffix.lower() == ".m3u8":
        try:
            targets = _hls_scout_targets(source, headers, absolute_times)
        except VideoError:
            raise
        except Exception as err:
            raise VideoError(f"HLS播放清单无法读取：{err}") from err

    def capture(pair: tuple[int, float, str, float]) -> tuple[int, float, Path, str | None]:
        index, raw_at, media_source, seek_at = pair
        at = float(raw_at)
        record_at = at - start if relative_timestamps else at
        target = out_dir / f"{file_prefix}_{record_at:010.3f}s.jpg"
        if not target.is_file():
            command = [exe, "-nostdin", "-hide_banner", "-loglevel", "error",
                       "-ss", f"{seek_at:.3f}", *_input_options(media_source, headers),
                       "-i", media_source, "-frames:v", "1", "-an",
                       "-vf", f"scale={width}:-2", "-q:v", str(quality),
                       "-y", str(target)]
            result = _run_cancelable(command, timeout=60, should_cancel=should_cancel)
            if result.returncode != 0 or not target.is_file():
                return index, record_at, target, result.stderr.strip()[-300:] or "侦察点没有画面"
        return index, record_at, target, None

    # 三条稀疏读取彼此独立；并发过高反而可能触发视频站限流，因此固定为3。
    consecutive_failures = 0
    # 智能时间表可能有两个时间点落在同一个3~5秒媒体段，只保留一次，避免
    # 保存内容完全相同的重复帧。
    if Path(urlsplit(source).path).suffix.lower() == ".m3u8":
        unique: list[tuple[int, float, str, float]] = []
        seen_media: set[str] = set()
        for target in targets:
            if target[2] in seen_media:
                continue
            seen_media.add(target[2])
            unique.append(target)
        targets = [(new_index, target[1], target[2], target[3])
                   for new_index, target in enumerate(unique)]
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vod-scout") as pool:
        for index, at, target, error in pool.map(capture, targets):
            if error:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise VideoError(f"远程录像稀疏定位失败：{error}")
            else:
                consecutive_failures = 0
                records.append({"index": len(records), "atSec": at,
                                "file": target.name, "bytes": target.stat().st_size})
            if on_progress:
                on_progress(index + 1, len(targets), at)
    if not records:
        raise VideoError("远程录像没有产生可用侦察画面。")
    return records


def _hls_scout_targets(source: str, headers: dict[str, str] | None,
                       times: list[float]) -> list[tuple[int, float, str, float]]:
    """将绝对录像时间映射到 HLS 媒体小段及段内时间。"""
    request_headers = {str(key): str(value) for key, value in (headers or {}).items()
                      if "\n" not in str(key) and "\r" not in str(key)
                      and "\n" not in str(value) and "\r" not in str(value)}
    try:
        with urlopen(Request(source, headers=request_headers), timeout=30) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
    except Exception as err:
        raise VideoError(f"HLS播放清单读取失败：{err}") from err
    if len(payload) > 16 * 1024 * 1024:
        raise VideoError("HLS播放清单超过16 MB，已停止读取。")
    text = payload.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line.startswith("#EXT-X-STREAM-INF") for line in lines):
        variants = [line for line in lines if not line.startswith("#")]
        if not variants:
            raise VideoError("HLS主清单中没有可用媒体线路。")
        return _hls_scout_targets(urljoin(source, variants[0]), headers, times)

    segments: list[tuple[float, float, str]] = []
    cursor = 0.0
    pending_duration: float | None = None
    for line in lines:
        if line.startswith("#EXTINF:"):
            try:
                pending_duration = float(line.split(":", 1)[1].split(",", 1)[0])
            except (ValueError, IndexError):
                pending_duration = None
        elif not line.startswith("#") and pending_duration is not None:
            duration = max(0.001, pending_duration)
            segments.append((cursor, cursor + duration, urljoin(source, line)))
            cursor += duration
            pending_duration = None
    if not segments:
        raise VideoError("HLS播放清单中没有可定位的媒体小段。")

    output: list[tuple[int, float, str, float]] = []
    segment_index = 0
    for index, raw_at in enumerate(times):
        at = float(raw_at)
        while segment_index + 1 < len(segments) and at >= segments[segment_index][1]:
            segment_index += 1
        start, _end, segment_url = segments[segment_index]
        # 从媒体小段开头解码最接近的关键帧。若先在只有数秒的 TS 小段内 seek，
        # 很容易跳过唯一关键帧并得到“Nothing was written”。侦察允许几秒误差。
        output.append((index, at, segment_url, 0.0))
    return output


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
    # OCR 传入的通常已经是单张 JPG/PNG。对单帧图片在输入前加
    # ``-ss 0`` 时，某些 ffmpeg 版本会返回 0 却不产生输出文件。
    # 只在真正需要跳转时才加 -ss，这样图片 OCR 和视频截帧都可用。
    seek = ["-ss", f"{at_sec:.3f}"] if at_sec > 0 else []
    result = _run([
        exe, "-nostdin", "-loglevel", "error", *seek, "-i", str(video),
        "-frames:v", "1", "-vf", crop, "-q:v", "2",
        "-y", str(out_file),
    ], timeout=120)
    if result.returncode != 0 or not out_file.is_file():
        raise VideoError(f"截图失败：{result.stderr.strip()[:300]}")
    return out_file
