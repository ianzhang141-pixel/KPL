"""本地控制台。默认 8020 端口，只监听 127.0.0.1。

端口选 8020 是为了和 LOL 那边错开：那边 8000 是 GPT 的网站、8010 是 lolab。
三个可以同时开着互不干扰。

只用标准库 http.server。目标和 lolab 一样：
**让完全不用终端的人也能把整条链路走完。**
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import (__version__, annotate, check, economy_s44, evaluate, hud, knowledge_s44,
               minimap, model, ocr, paths, rules, samples, schema, season_s44, sources,
               state as state_mod, store, video)

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 4 * 1024 * 1024


class Console:
    """服务器运行期间的可变状态：当前数据目录 + 当前后台任务。"""

    def __init__(self, data_dir: Path) -> None:
        self.lock = threading.Lock()
        self.data_dir = data_dir
        self.job: dict[str, Any] = _idle_job()

    def set_data_dir(self, path: Path) -> None:
        with self.lock:
            self.data_dir = path

    def start_job(self, title: str, work: Callable[[Callable[[str], None]], None]) -> dict[str, Any]:
        with self.lock:
            if self.job["running"]:
                return {"started": False, "reason": "已经有一个任务在跑，等它结束。"}
            self.job = {"running": True, "title": title, "lines": [], "error": None,
                        "done": 0, "total": 0, "finished": False,
                        "startedAt": time.strftime("%H:%M:%S")}

        def log(line: str) -> None:
            with self.lock:
                self.job["lines"].append(f"{time.strftime('%H:%M:%S')}  {line}")
                self.job["lines"] = self.job["lines"][-400:]

        def runner() -> None:
            try:
                work(log)
            except Exception as err:      # noqa: BLE001 - 后台出错必须让人看见
                with self.lock:
                    self.job["error"] = f"{type(err).__name__}: {err}"
                log(f"❌ {type(err).__name__}: {err}")
            finally:
                with self.lock:
                    self.job["running"] = False
                    self.job["finished"] = True

        threading.Thread(target=runner, daemon=True).start()
        return {"started": True}

    def progress(self, done: int | None = None, total: int | None = None) -> None:
        with self.lock:
            if done is not None:
                self.job["done"] = done
            if total is not None:
                self.job["total"] = total

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.job, ensure_ascii=False))


def _idle_job() -> dict[str, Any]:
    return {"running": False, "title": "", "lines": [], "error": None,
            "done": 0, "total": 0, "finished": False, "startedAt": ""}


CONSOLE: Console | None = None


def _console() -> Console:
    assert CONSOLE is not None, "服务器还没启动"
    return CONSOLE


# ---------------------------------------------------------------- 接口

def api_status() -> dict[str, Any]:
    data_dir = _console().data_dir
    rules_table = rules.load(data_dir)
    pending = rules.unverified(rules_table)
    return {
        "version": __version__,
        "dataDir": str(data_dir),
        "dataDirExists": data_dir.is_dir(),
        "inventory": store.inventory(data_dir) if data_dir.is_dir() else
                     {"games": [], "total": 0, "totalBytesHuman": "0 B"},
        "video": video.available(),
        "videoSources": sources.available(),
        "sourceBatches": sources.list_batches(data_dir),
        "ocr": ocr.status(),
        "rules": {
            "unverified": pending,
            "unverifiedCount": len(pending),
            "table": rules_table,
        },
        "profiles": {
            name: {**profile, "verdict": hud.describe(profile)}
            for name, profile in hud.load_profiles(data_dir).items()
        },
        "minimapNote": ocr.minimap_note(),
        "minimap": minimap.load_thresholds(data_dir),
        "sampleTypes": samples.public_types(),
        "seasonRules": season_s44.payload(),
        "economyRules": economy_s44.payload(),
        "generalKnowledge": knowledge_s44.payload(),
    }


def api_create_game(body: dict[str, Any]) -> dict[str, Any]:
    game_id = str(body.get("gameId") or "").strip()
    if not store.valid_game_id(game_id):
        return {"ok": False, "error":
                "比赛编号只能用字母、数字、下划线、点、短横线，"
                "建议格式：KPL2026S1_AG_vs_EST_G1"}
    data_dir = paths.ensure(_console().data_dir)
    try:
        sample_meta = samples.normalize_metadata(body)
    except samples.SampleError as err:
        return {"ok": False, "error": str(err)}
    meta = {k: body.get(k) for k in
            ("title", "blueTeam", "redTeam", "sourceKind", "sourceUrl", "note")
            if body.get(k) is not None}
    meta.update(sample_meta)
    if body.get("blueWin") is not None:
        meta["blueWin"] = bool(body["blueWin"])
    store.create_game(data_dir, game_id, meta)
    return {"ok": True, "gameId": game_id, "summary": store.game_summary(data_dir, game_id)}


def api_create_batch(body: dict[str, Any]) -> dict[str, Any]:
    """一次为多段录像建档，返回与文件输入顺序一致的编号。"""
    files = body.get("files")
    if not isinstance(files, list) or not files:
        return {"ok": False, "error": "请至少选择一个录像文件。"}
    if len(files) > 100:
        return {"ok": False, "error": "一次最多处理100个录像文件。"}
    try:
        common = samples.normalize_metadata(body)
    except samples.SampleError as err:
        return {"ok": False, "error": str(err)}
    data_dir = paths.ensure(_console().data_dir)
    prefix = str(body.get("batchPrefix") or time.strftime("VIDEO_%Y%m%d_%H%M%S"))
    prepared: list[tuple[str, int]] = []
    for index, item in enumerate(files, 1):
        if not isinstance(item, dict):
            return {"ok": False, "error": f"第{index}个录像信息不完整。"}
        name = Path(str(item.get("name") or f"录像{index}")).name[:180]
        try:
            size = int(item.get("sizeBytes") or 0)
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            return {"ok": False, "error": f"{name}是空文件或大小无法读取。"}
        prepared.append((name, size))

    created: list[dict[str, Any]] = []
    reserved: set[str] = set()
    for index, (name, size) in enumerate(prepared, 1):
        game_id = store.next_game_id(data_dir, prefix, index)
        while game_id in reserved:
            game_id = store.next_game_id(data_dir, game_id, 1)
        reserved.add(game_id)
        meta = {
            **common,
            "title": name,
            "sourceKind": "browser-video",
            "videoName": name,
            "videoSizeBytes": size,
            "batchPrefix": prefix[:80],
            "batchIndex": index,
        }
        store.create_game(data_dir, game_id, meta)
        created.append({"gameId": game_id, "fileName": name, "index": index})
    return {"ok": True, "created": created, "count": len(created)}


def api_sampling_plan(body: dict[str, Any]) -> dict[str, Any]:
    try:
        plan = video.sampling_plan(
            body.get("durationSec"), body.get("startSec") or 0,
            body.get("endSec") if body.get("endSec") not in (None, "") else None,
            str(body.get("mode") or "smart"), body.get("everySec") or 5,
        )
    except video.VideoError as err:
        return {"ok": False, "error": str(err)}
    return {"ok": True, **plan}


def api_source_preview(body: dict[str, Any]) -> dict[str, Any]:
    """解析多行网址/CSV，逐项返回；一个坏网址不影响其余项目。"""
    try:
        entries = sources.parse_submission(body.get("text"), body.get("csvText"))
    except sources.SourceError as err:
        return {"ok": False, "errorCode": err.code,
                "error": sources.ERROR_MESSAGES[err.code]}
    cookie_browser = str(body.get("cookieBrowser") or "").lower()
    if cookie_browser not in sources.COOKIE_BROWSERS:
        return {"ok": False, "errorCode": "INVALID_SOURCE",
                "error": "不支持这个浏览器登录状态。"}

    def resolve_entry(pair: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        input_index, entry = pair
        origin = entry["originUrl"]
        try:
            resolved_items = sources.resolve(origin, cookie_browser,
                                             metadata=entry.get("metadata") or {})
            output: list[dict[str, Any]] = []
            for resolved in resolved_items:
                public = resolved.public()
                public.update({
                    "inputIndex": input_index,
                    "status": "READY",
                    "duplicateGameId": sources.find_duplicate(
                        _console().data_dir, resolved.mediaFingerprint
                    ),
                })
                output.append(public)
            return output
        except sources.SourceError as err:
            try:
                detected = sources.detect(origin)
            except sources.SourceError:
                detected = {"platform": "unknown", "sourceType": "INVALID_SOURCE",
                            "originUrl": str(origin)}
            return [{
                **detected, "inputIndex": input_index, "title": "无法解析",
                "status": "FAILED", "errorCode": err.code,
                "error": sources.ERROR_MESSAGES[err.code],
                "metadata": entry.get("metadata") or {}, "sourceUrlStored": False,
            }]

    # 解析时间主要消耗在等待平台响应。并发限制为4，既缩短大清单预览时间，
    # 又避免对网站和本机造成过高瞬时压力；map 保持用户提交的原始顺序。
    workers = min(4, len(entries))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-preview") as pool:
        groups = pool.map(resolve_entry, enumerate(entries, 1))
        items = [item for group in groups for item in group]
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("platform") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"ok": True, "submitted": len(entries), "items": items,
            "resolved": sum(1 for item in items if item["status"] == "READY"),
            "failed": sum(1 for item in items if item["status"] == "FAILED"),
            "platformCounts": counts}


def _source_batch_id(data_dir: Path) -> str:
    base = time.strftime("SOURCE_%Y%m%d_%H%M%S")
    candidate, suffix = base, 2
    while sources.batch_path(data_dir, candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _source_processing_config(batch: dict[str, Any]) -> tuple[str, float, float, float | None]:
    """读取可恢复的处理参数；旧批次没有此字段时使用当时的默认值。"""
    raw = batch.get("processing") if isinstance(batch.get("processing"), dict) else {}
    mode = str(raw.get("samplingMode") or "smart")
    if mode not in video.SAMPLING_MODES:
        mode = "smart"
    try:
        every = float(raw.get("everySec") or 5)
        start_sec = float(raw.get("startSec") or 0)
        end_raw = raw.get("endSec")
        end_sec = float(end_raw) if end_raw not in (None, "") else None
    except (TypeError, ValueError):
        return "smart", 5.0, 0.0, None
    if (not math.isfinite(every) or every <= 0 or not math.isfinite(start_sec)
            or start_sec < 0 or (end_sec is not None and
                                 (not math.isfinite(end_sec) or end_sec <= start_sec))):
        return "smart", 5.0, 0.0, None
    return mode, every, start_sec, end_sec


def _job_processing_config(batch: dict[str, Any], job: dict[str, Any]) -> tuple[str, float, float, float | None]:
    mode, every, start_sec, end_sec = _source_processing_config(batch)
    if job.get("segmentStartSec") is None or job.get("segmentEndSec") is None:
        return mode, every, start_sec, end_sec
    try:
        segment_start = float(job["segmentStartSec"])
        segment_end = float(job["segmentEndSec"])
    except (TypeError, ValueError):
        return mode, every, start_sec, end_sec
    if not math.isfinite(segment_start) or not math.isfinite(segment_end) or segment_end <= segment_start:
        return mode, every, start_sec, end_sec
    return "smart", every, segment_start, segment_end


def _source_batches_work(data_dir: Path, batches: list[dict[str, Any]]) -> Callable[[Callable[[str], None]], None]:
    """为新批次和重启恢复批次生成同一条处理链，避免两套逻辑漂移。"""
    eligible = [(batch, job) for batch in batches
                for job in (batch.get("jobs") or [])
                if isinstance(job, dict) and job.get("status") in {"PENDING", "RUNNING"}]

    def work(log: Callable[[str], None]) -> None:
        cleaned = sources.cleanup_cache(data_dir)
        if cleaned["removedFiles"]:
            log(f"已清理 {cleaned['removedFiles']} 个过期临时缓存文件。")
        completed = scouted = failed = 0
        total = len(eligible)
        resolved_cache: dict[tuple[str, str, int | None], sources.ResolvedVideoSource] = {}
        _console().progress(0, total)
        for position, (batch, job) in enumerate(eligible, 1):
            batch_id = str(batch.get("batchId") or "")
            cookie_browser = str(batch.get("cookieBrowser") or "").lower()
            mode, every, start_sec, end_sec = _job_processing_config(batch, job)
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            job.update({"status": "RUNNING", "stage": "RESOLVING",
                        "startedAt": job.get("startedAt") or now,
                        "resumedAt": now if job.get("recoveredAfterRestart") else None,
                        "finishedAt": None, "errorCode": None, "errorMessage": None})
            sources.save_batch(data_dir, batch)
            prefix = "恢复" if job.get("recoveredAfterRestart") else "解析"
            log(f"[{position}/{total}] {prefix}来源：{job['title']}")
            last_error_code = "RESOLVER_ERROR"
            last_error_message = "无法处理该录像。"
            resolution_cookie_browser = cookie_browser
            for attempt in range(1, sources.MAX_RESOLVE_RETRIES + 1):
                job["attemptCount"] = attempt
                job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                sources.save_batch(data_dir, batch)
                try:
                    playlist_index = int(job["playlistIndex"]) if job.get("playlistIndex") else None
                    cache_key = (str(job["originUrl"]), resolution_cookie_browser, playlist_index)
                    resolved = resolved_cache.get(cache_key)
                    if resolved is None:
                        resolved_list = sources.resolve(
                            job["originUrl"], resolution_cookie_browser, playlist_index,
                        )
                        resolved = resolved_list[0]
                        resolved_cache[cache_key] = resolved
                    if not resolved.sourceUrl:
                        raise sources.SourceError("NO_VIDEO_STREAM")
                    job["stage"] = "PROBING"
                    sources.save_batch(data_dir, batch)
                    info = video.probe_input(resolved.sourceUrl, resolved.headers)
                    if info.durationSec <= 0:
                        raise video.VideoError("无法确认录像时长；暂不处理直播或无限流。")
                    job["stage"] = "STREAMING"
                    log(f"开始流式分析：{info.width}×{info.height}，{info.durationSec / 60:.1f}分钟")

                    def progress(done: int, frame_total: int, at_sec: float) -> None:
                        job["progress"] = round(done / max(1, frame_total), 4)
                        job["currentVideoTime"] = round(at_sec, 3)
                        _console().progress(position - 1 + job["progress"], total)
                        if done == 1 or done == frame_total or done % 10 == 0:
                            job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                            sources.save_batch(data_dir, batch)

                    if job.get("sourceParentGameId"):
                        records = video.sparse_seek_frames(
                            resolved.sourceUrl,
                            store.frames_dir(data_dir, job["gameId"]),
                            info.durationSec, every_sec=every, width=960,
                            headers=resolved.headers, on_progress=progress,
                            start_sec=start_sec, end_sec=end_sec, mode="smart",
                            relative_timestamps=True, file_prefix="stream",
                        )
                    else:
                        records = video.stream_extract_frames(
                            resolved.sourceUrl, store.frames_dir(data_dir, job["gameId"]),
                            info.durationSec, mode, every, start_sec, end_sec,
                            headers=resolved.headers, on_progress=progress,
                        )
                    segment_duration = ((end_sec - start_sec) if job.get("sourceParentGameId")
                                        and end_sec is not None else info.durationSec)
                    store.create_game(data_dir, job["gameId"], {
                        "videoDurationSec": round(segment_duration, 3),
                        "sourceVodDurationSec": round(info.durationSec, 3),
                        "videoWidth": info.width, "videoHeight": info.height,
                        "samplingMode": mode, "frameEverySec": every if mode == "fixed" else None,
                        "savedFrameCount": len(records), "sourceResolvedAt": resolved.resolvedAt,
                        "sourceUrlStored": False, "fullVideoStored": False,
                    })
                    job.update({"status": "DONE", "stage": "COMPLETE", "progress": 1.0,
                                "currentVideoTime": round(segment_duration, 3),
                                "errorCode": None, "errorMessage": None,
                                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                    completed += 1
                    log(f"✅ {job['gameId']}：保存 {len(records)} 张稀疏证据帧，没有保存完整录像。")
                    break
                except sources.SourceError as err:
                    last_error_code = err.code
                    last_error_message = sources.ERROR_MESSAGES[err.code]
                    if err.code not in {"NETWORK_ERROR", "MEDIA_URL_EXPIRED"}:
                        break
                    if err.code == "NETWORK_ERROR" and resolution_cookie_browser:
                        resolved_cache.pop((str(job["originUrl"]), resolution_cookie_browser,
                                            int(job["playlistIndex"]) if job.get("playlistIndex") else None), None)
                        resolution_cookie_browser = ""
                        log("读取浏览器登录状态超时，改用公开页面继续；不会影响后续登录受限录像。")
                    log(f"第{attempt}次解析失败，将重新获取媒体地址。")
                except video.VideoError as err:
                    if (mode == "smart" and "超过2000张基础帧" in str(err)
                            and start_sec == 0 and end_sec is None):
                        try:
                            job["stage"] = "SCOUTING"
                            sources.save_batch(data_dir, batch)
                            log("检测到长直播回放：先按60秒间隔生成低清侦察帧，再划分比赛区间。")

                            def scout_progress(done: int, frame_total: int, at_sec: float) -> None:
                                job["progress"] = round(done / max(1, frame_total), 4)
                                job["currentVideoTime"] = round(at_sec, 3)
                                _console().progress(position - 1 + job["progress"], total)
                                if done == 1 or done == frame_total or done % 10 == 0:
                                    job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                                    sources.save_batch(data_dir, batch)

                            records = video.sparse_seek_frames(
                                resolved.scoutSourceUrl or resolved.sourceUrl,
                                store.frames_dir(data_dir, job["gameId"]),
                                info.durationSec, every_sec=60.0, width=640,
                                headers=resolved.headers, on_progress=scout_progress,
                            )
                            store.create_game(data_dir, job["gameId"], {
                                "videoDurationSec": round(info.durationSec, 3),
                                "videoWidth": info.width, "videoHeight": info.height,
                                "samplingMode": "long-source-scout", "frameEverySec": 60.0,
                                "savedFrameCount": len(records),
                                "sourceResolvedAt": resolved.resolvedAt,
                                "sourceUrlStored": False, "fullVideoStored": False,
                                "sourceSegmentationRequired": True,
                                "analysisStatus": "SCOUT_COMPLETE_NEEDS_SEGMENTATION",
                            })
                            job.update({"status": "SCOUT_DONE", "stage": "SCOUT_COMPLETE",
                                        "progress": 1.0,
                                        "currentVideoTime": round(info.durationSec, 3),
                                        "errorCode": None, "errorMessage": None,
                                        "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                            scouted += 1
                            log(f"✅ {job['gameId']}：保存 {len(records)} 张低清侦察帧，等待自动划分比赛区间。")
                            break
                        except video.VideoError as scout_err:
                            last_error_code = sources.classify_media_error(scout_err)
                            last_error_message = sources.safe_error(scout_err)
                            if (last_error_code in {"NETWORK_ERROR", "MEDIA_URL_EXPIRED"}
                                    and attempt < sources.MAX_RESOLVE_RETRIES):
                                log(f"侦察流中断，第{attempt}次重新解析原始网页地址。")
                                continue
                            break
                    last_error_code = sources.classify_media_error(err)
                    last_error_message = sources.safe_error(err)
                    if (last_error_code in {"NETWORK_ERROR", "MEDIA_URL_EXPIRED"}
                            and attempt < sources.MAX_RESOLVE_RETRIES):
                        log(f"媒体流中断，第{attempt}次重新解析原始网页地址。")
                    else:
                        break
            if job.get("status") not in {"DONE", "SCOUT_DONE"}:
                job.update({"status": "FAILED", "stage": "FAILED",
                            "errorCode": last_error_code, "errorMessage": last_error_message,
                            "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                failed += 1
                log(f"❌ {job['gameId']}：{last_error_message} 后续任务继续。")
            job["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            sources.save_batch(data_dir, batch)
            _console().progress(position, total)
            if not any(item.get("status") in {"PENDING", "RUNNING"}
                       for item in (batch.get("jobs") or []) if isinstance(item, dict)):
                batch["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                batch_jobs = [item for item in (batch.get("jobs") or []) if isinstance(item, dict)]
                batch["result"] = {
                    "done": sum(1 for item in batch_jobs if item.get("status") == "DONE"),
                    "scoutDone": sum(1 for item in batch_jobs if item.get("status") == "SCOUT_DONE"),
                    "failed": sum(1 for item in batch_jobs if item.get("status") == "FAILED"),
                }
                sources.save_batch(data_dir, batch)
                parent_game_id = str(batch.get("parentGameId") or "")
                if parent_game_id and store.valid_game_id(parent_game_id):
                    failures = batch["result"]["failed"]
                    store.create_game(data_dir, parent_game_id, {
                        "analysisStatus": ("SEGMENT_EXTRACTION_COMPLETE" if failures == 0
                                           else "SEGMENT_EXTRACTION_PARTIAL"),
                        "segmentExtractionResult": batch["result"],
                        "segmentExtractionFinishedAt": batch["finishedAt"],
                    })
        log(f"录像任务完成：完整处理{completed}个，长录像侦察{scouted}个，失败{failed}个。")

    return work


def _start_persisted_source_batches(batches: list[dict[str, Any]], title: str) -> dict[str, Any]:
    eligible = sum(1 for batch in batches for job in (batch.get("jobs") or [])
                   if isinstance(job, dict) and job.get("status") in {"PENDING", "RUNNING"})
    if not eligible:
        return {"started": False, "reason": "没有等待恢复的录像任务。"}
    return _console().start_job(title, _source_batches_work(_console().data_dir, batches))


def api_source_resume(body: dict[str, Any]) -> dict[str, Any]:
    """恢复一个中断批次；失败项只有在明确要求时才重试。"""
    if _console().snapshot().get("running"):
        return {"started": False, "reason": "已有任务正在运行，请等它完成。"}
    batch_id = str(body.get("batchId") or "").strip()
    batch = sources.load_batch(_console().data_dir, batch_id) if batch_id else None
    if not batch:
        return {"started": False, "reason": "找不到这个录像批次。"}
    sources.prepare_interrupted_batch(_console().data_dir, batch)
    if bool(body.get("includeFailed")):
        for job in batch.get("jobs") or []:
            if isinstance(job, dict) and job.get("status") == "FAILED":
                job.update({"status": "PENDING", "stage": "RETRY_QUEUED",
                            "progress": 0.0, "currentVideoTime": 0.0,
                            "errorCode": None, "errorMessage": None, "finishedAt": None})
        sources.save_batch(_console().data_dir, batch)
    started = _start_persisted_source_batches([batch], f"恢复录像批次：{batch_id}")
    return {**started, "batchId": batch_id}


def resume_interrupted_source_batches() -> dict[str, Any]:
    """本地服务重启后自动接管未完成任务，不需要用户重新提交网址。"""
    if _console().snapshot().get("running"):
        return {"started": False, "reason": "已有任务正在运行。"}
    batches = sources.interrupted_batches(_console().data_dir)
    if not batches:
        return {"started": False, "reason": "没有中断任务。"}
    recovered = sum(sources.prepare_interrupted_batch(_console().data_dir, batch)
                    for batch in batches)
    started = _start_persisted_source_batches(batches, "自动恢复中断的录像任务")
    return {**started, "batchCount": len(batches), "recoveredJobs": recovered}


def api_source_start(body: dict[str, Any]) -> dict[str, Any]:
    """创建远程来源批次，并用一条连续媒体流逐场处理。"""
    if _console().snapshot().get("running"):
        return {"started": False, "reason": "已有任务正在运行，请等它完成。"}
    requested = body.get("items")
    if not isinstance(requested, list) or not requested:
        return {"started": False, "reason": "请先解析并选择至少一个录像来源。"}
    if len(requested) > sources.MAX_SOURCES:
        return {"started": False, "reason": f"一次最多处理{sources.MAX_SOURCES}个来源。"}
    try:
        common = samples.normalize_metadata(body)
    except samples.SampleError as err:
        return {"started": False, "reason": str(err)}
    try:
        every = float(body.get("everySec") or 5)
        start_sec = float(body.get("startSec") or 0)
        end_value = body.get("endSec")
        end_sec = float(end_value) if end_value not in (None, "") else None
    except (TypeError, ValueError):
        return {"started": False, "reason": "抽帧范围必须是数字。"}
    if (not math.isfinite(every) or not math.isfinite(start_sec) or every <= 0
            or start_sec < 0 or (end_sec is not None and
                                 (not math.isfinite(end_sec) or end_sec <= start_sec))):
        return {"started": False, "reason": "抽帧起止时间或间隔不合法。"}
    mode = str(body.get("samplingMode") or "smart")
    if mode not in video.SAMPLING_MODES:
        return {"started": False, "reason": "不支持这种抽帧方式。"}
    cookie_browser = str(body.get("cookieBrowser") or "").lower()
    if cookie_browser not in sources.COOKIE_BROWSERS:
        return {"started": False, "reason": "不支持这个浏览器登录状态。"}

    data_dir = paths.ensure(_console().data_dir)
    batch_id = _source_batch_id(data_dir)
    prefix = str(body.get("batchPrefix") or batch_id)
    jobs: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    for index, raw in enumerate(requested, 1):
        if not isinstance(raw, dict) or raw.get("status") not in (None, "READY"):
            continue
        try:
            detected = sources.detect(raw.get("originUrl"))
        except sources.SourceError:
            continue
        fingerprint = str(raw.get("mediaFingerprint") or "")[:64]
        duplicate = sources.find_duplicate(data_dir, fingerprint)
        if duplicate and not bool(body.get("allowDuplicates")):
            continue
        game_id = store.next_game_id(data_dir, prefix, index)
        csv_meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        title = str(raw.get("title") or f"远程录像{index}")[:300]
        meta = {
            **common,
            "title": title,
            "sourceKind": "remote-stream",
            "videoImportMode": "remote-stream-no-full-copy",
            "originUrl": detected["originUrl"],
            "sourceType": detected["sourceType"],
            "sourcePlatform": str(raw.get("platform") or detected["platform"])[:40],
            "resolverName": str(raw.get("resolverName") or "")[:80],
            "mediaFingerprint": fingerprint,
            "sourceUrlStored": False,
            "sourceBatchId": batch_id,
            "playlistIndex": raw.get("playlistIndex"),
            "blueTeam": str(csv_meta.get("team") or "")[:80],
            "redTeam": str(csv_meta.get("opponent") or "")[:80],
            "tournament": str(csv_meta.get("event") or common.get("tournament") or "")[:100],
            "matchDate": str(csv_meta.get("date") or "")[:40],
            "note": str(csv_meta.get("note") or "")[:500],
        }
        store.create_game(data_dir, game_id, meta)
        job = {
            "id": f"{batch_id}_{index:03d}", "batchId": batch_id,
            "gameId": game_id, "originUrl": detected["originUrl"],
            "platform": meta["sourcePlatform"], "sourceType": detected["sourceType"],
            "title": title, "playlistIndex": raw.get("playlistIndex"),
            "mediaFingerprint": fingerprint, "status": "PENDING", "stage": "QUEUED",
            "progress": 0.0, "currentVideoTime": 0.0, "attemptCount": 0,
            "errorCode": None, "errorMessage": None,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "startedAt": None, "finishedAt": None, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        jobs.append(job)
        created.append({"gameId": game_id, "title": title, "index": index})
    if not jobs:
        return {"started": False, "reason": "没有可创建的任务；失败项或重复录像已自动跳过。"}
    batch = {"batchId": batch_id, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "cookieBrowser": cookie_browser or None, "sourceUrlStored": False,
             "processing": {"samplingMode": mode, "everySec": every,
                            "startSec": start_sec, "endSec": end_sec},
             "jobs": jobs}
    sources.save_batch(data_dir, batch)
    started = _start_persisted_source_batches([batch], f"流式录像批次：{batch_id}")
    return {**started, "batchId": batch_id, "created": created,
            "count": len(created), "skipped": len(requested) - len(created)}


def api_source_segments_start(body: dict[str, Any]) -> dict[str, Any]:
    """把长回放的已核对区间拆成独立比赛，并开始高密度稀疏抽帧。"""
    if _console().snapshot().get("running"):
        return {"started": False, "reason": "已有任务正在运行，请等它完成。"}
    batch_id = str(body.get("batchId") or "").strip()
    parent_game_id = str(body.get("parentGameId") or "").strip()
    source_batch = sources.load_batch(_console().data_dir, batch_id)
    parent_meta = store.load_meta(_console().data_dir, parent_game_id)
    if not source_batch or not parent_meta:
        return {"started": False, "reason": "找不到长录像批次或侦察档案。"}
    parent_job = next((job for job in source_batch.get("jobs") or []
                       if isinstance(job, dict) and job.get("gameId") == parent_game_id), None)
    if not parent_job:
        return {"started": False, "reason": "长录像档案与来源批次不匹配。"}
    raw_segments = body.get("segments") or parent_meta.get("sourceSegments") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        return {"started": False, "reason": "尚未找到可处理的比赛区间。"}
    if len(raw_segments) > sources.MAX_SOURCES:
        return {"started": False, "reason": f"一次最多处理{sources.MAX_SOURCES}个区间。"}
    segments: list[tuple[float, float, str]] = []
    for index, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, dict) or str(raw.get("status") or "READY") != "READY":
            continue
        try:
            start_sec, end_sec = float(raw.get("startSec")), float(raw.get("endSec"))
        except (TypeError, ValueError):
            return {"started": False, "reason": f"第{index}个比赛区间不是有效时间。"}
        if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
                or start_sec < 0 or end_sec - start_sec < 60 or end_sec - start_sec > 3600):
            return {"started": False, "reason": f"第{index}个比赛区间时长不合理。"}
        segments.append((start_sec, end_sec, str(raw.get("note") or "完整对局")[:120]))
    segments.sort()
    if not segments or any(current[0] < previous[1]
                           for previous, current in zip(segments, segments[1:])):
        return {"started": False, "reason": "比赛区间为空或相互重叠。"}

    data_dir = paths.ensure(_console().data_dir)
    segment_batch_id = _source_batch_id(data_dir)
    jobs: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    derived_keys = {"gameId", "sourceSegments", "excludedSourceFragments",
                    "savedFrameCount", "analysisStatus", "samplingMode", "frameEverySec",
                    "sourceSegmentationRequired", "videoDurationSec", "videoWidth",
                    "videoHeight", "segmentBatchId"}
    inherited = {key: value for key, value in parent_meta.items() if key not in derived_keys}
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for index, (start_sec, end_sec, note) in enumerate(segments, 1):
        game_id = store.next_game_id(data_dir, f"{parent_game_id}_GAME", index)
        title = f"{parent_meta.get('title') or '长录像'} · 第{index}场"
        store.create_game(data_dir, game_id, {
            **inherited, "title": title, "sourceParentGameId": parent_game_id,
            "sourceBatchId": segment_batch_id, "segmentIndex": index,
            "segmentStartSec": start_sec, "segmentEndSec": end_sec,
            "segmentDurationSec": round(end_sec - start_sec, 3),
            "segmentEvidence": note, "segmentBoundaryConfidence": 0.8,
            "segmentBoundaryIsTrainingTruth": False,
            "sourceSegmentationRequired": False, "analysisStatus": "DENSE_EXTRACTION_QUEUED",
            "sourceUrlStored": False, "fullVideoStored": False,
        })
        jobs.append({
            "id": f"{segment_batch_id}_{index:03d}", "batchId": segment_batch_id,
            "gameId": game_id, "sourceParentGameId": parent_game_id,
            "originUrl": parent_job["originUrl"], "platform": parent_job.get("platform"),
            "sourceType": parent_job.get("sourceType"), "title": title,
            "playlistIndex": parent_job.get("playlistIndex"),
            "segmentStartSec": start_sec, "segmentEndSec": end_sec,
            "status": "PENDING", "stage": "QUEUED", "progress": 0.0,
            "currentVideoTime": 0.0, "attemptCount": 0,
            "errorCode": None, "errorMessage": None, "createdAt": now,
            "startedAt": None, "finishedAt": None, "updatedAt": now,
        })
        created.append({"gameId": game_id, "title": title, "index": index})
    segment_batch = {
        "batchId": segment_batch_id, "createdAt": now,
        "cookieBrowser": source_batch.get("cookieBrowser"), "sourceUrlStored": False,
        "parentBatchId": batch_id, "parentGameId": parent_game_id,
        "processing": {"samplingMode": "smart", "everySec": 5,
                       "startSec": 0, "endSec": None}, "jobs": jobs,
    }
    sources.save_batch(data_dir, segment_batch)
    store.create_game(data_dir, parent_game_id, {
        "analysisStatus": "SEGMENT_EXTRACTION_RUNNING",
        "sourceSegmentationRequired": False, "segmentBatchId": segment_batch_id,
        "segmentGameIds": [item["gameId"] for item in created],
    })
    started = _start_persisted_source_batches(
        [segment_batch], f"长录像分场处理：{parent_game_id}"
    )
    return {**started, "batchId": segment_batch_id, "created": created,
            "count": len(created), "parentGameId": parent_game_id}


def api_delete_game(body: dict[str, Any]) -> dict[str, Any]:
    game_id = str(body.get("gameId") or "")
    if not store.valid_game_id(game_id):
        return {"ok": False, "error": "比赛编号不合法。"}
    if str(body.get("confirm") or "") != game_id:
        return {"ok": False, "error": "删除是不可撤销的，请再输入一次比赛编号确认。"}
    removed = store.delete_game(_console().data_dir, game_id)
    return {"ok": removed, "error": "" if removed else "没有这场比赛。"}


def api_annotate(body: dict[str, Any]) -> dict[str, Any]:
    data_dir = paths.ensure(_console().data_dir)
    game_id = str(body.get("gameId") or "")
    if not store.valid_game_id(game_id) or not store.load_meta(data_dir, game_id):
        return {"ok": False, "error": f"没有这场比赛：{game_id}"}
    try:
        observation = annotate.build_observation(
            body.get("atSec"), body.get("payload") or {},
            rules.load(data_dir), body.get("frameFile"),
        )
    except annotate.AnnotateError as err:
        return {"ok": False, "error": str(err)}
    annotate.save(data_dir, game_id, observation)
    return {"ok": True, "saved": observation, "progress": annotate.progress(data_dir, game_id)}


def api_save_rule(body: dict[str, Any]) -> dict[str, Any]:
    key = str(body.get("key") or "")
    if key not in rules.DEFAULTS:
        return {"ok": False, "error": f"没有这条常量：{key}"}
    try:
        value = int(body.get("value"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "值必须是整数。"}
    if value < 0:
        return {"ok": False, "error": "值不能是负数。"}
    rules.save_override(paths.ensure(_console().data_dir), key, value,
                        str(body.get("by") or "用户核对"))
    return {"ok": True, "unverified": rules.unverified(rules.load(_console().data_dir))}


def api_save_profile(body: dict[str, Any]) -> dict[str, Any]:
    name = str(body.get("name") or "").strip() or "custom"
    if not store.valid_game_id(name):
        return {"ok": False, "error": "标定档案名只能用字母、数字、下划线、点、短横线。"}
    regions = body.get("regions") or {}
    bad = [k for k, v in regions.items() if not hud.valid_box(v)]
    if bad:
        return {"ok": False, "error": f"这些区域的坐标不合法：{'、'.join(bad)}。坐标必须是 0~1 的相对值。"}
    profile = {
        "name": body.get("label") or name,
        "note": body.get("note") or "",
        "aspect": body.get("aspect"),
        "calibratedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": regions,
    }
    hud.save_profile(paths.ensure(_console().data_dir), name, profile)
    saved = hud.get_profile(_console().data_dir, name)
    return {"ok": True, "profile": saved, "verdict": hud.describe(saved)}


def _frame_file(data_dir: Path, game_id: str, name: str) -> Path:
    """把 (比赛, 文件名) 解析成真实路径，挡住 ../。"""
    if not store.valid_game_id(game_id):
        raise ValueError("比赛编号不合法。")
    base = store.frames_dir(data_dir, game_id).resolve()
    target = (base / name).resolve()
    if not str(target).startswith(str(base) + "/") or not target.is_file():
        raise ValueError("没有这张帧图。")
    return target


def api_minimap_detect(body: dict[str, Any]) -> dict[str, Any]:
    """在标定页上实时试一次小地图识别。"""
    data_dir = _console().data_dir
    try:
        image = _frame_file(data_dir, str(body.get("gameId") or ""),
                            str(body.get("file") or ""))
    except ValueError as err:
        return {"ok": False, "error": str(err)}

    box = body.get("box")
    if not hud.valid_box(box):
        return {"ok": False, "error": "小地图区域坐标不合法，必须是 0~1 的相对值。"}

    thresholds = minimap.load_thresholds(data_dir)
    for key in ("blueMinB", "blueBOverR", "blueBOverG",
                "redMinR", "redROverB", "redROverG",
                "minBlobPixels", "maxBlobPixels"):
        if body.get(key) is not None:
            try:
                thresholds[key] = int(body[key])
            except (TypeError, ValueError):
                return {"ok": False, "error": f"{key} 必须是整数。"}

    try:
        result = minimap.detect(image, tuple(box), thresholds)
    except Exception as err:      # noqa: BLE001 - 标定页要看见失败原因
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}
    return {"ok": True, "result": result, "thresholds": thresholds}


def api_minimap_save(body: dict[str, Any]) -> dict[str, Any]:
    data_dir = paths.ensure(_console().data_dir)
    values = {k: v for k, v in (body.get("thresholds") or {}).items()
              if k != "verified"}
    try:
        path = minimap.save_thresholds(data_dir, values, verified=bool(body.get("verified")))
    except (TypeError, ValueError) as err:
        return {"ok": False, "error": str(err)}
    return {"ok": True, "path": str(path),
            "thresholds": minimap.load_thresholds(data_dir)}


def api_analysis(game_id: str) -> dict[str, Any]:
    """一场比赛的完整分析：胜率曲线 + 拐点 + 每个拐点发生了什么。

    没有训练好的模型时返回基线曲线，并把 `kind` 标成 `"baseline"`。
    **前端必须据此显示警告并把线画成虚线** —— 这条曲线是假设出来的。
    """
    data_dir = _console().data_dir
    if not store.valid_game_id(game_id):
        return {"ok": False, "error": "比赛编号不合法。"}
    meta = store.load_meta(data_dir, game_id)
    if not meta:
        return {"ok": False, "error": f"没有这场比赛：{game_id}"}

    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))
    if not observations:
        return {
            "ok": False,
            "stage": "no_observations",
            "error": "这场还没有任何观测数据，算不出曲线。",
            "next": "先去标注页逐帧填数，或者标定 HUD 之后跑识别。",
        }

    built = state_mod.build(meta, observations, rules.load(data_dir))

    # 目前没有任何训练好的模型 —— 这是事实，不是占位。
    # 等真的训出来了，这里从磁盘读模型，kind 会变成 "trained"。
    trained = _load_model(data_dir)

    curve = model.win_curve(built, trained)
    inflections = model.find_inflections(curve)
    return {
        "ok": True,
        "meta": built["meta"],
        "coverage": built["coverage"],
        "curve": curve,
        "inflections": [
            {**inf, "explain": model.explain_inflection(inf)} for inf in inflections
        ],
        "branchStatus": model.branch_analysis_status(),
        "baselinePriors": model.BASELINE_PRIORS if curve["kind"] == "baseline" else None,
    }


def _load_model(data_dir: Path) -> dict[str, Any] | None:
    """读取训练好的模型。现在永远返回 None —— 因为一个都还没训出来。

    **不要在这里塞一个假模型让页面好看。** 页面显示基线曲线并挂着警告，
    比显示一条假装是模型输出的曲线诚实得多。
    """
    path = data_dir / "kpl" / "model.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) and loaded.get("kind") == "trained" else None


def api_train(body: dict[str, Any]) -> dict[str, Any]:
    """在本机所有比赛上训练一版 V(s)。数据不够会拒绝 —— 那是设计。"""
    data_dir = _console().data_dir
    games = []
    for game_id in store.list_games(data_dir):
        meta = store.load_meta(data_dir, game_id)
        if not meta or meta.get("blueWin") is None:
            continue
        observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))
        if observations:
            games.append(state_mod.build(meta, observations, rules.load(data_dir)))

    try:
        trained = model.train(games)
    except model.TrainingRefused as err:
        return {"ok": False, "refused": True, "error": str(err),
                "gamesConsidered": len(games)}

    path = data_dir / "kpl" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(trained, handle, ensure_ascii=False, indent=2)
    return {"ok": True, "model": {k: v for k, v in trained.items()
                                  if k not in ("means", "stds")}}


def api_build_state(body: dict[str, Any]) -> dict[str, Any]:
    data_dir = _console().data_dir
    game_id = str(body.get("gameId") or "")
    meta = store.load_meta(data_dir, game_id) if store.valid_game_id(game_id) else None
    if not meta:
        return {"ok": False, "error": f"没有这场比赛：{game_id}"}
    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))
    if not observations:
        return {"ok": False, "error": "这场还没有任何观测，先去标注。"}
    built = state_mod.build(meta, observations, rules.load(data_dir))
    store.write_json_gz(store.state_path(data_dir, game_id), built)
    return {"ok": True, "meta": built["meta"], "coverage": built["coverage"],
            "rejectionCount": len(built["rejections"])}


def api_extract(body: dict[str, Any]) -> dict[str, Any]:
    """后台抽帧。录像可能很长，所以走后台任务而不是阻塞请求。"""
    data_dir = paths.ensure(_console().data_dir)
    game_id = str(body.get("gameId") or "")
    if not store.valid_game_id(game_id) or not store.load_meta(data_dir, game_id):
        return {"started": False, "reason": f"没有这场比赛：{game_id}"}
    source = Path(str(body.get("video") or "")).expanduser()
    if not source.is_file():
        return {"started": False, "reason": f"找不到录像文件：{source}"}
    try:
        every = float(body.get("everySec") or 30)
        start = float(body.get("startSec") or 0)
    except (TypeError, ValueError):
        return {"started": False, "reason": "间隔和起点必须是数字。"}
    end = body.get("endSec")
    end_sec = float(end) if end not in (None, "") else None

    def work(log: Callable[[str], None]) -> None:
        log(f"读取录像：{source.name}")
        info = video.probe(source)
        log(f"分辨率 {info.width}×{info.height}，时长 {info.durationSec / 60:.1f} 分钟")
        out_dir = store.frames_dir(data_dir, game_id)
        log(f"每 {every:.0f} 秒抽一帧 → {out_dir}")

        def progress(done: int, total: int) -> None:
            _console().progress(done, total)

        records = video.extract_frames(
            source, out_dir, every, start, end_sec, on_progress=progress
        )
        good = [r for r in records if r.get("file")]
        log(f"✅ 抽出 {len(good)} 帧，失败 {len(records) - len(good)} 帧")
        store.create_game(data_dir, game_id, {
            "videoPath": str(source), "videoWidth": info.width,
            "videoHeight": info.height, "frameEverySec": every,
            "frameStartSec": start,
        })
        log("提示：抽帧只是把画面切出来，里面还没有任何数字。")
        log("下一步是在标注页上逐帧填数，或者标定 HUD 之后跑识别。")

    return _console().start_job(f"抽帧：{game_id}", work)


def api_register_browser_video(body: dict[str, Any]) -> dict[str, Any]:
    """只记录用户选择的录像信息；录像本身不会上传或复制。"""
    data_dir = paths.ensure(_console().data_dir)
    game_id = str(body.get("gameId") or "")
    if not store.valid_game_id(game_id) or not store.load_meta(data_dir, game_id):
        return {"ok": False, "error": f"没有这场比赛：{game_id}"}
    try:
        duration = float(body.get("durationSec") or 0)
        width = int(body.get("width") or 0)
        height = int(body.get("height") or 0)
        size = int(body.get("sizeBytes") or 0)
        every = float(body.get("everySec") or 5)
        maximum_gap = float(body.get("maximumGapSec") or every)
        change_scan = float(body.get("changeScanEverySec") or 0)
        base_count = int(body.get("baseFrameCount") or 0)
        scan_count = int(body.get("scanPointCount") or 0)
        adaptive_count = int(body.get("adaptiveFrameCount") or 0)
        saved_count = int(body.get("savedFrameCount") or 0)
    except (TypeError, ValueError):
        return {"ok": False, "error": "录像信息不完整，浏览器无法读取时长或分辨率。"}
    if (not all(math.isfinite(value) for value in (duration, every, maximum_gap, change_scan))
            or duration <= 0 or width <= 0 or height <= 0 or size <= 0 or every <= 0
            or maximum_gap <= 0 or change_scan < 0
            or min(base_count, scan_count, adaptive_count, saved_count) < 0
            or max(base_count, saved_count) > video.MAX_BROWSER_FRAMES
            or scan_count > video.MAX_BROWSER_SCAN_POINTS):
        return {"ok": False, "error": "录像信息不完整，浏览器无法读取时长或分辨率。"}
    name = Path(str(body.get("fileName") or "本地录像")).name[:180]
    sampling_mode = str(body.get("samplingMode") or "fixed")
    if sampling_mode not in video.SAMPLING_MODES:
        return {"ok": False, "error": "不支持这种抽帧模式。"}
    store.create_game(data_dir, game_id, {
        "videoImportMode": "browser-local-no-copy",
        "videoName": name,
        "videoSizeBytes": size,
        "videoDurationSec": round(duration, 3),
        "videoWidth": width,
        "videoHeight": height,
        "frameEverySec": every if sampling_mode == "fixed" else None,
        "samplingMode": sampling_mode,
        "samplingMaxGapSec": maximum_gap,
        "baseFrameCount": base_count,
        "changeScanEverySec": change_scan or None,
        "scanPointCount": scan_count,
        "adaptiveFrameCount": adaptive_count,
        "savedFrameCount": saved_count,
    })
    return {"ok": True, "message": "只记录了录像信息，没有复制或上传整段录像。"}


def api_upload_browser_frame(
    game_id: str, at_sec: str, body: bytes, content_type: str
) -> dict[str, Any]:
    data_dir = paths.ensure(_console().data_dir)
    if not store.valid_game_id(game_id) or not store.load_meta(data_dir, game_id):
        return {"ok": False, "error": f"没有这场比赛：{game_id}"}
    try:
        return video.save_browser_frame(
            store.frames_dir(data_dir, game_id), float(at_sec), body, content_type
        )
    except (TypeError, ValueError, video.VideoError) as err:
        return {"ok": False, "error": str(err)}


def api_frames(game_id: str) -> dict[str, Any]:
    data_dir = _console().data_dir
    if not store.valid_game_id(game_id):
        return {"error": "比赛编号不合法。"}
    frames_dir = store.frames_dir(data_dir, game_id)
    meta = store.load_meta(data_dir, game_id) or {}
    files: list[dict[str, Any]] = []
    if frames_dir.is_dir():
        for path in sorted(frames_dir.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            at = 0.0
            stem = path.stem
            if "_" in stem and stem.rsplit("_", 1)[-1].endswith("s"):
                try:
                    at = float(stem.rsplit("_", 1)[-1][:-1])
                except ValueError:
                    at = 0.0
            files.append({"file": path.name, "atSec": at})
    annotated = {round(float(s), 1) for s in
                 annotate.progress(data_dir, game_id)["annotatedSeconds"]}
    for item in files:
        item["annotated"] = round(item["atSec"], 1) in annotated
    return {"gameId": game_id, "frames": files, "meta": meta,
            "progress": annotate.progress(data_dir, game_id)}


def api_state(game_id: str) -> dict[str, Any]:
    data_dir = _console().data_dir
    if not store.valid_game_id(game_id):
        return {"error": "比赛编号不合法。"}
    path = store.state_path(data_dir, game_id)
    if path.is_file():
        return store.read_json_gz(path)
    meta = store.load_meta(data_dir, game_id)
    if not meta:
        return {"error": f"没有这场比赛：{game_id}"}
    observations = list(store.read_jsonl_gz(store.observations_path(data_dir, game_id)))
    if not observations:
        return {"error": "这场还没有任何观测。"}
    return state_mod.build(meta, observations, rules.load(data_dir))


def api_set_data_dir(body: dict[str, Any]) -> dict[str, Any]:
    raw = str(body.get("path") or "").strip()
    if not raw:
        return {"ok": False, "error": "路径不能为空。"}
    path = Path(raw).expanduser()
    if not path.is_dir():
        return {"ok": False, "error": f"这个目录不存在：{path}"}
    _console().set_data_dir(path.resolve())
    return {"ok": True, "dataDir": str(path.resolve())}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = f"kplab/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        return      # 不要往终端刷请求日志，用户看的是控制台页面

    def do_GET(self) -> None:      # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send_file(WEB_DIR / "console.html")
            elif path == "/annotate":
                self._send_file(WEB_DIR / "annotate.html")
            elif path == "/calibrate":
                self._send_file(WEB_DIR / "calibrate.html")
            elif path == "/analysis":
                self._send_file(WEB_DIR / "analysis.html")
            elif path == "/debug":
                self._send_file(WEB_DIR / "debug.html")
            elif path == "/api/status":
                self._send_json(api_status())
            elif path == "/api/job":
                self._send_json(_console().snapshot())
            elif path == "/api/datadir/scan":
                self._send_json({"candidates": store.find_data_dir_candidates()})
            elif path == "/api/games":
                self._send_json(store.inventory(_console().data_dir))
            elif path.startswith("/api/frames/"):
                self._send_json(api_frames(_tail(path)))
            elif path.startswith("/api/state/"):
                self._send_json(api_state(_tail(path)))
            elif path.startswith("/api/analysis/"):
                self._send_json(api_analysis(_tail(path)))
            elif path.startswith("/api/check/"):
                self._send_json(check.run(_console().data_dir, _tail(path)))
            elif path.startswith("/api/evaluate/"):
                self._send_json(evaluate.run(_console().data_dir, _tail(path)))
            elif path == "/frame":
                self._send_frame(query)
            else:
                self._send_json({"error": "not found"}, status=404)
        except FileNotFoundError as err:
            self._send_json({"error": str(err)}, status=404)
        except Exception as err:      # noqa: BLE001 - 本地调试服务器，错误要看得见
            self._send_json({"error": f"{type(err).__name__}: {err}"}, status=500)

    def do_POST(self) -> None:      # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/frame/upload":
            self._receive_browser_frame(parse_qs(parsed.query))
            return
        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "/api/datadir": api_set_data_dir,
            "/api/game/create": api_create_game,
            "/api/game/batch-create": api_create_batch,
            "/api/game/delete": api_delete_game,
            "/api/annotate": api_annotate,
            "/api/rules": api_save_rule,
            "/api/profile": api_save_profile,
            "/api/state/build": api_build_state,
            "/api/extract": api_extract,
            "/api/train": api_train,
            "/api/minimap/detect": api_minimap_detect,
            "/api/minimap/save": api_minimap_save,
            "/api/video/register": api_register_browser_video,
            "/api/video/plan": api_sampling_plan,
            "/api/source/preview": api_source_preview,
            "/api/source/start": api_source_start,
            "/api/source/resume": api_source_resume,
            "/api/source/segments/start": api_source_segments_start,
        }
        handler = routes.get(path)
        if not handler:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            self._send_json(handler(self._body()))
        except Exception as err:      # noqa: BLE001
            self._send_json({"error": f"{type(err).__name__}: {err}"}, status=500)

    def _receive_browser_frame(self, query: dict[str, list[str]]) -> None:
        """接收浏览器本地解码的一张图；绝不接收整段录像。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json({"ok": False, "error": "收到的是空图片。"}, status=400)
            return
        if length > video.BROWSER_FRAME_MAX_BYTES:
            self._send_json({"ok": False, "error": "单张帧图超过 8 MB。"}, status=413)
            return
        body = self.rfile.read(length)
        result = api_upload_browser_frame(
            (query.get("game") or [""])[0],
            (query.get("atSec") or [""])[0],
            body,
            self.headers.get("Content-Type") or "",
        )
        self._send_json(result, status=200 if result.get("ok") else 400)

    def _send_frame(self, query: dict[str, list[str]]) -> None:
        """把抽出来的帧图发给标注页。

        路径是拼出来的，所以必须挡住 ../ —— 这个服务器虽然只监听本机，
        但浏览器里的任何页面都能向它发请求。
        """
        game_id = (query.get("game") or [""])[0]
        name = (query.get("file") or [""])[0]
        if not store.valid_game_id(game_id):
            self._send_json({"error": "比赛编号不合法"}, status=400)
            return
        base = store.frames_dir(_console().data_dir, game_id).resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base) + "/") or not target.is_file():
            self._send_json({"error": "没有这张图"}, status=404)
            return
        if target.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            self._send_json({"error": "不支持的文件类型"}, status=400)
            return
        body = target.read_bytes()
        kind = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        try:
            obj = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _tail(path: str) -> str:
    return unquote(path.rsplit("/", 1)[-1])


def serve(data_dir: Path, port: int = 8020) -> None:
    global CONSOLE
    CONSOLE = Console(data_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print()
    print("  ┌──────────────────────────────────────────────┐")
    print(f"  │  王者荣耀决策实验室：http://127.0.0.1:{port}    │")
    print("  └──────────────────────────────────────────────┘")
    print()
    print(f"  版本：kplab {__version__}")
    print(f"  数据目录：{data_dir}")
    print(f"  抽帧（ffmpeg）：{'可用' if video.available()['ok'] else '未安装 —— 不影响人工标注'}")
    source_info = sources.available()
    print(f"  网页录像解析（yt-dlp）："
          f"{'可用' if source_info['webPages'] else '未安装 —— 直接媒体地址仍可使用'}")
    auto = ocr.best_backend()
    print(f"  自动识别：{auto if auto else '无可用后端 —— 不影响人工标注'}")
    print()
    print("  在浏览器里打开上面那个地址，接下来全在网页上点。")
    print("  要停止：回到这个窗口按 Control + C。")
    print()
    resumed = resume_interrupted_source_batches()
    if resumed.get("started"):
        print(f"  已自动恢复 {resumed.get('batchCount', 0)} 个中断批次，"
              f"其中 {resumed.get('recoveredJobs', 0)} 个任务来自上次运行。")
        print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
