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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import (__version__, annotate, check, evaluate, hud, ocr, paths,
               rules, samples, schema, season_s44, state as state_mod, store, video)

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
        "sampleTypes": samples.public_types(),
        "seasonRules": season_s44.payload(),
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
            "/api/video/register": api_register_browser_video,
            "/api/video/plan": api_sampling_plan,
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
    auto = ocr.best_backend()
    print(f"  自动识别：{auto if auto else '无可用后端 —— 不影响人工标注'}")
    print()
    print("  在浏览器里打开上面那个地址，接下来全在网页上点。")
    print("  要停止：回到这个窗口按 Control + C。")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
