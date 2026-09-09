"""统一录像来源：本地文件、直接媒体地址和网页解析器。

长期保存的是用户提交的 ``originUrl``。带签名、会过期的真实媒体地址只在
任务执行期间存在于内存，既不写入日志，也不写进比赛元数据。
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


MAX_SOURCES = 100
MAX_PLAYLIST_ITEMS = 100
MAX_RESOLVE_RETRIES = 3
RESOLVE_TIMEOUT_SEC = 90
SUPPORTED_LOCAL_SUFFIXES = {".mp4", ".mkv", ".flv", ".ts", ".mov", ".m4v", ".webm"}
DIRECT_VIDEO_SUFFIXES = SUPPORTED_LOCAL_SUFFIXES | {".m3u8", ".mpd"}
COOKIE_BROWSERS = {"", "safari", "chrome", "firefox", "edge"}

ERROR_MESSAGES = {
    "UNSUPPORTED_SITE": "该网址暂不支持。",
    "VIDEO_NOT_FOUND": "录像页面不存在或已经删除。",
    "LOGIN_REQUIRED": "该录像需要登录后访问。",
    "ACCESS_DENIED": "当前账号或网络无权读取该录像。",
    "GEO_BLOCKED": "该录像受地区限制。",
    "MEDIA_URL_EXPIRED": "媒体地址已经过期，将重新解析。",
    "NETWORK_ERROR": "网络连接失败。",
    "RESOLVER_MISSING": "网页解析工具 yt-dlp 尚未安装。",
    "RESOLVER_ERROR": "网页中没有解析出可读取的录像。",
    "NO_VIDEO_STREAM": "页面中没有可用的视频流。",
    "SOURCE_TOO_LONG": "录像时长超过单场比赛范围，需要先进行分段侦察。",
    "DRM_PROTECTED": "该录像受 DRM 保护，系统不会尝试绕过。",
    "INVALID_SOURCE": "录像地址不合法。",
}


class SourceError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code if code in ERROR_MESSAGES else "RESOLVER_ERROR"
        self.detail = _redact(detail)
        message = ERROR_MESSAGES[self.code]
        if self.detail:
            message += f"（{self.detail[:240]}）"
        super().__init__(message)


@dataclass
class ResolvedVideoSource:
    platform: str
    sourceType: str
    originUrl: str
    sourceUrl: str | None
    title: str
    scoutSourceUrl: str | None = field(default=None, repr=False)
    analysisSourceUrl: str | None = field(default=None, repr=False)
    durationSec: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    mediaFingerprint: str = ""
    requiresLogin: bool = False
    resolverName: str = ""
    resolvedAt: str = ""
    sourceExpiresAt: str | None = None
    playlistIndex: int | None = None
    playlistCount: int = 1
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """给前端的安全副本：不暴露媒体签名地址、Cookie 或请求头。"""
        result = asdict(self)
        result.pop("sourceUrl", None)
        result.pop("scoutSourceUrl", None)
        result.pop("analysisSourceUrl", None)
        result.pop("headers", None)
        result["sourceUrlStored"] = False
        return result


def yt_dlp_path() -> str | None:
    return shutil.which("yt-dlp")


def available() -> dict[str, Any]:
    executable = yt_dlp_path()
    return {
        "ytDlp": executable,
        "webPages": bool(executable),
        "directHttp": True,
        "hls": True,
        "local": True,
        "maxSources": MAX_SOURCES,
        "cacheMaxGb": 2,
        "cacheRetentionHours": 24,
        "hint": "" if executable else "安装 yt-dlp 后可解析 B站、虎牙、斗鱼等网页；直接 MP4/m3u8 已可使用。",
    }


def _redact(text: Any) -> str:
    raw = str(text or "")
    for marker in ("Cookie:", "cookie:", "Authorization:", "authorization:"):
        if marker in raw:
            raw = raw.split(marker, 1)[0] + marker + " [已隐藏]"
    raw = re.sub(r"https?://[^\s'\"]+", "[媒体地址已隐藏]", raw)
    return raw.replace("\n", " ").strip()


def safe_error(text: Any) -> str:
    """日志和任务文件可用的错误文本，不泄露临时媒体地址或认证信息。"""
    return _redact(text)[:240]


def _clean_origin(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        raise SourceError("INVALID_SOURCE", "地址为空")
    if len(value) > 4096:
        raise SourceError("INVALID_SOURCE", "地址过长")
    local = Path(value).expanduser()
    if local.is_file():
        if local.suffix.lower() not in SUPPORTED_LOCAL_SUFFIXES:
            raise SourceError("INVALID_SOURCE", "不支持这个本地录像格式")
        return str(local.resolve())

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SourceError("INVALID_SOURCE", "只接受 http、https 或本地录像文件")
    if parsed.username or parsed.password:
        raise SourceError("INVALID_SOURCE", "网址中不能包含账号或密码")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise SourceError("INVALID_SOURCE", "不读取本机或局域网地址")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address and (address.is_private or address.is_loopback or address.is_link_local
                    or address.is_reserved or address.is_multicast):
        raise SourceError("INVALID_SOURCE", "不读取本机、局域网或保留地址")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/",
                       parsed.query, ""))


def detect(raw: Any) -> dict[str, str]:
    origin = _clean_origin(raw)
    local = Path(origin)
    if local.is_file():
        return {"sourceType": "LOCAL_FILE", "platform": "local", "originUrl": origin}
    parsed = urlsplit(origin)
    host = (parsed.hostname or "").lower()
    suffix = Path(parsed.path).suffix.lower()
    if suffix == ".m3u8":
        kind = "HLS"
    elif suffix == ".mpd":
        kind = "DASH"
    elif suffix in SUPPORTED_LOCAL_SUFFIXES:
        kind = "HTTP_VIDEO"
    elif host == "b23.tv" or host.endswith("bilibili.com"):
        kind = "BILIBILI_PAGE"
    elif host.endswith("huya.com"):
        kind = "HUYA_PAGE"
    elif host.endswith("douyu.com"):
        kind = "DOUYU_PAGE"
    else:
        kind = "GENERIC_WEB_PAGE"
    platform = {
        "BILIBILI_PAGE": "bilibili", "HUYA_PAGE": "huya",
        "DOUYU_PAGE": "douyu", "HLS": "direct", "DASH": "direct",
        "HTTP_VIDEO": "direct", "GENERIC_WEB_PAGE": "web",
    }[kind]
    return {"sourceType": kind, "platform": platform, "originUrl": origin}


def parse_submission(text: Any = "", csv_text: Any = "") -> list[dict[str, Any]]:
    """解析多行网址和 CSV；坏行单独报错，不让整批消失。"""
    entries: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            entries.append({"originUrl": value, "metadata": {}})
    if str(csv_text or "").strip():
        reader = csv.DictReader(StringIO(str(csv_text)))
        for row in reader:
            allowed = {"source_url", "origin_url", "team", "opponent",
                       "date", "event", "note"}
            normalized = {str(k or "").strip(): str(v or "").strip()[:500]
                          for k, v in row.items() if str(k or "").strip() in allowed}
            url = normalized.pop("source_url", "") or normalized.pop("origin_url", "")
            if url:
                entries.append({"originUrl": url, "metadata": normalized})
    if not entries:
        raise SourceError("INVALID_SOURCE", "没有找到录像网址")
    if len(entries) > MAX_SOURCES:
        raise SourceError("INVALID_SOURCE", f"一次最多提交{MAX_SOURCES}个录像来源")
    return entries


def _number(value: Any, integer: bool = False) -> int | float | None:
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(float(number)) and number >= 0 else None


def _fingerprint(platform: str, video_id: Any, origin: str, title: str,
                 duration: Any) -> str:
    parsed = urlsplit(origin)
    stable_origin = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, "", ""))
    identity = str(video_id or "").strip() or stable_origin
    material = f"{platform}|{identity}|{title.strip()}|{_number(duration) or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _classify_ytdlp_error(stderr: str) -> str:
    lower = stderr.lower()
    if "drm" in lower:
        return "DRM_PROTECTED"
    if any(word in lower for word in ("login", "sign in", "cookies", "登录")):
        return "LOGIN_REQUIRED"
    if any(word in lower for word in ("private video", "forbidden", "access denied", "403", "412")):
        return "ACCESS_DENIED"
    if any(word in lower for word in ("geo", "country", "region")):
        return "GEO_BLOCKED"
    if any(word in lower for word in ("not found", "deleted", "removed", "404")):
        return "VIDEO_NOT_FOUND"
    if any(word in lower for word in ("timed out", "network", "connection", "temporary failure")):
        return "NETWORK_ERROR"
    return "RESOLVER_ERROR"


def classify_media_error(detail: Any) -> str:
    lower = str(detail or "").lower()
    if "超过2000张基础帧" in lower:
        return "SOURCE_TOO_LONG"
    if "drm" in lower:
        return "DRM_PROTECTED"
    if "401" in lower or "login" in lower or "sign in" in lower:
        return "LOGIN_REQUIRED"
    if "404" in lower or "not found" in lower:
        return "VIDEO_NOT_FOUND"
    if "403" in lower or "412" in lower or "forbidden" in lower:
        return "ACCESS_DENIED"
    if any(word in lower for word in ("expired", "signature", "token has expired")):
        return "MEDIA_URL_EXPIRED"
    return "NETWORK_ERROR"


def _run_ytdlp(origin: str, cookie_browser: str = "") -> dict[str, Any]:
    executable = yt_dlp_path()
    if not executable:
        raise SourceError("RESOLVER_MISSING")
    browser = str(cookie_browser or "").lower()
    if browser not in COOKIE_BROWSERS:
        raise SourceError("INVALID_SOURCE", "不支持这个浏览器登录状态")
    command = [
        executable, "--dump-single-json", "--skip-download", "--no-warnings",
        "--socket-timeout", "20", "--extractor-retries", "2",
        "--fragment-retries", "2", "--yes-playlist", "--playlist-end", "101", origin,
    ]
    if browser:
        command[1:1] = ["--cookies-from-browser", browser]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=RESOLVE_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired as err:
        raise SourceError("NETWORK_ERROR", "网页解析超时") from err
    except OSError as err:
        raise SourceError("RESOLVER_ERROR", str(err)) from err
    if result.returncode != 0:
        raise SourceError(_classify_ytdlp_error(result.stderr), result.stderr[-800:])
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError) as err:
        raise SourceError("RESOLVER_ERROR", "yt-dlp 返回了无法识别的数据") from err
    if not isinstance(payload, dict):
        raise SourceError("NO_VIDEO_STREAM")
    return payload


def _media_url(payload: dict[str, Any]) -> tuple[str | None, str | None, str | None, dict[str, str]]:
    url = payload.get("url")
    candidates = payload.get("requested_formats") or []
    if not url and isinstance(candidates, list):
        videos = [item for item in candidates if isinstance(item, dict)
                  and item.get("url") and item.get("vcodec") not in (None, "none")]
        if videos:
            videos.sort(key=lambda item: (_number(item.get("height")) or 0,
                                          _number(item.get("tbr")) or 0), reverse=True)
            url = videos[0].get("url")
    scout_url = None
    analysis_url = None
    formats = payload.get("formats") or []
    if isinstance(formats, list):
        scout_formats = [item for item in formats if isinstance(item, dict)
                         and item.get("url") and item.get("vcodec") != "none"]
        scout_formats.sort(key=lambda item: (_number(item.get("height")) or 100000,
                                             _number(item.get("tbr")) or 100000))
        if scout_formats:
            scout_url = str(scout_formats[0]["url"])
            analysis_format = min(
                scout_formats,
                key=lambda item: abs((_number(item.get("height")) or 720) - 720),
            )
            analysis_url = str(analysis_format["url"])
    headers = payload.get("http_headers") if isinstance(payload.get("http_headers"), dict) else {}
    # 登录态请求头可以在解析任务的内存中交给 ffmpeg，但 public()、任务文件和
    # 日志都必须剔除它们。否则“用浏览器登录状态解析”会在取到媒体地址后又因
    # 缺少 Cookie 而播放失败。
    transient_headers = {str(key): str(value) for key, value in headers.items()}
    return str(url) if url else None, scout_url, analysis_url, transient_headers


def _from_ytdlp(payload: dict[str, Any], origin: str, platform: str,
                source_type: str, resolver_name: str,
                playlist_index: int | None = None, playlist_count: int = 1,
                metadata: dict[str, Any] | None = None) -> ResolvedVideoSource:
    source_url, scout_source_url, analysis_source_url, headers = _media_url(payload)
    webpage = str(payload.get("webpage_url") or origin)
    title = str(payload.get("title") or payload.get("fulltitle") or "未命名录像")[:300]
    video_id = payload.get("id")
    return ResolvedVideoSource(
        platform=platform,
        sourceType=source_type,
        originUrl=webpage,
        sourceUrl=source_url,
        title=title,
        scoutSourceUrl=scout_source_url,
        analysisSourceUrl=analysis_source_url,
        durationSec=_number(payload.get("duration")),
        width=_number(payload.get("width"), integer=True),
        height=_number(payload.get("height"), integer=True),
        fps=_number(payload.get("fps")),
        codec=str(payload.get("vcodec") or "") or None,
        mediaFingerprint=_fingerprint(platform, video_id, webpage, title, payload.get("duration")),
        requiresLogin=False,
        resolverName=resolver_name,
        resolvedAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        playlistIndex=playlist_index,
        playlistCount=playlist_count,
        headers=headers,
        metadata={**(metadata or {}), "platformVideoId": str(video_id or "")},
    )


class Resolver:
    name = "base"
    source_types: set[str] = set()

    def can_handle(self, detected: dict[str, str]) -> bool:
        return detected["sourceType"] in self.source_types

    def resolve(self, detected: dict[str, str], cookie_browser: str = "",
                playlist_index: int | None = None,
                metadata: dict[str, Any] | None = None) -> list[ResolvedVideoSource]:
        raise NotImplementedError


class LocalResolver(Resolver):
    name = "local-file"
    source_types = {"LOCAL_FILE"}

    def resolve(self, detected: dict[str, str], cookie_browser: str = "",
                playlist_index: int | None = None,
                metadata: dict[str, Any] | None = None) -> list[ResolvedVideoSource]:
        path = Path(detected["originUrl"])
        return [ResolvedVideoSource(
            platform="local", sourceType="LOCAL_FILE", originUrl=str(path),
            sourceUrl=str(path), title=path.name,
            mediaFingerprint=_fingerprint("local", f"{path}:{path.stat().st_size}",
                                          str(path), path.name, None),
            resolverName=self.name, resolvedAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            metadata=metadata or {},
        )]


class DirectMediaResolver(Resolver):
    name = "direct-media"
    source_types = {"HTTP_VIDEO", "HLS", "DASH"}

    def resolve(self, detected: dict[str, str], cookie_browser: str = "",
                playlist_index: int | None = None,
                metadata: dict[str, Any] | None = None) -> list[ResolvedVideoSource]:
        origin = detected["originUrl"]
        title = Path(urlsplit(origin).path).name or "远程录像"
        return [ResolvedVideoSource(
            platform="direct", sourceType=detected["sourceType"], originUrl=origin,
            sourceUrl=origin, title=title,
            mediaFingerprint=_fingerprint("direct", None, origin, title, None),
            resolverName=self.name, resolvedAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            metadata=metadata or {},
        )]


class YtDlpResolver(Resolver):
    def __init__(self, name: str, source_types: Iterable[str]) -> None:
        self.name = name
        self.source_types = set(source_types)

    def resolve(self, detected: dict[str, str], cookie_browser: str = "",
                playlist_index: int | None = None,
                metadata: dict[str, Any] | None = None) -> list[ResolvedVideoSource]:
        payload = _run_ytdlp(detected["originUrl"], cookie_browser)
        entries = payload.get("entries")
        if isinstance(entries, list):
            indexed = [(index, item) for index, item in enumerate(entries, 1)
                       if isinstance(item, dict)]
            if len(indexed) > MAX_PLAYLIST_ITEMS:
                raise SourceError("RESOLVER_ERROR", f"播放列表超过{MAX_PLAYLIST_ITEMS}项")
            if playlist_index is not None:
                indexed = [(index, item) for index, item in indexed if index == playlist_index]
            if not indexed:
                raise SourceError("NO_VIDEO_STREAM", "播放列表中没有可用项目")
            return [_from_ytdlp(item, detected["originUrl"], detected["platform"],
                                detected["sourceType"], self.name, index,
                                len(entries), metadata)
                    for index, item in indexed]
        return [_from_ytdlp(payload, detected["originUrl"], detected["platform"],
                            detected["sourceType"], self.name, playlist_index, 1, metadata)]


REGISTRY: list[Resolver] = [
    LocalResolver(), DirectMediaResolver(),
    YtDlpResolver("yt-dlp-bilibili", {"BILIBILI_PAGE"}),
    YtDlpResolver("yt-dlp-huya", {"HUYA_PAGE"}),
    YtDlpResolver("yt-dlp-douyu", {"DOUYU_PAGE"}),
    YtDlpResolver("yt-dlp-generic", {"GENERIC_WEB_PAGE"}),
]


def resolve(raw: Any, cookie_browser: str = "", playlist_index: int | None = None,
            metadata: dict[str, Any] | None = None) -> list[ResolvedVideoSource]:
    detected = detect(raw)
    for resolver in REGISTRY:
        if resolver.can_handle(detected):
            resolved = resolver.resolve(detected, cookie_browser, playlist_index, metadata)
            for item in resolved:
                if not item.sourceUrl and item.playlistCount <= 1:
                    raise SourceError("NO_VIDEO_STREAM")
            return resolved
    raise SourceError("UNSUPPORTED_SITE")


def batch_path(data_dir: Path, batch_id: str) -> Path:
    clean = "".join(char for char in batch_id if char.isalnum() or char in "_.-")[:80]
    if not clean or clean != batch_id:
        raise ValueError("批次编号不合法。")
    return data_dir / "kpl" / "source_batches" / f"{clean}.json"


def save_batch(data_dir: Path, batch: dict[str, Any]) -> Path:
    path = batch_path(data_dir, str(batch.get("batchId") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_batch(data_dir: Path, batch_id: str) -> dict[str, Any] | None:
    path = batch_path(data_dir, batch_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def list_batches(data_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    root = data_dir / "kpl" / "source_batches"
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        jobs = batch.get("jobs") if isinstance(batch.get("jobs"), list) else []
        active = next((job for job in jobs if isinstance(job, dict)
                       and job.get("status") == "RUNNING"), None)
        summaries.append({
            "batchId": batch.get("batchId"), "createdAt": batch.get("createdAt"),
            "total": len(jobs),
            "pending": sum(1 for job in jobs if job.get("status") == "PENDING"),
            "running": sum(1 for job in jobs if job.get("status") == "RUNNING"),
            "done": sum(1 for job in jobs if job.get("status") == "DONE"),
            "scoutDone": sum(1 for job in jobs if job.get("status") == "SCOUT_DONE"),
            "failed": sum(1 for job in jobs if job.get("status") == "FAILED"),
            "active": ({
                "gameId": active.get("gameId"), "title": active.get("title"),
                "stage": active.get("stage"), "progress": active.get("progress", 0),
                "currentVideoTime": active.get("currentVideoTime", 0),
                "updatedAt": active.get("updatedAt"),
            } if active else None),
        })
        if len(summaries) >= limit:
            break
    return summaries


def interrupted_batches(data_dir: Path) -> list[dict[str, Any]]:
    """读取因服务退出而遗留的待处理/运行中批次，按创建时间从旧到新返回。"""
    root = data_dir / "kpl" / "source_batches"
    if not root.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime):
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(batch, dict):
            continue
        jobs = batch.get("jobs") if isinstance(batch.get("jobs"), list) else []
        if any(job.get("status") in {"PENDING", "RUNNING"}
               for job in jobs if isinstance(job, dict)):
            result.append(batch)
    return result


def prepare_interrupted_batch(data_dir: Path, batch: dict[str, Any]) -> int:
    """把上次进程里的 RUNNING 安全退回队列；已完成和已失败任务保持不动。"""
    recovered = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    jobs = batch.get("jobs") if isinstance(batch.get("jobs"), list) else []
    for job in jobs:
        if not isinstance(job, dict) or job.get("status") != "RUNNING":
            continue
        job.update({"status": "PENDING", "stage": "RECOVERY_QUEUED",
                    "recoveredAfterRestart": True, "recoveredAt": now,
                    "updatedAt": now})
        recovered += 1
    if recovered:
        batch["recoveryCount"] = int(batch.get("recoveryCount") or 0) + 1
        save_batch(data_dir, batch)
    return recovered


def find_duplicate(data_dir: Path, fingerprint: str) -> str | None:
    if not fingerprint:
        return None
    root = data_dir / "kpl" / "games"
    if not root.is_dir():
        return None
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if meta.get("mediaFingerprint") == fingerprint:
            return str(meta.get("gameId") or meta_path.parent.name)
    return None


def cleanup_cache(data_dir: Path, retention_hours: float = 24,
                  max_bytes: int = 2 * 1024**3) -> dict[str, int]:
    """清理有限临时缓存；当前流式路径通常不会创建缓存文件。"""
    root = data_dir / "kpl" / "cache"
    if not root.is_dir():
        return {"removedFiles": 0, "removedBytes": 0, "remainingBytes": 0}
    now = time.time()
    files = [path for path in root.rglob("*") if path.is_file()]
    removed_files = removed_bytes = 0
    for path in files:
        try:
            if now - path.stat().st_mtime > retention_hours * 3600:
                size = path.stat().st_size
                path.unlink()
                removed_files += 1
                removed_bytes += size
        except OSError:
            continue
    remaining = sorted((path for path in root.rglob("*") if path.is_file()),
                       key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in remaining)
    for path in remaining:
        if total <= max_bytes:
            break
        try:
            size = path.stat().st_size
            path.unlink()
            total -= size
            removed_files += 1
            removed_bytes += size
        except OSError:
            continue
    return {"removedFiles": removed_files, "removedBytes": removed_bytes,
            "remainingBytes": max(0, total)}
