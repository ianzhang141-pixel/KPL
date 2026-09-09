"""比赛数据的落档与读取 —— 从第一天就压缩存。

## 为什么一上来就 gzip

LOL 项目的路线图里有一张表：未压缩存 5 万场 Timeline 是 98GB，
MacBook Air 装不下，所以「改成 gzip」被排进了必做项 AI1。
那是**先写错、再回头改**，期间采集的数据全都要重新落一遍。

王者荣耀这边只会更糟：LOL 的原始数据是几 MB 的 JSON，
这边的原始数据是录像和抽出来的帧图。所以这里没有「以后再压缩」的选项。

## 目录结构

    data/kpl/
      rules_override.json        用户核对过的游戏常量
      games/
        <gameId>/
          meta.json              这场是谁打谁、录像哪来的、赛事标签
          observations.jsonl.gz  原始观测，一行一帧，**只追加不改写**
          state.json.gz          由观测算出来的分钟级 State
          frames/                抽出来的画面（jpg），核对用
          annotations.jsonl.gz   人工标注，一行一条

## 「只追加不改写」是有意的

observations 是原始记录，等价于 LOL 的原始 Timeline JSON。
LOL 项目 ① 号问题就是原始 Timeline 从不落盘、用完就丢，
导致每改一次 State 定义就要把所有比赛重新下载一遍。

这边只要留着 observations，改 State 定义就只是重跑一次 `kplab state`，
不用重新看一遍录像。这是整个存储设计里最重要的一条。
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator


# ThreadingHTTPServer 会并行处理请求；gzip 追加不是线程安全的，同一文件同时写会损坏压缩流。
_APPEND_LOCK = threading.Lock()

from . import paths

GAME_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def valid_game_id(game_id: str) -> bool:
    """gameId 会直接拼进路径，必须挡住 ../ 之类的东西。"""
    return bool(game_id) and bool(GAME_ID_RE.match(game_id)) and game_id not in (".", "..")


def _require(game_id: str) -> str:
    if not valid_game_id(game_id):
        raise ValueError(
            f"比赛编号不合法：{game_id!r}。"
            "只允许字母、数字、下划线、点和短横线，例如 KPL2026S1_AG_vs_EST_G1。"
        )
    return game_id


# ---------------------------------------------------------------- 路径

def game_root(data_dir: Path, game_id: str) -> Path:
    return data_dir / "kpl" / "games" / _require(game_id)


def meta_path(data_dir: Path, game_id: str) -> Path:
    return game_root(data_dir, game_id) / "meta.json"


def observations_path(data_dir: Path, game_id: str) -> Path:
    return game_root(data_dir, game_id) / "observations.jsonl.gz"


def annotations_path(data_dir: Path, game_id: str) -> Path:
    return game_root(data_dir, game_id) / "annotations.jsonl.gz"


def state_path(data_dir: Path, game_id: str) -> Path:
    return game_root(data_dir, game_id) / "state.json.gz"


def frames_dir(data_dir: Path, game_id: str) -> Path:
    return game_root(data_dir, game_id) / "frames"


# ---------------------------------------------------------------- 读写

def write_json_gz(path: Path, obj: Any) -> Path:
    """原子写：先写临时文件再改名，中途断电不会留下半个损坏的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(obj, handle, ensure_ascii=False)
    tmp.replace(path)
    return path


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl_gz(path: Path, record: dict[str, Any]) -> Path:
    """追加一条记录。

    gzip 支持把多个压缩流首尾相接，解压时会当成一个连续的流读出来，
    所以「压缩」和「只追加」并不冲突 —— 每次追加就是接一个新的小流上去。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with gzip.open(path, "at", encoding="utf-8", compresslevel=6) as handle:
            handle.write(line)
    return path


def read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue        # 坏行跳过，不让一行毁掉整场
            if isinstance(obj, dict):
                yield obj


def rewrite_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> Path:
    """整份重写。只在「删除某场的某些标注」这种场合用，正常流程一律追加。"""
    with _APPEND_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return path


# ---------------------------------------------------------------- 比赛

def create_game(data_dir: Path, game_id: str, meta: dict[str, Any]) -> Path:
    root = game_root(data_dir, game_id)
    root.mkdir(parents=True, exist_ok=True)
    path = meta_path(data_dir, game_id)
    base = load_meta(data_dir, game_id) or {}
    base.update(meta)
    base.setdefault("gameId", game_id)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(base, handle, ensure_ascii=False, indent=2)
    return path


def load_meta(data_dir: Path, game_id: str) -> dict[str, Any] | None:
    path = meta_path(data_dir, game_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def list_games(data_dir: Path) -> list[str]:
    root = data_dir / "kpl" / "games"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def next_game_id(data_dir: Path, prefix: str, index: int) -> str:
    """为批量导入分配不覆盖旧数据的编号。"""
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix or "BATCH")).strip("_.-")
    clean = (clean or "BATCH")[:68]
    base = f"{clean}_{max(1, int(index)):03d}"[:80]
    candidate = base
    suffix = 2
    while game_root(data_dir, candidate).exists():
        tail = f"_{suffix}"
        candidate = base[:80 - len(tail)] + tail
        suffix += 1
    return candidate


def delete_game(data_dir: Path, game_id: str) -> bool:
    root = game_root(data_dir, game_id)
    if not root.is_dir():
        return False
    shutil.rmtree(root)
    return True


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def game_summary(data_dir: Path, game_id: str) -> dict[str, Any]:
    """一场比赛的盘点，不解析大文件，只看有没有 / 多大 / 几条。"""
    meta = load_meta(data_dir, game_id) or {}
    obs_path = observations_path(data_dir, game_id)
    ann_path = annotations_path(data_dir, game_id)
    st_path = state_path(data_dir, game_id)
    frames = frames_dir(data_dir, game_id)

    frame_files = 0
    if frames.is_dir():
        frame_files = sum(1 for p in frames.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    return {
        "gameId": game_id,
        "title": meta.get("title") or "",
        "tournament": meta.get("tournament") or "",
        "blueTeam": meta.get("blueTeam") or "",
        "redTeam": meta.get("redTeam") or "",
        "sourceKind": meta.get("sourceKind") or "",
        "sampleType": meta.get("sampleType") or "other",
        "sampleTypeLabel": meta.get("sampleTypeLabel") or "其他录像",
        "focalPlayer": meta.get("focalPlayer") or "",
        "focalTeam": meta.get("focalTeam") or "",
        "gamePatch": meta.get("gamePatch") or "",
        "trainingWeight": meta.get("trainingWeight"),
        "tags": meta.get("tags") or [],
        "blueWin": meta.get("blueWin"),
        "hasObservations": obs_path.is_file(),
        "observationCount": sum(1 for _ in read_jsonl_gz(obs_path)),
        "annotationCount": sum(1 for _ in read_jsonl_gz(ann_path)),
        "hasState": st_path.is_file(),
        "frameImages": frame_files,
        "bytes": {
            "observations": _size(obs_path),
            "state": _size(st_path),
            "annotations": _size(ann_path),
            "frames": _dir_bytes(frames),
        },
    }


def inventory(data_dir: Path) -> dict[str, Any]:
    games = [game_summary(data_dir, gid) for gid in list_games(data_dir)]
    total_bytes = sum(sum(g["bytes"].values()) for g in games)
    return {
        "dataDir": str(data_dir),
        "games": games,
        "total": len(games),
        "withObservations": sum(1 for g in games if g["hasObservations"]),
        "withState": sum(1 for g in games if g["hasState"]),
        "annotatedGames": sum(1 for g in games if g["annotationCount"] > 0),
        "totalBytes": total_bytes,
        "totalBytesHuman": human_bytes(total_bytes),
    }


def human_bytes(size: int) -> str:
    step = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:.1f} {unit}" if unit != "B" else f"{int(step)} B"
        step /= 1024
    return f"{step:.1f} TB"


def find_data_dir_candidates(home: Path | None = None, max_depth: int = 5) -> list[dict[str, Any]]:
    """在个人文件夹里找可能的 data 目录，供网页上一键选择。"""
    home = home or Path.home()
    skip = {
        "Library", "Applications", "Pictures", "Music", "Movies",
        "node_modules", "__pycache__", "venv", ".venv", "site-packages",
    }
    names = {"data", "Data", "storage", "var"}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(directory.iterdir())
        except (OSError, PermissionError):
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in skip:
                continue
            if entry.name in names:
                key = str(entry.resolve())
                if key not in seen:
                    seen.add(key)
                    has_kpl = (entry / "kpl").is_dir()
                    out.append({
                        "path": str(entry),
                        "hasKpl": has_kpl,
                        "games": len(list_games(entry)),
                        "score": (100 if has_kpl else 0) + len(list_games(entry)),
                    })
            walk(entry, depth + 1)

    walk(home, 0)
    out.sort(key=lambda c: (-c["score"], c["path"]))
    return out[:20]
