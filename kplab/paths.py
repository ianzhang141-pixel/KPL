"""数据目录定位。

沿用 lolab 的做法：命令行 > 环境变量 > 从当前目录向上找 > ./data。
"""

from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = ("data", "Data", "storage", "var")


def find_data_dir(explicit: str | None = None) -> Path:
    """返回项目的 data 目录。

    优先级：命令行 --data > 环境变量 KPLAB_DATA > 从当前目录向上找 data/ > ./data
    找不到时返回 ./data（由调用方决定要不要创建）。
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    env = os.environ.get("KPLAB_DATA")
    if env:
        return Path(env).expanduser().resolve()

    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        for name in _CANDIDATES:
            candidate = base / name
            if candidate.is_dir():
                return candidate
        # 到达 home 或文件系统根目录就停下，不要一路扫到 /
        if base == Path.home() or base.parent == base:
            break

    return here / "data"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def game_dir(data_dir: Path, game_id: str) -> Path:
    """一场比赛的所有数据都放在同一个目录下，方便整场删除 / 拷贝。"""
    return data_dir / "kpl" / "games" / game_id
