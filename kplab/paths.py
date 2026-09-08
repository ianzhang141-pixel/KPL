"""数据目录定位。

优先级：命令行 > 环境变量 > 项目范围内向上找 > 项目根目录/data。

向上查找必须有边界。系统目录（例如 macOS 的 ``/private/var``）不能因为
名字碰巧在候选列表里，就被误认成项目数据目录。
"""

from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = ("data", "Data", "storage", "var")


def _is_project_root(path: Path) -> bool:
    """判断一个目录是否足以作为向上查找的安全边界。"""
    return (path / ".git").exists() or (path / "kplab").is_dir()


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
    fallback_base = here
    home = Path.home().resolve()

    for base in (here, *here.parents):
        # 不检查文件系统根目录，也不越过最上层系统目录（/private、/var、
        # /Users 等）。在项目外运行时，宁可回退到 ./data，也不要猜中系统目录。
        if base.parent == base:
            break
        project_root = _is_project_root(base)
        if base.parent.parent == base.parent and not project_root and base != home:
            break

        fallback_base = base
        for name in _CANDIDATES:
            candidate = base / name
            if candidate.is_dir():
                return candidate

        # Git/源码项目根目录和用户主目录都是明确边界，不再继续向系统上层找。
        if project_root or base == home:
            break

    return fallback_base / "data"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def game_dir(data_dir: Path, game_id: str) -> Path:
    """一场比赛的所有数据都放在同一个目录下，方便整场删除 / 拷贝。"""
    return data_dir / "kpl" / "games" / game_id
