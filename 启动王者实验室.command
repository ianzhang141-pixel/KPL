#!/bin/zsh

# 双击即可启动。项目和数据目录都以这个文件所在位置为准。
cd -- "${0:A:h}" || exit 1
mkdir -p data
(sleep 1; open "http://127.0.0.1:8020") &
exec /usr/bin/env python3 -m kplab serve
