#!/usr/bin/env bash
set -euo pipefail

echo "=== 随机小脚本开始 ==="
echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "当前用户: $(whoami)"
echo "当前目录: $(pwd)"

echo "随机数: $RANDOM"
echo "再来一个随机数: $RANDOM"

echo "主目录内容预览:"
ls -1 "$HOME" | head -n 10

echo "=== 脚本结束 ==="
