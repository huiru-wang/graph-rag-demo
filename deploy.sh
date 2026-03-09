#!/bin/bash

# --- 自动部署脚本 ---
# 设置项目路径 (请根据服务器实际情况修改)
PROJECT_DIR="/root/graph-rag-demo"
SERVICE_NAME="graph-rag"

echo "🚀 开始部署更新..."

# 1. 进入项目目录
cd $PROJECT_DIR || { echo "❌ 目录 $PROJECT_DIR 不存在"; exit 1; }

# 2. 拉取最新代码
echo "📥 正在拉取 git 更新..."
git pull origin main
if [ $? -ne 0 ]; then
    echo "❌ Git pull 失败，请检查网络或冲突"
    exit 1
fi

# 3. 同步依赖 (使用 uv)
# 如果 pyproject.toml 或 uv.lock 有更新，这一步非常重要
echo "📦 正在同步 Python 依赖..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "⚠️ 未找到 uv 命令，跳过依赖同步 (请确保 uv 已安装并加入 PATH)"
fi

# 4. 重启系统服务
echo "🔄 正在重启 systemd 服务: $SERVICE_NAME..."
sudo systemctl restart $SERVICE_NAME

# 5. 检查服务状态
echo "📊 检查服务状态..."
sleep 2
systemctl status $SERVICE_NAME --no-pager | grep "Active:"

echo "✅ 部署完成！"
