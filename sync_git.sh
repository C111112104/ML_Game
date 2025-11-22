#!/usr/bin/env bash
set -euo pipefail

# === 設定區 ===
# 使用 HTTPS，不需要 SSH 金鑰
REPO_URL="https://github.com/C111112104/ML_Game.git"

# 專案所在資料夾
REPO_DIR="${HOME}/ML_Game"

# 要同步的分支
BRANCH_NAME="main"
# === 設定區結束 ===


echo "[INFO] 目標資料夾：${REPO_DIR}"

# 1. 如果專案不存在，就 clone
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "[INFO] 找不到 .git，第一次執行 clone..."
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
echo "[INFO] 目前路徑：$(pwd)"

# 2. 確保 origin 正確（如果之前用 SSH clone）
CURRENT_URL=$(git remote get-url origin)
if [ "${CURRENT_URL}" != "${REPO_URL}" ]; then
    echo "[INFO] 更新 origin URL → HTTPS"
    git remote set-url origin "${REPO_URL}"
fi

# 3. 拉取更新
echo "[INFO] 取得遠端更新 (fetch)..."
git fetch origin

# 4. 切換分支
if git rev-parse --verify "${BRANCH_NAME}" >/dev/null 2>&1; then
    git checkout "${BRANCH_NAME}"
else
    git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}"
fi

# 5. 顯示狀態
echo
echo "===== Git 狀態（同步前）====="
git status
echo "============================="
echo

# 6. 提醒如果本地有修改
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[WARN] 本地偵測到未 commit 的修改，pull 時可能遇到 conflict。"
fi

# 7. 拉取最新版本（rebase 模式）
echo "[INFO] 從 origin/${BRANCH_NAME} 拉取最新..."
git pull --rebase origin "${BRANCH_NAME}"

echo "[INFO] 同步完成 ✅"
git status
