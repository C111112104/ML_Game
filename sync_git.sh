#!/usr/bin/env bash
# sync_git.sh — 配置 Git Credential Helper，自動同步專案

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/C111112104/ML_Game.git}"
REPO_DIR="${REPO_DIR:-${HOME}/ML_Game}"
BRANCH_NAME="${BRANCH_NAME:-main}"

# 啟用 Git credential-helper (記憶5小時，預設 cache)
git config --global credential.helper 'cache --timeout=18000'

echo "✅ 已啟用 Git Credential Cache（五小時）"

# 首次 clone 或進入已存在的倉庫
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "📦 首次 clone，請輸入 GitHub 帳號及 Personal Access Token（密碼欄）"
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# 確認遠端 origin 是否為 HTTPS
current_url=$(git remote get-url origin)
if [[ "$current_url" != https:* ]]; then
    echo "🔄 調整成 HTTPS 遠端..."
    git remote set-url origin "${REPO_URL}"
fi

echo "✅ 已進入 ${REPO_DIR}"

git pull origin "${BRANCH_NAME}"

# 推送/同步代碼
echo "⌨️ 請輸入 commit 訊息（預設：Auto-sync）"
read -r COMMIT_MSG
COMMIT_MSG="${COMMIT_MSG:-Auto-sync}"

git add .
git commit -am "${COMMIT_MSG}" || echo "⚠️ 無檔案更新可 commit"
git push origin "${BRANCH_NAME}"

echo "🎉 完成 push、只需首次輸入密碼/Token，之後自動記住!"

# 用於取消保存（可選）
# git config --global --unset credential.helper


