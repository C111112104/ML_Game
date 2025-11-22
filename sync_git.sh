#!/usr/bin/env bash
# Interactive Git Menu for ML_Game
set -euo pipefail

# ========= 可覆寫的預設值（亦支援命令列參數） =========
REPO_URL="${REPO_URL:-https://github.com/C111112104/ML_Game.git}"
REPO_DIR="${REPO_DIR:-${HOME}/ML_Game}"
BRANCH_NAME="${BRANCH_NAME:-main}"
# ==================================================

# ---- 參數解析 (--url/--dir/--branch) ----
for arg in "$@"; do
  case "$arg" in
    --url=*)    REPO_URL="${arg#*=}";;
    --dir=*)    REPO_DIR="${arg#*=}";;
    --branch=*) BRANCH_NAME="${arg#*=}";;
    *) ;;
  esac
done

# ---- 顏色與輸出 ----
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "🔷 %s\n" "$*"; }
ok()   { printf "✅ %s\n" "$*"; }
warn() { printf "⚠️  %s\n" "$*"; }
err()  { printf "💥 %s\n" "$*" 1>&2; }

trap 'err "執行過程發生錯誤，請檢查上述訊息。"' ERR

# ---- 基礎檢查 & 進入倉庫 ----
ensure_repo() {
  info "目標資料夾：${REPO_DIR}"
  if [ ! -d "${REPO_DIR}/.git" ]; then
    warn "找不到 .git，第一次執行 clone..."
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
  cd "${REPO_DIR}"

  # 確保 origin URL 正確（HTTPS）
  local current_url
  current_url="$(git remote get-url origin 2>/dev/null || echo "")"
  if [ -z "$current_url" ]; then
    git remote add origin "${REPO_URL}"
  elif [ "$current_url" != "$REPO_URL" ]; then
    info "更新 origin URL → ${REPO_URL}"
    git remote set-url origin "${REPO_URL}"
  fi

  # 確保分支存在並切換
  git fetch origin --prune
  if git rev-parse --verify "${BRANCH_NAME}" >/dev/null 2>&1; then
    git checkout "${BRANCH_NAME}"
  else
    git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}" || git checkout -b "${BRANCH_NAME}"
  fi
}

# ---- 功能函式 ----
do_status() { ensure_repo; echo; bold "===== Git 狀態 ====="; git status; echo; }

do_sync() {
  ensure_repo
  bold "同步（fetch → rebase pull）"
  # 提醒本地未提交
  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "偵測到未提交的變更，pull 可能產生衝突。"
  fi
  git fetch origin
  git pull --rebase origin "${BRANCH_NAME}"
  ok "同步完成"
  git status
}

do_push() {
  ensure_repo
  bold "推送（add → commit → push）"
  git add -A

  if git diff --cached --quiet; then
    warn "沒有 staged 變更可提交。"
  else
    printf "✍️  請輸入提交訊息（預設：update）: "
    read -r msg
    msg="${msg:-update}"
    git commit -m "$msg"
  fi

  # 設定 upstream（首次推送）
  if ! git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
    info "設定 upstream: origin/${BRANCH_NAME}"
    git push -u origin "${BRANCH_NAME}"
  else
    git push
  fi
  ok "推送完成"
}

do_stash()     { ensure_repo; bold "儲存工作區變更 (stash)"; git stash push -u -m "work-in-progress $(date +%F_%T)"; ok "已建立 stash"; git stash list; }
do_stash_pop() { ensure_repo; bold "套用最近一次 stash"; git stash list | head -n1 || true; git stash pop || warn "無可套用的 stash"; }
do_log()       { ensure_repo; bold "最近 15 筆提交"; git log --oneline --graph --decorate -15; }
do_new_branch(){
  ensure_repo
  printf "🪄 新分支名稱: "
  read -r nb
  [ -z "$nb" ] && { warn "名稱不可為空"; return; }
  git checkout -b "$nb"
  ok "已建立並切換到分支: $nb"
}
do_switch_branch(){
  ensure_repo
  info "現有本地分支："
  git for-each-ref --format='%(refname:short)' refs/heads/
  printf "🔀 要切換的分支: "
  read -r sb
  [ -z "$sb" ] && { warn "名稱不可為空"; return; }
  git checkout "$sb"
  ok "已切換分支: $sb"
}
do_set_origin(){
  ensure_repo
  printf "🔗 新的遠端 URL (空白則維持): "
  read -r nu
  [ -z "$nu" ] && { warn "未變更"; return; }
  git remote set-url origin "$nu"
  ok "origin 已更新為：$nu"
}
do_tag(){
  ensure_repo
  printf "🏷️  標籤名稱: "
  read -r tn
  [ -z "$tn" ] && { warn "名稱不可為空"; return; }
  printf "📝 標籤說明(可空白): "
  read -r ta
  if [ -z "$ta" ]; then git tag "$tn"; else git tag -a "$tn" -m "$ta"; fi
  printf "是否推送標籤到遠端？(y/N): "
  read -r yn
  [[ "${yn,,}" == "y" ]] && git push origin "$tn"
  ok "標籤已建立"
}
do_backup_branch(){
  ensure_repo
  local ts="backup/${BRANCH_NAME}-$(date +%Y%m%d-%H%M%S)"
  git branch "$ts"
  ok "已建立備份分支：$ts"
}
do_amend_last(){
  ensure_repo
  printf "✍️  修正最近一次提交訊息(空白則沿用原訊息): "
  read -r m
  if [ -z "$m" ]; then git commit --amend --no-edit; else git commit --amend -m "$m"; fi
  ok "已修正最近一次提交"
}

# ---- 選單 ----
menu() {
  bold "Git 選單（倉庫：$REPO_DIR，分支：$BRANCH_NAME）"
  echo "輸入數字或別名執行："
  cat <<'EOF'
1) sync           - 同步遠端 (fetch + pull --rebase)
2) push           - 推送變更 (add + commit + push)
3) status         - 顯示狀態
4) log            - 顯示最近提交
5) stash          - 儲存暫存工作
6) stash-pop      - 套用最近暫存
7) new-branch     - 新建分支
8) switch-branch  - 切換分支
9) set-origin     - 變更遠端 URL
10) tag           - 建立標籤（可選推送）
11) backup-branch - 建立備份分支
12) amend-last    - 修正最近一次提交
q) quit           - 離開
EOF
  printf "👉 指令："
}

# ---- 主迴圈 ----
while true; do
  menu
  read -r cmd
  case "${cmd}" in
    1|sync)            do_sync;;
    2|push)            do_push;;
    3|status)          do_status;;
    4|log)             do_log;;
    5|stash)           do_stash;;
    6|stash-pop)       do_stash_pop;;
    7|new-branch)      do_new_branch;;
    8|switch-branch)   do_switch_branch;;
    9|set-origin)      do_set_origin;;
    10|tag)            do_tag;;
    11|backup-branch)  do_backup_branch;;
    12|amend-last)     do_amend_last;;
    q|quit|exit)       ok "Bye!"; exit 0;;
    *) warn "未知指令：$cmd";;
  esac
  echo
done

