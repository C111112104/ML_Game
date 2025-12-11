# ML_Game - 機器學習遊戲 AI 專題

> 使用強化學習 (Reinforcement Learning) 開發多款遊戲 AI 系統，涵蓋 DQN 等主流程解法

## 📚 專題內容

本專題包含兩個代表性案例：

### 1. 🏓 乒乓球遊戲 AI（Ping Pong Game）

- 利用 Deep Q-Network (DQN) 強化學習訓練遊戲 AI
- 實作細節與理論請參考：[乒乓球 AI 專題簡報](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)

**特色重點：**

- 深度強化學習智能體設計
- Bellman 方程、自動回放、目標網路等核心技術
- 經完整測試與實證

| 指標 | 目標值 |
| :-- | :-- |
| 接球成功率 | ≥85% |
| 推理延遲 | <30 ms |
| 訓練收斂 | ≤2 小時 |


---

### 2. 🎮 TetrAI - 俄羅斯方塊 AI 對戰系統

- 強化學習 AI 智能體可自動學習 Tetris 遊戲策略
- 支援進階行為（B2B、T-Spin、Ghost Piece 等）
- 嚴謹 Socket 架構（Processing 遊戲引擎＋Python 智能體）
- 詳細解構與流程圖，請見：[TetrAI 專題詳細提案](https://github.com/C111112104/ML_Game/blob/main/doc/TetrAI_Proposal.md)

**核心特性：**

- 多代理訓練（未來將支援 PvP 對戰）
- DQN 與遺傳演算法混合策略
- 高維度特徵工程與資料驅動學習
- 完整模組分解、架構流程、系統分析
- 所有關鍵架構圖直接用 Markdown 標記語法書寫

---

## 🗂️ 內容導覽

| 區塊 | 內容 | 重點補充 |
| :-- | :-- | :-- |
| **需求分析** | 功能、規格、效能、驗收方法 | 8 項核心定義 |
| **系統分析** | 系統架構、資料流、特徵分析 | 包含詳細架構流程 |
| **系統設計** | 模組結構、分支圖、架構圖 | Markdown 圖一致呈現 |
| **編碼實現** | .py / .java 主核心程式 | 俄羅斯方塊分階段更新 |
| **驗證測試** | 評估指標、效能測量 | 測試/可靠性報告 |


---

🕹️ Tetris AI 專案完整執行 SOP (Markdown 格式)此 SOP 旨在指導使用者成功運行 Tetris AI 專案，該專案分為兩個必須同時運行的組件：Processing (遊戲前端/伺服器) 和 Python (AI 後端/客戶端)。🎯 必備工具與環境設定項目說明取得方式Processing IDE用於執行遊戲介面和網路伺服器。官方網站下載並安裝。Python 3用於執行 AI 訓練和決策的核心邏輯。官方網站下載並安裝。專案原始碼包含所有的 .pde (Processing) 和 .py (Python) 檔案。🛠️ 步驟一：處理 Processing 環境依賴項這是確保遊戲前端能夠順利啟動的關鍵步驟，特別是解決 Minim 函式庫的錯誤。1.1 安裝 Minim 函式庫 (解決 ddf.minim 錯誤)開啟 Processing IDE。點選頂部菜單 「Sketch」 (程式) -> 「Import Library...」 (匯入函式庫)。點選彈出子菜單最下方的 「Add Library...」 (新增函式庫)。在彈出的 Contribution Manager 視窗中，搜尋 minim。點選 「Minim」 函式庫，然後點選右下角的 「Install」 (安裝)。建議： 安裝完成後，重新啟動 Processing IDE。1.2 Python 函式庫安裝打開終端機 (Terminal/Command Prompt)，安裝 AI 運行所需的核心依賴：Bash# 基礎依賴
pip install keras numpy pillow tqdm tensorboard opencv-python

# 選擇後端：建議使用 Jax 以獲得更快的訓練/預測速度
pip install jax[cuda12]  # 如果有NVIDIA GPU
# 或
pip install jax         # 如果沒有GPU
提示： 如果決定使用 Jax 作為後端，執行 Python 腳本前，請在終端機中設定：export KERAS_BACKEND="jax"🚀 步驟二：啟動 Processing 遊戲前端 (伺服器)遊戲前端必須先啟動，以便在特定端口上監聽，等待 Python AI 的連線。開啟專案： 使用 Processing IDE 開啟包含所有 .pde 檔案的專案資料夾 (例如 main.pde)。啟動伺服器： 點擊 IDE 頂部的 「執行 (Run)」 按鈕 (▶️)。確認狀態：一個名為 "TetrAI" 的遊戲視窗會彈出。Processing 控制台 (Console) 應該顯示 ServerHandler 已啟動並正在等待連線的訊息。核心功能： 程式碼中的 serverhandler.startServer() 啟動了遊戲的伺服器，準備接收來自 AI 的動作指令。🧠 步驟三：運行 Python AI 後端 (客戶端)一旦遊戲伺服器運行，就可以啟動 AI 模型作為客戶端連線，開始遊戲。選擇模式 (二選一)模式目的執行指令說明1. 訓練模式讓 AI 從零開始學習 (DQL)，探索並優化遊戲策略。python3 run.py程式將開始多個回合 (Episodes) 的訓練。2. 執行模式使用一個已訓練好的模型來玩遊戲，測試其最終性能。python3 run_model.py sample.keras載入 sample.keras 模型並直接開始遊戲。執行與監控在終端機中輸入您選擇模式的指令並執行。遊戲開始： Python 客戶端將立即連線到 Processing 伺服器。一旦連線成功，Processing 視窗中的 Tetris 遊戲就會開始自動進行。監控訓練 (僅限訓練模式)：開啟另一個終端機視窗，輸入：tensorboard --logdir ./logs開啟瀏覽器訪問 TensorBoard 提供的網址，即可視覺化監控 AI 的分數和訓練進度。✅ 流程總結組件執行順序執行指令/操作Processing1 (先)**開啟 IDE 並點擊 「Run」 (啟動伺服器/遊戲介面)Python2 (後)**終端機執行 python3 run.py 或 python3 run_model.py [model] (啟動客戶端/AI)

---

## 🔗 相關文檔

- [乒乓球 AI 專題簡報](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)
- [TetrAI 俄羅斯方塊 AI 專題提案](https://github.com/C111112104/ML_Game/blob/main/doc/TetrAI_Proposal.md)  ← 含多張系統流程圖、分析架構圖（推薦線上瀏覽）

---

## 📣 補充

- **TetrAI 方案持續更新中，歡迎 PR 與 Issue 討論！**
- 圖表建議直接在 GitHub 上瀏覽以獲得最佳互動體驗。
- 任何技術討論、錯誤修正請透過 Issue/New Discussion 提出。

---
