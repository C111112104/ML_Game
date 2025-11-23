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

## 🔗 相關文檔

- [乒乓球 AI 專題簡報](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)
- [TetrAI 俄羅斯方塊 AI 專題提案](https://github.com/C111112104/ML_Game/blob/main/doc/TetrAI_Proposal.md)  ← 含多張系統流程圖、分析架構圖（推薦線上瀏覽）

---

## 📣 補充

- **TetrAI 方案持續更新中，歡迎 PR 與 Issue 討論！**
- 圖表建議直接在 GitHub 上瀏覽以獲得最佳互動體驗。
- 任何技術討論、錯誤修正請透過 Issue/New Discussion 提出。

---
