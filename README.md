# ML_Game - 機器學習遊戲 AI 專題

> 使用強化學習 (Reinforcement Learning) 開發多款遊戲 AI 系統，涵蓋 DQN 等主流程解法

## 🚀 專案簡介

本專題致力於使用強化學習技術開發遊戲 AI 系統，目前包含兩個代表性案例：

1.  **🏓 乒乓球遊戲 AI 系統 (Ping Pong Game AI)**：利用 Deep Q-Network (DQN) 訓練 AI 代理，使其學會高效的乒乓球遊戲策略。
2.  **🎮 TetrAI - 俄羅斯方塊對戰 AI**：強化學習 AI 智能體可自動學習 Tetris 遊戲策略，支援進階行為（B2B、T-Spin、Ghost Piece 等）。

---

## 📚 目錄

- [🚀 專案簡介](#-專案簡介)
- [1. 🏓 乒乓球遊戲 AI 系統](#1--乒乓球遊戲-ai-系統)
- [2. 🎮 TetrAI - 俄羅斯方塊對戰 AI](#2--tetrai---俄羅斯方塊對戰-ai)
- [🗂️ 內容導覽](#-內容導覽)
- [🕹️ Tetris AI 專案完整執行 SOP](#-tetris-ai-專案完整執行-sop)
- [📄 授權](#-授權)
- [👥 分工表](#-分工表)

---

## 1. 🏓 乒乓球遊戲 AI 系統

本專案的核心是使用 **Deep Q-Network (DQN)** 強化學習演算法，訓練一個 AI 代理來學習並掌握乒乓球遊戲的策略。

### 核心技術
- **深度強化學習 (DRL)**：結合深度學習與強化學習，處理高維度狀態空間。
- **Bellman 方程應用**：用於計算最優動作價值函數。
- **經驗回放 (Experience Replay)**：打破數據相關性，提高訓練穩定性。
- **目標網絡 (Target Network)**：穩定 Q 值更新，防止訓練發散。

| 指標 | 目標值 |
| :--- | :--- |
| 接球成功率 | ≥85% |
| 推理延遲 | <30 ms |
| 訓練收斂 | ≤2 小時 |

🔗 [**線上瀏覽 - 乒乓球 AI 專題報告**](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)

---

## 2. 🎮 TetrAI - 俄羅斯方塊對戰 AI

- 強化學習 AI 智能體可自動學習 Tetris 遊戲策略。
- 嚴謹 Socket 架構（Processing 遊戲引擎 ＋ Python 智能體）。
- **核心特性**：多代理訓練、DQN 與遺傳演算法混合策略、高維度特徵工程。

- DEMO: [https://github.com/user-attachments/assets/882650b6-983f-4413-ace2-96f905f11f89](https://github.com/user-attachments/assets/882650b6-983f-4413-ace2-96f905f11f89)

🔗 [**TetrAI 專題詳細提案**](https://github.com/C111112104/ML_Game/blob/main/doc/TetrAI_Proposal.md)

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

## 🕹️ Tetris AI 專案完整執行 SOP

本專案分為兩個必須同時運行的組件：Processing (遊戲前端/伺服器) 和 Python (AI 後端/客戶端)。

### 🎯 必備工具與環境設定
| 項目 | 說明 | 取得方式 |
| :--- | :--- | :--- |
| Processing IDE | 用於執行遊戲介面和網路伺服器 | 官方網站下載 |
| Python 3 | 用於執行 AI 訓練和決策 | 官方網站下載 |

### 🛠️ 步驟一：環境依賴安裝
1. **Processing**: 安裝 `Minim` 函式庫。
2. **Python**: `pip install keras numpy pillow tqdm tensorboard opencv-python jax`

### 🚀 步驟二：啟動 Processing 遊戲前端
開啟 `main.pde` 並點擊 **Run (▶️)**。

### 🧠 步驟三：運行 Python AI 後端
- **訓練模式**: `python3 run.py`
- **執行模式**: `python3 run_model.py sample.keras`

---

## 📄 授權
本專案採用 [MIT License](LICENSE) 授權。

---

## 👥 分工表

| 組員 | 工作量 |
| :--- | :--- |
| 博皓 | 33% |
| 亞倫 | 33% |
| 東穎 | 33% |