# ML_Game - 機器學習遊戲 AI 專題

> 使用強化學習 (Reinforcement Learning) 開發多款遊戲 AI 系統，涵蓋 DQN 等主流程解法

## 🚀 專案簡介

本專題致力於使用強化學習技術開發遊戲 AI 系統，目前包含兩個代表性案例：

1.  **🏓 乒乓球遊戲 AI 系統 (Ping Pong Game AI)**：利用 Deep Q-Network (DQN) 訓練 AI 代理，使其學會高效的乒乓球遊戲策略。
2.  **🎮 Doodle Jump 遊戲 AI 系統**：基於 Q-Learning 演算法，訓練角色在 HTML5 環境中自動跳躍與生存。

---

## 📚 目錄

- [🚀 專案簡介](#-專案簡介)
- [1. 🏓 乒乓球遊戲 AI 系統](#1--乒乓球遊戲-ai-系統)
- [2. 🎮 Doodle Jump 遊戲 AI 系統](#2--doodle-jump-遊戲-ai-系統)
- [📝 簡報 Todo](#-簡報-todo)
- [👥 分工表](#-分工表)
- [📂 檔案結構](#-檔案結構)

---

## 1. 🏓 乒乓球遊戲 AI 系統

本專案的核心是使用 **Deep Q-Network (DQN)** 強化學習演算法，訓練一個 AI 代理來學習並掌握乒乓球遊戲的策略。

### 核心技術
- **深度強化學習 (DRL)**：結合深度學習與強化學習，處理高維度狀態空間。
- **Bellman 方程應用**：用於計算最優動作價值函數。
- **經驗回放 (Experience Replay)**：打破數據相關性，提高訓練穩定性。
- **目標網絡 (Target Network)**：穩定 Q 值更新，防止訓練發散。

![Ping Pong Training Interface](https://github.com/user-attachments/assets/49cd8c15-afc2-4561-94d7-9488ea3d5e6f)
*圖 1：乒乓球 AI 訓練數據*

- DEMO:

https://github.com/user-attachments/assets/162027fa-aeb2-4d6c-8088-23f69ab33ba1

*影片 1：乒乓球 AI 實際運行展示*

🔗 [**線上瀏覽 - 乒乓球 AI 專題報告**](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)

---

## 2. 🎮 Doodle Jump 遊戲 AI 系統

- 使用 **Q-Learning** 演算法訓練 AI 在 HTML5 版 Doodle Jump 中生存。
- 支援瀏覽器端實時訓練與視覺化（Chart.js）。
- **核心特性**：狀態空間離散化、獎勵機制設計、即時決策。

- DEMO:

![Doodle Jump AI Demo](doodle-jump-machine-learning/report/dj.png)
*圖 2：Doodle Jump AI 遊戲畫面與訓練數據*

🔗 [**線上瀏覽 - Doodle Jump AI 專題報告**](https://github.com/C111112104/ML_Game/blob/main/doc/doodle-jump-present.md)

---

## 📝 簡報 Todo

| 日期 | 項目 |
| :-- | :-- |
| 1211 | 關於 `ping-pong-ai-project.md`：1) Dueling 的目標 Q 值在 `Ben_DQN.py` 中目標 Q 值是多少？ 2) Dueling Q 值計算公式中的 A 值代表 reward 嗎？在 `Ben_DQN.py` 中每次給的 reward 是多少？ |
| 1218 | (空 / 待補) |

1. 補 breakdown diargam , API table
2. breakdown diargam , API table 需要對應
3. 對各個 breakdown 的 submodule 單元測試 結果，貼到簡報
4. 找 loss function 曲線結果圖

---

## 👥 分工表

| 組員 | 工作量 |
| :--- | :--- |
| 博皓 | 33% |
| 亞倫 | 33% |
| 東穎 | 33% |

---

## 📂 檔案結構

```text
.
├── doodle-jump-machine-learning/ # Doodle Jump AI 原始碼 (HTML5 + JS)
│   ├── index.html         # 遊戲入口網頁
│   ├── scripts/           # 遊戲邏輯與 AI 演算法
│   │   ├── GameLogic.js   # 遊戲主邏輯
│   │   └── QLearning.js   # Q-Learning 核心
│   └── report/            # 訓練結果圖表
├── doc/                   # 專案技術文件與詳細報告
│   ├── ping-pong-ai-project.md # 乒乓球遊戲 AI 技術分析與驗證
│   ├── doodle-jump-present.md  # Doodle Jump AI 系統設計報告
│   └── TetrAI_Proposal.md      # (歷史) TetrAI 提案文件
├── train-pong/            # 乒乓球 AI 訓練模組與歷史數據
│   ├── saved_models/      # 存放訓練過程中的模型權重檔 (.pth)
│   ├── BEN_DQN.py         # Dueling Double DQN 模型架構實現
│   ├── ann_train.py       # 訓練主程式與環境對接邏輯
│   ├── training_log.csv   # 訓練記錄檔
│   └── runs/              # TensorBoard 視覺化訓練日誌
├── sync_git.sh            # 自動化 Git 同步工作腳本
└── README.md              # 專案總覽與導覽說明
```
