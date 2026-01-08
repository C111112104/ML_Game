# ML_Game - 機器學習遊戲 AI 專題

> 使用強化學習 (Reinforcement Learning) 開發多款遊戲 AI 系統，涵蓋 DQN 等主流程解法

## 📚 專題內容

本專題包含兩個代表性案例：

### 1. 🏓 乒乓球遊戲 AI（Ping Pong Game）

- 利用 Deep Q-Network (DQN) 強化學習訓練遊戲 AI
- 實作細節與理論請參考：[乒乓球 AI 專題簡報](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)
<img width="605" height="293" alt="image" src="https://github.com/user-attachments/assets/49cd8c15-afc2-4561-94d7-9488ea3d5e6f" />
- DEMO
 


https://github.com/user-attachments/assets/162027fa-aeb2-4d6c-8088-23f69ab33ba1




---

### 2. 🎮 TetrAI - 俄羅斯方塊 AI 對戰系統

- 強化學習 AI 智能體可自動學習 Tetris 遊戲策略
- 支援進階行為（B2B、T-Spin、Ghost Piece 等）
- 詳細解構與流程圖，請見：[TetrAI 專題詳細提案](https://github.com/C111112104/ML_Game/blob/main/doc/TetrAI_Proposal.md)

- DEMO:
- https://github.com/user-attachments/assets/882650b6-983f-4413-ace2-96f905f11f89
- 
---

## Tetris AI 專案 SOP

```mermaid
graph LR
  A[SOP 目標\n同時啟動 Processing（前端／伺服器）與 Python 智能體（後端）] --> B[主要相依套件\n詳見下方套件說明]
  B --> C[後端選擇\n建議使用 Jax 加速訓練與推論]
  C --> D[環境設定\n如使用 Jax，請將 KERAS_BACKEND 設為 jax]
  D --> E[啟動 Processing（遊戲前端／伺服器）\n確認伺服器埠號與設定]
  E --> F[啟動 Python 智能體（後端）\n確認 Socket 主機／埠與 Processing 對應]
  F --> G[驗證與 DEMO\n檢查連線、觀察智能體行為、查看日誌]
  G --> H[備註\n原說明文件有截斷／亂碼，建議補上完整 SOP 與啟動步驟]
```

---

## 簡報 Todo

| 日期 | 項目 |
| :-- | :-- |
| 1211 | 關於 `ping-pong-ai-project.md`：1) Dueling 的目標 Q 值在 `Ben_DQN.py` 中目標 Q 值是多少？ 2) Dueling Q 值計算公式中的 A 值代表 reward 嗎？在 `Ben_DQN.py` 中每次給的 reward 是多少？ |
| 1218 | (空 / 待補) |

---
