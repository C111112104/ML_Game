# ML_Game - 機器學習遊戲 AI 專題

> 使用強化學習 (Reinforcement Learning) 開發遊戲 AI 系統

## 📚 專題內容

本專題包含兩個機器學習應用案例：

### 1. 🏓 乒乓球遊戲 AI 系統 (Ping Pong Game)


## 📊 研究簡報

📌 **完整專題簡報（含圖表與代碼）：**

🔗 [線上瀏覽 - 乒乓球 AI 專題簡報](https://github.com/C111112104/ML_Game/blob/main/doc/ping-pong-ai-project.md)

使用 **Deep Q-Network (DQN)** 強化學習演算法，訓練 AI 代理學習乒乓球遊戲策略。

**核心特性：**
- ✅ 深度強化學習 (DRL) 實現
- ✅ Bellman 方程應用
- ✅ 經驗回放 (Experience Replay) 優化
- ✅ 目標網絡 (Target Network) 穩定化
- ✅ 完整的測試與驗證流程

**目標指標：**
| 指標 | 目標值 | 
|------|--------|
| 接球成功率 | ≥85% |
| 推理延遲 | <30 ms |
| 訓練收斂時間 | ≤2 小時 |

---

### 2. 🎮 TetrAI - 俄羅斯方塊對戰 AI (待開發)

進階項目：使用多代理強化學習訓練能與人類競爭的方塊遊戲 AI。

**預計功能：**
- 支援 B2B、T-Spin、Tetris、Ghost Piece 等進階機制
- Socket Server 架構（Processing + Python）
- 遺傳算法與 DQN 混合策略

---

**簡報內容結構：**

| 部分 | 內容描述 | 重點 |
|------|---------|------|
| **需求分析** | 功能需求、規格、效能、驗收方法 | 8 個核心功能定義 |
| **系統分析** | 系統邊界、用例圖、資料流圖 | DFD 完整設計 |
| **系統設計** | 模組分支圖、MSC 圖、架構圖 | ⭐ 含 Mermaid 圖表 |
| **編碼實現** | game_engine.py、dqn_agent.py、train.py | 完整 Python 代碼 |
| **驗證與測試** | 測試計劃、評估指標、性能基準 | 實測結果展示 |

