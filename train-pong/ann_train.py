import torch
import numpy as np
import cv2
import os
from BEN_DQN import Agent, PaiaPongEnv

# -------------------------------
# -------- Settings -------------
# -------------------------------
MODEL_DIR = "saved_models"
LATEST_MODEL = os.path.join(MODEL_DIR, "dqn_latest.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TESTS = 5  # 要重複測試幾次

# -------------------------------
# -------- Create Agent ----------
# -------------------------------
state_dim = 8
action_dim = 3
agent = Agent(state_dim, action_dim, device)

# -------------------------------
# -------- Load Model -----------
# -------------------------------
if os.path.exists(LATEST_MODEL):
    agent.online_net.load_state_dict(torch.load(LATEST_MODEL, map_location=device))
    agent.target_net.load_state_dict(agent.online_net.state_dict())
    print("✅ Loaded latest model")
else:
    print(f"⚠️ {LATEST_MODEL} not found! Running visual test without pre-trained model.")

# -------------------------------
# -------- Visual Test ----------
# -------------------------------
agent.epsilon = 0.0  # 完全 exploitation
print(f"🎮 Start visual test. Running {NUM_TESTS} episodes.")

for test_num in range(1, NUM_TESTS + 1):
    env = PaiaPongEnv()
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0

    print(f"\n--- Test Episode {test_num} ---")

    while not done:
        # 將 state 轉成 tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(agent.online_net(state_tensor)).item()

        # 執行一步
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # 渲染遊戲畫面
        env.render()

        # 判斷遊戲是否結束
        done = terminated or truncated

    env.close()
    cv2.destroyAllWindows()
    print(f"Test Episode {test_num} finished! Steps: {step_count}, Total reward: {total_reward:.2f}")

print("🎮 All visual tests finished!")
