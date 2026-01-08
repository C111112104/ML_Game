import numpy as np
import torch
import cv2
import time
import psutil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# 讀 CSV
df = pd.read_csv(r"C:\Users\USER\Desktop\training_log.csv")  # 改成你的路徑

# 畫 reward 曲線
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['avg_reward'], label='Avg reward')
plt.plot(df['episode'], df['avg_reward_100'], label='100-episode avg')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()
plt.grid()
plt.show()

# 畫 epsilon 曲線
plt.figure(figsize=(10,5))
plt.plot(df['episode'], df['epsilon'], label='Epsilon')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')
plt.grid()
plt.show()

import csv
import os
CSV_PATH = "training_log.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "episode",
            "avg_reward",
            "avg_reward_100",
            "avg_loss",
            "epsilon",
            "hits",
            "mem_MB",
            "elapsed_hours"
        ])
from collections import deque

from gymnasium.vector import SyncVectorEnv

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/paia_pong")
from collections import deque

# 用滑動視窗記錄最近 100 集 reward（可自己調整）
reward_window = deque(maxlen=100)
loss_window = deque(maxlen=100)
epsilon_window = deque(maxlen=100)
hits_window = deque(maxlen=100)


MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
# -------------------------------
# -------- PAIA Pong Env -------
# -------------------------------

class PaiaPongEnv:
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode  # <- 新增這行
        self.W = 200
        self.H = 500
        self.paddle_w = 40
        self.paddle_h = 10
        self.ball_r = 5
        self.paddle_speed = 5
        self.action_space_n = 3  # 0=stay,1=left,2=right

        # P1
        self.paddle_x = 80
        self.paddle_y = 420

        # P2
        self.paddle2_x = 80
        self.paddle2_y = 50

        self.total_hits = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.ball_x = self.W / 2
        self.ball_y = self.H / 2
        v = np.random.choice([7, 10])
        self.ball_vx = np.random.choice([-v, v])
        self.ball_vy = np.random.choice([-v, v])

        self.paddle_x = 80
        self.paddle_y = 420

        self.paddle2_x = 80
        self.paddle2_y = 50

        self.total_hits = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.ball_x / self.W,
            self.ball_y / self.H,
            self.ball_vx / 15.0,
            self.ball_vy / 15.0,
            self.paddle_x / self.W,
            self.paddle_y / self.H,
            self.paddle2_x / self.W,
            self.paddle2_y / self.H
        ], dtype=np.float32)

    def step(self, action):
        # --- P1 移動 ---
        if action == 1:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed
        self.paddle_x = np.clip(self.paddle_x, 0, self.W - self.paddle_w)

        # --- P2 簡單 AI ---
        # ---- P2 預測落點 ----
        if self.ball_vy < 0:  # 球往上
            t = (self.paddle2_y - self.ball_y) / self.ball_vy
            predict_x = self.ball_x + self.ball_vx * t
        else:
            predict_x = self.ball_x

        # 牆壁反射修正
        while predict_x < 0 or predict_x > self.W:
            if predict_x < 0:
                predict_x = -predict_x
            elif predict_x > self.W:
                predict_x = 2 * self.W - predict_x

        target_x = predict_x - self.paddle_w / 2
        if target_x < self.paddle2_x:
            self.paddle2_x -= self.paddle_speed
        elif target_x > self.paddle2_x:
            self.paddle2_x += self.paddle_speed

        # --- 分段更新球的位置，避免穿板 ---
        max_step = max(abs(self.ball_vx), abs(self.ball_vy))
        substeps = int(max_step / 2) + 1  # 將步長拆成小段
        reward = 0.0
        done = False

        for _ in range(substeps):
            self.ball_x += self.ball_vx / substeps
            self.ball_y += self.ball_vy / substeps

            # --- 碰撞牆壁 ---
            if self.ball_x - self.ball_r < 0:
                self.ball_x = self.ball_r
                self.ball_vx *= -1
            elif self.ball_x + self.ball_r > self.W:
                self.ball_x = self.W - self.ball_r
                self.ball_vx *= -1

            # --- P1 碰撞 ---
            if (self.ball_y + self.ball_r >= self.paddle_y and
                    self.ball_y - self.ball_r <= self.paddle_y + self.paddle_h and
                    self.ball_x + self.ball_r >= self.paddle_x and
                    self.ball_x - self.ball_r <= self.paddle_x + self.paddle_w and
                    self.ball_vy > 0):  # 球往下打
                self.ball_y = self.paddle_y - self.ball_r
                hit_pos = (self.ball_x - self.paddle2_x) / self.paddle_w  # 0~1
                angle = (hit_pos - 0.5) * 2  # -1 ~ 1
                self.ball_vx += angle * 2
                self.ball_vx = np.clip(self.ball_vx, -12, 12)

                self.total_hits += 1
                reward += 0.5 + 0.1 * self.total_hits

            # --- P2 碰撞 ---
            if (self.ball_y - self.ball_r <= self.paddle2_y + self.paddle_h and
                    self.ball_y + self.ball_r >= self.paddle2_y and
                    self.ball_x + self.ball_r >= self.paddle2_x and
                    self.ball_x - self.ball_r <= self.paddle2_x + self.paddle_w and
                    self.ball_vy < 0):  # 球往上打
                self.ball_y = self.paddle2_y + self.paddle_h + self.ball_r
                hit_pos = (self.ball_x - self.paddle2_x) / self.paddle_w  # 0~1
                angle = (hit_pos - 0.5) * 2  # -1 ~ 1
                self.ball_vx += angle * 2
                self.ball_vx = np.clip(self.ball_vx, -12, 12)

                self.total_hits += 1

            # --- 出界判定 ---
            if self.ball_y - self.ball_r > self.H:
                done = True
                reward = -2.0  # 原本 -1

                break
            elif self.ball_y + self.ball_r < 0:
                done = True
                reward = 1.0
                break

        return self._get_obs(), reward, done, False, {}

    def render(self):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # ball
        cv2.circle(img, (int(self.ball_x), int(self.ball_y)), self.ball_r, (255, 255, 255), -1)

        # P1
        cv2.rectangle(img, (int(self.paddle_x), int(self.paddle_y)),
                      (int(self.paddle_x + self.paddle_w), int(self.paddle_y + self.paddle_h)), (255, 255, 255), -1)

        # P2
        cv2.rectangle(img, (int(self.paddle2_x), int(self.paddle2_y)),
                      (int(self.paddle2_x + self.paddle_w), int(self.paddle2_y + self.paddle_h)), (255, 255, 255), -1)

        cv2.imshow("PAIA Pong DQN", img)
        cv2.waitKey(16)

    @property
    def observation_space(self):
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    @property
    def action_space(self):
        from gymnasium import spaces
        return spaces.Discrete(self.action_space_n)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass


# -------------------------------
# --------- DQN MLP Model -------
# -------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------
# --------- Replay Buffer -------
# -------------------------------
class ReplayBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idx])
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# --------- Agent ---------------
# -------------------------------
class Agent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.online_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(50000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.online_net(state)
            return int(torch.argmax(q).item())

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        q_values = self.online_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -------------------------------
# -------- Evaluation -----------
# -------------------------------
def evaluate_agent(agent, env, n_episodes=10, max_steps=1000):
    """
    使用固定策略 (epsilon=0) 評估 agent
    回傳平均 reward
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 完全 exploitation

    rewards = []
    hits_list = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        hits = 0

        while not done and step_count < max_steps:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            # 如果你的 env 有 hits 計數，可以這樣加
            if hasattr(env, "total_hits"):
                hits = env.total_hits
            done = terminated or truncated

        rewards.append(episode_reward)
        hits_list.append(hits)
        print(f"  Eval Episode {ep+1}: reward={episode_reward:.3f} | hits={hits}")

    agent.epsilon = old_epsilon  # 恢復訓練 epsilon
    avg_reward = sum(rewards) / len(rewards)
    avg_hits = sum(hits_list) / len(hits_list)
    print(f"  >> Avg reward: {avg_reward:.3f} | Avg hits: {avg_hits:.2f}")
    return avg_reward

# -------------------------------
# --------- Training Loop -------
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_ENVS = 32
    MAX_EPISODES = 1000
    MAX_STEPS = 5000
    BATCH_SIZE = 32
    TARGET_UPDATE = 50
    EVAL_INTERVAL = 50
    EVAL_EPISODES = 5
    SAVE_INTERVAL = 50  # 每 50 集存檔


    # vector env
    def make_env():
        return PaiaPongEnv()


    envs = SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    obs, _ = envs.reset()
    obs = obs.astype(np.float32)
    state_dim = 8
    action_dim = 3
    agent = Agent(state_dim, action_dim, device)

    # -------------------------------
    # ---- Load latest model once ---
    # -------------------------------
    LATEST_MODEL = os.path.join(MODEL_DIR, "dqn_latest.pth")
    LATEST_BUFFER = os.path.join(MODEL_DIR, "replay_buffer_latest.pkl")

    if os.path.exists(LATEST_MODEL):
        agent.online_net.load_state_dict(
            torch.load(LATEST_MODEL, map_location=device)
        )
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print("✅ Loaded latest model")

    if os.path.exists(LATEST_BUFFER):
        with open(LATEST_BUFFER, "rb") as f:
            loaded_memory = pickle.load(f)

        if len(loaded_memory) > 0:
            agent.memory = loaded_memory
            print(f"✅ Loaded replay buffer, size={len(agent.memory)}")
        else:
            print("⚠️ Found replay buffer but it's empty, ignore it")
    else:
        print("ℹ️ No replay buffer found, start fresh")

    total_step = 0
    SAVE_INTERVAL = 50  # 每 50 集存檔
    reward_window = deque(maxlen=100)  # 如果還沒定義
    loss_window = deque(maxlen=100)  # 如果還沒定義
    start_time = time.time()

    # -------------------------------
    # -------- Training Loop --------
    # -------------------------------

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = envs.reset()
        total_rewards = np.zeros(NUM_ENVS)
        episode_loss = 0
        episode_steps = 0

        for step in range(MAX_STEPS):
            actions = np.array([agent.act(o) for o in obs])
            next_obs, rewards, terminated, truncated, info = envs.step(actions)

            for i in range(NUM_ENVS):
                done_i = terminated[i] or truncated[i]  # ✅ i 在這裡才存在
                agent.memory.push(
                    obs[i],
                    actions[i],
                    rewards[i],
                    next_obs[i],
                    done_i
                )

            obs = next_obs
            total_step += 1

            agent.decay_epsilon()
            loss = agent.train(BATCH_SIZE)
            episode_loss += loss
            episode_steps += 1
            total_rewards += rewards

            if step % TARGET_UPDATE == 0:
                agent.update_target()

            if np.all(terminated):
                break

        avg_reward = total_rewards.mean()
        reward_window.append(avg_reward)
        if len(reward_window) > 100:
            reward_window.pop(0)

        avg_reward_100 = np.mean(reward_window)
        reward_std_100 = np.std(reward_window)
        avg_loss = episode_loss / max(episode_steps, 1)

        elapsed_time = time.time() - start_time

        # 獲取目前系統記憶體用量 (MB)
        process = psutil.Process(os.getpid())
        mem_used = process.memory_info().rss / (1024 * 1024)  # MB
        hits_sum = sum(env.total_hits for env in envs.envs)

        # TensorBoard 紀錄
        writer.add_scalar("train/avg_reward", float(avg_reward), episode)
        writer.add_scalar("train/avg_reward_100", float(avg_reward_100), episode)
        writer.add_scalar("train/reward_std_100", float(reward_std_100), episode)
        writer.add_scalar("train/avg_loss", float(avg_loss), episode)
        writer.add_scalar("train/epsilon", float(agent.epsilon), episode)
        writer.add_scalar("train/mem_MB", float(mem_used), episode)
        writer.add_scalar("train/elapsed_hours", float(elapsed_time / 3600), episode)
        writer.add_scalar("train/hits", int(hits_sum), episode)
        writer.flush()

        print(
            f"Episode {episode} | "
            f"Avg reward: {avg_reward:.3f} | "
            f"100-episode avg: {avg_reward_100:.3f} | "
            f"Reward std: {reward_std_100:.3f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Mem: {mem_used:.1f} MB | "
            f"Time: {elapsed_time / 3600:.2f} hrs"
            f"Writing to TensorBoard:", episode

        )


        # -------------------------------
        # -------- Evaluation -----------
        # -------------------------------
        if episode % 10 == 0:
            eval_env = PaiaPongEnv()
            eval_reward = evaluate_agent(
                agent, eval_env, n_episodes=EVAL_EPISODES
            )
            with open(CSV_PATH, "a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    episode,
                    avg_reward,
                    avg_reward_100,
                    avg_loss,
                    agent.epsilon,
                    hits_sum,
                    mem_used,
                    elapsed_time / 3600
                ])

            print(
                f"--- Evaluation at Episode {episode}: "
                f"Avg reward = {eval_reward:.3f} ---"
            )


        print(
            f"Episode {episode} | "
            f"Avg reward: {total_rewards.mean():.3f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Loss: {loss:.4f} | "
            f"Hits: {hits_sum}"
        )

        # -------------------------------
        # -------- Save models ----------
        # -------------------------------
        if episode % 10 == 0:
            model_path = os.path.join(
                MODEL_DIR, f"dqn_episode_{episode}.pth"
            )
            torch.save(agent.online_net.state_dict(), model_path)

            buffer_path = os.path.join(
                MODEL_DIR, f"replay_buffer_episode_{episode}.pkl"
            )
            if len(agent.memory) > 0:
                with open(buffer_path, "wb") as f:
                    pickle.dump(agent.memory, f)
                print(f"💾 Replay buffer saved, size={len(agent.memory)}")
            else:
                print("⚠️ Replay buffer empty, not saved")

            print(f"💾 Model saved at {model_path}")
            print(f"💾 Replay buffer saved at {buffer_path}")

    # -------------------------------
    # ---- Save latest model --------
    # -------------------------------
    LATEST_MODEL = os.path.join(MODEL_DIR, "dqn_latest.pth")
    LATEST_BUFFER = os.path.join(MODEL_DIR, "replay_buffer_latest.pkl")

    torch.save(agent.online_net.state_dict(), LATEST_MODEL)
    with open(LATEST_BUFFER, "wb") as f:
        pickle.dump(agent.memory, f)

    print("✅ Latest model & replay buffer saved")

    # ===============================
    # ===== Visual Test (Render) =====
    # ===============================
    print("🎮 Start visual test")

    test_env = PaiaPongEnv()
    state, _ = test_env.reset()
    done = False

    agent.epsilon = 0.0  # 關掉探索（重要）

    while not done:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            action = torch.argmax(agent.online_net(state_tensor)).item()

        state, reward, terminated, truncated, _ = test_env.step(action)
        test_env.render()

        done = terminated or truncated

    test_env.close()

    # ----------------- 訓練結束後只存最新 -----------------
    final_model_path = os.path.join(MODEL_DIR, "dqn_latest.pth")
    torch.save(agent.online_net.state_dict(), final_model_path)
    final_buffer_path = os.path.join(MODEL_DIR, "replay_buffer_latest.pkl")
    with open(final_buffer_path, "wb") as f:
        pickle.dump(agent.memory, f)
    print(f"Final Model saved at {final_model_path}")
    print(f"Final Replay buffer saved at {final_buffer_path}")
    print(f"📦 ReplayBuffer size now = {len(agent.memory)}")
    print(f"📦 ReplayBuffer size after episode {episode}: {len(agent.memory)}")

#http://localhost:6006/
#tensorboard --logdir=runs
