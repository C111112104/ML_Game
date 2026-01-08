import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


class PaiaPongEnv(gym.Env):
    """
    PAIA Pong with Spin Mechanism
    Action: 0 stay, 1 left, 2 right
    Observation: (6,)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode

        # screen
        self.W = 200
        self.H = 500

        # sizes (from README)
        self.BALL_R = 5          # 10x10
        self.PADDLE_W = 40
        self.PADDLE_H = 10

        # speed
        self.PADDLE_SPEED = 5
        self.AI_SPEED = 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.reset()

    # -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ball
        self.ball_x = self.W // 2
        self.ball_y = self.H // 2
        v = np.random.choice([7, 10])
        self.ball_vx = np.random.choice([-v, v])
        self.ball_vy = np.random.choice([-v, v])

        # paddles
        self.p1_x = 80
        self.p1_y = 420
        self.p2_x = 80
        self.p2_y = 70

        self.p1_dx = 0
        self.p2_dx = 0

        return self._get_obs(), {}

    # -------------------------------------------------
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # ===== bottom paddle (agent) =====
        old_x = self.p1_x
        if action == 1:
            self.p1_x -= self.PADDLE_SPEED
        elif action == 2:
            self.p1_x += self.PADDLE_SPEED

        self.p1_x = np.clip(self.p1_x, 0, self.W - self.PADDLE_W)
        self.p1_dx = self.p1_x - old_x

        # ===== top paddle AI =====
        old_x = self.p2_x
        center = self.p2_x + self.PADDLE_W / 2
        if self.ball_x > center:
            self.p2_x += self.AI_SPEED
        elif self.ball_x < center:
            self.p2_x -= self.AI_SPEED

        self.p2_x = np.clip(self.p2_x, 0, self.W - self.PADDLE_W)
        self.p2_dx = self.p2_x - old_x

        # ===== move ball =====
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # wall bounce
        if self.ball_x <= 0 or self.ball_x >= self.W:
            self.ball_vx *= -1

        # ===== paddle collision =====
        if self._hit(self.p1_x, self.p1_y):
            self._apply_spin(self.p1_dx)
            self.ball_vy = -abs(self.ball_vy)
            reward += 0.1

        if self._hit(self.p2_x, self.p2_y):
            self._apply_spin(self.p2_dx)
            self.ball_vy = abs(self.ball_vy)

        # ===== scoring =====
        if self.ball_y <= 0:
            reward = +1.0
            terminated = True

        if self.ball_y >= self.H:
            reward = -1.0
            terminated = True

        if self.render_mode:
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    # -------------------------------------------------
    def _apply_spin(self, paddle_dx):
        """切球機制（依 README）"""
        if paddle_dx == 0:
            return

        if np.sign(paddle_dx) == np.sign(self.ball_vx):
            self.ball_vx += 3 * np.sign(self.ball_vx)
        else:
            self.ball_vx *= -1

    # -------------------------------------------------
    def _hit(self, px, py):
        return (
            px <= self.ball_x <= px + self.PADDLE_W and
            py <= self.ball_y <= py + self.PADDLE_H
        )

    # -------------------------------------------------
    def _get_obs(self):
        return np.array([
            self.ball_x / self.W,
            self.ball_y / self.H,
            self.ball_vx / 15,
            self.ball_vy / 15,
            self.p1_x / self.W,
            self.p2_x / self.W,
        ], dtype=np.float32)

    # -------------------------------------------------
    def render(self):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # ball
        cv2.circle(
            img,
            (int(self.ball_x), int(self.ball_y)),
            self.BALL_R,
            (255, 255, 255),
            -1
        )

        # paddles
        cv2.rectangle(
            img,
            (int(self.p1_x), int(self.p1_y)),
            (int(self.p1_x + self.PADDLE_W), int(self.p1_y + self.PADDLE_H)),
            (0, 0, 255),
            -1
        )

        cv2.rectangle(
            img,
            (int(self.p2_x), int(self.p2_y)),
            (int(self.p2_x + self.PADDLE_W), int(self.p2_y + self.PADDLE_H)),
            (255, 0, 0),
            -1
        )

        cv2.imshow("PAIA Pong", img)
        cv2.waitKey(16)
