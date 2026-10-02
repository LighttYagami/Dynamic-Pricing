import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Shared constants — imported by train.py and analysis.py
ALPHA         = 100.0
BETA          = 2.0
SIGMA         = 5.0
T             = 6 #12
I0            = 100.0 #450.0 #600.0
P_MIN         = 10.0
P_MAX         = 50.0
SALVAGE       = 5.0 #2.0
HOLDING_COST  = 0.1          # per unit of inventory held per step
ACTION_DELTAS = [-0.10, -0.05, 0.0, 0.05, 0.10]


class DynamicPricingEnv(gym.Env):
    """
    Single-product dynamic pricing over T steps.
    State  : 7D — [price, demand(t-1..t-3), inventory, time, demand-change]
    Actions: 5 discrete price adjustments {-10%,-5%,0%,+5%,+10%}
    Reward : p*d_sold + lambda*delta_d - HOLDING_COST*inventory  (terminal: also -SALVAGE*inventory)
    """
    metadata = {"render_modes": []}

    def __init__(self, beta=None, lam=0.5):
        super().__init__()
        self.beta = float(beta) if beta is not None else BETA
        self.lam  = float(lam)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _demand(self, price):
        return float(np.clip(
            ALPHA - self.beta * price + np.random.normal(0, SIGMA), 0.0, None
        ))

    def _obs(self):
        return np.array([
            self.price          / P_MAX,
            self.d_hist[0]      / ALPHA,
            self.d_hist[1]      / ALPHA,
            self.d_hist[2]      / ALPHA,
            self.inventory      / I0,
            self.t              / T,
            (self.d_hist[2] - self.d_hist[1]) / ALPHA,
        ], dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.price     = (P_MIN + P_MAX) / 2.0   # start at midpoint = 30
        self.inventory = float(I0)
        self.d_hist    = [0.0, 0.0, 0.0]
        self.t         = 0
        return self._obs(), {}

    def step(self, action):
        self.price = float(np.clip(
            self.price * (1.0 + ACTION_DELTAS[action]), P_MIN, P_MAX
        ))
        d_raw      = self._demand(self.price)
        d_sold     = float(min(d_raw, self.inventory))
        self.inventory = max(0.0, self.inventory - d_sold)
        delta_d    = d_sold - self.d_hist[-1]
        revenue    = self.price * d_sold
        reward     = revenue + self.lam * delta_d - HOLDING_COST * self.inventory
        self.d_hist = [self.d_hist[1], self.d_hist[2], d_sold]
        self.t     += 1
        done        = (self.t >= T)
        if done:
            reward -= SALVAGE * self.inventory
        info = {
            "price":     self.price,
            "demand":    d_sold,
            "inventory": self.inventory,
            "revenue":   revenue,
        }
        return self._obs(), reward, done, False, info
