"""
Dynamic Pricing RL — Training Script
Run: python train.py
Outputs: results/results.pkl  +  models/  directory
"""

import os, time, pickle, warnings, csv
import numpy as np
import torch
warnings.filterwarnings("ignore")

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env import (DynamicPricingEnv,
                 ALPHA, BETA, SIGMA, T, I0, P_MIN, P_MAX, SALVAGE,
                 HOLDING_COST, ACTION_DELTAS)

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_DEFAULT  = 0.5
LAMBDA_ABLATION = [0.0, 0.1, 0.5, 1.0]
EVAL_BETAS      = [1.0, 2.0, 3.5]
N_SEEDS         = 3
EVAL_EPISODES   = 100
MA_WINDOW       = 50

# Rule-based
RB_DELTA  = 2.0
RB_THRESH = ALPHA / 2.0          # 50 — expected demand at optimal static price

# Q-Learning
QL_EPISODES          = 3000
QL_ABLATION_EPISODES = 1000
QL_LR                = 0.1
QL_GAMMA             = 0.95
QL_EPS_START         = 1.0
QL_EPS_END           = 0.05
QL_EPS_DECAY         = 500       # episodes over which epsilon decays linearly
QL_PRICE_BINS        = 5
QL_DEMAND_BINS       = 4
QL_INV_BINS          = 3

# DQN (SB3)
DQN_TIMESTEPS          = 150_000
DQN_ABLATION_TIMESTEPS = 50_000
DQN_LR                 = 1e-3
DQN_BUFFER             = 10_000
DQN_BATCH              = 64
DQN_GAMMA              = 0.95
DQN_TARGET_UPD         = 500
DQN_EPS_FRAC           = 0.3
DQN_EPS_END            = 0.05

# PPO (SB3)
PPO_TIMESTEPS          = 2_000_000 #300_000
PPO_ABLATION_TIMESTEPS = 75_000
PPO_LR                 = 3e-4
PPO_N_ENVS             = 4
PPO_N_STEPS            = 256
PPO_BATCH              = 64
PPO_GAMMA              = 0.95
PPO_GAE_LAM            = 0.95
PPO_ENT_COEF           = 0.01
PPO_N_EPOCHS           = 5 #10

# ── Run flags — set False to skip that section ────────────────────────────────
RUN_ABLATION   = False   # set True once main results look good
RUN_ROBUSTNESS = False   # set True once main results look good

CONFIG = dict(
    ALPHA=ALPHA, BETA=BETA, SIGMA=SIGMA, T=T, I0=I0,
    P_MIN=P_MIN, P_MAX=P_MAX, SALVAGE=SALVAGE, HOLDING_COST=HOLDING_COST,
    LAMBDA_DEFAULT=LAMBDA_DEFAULT, LAMBDA_ABLATION=LAMBDA_ABLATION,
    EVAL_BETAS=EVAL_BETAS, N_SEEDS=N_SEEDS, EVAL_EPISODES=EVAL_EPISODES,
    MA_WINDOW=MA_WINDOW, QL_EPISODES=QL_EPISODES,
    DQN_TIMESTEPS=DQN_TIMESTEPS, PPO_TIMESTEPS=PPO_TIMESTEPS,
    PPO_N_ENVS=PPO_N_ENVS, DEVICE=DEVICE,
)

# ── Q-Learning discretisation ─────────────────────────────────────────────────
_P_EDGES = np.linspace(P_MIN, P_MAX, QL_PRICE_BINS  + 1)
_D_EDGES = np.linspace(0,     ALPHA,  QL_DEMAND_BINS + 1)
_I_EDGES = np.linspace(0,     I0,     QL_INV_BINS    + 1)

def _disc(price, demand, inventory):
    p = int(np.clip(np.digitize(price,     _P_EDGES) - 1, 0, QL_PRICE_BINS  - 1))
    d = int(np.clip(np.digitize(demand,    _D_EDGES) - 1, 0, QL_DEMAND_BINS - 1))
    i = int(np.clip(np.digitize(inventory, _I_EDGES) - 1, 0, QL_INV_BINS    - 1))
    return p, d, i

# ── SB3 callback: logs per-episode reward & revenue ──────────────────────────
class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards  = []
        self.ep_revenues = []
        self._cur_r   = None
        self._cur_rev = None

    def _on_step(self):
        n = len(self.locals["rewards"])
        if self._cur_r is None:
            self._cur_r   = np.zeros(n)
            self._cur_rev = np.zeros(n)
        for i in range(n):
            self._cur_r[i]   += self.locals["rewards"][i]
            self._cur_rev[i] += self.locals["infos"][i].get("revenue", 0.0)
            if self.locals["dones"][i]:
                self.ep_rewards.append(float(self._cur_r[i]))
                self.ep_revenues.append(float(self._cur_rev[i]))
                self._cur_r[i] = self._cur_rev[i] = 0.0
        return True

# ── Shared evaluation utilities ───────────────────────────────────────────────
def _eval_episodes(agent_fn, beta=BETA, lam=LAMBDA_DEFAULT,
                   n_ep=EVAL_EPISODES, seed=0):
    """Run n_ep episodes with a frozen agent. Returns (rewards, revenues, inventories) lists."""
    rewards, revenues, inventories = [], [], []
    for ep in range(n_ep):
        env = DynamicPricingEnv(beta=beta, lam=lam)
        obs, _ = env.reset(seed=seed + ep)
        ep_r = ep_rev = 0.0
        done = False
        while not done:
            action = agent_fn(obs, env)
            obs, r, done, _, info = env.step(action)
            ep_r   += r
            ep_rev += info["revenue"]
        rewards.append(ep_r)
        revenues.append(ep_rev)
        inventories.append(env.inventory)
    return rewards, revenues, inventories

def _collect_trajectory(agent_fn, beta=BETA, lam=LAMBDA_DEFAULT, seed=42):
    """Run one episode and return full trajectory for plotting."""
    env = DynamicPricingEnv(beta=beta, lam=lam)
    obs, _ = env.reset(seed=seed)
    prices, demands, inventories, revenues = [], [], [], []
    done = False
    while not done:
        action = agent_fn(obs, env)
        obs, _, done, _, info = env.step(action)
        prices.append(info["price"])
        demands.append(info["demand"])
        inventories.append(info["inventory"])
        revenues.append(info["revenue"])
    return {"prices": prices, "demands": demands,
            "inventories": inventories, "revenues": revenues}

def convergence_ep(reward_history, window=MA_WINDOW, threshold=0.90):
    """Episodes to reach threshold × peak of moving-average reward."""
    if len(reward_history) < window:
        return len(reward_history)
    ma = np.convolve(reward_history, np.ones(window) / window, mode="valid")
    peak = ma.max()
    hits = np.where(ma >= threshold * peak)[0]
    return int(hits[0] + window) if len(hits) else len(reward_history)

# ── Baseline agent functions ──────────────────────────────────────────────────
def _static_fn(obs, env):
    p_star     = float(np.clip(ALPHA / (2.0 * env.beta), P_MIN, P_MAX))
    candidates = [env.price * (1.0 + d) for d in ACTION_DELTAS]
    return int(np.argmin([abs(c - p_star) for c in candidates]))

def _rule_fn(obs, env):
    last_d     = env.d_hist[-1]
    target     = float(np.clip(
        env.price + RB_DELTA * np.sign(last_d - RB_THRESH), P_MIN, P_MAX
    ))
    candidates = [env.price * (1.0 + d) for d in ACTION_DELTAS]
    return int(np.argmin([abs(c - target) for c in candidates]))

# ── Q-Learning ────────────────────────────────────────────────────────────────
def train_ql(lam=LAMBDA_DEFAULT, seed=0, episodes=QL_EPISODES):
    np.random.seed(seed)
    Q = np.zeros((QL_PRICE_BINS, QL_DEMAND_BINS, QL_INV_BINS, len(ACTION_DELTAS)))
    ep_rewards, ep_revenues, epsilons = [], [], []
    env = DynamicPricingEnv(lam=lam)

    for ep in range(episodes):
        eps = max(QL_EPS_END,
                  QL_EPS_START - (QL_EPS_START - QL_EPS_END) * ep / QL_EPS_DECAY)
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        ep_r = ep_rev = 0.0
        done = False
        while not done:
            price, demand, inv = env.price, env.d_hist[-1], env.inventory
            s = _disc(price, demand, inv)
            action = (np.random.randint(len(ACTION_DELTAS))
                      if np.random.random() < eps
                      else int(np.argmax(Q[s])))
            obs, r, done, _, info = env.step(action)
            s2 = _disc(info["price"], info["demand"], info["inventory"])
            Q[s][action] += QL_LR * (r + QL_GAMMA * Q[s2].max() - Q[s][action])
            ep_r   += r
            ep_rev += info["revenue"]
        ep_rewards.append(ep_r)
        ep_revenues.append(ep_rev)
        epsilons.append(eps)

    return Q, ep_rewards, ep_revenues, epsilons

def eval_ql(Q, beta=BETA, lam=LAMBDA_DEFAULT, n_ep=EVAL_EPISODES, seed=0):
    def agent_fn(obs, env):
        s = _disc(env.price, env.d_hist[-1], env.inventory)
        return int(np.argmax(Q[s]))
    rew, rev, inv = _eval_episodes(agent_fn, beta=beta, lam=lam, n_ep=n_ep, seed=seed)
    return rew, rev, inv

def ql_trajectory(Q, beta=BETA, lam=LAMBDA_DEFAULT, seed=42):
    def agent_fn(obs, env):
        s = _disc(env.price, env.d_hist[-1], env.inventory)
        return int(np.argmax(Q[s]))
    return _collect_trajectory(agent_fn, beta=beta, lam=lam, seed=seed)

# ── DQN (SB3) ─────────────────────────────────────────────────────────────────
def train_dqn(lam=LAMBDA_DEFAULT, seed=0, timesteps=DQN_TIMESTEPS):
    env = DynamicPricingEnv(lam=lam)
    cb  = EpisodeLogger()
    model = DQN(
        "MlpPolicy", env,
        learning_rate          = DQN_LR,
        buffer_size            = DQN_BUFFER,
        batch_size             = DQN_BATCH,
        gamma                  = DQN_GAMMA,
        target_update_interval = DQN_TARGET_UPD,
        exploration_fraction   = DQN_EPS_FRAC,
        exploration_final_eps  = DQN_EPS_END,
        policy_kwargs          = dict(net_arch=[64, 64]),
        device                 = DEVICE,
        seed                   = seed,
        verbose                = 0,
    )
    model.learn(total_timesteps=timesteps, callback=cb)
    return model, cb.ep_rewards, cb.ep_revenues

def eval_dqn(model, beta=BETA, lam=LAMBDA_DEFAULT, n_ep=EVAL_EPISODES, seed=0):
    def agent_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    rew, rev, inv = _eval_episodes(agent_fn, beta=beta, lam=lam, n_ep=n_ep, seed=seed)
    return rew, rev, inv

def dqn_trajectory(model, beta=BETA, lam=LAMBDA_DEFAULT, seed=42):
    def agent_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _collect_trajectory(agent_fn, beta=beta, lam=lam, seed=seed)

# ── PPO (SB3) ─────────────────────────────────────────────────────────────────
def _make_env(lam):
    def _init():
        return DynamicPricingEnv(lam=lam)
    return _init

def train_ppo(lam=LAMBDA_DEFAULT, seed=0, timesteps=PPO_TIMESTEPS):
    vec_env = DummyVecEnv([_make_env(lam) for _ in range(PPO_N_ENVS)])
    cb      = EpisodeLogger()
    model   = PPO(
        "MlpPolicy", vec_env,
        learning_rate = PPO_LR,
        n_steps       = PPO_N_STEPS,
        batch_size    = PPO_BATCH,
        gamma         = PPO_GAMMA,
        gae_lambda    = PPO_GAE_LAM,
        ent_coef      = PPO_ENT_COEF,
        n_epochs      = PPO_N_EPOCHS,
        policy_kwargs = dict(net_arch=[64, 64]),
        device        = DEVICE,
        seed          = seed,
        verbose       = 0,
    )
    model.learn(total_timesteps=timesteps, callback=cb)
    return model, cb.ep_rewards, cb.ep_revenues

def eval_ppo(model, beta=BETA, lam=LAMBDA_DEFAULT, n_ep=EVAL_EPISODES, seed=0):
    def agent_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    rew, rev, inv = _eval_episodes(agent_fn, beta=beta, lam=lam, n_ep=n_ep, seed=seed)
    return rew, rev, inv

def ppo_trajectory(model, beta=BETA, lam=LAMBDA_DEFAULT, seed=42):
    def agent_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _collect_trajectory(agent_fn, beta=beta, lam=lam, seed=seed)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print(f"Device : {DEVICE}")
    print(f"Config : T={T}, I0={I0}, λ={LAMBDA_DEFAULT}, seeds={N_SEEDS}")
    print(f"Flags  : RUN_ABLATION={RUN_ABLATION}  RUN_ROBUSTNESS={RUN_ROBUSTNESS}\n")

    # Load existing results if available (resume support)
    _pkl = "results/results.pkl"
    if os.path.exists(_pkl):
        with open(_pkl, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {_pkl}")
        print(f"  Completed sections: { {k for k in results['eval']} }\n")
    else:
        results = {
            "config":      CONFIG,
            "training":    {},
            "eval":        {},
            "trajectories":{},
            "ablation":    {},
            "robustness":  {},
        }

    def _save_csv():
        rows = []
        for agent in ("static", "rule_based", "ql", "dqn", "ppo"):
            if agent not in results["eval"]:
                continue
            ev  = results["eval"][agent]
            rev = ev.get("revenues",    [])
            rew = ev.get("rewards",     [])
            inv = ev.get("inventories", [])
            rows.append({
                "agent":         agent,
                "mean_revenue":  round(float(np.mean(rev)), 2) if rev else "",
                "std_revenue":   round(float(np.std(rev)),  2) if rev else "",
                "mean_reward":   round(float(np.mean(rew)), 2) if rew else "",
                "std_reward":    round(float(np.std(rew)),  2) if rew else "",
                "mean_inv_left": round(float(np.mean(inv)), 2) if inv else "",
                "std_inv_left":  round(float(np.std(inv)),  2) if inv else "",
                "conv_ep":       ev.get("conv_ep", ""),
            })
        if not rows:
            return
        csv_path = "results/results_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  --> results/results_summary.csv saved")

    def _save():
        with open(_pkl, "wb") as f:
            pickle.dump(results, f)
        print("  --> results/results.pkl saved")
        _save_csv()

    def _done(section):
        """Return True if this section already has results."""
        if section in ("static", "rule_based"):
            return section in results["eval"]
        if section in ("ql", "dqn", "ppo"):
            return section in results["training"]
        if section == "ablation":
            return bool(results.get("ablation"))
        if section == "robustness":
            return bool(results.get("robustness"))
        return False

    # ── [1/6] Baselines ───────────────────────────────────────────────────────
    if _done("static"):
        print("[1/6] Baselines... SKIPPED")
        print(f"  Static     : rev={np.mean(results['eval']['static']['revenues']):.1f}"
              f"  inv={np.mean(results['eval']['static']['inventories']):.1f}")
        print(f"  Rule-Based : rev={np.mean(results['eval']['rule_based']['revenues']):.1f}"
              f"  inv={np.mean(results['eval']['rule_based']['inventories']):.1f}")
    else:
        print("[1/6] Baselines...")
        s_rew, s_rev, s_inv, rb_rew, rb_rev, rb_inv = [], [], [], [], [], []
        for s in range(N_SEEDS):
            r, rev, inv = _eval_episodes(_static_fn, seed=s * 100)
            s_rew += r;  s_rev += rev;  s_inv += inv
            r, rev, inv = _eval_episodes(_rule_fn,   seed=s * 100)
            rb_rew += r; rb_rev += rev; rb_inv += inv
        results["eval"]["static"]     = {"rewards": s_rew,  "revenues": s_rev,  "inventories": s_inv}
        results["eval"]["rule_based"] = {"rewards": rb_rew, "revenues": rb_rev, "inventories": rb_inv}
        results["trajectories"]["static"]      = _collect_trajectory(_static_fn)
        results["trajectories"]["rule_based"]  = _collect_trajectory(_rule_fn)
        print(f"  Static     : rev={np.mean(s_rev):.1f} ± {np.std(s_rev):.1f}  inv_left={np.mean(s_inv):.1f}")
        print(f"  Rule-Based : rev={np.mean(rb_rev):.1f} ± {np.std(rb_rev):.1f}  inv_left={np.mean(rb_inv):.1f}")
        _save()

    # ── [2/6] Q-Learning ─────────────────────────────────────────────────────
    if _done("ql"):
        print("\n[2/6] Q-Learning... SKIPPED")
        print(f"  eval_rev={np.mean(results['eval']['ql']['revenues']):.1f}"
              f"  inv_left={np.mean(results['eval']['ql']['inventories']):.1f}")
        best_ql = np.load("models/ql_seed0.npy")
    else:
        print("\n[2/6] Q-Learning...")
        ql_train_rew, ql_train_rev, ql_epsilons = [], [], None
        ql_eval_rew,  ql_eval_rev, ql_eval_inv  = [], [], []
        best_ql = None
        for s in range(N_SEEDS):
            Q, ep_r, ep_rev, eps = train_ql(seed=s)
            np.save(f"models/ql_seed{s}.npy", Q)
            ql_train_rew.append(ep_r)
            ql_train_rev.append(ep_rev)
            if ql_epsilons is None:
                ql_epsilons = eps
            r, rev, inv = eval_ql(Q, seed=s * 100)
            ql_eval_rew += r;  ql_eval_rev += rev;  ql_eval_inv += inv
            if best_ql is None:
                best_ql = Q
            print(f"  Seed {s}: last-100 train_rev={np.mean(ep_rev[-100:]):.1f}"
                  f"  eval_rev={np.mean(rev):.1f}  inv_left={np.mean(inv):.1f}")
        results["training"]["ql"] = {
            "rewards":  ql_train_rew,
            "revenues": ql_train_rev,
            "epsilons": ql_epsilons,
        }
        results["eval"]["ql"] = {
            "rewards":     ql_eval_rew,
            "revenues":    ql_eval_rev,
            "inventories": ql_eval_inv,
            "conv_ep":     convergence_ep(ql_train_rew[0]),
        }
        results["trajectories"]["ql"] = ql_trajectory(best_ql)
        _save()

    # ── [3/6] DQN ────────────────────────────────────────────────────────────
    if _done("dqn"):
        print("\n[3/6] DQN... SKIPPED")
        print(f"  eval_rev={np.mean(results['eval']['dqn']['revenues']):.1f}"
              f"  inv_left={np.mean(results['eval']['dqn']['inventories']):.1f}")
        best_dqn = DQN.load("models/dqn_seed0")
    else:
        print("\n[3/6] DQN...")
        dqn_train_rew, dqn_train_rev = [], []
        dqn_eval_rew,  dqn_eval_rev, dqn_eval_inv = [], [], []
        best_dqn = None
        for s in range(N_SEEDS):
            model, ep_r, ep_rev = train_dqn(seed=s)
            model.save(f"models/dqn_seed{s}")
            dqn_train_rew.append(ep_r)
            dqn_train_rev.append(ep_rev)
            r, rev, inv = eval_dqn(model, seed=s * 100)
            dqn_eval_rew += r;  dqn_eval_rev += rev;  dqn_eval_inv += inv
            if best_dqn is None:
                best_dqn = model
            print(f"  Seed {s}: last-50 train_rev={np.mean(ep_rev[-50:]):.1f}"
                  f"  eval_rev={np.mean(rev):.1f}  inv_left={np.mean(inv):.1f}")
        results["training"]["dqn"] = {"rewards": dqn_train_rew, "revenues": dqn_train_rev}
        results["eval"]["dqn"] = {
            "rewards":     dqn_eval_rew,
            "revenues":    dqn_eval_rev,
            "inventories": dqn_eval_inv,
            "conv_ep":     convergence_ep(dqn_train_rew[0]),
        }
        results["trajectories"]["dqn"] = dqn_trajectory(best_dqn)
        _save()

    # ── [4/6] PPO ────────────────────────────────────────────────────────────
    if _done("ppo"):
        print("\n[4/6] PPO... SKIPPED")
        print(f"  eval_rev={np.mean(results['eval']['ppo']['revenues']):.1f}"
              f"  inv_left={np.mean(results['eval']['ppo']['inventories']):.1f}")
        best_ppo = PPO.load("models/ppo_seed0")
    else:
        print("\n[4/6] PPO...")
        ppo_train_rew, ppo_train_rev = [], []
        ppo_eval_rew,  ppo_eval_rev, ppo_eval_inv = [], [], []
        best_ppo = None
        for s in range(N_SEEDS):
            model, ep_r, ep_rev = train_ppo(seed=s)
            model.save(f"models/ppo_seed{s}")
            ppo_train_rew.append(ep_r)
            ppo_train_rev.append(ep_rev)
            r, rev, inv = eval_ppo(model, seed=s * 100)
            ppo_eval_rew += r;  ppo_eval_rev += rev;  ppo_eval_inv += inv
            if best_ppo is None:
                best_ppo = model
            print(f"  Seed {s}: last-50 train_rev={np.mean(ep_rev[-50:]):.1f}"
                  f"  eval_rev={np.mean(rev):.1f}  inv_left={np.mean(inv):.1f}")
        results["training"]["ppo"] = {"rewards": ppo_train_rew, "revenues": ppo_train_rev}
        results["eval"]["ppo"] = {
            "rewards":     ppo_eval_rew,
            "revenues":    ppo_eval_rev,
            "inventories": ppo_eval_inv,
            "conv_ep":     convergence_ep(ppo_train_rew[0]),
        }
        results["trajectories"]["ppo"] = ppo_trajectory(best_ppo)
        _save()

    # ── [5/6] Lambda ablation ─────────────────────────────────────────────────
    if not RUN_ABLATION:
        print("\n[5/6] Lambda ablation... SKIPPED (RUN_ABLATION=False)")
    elif _done("ablation"):
        print("\n[5/6] Lambda ablation... SKIPPED (already in results)")
    else:
        print("\n[5/6] Lambda ablation...")
        for lam in LAMBDA_ABLATION:
            results["ablation"][lam] = {}
            # Reuse main training results for the default lambda — no retraining needed
            if lam == LAMBDA_DEFAULT:
                print(f"  λ={lam}  (reusing main results)")
                for agent in ("ql", "dqn", "ppo"):
                    results["ablation"][lam][agent] = {
                        "revenues":    results["eval"][agent]["revenues"],
                        "inventories": results["eval"][agent]["inventories"],
                    }
                _save()
                continue
            print(f"  λ={lam}")
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                Q, _, _, _ = train_ql(lam=lam, seed=s, episodes=QL_ABLATION_EPISODES)
                _, rev, inv = eval_ql(Q, lam=lam, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["ablation"][lam]["ql"] = {"revenues": rev_all, "inventories": inv_all}
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                model, _, _ = train_dqn(lam=lam, seed=s, timesteps=DQN_ABLATION_TIMESTEPS)
                _, rev, inv = eval_dqn(model, lam=lam, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["ablation"][lam]["dqn"] = {"revenues": rev_all, "inventories": inv_all}
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                model, _, _ = train_ppo(lam=lam, seed=s, timesteps=PPO_ABLATION_TIMESTEPS)
                _, rev, inv = eval_ppo(model, lam=lam, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["ablation"][lam]["ppo"] = {"revenues": rev_all, "inventories": inv_all}
            _save()

    # ── [6/6] Robustness sweep (no retraining) ────────────────────────────────
    if not RUN_ROBUSTNESS:
        print("\n[6/6] Robustness sweep... SKIPPED (RUN_ROBUSTNESS=False)")
    elif _done("robustness"):
        print("\n[6/6] Robustness sweep... SKIPPED (already in results)")
    else:
        print("\n[6/6] Robustness sweep (frozen policies)...")
        ql_models  = [np.load(f"models/ql_seed{s}.npy") for s in range(N_SEEDS)]
        dqn_models = [DQN.load(f"models/dqn_seed{s}")   for s in range(N_SEEDS)]
        ppo_models = [PPO.load(f"models/ppo_seed{s}")   for s in range(N_SEEDS)]
        for beta in EVAL_BETAS:
            results["robustness"][beta] = {}
            # Reuse main eval results for the training beta — no re-evaluation needed
            if beta == BETA:
                print(f"  β={beta}  (reusing main results)")
                for agent in ("static", "rule_based", "ql", "dqn", "ppo"):
                    results["robustness"][beta][agent] = {
                        "revenues":    results["eval"][agent]["revenues"],
                        "inventories": results["eval"][agent]["inventories"],
                    }
                continue
            print(f"  β={beta}")
            for tag, fn in [("static", _static_fn), ("rule_based", _rule_fn)]:
                rev_all, inv_all = [], []
                for s in range(N_SEEDS):
                    _, rev, inv = _eval_episodes(fn, beta=beta, seed=s * 100)
                    rev_all += rev;  inv_all += inv
                results["robustness"][beta][tag] = {"revenues": rev_all, "inventories": inv_all}
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                _, rev, inv = eval_ql(ql_models[s], beta=beta, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["robustness"][beta]["ql"] = {"revenues": rev_all, "inventories": inv_all}
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                _, rev, inv = eval_dqn(dqn_models[s], beta=beta, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["robustness"][beta]["dqn"] = {"revenues": rev_all, "inventories": inv_all}
            rev_all, inv_all = [], []
            for s in range(N_SEEDS):
                _, rev, inv = eval_ppo(ppo_models[s], beta=beta, seed=s * 100)
                rev_all += rev;  inv_all += inv
            results["robustness"][beta]["ppo"] = {"revenues": rev_all, "inventories": inv_all}
        _save()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min")
    print("Saved: results/results.pkl  |  models/")
