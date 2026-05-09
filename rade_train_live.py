"""
rade_train_live.py
--------------
Actor-Critic training with MABSE belief state.

Replaces rade_train.py. No LLM dependency.
Uses real Bitcoin data from PostgreSQL via MABSE.

Key differences from rade_train.py:
  - State: 9D (adds b̂^s, b̂^m, b̂^l) vs 6/7D
  - Belief: Multi-scale temporal vs single-scale cosine
  - Threshold: KDE adaptive vs fixed (100MB, 1100s, 15000)
  - LLM: removed entirely
  - Reward: 4-term (no LLM shaping term)

Usage:
    # With live data (run btc_live_runner.py first for ≥1h)
    python rade_train_live.py --steps 2000 --use-live

    # Synthetic fallback (no DB needed)
    python rade_train_live.py --steps 2000 --no-live

    # Full ablation
    python rade_train_live.py --steps 2000 --ablation-type mabse_full --use-live
    python rade_train_live.py --steps 2000 --ablation-type ac_base    --no-live
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from btc_live_collector import LiveBitcoinCollector, LiveNetworkSnapshot
from rade_belief import MABSE, MABSEState

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ACTIONS     = ["probe", "skip", "escalate"]
ACTION_COST = {"probe": 0.08, "skip": 0.01, "escalate": 0.12}
ACTION_ENC  = {"probe": 1.0,  "skip": 0.0,  "escalate": -1.0}

STATE_DIM_MABSE = 9   # full MABSE state
STATE_DIM_BASE  = 6   # ablation without beliefs


# ──────────────────────────────────────────────
# Running state normaliser (Welford)
# ──────────────────────────────────────────────

class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.n    = 0
        self.mean = torch.zeros(dim)
        self.m2   = torch.zeros(dim)
        self.eps  = eps

    def update(self, x: torch.Tensor):
        self.n += 1
        d  = x - self.mean
        self.mean += d / self.n
        self.m2   += d * (x - self.mean)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.n < 2:
            return x
        var = self.m2 / max(self.n - 1, 1)
        return (x - self.mean) / torch.sqrt(var + self.eps)


# ──────────────────────────────────────────────
# Actor-Critic network
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.body   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, len(ACTIONS))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ──────────────────────────────────────────────
# Reward function (4-term, no LLM)
# ──────────────────────────────────────────────

def utility(anomalous: bool, action: str) -> float:
    """
    U(s_t, a_t):
        escalate + anomaly  → +1.0
        probe    + anomaly  → +0.5
        skip     + normal   → +0.4
        mismatch            → -0.3
    """
    if action == "escalate" and anomalous:  return 1.0
    if action == "probe"    and anomalous:  return 0.5
    if action == "skip"     and not anomalous: return 0.4
    return -0.3


def compute_reward(
    anomalous:     bool,
    action:        str,
    ensemble_belief: float,
    terminal_miss: bool,
    alpha:         float = 1.0,
    lambda_cost:   float = 0.15,
    beta:          float = 0.5,
    gamma_fn:      float = 0.3,
) -> float:
    """
    r_t = α·U - λ·c(a) - β·b̄·𝟙[skip] - γ_FN·𝟙[miss]

    Term 1: detection utility
    Term 2: probe cost penalty
    Term 3: RAG risk penalty (ensemble belief gated)
    Term 4: terminal false-negative penalty
    """
    u    = utility(anomalous, action)
    cost = ACTION_COST[action]
    rag  = beta * ensemble_belief if action == "skip" else 0.0
    fn   = gamma_fn if terminal_miss else 0.0
    return alpha * u - lambda_cost * cost - rag - fn


# ──────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────

def compute_gae(
    rewards:    List[float],
    values:     List[float],
    last_value: float,
    gamma:      float = 0.97,
    lam:        float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adv, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        v_next = last_value if t == len(rewards) - 1 else values[t + 1]
        delta  = rewards[t] + gamma * v_next - values[t]
        gae    = delta + gamma * lam * gae
        adv.append(gae)
    adv.reverse()
    advantages = torch.tensor(adv, dtype=torch.float32)
    returns    = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


# ──────────────────────────────────────────────
# Synthetic environment fallback
# ──────────────────────────────────────────────

class SyntheticBitcoinEnv:
    """
    Synthetic fallback when live data is unavailable.
    Parameters from calibrated distributions (btc_calibrate.py output).
    """
    import random as _random

    def __init__(
        self,
        seed:          int   = 42,
        mu_mempool:    float = 3.21,
        sigma_mempool: float = 0.87,
        lambda_inter:  float = 1 / 592,
    ):
        import random
        self.rng           = random.Random(seed)
        self.mu_mempool    = mu_mempool
        self.sigma_mempool = sigma_mempool
        self.lambda_inter  = lambda_inter

        self._m = self._sample_mempool()
        self._p = int(self.rng.uniform(300, 650_000))
        self._b = self._sample_inter()

    def _sample_mempool(self) -> float:
        import random
        log_val = self.rng.gauss(self.mu_mempool, self.sigma_mempool)
        return max(0.5, math.exp(log_val))

    def _sample_inter(self) -> float:
        return self.rng.expovariate(self.lambda_inter)

    def step(self) -> LiveNetworkSnapshot:
        self._m = self._m * 0.9 + self._sample_mempool() * 0.1
        self._p = max(100, int(self._p * 0.9 + self.rng.uniform(300, 650_000) * 0.1))
        self._b = self._b * 0.7 + self._sample_inter() * 0.3
        return LiveNetworkSnapshot(
            collected_at         = datetime.now(timezone.utc),
            mempool_size_mb      = round(self._m, 2),
            pending_tx           = self._p,
            inter_block_time_sec = round(self._b, 1),
            fee_rate_fast_sat_vb = round(self.rng.uniform(5, 80), 1),
            fee_rate_med_sat_vb  = round(self.rng.uniform(2, 40), 1),
            hashrate_eh_s        = round(self.rng.uniform(600, 800), 1),
            difficulty           = self.rng.uniform(8e13, 1e14),
            source               = "synthetic",
        )


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

@dataclass
class StepRecord:
    state:      torch.Tensor
    action_idx: int
    value:      float
    reward:     float


def parse_args():
    p = argparse.ArgumentParser(description="MABSE Actor-Critic training")
    p.add_argument("--steps",          type=int,   default=2000)
    p.add_argument("--rollout",        type=int,   default=64)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--lambda-cost",    type=float, default=0.15)
    p.add_argument("--alpha",          type=float, default=1.0)
    p.add_argument("--beta",           type=float, default=0.5)
    p.add_argument("--gamma-fn",       type=float, default=0.3)
    p.add_argument("--gamma",          type=float, default=0.97)
    p.add_argument("--gae-lambda",     type=float, default=0.95)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--c-v",            type=float, default=0.5)
    p.add_argument("--c-e",            type=float, default=0.01)
    p.add_argument("--reward-clip",    type=float, default=5.0)
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--use-live",       action="store_true",
                   help="Use real Bitcoin data from PostgreSQL via MABSE")
    p.add_argument("--no-live",        action="store_true",
                   help="Force synthetic environment")
    p.add_argument("--ablation-type",
                   choices=["mabse_full", "mabse_no_multiscale",
                             "mabse_no_kde", "ac_base"],
                   default="mabse_full")
    p.add_argument("--no-db",          action="store_true")
    p.add_argument("--policy-name",    type=str, default="mabse_a2c")
    p.add_argument("--notes",          type=str, default="")
    # MABSE params
    p.add_argument("--tau-short",      type=float, default=10.0)
    p.add_argument("--tau-mid",        type=float, default=60.0)
    p.add_argument("--tau-long",       type=float, default=1440.0)
    p.add_argument("--kde-bandwidth",  type=float, default=5.0)
    # Calibration params (from btc_calibrate.py output)
    p.add_argument("--mu-mempool",     type=float, default=3.21)
    p.add_argument("--sigma-mempool",  type=float, default=0.87)
    p.add_argument("--lambda-inter",   type=float, default=1/592)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Ablation: which components to use
    use_mabse     = args.ablation_type in ("mabse_full", "mabse_no_kde")
    use_kde       = args.ablation_type in ("mabse_full", "mabse_no_multiscale")
    use_multiscale = args.ablation_type == "mabse_full"
    state_dim     = STATE_DIM_MABSE if use_mabse else STATE_DIM_BASE

    # ── environment ────────────────────────────
    use_live = args.use_live and not args.no_live
    if use_live:
        collector = LiveBitcoinCollector()
        dsn       = os.environ.get(
            "DATABASE_URL", "postgresql://btcqa:btcqa@localhost:5432/btcqa"
        )
        mabse = MABSE(
            db_dsn        = dsn,
            tau_short     = args.tau_short,
            tau_mid       = args.tau_mid,
            tau_long      = args.tau_long if use_multiscale else args.tau_mid,
            k_neighbors   = 6,
            kde_bandwidth = args.kde_bandwidth,
        )
        print(f"[MABSE] Live mode. DB episodes: {mabse.n_episodes()}")
    else:
        syn_env = SyntheticBitcoinEnv(
            seed          = args.seed,
            mu_mempool    = args.mu_mempool,
            sigma_mempool = args.sigma_mempool,
            lambda_inter  = args.lambda_inter,
        )
        mabse   = None
        collector = None
        print("[MABSE] Synthetic mode (no live data)")

    # ── model & optimiser ──────────────────────
    model    = ActorCritic(in_dim=state_dim, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    norm      = RunningNorm(dim=state_dim)

    # ── DB logging ─────────────────────────────
    db     = None
    run_id = None
    if not args.no_db:
        try:
            from db import EpisodeDB
            db = EpisodeDB.from_env()
            db.init_schema()
            run_id = db.create_experiment(
                policy        = args.policy_name,
                ablation_type = args.ablation_type,
                lambda_cost   = args.lambda_cost,
                seed          = args.seed,
                notes         = args.notes or None,
            )
            print(f"[DB] run_id={run_id}")
        except Exception as e:
            print(f"[DB] skipped: {e}")

    # ── training state ─────────────────────────
    prev_snap: Optional[LiveNetworkSnapshot] = None
    prev_action = "skip"
    rollout_buf: List[StepRecord] = []
    all_rewards: List[float]      = []

    def get_snap():
        if use_live:
            return collector.collect()
        else:
            return syn_env.step()

    def build_state_vector(
        snap:        LiveNetworkSnapshot,
        prev_snap_:  Optional[LiveNetworkSnapshot],
        prev_action_: str,
        step:        int,
    ) -> torch.Tensor:
        if use_mabse and mabse is not None:
            ms: MABSEState = mabse.compute(
                mempool_mb      = snap.mempool_size_mb,
                pending_tx      = snap.pending_tx,
                inter_block_sec = snap.inter_block_time_sec,
                prev_mempool_mb = prev_snap_.mempool_size_mb if prev_snap_ else 0.0,
                prev_pending_tx = prev_snap_.pending_tx      if prev_snap_ else 0,
                prev_action     = prev_action_,
                current_step    = step,
            )
            return torch.tensor(ms.to_vector(), dtype=torch.float32)
        else:
            # AC base: 6D state
            m  = min(snap.mempool_size_mb / 150.0, 1.0)
            p  = min(snap.pending_tx / 700_000.0,  1.0)
            b  = min(snap.inter_block_time_sec / 1200.0, 1.0)
            dm = (snap.mempool_size_mb - (prev_snap_.mempool_size_mb if prev_snap_ else 0)) / 30.0
            dp = (snap.pending_tx - (prev_snap_.pending_tx if prev_snap_ else 0)) / 50_000.0
            h  = ACTION_ENC.get(prev_action_, 0.0)
            return torch.tensor([m, p, b, dm, dp, h], dtype=torch.float32)

    def is_anomalous(snap: LiveNetworkSnapshot, state_: torch.Tensor) -> bool:
        if use_mabse and mabse is not None and use_kde:
            ms_state = mabse.compute(
                mempool_mb      = snap.mempool_size_mb,
                pending_tx      = snap.pending_tx,
                inter_block_sec = snap.inter_block_time_sec,
            )
            return mabse.is_anomalous(
                ms_state,
                snap.mempool_size_mb,
                snap.pending_tx,
                snap.inter_block_time_sec,
            )
        # Fixed thresholds (paper default)
        return (
            snap.mempool_size_mb      > 100.0
            or snap.inter_block_time_sec > 1100.0
            or snap.pending_tx           > 15_000
        )

    # ── main loop ──────────────────────────────
    curr_snap = get_snap()

    for step in range(1, args.steps + 1):
        state_vec = build_state_vector(curr_snap, prev_snap, prev_action, step)
        norm.update(state_vec)
        state_n   = norm.normalize(state_vec)

        logits, value = model(state_n.unsqueeze(0))
        probs  = F.softmax(logits, dim=-1).squeeze(0)
        dist   = Categorical(probs=probs)
        a_idx  = int(dist.sample().item())
        action = ACTIONS[a_idx]

        # Anomaly detection (fixed or KDE-adaptive)
        anomalous = is_anomalous(curr_snap, state_vec)

        # Ensemble belief for RAG risk term
        if use_mabse and mabse is not None:
            ms_state = mabse.compute(
                mempool_mb      = curr_snap.mempool_size_mb,
                pending_tx      = curr_snap.pending_tx,
                inter_block_sec = curr_snap.inter_block_time_sec,
                current_step    = step,
            )
            ens_belief = ms_state.ensemble_belief
        else:
            ens_belief = 0.0

        terminal_miss = anomalous and action == "skip"
        reward = compute_reward(
            anomalous      = anomalous,
            action         = action,
            ensemble_belief= ens_belief,
            terminal_miss  = terminal_miss,
            alpha          = args.alpha,
            lambda_cost    = args.lambda_cost,
            beta           = args.beta,
            gamma_fn       = args.gamma_fn,
        )
        reward = max(-args.reward_clip, min(args.reward_clip, reward))

        rollout_buf.append(StepRecord(
            state      = state_n,
            action_idx = a_idx,
            value      = float(value.item()),
            reward     = float(reward),
        ))
        all_rewards.append(float(reward))

        # Persist to DB
        if db and run_id:
            try:
                from memory import Episode
                ep_obj = Episode(
                    mempool_size      = curr_snap.mempool_size_mb,
                    pending_tx        = curr_snap.pending_tx,
                    inter_block_time  = curr_snap.inter_block_time_sec,
                    action            = action,
                    reward            = reward,
                    cost              = ACTION_COST[action],
                    detected          = anomalous and action != "skip",
                    step              = step,
                    timestamp         = curr_snap.collected_at.timestamp(),
                )
                db.insert_episode(run_id, ep_obj, step=step)
            except Exception:
                pass

        prev_snap   = curr_snap
        prev_action = action
        curr_snap   = get_snap()

        # ── GAE-A2C update ─────────────────────
        should_update = (len(rollout_buf) >= args.rollout) or (step == args.steps)
        if not should_update:
            continue

        next_state = build_state_vector(curr_snap, prev_snap, prev_action, step + 1)
        next_state = norm.normalize(next_state)
        with torch.no_grad():
            _, next_val = model(next_state.unsqueeze(0))
        bootstrap = float(next_val.item())

        rewards = [r.reward for r in rollout_buf]
        values  = [r.value  for r in rollout_buf]
        adv, ret = compute_gae(rewards, values, bootstrap, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states  = torch.stack([r.state      for r in rollout_buf])
        actions = torch.tensor([r.action_idx for r in rollout_buf], dtype=torch.long)

        logits_b, values_b = model(states)
        d          = Categorical(logits=logits_b)
        logp_b     = d.log_prob(actions)
        entropy_b  = d.entropy().mean()

        policy_loss = -(logp_b * adv.detach()).mean()
        value_loss  = F.mse_loss(values_b, ret.detach())
        total_loss  = policy_loss + args.c_v * value_loss - args.c_e * entropy_b

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        rollout_buf = []
        mean_r = sum(all_rewards[-100:]) / max(1, len(all_rewards[-100:]))
        print(
            f"[step={step:04d}] mean_r100={mean_r:+.4f} "
            f"loss={float(total_loss.item()):.4f} "
            f"belief_ens={ens_belief:.3f} "
            f"anomalous={anomalous} "
            f"ablation={args.ablation_type}"
        )

    # ── final rollup ───────────────────────────
    if db and run_id:
        try:
            stats = db.get_experiment_stats(run_id)
            denom = stats.get("mean_cost") or 1e-8
            db.insert_rollup(run_id, {
                "mean_return":    stats.get("mean_return"),
                "detection_rate": stats.get("detection_rate"),
                "mean_cost":      stats.get("mean_cost"),
                "mean_ttd":       stats.get("mean_ttd"),
                "efficiency_eta": (stats.get("detection_rate") or 0) / denom,
                "total_episodes": stats.get("total"),
            })
            db.close()
            print(f"[DB] rollup saved run_id={run_id}")
        except Exception as e:
            print(f"[DB] rollup failed: {e}")

    print(f"[DONE] steps={args.steps} ablation={args.ablation_type}")


if __name__ == "__main__":
    main()
