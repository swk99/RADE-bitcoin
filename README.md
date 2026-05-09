# 🔍 RADE: Retrieval-Augmented Diagnostic Engine

> **Cost-Aware Bitcoin Node Diagnostics via Actor-Critic RL with Multi-scale Belief Estimation**
>
> 📄 *Submitted to ACM ICAIF 2025* | 🏫 Goldsmiths, University of London | ✍️ Seonwoo Kim

---

## 🤔 What is this?

So basically... Bitcoin nodes need to be monitored constantly. But here's the thing — **monitoring costs money** (CPU, I/O, engineer time), and you can't just probe every 10 seconds forever 😅

This project treats Bitcoin node diagnosis as a **sequential decision problem** (POMDP):
- At each step, an agent decides: `probe` 🔍, `skip` ⏭️, or `escalate` 🚨
- The agent wants to **detect anomalies** while **minimising probe cost**
- Too aggressive → expensive. Too lazy → misses anomalies. Finding the balance is the whole point!

### 🌟 Key Contributions

| Feature | What it does |
|---|---|
| 🧠 **Multi-scale Belief** | 3 temporal scales (10min / 1hr / 24hr) for richer anomaly evidence |
| 📊 **KDE Adaptive Threshold** | Dynamic anomaly detection that *learns* from real data (bye bye fixed thresholds!) |
| 🗄️ **Live Bitcoin Data** | Real mempool.space data → calibrated MDP parameters |
| 🚫 **No LLM needed** | Fully data-driven, no API costs, runs offline |

---

## 💡 Why does this matter?

Here's a fun story 😅

The original anomaly threshold for Bitcoin pending transactions was `> 15,000 tx`.

**But in 2026?** The network regularly sits at **40,000~55,000 tx** under completely normal conditions.

```
Fixed threshold (2019): 15,000 tx  ❌ flags EVERYTHING as anomaly
Real network (2026):    ~45,000 tx  ✅ totally normal
```

This is exactly why **adaptive thresholds** matter! Our KDE-based approach automatically learns the current distribution and adjusts. The fixed threshold world is crumbling 🏚️ and RADE is here to save it 🦸

---

## 🗂️ Project Structure

```
bitcoin-qa-orchestrator/
│
├── 📡 Data Collection
│   ├── btc_live_collector.py   ← Fetches real data from mempool.space API
│   ├── btc_live_db.py          ← PostgreSQL schema (live_snapshots table)
│   ├── btc_live_runner.py      ← Scheduler (runs every 10 min automatically)
│   └── btc_calibrate.py        ← Fits MDP params from collected data
│
├── 🧠 Core Algorithm
│   ├── rade_belief.py          ← RADE: Multi-scale belief + KDE threshold
│   └── rade_train_live.py      ← GAE-A2C training loop
│
├── 🗄️ Persistence
│   ├── db.py                   ← Experiment logging (PostgreSQL)
│   └── memory.py               ← ChromaDB episodic memory
│
└── 📋 requirements.txt
```

---

## 🚀 Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ running locally
- mempool.space API access (free, no key needed! 🎉)

### Installation

```bash
# Clone the repo
git clone https://github.com/seonwoojh/bitcoin-qa-orchestrator.git
cd bitcoin-qa-orchestrator

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Create PostgreSQL database
createdb btcqa

# Set your DATABASE_URL (or use the default)
# Default: postgresql://btcqa:btcqa@localhost:5432/btcqa

# Initialise schema (run once!)
python -c "
from btc_live_db import LiveSnapshotDB
db = LiveSnapshotDB.from_env()
db.init_schema()
print('✅ DB schema ready!')
"
```

---

## 📡 Step 1: Start Collecting Live Bitcoin Data

This is the **most important step** — everything else depends on having real data!

```bash
# Start collecting (runs every 10 minutes, forever)
python btc_live_runner.py --calibrate-every 144

# Quick test first (30 second intervals, 5 steps, no DB)
python btc_live_runner.py --interval 30 --steps 5 --no-db
```

You'll see something like this 👇

```
╔══════════════════════════════════════════════════╗
║       RADE — Bitcoin Live Data Collector         ║
╚══════════════════════════════════════════════════╝

[000001] 2026-05-09 07:35:00 UTC  ▶ 수집 중...
  🌐  소스     : live
  ✅ 정상
  📊 Mempool  :    33.91 MB  │  Pending:    46,253 tx  │  Inter-block: 587s
  💸 Fee      : fast=2.0 sat/vb  │  mid=1.0 sat/vb
  ⛏  Hashrate : 918.8 EH/s
  📈 누적     : 수집=1  이상=0 (0.0%)  오류=0

  ⏳ [████░░░░░░░░░░░░░░░░]  483s 남음 │ 다음 수집 07:45:00 UTC
```

> 💡 **Why keep it running?**
> RADE needs enough data to fit the distribution. The more data, the better the calibration!
> Minimum: **144 snapshots** (~24 hours) for a meaningful calibration run.

### 🔍 Check Collection Status (anytime!)

```bash
python btc_live_runner.py --status
```

```
====================================================
  RADE — Live Bitcoin Data Collection Status
====================================================
  Total snapshots : 144
    live          : 144
    synthetic     : 0
  Date range      : 2026-05-08 → 2026-05-09
  Anomalous       : 144 (100.0%)  ← this is expected at first!
====================================================
```

> 🙃 **Don't panic if anomaly rate is 100%!**
> The initial thresholds are set to 2019 values (`pending_tx > 15,000`).
> In 2026, the normal baseline is ~45,000 tx — so everything looks "anomalous" at first.
> This is actually a **feature, not a bug** — it demonstrates exactly why adaptive thresholds matter!
> The KDE will fix this automatically as data accumulates 📈

---

## 📊 Step 2: Calibrate MDP Parameters

Once you have **at least 144 snapshots** (~1 day), run calibration:

```bash
# Calibrate from last 1 day of data
python btc_calibrate.py --days 1 --plot --output params.json

# After 7 days (more accurate!)
python btc_calibrate.py --days 7 --plot --output params_7d.json
```

Expected output:

```
=======================================================
  RADE MDP Calibration Results
=======================================================
  Samples   : 144  (1 days, source=all)

  Mempool size (MB)
    Distribution : LogNormal(mu=3.50, sigma=0.03)
    Mean (approx): 33.1 MB
    P95          : 34.3 MB  ← anomaly threshold

  Inter-block time (s)
    Distribution : Exponential(lambda=0.001667)
    Mean         : 599.9 s  (paper: 592 s) ✅
    P95          : 1354 s  ← anomaly threshold

  Pending tx
    P95          : 55,798  ← anomaly threshold (vs old: 15,000!)

  Anomaly rate   : 18.5%  (paper target: ~5%)
=======================================================
```

> 📈 **The threshold convergence story:**
> This is the whole point of RADE! Watch how `P95 pending` evolves:
> ```
> n=11  snapshots → P95 pending: 47,891  (anomaly rate: 100%)
> n=28  snapshots → P95 pending: 55,798  (anomaly rate: 100%)
> n=144 snapshots → P95 pending: ???     (anomaly rate: ~18%?)
> n=1008 snapshots → P95 pending: ???    (anomaly rate: ~5%?)
> ```
> The KDE adaptive threshold is *learning* the real distribution!
> Fixed thresholds could never do this 🎯

---

## 🧠 Step 3: Train RADE

### With Live Data (recommended!)

```bash
# Full RADE with multi-scale belief + KDE adaptive threshold
python rade_train_live.py \
  --steps 2000 \
  --ablation-type mabse_full \
  --use-live

# With calibrated parameters from Step 2
python rade_train_live.py \
  --steps 2000 \
  --ablation-type mabse_full \
  --use-live \
  --mu-mempool 3.50 \
  --sigma-mempool 0.03 \
  --lambda-inter 0.001667
```

### Without Live Data (synthetic fallback)

```bash
# Uses calibrated synthetic environment
python rade_train_live.py \
  --steps 2000 \
  --ablation-type ac_base \
  --no-live \
  --no-db
```

---

## 🔬 Step 4: Run Ablation Study

This is for the paper! 📄 We compare 4 variants:

```bash
# 1️⃣ AC baseline (no belief, no adaptive threshold)
python rade_train_live.py --steps 2000 --ablation-type ac_base --no-live

# 2️⃣ RADE-SS (single-scale belief only)
python rade_train_live.py --steps 2000 --ablation-type mabse_no_multiscale --use-live

# 3️⃣ RADE-FT (multi-scale belief, fixed threshold)
python rade_train_live.py --steps 2000 --ablation-type mabse_no_kde --use-live

# 4️⃣ RADE full (everything on!)
python rade_train_live.py --steps 2000 --ablation-type mabse_full --use-live
```

| Variant | Multi-scale Belief | KDE Threshold | State Dim |
|---|:---:|:---:|:---:|
| AC base | ❌ | ❌ | 6D |
| RADE-SS | ⚡ single | ❌ | 7D |
| RADE-FT | ✅ | ❌ | 9D |
| **RADE (full)** | ✅ | ✅ | **9D** |

---

## 📐 Algorithm Overview

### Multi-scale Belief State

At each step $t$, RADE retrieves past episodes from PostgreSQL and computes **3 independent beliefs**:

$$\hat{b}^\tau_t = \sum_{k \in \mathcal{N}_\tau(t)} \text{sim}(\mathbf{s}_t, \mathbf{s}_k) \cdot w_k^\tau \cdot \mathbf{1}[z_k = 1]$$

where the time-decay weight is:

$$w_k^\tau = \frac{\exp(-\text{age}_k / \tau)}{\sum_j \exp(-\text{age}_j / \tau)}$$

Three scales:
- 🟢 **Short** ($\tau_s$ = 10 min): Recent block-level anomalies
- 🟡 **Mid** ($\tau_m$ = 60 min): Fee pressure patterns
- 🔴 **Long** ($\tau_l$ = 1440 min): Regime-level context

### KDE Adaptive Threshold

Instead of fixed `mempool > 100MB`, RADE uses:

$$\theta_t = \hat{F}^{-1}(1 - \alpha_t), \quad \alpha_t = \alpha_0 \cdot \exp(-\lambda_\alpha \cdot \bar{b}_t)$$

When belief is high → $\alpha_t$ decreases → threshold tightens → more sensitive detection! 🎯

### Four-Term Reward

$$r_t = \underbrace{\alpha \cdot U(s_t,a_t)}_{\text{utility}} - \underbrace{\lambda \cdot c(a_t)}_{\text{cost}} - \underbrace{\beta \cdot \bar{b}_t \cdot \mathbf{1}[a_t=\text{skip}]}_{\text{RAG risk}} - \underbrace{\gamma_{FN} \cdot \mathbf{1}[\text{miss}]}_{\text{FN penalty}}$$

---

## 🗄️ Data Pipeline

```
mempool.space API
      │
      ▼
btc_live_collector.py   ←── fetches every 10 min
      │
      ▼
btc_live_db.py          ←── PostgreSQL (live_snapshots)
      │
      ├──► btc_calibrate.py    ←── fits LogNormal + Exponential
      │         │
      │         ▼
      │    params.json (μ, σ, λ, P95 thresholds)
      │
      ▼
rade_belief.py          ←── multi-scale belief + KDE threshold
      │
      ▼
rade_train_live.py      ←── GAE-A2C training
      │
      ▼
db.py                   ←── experiment results (PostgreSQL)
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Why? |
|---|---|---|
| $\gamma$ | 0.97 | Long-horizon discount |
| $\lambda_{\text{GAE}}$ | 0.95 | Bias-variance balance |
| $\tau_s / \tau_m / \tau_l$ | 10 / 60 / 1440 min | Bitcoin block / hour / day |
| $K$ neighbours | 6 | Top-K retrieval per scale |
| KDE bandwidth | Silverman's rule | Minimises MISE |
| $\alpha_0$ | 0.05 | 5% base false alarm rate |
| $\lambda_\alpha$ | 2.0 | Belief-gate sensitivity |
| Hidden size | 128 | Actor-Critic network |
| Rollout $N$ | 64 | GAE window |

---

## 📈 Expected Results

As more data is collected, you should see:

```
Snapshots  → P95 pending  → Anomaly rate
11         → 47,891       → 100%   😱
28         → 55,798       → 100%   😬
144        → ???          → ~18%?  📉
1,008      → ???          → ~5%    ✅
```

This convergence curve is the **core empirical result** of the paper! 🎯

---

## 🧪 Reproducing Paper Results

```bash
# 1. Collect data for 7+ days
python btc_live_runner.py --calibrate-every 144

# 2. Calibrate
python btc_calibrate.py --days 7 --plot --output params_7d.json

# 3. Run all ablations (λ grid × 5 seeds = 120 experiments)
for ABLATION in ac_base mabse_no_multiscale mabse_no_kde mabse_full; do
  for LAMBDA in 0.05 0.10 0.20 0.30 0.40 0.50; do
    for SEED in 41 42 43 44 45; do
      python rade_train_live.py \
        --steps 2000 \
        --ablation-type $ABLATION \
        --lambda-cost $LAMBDA \
        --seed $SEED \
        --use-live
    done
  done
done
```

---

## 📚 Key References

- Krishnamurthy (2016) — POMDP foundations
- Joseph et al. (2020, 2022) — AC for anomaly detection
- Schulman et al. (2016) — GAE
- Ng et al. (1999) — Potential-based reward shaping
- Lindstrom et al. (2020) — Functional KDE for time series
- iADCPS (2025) — Dynamic thresholds for cyber-physical systems
- Neudecker et al. (2016) — Bitcoin P2P network analysis
- Gündlach et al. (2024) — Bitcoin confirmation times

---

## 📝 Citation

```bibtex
@inproceedings{kim2025rade,
  author    = {Kim, Seonwoo},
  title     = {{RADE}: Retrieval-Augmented Diagnostic Engine with
               Multi-scale Belief Estimation for Cost-Aware
               Bitcoin Node Diagnostics via Actor-Critic {RL}},
  booktitle = {ACM International Conference on AI in Finance (ICAIF)},
  year      = {2025}
}
```

---

## 🙋 Author

**Seonwoo Kim** | MSc Computing (Data Science) | Goldsmiths, University of London

📧 skim008@gold.ac.uk | 🔗 [ORCID](https://orcid.org/0009-0005-9599-0514)

---

*Made with ☕ and a lot of Bitcoin mempool data 📊*
