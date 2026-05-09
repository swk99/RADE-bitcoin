"""
btc_live_collector.py
---------------------
Real Bitcoin network data collector via mempool.space public REST API.
No API key required. Falls back to synthetic data if API is unavailable.

Collected signals:
  - mempool_size_mb     : total unconfirmed transaction weight (MB)
  - pending_tx          : number of unconfirmed transactions
  - inter_block_time_sec: seconds since last confirmed block
  - fee_rate_fast_sat_vb: recommended fast fee (sat/vbyte)
  - fee_rate_med_sat_vb : recommended medium fee (sat/vbyte)
  - hashrate_eh_s       : current estimated network hashrate (EH/s)
  - difficulty          : current mining difficulty

Usage:
    collector = LiveBitcoinCollector()
    snap = collector.collect()
    print(snap)
"""

from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)

MEMPOOL_BASE = "https://mempool.space/api"
DEFAULT_TIMEOUT = 15   # seconds per request
RETRY_BACKOFF   = [1, 2, 4]  # retry wait times (seconds)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class LiveNetworkSnapshot:
    """One point-in-time observation of the Bitcoin network."""
    collected_at:          datetime     # UTC
    mempool_size_mb:       float        # MB
    pending_tx:            int          # count
    inter_block_time_sec:  float        # seconds since last block
    fee_rate_fast_sat_vb:  float        # sat/vbyte
    fee_rate_med_sat_vb:   float        # sat/vbyte
    hashrate_eh_s:         float        # EH/s  (0.0 if unavailable)
    difficulty:            float        # raw difficulty (0.0 if unavailable)
    source:                str          # "live" | "synthetic"
    error:                 Optional[str] = None  # last error if partial fetch

    def to_dict(self) -> dict:
        d = asdict(self)
        d["collected_at"] = self.collected_at.isoformat()
        return d

    # Derived normalised features used by rade_train.py
    @property
    def norm_mempool(self) -> float:
        return min(self.mempool_size_mb / 150.0, 1.0)

    @property
    def norm_pending(self) -> float:
        return min(self.pending_tx / 700_000.0, 1.0)

    @property
    def norm_inter_block(self) -> float:
        return min(self.inter_block_time_sec / 1_200.0, 1.0)

    def is_anomalous(self) -> bool:
        """Oracle from Eq.(6) in the RADE paper."""
        return (
            self.mempool_size_mb > 100.0
            or self.inter_block_time_sec > 1_100.0
            or self.pending_tx > 15_000
        )


# ──────────────────────────────────────────────
# Collector
# ──────────────────────────────────────────────

class LiveBitcoinCollector:
    """
    Collects real Bitcoin network snapshots from mempool.space.

    Parameters
    ----------
    base_url : str
        Root of the mempool.space REST API.
    timeout  : int
        Per-request timeout in seconds.
    retries  : int
        Number of retry attempts on transient errors.
    seed     : int
        RNG seed for synthetic fallback.
    """

    def __init__(
        self,
        base_url: str = MEMPOOL_BASE,
        timeout:  int = DEFAULT_TIMEOUT,
        retries:  int = 2,
        seed:     int = 42,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.retries  = retries
        self._rng     = random.Random(seed)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "btc-qa-rade/1.0"})
        self._last_block_ts: Optional[int] = None

    # ── low-level HTTP ──────────────────────────

    def _get(self, path: str) -> dict | list | int | str:
        url = f"{self.base_url}{path}"
        last_exc: Exception = RuntimeError("no attempt")
        for wait in [0] + RETRY_BACKOFF[: self.retries]:
            if wait:
                time.sleep(wait)
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return resp.json()
                try:
                    return int(resp.text.strip())
                except ValueError:
                    return resp.text.strip()
            except Exception as exc:
                last_exc = exc
                log.warning("GET %s failed: %s", url, exc)
        raise last_exc

    # ── individual signal fetchers ───────────────

    def _fetch_mempool(self) -> tuple[float, int]:
        """Returns (mempool_size_mb, pending_tx_count)."""
        data = self._get("/mempool")
        vsize_bytes = float(data["vsize"])   # virtual bytes
        count       = int(data["count"])
        return vsize_bytes / 1_000_000.0, count

    def _fetch_inter_block_time(self) -> float:
        """
        Seconds since last confirmed block.
        Uses the timestamp of the tip block compared to now.
        Falls back to 600.0 on first call (no prior reference).
        """
        tip_height = self._get("/blocks/tip/height")
        block_hash = self._get(f"/block-height/{tip_height}")
        block_info = self._get(f"/block/{block_hash}")
        block_ts   = int(block_info["timestamp"])

        now_ts = int(datetime.now(timezone.utc).timestamp())
        inter  = max(1.0, float(now_ts - block_ts))
        self._last_block_ts = block_ts
        return inter

    def _fetch_fees(self) -> tuple[float, float]:
        """Returns (fast_sat_vb, medium_sat_vb)."""
        data = self._get("/v1/fees/recommended")
        fast = float(data.get("fastestFee", data.get("hourFee", 10.0)))
        med  = float(data.get("halfHourFee", data.get("hourFee", 5.0)))
        return fast, med

    def _fetch_hashrate(self) -> tuple[float, float]:
        """Returns (hashrate_EH_s, difficulty). May fail silently."""
        data      = self._get("/v1/mining/hashrate/1m")
        hashrates = data.get("hashrates", [])
        if hashrates:
            # last entry is most recent
            hr_hash_per_sec = float(hashrates[-1].get("avgHashrate", 0))
            hr_eh_s = hr_hash_per_sec / 1e18
        else:
            hr_eh_s = 0.0
        difficulty = float(data.get("difficulty", [{}])[-1].get("difficulty", 0.0)) \
            if data.get("difficulty") else 0.0
        return hr_eh_s, difficulty

    # ── public interface ─────────────────────────

    def collect_live(self) -> LiveNetworkSnapshot:
        """
        Fetches all signals from mempool.space.
        Raises on unrecoverable errors.
        """
        errors: list[str] = []

        # Required signals (raise if fail)
        mempool_mb, pending = self._fetch_mempool()
        inter_block         = self._fetch_inter_block_time()

        # Optional signals (degrade gracefully)
        try:
            fee_fast, fee_med = self._fetch_fees()
        except Exception as e:
            fee_fast, fee_med = 10.0, 5.0
            errors.append(f"fees: {e}")

        try:
            hashrate, difficulty = self._fetch_hashrate()
        except Exception as e:
            hashrate, difficulty = 0.0, 0.0
            errors.append(f"hashrate: {e}")

        return LiveNetworkSnapshot(
            collected_at          = datetime.now(timezone.utc),
            mempool_size_mb       = round(mempool_mb,  4),
            pending_tx            = pending,
            inter_block_time_sec  = round(inter_block, 1),
            fee_rate_fast_sat_vb  = round(fee_fast,    2),
            fee_rate_med_sat_vb   = round(fee_med,     2),
            hashrate_eh_s         = round(hashrate,    4),
            difficulty            = difficulty,
            source                = "live",
            error                 = "; ".join(errors) or None,
        )

    def collect_synthetic(self) -> LiveNetworkSnapshot:
        """Synthetic fallback preserving realistic ranges."""
        rng = self._rng
        return LiveNetworkSnapshot(
            collected_at          = datetime.now(timezone.utc),
            mempool_size_mb       = round(rng.uniform(5.0, 140.0),  2),
            pending_tx            = int(rng.uniform(300, 650_000)),
            inter_block_time_sec  = round(rng.expovariate(1 / 592), 1),
            fee_rate_fast_sat_vb  = round(rng.uniform(5.0, 80.0),   2),
            fee_rate_med_sat_vb   = round(rng.uniform(2.0, 40.0),   2),
            hashrate_eh_s         = round(rng.uniform(600, 800),     2),
            difficulty            = rng.uniform(8e13, 1e14),
            source                = "synthetic",
        )

    def collect(self) -> LiveNetworkSnapshot:
        """Try live; fall back to synthetic on any error."""
        try:
            snap = self.collect_live()
            log.info(
                "live | mempool=%.1fMB pending=%d inter_block=%.0fs anomaly=%s",
                snap.mempool_size_mb, snap.pending_tx,
                snap.inter_block_time_sec, snap.is_anomalous(),
            )
            return snap
        except Exception as exc:
            log.warning("live fetch failed (%s) — using synthetic", exc)
            return self.collect_synthetic()


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    c = LiveBitcoinCollector()
    snap = c.collect()
    import json
    print(json.dumps(snap.to_dict(), indent=2))
    print(f"\nAnomalous: {snap.is_anomalous()}")
    print(f"Norm state: m={snap.norm_mempool:.3f} "
          f"p={snap.norm_pending:.3f} b={snap.norm_inter_block:.3f}")
