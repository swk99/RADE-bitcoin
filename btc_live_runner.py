"""
btc_live_runner.py
------------------
RADE — Scheduled Bitcoin network data collector.
Runs indefinitely, collecting one snapshot every --interval seconds
(default: 600 = 10 minutes, matching Bitcoin's block target).

Usage:
    python btc_live_runner.py                          # 10분마다 무한 수집
    python btc_live_runner.py --interval 30 --steps 10 --no-db  # 빠른 테스트
    python btc_live_runner.py --calibrate-every 144    # 하루마다 auto calibrate
    python btc_live_runner.py --status                 # 수집 현황 확인

Environment:
    DATABASE_URL  PostgreSQL DSN (default: postgresql://btcqa:btcqa@localhost/btcqa)
    MEMPOOL_URL   Override mempool.space base URL
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)
_STOP = False

BANNER = """
╔══════════════════════════════════════════════════╗
║       RADE — Bitcoin Live Data Collector         ║
╚══════════════════════════════════════════════════╝"""


def _handle_signal(sig, frame):
    global _STOP
    print(f"\n[RADE] Signal {sig} — 현재 스텝 완료 후 종료합니다.")
    _STOP = True


# ──────────────────────────────────────────────
# Status report
# ──────────────────────────────────────────────

def print_status():
    from btc_live_db import LiveSnapshotDB
    db = LiveSnapshotDB.from_env()
    db.init_schema()

    total = db.count()
    live  = db.count(source="live")
    synth = db.count(source="synthetic")
    cal   = db.latest_calibration_params()

    print("\n" + "=" * 52)
    print("  RADE — Live Bitcoin Data Collection Status")
    print("=" * 52)
    print(f"  Total snapshots : {total:,}")
    print(f"    live          : {live:,}")
    print(f"    synthetic     : {synth:,}")

    if total > 0:
        df = db.to_dataframe(days=9999)
        if len(df):
            first = df["collected_at"].iloc[0]
            last  = df["collected_at"].iloc[-1]
            anom  = df["is_anomalous"].sum()
            print(f"  Date range      : {first} → {last}")
            print(f"  Anomalous       : {anom:,} ({anom/len(df)*100:.1f}%)")

    if cal:
        print(f"\n  Latest calibration (id={cal['id']})")
        print(f"    Computed at    : {cal['computed_at']}")
        print(f"    Samples used   : {cal['n_samples']:,}")
        print(f"    mu_mempool     : {cal['mu_mempool']}")
        print(f"    sigma_mempool  : {cal['sigma_mempool']}")
        print(f"    lambda_inter   : {cal['lambda_inter']}")
        print(f"    anomaly_rate   : {cal['anomaly_rate']*100:.1f}%")
        print(f"    P95 thresholds : mempool={cal['p95_mempool_mb']}MB  "
              f"inter={cal['p95_inter_sec']}s  pending={cal['p95_pending']:,}")
    else:
        print("\n  No calibration run yet.")
        if total < 144:
            needed = 144 - total
            print(f"  → Need ~{needed} more snapshots "
                  f"(~{needed*10//60}h at 10min interval)")

    print("=" * 52)
    db.close()


# ──────────────────────────────────────────────
# Countdown sleep
# ──────────────────────────────────────────────

def _sleep_with_countdown(seconds: float, step: int, total_steps: int):
    """10초마다 카운트다운 출력. Ctrl+C 로 중단 가능."""
    global _STOP
    if seconds <= 0:
        return

    next_at   = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    remaining = seconds

    while remaining > 0 and not _STOP:
        chunk     = min(10.0, remaining)
        time.sleep(chunk)
        remaining -= chunk

        if _STOP:
            break

        filled   = int((1.0 - remaining / seconds) * 20)
        bar      = "█" * filled + "░" * (20 - filled)
        next_str = next_at.strftime("%H:%M:%S")
        step_str = f"{step}/{total_steps}" if total_steps else f"{step}/∞"

        print(
            f"\r  ⏳ [{bar}] {int(remaining):3d}s 남음 │ "
            f"다음 수집 {next_str} UTC │ step {step_str}   ",
            end="", flush=True,
        )

    # 줄 정리
    print("\r" + " " * 85 + "\r", end="", flush=True)


# ──────────────────────────────────────────────
# Main collection loop
# ──────────────────────────────────────────────

def run(
    interval:        int  = 600,
    steps:           int  = 0,
    no_db:           bool = False,
    calibrate_every: int  = 0,
    calibrate_days:  int  = 90,
    verbose:         bool = False,
):
    from btc_live_collector import LiveBitcoinCollector
    from btc_live_db        import LiveSnapshotDB

    print(BANNER)

    collector = LiveBitcoinCollector(
        base_url=os.environ.get("MEMPOOL_URL", "https://mempool.space/api"),
    )

    db = None
    if not no_db:
        db = LiveSnapshotDB.from_env()
        db.init_schema()
        print(f"\n[DB] PostgreSQL 연결 완료. 기존 스냅샷: {db.count():,}개")

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    step            = 0
    total_collected = 0
    total_errors    = 0
    total_anomalous = 0

    iv_str  = f"{interval}s ({interval//60}분)" if interval >= 60 else f"{interval}s"
    cal_str = f"{calibrate_every}회마다" if calibrate_every else "없음"

    print(f"\n[설정] 수집 간격={iv_str}  "
          f"스텝={steps or '무제한'}  "
          f"DB={'OFF' if no_db else 'ON'}  "
          f"AutoCalib={cal_str}")
    print(f"[시작] {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    print("─" * 80)

    while not _STOP:
        step += 1
        if steps and step > steps:
            print(f"\n[완료] --steps {steps} 도달. 종료.")
            break

        t_start  = time.monotonic()
        now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # ── 수집 ──────────────────────────────
        print(f"\n[{step:06d}] {now_str} UTC  ▶ 수집 중...", flush=True)
        snap = collector.collect()
        total_collected += 1
        if snap.error:
            total_errors += 1

        anomalous = snap.is_anomalous()
        if anomalous:
            total_anomalous += 1

        # ── DB 저장 ───────────────────────────
        row_id = None
        if db:
            try:
                row_id = db.insert(snap)
            except Exception as exc:
                print(f"  [DB ERROR] {exc}", flush=True)

        # ── 결과 출력 ─────────────────────────
        status_icon = "⚠️  이상" if anomalous else "✅ 정상"
        src_icon    = "🌐" if snap.source == "live" else "🔧"
        anom_rate   = total_anomalous / total_collected * 100

        print(f"  {src_icon}  소스     : {snap.source}", flush=True)
        print(f"  {status_icon}", flush=True)
        print(
            f"  📊 Mempool  : {snap.mempool_size_mb:8.2f} MB  │  "
            f"Pending: {snap.pending_tx:>9,} tx  │  "
            f"Inter-block: {snap.inter_block_time_sec:6.0f}s",
            flush=True,
        )
        print(
            f"  💸 Fee      : fast={snap.fee_rate_fast_sat_vb:.1f} sat/vb  │  "
            f"mid={snap.fee_rate_med_sat_vb:.1f} sat/vb",
            flush=True,
        )
        if snap.hashrate_eh_s > 0:
            print(f"  ⛏  Hashrate : {snap.hashrate_eh_s:.1f} EH/s", flush=True)
        print(
            f"  📈 누적     : 수집={total_collected:,}  "
            f"이상={total_anomalous:,} ({anom_rate:.1f}%)  "
            f"오류={total_errors:,}"
            + (f"  [DB id={row_id}]" if row_id and verbose else ""),
            flush=True,
        )
        if snap.error:
            print(f"  ❌ 오류     : {snap.error}", flush=True)

        # ── 자동 calibration ──────────────────
        if calibrate_every and db and (step % calibrate_every == 0):
            print(f"\n[Calibration] step={step} → MDP 파라미터 재추정 중...",
                  flush=True)
            try:
                from btc_calibrate import calibrate
                params = calibrate(
                    days    = calibrate_days,
                    save_db = True,
                    notes   = f"auto_step={step}",
                )
                print(
                    f"[Calibration] ✅ 완료: "
                    f"μ_m={params['mu_mempool']}  "
                    f"σ_m={params['sigma_mempool']}  "
                    f"λ_b={params['lambda_inter']:.6f}  "
                    f"anomaly={params['anomaly_rate']*100:.1f}%",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Calibration] ❌ 실패: {exc}", flush=True)

        # ── 대기 (카운트다운) ─────────────────
        elapsed = time.monotonic() - t_start
        sleep   = max(0.0, interval - elapsed)

        if sleep > 0 and not _STOP:
            _sleep_with_countdown(sleep, step, steps)

    # ── 최종 요약 ─────────────────────────────
    print("\n" + "=" * 52)
    print("  RADE 수집 종료")
    print("=" * 52)
    print(f"  총 수집   : {total_collected:,}회")
    print(f"  이상 탐지 : {total_anomalous:,}회 "
          f"({total_anomalous/max(1,total_collected)*100:.1f}%)")
    print(f"  오류      : {total_errors:,}회")
    print(f"  종료 시각 : "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 52)

    if db:
        db.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RADE — Collect live Bitcoin network snapshots"
    )
    p.add_argument("--interval", type=int, default=600,
                   help="Seconds between collections (default: 600)")
    p.add_argument("--steps",    type=int, default=0,
                   help="Stop after N steps (0=unlimited)")
    p.add_argument("--no-db",    action="store_true",
                   help="Skip PostgreSQL writes")
    p.add_argument("--calibrate-every", type=int, default=0, metavar="N",
                   help="Auto-calibrate every N snapshots")
    p.add_argument("--calibrate-days",  type=int, default=90,
                   help="Days to use for calibration (default: 90)")
    p.add_argument("--verbose",  action="store_true",
                   help="Show DB row IDs")
    p.add_argument("--status",   action="store_true",
                   help="Show collection status and exit")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.WARNING,
        format  = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
    )
    args = parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    run(
        interval        = args.interval,
        steps           = args.steps,
        no_db           = args.no_db,
        calibrate_every = args.calibrate_every,
        calibrate_days  = args.calibrate_days,
        verbose         = args.verbose,
    )
