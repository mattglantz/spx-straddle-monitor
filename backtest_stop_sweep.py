#!/usr/bin/env python3
"""
Stop-Floor Sweep — tests different minimum stop distances to find optimal size.
Runs the full backtest for each stop floor value and compares results.
"""

import sys
from pathlib import Path
from backtest import run_backtest
import numpy as np


STOP_FLOORS = [4, 6, 8, 10, 12, 14, 16, 18, 20]


def run_sweep(start_date=None, end_date=None, walk_forward=False):
    results = []

    for floor in STOP_FLOORS:
        print(f"\n--- Running stop_floor = {floor} pts ---")
        r = run_backtest(
            start_date=start_date,
            end_date=end_date,
            walk_forward=walk_forward,
            stop_floor=float(floor),
        )

        if r.trades_taken == 0:
            results.append({"floor": floor, "trades": 0})
            continue

        win_rate = r.wins / r.trades_taken * 100
        avg_win = r.gross_wins / r.wins if r.wins else 0
        avg_loss = r.gross_losses / r.losses if r.losses else 0
        pf = r.gross_wins / r.gross_losses if r.gross_losses > 0 else float("inf")

        daily_pnls = [t["pnl"] for t in r.trade_log]
        sharpe = 0.0
        if len(daily_pnls) > 1:
            mean_pnl = np.mean(daily_pnls)
            std_pnl = np.std(daily_pnls, ddof=1)
            if std_pnl > 0:
                sharpe = (mean_pnl / std_pnl) * np.sqrt(252)

        # Average stop distance actually used
        stop_dists = [abs(t["entry"] - t["stop"]) for t in r.trade_log]
        avg_stop_dist = np.mean(stop_dists) if stop_dists else 0

        results.append({
            "floor": floor,
            "trades": r.trades_taken,
            "wins": r.wins,
            "losses": r.losses,
            "stops": r.stops,
            "time_exits": r.time_exits,
            "win_rate": win_rate,
            "total_pnl": r.total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "sharpe": sharpe,
            "max_dd": r.max_drawdown,
            "avg_stop_dist": avg_stop_dist,
            "pnl_per_trade": r.total_pnl / r.trades_taken,
        })

    return results


def print_sweep(results):
    print("\n")
    print("=" * 110)
    print("STOP-FLOOR SWEEP RESULTS")
    print("=" * 110)
    print(f"{'Floor':>6} | {'Trades':>6} | {'WR%':>6} | {'Stops':>5} | {'TimeEx':>6} | "
          f"{'P&L pts':>9} | {'$/trade':>9} | {'AvgW':>6} | {'AvgL':>6} | {'PF':>5} | "
          f"{'Sharpe':>6} | {'MaxDD':>8} | {'AvgStop':>7}")
    print("-" * 110)

    best_pnl = max(r["total_pnl"] for r in results if r["trades"] > 0)
    best_sharpe = max(r["sharpe"] for r in results if r["trades"] > 0)

    for r in results:
        if r["trades"] == 0:
            print(f"{r['floor']:>5}pt | {'NO TRADES':>6}")
            continue

        pnl_flag = " <-- BEST P&L" if r["total_pnl"] == best_pnl else ""
        sharpe_flag = " <-- BEST SHARPE" if r["sharpe"] == best_sharpe and not pnl_flag else ""
        flag = pnl_flag or sharpe_flag

        print(f"{r['floor']:>5}pt | {r['trades']:>6} | {r['win_rate']:>5.1f}% | {r['stops']:>5} | "
              f"{r['time_exits']:>6} | {r['total_pnl']:>+8.1f} | "
              f"${r['pnl_per_trade'] * 50:>+7.0f} | {r['avg_win']:>+5.1f} | {r['avg_loss']:>-5.1f} | "
              f"{r['profit_factor']:>5.2f} | {r['sharpe']:>+5.2f} | "
              f"{r['max_dd']:>7.1f} | {r['avg_stop_dist']:>6.1f}{flag}")

    print("=" * 110)
    print("  $/trade = avg P&L per trade in dollars ($50/pt)")
    print("  AvgStop = average actual stop distance used (pts from entry)")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sweep stop-floor values")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    results = run_sweep(
        start_date=args.start,
        end_date=args.end,
        walk_forward=args.walk_forward,
    )
    print_sweep(results)
