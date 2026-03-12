"""Simulate how the new guards would have affected today's trades."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sqlite3, json

conn = sqlite3.connect('trading_journal.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, timestamp, price, verdict, confidence, target, stop, status, pnl, contracts, signals "
    "FROM trades WHERE timestamp LIKE '2026-03-06%' ORDER BY id"
).fetchall()

print("=== RETROACTIVE GUARD SIMULATION ===")
print("Would these guards have prevented today's losses?\n")

saved = 0
for r in rows:
    sigs = json.loads(r['signals'] or '{}')
    verdict = r['verdict'].upper()
    mtf = sigs.get('mtf', '')
    div = sigs.get('divergence', '')
    fractal = sigs.get('fractal', '')
    fractal_conf = int(sigs.get('fractal_conf', 0))
    confl = sigs.get('confluence', '')
    pnl = (r['pnl'] or 0) * 50 * (r['contracts'] or 1)

    blocked = False
    guard = ''

    v_bull = any(k in verdict for k in ('BULL', 'LONG'))
    v_bear = any(k in verdict for k in ('BEAR', 'SHORT'))
    div_high = any(k in div for k in ('HIGH', 'EXTREME'))

    # Guard 1
    if v_bull and 'FULL BEARISH' in mtf and div_high:
        blocked = True
        guard = 'CONFLICT GUARD (BULL vs FULL BEARISH MTF + HIGH DIV)'
    elif v_bear and 'FULL BULLISH' in mtf and div_high:
        blocked = True
        guard = 'CONFLICT GUARD (BEAR vs FULL BULLISH MTF + HIGH DIV)'

    # Guard 3
    if not blocked:
        frac_bull = 'BULL' in fractal
        frac_bear = 'BEAR' in fractal
        contradicts = (
            (v_bull and frac_bear and fractal_conf >= 60) or
            (v_bear and frac_bull and fractal_conf >= 60)
        )
        if contradicts and confl not in ('STRONG', 'MODERATE'):
            blocked = True
            guard = f'FRACTAL CONTRADICTION ({verdict} vs {fractal} {fractal_conf}%, confl={confl})'

    if blocked:
        saved += pnl if pnl < 0 else 0
        mark = 'BLOCKED'
    else:
        mark = 'allowed'

    print(f"#{r['id']} {r['verdict']} {r['confidence']}% @ {r['price']:.2f} -> {r['status']} ${pnl:+,.0f}")
    print(f"   MTF={mtf} | DIV={div} | Fractal={fractal}({fractal_conf}%) | Confl={confl}")
    print(f"   >>> {mark} {guard}")
    print()

print(f"=== Would have saved ${abs(saved):,.0f} of today's ${6000:,} loss ===")
conn.close()
