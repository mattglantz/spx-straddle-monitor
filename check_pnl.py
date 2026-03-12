import sqlite3
from datetime import datetime
import trade_status as ts

conn = sqlite3.connect('trading_journal.db')
conn.row_factory = sqlite3.Row

# Show table schema
cols = conn.execute("PRAGMA table_info(trades)").fetchall()
print("Schema:", [c['name'] for c in cols])
print()

# Today's trades
rows = conn.execute("SELECT * FROM trades WHERE timestamp LIKE '2026-03-03%' ORDER BY id").fetchall()
print(f"Total trades today: {len(rows)}")
print()

for r in rows:
    d = dict(r)
    print(f"ID:{d['id']} | {d['timestamp']} | {d['verdict']} | Entry:{d['price']} | Target:{d['target']} | Stop:{d['stop']} | Status:{d['status']} | PnL:{d['pnl']:.2f} | Conf:{d['confidence']}")

print()
closed = [dict(r) for r in rows if dict(r)['status'] in ts.CLOSED_STATUSES]
floating = [dict(r) for r in rows if dict(r)['status'] == ts.FLOATING]
realized = sum(t['pnl'] for t in closed)
float_pnl = sum(t['pnl'] for t in floating)

print(f"Closed trades: {len(closed)}")
print(f"Floating trades: {len(floating)}")
print(f"Realized P&L: {realized:+.2f} pts (${realized*50:+,.0f})")
print(f"Floating P&L: {float_pnl:+.2f} pts (${float_pnl*50:+,.0f})")
print(f"Net P&L: {realized+float_pnl:+.2f} pts (${(realized+float_pnl)*50:+,.0f})")

wins = sum(1 for t in closed if t['pnl'] > 0)
losses = sum(1 for t in closed if t['pnl'] < 0)
wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wr:.0f}%")

# Also check what the bot is reporting
print()
print("--- What bot reports via get_today_stats ---")
all_trades = [dict(r) for r in rows]
closed2 = [t for t in all_trades if t['status'] in ts.CLOSED_STATUSES]
floating2 = [t for t in all_trades if t['status'] == ts.FLOATING]
r2 = sum(t['pnl'] for t in closed2)
f2 = sum(t['pnl'] for t in floating2)
print(f"realized={r2:+.2f}, floating={f2:+.2f}, net={r2+f2:+.2f}")

conn.close()
