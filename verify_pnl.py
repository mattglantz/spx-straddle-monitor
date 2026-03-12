"""Verify each trade's P&L calculation is correct."""

trades = [
    # ID, verdict, entry, target, stop, status, recorded_pnl
    (8,  "BEARISH", 6776.75, 6741.0, 6795.0, "LOSS",  -22.25),
    (9,  "BULLISH", 6799.0,  6810.0, 6787.0, "WIN",   11.00),
    (10, "BEARISH", 6822.25, 6807.0, 6831.0, "LOSS",  -16.00),
    (11, "BULLISH", 6838.25, 6845.0, 6834.0, "LOSS",  -4.25),
]

print("=== P&L VERIFICATION ===\n")
total_ok = True

for tid, verdict, entry, target, stop, status, recorded_pnl in trades:
    is_long = "BULL" in verdict

    if status == "WIN":
        if is_long:
            expected_pnl = target - entry
        else:
            expected_pnl = entry - target
    elif status == "LOSS":
        if is_long:
            expected_pnl = stop - entry  # Long loss: stop is below entry, so result is negative
        else:
            expected_pnl = entry - stop  # Short loss: stop is above entry, so result is negative
    else:
        expected_pnl = None

    match = abs(recorded_pnl - expected_pnl) < 0.01 if expected_pnl is not None else "?"

    print(f"Trade {tid}: {verdict} @ {entry}")
    print(f"  Target: {target} | Stop: {stop} | Status: {status}")
    print(f"  Recorded PnL: {recorded_pnl:+.2f} pts (${recorded_pnl*50:+,.0f})")
    print(f"  Expected PnL: {expected_pnl:+.2f} pts (${expected_pnl*50:+,.0f})")
    print(f"  {'OK' if match else 'MISMATCH'}")

    if not match:
        total_ok = False
        print(f"  DIFFERENCE: {recorded_pnl - expected_pnl:+.2f} pts")
    print()

# Check trade 8 more carefully - BEARISH entry 6776.75, stop 6795
# For a SHORT: loss = -(stop - entry) = -(6795 - 6776.75) = -18.25
# But recorded is -22.25. Where does -22.25 come from?
print("=== TRADE 8 DEEP DIVE ===")
print(f"BEARISH (short) @ 6776.75, stop 6795.0")
print(f"Expected stop loss: -(6795.0 - 6776.75) = {-(6795.0 - 6776.75):+.2f}")
print(f"Recorded: -22.25")
print(f"If price at close was 6799.0: loss = -(6799.0 - 6776.75) = {-(6799.0 - 6776.75):+.2f}")
print(f"If time-exit at some price X: -22.25 means X = 6776.75 + 22.25 = {6776.75 + 22.25}")
print()

# Check trade 10 - BEARISH entry 6822.25, stop 6831.0
print("=== TRADE 10 DEEP DIVE ===")
print(f"BEARISH (short) @ 6822.25, stop 6831.0")
print(f"Expected stop loss: -(6831.0 - 6822.25) = {-(6831.0 - 6822.25):+.2f}")
print(f"Recorded: -16.00")
print(f"If price at exit was X: -16.00 means X = 6822.25 + 16.00 = {6822.25 + 16.00}")
print(f"Log showed 'No path data, float PnL from current price: -15.25' at 14:21")
print(f"That implies price was 6822.25 + 15.25 = {6822.25 + 15.25} at 14:21")

print()
print("=== SUMMARY ===")
total = sum(t[6] for t in trades)
print(f"Total Realized: {total:+.2f} pts (${total*50:+,.0f})")
