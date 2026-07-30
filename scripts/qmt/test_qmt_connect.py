import sys, os
# Add QMT's xtquant to path
qmt_root = r"H:\国金证券QMT量化终端"
xt_path = os.path.join(qmt_root, "bin.x64", "Lib", "site-packages")
sys.path.insert(0, xt_path)

from xtquant import xttrader, xtdata

session_id = 10086
mini_path = os.path.join(qmt_root, "bin.x64")

print(f"QMT root: {qmt_root}")
print(f"xtquant path: {xt_path}")
print(f"mini_path: {mini_path}")
print(f"session_id: {session_id}")

# Create trader
t = xttrader.XtQuantTrader(mini_path, session_id)
print(f"XtQuantTrader created")

# Try connect
result = t.connect()
print(f"connect() returned: {result}")

if result == 0:
    print("SUCCESS: Connected to QMT!")

    # Query account status
    try:
        accs = t.query_stock_accounts()
        print(f"Accounts: {accs}")
    except Exception as e:
        print(f"query_stock_accounts error: {e}")

    # Query positions
    try:
        positions = t.query_stock_positions(account_id="8890836643")
        print(f"Positions: {positions}")
    except Exception as e:
        print(f"query_stock_positions error: {e}")

    # Query asset
    try:
        asset = t.query_stock_asset(account_id="8890836643")
        print(f"Asset: {asset}")
    except Exception as e:
        print(f"query_stock_asset error: {e}")
else:
    print(f"FAILED: connect returned {result}")
    print("Possible reasons:")
    print("  - QMT client not running")
    print("  - Wrong path")
    print("  - Session ID conflict")

# Disconnect
try:
    t.disconnect()
    print("Disconnected")
except:
    pass
