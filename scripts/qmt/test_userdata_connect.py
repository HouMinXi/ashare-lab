import sys, os
qmt_root = r"H:\国金证券QMT量化终端"
xt_path = os.path.join(qmt_root, "bin.x64", "Lib", "site-packages")
sys.path.insert(0, xt_path)

from xtquant import xttrader

# Try userdata path
userdata_path = os.path.join(qmt_root, "userdata")
print(f"Trying userdata path: {userdata_path}")

t = xttrader.XtQuantTrader(userdata_path, 10086)
result = t.connect()
print(f"connect() = {result}")

if result == 0:
    print("SUCCESS!")
    try:
        accs = t.query_stock_accounts()
        print(f"Accounts: {accs}")
    except Exception as e:
        print(f"query error: {e}")
else:
    # Try with callback
    print("\nTrying with callback...")
    class MyCallback(xttrader.XtQuantTraderCallback):
        def on_disconnected(self):
            print("  disconnected!")
        def on_stock_order(self, order):
            print(f"  order: {order}")
        def on_order_error(self, order_error):
            print(f"  order_error: {order_error}")
        def on_stock_trade(self, trade):
            print(f"  trade: {trade}")
        def on_order_stock_async_response(self, response):
            print(f"  async response: {response}")

    cb = MyCallback()
    t2 = xttrader.XtQuantTrader(userdata_path, 10087, cb)
    result2 = t2.connect()
    print(f"connect() with callback = {result2}")
