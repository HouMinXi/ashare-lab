"""Pre-populate baostock industry cache. Checkpoint-based: processes up to
BATCH_SIZE symbols per run, saves progress, exits. Timer reruns until done.
SIGALRM per-query timeout prevents baostock server hangs."""
import baostock as bs, csv, time, signal, sys
from pathlib import Path

CACHE_FILE = Path("/home/houminxi/code/ashare-lab/data/baostock_cache/industry.csv")
CHECKPOINT = Path("/home/houminxi/code/ashare-lab/data/baostock_cache/.industry_checkpoint")
BATCH_SIZE = 400
QUERY_TIMEOUT = 15  # seconds per query
TRADE_DATE = "2026-07-10"

class QueryTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise QueryTimeout("query timed out")

CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load already-done codes from cache
done_codes = set()
if CACHE_FILE.exists():
    with open(CACHE_FILE, newline="") as f:
        for row in csv.DictReader(f):
            done_codes.add(row["code"])

# Load checkpoint (last index processed)
start_idx = 0
if CHECKPOINT.exists():
    start_idx = int(CHECKPOINT.read_text().strip())

# Get universe
import qlib
qlib.init(provider_uri="/home/houminxi/.qlib/qlib_data/cn_data", region="cn")
from qlib.data import D
instruments = D.instruments()
all_syms = [str(s) for s in D.list_instruments(instruments=instruments, as_list=True)]

# Filter: skip BJ, skip already in cache
def _to_bs_code(sym):
    code = sym.lower()
    if len(code) == 6:
        pfx = "sh" if code.startswith("6") else "sz"
        return pfx + "." + code
    elif not code.startswith(("sh.", "sz.")):
        pfx = "sh" if sym.startswith(("SH", "6")) else "sz"
        return pfx + "." + sym[-6:]
    return code

remaining = []
for sym in all_syms[start_idx:]:
    if sym.upper().startswith("BJ"):
        continue
    bs_code = _to_bs_code(sym)
    if bs_code not in done_codes:
        remaining.append((sym, bs_code))

print(f"total={len(all_syms)} start={start_idx} remaining={len(remaining)} cached={len(done_codes)}", flush=True)
if not remaining:
    print("DONE: all symbols cached")
    sys.exit(0)

bs.login()

new_rows = []
t0 = time.time()
signal.signal(signal.SIGALRM, _alarm_handler)
processed = 0
timeouts = 0

for sym, bs_code in remaining[:BATCH_SIZE]:
    try:
        signal.alarm(QUERY_TIMEOUT)
        rs = bs.query_stock_industry(code=bs_code, date=TRADE_DATE)
        signal.alarm(0)
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            if len(row) > 3 and row[3]:
                new_rows.append({"date": TRADE_DATE, "code": bs_code, "industry": row[3]})
                done_codes.add(bs_code)
    except QueryTimeout:
        signal.alarm(0)
        timeouts += 1
    except Exception:
        signal.alarm(0)
    processed += 1
    if processed % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {processed}/{min(BATCH_SIZE, len(remaining))} ({len(new_rows)} matched, {timeouts} timeout) {elapsed:.0f}s", flush=True)

signal.alarm(0)
bs.logout()

# Append to cache
if new_rows:
    write_header = not CACHE_FILE.exists()
    try:
        with open(CACHE_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "code", "industry"])
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
    except OSError as exc:
        print(f"cache write failed: {exc}", file=sys.stderr)

# Update checkpoint
new_idx = start_idx + len(all_syms[start_idx:start_idx + BATCH_SIZE])
CHECKPOINT.write_text(str(new_idx))

elapsed = time.time() - t0
print(f"BATCH DONE: {processed} processed, {len(new_rows)} new, {timeouts} timeout, {elapsed:.0f}s", flush=True)
print(f"checkpoint: {new_idx}/{len(all_syms)}", flush=True)
