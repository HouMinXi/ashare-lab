# ADR-0002: X500-to-Windows wiring pattern

**Date**: 2026-06-28
**Status**: accepted
**Deciders**: Minxi Hou
**Supersedes**: clarifies Phase 7-8 connection mechanism in ADR-0001

## Context

ashare-lab runs on two machines in the same LAN (192.168.100.x):

- X500 (Linux, Fedora 44, 7x24, no GPU): data pipeline, paper engine,
  sentiment veto, WeChat report, SQLite ledger
- GPU box (Windows 11, RTX 3080 20GB, admin@192.168.100.11): model
  training, model inference, AND broker execution via Windows client
  (no Linux broker API exists)

Both training and trading are Windows-only. X500 is the brain (strategy,
risk, monitoring) but cannot execute either compute or orders directly.
The question is how they communicate.

The machines already have SSH/SCP connectivity -- GPU training results
have been transferred via SCP since Phase 2. Windows 10+ includes a
built-in OpenSSH server.

## Decision

Use SSH/SCP file exchange over LAN as the sole communication channel.
No REST API, no message queue, no shared filesystem. The protocol:

### Daily flow (Phase 7+8 combined)

```
15:00  Market closes
15:30  [Windows] Task Scheduler triggers predict.py
         -> writes predictions/{date}.parquet locally
15:35  [Windows] scp predictions/{date}.parquet x500:code/ashare-lab/predictions/
16:00  [X500] systemd timer triggers paper pipeline
         -> reads predictions/{date}.parquet
         -> runs sentiment veto (eastmoney + deepseek)
         -> runs risk checks
         -> generates paper orders
         -> writes orders/{date}.json
         -> sends WeChat report
16:05  [X500] scp orders/{date}.json gpu-box:ashare-lab/orders/
09:25  [Windows] QMT agent reads orders/{date}.json
         -> submits at 09:30 open
09:35  [Windows] QMT agent writes fills/{date}.json
         -> scp fills/{date}.json x500:code/ashare-lab/fills/
16:10  [X500] next day pipeline reconciles paper vs real fills
```

### File contracts

| File | Producer | Consumer | Format |
|------|----------|----------|--------|
| predictions/{date}.parquet | Windows predict.py | X500 pipeline | parquet (symbol, score columns) |
| orders/{date}.json | X500 pipeline | Windows QMT agent | JSON [{side, symbol, qty, price_limit}] |
| fills/{date}.json | Windows QMT agent | X500 reconciliation | JSON [{symbol, side, qty, fill_price, fill_time}] |

### Health checks

- [X500] If predictions/{date}.parquet missing by 16:00, send WeChat
  alert and skip trading (paper-only day)
- [Windows] If orders/{date}.json missing by 09:25, skip real orders
  (no orders = no trades, safe default)
- [X500] If fills/{date}.json missing by next pipeline run, log
  "reconciliation skipped" (non-blocking)

### Why X500 stays in the loop

1. **7x24 uptime**: Windows reboots for updates, X500 does not
2. **Risk gate**: sentiment veto and drawdown kill switch run on X500.
   Windows QMT agent is a dumb executor -- it submits whatever
   orders/{date}.json says. Risk decisions live on the reliable machine.
3. **Single ledger**: SQLite on X500 is the source of truth
4. **Report independence**: WeChat reports send even if Windows is off
5. **Fail-safe default**: missing file = no action. Trading without
   risk checks is structurally impossible.

## Alternatives Considered

### Alternative A: REST API on Windows
- **Pros**: Real-time communication, can query status
- **Cons**: Requires running server, firewall config, more code
- **Why not**: Daily batch does not need real-time. SCP is simpler
  and already proven in this project.

### Alternative B: Shared filesystem (SMB mount)
- **Pros**: Both machines see the same files
- **Cons**: SQLite over SMB is unreliable (locking), adds network
  dependency to every file read
- **Why not**: SQLite explicitly warns against network filesystems.

### Alternative C: Move everything to Windows
- **Pros**: No wiring needed, single machine
- **Cons**: Windows Update kills 7x24, GPU box is multi-tenant,
  no separation between risk gate and executor
- **Why not**: Reliability regression.

### Alternative D: Cloud Windows VPS
- **Pros**: Dedicated 7x24 Windows
- **Cons**: Monthly cost, regulatory gray area, extra machine
- **Why not**: GPU box is on the same LAN. Revisit if it proves
  unreliable during trading hours.

## Consequences

### Positive
- Zero new infrastructure (SSH/SCP already works)
- Each file contract independently testable
- Fail-safe: missing file = no action
- Debuggable: every exchange is a file on disk

### Negative
- Batch latency (30 min between close and pipeline). Not suitable
  for intraday (out of scope -- daily frequency only)
- Windows Task Scheduler less reliable than systemd. Mitigation:
  X500 health check alerts on missing prediction.
- Both machines must be up during 15:30-16:10 window. Mitigation:
  both on UPS, incomplete days recoverable via backfill.

### Risks
- **Windows SSH server disabled by update**: pin OpenSSH Server feature
- **File format drift**: version field in JSON from day one
- **Clock skew**: both on same LAN with NTP, filenames use trade date
