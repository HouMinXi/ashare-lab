# R5 Solution: Database Corruption Silent Data Loss

## Problem Statement

SQLite `paper.db` (192KB) has no integrity check before backup. If the database becomes corrupted:

1. `init_schema()` runs `CREATE TABLE IF NOT EXISTS` - tables appear to exist but data is gone
2. `hot_backup()` overwrites daily backup with the corrupted/empty database
3. Historical NAV, positions, trades, signals all permanently lost

## Root Cause Analysis

From code analysis of `ledger.py` and `pipeline.py`:

```python
# pipeline.py line 492 - init_schema runs every pipeline run
ctx.conn = get_connection(ctx.db_path)
init_schema(ctx.conn)  # No integrity check first!

# pipeline.py line 1493 - backup happens after settle
backup_path = PROJECT_ROOT / "backups" / f"paper_{ctx.trade_date}.db"
hot_backup(ctx.db_path, backup_path)  # Backs up whatever is there

# ledger.py line 672 - hot_backup has no integrity check
def hot_backup(db_path: Path, backup_path: Path) -> None:
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)  # Copies corrupted data if source is corrupt
    finally:
        dst.close()
        src.close()
```

## Solution Architecture

### Layer 1: Integrity Check (Prevention)

Add `check_db_integrity()` function to `ledger.py`:

```python
import hashlib
from dataclasses import dataclass
from datetime import date

@dataclass
class IntegrityResult:
    """Result of SQLite integrity check."""
    is_healthy: bool
    detail: str
    check_duration_ms: float

@dataclass
class BackupResult:
    """Result of backup operation."""
    success: bool
    backup_path: Path
    sha256: str
    size_bytes: int
    integrity_verified: bool
    error: str | None = None


def check_db_integrity(db_path: Path) -> IntegrityResult:
    """Run PRAGMA integrity_check on database using separate connection.

    Uses a short-lived connection to avoid interfering with the main
    application connection. Safe to run during normal operation in WAL mode.

    Returns:
        IntegrityResult with is_healthy=True if database passes check.
    """
    import time

    start = time.monotonic()
    try:
        # Separate connection - doesn't interfere with main connection
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()

        duration_ms = (time.monotonic() - start) * 1000
        is_healthy = result is not None and result[0] == "ok"
        detail = result[0] if result else "no result"

        return IntegrityResult(
            is_healthy=is_healthy,
            detail=detail,
            check_duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        return IntegrityResult(
            is_healthy=False,
            detail=f"connection error: {e}",
            check_duration_ms=duration_ms,
        )


def attempt_wal_repair(db_path: Path) -> tuple[bool, str]:
    """Attempt to repair corruption via WAL checkpoint.

    PRAGMA wal_checkpoint(TRUNCATE) flushes the WAL file into the
    main database and truncates it. This fixes most WAL-related
    corruption caused by interrupted writes.

    Returns:
        Tuple of (repair_succeeded, detail_message).
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except Exception as e:
        return False, f"WAL checkpoint failed: {e}"

    # Verify repair worked
    result = check_db_integrity(db_path)
    if result.is_healthy:
        return True, "WAL checkpoint repair successful"
    else:
        return False, f"WAL checkpoint did not fix corruption: {result.detail}"
```

### Layer 2: Safe Backup with Integrity Gating

Modify `hot_backup()` to run integrity check first and compute SHA-256:

```python
def hot_backup(db_path: Path, backup_path: Path, verify: bool = True) -> BackupResult:
    """Create a consistent backup using the sqlite3 backup API.

    Runs integrity check BEFORE backup to prevent backing up corrupted data.
    Computes SHA-256 hash of backup file for verification.

    Args:
        db_path: Path to source database.
        backup_path: Path to backup destination.
        verify: If True, verify backup integrity after creation.

    Returns:
        BackupResult with success status and SHA-256 hash.
    """
    # Step 1: Integrity check BEFORE backup
    integrity = check_db_integrity(db_path)
    if not integrity.is_healthy:
        # Attempt WAL repair
        repaired, repair_msg = attempt_wal_repair(db_path)
        if not repaired:
            return BackupResult(
                success=False,
                backup_path=backup_path,
                sha256="",
                size_bytes=0,
                integrity_verified=False,
                error=f"Source database corrupt: {integrity.detail}. {repair_msg}",
            )
        # Re-check after repair
        integrity = check_db_integrity(db_path)
        if not integrity.is_healthy:
            return BackupResult(
                success=False,
                backup_path=backup_path,
                sha256="",
                size_bytes=0,
                integrity_verified=False,
                error=f"Source database still corrupt after repair: {integrity.detail}",
            )

    # Step 2: Create backup using atomic API
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # Step 3: Compute SHA-256 hash
    sha256 = _compute_sha256(backup_path)
    size_bytes = backup_path.stat().st_size

    # Step 4: Verify backup integrity (optional but recommended)
    integrity_verified = False
    if verify:
        backup_integrity = check_db_integrity(backup_path)
        integrity_verified = backup_integrity.is_healthy
        if not integrity_verified:
            return BackupResult(
                success=False,
                backup_path=backup_path,
                sha256=sha256,
                size_bytes=size_bytes,
                integrity_verified=False,
                error=f"Backup verification failed: {backup_integrity.detail}",
            )

    return BackupResult(
        success=True,
        backup_path=backup_path,
        sha256=sha256,
        size_bytes=size_bytes,
        integrity_verified=integrity_verified,
    )


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_backup(backup_path: Path) -> IntegrityResult:
    """Verify a backup file is intact and not corrupted.

    Opens the backup file and runs PRAGMA integrity_check.
    Use this to verify backups before restoring from them.

    Returns:
        IntegrityResult with is_healthy=True if backup is valid.
    """
    if not backup_path.exists():
        return IntegrityResult(
            is_healthy=False,
            detail=f"Backup file not found: {backup_path}",
            check_duration_ms=0.0,
        )
    return check_db_integrity(backup_path)
```

### Layer 3: Golden Backup (Immutable Recovery Point)

Add monthly immutable backup that never gets overwritten:

```python
def golden_backup(db_path: Path, backup_dir: Path) -> BackupResult:
    """Create monthly immutable backup that is never deleted.

    Golden backups provide a recovery point that survives the normal
    backup rotation. Only one golden backup per month (named by month).

    The file is set to read-only (chmod 444) to prevent accidental
    deletion or overwriting.

    Args:
        db_path: Path to source database.
        backup_dir: Base backup directory (e.g., PROJECT_ROOT / "backups").

    Returns:
        BackupResult with success status.
    """
    today = date.today()
    golden_dir = backup_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    # Name by month - only one golden backup per month
    golden_path = golden_dir / f"paper_golden_{today.strftime('%Y-%m')}.db"

    # Check if golden backup already exists for this month
    if golden_path.exists():
        # Verify existing golden backup is still good
        existing_check = verify_backup(golden_path)
        if existing_check.is_healthy:
            return BackupResult(
                success=True,
                backup_path=golden_path,
                sha256=_compute_sha256(golden_path),
                size_bytes=golden_path.stat().st_size,
                integrity_verified=True,
                error=None,  # Existing golden backup is fine
            )
        # Existing golden backup is corrupt - replace it
        golden_path.unlink()

    # Create golden backup with integrity verification
    result = hot_backup(db_path, golden_path, verify=True)

    # Set read-only permissions to prevent accidental modification
    if result.success:
        golden_path.chmod(0o444)  # Read-only for all

    return result


def is_first_trading_day_of_month() -> bool:
    """Check if today is likely the first trading day of the month.

    Simple heuristic: check if we haven't created a golden backup this month.
    More robust than checking calendar dates (handles weekends/holidays).
    """
    config = load_config()
    db_path = PROJECT_ROOT / config["paper"]["db_path"]
    backup_dir = PROJECT_ROOT / "backups"
    golden_dir = backup_dir / "golden"

    if not golden_dir.exists():
        return True  # No golden backups yet

    # Check if golden backup exists for current month
    current_month = date.today().strftime("%Y-%m")
    golden_path = golden_dir / f"paper_golden_{current_month}.db"
    return not golden_path.exists()
```

### Layer 4: Modified Pipeline Integration

Update `_step12_backup_and_finalize()` in `pipeline.py`:

```python
def _step12_backup_and_finalize(ctx: DailyRunContext) -> None:
    """Commit, hot-backup, cleanup old backups, record settled run."""
    ctx.conn.commit()

    # Daily backup with integrity check
    backup_path = PROJECT_ROOT / "backups" / f"paper_{ctx.trade_date}.db"
    backup_result = hot_backup(ctx.db_path, backup_path, verify=True)

    if not backup_result.success:
        logger.error(
            "BACKUP FAILED for %s: %s",
            ctx.trade_date,
            backup_result.error,
        )
        # Alert operator - this is critical
        _alert_backup_failure(ctx.trade_date, backup_result.error)
        # Don't fail the pipeline - data is still in the main DB
        # But operator must investigate immediately

    # Golden backup on first trading day of month
    if is_first_trading_day_of_month():
        golden_result = golden_backup(ctx.db_path, PROJECT_ROOT / "backups")
        if golden_result.success:
            logger.info(
                "Golden backup created: %s (SHA-256: %s)",
                golden_result.backup_path,
                golden_result.sha256[:16],  # First 16 chars for logging
            )
        else:
            logger.error("Golden backup FAILED: %s", golden_result.error)

    # Cleanup old backups (preserves golden backups)
    cleanup_old_backups(PROJECT_ROOT / "backups", ctx.paper_cfg["backup_retention_days"])

    record_run(ctx.conn, ctx.trade_date, "settled")
    ctx.conn.commit()


def _alert_backup_failure(trade_date: str, error: str) -> None:
    """Alert operator about backup failure.

    Uses the same alert mechanism as crash alerts.
    """
    try:
        import subprocess
        import sys
        subprocess.Popen(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "alert.py"),
             "1", "backup_failure", "0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # Alert is best-effort
    logger.critical("BACKUP FAILURE ALERT: %s - %s", trade_date, error)
```

### Layer 5: Recovery Procedure

Add recovery functions to `ledger.py`:

```python
def restore_from_backup(backup_path: Path, db_path: Path) -> tuple[bool, str]:
    """Restore database from a backup file.

    Steps:
    1. Verify backup integrity
    2. Backup current database (even if corrupt)
    3. Copy backup to database location
    4. Verify restored database

    Args:
        backup_path: Path to backup file to restore from.
        db_path: Path to database to restore to.

    Returns:
        Tuple of (success, message).
    """
    # Step 1: Verify backup integrity
    integrity = verify_backup(backup_path)
    if not integrity.is_healthy:
        return False, f"Backup is corrupt: {integrity.detail}"

    # Step 2: Backup current database (even if corrupt, for forensics)
    if db_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt_backup = db_path.parent / f"paper_corrupt_{timestamp}.db"
        try:
            shutil.copy2(db_path, corrupt_backup)
        except Exception:
            pass  # Best effort

    # Step 3: Copy backup to database location
    try:
        shutil.copy2(backup_path, db_path)
    except Exception as e:
        return False, f"Failed to copy backup: {e}"

    # Step 4: Verify restored database
    restored_integrity = check_db_integrity(db_path)
    if not restored_integrity.is_healthy:
        return False, f"Restored database is corrupt: {restored_integrity.detail}"

    return True, f"Database restored successfully from {backup_path}"


def find_best_backup(backup_dir: Path, target_date: str | None = None) -> Path | None:
    """Find the best available backup for recovery.

    Priority:
    1. Daily backup for target_date (if specified)
    2. Most recent daily backup that passes integrity check
    3. Most recent golden backup

    Args:
        backup_dir: Base backup directory.
        target_date: Optional specific date to restore (YYYY-MM-DD).

    Returns:
        Path to best backup, or None if no valid backup found.
    """
    # Try specific date first
    if target_date:
        daily_path = backup_dir / f"paper_{target_date}.db"
        if daily_path.exists():
            integrity = verify_backup(daily_path)
            if integrity.is_healthy:
                return daily_path

    # Try most recent daily backups (newest first)
    daily_backups = sorted(
        backup_dir.glob("paper_????-??-??.db"),
        reverse=True,
    )
    for backup in daily_backups:
        integrity = verify_backup(backup)
        if integrity.is_healthy:
            return backup

    # Fall back to golden backups
    golden_dir = backup_dir / "golden"
    if golden_dir.exists():
        golden_backups = sorted(
            golden_dir.glob("paper_golden_????-??.db"),
            reverse=True,
        )
        for backup in golden_backups:
            integrity = verify_backup(backup)
            if integrity.is_healthy:
                return backup

    return None
```

### Layer 6: Modified Cleanup (Preserves Golden Backups)

Update `cleanup_old_backups()` to preserve golden backups:

```python
def cleanup_old_backups(backup_dir: Path, retention_days: int) -> None:
    """Delete backup files older than retention_days.

    Golden backups in backup_dir/golden/ are NEVER deleted.
    Only daily backups in the main backup_dir are subject to rotation.
    """
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=retention_days)
    golden_dir = backup_dir / "golden"

    for f in backup_dir.glob("paper_*.db"):
        # Skip golden backups directory
        if golden_dir and f.parent == golden_dir:
            continue

        # Expected filename: paper_YYYY-MM-DD.db
        stem = f.stem  # paper_YYYY-MM-DD
        try:
            date_str = stem[len("paper_"):]
            file_date = date.fromisoformat(date_str)
        except (ValueError, IndexError):
            continue
        if file_date < cutoff:
            f.unlink()
```

## Implementation Checklist

- [ ] Add `IntegrityResult` and `BackupResult` dataclasses to `ledger.py`
- [ ] Add `check_db_integrity()` function to `ledger.py`
- [ ] Add `attempt_wal_repair()` function to `ledger.py`
- [ ] Add `_compute_sha256()` helper to `ledger.py`
- [ ] Add `verify_backup()` function to `ledger.py`
- [ ] Modify `hot_backup()` to run integrity check and compute hash
- [ ] Add `golden_backup()` function to `ledger.py`
- [ ] Add `is_first_trading_day_of_month()` helper to `ledger.py`
- [ ] Add `restore_from_backup()` function to `ledger.py`
- [ ] Add `find_best_backup()` function to `ledger.py`
- [ ] Update `cleanup_old_backups()` to preserve golden backups
- [ ] Update `_step12_backup_and_finalize()` in `pipeline.py`
- [ ] Add `_alert_backup_failure()` helper to `pipeline.py`
- [ ] Test: corrupt database, verify detection
- [ ] Test: golden backup creation on first of month
- [ ] Test: recovery from golden backup

## Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| Integrity check functions | ~60 | Low |
| Safe backup with hash | ~40 | Low |
| Golden backup logic | ~30 | Low |
| Pipeline integration | ~20 | Low |
| Recovery functions | ~50 | Medium |
| Cleanup preservation | ~10 | Low |
| **Total** | **~210** | **Low-Medium** |

**Time estimate**: 2-3 hours implementation + 1-2 hours testing

## Recovery Runbook

### Scenario 1: Corruption Detected During Pipeline

```
1. Pipeline detects corruption via integrity check
2. Backup is skipped (corrupt data not backed up)
3. Alert sent to operator
4. Operator investigates:
   - Check logs for corruption details
   - Run: sqlite3 paper.db "PRAGMA integrity_check;"
   - If WAL-related: attempt WAL checkpoint repair
   - If unrecoverable: proceed to Scenario 2
```

### Scenario 2: Database Unrecoverable, Need Restore

```
1. Stop the pipeline (prevent further damage)
2. Find best backup:
   python -c "
   from ashare_lab.paper.ledger import find_best_backup
   from pathlib import Path
   best = find_best_backup(Path('backups'))
   print(f'Restore from: {best}')
   "

3. Verify backup integrity:
   sqlite3 backups/paper_2026-07-15.db "PRAGMA integrity_check;"
   # Must return "ok"

4. Restore:
   python -c "
   from ashare_lab.paper.ledger import restore_from_backup
   from pathlib import Path
   success, msg = restore_from_backup(
       Path('backups/paper_2026-07-15.db'),
       Path('paper.db')
   )
   print(msg)
   "

5. Verify restored database:
   sqlite3 paper.db "PRAGMA integrity_check;"
   sqlite3 paper.db "SELECT COUNT(*) FROM nav;"

6. Re-run pipeline for any missing days:
   python -m ashare_lab.paper backfill 2026-07-15 2026-07-15
```

### Scenario 3: All Daily Backups Corrupt (Use Golden Backup)

```
1. List golden backups:
   ls -la backups/golden/

2. Find most recent golden backup:
   python -c "
   from ashare_lab.paper.ledger import verify_backup
   from pathlib import Path
   for p in sorted(Path('backups/golden').glob('*.db'), reverse=True):
       r = verify_backup(p)
       if r.is_healthy:
           print(f'Valid golden backup: {p}')
           break
   "

3. Restore from golden backup (same as Scenario 2, step 4)

4. Re-run pipeline for all days since golden backup:
   python -m ashare_lab.paper backfill 2026-07-01 2026-07-15
```

## Testing Strategy

### Test 1: Corruption Detection

```python
def test_corruption_detection():
    """Verify integrity check detects corruption."""
    import tempfile
    import os

    # Create a corrupt database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        f.write(b'corrupt data')
        corrupt_path = Path(f.name)

    try:
        result = check_db_integrity(corrupt_path)
        assert not result.is_healthy
        assert "error" in result.detail.lower() or "malformed" in result.detail.lower()
    finally:
        os.unlink(corrupt_path)
```

### Test 2: Golden Backup Preservation

```python
def test_golden_backup_preserved():
    """Verify golden backups are not deleted by cleanup."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir)
        golden_dir = backup_dir / "golden"
        golden_dir.mkdir()

        # Create a golden backup
        golden_file = golden_dir / "paper_golden_2026-07.db"
        golden_file.write_text("golden backup data")

        # Create an old daily backup
        old_daily = backup_dir / "paper_2026-01-01.db"
        old_daily.write_text("old daily backup")

        # Run cleanup with 7-day retention
        cleanup_old_backups(backup_dir, retention_days=7)

        # Golden backup should still exist
        assert golden_file.exists()
        # Old daily backup should be deleted
        assert not old_daily.exists()
```

### Test 3: Backup Integrity Verification

```python
def test_backup_integrity_gating():
    """Verify backup fails when source is corrupt."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        backup_path = Path(tmpdir) / "backup.db"

        # Create corrupt database
        db_path.write_bytes(b'corrupt')

        result = hot_backup(db_path, backup_path, verify=True)
        assert not result.success
        assert "corrupt" in result.error.lower()
```

## Key Design Decisions

1. **Integrity check BEFORE backup**: Prevents backing up corrupted data. This is the critical fix for R5.

2. **Golden backup never deleted**: Monthly immutable snapshots survive the normal rotation. Provides recovery even if corruption persists for weeks.

3. **SHA-256 hash verification**: Detects silent corruption during backup transfer or storage.

4. **Backup verification opens the backup**: Catches corruption that occurred during the backup process itself.

5. **WAL repair attempt**: Minor corruption (WAL-related) can often be fixed automatically before giving up.

6. **Operator alerting**: Backup failures trigger immediate alerts so humans can investigate.

7. **Read-only golden backups**: chmod 444 prevents accidental deletion or overwriting.

## References

- SQLite integrity_check documentation: https://sqlite.org/pragma.html#pragma_integrity_check
- SQLite corruption prevention: https://sqlite.org/howtocorrupt.html
- Atomic SQLite backups: https://greyforge.tech/chronicles/sqlite-checkpoint-atomic-backups
- Production SQLite patterns: https://llmbestpractices.com/backend/sqlite
- Trading system DR: https://theplanet.cloud/backup-and-dr-architectures-for-commodity-trading-platforms
