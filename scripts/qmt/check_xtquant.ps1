# Check xtquant in QMT bin.x64
$qmt = Get-ChildItem "H:\" -Directory -Filter "*QMT*"
if ($qmt) {
    $bin = Join-Path $qmt.FullName "bin.x64"
    Write-Host "QMT bin: $bin"
    Get-ChildItem $bin -Filter "xtquant*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "FOUND: $($_.FullName)"
    }
    # Check if xtquant is a Python package
    $xt = Join-Path $bin "xtquant"
    if (Test-Path $xt) {
        Write-Host "xtquant package dir: $xt"
        Get-ChildItem $xt -Filter "*.py" | Select-Object -First 5 | ForEach-Object {
            Write-Host "  PY: $($_.Name)"
        }
    }
}

# Also check iQuant
$iq = Get-ChildItem "H:\" -Directory -Filter "*iQuant*"
if ($iq) {
    $bin = Join-Path $iq.FullName "bin.x64"
    Write-Host "`niQuant bin: $bin"
    Get-ChildItem $bin -Filter "xtquant*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "FOUND: $($_.FullName)"
    }
}
