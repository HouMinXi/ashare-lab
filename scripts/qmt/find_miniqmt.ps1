# Find miniQMT executable
$qmtRoot = (Get-ChildItem "H:\" -Directory -Filter "*QMT*").FullName
Write-Host "QMT root: $qmtRoot"

# Common miniQMT locations
$candidates = @(
    "$qmtRoot\bin.x64\miniQMT.exe",
    "$qmtRoot\bin.x64\XtMiniQmt.exe",
    "$qmtRoot\bin.x64\XtQuant.exe",
    "$qmtRoot\bin.x64\pythonw.exe",
    "$qmtRoot\mpython\miniQMT.exe"
)
foreach ($c in $candidates) {
    if (Test-Path $c) { Write-Host "FOUND: $c" }
}

# Search for mini/qmt executables in bin.x64
Write-Host "`n=== bin.x64 *.exe ==="
Get-ChildItem "$qmtRoot\bin.x64" -Filter "*.exe" | ForEach-Object {
    Write-Host "  $($_.Name)"
}

# Check if there's a start script
Write-Host "`n=== start/bat/cmd files ==="
Get-ChildItem $qmtRoot -Filter "*.bat" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.Name)"
}
Get-ChildItem $qmtRoot -Filter "*.cmd" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.Name)"
}
