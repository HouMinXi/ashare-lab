# Check QMT's built-in Python
$qmtRoot = (Get-ChildItem "H:\" -Directory -Filter "*QMT*").FullName
$mpy = "$qmtRoot\mpython"
Write-Host "mpython dir: $mpy"
if (Test-Path $mpy) {
    Get-ChildItem $mpy -Filter "python*" | ForEach-Object {
        Write-Host "  $($_.Name)"
    }
    # Check Python version
    $pyExe = Get-ChildItem $mpy -Filter "python.exe" -Recurse | Select-Object -First 1
    if ($pyExe) {
        Write-Host "Python exe: $($pyExe.FullName)"
        & $pyExe.FullName --version
    }
}
