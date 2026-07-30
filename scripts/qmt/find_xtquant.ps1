# Find xtquant module
Get-ChildItem "H:\" -Directory | ForEach-Object {
    $xt = Join-Path $_.FullName "bin\xtquant"
    if (Test-Path $xt) { Write-Host "FOUND: $xt" }
    $xt2 = Join-Path $_.FullName "xtquant"
    if (Test-Path $xt2) { Write-Host "FOUND: $xt2" }
}

# Also check subdirectories
Get-ChildItem "H:\" -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match "xtquant|bin|lib") {
            Write-Host "DIR: $($_.FullName)"
        }
    }
}
