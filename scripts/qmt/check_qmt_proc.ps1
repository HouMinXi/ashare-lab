Get-Process | Where-Object {
    $_.ProcessName -match "QMT|qmt|xuntou|guojin|mini|trader|国金"
} | Select-Object ProcessName, Id, Path | Format-Table -AutoSize
