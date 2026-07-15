# gpu_monitor.ps1 — Dual-GPU monitoring for win-gpu (RTX 3080 + RTX 3060 Ti)
# Usage: powershell -ExecutionPolicy Bypass -File H:\ashare-lab\scripts\gpu_monitor.ps1 [-Interval 30] [-LogPath H:\ashare-lab\logs\gpu_monitor.csv]
# Collects: GPU temp, VRAM, power draw, utilization for both GPUs + CPU + system RAM.
# Writes CSV log. Prints warnings to stderr when thresholds exceeded.

param(
    [int]$Interval = 30,
    [string]$LogPath = "H:\ashare-lab\logs\gpu_monitor.csv",
    [string]$HWiNFOCsv = "H:\tmp\hwinfo.csv"
)

# Thresholds
$GPU_TEMP_WARN = 85       # Celsius
$GPU_VRAM_PCT_WARN = 95   # percent of total
$GPU_POWER_3080_WARN = 350 # Watts (3080 TDP 320W, transient spikes 400-600W)
$GPU_POWER_3060TI_WARN = 200 # Watts (3060 Ti TDP 200W)
$TOTAL_POWER_WARN = 700   # Watts (850W PSU, 82% threshold)
$SYS_RAM_WARN_MB = 28000  # MB (32GB total, warn at 28GB)
$CPU_TDP_W = 65           # i5-12490F base TDP (PL1)
$CPU_MAX_W = 117          # i5-12490F max turbo (PL2)
$MISC_POWER_W = 80        # Motherboard + SSD + fans + RAM (fixed overhead)

# Ensure log directory exists
$logDir = Split-Path $LogPath -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Write CSV header if file doesn't exist or is empty
if (-not (Test-Path $LogPath) -or (Get-Item $LogPath).Length -eq 0) {
    "timestamp,gpu0_temp_c,gpu0_vram_used_mb,gpu0_vram_total_mb,gpu0_power_w,gpu0_util_pct,gpu1_temp_c,gpu1_vram_used_mb,gpu1_vram_total_mb,gpu1_power_w,gpu1_util_pct,cpu_util_pct,cpu_power_est_w,sys_ram_used_mb,sys_ram_total_mb,sys_power_est_w,warning" | Out-File -FilePath $LogPath -Encoding utf8
}

Write-Host "GPU Monitor started. Interval=${Interval}s, Log=$LogPath" -ForegroundColor Green
Write-Host "Thresholds: GPU temp>${GPU_TEMP_WARN}C, VRAM>${GPU_VRAM_PCT_WARN}%, Power 3080>${GPU_POWER_3080_WARN}W 3060Ti>${GPU_POWER_3060TI_WARN}W, Total>${TOTAL_POWER_WARN}W, RAM>${SYS_RAM_WARN_MB}MB" -ForegroundColor Yellow

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $warnings = @()

    # nvidia-smi query for both GPUs
    try {
        $smiOut = & nvidia-smi --query-gpu=index,temperature.gpu,memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader,nounits 2>$null
        $gpu0 = ($smiOut[0] -split ',\s*').Trim()
        $gpu1 = ($smiOut[1] -split ',\s*').Trim()

        $g0_temp = [int]$gpu0[1]
        $g0_vram_used = [int]$gpu0[2]
        $g0_vram_total = [int]$gpu0[3]
        $g0_power = [math]::Round([double]$gpu0[4], 1)
        $g0_util = [int]$gpu0[5]

        $g1_temp = [int]$gpu1[1]
        $g1_vram_used = [int]$gpu1[2]
        $g1_vram_total = [int]$gpu1[3]
        $g1_power = [math]::Round([double]$gpu1[4], 1)
        $g1_util = [int]$gpu1[5]

        # Temperature warnings
        if ($g0_temp -ge $GPU_TEMP_WARN) { $warnings += "GPU0 temp ${g0_temp}C" }
        if ($g1_temp -ge $GPU_TEMP_WARN) { $warnings += "GPU1 temp ${g1_temp}C" }

        # VRAM warnings
        if ($g0_vram_total -gt 0 -and ($g0_vram_used / $g0_vram_total * 100) -ge $GPU_VRAM_PCT_WARN) {
            $warnings += "GPU0 VRAM ${g0_vram_used}/${g0_vram_total}MB"
        }
        if ($g1_vram_total -gt 0 -and ($g1_vram_used / $g1_vram_total * 100) -ge $GPU_VRAM_PCT_WARN) {
            $warnings += "GPU1 VRAM ${g1_vram_used}/${g1_vram_total}MB"
        }

        # Power warnings
        if ($g0_power -ge $GPU_POWER_3080_WARN) { $warnings += "GPU0 power ${g0_power}W" }
        if ($g1_power -ge $GPU_POWER_3060TI_WARN) { $warnings += "GPU1 power ${g1_power}W" }
        $totalGpuPower = $g0_power + $g1_power
        if ($totalGpuPower -ge $TOTAL_POWER_WARN) { $warnings += "Total GPU power ${totalGpuPower}W" }

    } catch {
        $g0_temp = -1; $g0_vram_used = -1; $g0_vram_total = -1; $g0_power = -1; $g0_util = -1
        $g1_temp = -1; $g1_vram_used = -1; $g1_vram_total = -1; $g1_power = -1; $g1_util = -1
        $warnings += "nvidia-smi failed"
    }

    # System RAM + CPU utilization
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $sysTotalMB = [math]::Round($os.TotalVisibleMemorySize / 1024)
        $sysFreeMB = [math]::Round($os.FreePhysicalMemory / 1024)
        $sysUsedMB = $sysTotalMB - $sysFreeMB
        if ($sysUsedMB -ge $SYS_RAM_WARN_MB) { $warnings += "SysRAM ${sysUsedMB}/${sysTotalMB}MB" }
    } catch {
        $sysUsedMB = -1; $sysTotalMB = -1
    }

    # CPU utilization (average across all cores) + power
    try {
        $cpuLoad = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        $cpuUtil = [math]::Round($cpuLoad, 0)
    } catch {
        $cpuUtil = -1
    }

    # CPU power: prefer HWiNFO64 CSV (column 132 = CPU Package Power [W])
    $cpuPowerEst = -1
    if (Test-Path $HWiNFOCsv) {
        try {
            $lastLine = Get-Content $HWiNFOCsv -Tail 1 -ErrorAction Stop
            $fields = $lastLine.Split(',')
            if ($fields.Count -gt 132) {
                $raw = $fields[132].Trim('"')
                if ($raw -match '^[\d.]+$') { $cpuPowerEst = [math]::Round([double]$raw, 0) }
            }
        } catch {}
    }
    # Fallback: linear interpolation from CPU utilization
    if ($cpuPowerEst -lt 0 -and $cpuUtil -ge 0) {
        $cpuPowerEst = [math]::Round($CPU_TDP_W * 0.15 + ($CPU_MAX_W - $CPU_TDP_W * 0.15) * ($cpuUtil / 100), 0)
    }

    # Total system power estimate
    $totalSysPower = if ($g0_power -ge 0 -and $g1_power -ge 0 -and $cpuPowerEst -ge 0) {
        [math]::Round($g0_power + $g1_power + $cpuPowerEst + $MISC_POWER_W, 0)
    } else { -1 }
    if ($totalSysPower -ge $TOTAL_POWER_WARN) { $warnings += "System power ~${totalSysPower}W" }

    $warnStr = if ($warnings.Count -gt 0) { $warnings -join "; " } else { "" }

    # CSV line
    "$ts,$g0_temp,$g0_vram_used,$g0_vram_total,$g0_power,$g0_util,$g1_temp,$g1_vram_used,$g1_vram_total,$g1_power,$g1_util,$cpuUtil,$cpuPowerEst,$sysUsedMB,$sysTotalMB,$totalSysPower,$warnStr" |
        Out-File -FilePath $LogPath -Append -Encoding utf8

    # Console output (compact)
    $line = "$ts | 3080: ${g0_temp}C ${g0_vram_used}MB ${g0_power}W ${g0_util}% | 3060Ti: ${g1_temp}C ${g1_vram_used}MB ${g1_power}W ${g1_util}% | CPU: ${cpuUtil}% ~${cpuPowerEst}W | RAM: ${sysUsedMB}/${sysTotalMB}MB | SYS: ~${totalSysPower}W"
    if ($warnings.Count -gt 0) {
        Write-Host "$line *** WARNING: $warnStr" -ForegroundColor Red
    } else {
        Write-Host $line
    }

    Start-Sleep -Seconds $Interval
}
