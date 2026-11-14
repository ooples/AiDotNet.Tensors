# Waits until no other process is loading the GPU this job is about to measure, up to -MaxWaitMinutes.
#
# The self-hosted CUDA runners share their card with research runs on the same machine. On 2026-11-13 another
# process held 13.5 to 19.5 GB of the RTX 3090's 24 GB at 100 % utilisation through a nightly run: the suites spilled
# into WDDM system-memory fallback and ran past the watchdog, and device-wide measurements (free memory, timings) read
# the other process's activity as ours. The runner's service account cannot list another user's GPU processes, so this
# judges by the device totals before the job starts any GPU work of its own: busy while memory in use exceeds
# -MaxBusyMemoryGb or the mean utilisation over a few samples reaches -MaxBusyUtilizationPercent.
#
# On timeout the job proceeds (a nightly run is better late and annotated than missing) with a warning naming what was
# seen, so a failure under load is attributable. Every wait is written to the job summary.
param(
    [int] $MaxWaitMinutes = 30,
    [double] $MaxBusyMemoryGb = 3,
    [int] $MaxBusyUtilizationPercent = 40,
    [string] $Label = 'GPU work'
)

function Read-Gpu {
    $samples = 1..3 | ForEach-Object {
        if ($_ -gt 1) { Start-Sleep -Seconds 5 }
        # NVSMI_GPU (set by the workflow's GPU step from the runner's CUDA_VISIBLE_DEVICES pin) selects this runner's
        # card on a multi-GPU machine; otherwise the first line is the only card.
        $select = if ($env:NVSMI_GPU) { @('-i', $env:NVSMI_GPU) } else { @() }
        $line = & nvidia-smi @select --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0 -or -not $line) { throw "nvidia-smi failed ($LASTEXITCODE)." }
        $parts = ($line | Select-Object -First 1).Split(',') | ForEach-Object { [double]$_.Trim() }
        [pscustomobject]@{ MemoryGb = $parts[0] / 1024; Utilization = $parts[1] }
    }
    [pscustomobject]@{
        MemoryGb = [math]::Round(($samples | Measure-Object MemoryGb -Maximum).Maximum, 1)
        Utilization = [math]::Round(($samples | Measure-Object Utilization -Average).Average)
    }
}

$start = Get-Date
$deadline = $start.AddMinutes($MaxWaitMinutes)
while ($true) {
    $gpu = Read-Gpu
    $busy = $gpu.MemoryGb -gt $MaxBusyMemoryGb -or $gpu.Utilization -ge $MaxBusyUtilizationPercent
    $waited = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    if (-not $busy) {
        $msg = "GPU quiet before ${Label}: $($gpu.MemoryGb) GB in use, $($gpu.Utilization) % utilisation (waited $waited min)."
        Write-Output $msg
        $msg | Out-File -Append -FilePath $env:GITHUB_STEP_SUMMARY
        exit 0
    }
    if ((Get-Date) -ge $deadline) {
        $msg = "GPU still busy before ${Label} after $waited min: $($gpu.MemoryGb) GB in use, $($gpu.Utilization) % utilisation by other processes. Results below were measured under that load."
        Write-Output "::warning::$msg"
        $msg | Out-File -Append -FilePath $env:GITHUB_STEP_SUMMARY
        exit 0
    }
    Write-Output "$(Get-Date -Format u) GPU busy ($($gpu.MemoryGb) GB, $($gpu.Utilization) %); waiting before $Label."
    Start-Sleep -Seconds 60
}
