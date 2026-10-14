<#
.SYNOPSIS
  #1261: runs the CUDA GEMM grid as several clean process launches per shape and summarizes the evidence.

.DESCRIPTION
  Each shape runs in its own benchmark process (`--direct-ptx-dense-linear-full R --only=gemm-grid-NAME`), and the
  whole grid is swept -Launches times, so every cell gets Launches x RunsPerLaunch independent run medians from
  processes started at different times (different clocks and thermal states). One process per shape also bounds
  memory and lets a launch that the clean-evidence gates refuse (a foreign Python or GPU process appeared, the GPU
  was too hot) be retried without discarding hours of other shapes. Any other failure stops the sweep.

.EXAMPLE
  pwsh -File tests/AiDotNet.Tensors.Benchmarks/Profiling/run-cuda-gemm-grid.ps1 `
      -Executable tests/AiDotNet.Tensors.Benchmarks/bin/Release/net10.0/AiDotNet.Tensors.Benchmarks.exe `
      -Python C:\path\to\cuda-torch\python.exe -SmCount 24 -Fp32LanesPerSm 64 -CooldownCelsius 62
#>
param(
    [Parameter(Mandatory)][string]$Executable,
    # CUDA-enabled PyTorch interpreter for the PyTorch lanes.
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][int]$SmCount,
    [Parameter(Mandatory)][int]$Fp32LanesPerSm,
    [int]$Launches = 3,
    [int]$RunsPerLaunch = 3,
    # Shape names (see --list-gemm-grid); empty runs the whole grid.
    [string[]]$Shapes = @(),
    [string]$OutputDirectory = 'artifacts/cuda-gemm-grid',
    # Cooldown target before each route; a laptop GPU idles too warm for the default 55 C.
    [int]$CooldownCelsius = 0,
    [int]$MaxAttempts = 6,
    [int]$RetryDelaySeconds = 60
)
$ErrorActionPreference = 'Stop'

$Executable = (Resolve-Path $Executable).Path
$Python = (Resolve-Path $Python).Path
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
$known = @(& $Executable --list-gemm-grid)
if ($known.Count -eq 0) { throw 'The benchmark listed no GEMM grid shapes.' }
# `pwsh -File run-cuda-gemm-grid.ps1 -Shapes a,b` arrives as ONE string, because -File does not parse arrays; split it.
$Shapes = @($Shapes | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Shapes.Count -eq 0) { $Shapes = $known }
$unknown = @($Shapes | Where-Object { $_ -notin $known })
if ($unknown.Count -gt 0) { throw "Unknown GEMM grid shape(s): $($unknown -join ', '). Known: $($known -join ', ')" }

$env:PYTHON = $Python
if ($CooldownCelsius -gt 0) { $env:AIDOTNET_GEMM_GRID_COOLDOWN_CELSIUS = "$CooldownCelsius" }

# Refusals by the clean-evidence gates are environmental and worth a retry; anything else is a real failure.
$retryable = 'Foreign GPU workload|foreign Python workload|Python workload detected|exceeds the \d+ C evidence ceiling|' +
             'not benchmark-ready|stayed at \d+ C|quiescen'

$started = Get-Date
for ($launch = 1; $launch -le $Launches; $launch++) {
    foreach ($shape in $Shapes) {
        $log = Join-Path $OutputDirectory ("{0}.launch{1}.log" -f $shape, $launch)
        if (Test-Path $log) {
            Write-Host "[$shape launch $launch] already recorded, skipping"
            continue
        }
        for ($attempt = 1; ; $attempt++) {
            $partial = "$log.partial"
            $timer = [Diagnostics.Stopwatch]::StartNew()
            & $Executable --direct-ptx-dense-linear-full $RunsPerLaunch "--only=gemm-grid-$shape" *> $partial
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Move-Item $partial $log
                Write-Host ("[{0} launch {1}] ok in {2:F0} s" -f $shape, $launch, $timer.Elapsed.TotalSeconds)
                break
            }
            $text = Get-Content $partial -Raw
            $failed = "$log.failed$attempt"
            Move-Item -Force $partial $failed
            if ($text -notmatch $retryable) {
                throw "[$shape launch $launch] failed with exit code $code for a non-environmental reason; see $failed"
            }
            if ($attempt -ge $MaxAttempts) {
                throw "[$shape launch $launch] the clean-evidence gates refused $MaxAttempts attempts; see $failed"
            }
            Write-Host "[$shape launch $launch] attempt $attempt refused by a clean-evidence gate; retrying in $RetryDelaySeconds s"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
}
Write-Host ("Sweep finished in {0:F1} h" -f ((Get-Date) - $started).TotalHours)

$summary = Join-Path $OutputDirectory 'summary.md'
$logs = @(Get-ChildItem $OutputDirectory -Filter '*.log' | ForEach-Object FullName)
& $Python -I (Join-Path $PSScriptRoot 'summarize_cuda_gemm_grid.py') --sm-count $SmCount `
    --fp32-lanes-per-sm $Fp32LanesPerSm @logs | Set-Content -Path $summary -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "The summarizer failed with exit code $LASTEXITCODE." }
Write-Host "Summary: $summary"
