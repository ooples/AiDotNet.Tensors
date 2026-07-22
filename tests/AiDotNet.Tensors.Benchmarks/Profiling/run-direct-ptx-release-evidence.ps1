[CmdletBinding()]
param(
    [ValidateRange(3, 20)]
    [int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-direct-ptx-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [switch]$SkipBuild,
    [switch]$SkipExternal,
    [switch]$Issue834Only,
    [switch]$Issue835Only,
    [ValidateRange(0, 10)]
    [int]$ContaminationRetries = 4,
    [switch]$AllowDirty
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$project = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\AiDotNet.Tensors.Benchmarks.csproj'
$targetDll = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\bin\Release\net10.0\AiDotNet.Tensors.Benchmarks.dll'
$pythonRoot = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\BaselineRunners\py'
$evidenceRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($evidenceRoot) | Out-Null

function New-EvidenceSuite([string]$Name, [string]$Command, [string[]]$Arguments) {
    [pscustomobject]@{
        Name = $Name
        Command = $Command
        Arguments = $Arguments
    }
}

function Format-Command([string]$Command, [string[]]$Arguments) {
    $quoted = @($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    })
    return (@($Command) + $quoted) -join ' '
}

function Read-QkvDotnetRows([string]$Path) {
    $prefix = 'qkv_evidence_json='
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.StartsWith($prefix, [StringComparison]::Ordinal)
    } | ForEach-Object {
        $_.Substring($prefix.Length) | ConvertFrom-Json
    })
}

function Read-QkvPythonRows([string]$Path) {
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.TrimStart().StartsWith('{', [StringComparison]::Ordinal)
    } | ForEach-Object { $_ | ConvertFrom-Json })
}

function Assert-QkvDecodeThroughput([object[]]$Rows, [string]$Source) {
    foreach ($row in $Rows) {
        $deviceMedian = [double]$row.device_median_us
        $endToEndMedian = [double]$row.e2e_median_us
        $deviceTokens = [double]$row.device_tokens_per_second
        $endToEndTokens = [double]$row.e2e_tokens_per_second
        if ($deviceMedian -le 0 -or $endToEndMedian -le 0 -or
            $deviceTokens -le 0 -or $endToEndTokens -le 0) {
            throw "QKV release gate found missing or non-positive decode throughput in $Source for '$($row.shape)' '$($row.method)'."
        }
        $expectedDeviceTokens = 1e6 / $deviceMedian
        $expectedEndToEndTokens = 1e6 / $endToEndMedian
        if ([Math]::Abs($deviceTokens - $expectedDeviceTokens) -gt [Math]::Max(1e-6, $expectedDeviceTokens * 1e-9) -or
            [Math]::Abs($endToEndTokens - $expectedEndToEndTokens) -gt [Math]::Max(1e-6, $expectedEndToEndTokens * 1e-9)) {
            throw "QKV release gate found inconsistent decode throughput in $Source for '$($row.shape)' '$($row.method)'."
        }
    }
}

function Assert-QkvReleaseGate([string]$Root, [int]$RunCount, [bool]$IncludeExternal) {
    $shapes = @('decode-h4', 'decode-h8', 'decode-h16')
    $verdicts = [System.Collections.Generic.List[object]]::new()
    for ($run = 1; $run -le $RunCount; $run++) {
        $prefix = 'run-{0:D2}' -f $run
        $dotnetPath = Join-Path $Root ($prefix + '-qkv-rope-cache.log')
        $dotnetRows = @(Read-QkvDotnetRows $dotnetPath)
        if ($dotnetRows.Count -ne 9) {
            throw "QKV release gate expected 9 .NET rows in '$dotnetPath'; found $($dotnetRows.Count)."
        }
        Assert-QkvDecodeThroughput $dotnetRows $dotnetPath
        $pythonRows = @()
        if ($IncludeExternal) {
            $pythonPath = Join-Path $Root ($prefix + '-qkv-rope-cache-pytorch.log')
            $pythonRows = @(Read-QkvPythonRows $pythonPath | Where-Object { $_.status -eq 'ok' })
            if ($pythonRows.Count -ne 6) {
                throw "QKV release gate expected 6 PyTorch rows in '$pythonPath'; found $($pythonRows.Count)."
            }
            Assert-QkvDecodeThroughput $pythonRows $pythonPath
        }

        foreach ($shape in $shapes) {
            $direct = @($dotnetRows | Where-Object {
                $_.shape -eq $shape -and $_.method -eq 'Direct PTX CUDA graph'
            })
            $directEager = @($dotnetRows | Where-Object {
                $_.shape -eq $shape -and $_.method -eq 'Direct PTX fused'
            })
            $current = @($dotnetRows | Where-Object {
                $_.shape -eq $shape -and $_.method -eq 'AiDotNet cuBLAS+NVRTC'
            })
            if ($direct.Count -ne 1 -or $directEager.Count -ne 1 -or $current.Count -ne 1) {
                throw "QKV release gate has an incomplete or duplicate .NET method set for run $run '$shape'."
            }
            $candidate = $direct[0]
            $candidateRows = @($candidate, $directEager[0])
            if (@($candidateRows | Where-Object {
                [double]$_.max_error -gt 2e-5 -or
                [long]$_.managed_bytes -ne 0 -or
                [long]$_.temporary_device_bytes -ne 0 -or
                [int]$_.registers_per_thread -gt 48 -or
                [int]$_.static_shared_bytes -ne 0 -or
                [int]$_.local_bytes_per_thread -ne 0 -or
                [int]$_.active_blocks_per_sm -lt 8
            }).Count -ne 0) {
                throw "QKV release resource/correctness gate failed for run $run '$shape'."
            }

            $peers = @($current[0])
            if ($IncludeExternal) {
                $peers += @($pythonRows | Where-Object { $_.shape -eq $shape })
            }
            foreach ($peer in $peers) {
                if ([double]$peer.max_error -gt 2e-5) {
                    throw "QKV peer '$($peer.method)' exceeded the correctness tolerance for run $run '$shape'."
                }
                $deviceSpeedup = [double]$peer.device_median_us / [double]$candidate.device_median_us
                $endToEndSpeedup = [double]$peer.e2e_median_us / [double]$candidate.e2e_median_us
                $p95Ratio = [double]$candidate.device_p95_us / [double]$peer.device_p95_us
                if ($deviceSpeedup -lt 1.10 -or $endToEndSpeedup -lt 1.10 -or $p95Ratio -gt 1.10) {
                    throw "QKV championship gate failed for run $run '$shape' versus '$($peer.method)': device=$deviceSpeedup, E2E=$endToEndSpeedup, P95 ratio=$p95Ratio."
                }
                $verdicts.Add([ordered]@{
                    run = $run
                    shape = $shape
                    competitor = $peer.method
                    device_median_speedup = $deviceSpeedup
                    e2e_median_speedup = $endToEndSpeedup
                    device_p95_ratio = $p95Ratio
                })
            }
        }
    }
    $gatePath = Join-Path $Root 'qkv-release-gate.json'
    [ordered]@{
        status = 'pass'
        required_device_and_e2e_median_speedup = 1.10
        maximum_device_p95_ratio = 1.10
        maximum_error = 2e-5
        runs = $RunCount
        external_competitors_included = $IncludeExternal
        verdicts = @($verdicts)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gatePath -Encoding utf8
}

function Get-GpuSnapshot {
    $output = & nvidia-smi `
        '--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used' `
        '--format=csv,noheader,nounits' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi snapshot failed: $output" }
    return ($output -join [Environment]::NewLine).Trim()
}

function Assert-GpuReady([string]$Label, [switch]$AfterSuite) {
    $status = & nvidia-smi '--query-gpu=utilization.gpu,memory.used,temperature.gpu' '--format=csv,noheader,nounits' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "[$Label] nvidia-smi status failed: $status" }
    $cells = @((($status -join '') -split ',') | ForEach-Object { $_.Trim() })
    if ($cells.Count -ge 3) {
        $utilization = 0
        $usedMegabytes = 0
        $temperatureCelsius = 0
        if ([int]::TryParse($cells[0], [ref]$utilization) -and
            [int]::TryParse($cells[1], [ref]$usedMegabytes) -and
            [int]::TryParse($cells[2], [ref]$temperatureCelsius)) {
            if ($temperatureCelsius -gt 75) {
                throw "[$Label] GPU temperature $temperatureCelsius C exceeds the 75 C evidence ceiling."
            }
            # A process that just exited can still own the current utilization
            # sample. Apply idle/memory admission only before a suite; after a
            # suite, require no foreign compute and a safe temperature.
            if (-not $AfterSuite -and ($utilization -gt 20 -or $usedMegabytes -gt 2048)) {
                throw "[$Label] GPU is not benchmark-ready (utilization=$utilization%, memory.used=$usedMegabytes MiB, temperature=$temperatureCelsius C)."
            }
        }
    }

    $pmon = @(& nvidia-smi pmon -c 1 -s u 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "[$Label] nvidia-smi pmon failed: $($pmon -join ' ')" }
    $conflicts = @()
    foreach ($line in $pmon) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $parts = $trimmed -split '\s+'
        if ($parts.Count -lt 9) { continue }
        $processType = $parts[2]
        $smPercent = 0
        $isComputeOnly = $processType -eq 'C'
        # WDDM's single pmon sample can report stale C+G percentages after the
        # benchmark process exits (including values inconsistent with a 1%
        # whole-device snapshot). Enforce mixed graphics/compute admission at
        # the stable pre-suite boundary; the post boundary still rejects every
        # compute-only process and unsafe temperature.
        $isActiveMixed = -not $AfterSuite -and $processType.Contains('C') -and
            [int]::TryParse($parts[3], [ref]$smPercent) -and $smPercent -gt 5
        if ($isComputeOnly -or $isActiveMixed) {
            $conflicts += "pid=$($parts[1]) $($parts[-1]) type=$processType sm=$($parts[3])%"
        }
    }
    if ($conflicts.Count -ne 0) {
        throw "[$Label] Foreign GPU workload detected; clean benchmark refused: $($conflicts -join '; ')"
    }
}

Push-Location $repoRoot
try {
    if ($Issue834Only -and $Issue835Only) {
        throw '-Issue834Only and -Issue835Only are mutually exclusive.'
    }
    $gitCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the Git commit for the evidence manifest.' }
    $dirtyLines = @(& git status --porcelain)
    if ($dirtyLines.Count -ne 0 -and -not $AllowDirty) {
        throw 'Release evidence requires a clean worktree. Commit the exact candidate first or pass -AllowDirty for diagnostic-only capture.'
    }

    if (-not $SkipBuild) {
        $buildLog = Join-Path $evidenceRoot 'build.log'
        "# command=dotnet build `"$project`" -c Release -f net10.0" |
            Set-Content -LiteralPath $buildLog -Encoding utf8
        & dotnet build $project -c Release -f net10.0 2>&1 |
            Out-File -LiteralPath $buildLog -Append -Encoding utf8
        if ($LASTEXITCODE -ne 0) {
            throw "Release benchmark build failed with exit code $LASTEXITCODE. See '$buildLog'."
        }
    }
    if (-not (Test-Path -LiteralPath $targetDll -PathType Leaf)) {
        throw "Benchmark target is missing at '$targetDll'."
    }

    $suites = [System.Collections.Generic.List[object]]::new()
    if (-not $Issue834Only -and -not $Issue835Only) {
        $suites.Add((New-EvidenceSuite 'online-attention' 'dotnet' @($targetDll, '--direct-ptx-online-attention')))
        $suites.Add((New-EvidenceSuite 'gpu-matrix' 'dotnet' @($targetDll, '--direct-ptx-gpu-matrix')))
        $suites.Add((New-EvidenceSuite 'residual-rmsnorm' 'dotnet' @($targetDll, '--direct-ptx-residual-rmsnorm')))
        if (-not $SkipExternal) {
            $suites.Add((New-EvidenceSuite 'external-gpu-baselines' 'dotnet' @($targetDll, '--direct-ptx-external-gpu-baselines')))
        }
    }

    if (-not $Issue835Only) {
        $suites.Add((New-EvidenceSuite 'attention-family' 'dotnet' @($targetDll, '--direct-ptx-attention-family', '1')))
        $suites.Add((New-EvidenceSuite 'decode' 'dotnet' @($targetDll, '--direct-ptx-decode', '1')))
        $suites.Add((New-EvidenceSuite 'paged-prefill' 'dotnet' @($targetDll, '--direct-ptx-paged-prefill', '1')))
        $suites.Add((New-EvidenceSuite 'attention-backward' 'dotnet' @($targetDll, '--direct-ptx-attention-backward', '1')))
        $suites.Add((New-EvidenceSuite 'flash-attention-backward' 'dotnet' @($targetDll, '--direct-ptx-flash-attention-backward', '1')))
    }
    if (-not $Issue834Only) {
        $suites.Add((New-EvidenceSuite 'qkv-rope-cache' 'dotnet' @(
            $targetDll, '--direct-ptx-qkv-rope-cache', '1', '--no-external')))
    }

    if (-not $SkipExternal) {
        $python = (Get-Command python -ErrorAction Stop).Source
        if (-not $Issue835Only) {
            $suites.Add((New-EvidenceSuite 'attention-family-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_attention_family_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'decode-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_decode_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'paged-prefill-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_paged_prefill_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'attention-backward-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_attention_backward_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'flash-attention-backward-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_flash_attention_backward_competitors.py'), '--runs', '1')))
        }
        if (-not $Issue834Only) {
            $suites.Add((New-EvidenceSuite 'qkv-rope-cache-pytorch' $python @(
                (Join-Path $pythonRoot 'run_direct_ptx_qkv_rope_cache_competitors.py'), '--runs', '1', '--json-lines')))
        }
    }

    $previousDirectPtx = $env:AIDOTNET_DIRECT_PTX
    $previousAutotune = $env:AIDOTNET_DIRECT_PTX_AUTOTUNE
    $env:AIDOTNET_DIRECT_PTX = '1'
    $env:AIDOTNET_DIRECT_PTX_AUTOTUNE = '0'
    try {
        for ($run = 1; $run -le $Runs; $run++) {
            foreach ($suite in $suites) {
                $label = "run-$('{0:D2}' -f $run)-$($suite.Name)"
                $captured = $false
                for ($attempt = 1; $attempt -le 1 + $ContaminationRetries; $attempt++) {
                    $ready = $false
                    $consecutiveReadySamples = 0
                    for ($poll = 1; $poll -le 30; $poll++) {
                        try {
                            Assert-GpuReady "$label-start"
                            $consecutiveReadySamples++
                            if ($consecutiveReadySamples -ge 3) {
                                $ready = $true
                                break
                            }
                        }
                        catch {
                            $consecutiveReadySamples = 0
                            if ($poll -eq 30) { throw }
                        }
                        Start-Sleep -Seconds 1
                    }
                    if (-not $ready) { throw "GPU readiness polling ended unexpectedly for '$label'." }

                    $log = Join-Path $evidenceRoot ("{0}.log" -f $label)
                    "# independent process $run/$Runs; suite=$($suite.Name); attempt=$attempt; started_utc=$([DateTime]::UtcNow.ToString('O'))" |
                        Set-Content -LiteralPath $log -Encoding utf8
                    "# git_commit=$gitCommit; dirty_worktree=$($dirtyLines.Count -ne 0); AIDOTNET_DIRECT_PTX=1; AIDOTNET_DIRECT_PTX_AUTOTUNE=0" |
                        Add-Content -LiteralPath $log -Encoding utf8
                    "# host_os=$([System.Runtime.InteropServices.RuntimeInformation]::OSDescription); powershell=$($PSVersionTable.PSVersion); dotnet=$(& dotnet --version)" |
                        Add-Content -LiteralPath $log -Encoding utf8
                    "# GPU name, uuid, driver, pstate, SM MHz, memory MHz, C, W, limit W, utilization %, memory MiB: $(Get-GpuSnapshot)" |
                        Add-Content -LiteralPath $log -Encoding utf8
                    "# command=$(Format-Command $suite.Command $suite.Arguments)" |
                        Add-Content -LiteralPath $log -Encoding utf8

                    # Keep the GPU-accelerated terminal out of the measured interval.
                    # The complete TUI is emitted only to the immutable process log.
                    $arguments = $suite.Arguments
                    $savedErrorAction = $ErrorActionPreference
                    try {
                        # Windows PowerShell wraps a native process's stderr as
                        # ErrorRecord objects. PyTorch uses stderr for backend
                        # eligibility warnings, so capture both streams and use
                        # the native exit code as the sole success criterion.
                        $ErrorActionPreference = 'Continue'
                        & $suite.Command @arguments 2>&1 |
                            Out-File -LiteralPath $log -Append -Encoding utf8
                        $exitCode = $LASTEXITCODE
                    }
                    finally {
                        $ErrorActionPreference = $savedErrorAction
                    }
                    "# ending_gpu=$(Get-GpuSnapshot)" | Add-Content -LiteralPath $log -Encoding utf8
                    "# completed_utc=$([DateTime]::UtcNow.ToString('O')); exit_code=$exitCode" |
                        Add-Content -LiteralPath $log -Encoding utf8
                    if ($exitCode -ne 0) {
                        throw "Evidence suite '$($suite.Name)' run $run failed with exit code $exitCode. See '$log'."
                    }

                    $contamination = $null
                    try { Assert-GpuReady "$label-end" -AfterSuite }
                    catch { $contamination = $_.Exception.Message }
                    if ($contamination) {
                        "# rejected_environment=$contamination" | Add-Content -LiteralPath $log -Encoding utf8
                        $rejected = Join-Path $evidenceRoot ("{0}-attempt-{1:D2}.rejected.txt" -f $label, $attempt)
                        Move-Item -LiteralPath $log -Destination $rejected -Force
                        if ($attempt -gt $ContaminationRetries) {
                            throw "Evidence suite '$($suite.Name)' run $run remained contaminated after $attempt attempts. Last reason: $contamination"
                        }
                        Start-Sleep -Seconds 2
                        continue
                    }

                    $captured = $true
                    break
                }
                if (-not $captured) { throw "No clean evidence was captured for '$label'." }
            }
        }
    }
    finally {
        $env:AIDOTNET_DIRECT_PTX = $previousDirectPtx
        $env:AIDOTNET_DIRECT_PTX_AUTOTUNE = $previousAutotune
    }

    if (-not $Issue834Only) {
        Assert-QkvReleaseGate $evidenceRoot $Runs (-not [bool]$SkipExternal)
    }

    $files = Get-ChildItem -LiteralPath $evidenceRoot -File | Where-Object {
        $_.Extension -eq '.log' -or $_.Name -eq 'qkv-release-gate.json'
    } | Sort-Object Name | ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [ordered]@{ file = $_.Name; sha256 = $hash.Hash.ToLowerInvariant() }
    }
    $rejectedFiles = Get-ChildItem -LiteralPath $evidenceRoot -Filter '*.rejected.txt' | Sort-Object Name | ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [ordered]@{ file = $_.Name; sha256 = $hash.Hash.ToLowerInvariant() }
    }
    $manifest = [ordered]@{
        generated_utc = [DateTime]::UtcNow.ToString('O')
        git_commit = $gitCommit
        dirty_worktree = $dirtyLines.Count -ne 0
        requested_independent_runs = $Runs
        contamination_retries_per_suite = $ContaminationRetries
        issue_834_only = [bool]$Issue834Only
        issue_835_only = [bool]$Issue835Only
        external_gpu_baselines_included = -not [bool]$SkipExternal
        feature_gates = [ordered]@{
            AIDOTNET_DIRECT_PTX = '1'
            AIDOTNET_DIRECT_PTX_AUTOTUNE = '0'
        }
        commands = [ordered]@{
            build = "dotnet build `"$project`" -c Release -f net10.0"
            suites = @($suites | ForEach-Object { Format-Command $_.Command $_.Arguments })
        }
        files = @($files)
        rejected_environment_attempts = @($rejectedFiles)
    } | ConvertTo-Json -Depth 5
    $manifestPath = Join-Path $evidenceRoot 'manifest.json'
    [System.IO.File]::WriteAllText($manifestPath, $manifest + [Environment]::NewLine)
    Write-Host "Evidence capture completed: $evidenceRoot"
    Write-Host 'Run each required Nsight target separately; performance-counter access is a mandatory release gate.'
}
finally {
    Pop-Location
}
