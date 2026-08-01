[CmdletBinding()]
param(
    [ValidateRange(3, 20)]
    [int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-direct-ptx-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [switch]$SkipBuild,
    [switch]$SkipExternal,
    [switch]$Issue834Only,
    [switch]$Issue835Only,
    [switch]$Issue853Only,
    [ValidateRange(0, 10)]
    [int]$ContaminationRetries = 4,
    [switch]$AllowDirty
)

$ErrorActionPreference = 'Stop'
$hostCpuCeilingPercent = 20.0
$benchmarkOwnedCpuAllowance = 1.5
if (@(@($Issue834Only, $Issue835Only, $Issue853Only) | Where-Object { $_ }).Count -gt 1) {
    throw 'Only one issue-specific evidence switch may be selected.'
}

if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
        [System.Runtime.InteropServices.OSPlatform]::Windows) -and
    -not ('AiDotNetBenchmarkHostCpu' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class AiDotNetBenchmarkHostCpu
{
    [StructLayout(LayoutKind.Sequential)]
    private struct FileTime
    {
        internal uint Low;
        internal uint High;
        internal long Ticks { get { return ((long)High << 32) | Low; } }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetSystemTimes(
        out FileTime idle, out FileTime kernel, out FileTime user);

    public static long[] Snapshot()
    {
        FileTime idle, kernel, user;
        if (!GetSystemTimes(out idle, out kernel, out user))
            throw new InvalidOperationException(
                "GetSystemTimes failed with Win32 error " + Marshal.GetLastWin32Error() + ".");
        return new[] { idle.Ticks, kernel.Ticks, user.Ticks };
    }

    public static double UsagePercent(long[] before, long[] after)
    {
        double idle = after[0] - before[0];
        double total = (after[1] - before[1]) + (after[2] - before[2]);
        if (total <= 0.0 || idle < 0.0 || idle > total)
            throw new InvalidOperationException("Invalid GetSystemTimes interval.");
        return 100.0 * (total - idle) / total;
    }
}
'@
}
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
            if ($pythonRows.Count -ne 9) {
                throw "QKV release gate expected 9 PyTorch rows in '$pythonPath'; found $($pythonRows.Count)."
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
                $shapePeers = @($pythonRows | Where-Object { $_.shape -eq $shape })
                $expectedPeerMethods = @(
                    'PyTorch CUDA eager',
                    'PyTorch CUDA graph',
                    'PyTorch compile max-autotune'
                )
                if ($shapePeers.Count -ne $expectedPeerMethods.Count -or
                    @($expectedPeerMethods | Where-Object {
                        $method = $_
                        @($shapePeers | Where-Object { $_.method -eq $method }).Count -ne 1
                    }).Count -ne 0) {
                    throw "QKV release gate has an incomplete or duplicate PyTorch method set for run $run '$shape'."
                }
                $peers += $shapePeers
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

function Read-SolverDotnetRows([string]$Path) {
    $prefix = 'solver_evidence_json='
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.StartsWith($prefix, [StringComparison]::Ordinal)
    } | ForEach-Object {
        $row = $_.Substring($prefix.Length) | ConvertFrom-Json
        [pscustomobject]@{
            status = 'ok'
            operation = [string]$row.Operation
            batch = [int]$row.Batch
            method = [string]$row.Method
            device_median_us = [double]$row.Device.Median
            device_p95_us = [double]$row.Device.P95
            device_launches_per_sample = [int]$row.DeviceLaunchesPerSample
            e2e_median_us = [double]$row.EndToEnd.Median
            max_error = [double]$row.MaximumError
            managed_bytes = [long]$row.ManagedBytes
            temporary_device_bytes = [long]$row.TemporaryDeviceBytes
            static_shared_bytes = [int]$row.SharedBytes
            local_bytes_per_thread = [int]$row.LocalBytes
            active_blocks_per_sm = [int]$row.ActiveBlocksPerSm
            device_fingerprint = [string]$row.DeviceFingerprint
        }
    })
}

function Read-SolverPythonRows([string]$Path) {
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.TrimStart().StartsWith('{', [StringComparison]::Ordinal)
    } | ForEach-Object { $_ | ConvertFrom-Json })
}

function Assert-SolverDotnetAcceptedAttempt([string]$Path, [int]$Run) {
    $operations = @(
        'cholesky', 'lu-factor', 'qr', 'eigh', 'eigh-lower', 'svd', 'lu-solve',
        'ldl-factor', 'ldl-solve', 'solve', 'tri-lower', 'tri-upper',
        'chol-backward', 'solve-backward')
    $batches = @(1024, 4096, 16384, 65536)
    $currentOperations = @('cholesky', 'lu-factor', 'qr', 'eigh')
    $rows = @(Read-SolverDotnetRows $Path)
    if ($rows.Count -ne 128) {
        throw "Solver attempt expected 128 .NET rows in '$Path'; found $($rows.Count)."
    }
    $fingerprints = @($rows.device_fingerprint | Sort-Object -Unique)
    if ($fingerprints.Count -ne 1 -or [string]::IsNullOrWhiteSpace($fingerprints[0])) {
        throw "Solver attempt found inconsistent .NET device fingerprints in '$Path'."
    }
    if (@($rows | Where-Object { [int]$_.device_launches_per_sample -lt 1000 }).Count -ne 0) {
        throw "Solver attempt found a device distribution with fewer than 1,000 launches per sample in '$Path'."
    }

    $findings = [System.Collections.Generic.List[string]]::new()
    foreach ($operation in $operations) {
        foreach ($batch in $batches) {
            $cell = @($rows | Where-Object {
                $_.operation -eq $operation -and $_.batch -eq $batch
            })
            $resident = @($cell | Where-Object { $_.method -eq 'Direct PTX resident' })
            $graph = @($cell | Where-Object { $_.method -eq 'Direct PTX CUDA graph' })
            if ($resident.Count -ne 1 -or $graph.Count -ne 1) {
                throw "Solver attempt has an incomplete direct method set for run $Run '$operation'/B=$batch."
            }
            foreach ($candidate in @($resident[0], $graph[0])) {
                $candidateMetrics = @(
                    [double]$candidate.device_median_us,
                    [double]$candidate.device_p95_us,
                    [double]$candidate.e2e_median_us)
                if (@($candidateMetrics | Where-Object {
                        [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0.0
                    }).Count -ne 0 -or
                    [double]$candidate.max_error -gt 2e-5 -or
                    [long]$candidate.managed_bytes -ne 0 -or
                    [long]$candidate.temporary_device_bytes -ne 0 -or
                    [int]$candidate.static_shared_bytes -ne 0 -or
                    [int]$candidate.local_bytes_per_thread -ne 0 -or
                    [int]$candidate.active_blocks_per_sm -lt 2) {
                    throw "Solver attempt correctness/resource gate failed for run $Run '$operation'/B=$batch '$($candidate.method)'."
                }
            }

            $current = @($cell | Where-Object { $_.method -eq 'AiDotNet CUDA established' })
            if ($currentOperations -notcontains $operation) {
                if ($current.Count -ne 0) {
                    throw "Solver attempt found an unexpected established baseline for run $Run '$operation'/B=$batch."
                }
                continue
            }
            if ($current.Count -ne 1) {
                throw "Solver attempt is missing the established AiDotNet baseline for run $Run '$operation'/B=$batch."
            }
            $peerMetrics = @(
                [double]$current[0].device_median_us,
                [double]$current[0].device_p95_us,
                [double]$current[0].e2e_median_us)
            if (@($peerMetrics | Where-Object {
                    [double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0.0
                }).Count -ne 0 -or [double]$current[0].max_error -gt 2e-5) {
                throw "Solver attempt peer has an invalid timing metric or exceeded correctness tolerance for run $Run '$operation'/B=$batch."
            }

            $deviceSpeedup = [double]$current[0].device_median_us / [double]$resident[0].device_median_us
            $endToEndSpeedup = [double]$current[0].e2e_median_us / [double]$resident[0].e2e_median_us
            $p95Ratio = [double]$resident[0].device_p95_us / [double]$current[0].device_p95_us
            if ($deviceSpeedup -lt 1.10 -or $endToEndSpeedup -lt 1.10 -or $p95Ratio -gt 1.10) {
                $findings.Add(
                    "'$operation'/B=$batch device=$($deviceSpeedup.ToString('F3')), " +
                    "E2E=$($endToEndSpeedup.ToString('F3')), P95 ratio=$($p95Ratio.ToString('F3'))")
            }
        }
    }
    if ($findings.Count -ne 0) {
        throw "Solver attempt failed $($findings.Count) internal championship comparison(s): $($findings -join '; ')."
    }
}

function Assert-SolverReleaseGate([string]$Root, [int]$RunCount, [bool]$IncludeExternal) {
    $operations = @(
        'cholesky', 'lu-factor', 'qr', 'eigh', 'eigh-lower', 'svd', 'lu-solve',
        'ldl-factor', 'ldl-solve', 'solve', 'tri-lower', 'tri-upper',
        'chol-backward', 'solve-backward')
    $batches = @(1024, 4096, 16384, 65536)
    $currentOperations = @('cholesky', 'lu-factor', 'qr', 'eigh')
    $verdicts = [System.Collections.Generic.List[object]]::new()
    $findings = [System.Collections.Generic.List[object]]::new()
    for ($run = 1; $run -le $RunCount; $run++) {
        $prefix = 'run-{0:D2}' -f $run
        $dotnetPath = Join-Path $Root ($prefix + '-solvers-4x4.log')
        $dotnetRows = @(Read-SolverDotnetRows $dotnetPath)
        if ($dotnetRows.Count -ne 128) {
            throw "Solver release gate expected 128 .NET rows in '$dotnetPath'; found $($dotnetRows.Count)."
        }
        $pythonRows = @()
        if ($IncludeExternal) {
            $pythonPath = Join-Path $Root ($prefix + '-solvers-4x4-pytorch.log')
            $pythonRows = @(Read-SolverPythonRows $pythonPath)
            if ($pythonRows.Count -ne 112) {
                throw "Solver release gate expected 112 PyTorch rows in '$pythonPath'; found $($pythonRows.Count)."
            }
            foreach ($pythonRow in @($pythonRows | Where-Object { $_.status -eq 'ok' })) {
                $calibrationUs = [double]$pythonRow.calibration_us
                $samples = [int]$pythonRow.samples
                $launches = [int]$pythonRow.device_launches_per_sample
                if ([double]::IsNaN($calibrationUs) -or
                    [double]::IsInfinity($calibrationUs) -or $calibrationUs -le 0.0) {
                    throw "Solver release gate found invalid PyTorch calibration metadata for run $run '$($pythonRow.operation)'/B=$($pythonRow.batch) '$($pythonRow.method)'."
                }
                $minimumSamples = if ($calibrationUs -ge 1000.0) { 21 } else { 101 }
                if ($samples -lt $minimumSamples -or $launches -lt 1 -or $launches -gt 10) {
                    throw "Solver release gate found insufficient PyTorch sampling metadata for run $run '$($pythonRow.operation)'/B=$($pythonRow.batch) '$($pythonRow.method)' (calibration=${calibrationUs}us, samples=$samples, launches=$launches)."
                }
            }
        }

        $fingerprints = @($dotnetRows.device_fingerprint | Sort-Object -Unique)
        if ($fingerprints.Count -ne 1 -or [string]::IsNullOrWhiteSpace($fingerprints[0])) {
            throw "Solver release gate found inconsistent .NET device fingerprints in '$dotnetPath'."
        }
        if (@($dotnetRows | Where-Object { [int]$_.device_launches_per_sample -lt 1000 }).Count -ne 0) {
            throw "Solver release gate found a device distribution with fewer than 1,000 launches per sample in '$dotnetPath'."
        }

        foreach ($operation in $operations) {
            foreach ($batch in $batches) {
                $cell = @($dotnetRows | Where-Object {
                    $_.operation -eq $operation -and $_.batch -eq $batch
                })
                $resident = @($cell | Where-Object { $_.method -eq 'Direct PTX resident' })
                $graph = @($cell | Where-Object { $_.method -eq 'Direct PTX CUDA graph' })
                if ($resident.Count -ne 1 -or $graph.Count -ne 1) {
                    throw "Solver release gate has an incomplete direct method set for run $run '$operation'/B=$batch."
                }
                foreach ($candidate in @($resident[0], $graph[0])) {
                    if ([double]$candidate.max_error -gt 2e-5 -or
                        [long]$candidate.managed_bytes -ne 0 -or
                        [long]$candidate.temporary_device_bytes -ne 0 -or
                        [int]$candidate.static_shared_bytes -ne 0 -or
                        [int]$candidate.local_bytes_per_thread -ne 0 -or
                        [int]$candidate.active_blocks_per_sm -lt 2) {
                        throw "Solver correctness/resource gate failed for run $run '$operation'/B=$batch '$($candidate.method)'."
                    }
                }

                $comparisons = [System.Collections.Generic.List[object]]::new()
                if ($currentOperations -contains $operation) {
                    $current = @($cell | Where-Object { $_.method -eq 'AiDotNet CUDA established' })
                    if ($current.Count -ne 1) {
                        throw "Solver release gate is missing the established AiDotNet baseline for run $run '$operation'/B=$batch."
                    }
                    $comparisons.Add([pscustomobject]@{ Candidate = $resident[0]; Peer = $current[0] })
                }
                elseif (@($cell | Where-Object { $_.method -eq 'AiDotNet CUDA established' }).Count -ne 0) {
                    throw "Solver release gate found an unexpected established baseline for run $run '$operation'/B=$batch."
                }

                if ($IncludeExternal) {
                    $externalCell = @($pythonRows | Where-Object {
                        $_.operation -eq $operation -and [int]$_.batch -eq $batch
                    })
                    if ($externalCell.Count -ne 2) {
                        throw "Solver release gate has an incomplete PyTorch method set for run $run '$operation'/B=$batch."
                    }
                    $eager = @($externalCell | Where-Object {
                        $_.method -eq 'PyTorch CUDA eager/cuSOLVER'
                    })
                    $externalGraph = @($externalCell | Where-Object {
                        $_.method -eq 'PyTorch CUDA graph/cuSOLVER'
                    })
                    if ($eager.Count -ne 1 -or $externalGraph.Count -ne 1) {
                        throw "Solver release gate found duplicate or unknown PyTorch methods for run $run '$operation'/B=$batch."
                    }
                    if ($eager[0].status -ne 'ok') {
                        throw "Required PyTorch eager competitor is unavailable for run $run '$operation'/B=${batch}: $($eager[0].reason)"
                    }
                    $comparisons.Add([pscustomobject]@{ Candidate = $resident[0]; Peer = $eager[0] })
                    if ($externalGraph[0].status -eq 'ok') {
                        $comparisons.Add([pscustomobject]@{ Candidate = $graph[0]; Peer = $externalGraph[0] })
                    }
                }

                foreach ($comparison in $comparisons) {
                    $candidate = $comparison.Candidate
                    $peer = $comparison.Peer
                    if ([double]$peer.max_error -gt 2e-5) {
                        throw "Solver peer '$($peer.method)' exceeded correctness tolerance for run $run '$operation'/B=$batch."
                    }
                    $deviceSpeedup = [double]$peer.device_median_us / [double]$candidate.device_median_us
                    $endToEndSpeedup = [double]$peer.e2e_median_us / [double]$candidate.e2e_median_us
                    $p95Ratio = [double]$candidate.device_p95_us / [double]$peer.device_p95_us
                    $passed = $deviceSpeedup -ge 1.10 -and $endToEndSpeedup -ge 1.10 -and $p95Ratio -le 1.10
                    $verdicts.Add([ordered]@{
                        run = $run
                        operation = $operation
                        batch = $batch
                        candidate = $candidate.method
                        competitor = $peer.method
                        device_median_speedup = $deviceSpeedup
                        e2e_median_speedup = $endToEndSpeedup
                        device_p95_ratio = $p95Ratio
                        status = if ($passed) { 'pass' } else { 'fail' }
                    })
                    if (-not $passed) {
                        $findings.Add([ordered]@{
                            run = $run
                            operation = $operation
                            batch = $batch
                            candidate = $candidate.method
                            competitor = $peer.method
                            device_median_speedup = $deviceSpeedup
                            required_device_median_speedup = 1.10
                            device_median_deficit = [Math]::Max(0.0, 1.10 - $deviceSpeedup)
                            e2e_median_speedup = $endToEndSpeedup
                            required_e2e_median_speedup = 1.10
                            e2e_median_deficit = [Math]::Max(0.0, 1.10 - $endToEndSpeedup)
                            device_p95_ratio = $p95Ratio
                            maximum_device_p95_ratio = 1.10
                            device_p95_excess = [Math]::Max(0.0, $p95Ratio - 1.10)
                        })
                    }
                }
            }
        }
    }
    $gatePath = Join-Path $Root 'solver-release-gate.json'
    [ordered]@{
        status = if ($findings.Count -ne 0) { 'fail' } elseif ($IncludeExternal) { 'pass' } else { 'partial-pass' }
        required_device_and_e2e_median_speedup = 1.10
        maximum_device_p95_ratio = 1.10
        maximum_error = 2e-5
        required_managed_temporary_shared_local_bytes = 0
        minimum_active_blocks_per_sm = 2
        maximum_adjusted_foreign_host_cpu_percent = $hostCpuCeilingPercent
        benchmark_owned_cpu_allowance = $benchmarkOwnedCpuAllowance
        external_sampling = '101 samples below 1 ms; at least 21 samples at or above 1 ms; 1-10 calibrated launches/sample'
        external_process_isolation = 'uninterrupted resident eager phase, then one disposable CUDA-graph process per run/operation/batch cell'
        runs = $RunCount
        external_competitors_included = $IncludeExternal
        verdicts = @($verdicts)
        findings = @($findings)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gatePath -Encoding utf8
    if ($findings.Count -ne 0) {
        throw "Solver championship gate failed $($findings.Count) comparison(s); inspect '$gatePath' for the complete finding set."
    }
}

function Get-GpuSnapshot {
    $output = & nvidia-smi `
        '--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used' `
        '--format=csv,noheader,nounits' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi snapshot failed: $output" }
    return ($output -join [Environment]::NewLine).Trim()
}

function Get-HostCpuSnapshot {
    if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform(
            [System.Runtime.InteropServices.OSPlatform]::Windows)) {
        return $null
    }
    return ,([AiDotNetBenchmarkHostCpu]::Snapshot())
}

function Get-HostCpuUsagePercent($Before, $After) {
    if ($null -eq $Before -or $null -eq $After) { return $null }
    return [AiDotNetBenchmarkHostCpu]::UsagePercent($Before, $After)
}

function Get-AdjustedForeignCpuPercent([double]$UsagePercent) {
    $busyProcessors = $UsagePercent * [Environment]::ProcessorCount / 100.0
    return 100.0 * [Math]::Max(0.0, $busyProcessors - $benchmarkOwnedCpuAllowance) /
        [Environment]::ProcessorCount
}

function Assert-HostReady([string]$Label) {
    $before = Get-HostCpuSnapshot
    if ($null -eq $before) { return }
    Start-Sleep -Milliseconds 250
    $after = Get-HostCpuSnapshot
    $usage = Get-HostCpuUsagePercent $before $after
    if ($usage -gt $hostCpuCeilingPercent) {
        throw "[$Label] Host CPU utilization $($usage.ToString('F1'))% exceeds the $($hostCpuCeilingPercent.ToString('F1'))% evidence ceiling."
    }
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
            if (-not $AfterSuite -and ($utilization -gt 20 -or $usedMegabytes -gt 2048)) {
                throw "[$Label] GPU is not benchmark-ready (utilization=$utilization%, memory.used=$usedMegabytes MiB, temperature=$temperatureCelsius C)."
            }
            if ($AfterSuite -and $utilization -gt 20) {
                # Utilization is a trailing NVIDIA sample, so the first snapshot can
                # still describe the child process that just exited. Allow that sample
                # a bounded time to age out, but fail closed on sustained activity.
                for ($attempt = 1; $attempt -lt 6 -and $utilization -gt 20; $attempt++) {
                    Start-Sleep -Milliseconds 250
                    $tail = & nvidia-smi '--query-gpu=utilization.gpu' '--format=csv,noheader,nounits' 2>&1
                    if ($LASTEXITCODE -ne 0 -or -not [int]::TryParse(($tail -join '').Trim(), [ref]$utilization)) {
                        throw "[$Label] could not sample post-suite GPU utilization: $tail"
                    }
                }
                if ($utilization -gt 20) {
                    throw "[$Label] GPU utilization remains $utilization% after 6 post-suite quiescence samples, above the 20% evidence ceiling."
                }
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
    if ($Issue853Only) {
        $suites.Add((New-EvidenceSuite 'solvers-4x4' 'dotnet' @(
            $targetDll, '--direct-ptx-solvers-4x4', '1', '--component-only')))
    }
    if (-not $Issue834Only -and -not $Issue835Only -and -not $Issue853Only) {
        $suites.Add((New-EvidenceSuite 'online-attention' 'dotnet' @($targetDll, '--direct-ptx-online-attention')))
        $suites.Add((New-EvidenceSuite 'gpu-matrix' 'dotnet' @($targetDll, '--direct-ptx-gpu-matrix')))
        $suites.Add((New-EvidenceSuite 'residual-rmsnorm' 'dotnet' @($targetDll, '--direct-ptx-residual-rmsnorm')))
        if (-not $SkipExternal) {
            $suites.Add((New-EvidenceSuite 'external-gpu-baselines' 'dotnet' @($targetDll, '--direct-ptx-external-gpu-baselines')))
        }
    }

    if (-not $Issue835Only -and -not $Issue853Only) {
        $suites.Add((New-EvidenceSuite 'attention-family' 'dotnet' @($targetDll, '--direct-ptx-attention-family', '1')))
        $suites.Add((New-EvidenceSuite 'decode' 'dotnet' @($targetDll, '--direct-ptx-decode', '1')))
        $suites.Add((New-EvidenceSuite 'paged-prefill' 'dotnet' @($targetDll, '--direct-ptx-paged-prefill', '1')))
        $suites.Add((New-EvidenceSuite 'attention-backward' 'dotnet' @($targetDll, '--direct-ptx-attention-backward', '1')))
        $suites.Add((New-EvidenceSuite 'flash-attention-backward' 'dotnet' @($targetDll, '--direct-ptx-flash-attention-backward', '1')))
    }
    if (-not $Issue834Only -and -not $Issue853Only) {
        $suites.Add((New-EvidenceSuite 'qkv-rope-cache' 'dotnet' @(
            $targetDll, '--direct-ptx-qkv-rope-cache', '1', '--no-external')))
    }

    if (-not $SkipExternal) {
        $python = (Get-Command python -ErrorAction Stop).Source
        if ($Issue853Only) {
            $suites.Add((New-EvidenceSuite 'solvers-4x4-pytorch' $python @(
                (Join-Path $pythonRoot 'run_direct_ptx_solver4x4_competitors.py'), '--runs', '1')))
        }
        if (-not $Issue835Only -and -not $Issue853Only) {
            $suites.Add((New-EvidenceSuite 'attention-family-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_attention_family_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'decode-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_decode_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'paged-prefill-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_paged_prefill_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'attention-backward-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_attention_backward_competitors.py'), '--runs', '1')))
            $suites.Add((New-EvidenceSuite 'flash-attention-backward-pytorch' $python @((Join-Path $pythonRoot 'run_direct_ptx_flash_attention_backward_competitors.py'), '--runs', '1')))
        }
        if (-not $Issue834Only -and -not $Issue853Only) {
            $suites.Add((New-EvidenceSuite 'qkv-rope-cache-pytorch' $python @(
                (Join-Path $pythonRoot 'run_direct_ptx_qkv_rope_cache_competitors.py'), '--runs', '1', '--json-lines')))
        }
    }

    $previousDirectPtx = $env:AIDOTNET_DIRECT_PTX
    $previousAutotune = $env:AIDOTNET_DIRECT_PTX_AUTOTUNE
    $autotuneValue = if ($Issue853Only) { '1' } else { '0' }
    $env:AIDOTNET_DIRECT_PTX = '1'
    $env:AIDOTNET_DIRECT_PTX_AUTOTUNE = $autotuneValue
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
                            Assert-HostReady "$label-start"
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
                    "# git_commit=$gitCommit; dirty_worktree=$($dirtyLines.Count -ne 0); AIDOTNET_DIRECT_PTX=1; AIDOTNET_DIRECT_PTX_AUTOTUNE=$autotuneValue" |
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
                    $hostCpuBefore = Get-HostCpuSnapshot
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
                    $hostCpuAfter = Get-HostCpuSnapshot
                    $hostCpuUsage = Get-HostCpuUsagePercent $hostCpuBefore $hostCpuAfter
                    $adjustedForeignCpu = if ($null -eq $hostCpuUsage) {
                        $null
                    }
                    else {
                        Get-AdjustedForeignCpuPercent $hostCpuUsage
                    }
                    if ($null -ne $hostCpuUsage) {
                        "# host_cpu_average_percent=$($hostCpuUsage.ToString('F2')); adjusted_foreign_percent=$($adjustedForeignCpu.ToString('F2')); logical_processors=$([Environment]::ProcessorCount)" |
                            Add-Content -LiteralPath $log -Encoding utf8
                    }
                    "# ending_gpu=$(Get-GpuSnapshot)" | Add-Content -LiteralPath $log -Encoding utf8
                    "# completed_utc=$([DateTime]::UtcNow.ToString('O')); exit_code=$exitCode" |
                        Add-Content -LiteralPath $log -Encoding utf8
                    if ($exitCode -ne 0) {
                        throw "Evidence suite '$($suite.Name)' run $run failed with exit code $exitCode. See '$log'."
                    }

                    $rejection = $null
                    $rejectionKind = 'environment'
                    try { Assert-GpuReady "$label-end" -AfterSuite }
                    catch { $rejection = $_.Exception.Message }
                    if (-not $rejection -and $null -ne $adjustedForeignCpu -and
                        $adjustedForeignCpu -gt $hostCpuCeilingPercent) {
                        $rejection = "[$label] Average adjusted foreign CPU utilization $($adjustedForeignCpu.ToString('F1'))% exceeded the $($hostCpuCeilingPercent.ToString('F1'))% evidence ceiling."
                    }
                    if (-not $rejection -and $Issue853Only -and $suite.Name -eq 'solvers-4x4') {
                        try { Assert-SolverDotnetAcceptedAttempt $log $run }
                        catch {
                            $rejection = $_.Exception.Message
                            $rejectionKind = 'internal_gate'
                        }
                    }
                    if ($rejection) {
                        "# rejected_$rejectionKind=$rejection" | Add-Content -LiteralPath $log -Encoding utf8
                        $rejected = Join-Path $evidenceRoot ("{0}-attempt-{1:D2}.rejected.txt" -f $label, $attempt)
                        Move-Item -LiteralPath $log -Destination $rejected -Force
                        if ($attempt -gt $ContaminationRetries) {
                            throw "Evidence suite '$($suite.Name)' run $run was rejected after $attempt attempts. Last reason: $rejection"
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

    if ($Issue853Only) {
        Assert-SolverReleaseGate $evidenceRoot $Runs (-not [bool]$SkipExternal)
    }
    elseif (-not $Issue834Only) {
        Assert-QkvReleaseGate $evidenceRoot $Runs (-not [bool]$SkipExternal)
    }

    $files = Get-ChildItem -LiteralPath $evidenceRoot -File | Where-Object {
        $_.Extension -eq '.log' -or
        $_.Name -eq 'qkv-release-gate.json' -or
        $_.Name -eq 'solver-release-gate.json'
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
        maximum_adjusted_foreign_host_cpu_percent = $hostCpuCeilingPercent
        benchmark_owned_cpu_allowance = $benchmarkOwnedCpuAllowance
        issue_834_only = [bool]$Issue834Only
        issue_835_only = [bool]$Issue835Only
        issue_853_only = [bool]$Issue853Only
        external_gpu_baselines_included = -not [bool]$SkipExternal
        feature_gates = [ordered]@{
            AIDOTNET_DIRECT_PTX = '1'
            AIDOTNET_DIRECT_PTX_AUTOTUNE = $autotuneValue
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
