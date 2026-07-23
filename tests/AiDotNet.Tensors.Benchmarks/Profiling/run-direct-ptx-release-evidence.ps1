[CmdletBinding()]
param(
    [ValidateRange(3, 20)]
    [int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-direct-ptx-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [switch]$SkipBuild,
    [switch]$SkipExternal,
    [switch]$Issue834Only,
    [switch]$Issue835Only,
    [switch]$Issue836Only,
    [string]$DenseLinearNcuCsv,
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

function Read-DenseLinearDotnetRows([string]$Path) {
    $prefix = 'dense_linear_evidence_json='
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.StartsWith($prefix, [StringComparison]::Ordinal)
    } | ForEach-Object {
        $_.Substring($prefix.Length) | ConvertFrom-Json
    })
}

function Read-DenseLinearEnvironment([string]$Path) {
    $prefix = 'dense_linear_environment_json='
    return @(Get-Content -LiteralPath $Path | Where-Object {
        $_.StartsWith($prefix, [StringComparison]::Ordinal)
    } | ForEach-Object {
        $_.Substring($prefix.Length) | ConvertFrom-Json
    })
}

function Assert-DenseLinearRow([object]$Row, [string]$Source) {
    foreach ($property in @(
        'work_flops', 'device_mean_us', 'device_median_us', 'device_p95_us',
        'device_p99_us', 'e2e_mean_us', 'e2e_median_us', 'e2e_p95_us',
        'e2e_p99_us', 'tflops', 'gflops', 'tolerance')) {
        $value = [double]$Row.$property
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0) {
            throw "Dense-linear evidence has invalid $property='$($Row.$property)' for '$($Row.method)' in '$Source'."
        }
    }
    $error = [double]$Row.max_error
    if ([double]::IsNaN($error) -or [double]::IsInfinity($error) -or $error -lt 0) {
        throw "Dense-linear evidence has invalid max_error='$($Row.max_error)' for '$($Row.method)' in '$Source'."
    }
    foreach ($property in @(
        'managed_bytes', 'temporary_device_allocation_count',
        'temporary_device_bytes', 'output_device_bytes', 'peak_device_bytes')) {
        if ($null -ne $Row.$property -and [long]$Row.$property -lt 0) {
            throw "Dense-linear evidence has invalid $property='$($Row.$property)' for '$($Row.method)' in '$Source'."
        }
    }
    if ([double]$Row.device_p95_us -lt [double]$Row.device_median_us -or
        [double]$Row.device_p99_us -lt [double]$Row.device_p95_us -or
        [double]$Row.e2e_p95_us -lt [double]$Row.e2e_median_us -or
        [double]$Row.e2e_p99_us -lt [double]$Row.e2e_p95_us) {
        throw "Dense-linear evidence has non-monotonic percentiles for '$($Row.method)' in '$Source'."
    }
    $expectedOperationsPerCall = if ([string]$Row.method -like '* graph') { 50 } else { 1 }
    if ([int]$Row.logical_operations_per_call -ne $expectedOperationsPerCall) {
        throw "Dense-linear evidence expected logical_operations_per_call=$expectedOperationsPerCall for '$($Row.method)' in '$Source'; found '$($Row.logical_operations_per_call)'."
    }
}

function Read-DenseLinearNcuProof([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $metricNames = @(
        'sass__inst_executed_register_spilling',
        'sass__inst_executed_register_spilling_mem_local',
        'sass__inst_executed_register_spilling_mem_shared',
        'smsp__sass_inst_executed_op_local.sum',
        'smsp__sass_inst_executed_op_local_ld.sum',
        'smsp__sass_inst_executed_op_local_st.sum',
        'l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum',
        'l1tex__t_requests_pipe_lsu_mem_local_op_st.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_cmd_read.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_cmd_write.sum',
        'launch__registers_per_thread',
        'launch__shared_mem_per_block_static',
        'launch__shared_mem_per_block_dynamic',
        'sm__maximum_warps_per_active_cycle_pct',
        'sm__warps_active.avg.pct_of_peak_sustained_active'
    )
    $zeroMetricNames = @($metricNames[0..7])
    $metricSums = @{}
    $csvLines = @(Get-Content -LiteralPath $resolved)
    $headerIndex = -1
    for ($index = 0; $index -lt $csvLines.Count; $index++) {
        if ($csvLines[$index].StartsWith('"ID","Process ID","Process Name"', [StringComparison]::Ordinal)) {
            $headerIndex = $index
            break
        }
    }
    if ($headerIndex -lt 0) {
        throw "Dense-linear Nsight proof does not contain a raw CSV header: '$resolved'."
    }
    $records = @(($csvLines[$headerIndex..($csvLines.Count - 1)] -join "`n") |
        ConvertFrom-Csv | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_.'Kernel Name')
        })
    if ($records.Count -ne 16) {
        throw "Dense-linear Nsight proof expected 16 launch rows; found $($records.Count) in '$resolved'."
    }
    foreach ($metricName in $metricNames) {
        $metricSum = 0.0
        if ($records[0].PSObject.Properties.Name -notcontains $metricName) {
            throw "Dense-linear Nsight proof is missing '$metricName' in '$resolved'."
        }
        foreach ($record in $records) {
            $text = [string]$record.$metricName
            $value = 0.0
            if ([string]::IsNullOrWhiteSpace($text) -or
                -not [double]::TryParse(
                    $text,
                    [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands,
                    [Globalization.CultureInfo]::InvariantCulture,
                    [ref]$value) -or
                [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
                throw "Dense-linear Nsight proof has an invalid '$metricName' value '$text' in '$resolved'."
            }
            if ($zeroMetricNames -contains $metricName -and $value -ne 0) {
                throw "Dense-linear Nsight zero-spill proof failed: '$metricName'='$value' for '$($record.'Kernel Name')'."
            }
            $metricSum += $value
        }
        $metricSums[$metricName] = $metricSum
    }
    $hash = Get-FileHash -LiteralPath $resolved -Algorithm SHA256
    return [ordered]@{
        status = 'pass'
        file = $resolved
        sha256 = $hash.Hash.ToLowerInvariant()
        exact_launch_rows = $records.Count
        zero_metric_groups = $zeroMetricNames.Count
        shared_bank_metric_groups = 2
        shared_bank_read_conflicts_total =
            $metricSums['l1tex__data_bank_conflicts_pipe_lsu_cmd_read.sum']
        shared_bank_write_conflicts_total =
            $metricSums['l1tex__data_bank_conflicts_pipe_lsu_cmd_write.sum']
        requested_metric_groups = $metricNames.Count
    }
}

function Write-DenseLinearMarkdown(
    [string]$Root,
    [object[]]$Rows,
    [object[]]$Environments,
    [object[]]$Verdicts,
    [int]$RunCount,
    [bool]$IncludeExternal,
    [object]$NcuProof) {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('# Issue #836 direct-PTX dense/linear evidence')
    $lines.Add('')
    $lines.Add("Generated from $RunCount independent clean process runs; each cell uses at least 30 logical warmups and 101 samples. Eager samples contain 50 ordinary launches; graph samples replay one captured sequence of 50 logical operations and normalize per operation.")
    $lines.Add('This graph contract measures GPU execution inside a model-scale captured graph instead of repeatedly exposing one-node cuGraphLaunch host-submission latency. Latency columns are normalized per logical operation; rate is GEMM-equivalent TFLOPS / GFLOPS. R/S/D/L/B means registers/thread, static shared bytes, dynamic shared bytes, local bytes/thread, and active blocks/SM; T/max is launched/max threads, and PTX/SASS are driver-reported versions.')
    $lines.Add('Temporary device bytes exclude required result storage. .NET rows report logical device-allocation count/bytes, including pooled buffers; PyTorch reports raw peak growth and de-aliased result-storage bytes in the source JSONL.')
    $lines.Add('')
    $lines.Add('## Environment fingerprint')
    $lines.Add('')
    $lines.Add('| Run | OS / framework | process | GPU / UUID | SM / driver | SM limits | benchmark contract |')
    $lines.Add('|---:|---|---|---|---|---|---|')
    foreach ($environment in @($Environments | Sort-Object evidence_run)) {
        $lines.Add("| $($environment.evidence_run) | $($environment.os) / $($environment.framework) | $($environment.process_architecture), $($environment.processor_count) logical CPUs, server GC=$($environment.server_gc) | $($environment.gpu) / $($environment.gpu_uuid) | $($environment.compute_capability) / $($environment.cuda_driver_version) | $($environment.max_threads_per_sm) threads/SM | $($environment.warmups) logical warmups, $($environment.samples) samples, eager=$($environment.launches_per_device_sample)/sample, graph=$($environment.graph_operations_per_replay)/replay |")
    }
    $pythonFingerprints = @($Rows | Where-Object {
        $null -ne $_.pytorch_version -and $null -ne $_.python_version
    } | Group-Object python_version,pytorch_version,pytorch_cuda_version,device_name,compute_capability,float32_matmul_precision)
    if ($pythonFingerprints.Count -ne 0) {
        $lines.Add('')
        $lines.Add('Python competitors: ' + (@($pythonFingerprints | ForEach-Object {
            $row = $_.Group[0]
            "Python $($row.python_version), PyTorch $($row.pytorch_version), CUDA $($row.pytorch_cuda_version), $($row.device_name) SM $($row.compute_capability), FP32 precision=$($row.float32_matmul_precision)"
        }) -join '; '))
    }
    $lines.Add('')
    $lines.Add('## Promotion gate')
    $lines.Add('')
    if ($null -eq $NcuProof) {
        $lines.Add('Nsight executed-spill proof: **HOLD** (no authenticated dense-linear CSV was supplied).')
    } else {
        $lines.Add("Nsight executed-spill proof: **PASS** - $($NcuProof.exact_launch_rows) exact launches, $($NcuProof.zero_metric_groups) zero local/spill metric groups, SHA-256 ``$($NcuProof.sha256)``.")
        $lines.Add("Nsight shared-bank evidence: read-conflict total $($NcuProof.shared_bank_read_conflicts_total), write-conflict total $($NcuProof.shared_bank_write_conflicts_total) across the same exact launches.")
    }
    $lines.Add('')
    $lines.Add('| Operation | Eligible competitors evaluated | min device speedup | min E2E speedup | max P95 ratio | timing verdict |')
    $lines.Add('|---|---|---:|---:|---:|---|')
    foreach ($operation in @($Rows.operation | Sort-Object -Unique)) {
        $operationVerdicts = @($Verdicts | Where-Object { $_.operation -eq $operation })
        $competitors = @($operationVerdicts.competitor | Sort-Object -Unique) -join ' / '
        $minDevice = ($operationVerdicts.device_median_speedup | Measure-Object -Minimum).Minimum
        $minE2e = ($operationVerdicts.e2e_median_speedup | Measure-Object -Minimum).Minimum
        $maxP95 = ($operationVerdicts.device_p95_ratio | Measure-Object -Maximum).Maximum
        $passed = @($operationVerdicts | Where-Object { -not $_.timing_gate_passed }).Count -eq 0
        $verdict = if ($passed) { '**TIMING PASS**' } else { 'FAIL - remains experimental' }
        $lines.Add("| $operation | $competitors | $('{0:F2}x' -f $minDevice) | $('{0:F2}x' -f $minE2e) | $('{0:F2}x' -f $maxP95) | $verdict |")
    }

    $lines.Add('')
    $lines.Add('## Full grouped results')
    foreach ($operation in @($Rows.operation | Sort-Object -Unique)) {
        $operationRows = @($Rows | Where-Object { $_.operation -eq $operation })
        $shape = $operationRows[0].shape
        $methodGroups = @($operationRows | Group-Object method)
        $winner = $methodGroups | Sort-Object {
            (@($_.Group.device_median_us) | Measure-Object -Average).Average
        } | Select-Object -First 1
        $lines.Add('')
        $lines.Add("### $operation - $shape")
        $lines.Add('')
        $lines.Add('| Method | med R1/R2/R3 us | worst P95 | worst P99 | avg mean | E2E med R1/R2/R3 | E2E worst P95 | E2E worst P99 | E2E avg mean | TFLOPS / GFLOPS | managed B | temp alloc/B | max error | R/S/D/L/B; T/max; PTX/SASS | verdict |')
        $lines.Add('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
        foreach ($methodGroup in $methodGroups) {
            $group = @($methodGroup.Group | Sort-Object evidence_run)
            $method = $methodGroup.Name
            if ($methodGroup.Name -eq $winner.Name) { $method = "**$method - WINNER**" }
            $deviceMedians = @($group | ForEach-Object { '{0:F2}' -f [double]$_.device_median_us }) -join '/'
            $e2eMedians = @($group | ForEach-Object { '{0:F2}' -f [double]$_.e2e_median_us }) -join '/'
            $worstP95 = ($group.device_p95_us | Measure-Object -Maximum).Maximum
            $worstP99 = ($group.device_p99_us | Measure-Object -Maximum).Maximum
            $mean = ($group.device_mean_us | Measure-Object -Average).Average
            $worstE2eP95 = ($group.e2e_p95_us | Measure-Object -Maximum).Maximum
            $worstE2eP99 = ($group.e2e_p99_us | Measure-Object -Maximum).Maximum
            $e2eMean = ($group.e2e_mean_us | Measure-Object -Average).Average
            $rates = @($group.gflops | Sort-Object)
            $gflops = [double]$rates[[Math]::Floor($rates.Count / 2)]
            $managedValues = @($group | Where-Object { $null -ne $_.managed_bytes } |
                ForEach-Object { [long]$_.managed_bytes })
            $managed = if ($managedValues.Count -eq 0) { 'n/a' } else {
                [string](($managedValues | Measure-Object -Maximum).Maximum)
            }
            $tempValues = @($group | Where-Object { $null -ne $_.temporary_device_bytes } |
                ForEach-Object { [long]$_.temporary_device_bytes })
            $temp = if ($tempValues.Count -eq 0) { 'n/a' } else {
                [string](($tempValues | Measure-Object -Maximum).Maximum)
            }
            $tempCountValues = @($group |
                Where-Object { $null -ne $_.temporary_device_allocation_count } |
                ForEach-Object { [long]$_.temporary_device_allocation_count })
            $tempCount = if ($tempCountValues.Count -eq 0) { 'n/a' } else {
                [string](($tempCountValues | Measure-Object -Maximum).Maximum)
            }
            $maxError = ($group.max_error | Measure-Object -Maximum).Maximum
            $resource = @($group | Where-Object { $null -ne $_.registers -and [int]$_.registers -ge 0 } |
                Select-Object -First 1)
            $resources = if ($resource.Count -eq 0) { 'n/a' } else {
                "$($resource[0].registers)/$($resource[0].static_shared_bytes)/$($resource[0].dynamic_shared_bytes)/$($resource[0].local_bytes_per_thread)/$($resource[0].active_blocks_per_sm); $($resource[0].block_threads)/$($resource[0].max_threads_per_block); $($resource[0].ptx_version)/$($resource[0].binary_version)"
            }
            $verdict = '-'
            if ($methodGroup.Name -like 'Direct PTX* graph') {
                $operationVerdicts = @($Verdicts | Where-Object { $_.operation -eq $operation })
                $passed = @($operationVerdicts | Where-Object { -not $_.timing_gate_passed }).Count -eq 0
                $minDevice = ($operationVerdicts.device_median_speedup | Measure-Object -Minimum).Minimum
                $minE2e = ($operationVerdicts.e2e_median_speedup | Measure-Object -Minimum).Minimum
                $maxRatio = ($operationVerdicts.device_p95_ratio | Measure-Object -Maximum).Maximum
                $verdict = if ($passed) {
                    "**TIMING PASS - $('{0:F2}x' -f $minDevice) device / $('{0:F2}x' -f $minE2e) E2E; $('{0:F2}x' -f $maxRatio) P95**"
                } else {
                    "FAIL - $('{0:F2}x' -f $minDevice) device / $('{0:F2}x' -f $minE2e) E2E; $('{0:F2}x' -f $maxRatio) P95"
                }
            }
            $lines.Add("| $method | $deviceMedians | $('{0:F2}' -f $worstP95) | $('{0:F2}' -f $worstP99) | $('{0:F2}' -f $mean) | $e2eMedians | $('{0:F2}' -f $worstE2eP95) | $('{0:F2}' -f $worstE2eP99) | $('{0:F2}' -f $e2eMean) | $('{0:F3} / {1:F2}' -f ($gflops / 1000.0), $gflops) | $managed | $tempCount/$temp | $('{0:E2}' -f $maxError) | $resources | $verdict |")
        }
    }
    $lines.Add('')
    $lines.Add('Timing qualification is not release promotion. Executed Nsight spill/local-memory counters are still mandatory for every promoted specialization.')
    [IO.File]::WriteAllLines((Join-Path $Root 'dense-linear-results.md'), $lines)
}

function Assert-DenseLinearEvidence(
    [string]$Root,
    [int]$RunCount,
    [bool]$IncludeExternal,
    [object]$NcuProof) {
    $operations = @(
        'decode-gelu', 'gemm-fp32', 'fused-gelu', 'batched-gemm',
        'gemm-fp16', 'fused-gelu-fp16-m16-k512',
        'fused-gelu-fp16-m16-k1024', 'lora',
        'linear-ce-index', 'linear-backward-relu',
        'dot', 'outer', 'batched-dot', 'strided-dot'
    )
    $verdicts = [System.Collections.Generic.List[object]]::new()
    $allRows = [System.Collections.Generic.List[object]]::new()
    $environments = [System.Collections.Generic.List[object]]::new()
    $allPassed = $true
    for ($run = 1; $run -le $RunCount; $run++) {
        $prefix = 'run-{0:D2}' -f $run
        $dotnetPath = Join-Path $Root ($prefix + '-dense-linear.log')
        $environmentRows = @(Read-DenseLinearEnvironment $dotnetPath)
        if ($environmentRows.Count -ne 1) {
            throw "Dense-linear evidence expected one environment row in '$dotnetPath'; found $($environmentRows.Count)."
        }
        $environmentRows[0] | Add-Member -NotePropertyName evidence_run -NotePropertyValue $run -Force
        if ([int]$environmentRows[0].graph_operations_per_replay -ne 50) {
            throw "Dense-linear evidence expected a 50-operation graph replay contract in '$dotnetPath'."
        }
        $environments.Add($environmentRows[0])
        $dotnetRows = @(Read-DenseLinearDotnetRows $dotnetPath)
        if ($dotnetRows.Count -ne 60) {
            throw "Dense-linear evidence expected 60 .NET rows in '$dotnetPath'; found $($dotnetRows.Count)."
        }
        foreach ($row in $dotnetRows) {
            Assert-DenseLinearRow $row $dotnetPath
            $row | Add-Member -NotePropertyName evidence_run -NotePropertyValue $run -Force
            $allRows.Add($row)
        }
        $pythonRows = @()
        if ($IncludeExternal) {
            $pythonPath = Join-Path $Root ($prefix + '-dense-linear-pytorch.log')
            $pythonRows = @(Read-QkvPythonRows $pythonPath | Where-Object { $_.status -eq 'ok' })
            if ($pythonRows.Count -ne 56) {
                throw "Dense-linear evidence expected 56 PyTorch rows (eager, graph, compile, and compile graph) in '$pythonPath'; found $($pythonRows.Count)."
            }
            foreach ($row in $pythonRows) {
                Assert-DenseLinearRow $row $pythonPath
                $row | Add-Member -NotePropertyName evidence_run -NotePropertyValue $run -Force
                $allRows.Add($row)
            }
        }

        foreach ($operation in $operations) {
            $directRows = @($dotnetRows | Where-Object {
                $_.operation -eq $operation -and $_.method -like 'Direct PTX*'
            })
            $candidate = @($directRows | Where-Object { $_.method -like 'Direct PTX* graph' })
            if ($candidate.Count -ne 1 -or $directRows.Count -ne 2) {
                throw "Dense-linear evidence has an incomplete Direct PTX method set for run $run '$operation'."
            }
            if (@($directRows | Where-Object {
                [double]$_.max_error -gt [double]$_.tolerance -or
                [long]$_.managed_bytes -ne 0 -or
                [long]$_.temporary_device_allocation_count -ne 0 -or
                [long]$_.temporary_device_bytes -ne 0 -or
                [int]$_.dynamic_shared_bytes -ne 0 -or
                [int]$_.local_bytes_per_thread -ne 0 -or
                [int]$_.block_threads -le 0 -or
                [int]$_.max_threads_per_block -lt [int]$_.block_threads -or
                [int]$_.ptx_version -le 0 -or
                [int]$_.binary_version -le 0 -or
                $_.module_image_kind -ne 'EmbeddedCubin' -or
                [string]::IsNullOrWhiteSpace([string]$_.cubin_sha256) -or
                [string]::IsNullOrWhiteSpace([string]$_.cubin_source_key) -or
                [string]::IsNullOrWhiteSpace([string]$_.compiler_log)
            }).Count -ne 0) {
                throw "Dense-linear resource/correctness gate failed for run $run '$operation'."
            }

            $peers = @($dotnetRows | Where-Object {
                $_.operation -eq $operation -and $_.method -notlike 'Direct PTX*'
            })
            if ($IncludeExternal) {
                $shapePeers = @($pythonRows | Where-Object { $_.operation -eq $operation })
                if ($shapePeers.Count -ne 4) {
                    throw "Dense-linear evidence has an incomplete PyTorch method set for run $run '$operation'."
                }
                $peers += $shapePeers
            }
            if ($peers.Count -eq 0) {
                throw "Dense-linear evidence has no eligible competitor for run $run '$operation'."
            }
            foreach ($peer in $peers) {
                $peerTolerance = if ($null -ne $peer.tolerance) { [double]$peer.tolerance } else { 2e-4 }
                if ([double]$peer.max_error -gt $peerTolerance) {
                    throw "Dense-linear peer '$($peer.method)' exceeded tolerance for run $run '$operation'."
                }
            }
            foreach ($peer in $peers) {
                $deviceSpeedup = [double]$peer.device_median_us / [double]$candidate[0].device_median_us
                $endToEndSpeedup = [double]$peer.e2e_median_us / [double]$candidate[0].e2e_median_us
                $p95Ratio = [double]$candidate[0].device_p95_us / [double]$peer.device_p95_us
                $timingPassed = $deviceSpeedup -ge 1.10 -and
                    $endToEndSpeedup -ge 1.10 -and $p95Ratio -le 1.10
                if (-not $timingPassed) { $allPassed = $false }
                $verdicts.Add([ordered]@{
                    run = $run
                    operation = $operation
                    candidate = $candidate[0].method
                    competitor = $peer.method
                    device_median_speedup = $deviceSpeedup
                    e2e_median_speedup = $endToEndSpeedup
                    device_p95_ratio = $p95Ratio
                    timing_gate_passed = $timingPassed
                })
            }
        }
    }
    $gatePath = Join-Path $Root 'dense-linear-release-gate.json'
    $spillProofPassed = $null -ne $NcuProof -and $NcuProof.status -eq 'pass'
    $promotionGatePassed = $allPassed -and $spillProofPassed
    [ordered]@{
        status = if ($promotionGatePassed) {
            'promotion-gate-pass'
        } elseif ($allPassed) {
            'timing-pass-spill-proof-hold'
        } else {
            'timing-fail'
        }
        promotion_gate_passed = $promotionGatePassed
        release_promoted = $false
        release_blocker = if ($promotionGatePassed) {
            'None. The measured candidate qualifies for an explicit production-routing change and post-promotion revalidation.'
        } elseif ($allPassed) {
            'Executed Nsight spill/local-memory counters are not available for every timing-qualified specialization.'
        } else {
            'At least one candidate/competitor timing pair failed.'
        }
        required_device_and_e2e_median_speedup = 1.10
        maximum_device_p95_ratio = 1.10
        measurement_contract = 'Eager: 50 ordinary launches per device sample. Graph: one captured 50-operation sequence per sample, normalized per logical operation.'
        error_tolerance_policy = 'Per-row operation-specific tolerance; 2e-3 for FP16 fused-linear Tensor-Core shapes and 2e-4 otherwise.'
        runs = $RunCount
        external_competitors_included = $IncludeExternal
        ncu_proof = $NcuProof
        verdicts = @($verdicts)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gatePath -Encoding utf8
    Write-DenseLinearMarkdown $Root @($allRows) @($environments) @($verdicts) $RunCount $IncludeExternal $NcuProof
}

function Get-GpuSnapshot {
    $output = & nvidia-smi `
        '--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used' `
        '--format=csv,noheader,nounits' 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi snapshot failed: $output" }
    return ($output -join [Environment]::NewLine).Trim()
}

function Assert-GpuReady([string]$Label, [switch]$AfterSuite) {
    $pythonProcesses = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -in @('python', 'python3', 'pythonw')
    })
    if ($pythonProcesses.Count -ne 0) {
        $pythonConflicts = @($pythonProcesses | ForEach-Object {
            "pid=$($_.Id) $($_.ProcessName)"
        }) -join '; '
        throw "[$Label] OS-level Python workload detected before CUDA registration; clean benchmark refused: $pythonConflicts"
    }

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
    $issueOnlyCount = @(@($Issue834Only, $Issue835Only, $Issue836Only) |
        Where-Object { $_ }).Count
    if ($issueOnlyCount -gt 1) {
        throw '-Issue834Only, -Issue835Only, and -Issue836Only are mutually exclusive.'
    }
    if (-not [string]::IsNullOrWhiteSpace($DenseLinearNcuCsv) -and -not $Issue836Only) {
        throw '-DenseLinearNcuCsv is valid only with -Issue836Only.'
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
    if ($Issue836Only) {
        $suites.Add((New-EvidenceSuite 'dense-linear' 'dotnet' @(
            $targetDll, '--direct-ptx-dense-linear-full', '1', '--no-python')))
        if (-not $SkipExternal) {
            $python = (Get-Command python -ErrorAction Stop).Source
            $suites.Add((New-EvidenceSuite 'dense-linear-pytorch' $python @(
                (Join-Path $pythonRoot 'run_direct_ptx_dense_linear_full_competitors.py'),
                '--runs', '1', '--json-lines')))
        }
    }
    elseif (-not $Issue834Only -and -not $Issue835Only) {
        $suites.Add((New-EvidenceSuite 'online-attention' 'dotnet' @($targetDll, '--direct-ptx-online-attention')))
        $suites.Add((New-EvidenceSuite 'gpu-matrix' 'dotnet' @($targetDll, '--direct-ptx-gpu-matrix')))
        $suites.Add((New-EvidenceSuite 'residual-rmsnorm' 'dotnet' @($targetDll, '--direct-ptx-residual-rmsnorm')))
        if (-not $SkipExternal) {
            $suites.Add((New-EvidenceSuite 'external-gpu-baselines' 'dotnet' @($targetDll, '--direct-ptx-external-gpu-baselines')))
        }
    }

    if (-not $Issue836Only -and -not $Issue835Only) {
        $suites.Add((New-EvidenceSuite 'attention-family' 'dotnet' @($targetDll, '--direct-ptx-attention-family', '1')))
        $suites.Add((New-EvidenceSuite 'decode' 'dotnet' @($targetDll, '--direct-ptx-decode', '1')))
        $suites.Add((New-EvidenceSuite 'paged-prefill' 'dotnet' @($targetDll, '--direct-ptx-paged-prefill', '1')))
        $suites.Add((New-EvidenceSuite 'attention-backward' 'dotnet' @($targetDll, '--direct-ptx-attention-backward', '1')))
        $suites.Add((New-EvidenceSuite 'flash-attention-backward' 'dotnet' @($targetDll, '--direct-ptx-flash-attention-backward', '1')))
    }
    if (-not $Issue836Only -and -not $Issue834Only) {
        $suites.Add((New-EvidenceSuite 'qkv-rope-cache' 'dotnet' @(
            $targetDll, '--direct-ptx-qkv-rope-cache', '1', '--no-external')))
    }

    if (-not $Issue836Only -and -not $SkipExternal) {
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
    $previousPath = $env:PATH
    $env:AIDOTNET_DIRECT_PTX = '1'
    $env:AIDOTNET_DIRECT_PTX_AUTOTUNE = '0'
    $nativeRuntime = Join-Path (Split-Path $targetDll -Parent) 'runtimes\win-x64\native'
    if (Test-Path -LiteralPath $nativeRuntime -PathType Container) {
        $env:PATH = $nativeRuntime + [IO.Path]::PathSeparator + $env:PATH
    }
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
        $env:PATH = $previousPath
    }

    $denseLinearNcuProof = $null
    if ($Issue836Only) {
        $denseLinearNcuProof = Read-DenseLinearNcuProof $DenseLinearNcuCsv
    }
    if (-not $Issue834Only -and -not $Issue836Only) {
        Assert-QkvReleaseGate $evidenceRoot $Runs (-not [bool]$SkipExternal)
    }
    if ($Issue836Only) {
        Assert-DenseLinearEvidence $evidenceRoot $Runs (-not [bool]$SkipExternal) $denseLinearNcuProof
    }

    $files = Get-ChildItem -LiteralPath $evidenceRoot -File | Where-Object {
        $_.Extension -eq '.log' -or $_.Extension -eq '.md' -or
            $_.Name.EndsWith('-release-gate.json', [StringComparison]::Ordinal)
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
        issue_836_only = [bool]$Issue836Only
        dense_linear_ncu_proof = $denseLinearNcuProof
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
