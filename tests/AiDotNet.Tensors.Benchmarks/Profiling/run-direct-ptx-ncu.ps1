param(
    [ValidateSet('attention', 'residual-rmsnorm', 'decode', 'paged-prefill', 'attention-backward', 'flash-attention-backward')]
    [string]$Target = 'attention',
    [string]$OutputCsv = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-direct-ptx-ncu-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.csv')),
    [string]$NcuPath = $env:NSIGHT_COMPUTE_CLI
)

$ErrorActionPreference = 'Stop'
$ncuSource = if ($NcuPath) {
    if (-not (Test-Path -LiteralPath $NcuPath -PathType Leaf)) {
        throw "Nsight Compute CLI was not found at '$NcuPath'."
    }
    (Resolve-Path -LiteralPath $NcuPath).Path
} else {
    (Get-Command ncu -ErrorAction Stop).Source
}
$targetDll = Join-Path $PSScriptRoot '..\bin\Release\net10.0\AiDotNet.Tensors.Benchmarks.dll'
if (-not (Test-Path -LiteralPath $targetDll -PathType Leaf)) {
    throw "Benchmark target is missing. Build AiDotNet.Tensors.Benchmarks in Release/net10.0 first."
}
$targetDll = (Resolve-Path -LiteralPath $targetDll).Path
$switch = switch ($Target) {
    'attention' { '--direct-ptx-profile-attention' }
    'residual-rmsnorm' { '--direct-ptx-profile-residual-rmsnorm' }
    'decode' { '--direct-ptx-profile-decode' }
    'paged-prefill' { '--direct-ptx-profile-paged-prefill' }
    'attention-backward' { '--direct-ptx-profile-attention-backward' }
    'flash-attention-backward' { '--direct-ptx-profile-flash-attention-backward' }
}
$kernel = switch ($Target) {
    'attention' { 'regex:aidotnet_online_attention_128x64' }
    'residual-rmsnorm' { 'regex:aidotnet_fused_residual_rmsnorm_d64' }
    'decode' { 'regex:aidotnet_(flash|paged)_decode_d64' }
    'paged-prefill' { 'regex:aidotnet_paged_prefill_d64' }
    'attention-backward' { 'regex:aidotnet_attention_backward_(delta|dq|dkv)_d64' }
    'flash-attention-backward' { 'regex:aidotnet_flash_attention_backward_(dq|dkv)_d64' }
}
$expectedLaunches = if ($Target -eq 'attention') { 16 } else { 4 }
$metricNames = @(
    'sass__inst_executed_register_spilling',
    'sass__inst_executed_register_spilling_mem_local',
    'sass__inst_executed_local_loads',
    'sass__inst_executed_local_stores',
    'launch__registers_per_thread',
    'launch__shared_mem_per_block_static',
    'launch__shared_mem_per_block_dynamic',
    'sm__maximum_warps_per_active_cycle_pct',
    'sm__warps_active.avg.pct_of_peak_sustained_active'
)
$metrics = $metricNames -join ','

& $ncuSource `
    --target-processes all `
    --kernel-name $kernel `
    --metrics $metrics `
    --csv `
    --page raw `
    --force-overwrite `
    --log-file $OutputCsv `
    dotnet $targetDll $switch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

dotnet $targetDll --direct-ptx-verify-ncu $OutputCsv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# One deterministic launch is emitted for every admitted specialization:
# attention = four sequence buckets x causal/plain x fused/unfused; residual =
# four row buckets. Requiring one raw row per requested metric and launch keeps
# a partial capture from being mistaken for whole-domain zero-spill evidence.
foreach ($metricName in $metricNames) {
    $pattern = '"' + [Regex]::Escape($metricName) + '(?:\.[^",]+)?",'
    $matchCount = @(
        Select-String -LiteralPath $OutputCsv -Pattern $pattern -AllMatches | ForEach-Object {
            $_.Matches
        }
    ).Count
    if ($matchCount -ne $expectedLaunches) {
        throw "Nsight evidence is incomplete for '$metricName': expected $expectedLaunches launch rows, found $matchCount."
    }
}

Write-Host "Nsight evidence verified for $expectedLaunches '$Target' specializations: $OutputCsv"
