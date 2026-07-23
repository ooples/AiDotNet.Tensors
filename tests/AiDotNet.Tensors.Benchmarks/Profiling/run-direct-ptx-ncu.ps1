param(
    [ValidateSet('attention', 'normalization', 'residual-rmsnorm', 'residual-layernorm-gelu', 'decode', 'paged-prefill', 'attention-backward', 'flash-attention-backward', 'qkv-rope-cache', 'fused-linear', 'mixed-linear', 'mixed-linear-m16', 'w8a8-linear')]
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
    'normalization' { '--direct-ptx-profile-normalization' }
    'residual-rmsnorm' { '--direct-ptx-profile-residual-rmsnorm' }
    'residual-layernorm-gelu' { '--direct-ptx-profile-residual-layernorm-gelu' }
    'decode' { '--direct-ptx-profile-decode' }
    'paged-prefill' { '--direct-ptx-profile-paged-prefill' }
    'attention-backward' { '--direct-ptx-profile-attention-backward' }
    'flash-attention-backward' { '--direct-ptx-profile-flash-attention-backward' }
    'qkv-rope-cache' { '--direct-ptx-profile-qkv-rope-cache' }
    'fused-linear' { '--direct-ptx-profile-fused-linear' }
    'mixed-linear' { '--direct-ptx-profile-mixed-linear' }
    'mixed-linear-m16' { '--direct-ptx-profile-mixed-linear-m16' }
    'w8a8-linear' { '--direct-ptx-profile-w8a8-linear' }
}
$kernel = switch ($Target) {
    'attention' { 'regex:aidotnet_online_attention_128x64' }
    'normalization' { 'regex:aidotnet_.*norm.*' }
    'residual-rmsnorm' { 'regex:aidotnet_fused_residual_rmsnorm_d64' }
    'residual-layernorm-gelu' { 'regex:aidotnet_fused_residual_bias_layernorm_gelu_d64' }
    'decode' { 'regex:aidotnet_(flash|paged)_decode_d64' }
    'paged-prefill' { 'regex:aidotnet_paged_prefill_d64' }
    'attention-backward' { 'regex:aidotnet_attention_backward_(delta|dq|dkv)_d64' }
    'flash-attention-backward' { 'regex:aidotnet_flash_attention_backward_(dq|dkv)_d64' }
    'qkv-rope-cache' { 'regex:aidotnet_qkv_rope_cache_d64' }
    'fused-linear' { 'regex:aidotnet_fused_linear_gelu_m1' }
    'mixed-linear' { 'regex:aidotnet_fused_linear_gelu_fp16_m1' }
    'mixed-linear-m16' { 'regex:aidotnet_fused_linear_gelu_fp16_m16' }
    'w8a8-linear' { 'regex:aidotnet_fused_linear_gelu_w8a8_m1' }
}
$expectedLaunches = switch ($Target) {
    'attention' { 16 }
    'normalization' { 71 }
    'residual-rmsnorm' { 4 }
    'residual-layernorm-gelu' { 10 }
    'decode' { 2 }
    'paged-prefill' { 1 }
    'attention-backward' { 3 }
    'flash-attention-backward' { 2 }
    'qkv-rope-cache' { 3 }
    'fused-linear' { 10 }
    'mixed-linear' { 10 }
    'mixed-linear-m16' { 10 }
    'w8a8-linear' { 10 }
}
$metricNames = @(
    'smsp__sass_inst_executed_op_local.sum',
    'smsp__sass_inst_executed_op_local_ld.sum',
    'smsp__sass_inst_executed_op_local_st.sum',
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

# One deterministic launch is emitted for every targeted kernel identity;
# normalization includes all 71 logical identities (66 distinct cubins), while
# attention and residual-RMSNorm enumerate their sequence/row, causal, and
# fusion variants. Nsight Compute 2026.2 raw CSV has one wide data row per
# launch. Require every requested column on every expected launch so a partial
# capture cannot be mistaken for complete evidence.
$csvLines = @(Get-Content -LiteralPath $OutputCsv)
$headerIndex = -1
for ($index = 0; $index -lt $csvLines.Count; $index++) {
    if ($csvLines[$index].StartsWith('"ID","Process ID","Process Name"', [StringComparison]::Ordinal)) {
        $headerIndex = $index
        break
    }
}
if ($headerIndex -lt 0) { throw 'Nsight evidence does not contain a raw CSV header.' }
$records = @(($csvLines[$headerIndex..($csvLines.Count - 1)] -join "`n") |
    ConvertFrom-Csv | Where-Object { -not [string]::IsNullOrWhiteSpace($_.'Kernel Name') })
if ($records.Count -ne $expectedLaunches) {
    throw "Nsight evidence is incomplete for '$Target': expected $expectedLaunches launch rows, found $($records.Count)."
}
foreach ($metricName in $metricNames) {
    if ($records[0].PSObject.Properties.Name -notcontains $metricName) {
        throw "Nsight evidence is missing the '$metricName' column."
    }
    $valueCount = @($records | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_.$metricName)
    }).Count
    if ($valueCount -ne $expectedLaunches) {
        throw "Nsight evidence is incomplete for '$metricName': expected $expectedLaunches values, found $valueCount."
    }
}

Write-Host "Nsight evidence verified for $expectedLaunches '$Target' specializations: $OutputCsv"
