param(
    [ValidateSet('attention', 'residual-rmsnorm', 'decode', 'paged-prefill', 'attention-backward', 'flash-attention-backward', 'fused-linear', 'mixed-linear')]
    [string]$Target = 'attention',
    [string]$OutputCsv = 'direct-ptx-ncu.csv',
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
    'fused-linear' { '--direct-ptx-profile-fused-linear' }
    'mixed-linear' { '--direct-ptx-profile-mixed-linear' }
}
$kernel = switch ($Target) {
    'attention' { 'regex:aidotnet_online_attention_128x64' }
    'residual-rmsnorm' { 'regex:aidotnet_fused_residual_rmsnorm_d64' }
    'decode' { 'regex:aidotnet_(flash|paged)_decode_d64' }
    'paged-prefill' { 'regex:aidotnet_paged_prefill_d64' }
    'attention-backward' { 'regex:aidotnet_attention_backward_(delta|dq|dkv)_d64' }
    'flash-attention-backward' { 'regex:aidotnet_flash_attention_backward_(dq|dkv)_d64' }
    'fused-linear' { 'regex:aidotnet_fused_linear_gelu_m1' }
    'mixed-linear' { 'regex:aidotnet_fused_linear_gelu_fp16_m1' }
}
$metrics = @(
    'sass__inst_executed_register_spilling',
    'sass__inst_executed_register_spilling_mem_local',
    'sass__inst_executed_local_loads',
    'sass__inst_executed_local_stores',
    'derived__local_spilling_requests'
) -join ','

& $ncuSource `
    --target-processes all `
    --kernel-name $kernel `
    --metrics $metrics `
    --csv `
    --log-file $OutputCsv `
    dotnet $targetDll $switch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

dotnet $targetDll --direct-ptx-verify-ncu $OutputCsv
exit $LASTEXITCODE
