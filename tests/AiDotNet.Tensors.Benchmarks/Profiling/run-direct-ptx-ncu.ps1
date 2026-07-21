param(
    [ValidateSet('attention', 'residual-rmsnorm', 'decode')]
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
}
$kernel = switch ($Target) {
    'attention' { 'regex:aidotnet_online_attention_128x64' }
    'residual-rmsnorm' { 'regex:aidotnet_fused_residual_rmsnorm_d64' }
    'decode' { 'regex:aidotnet_(flash|paged)_decode_d64' }
}
$metrics = @(
    'sass__inst_executed_register_spilling',
    'sass__inst_executed_register_spilling_mem_local',
    'sass__inst_executed_local_loads',
    'sass__inst_executed_local_stores'
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
