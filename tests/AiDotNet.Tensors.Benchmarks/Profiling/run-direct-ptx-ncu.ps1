param(
    [ValidateSet('attention', 'residual-rmsnorm')]
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
$switch = if ($Target -eq 'attention') {
    '--direct-ptx-profile-attention'
} else {
    '--direct-ptx-profile-residual-rmsnorm'
}
$kernel = if ($Target -eq 'attention') {
    'regex:aidotnet_online_attention_128x64'
} else {
    'regex:aidotnet_fused_residual_rmsnorm_d64'
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
