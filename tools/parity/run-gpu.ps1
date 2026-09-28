<#
.SYNOPSIS
  Runs the GPU lane of the PyTorch parity harness (epic #1055) on a local NVIDIA machine.

.DESCRIPTION
  Hosted CI has no GPU, so this lane runs where one exists. It:
    1. refuses to run unless the Python it uses has a CUDA build of torch (a CPU-only wheel would
       silently measure the wrong thing);
    2. builds the test project;
    3. gates on Category=PyTorchParityGpu (the residency and CUDA head-to-head ratchets) plus the
       transfer-count smoke tests, then reports Category=PyTorchParityGpuTarget (zero crossings,
       PyTorch-speed parity) without gating on it;
    4. copies the result artifacts into parity/results/<machine>/ for review and commit.

.PARAMETER Python
  The Python to use. Defaults to $env:PARITY_PYTHON, then 'python'.

.EXAMPLE
  pwsh tools/parity/run-gpu.ps1
#>
[CmdletBinding()]
param([string]$Python = $(if ($env:PARITY_PYTHON) { $env:PARITY_PYTHON } else { 'python' }))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root

$probe = & $Python -c "import torch, sys; print(torch.__version__); sys.exit(0 if torch.cuda.is_available() else 4)" 2>&1
if ($LASTEXITCODE -eq 4) {
    [Console]::Error.WriteLine("torch $probe has no CUDA device. Install a CUDA build (https://pytorch.org/get-started/locally/) before running the GPU lane.")
    exit 4
}
if ($LASTEXITCODE -ne 0) {
    [Console]::Error.WriteLine("Could not import torch with '$Python': $probe")
    exit 3
}
Write-Host "torch $probe with CUDA"
$env:PARITY_PYTHON = $Python

dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -f net10.0 -m:2 -p:UseSharedCompilation=false
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Artifacts older than this run are someone else's evidence; only files written after it are copied.
$runStart = Get-Date

# Gate: the ratchets, plus the transfer-count smoke tests, so "zero crossings" can only mean no transfers,
# never a probe that stopped reporting.
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --no-build -f net10.0 `
    --filter 'Category=PyTorchParityGpu|FullyQualifiedName~GpuResidencyTests.Scope_' --logger 'console;verbosity=detailed'
$testExit = $LASTEXITCODE

# Targets: zero crossings and PyTorch-speed parity. Red until reached; reported, never gating.
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj --no-build -f net10.0 `
    --filter 'Category=PyTorchParityGpuTarget' --logger 'console;verbosity=normal'
Write-Host ("GPU parity targets: " + $(if ($LASTEXITCODE -eq 0) { 'all met' } else { 'not yet met (see above); not gating' }))

# Named by the harness's machine key (os-arch-cpus-cpu model-device-gpu model), never the host name: results are
# meant to be committed, and a host name does not belong in the repository.
$artifacts = @(Get-ChildItem (Join-Path ([IO.Path]::GetTempPath()) 'aidotnet-parity') -Filter '*-cuda-latest.json' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -ge $runStart })
if ($artifacts.Count -eq 0) {
    Write-Host 'No CUDA head-to-head artifact was written by this run; nothing copied.'
    exit $testExit
}
$machine = (Get-Content $artifacts[0].FullName -Raw | ConvertFrom-Json).machineKey
$dest = Join-Path $root "parity/results/$machine"
New-Item -ItemType Directory -Force $dest | Out-Null
$artifacts | Copy-Item -Destination $dest -Force
Write-Host "Artifacts ($($artifacts.Count), this run only): $dest"
exit $testExit