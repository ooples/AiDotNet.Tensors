# Copyright (c) AiDotNet. All rights reserved.
#
# Issue #327 consumer-side repro driver. Clones HarmonicEngine at a
# pinned commit, points its NuGet feed at this repo's locally-built
# AiDotNet.Tensors package, and runs the consumer Transformer Train
# benchmark that originally surfaced the 6-10× gap vs PyTorch CPU.
#
# Used to validate that the AiDotNet.Tensors-side fixes in this PR
# (Phase 3 SgemmDirect gate, Phase 4 forward-intermediate pooling, etc.)
# actually transfer through to the consumer Train wall-clock the
# issue tracks.
#
# Usage:
#   pwsh tools/issue327-consumer-repro.ps1            # default: 100 steps
#   pwsh tools/issue327-consumer-repro.ps1 -Steps 5000 -Seeds 8
#   AIDOTNET_RUN_CONSUMER_INTEGRATION=1 pwsh tools/issue327-consumer-repro.ps1
#
# Prerequisites:
#   - dotnet SDK 10.0+ on PATH
#   - git on PATH
#   - HarmonicEngine repo accessible (public on github.com/ooples)
#
# This script is INTENTIONALLY not wired into CI — the consumer
# integration takes 3+ hours per full 8-seed run. Manual pre-PR
# validation only. The xUnit gate at
# tests/AiDotNet.Tensors.Tests/Engines/Issue327TransformerTrainPerfTests.cs
# is the CI tripwire.

param(
    [int]$Steps = 100,
    [int]$Seeds = 1,
    [string]$HarmonicEngineRef = "main",
    [string]$WorkDir = "$env:TEMP\issue327-consumer-repro"
)

$ErrorActionPreference = "Stop"

# Gate on env var unless explicitly invoked with -Force semantics. The
# consumer integration is a manual pre-PR step, not an automatic CI gate.
if ($env:AIDOTNET_RUN_CONSUMER_INTEGRATION -ne "1" -and -not $PSBoundParameters.ContainsKey('Steps')) {
    Write-Host "Skipping consumer integration: AIDOTNET_RUN_CONSUMER_INTEGRATION != 1"
    Write-Host "Pass -Steps <N> explicitly, or set AIDOTNET_RUN_CONSUMER_INTEGRATION=1 to run."
    exit 0
}

# Repo root: this script lives in tools/, so .. is the repo root.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TensorsCsproj = Join-Path $RepoRoot "src\AiDotNet.Tensors\AiDotNet.Tensors.csproj"

if (-not (Test-Path $TensorsCsproj)) {
    Write-Error "AiDotNet.Tensors.csproj not found at $TensorsCsproj"
    exit 1
}

Write-Host "=== Issue #327 consumer repro ==="
Write-Host "  Tensors source : $RepoRoot"
Write-Host "  Work dir       : $WorkDir"
Write-Host "  Steps / seed   : $Steps"
Write-Host "  Seeds          : $Seeds"
Write-Host "  HarmonicEngine : $HarmonicEngineRef"
Write-Host ""

# Step 1: Build AiDotNet.Tensors locally + pack as a NuGet package.
# The consumer integration needs a versioned NuGet to depend on.
$TensorsNupkgVersion = "0.0.0-issue327-local"
$TensorsNupkgDir = Join-Path $WorkDir "nupkg-local"
New-Item -Path $TensorsNupkgDir -ItemType Directory -Force | Out-Null

Write-Host "[1/4] Building AiDotNet.Tensors locally..."
& dotnet pack $TensorsCsproj `
    -c Release `
    -p:Version=$TensorsNupkgVersion `
    -p:PackageVersion=$TensorsNupkgVersion `
    -o $TensorsNupkgDir `
    --nologo

if ($LASTEXITCODE -ne 0) {
    Write-Error "AiDotNet.Tensors pack failed (exit $LASTEXITCODE)"
    exit 1
}

# Step 2: Clone HarmonicEngine at the requested ref.
$HarmonicEngineDir = Join-Path $WorkDir "HarmonicEngine"
if (Test-Path $HarmonicEngineDir) {
    Write-Host "[2/4] Updating HarmonicEngine checkout..."
    Push-Location $HarmonicEngineDir
    & git fetch --quiet origin
    & git checkout --quiet $HarmonicEngineRef
    & git reset --hard --quiet "origin/$HarmonicEngineRef"
    Pop-Location
} else {
    Write-Host "[2/4] Cloning HarmonicEngine..."
    & git clone --quiet --depth 1 --branch $HarmonicEngineRef `
        https://github.com/ooples/HarmonicEngine.git $HarmonicEngineDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git clone HarmonicEngine failed (exit $LASTEXITCODE)"
        exit 1
    }
}

# Step 3: Configure local NuGet feed so HarmonicEngine picks up our build.
$NugetConfig = Join-Path $HarmonicEngineDir "NuGet.config"
Write-Host "[3/4] Pointing NuGet feed at $TensorsNupkgDir..."
@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local-issue327" value="$TensorsNupkgDir" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
  </packageSources>
</configuration>
"@ | Set-Content -Path $NugetConfig -Encoding UTF8

# Step 4: Run the consumer Transformer Train benchmark.
Write-Host "[4/4] Running consumer Transformer Train (steps=$Steps, seeds=$Seeds)..."
Push-Location $HarmonicEngineDir

$reproFilter = "FullyQualifiedName~Phase_PA_WT2_Transformer_d128L4_Batched"
$envOverrides = @{
    "ISSUE327_STEPS" = $Steps.ToString()
    "ISSUE327_SEEDS" = $Seeds.ToString()
}

foreach ($key in $envOverrides.Keys) {
    Set-Item -Path "env:$key" -Value $envOverrides[$key]
}

& dotnet test tests\HarmonicEngine.Tests.csproj `
    -p:UseLocalAiDotNet=false `
    -p:AiDotNetTensorsVersion=$TensorsNupkgVersion `
    --filter $reproFilter `
    --logger "console;verbosity=normal"

$testExit = $LASTEXITCODE
Pop-Location

Write-Host ""
Write-Host "=== Consumer repro complete ==="
Write-Host "  Logs        : $HarmonicEngineDir\logs\"
Write-Host "  Test exit   : $testExit"
Write-Host ""
Write-Host "Issue #327 close criteria:"
Write-Host "  - Per-step wall ≤ 100 ms (issue close)"
Write-Host "  - Per-step wall ≤  50 ms (stretch / PyTorch parity)"
Write-Host "  - Active cores ≥ 20 (issue close); ≥ 28 (stretch)"
Write-Host ""
Write-Host "Compare the measured Phase_PA_WT2_Transformer ms/step against"
Write-Host "those targets in the test stdout. Update PR #331 body with"
Write-Host "the consumer-measured numbers if they differ materially from"
Write-Host "the BDN harness numbers in tests/AiDotNet.Tensors.Benchmarks."

exit $testExit
