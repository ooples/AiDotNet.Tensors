# Copyright (c) AiDotNet. All rights reserved.
#
# Issue #327 consumer-side repro driver. Clones HarmonicEngine,
# points its NuGet feed at this repo's locally-built AiDotNet.Tensors
# package, and runs the consumer Transformer Train benchmark.
#
# Usage:
#   pwsh tools/issue327-consumer-repro.ps1 -Run                     # run with defaults
#   pwsh tools/issue327-consumer-repro.ps1 -Steps 5000 -Seeds 8     # full 8-seed
#   AIDOTNET_RUN_CONSUMER_INTEGRATION=1 pwsh tools/issue327-consumer-repro.ps1
#
# -HarmonicEngineRef accepts a branch, tag, or full commit SHA. Pinning
# a SHA is recommended for reproducibility (default branch is volatile).
#
# Prerequisites: dotnet SDK 10.0+, git, network access to github.com/ooples/HarmonicEngine.
# Not wired into CI — the full run takes 3+ hours.

param(
    [switch]$Run,
    [int]$Steps = 100,
    [int]$Seeds = 1,
    # Default tracks the current consumer main branch. Override with a
    # SHA (e.g. -HarmonicEngineRef abc1234567890abc...) for reproducible
    # bisection. Branch / tag / SHA all accepted.
    [string]$HarmonicEngineRef = "main",
    # [System.IO.Path]::GetTempPath() resolves to %TEMP% on Windows and
    # $TMPDIR (with /tmp fallback) on POSIX hosts, so the default works
    # on every supported PowerShell host. $env:TEMP would be null on
    # macOS / Linux and Join-Path would fail before any work happened.
    [string]$WorkDir = (Join-Path ([System.IO.Path]::GetTempPath()) "issue327-consumer-repro")
)

$ErrorActionPreference = "Stop"

# Run gate: require either explicit -Run, env var, or any non-default
# arg to be set (to support `-Steps 5000` shorthand). Helps prevent
# accidental 3-hour runs from a no-arg invocation.
if (-not $Run -and $env:AIDOTNET_RUN_CONSUMER_INTEGRATION -ne "1" `
    -and -not $PSBoundParameters.ContainsKey('Steps') `
    -and -not $PSBoundParameters.ContainsKey('Seeds') `
    -and -not $PSBoundParameters.ContainsKey('HarmonicEngineRef')) {
    Write-Host "No -Run / explicit arg / AIDOTNET_RUN_CONSUMER_INTEGRATION env."
    Write-Host "Usage: pwsh tools/issue327-consumer-repro.ps1 -Run"
    Write-Host "       pwsh tools/issue327-consumer-repro.ps1 -Steps 5000 -Seeds 8"
    exit 0
}

# Repo root: this script lives in tools/, so .. is the repo root.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TensorsCsproj = Join-Path (Join-Path $RepoRoot "src") (Join-Path "AiDotNet.Tensors" "AiDotNet.Tensors.csproj")

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
    # Reset to the ref. The previous code did `git checkout $ref` +
    # `git reset --hard $ref` — that's correct for SHAs and tags
    # (they're immutable), but for BRANCH refs (e.g. `main`) the local
    # branch stays pinned to whatever commit it had before the fetch.
    # `git fetch` only updates `origin/<branch>`, not the local
    # `<branch>`, so the reproducer would silently test an outdated
    # consumer checkout.
    #
    # Probe whether the requested ref is a remote branch. If yes,
    # hard-reset onto `origin/<ref>` so we pick up upstream commits.
    # If no (SHA or tag), fall back to the direct-checkout path —
    # SHAs / tags are immutable so the prior behavior was already
    # correct for them.
    & git show-ref --quiet --verify "refs/remotes/origin/$HarmonicEngineRef"
    if ($LASTEXITCODE -eq 0) {
        # Branch ref — force-create local branch tracking origin/<ref>
        # so subsequent runs converge on the same state regardless of
        # whatever the local branch was at.
        & git checkout --quiet -B $HarmonicEngineRef "origin/$HarmonicEngineRef"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "git checkout -B $HarmonicEngineRef origin/$HarmonicEngineRef failed (exit $LASTEXITCODE)"
            Pop-Location
            exit 1
        }
        & git reset --hard --quiet "origin/$HarmonicEngineRef"
    } else {
        # SHA or tag — immutable, direct checkout converges trivially.
        & git checkout --quiet $HarmonicEngineRef
        if ($LASTEXITCODE -ne 0) {
            Write-Error "git checkout of $HarmonicEngineRef failed (exit $LASTEXITCODE)"
            Pop-Location
            exit 1
        }
        & git reset --hard --quiet $HarmonicEngineRef 2>$null
    }
    Pop-Location
} else {
    Write-Host "[2/4] Cloning HarmonicEngine..."
    # Clone the default branch first (full clone — `git checkout <sha>`
    # below requires the SHA's object to be present, which a shallow
    # clone wouldn't guarantee), then `git checkout` to the requested
    # ref. `git clone --branch` only accepts branches / tags — passing
    # a SHA fails. Two-step clone handles branch, tag, and SHA.
    & git clone --quiet https://github.com/ooples/HarmonicEngine.git $HarmonicEngineDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git clone HarmonicEngine failed (exit $LASTEXITCODE)"
        exit 1
    }
    Push-Location $HarmonicEngineDir
    & git checkout --quiet $HarmonicEngineRef
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git checkout $HarmonicEngineRef failed (exit $LASTEXITCODE)"
        Pop-Location
        exit 1
    }
    Pop-Location
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

& dotnet test (Join-Path "tests" "HarmonicEngine.Tests.csproj") `
    -p:UseLocalAiDotNet=false `
    -p:AiDotNetTensorsVersion=$TensorsNupkgVersion `
    --filter $reproFilter `
    --logger "console;verbosity=normal"

$testExit = $LASTEXITCODE
Pop-Location

Write-Host ""
Write-Host "=== Consumer repro complete ==="
Write-Host "  Logs        : $(Join-Path $HarmonicEngineDir 'logs')"
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
