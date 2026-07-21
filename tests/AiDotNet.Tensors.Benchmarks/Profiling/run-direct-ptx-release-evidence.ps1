[CmdletBinding()]
param(
    [ValidateRange(3, 20)]
    [int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-direct-ptx-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [switch]$SkipBuild,
    [switch]$SkipExternal,
    [switch]$AllowDirty
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$project = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\AiDotNet.Tensors.Benchmarks.csproj'
$targetDll = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\bin\Release\net10.0\AiDotNet.Tensors.Benchmarks.dll'
$evidenceRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($evidenceRoot) | Out-Null

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
            Set-Content -LiteralPath $buildLog
        & dotnet build $project -c Release -f net10.0 2>&1 |
            Out-File -LiteralPath $buildLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Release benchmark build failed with exit code $LASTEXITCODE. See '$buildLog'."
        }
    }
    if (-not (Test-Path -LiteralPath $targetDll -PathType Leaf)) {
        throw "Benchmark target is missing at '$targetDll'."
    }

    $suites = [ordered]@{
        'online-attention' = '--direct-ptx-online-attention'
        'gpu-matrix' = '--direct-ptx-gpu-matrix'
        'residual-rmsnorm' = '--direct-ptx-residual-rmsnorm'
    }
    if (-not $SkipExternal) {
        $suites['external-gpu-baselines'] = '--direct-ptx-external-gpu-baselines'
    }

    for ($run = 1; $run -le $Runs; $run++) {
        foreach ($suite in $suites.GetEnumerator()) {
            $log = Join-Path $evidenceRoot ("run-{0:D2}-{1}.log" -f $run, $suite.Key)
            "# independent process $run/$Runs; suite=$($suite.Key); started_utc=$([DateTime]::UtcNow.ToString('O'))" |
                Set-Content -LiteralPath $log
            "# command=dotnet `"$targetDll`" $($suite.Value)" |
                Add-Content -LiteralPath $log
            # Do not mirror the wide TUI to a GPU-accelerated terminal while a
            # later shape is being measured. The complete output remains in the
            # immutable per-process log and is printed only after capture ends.
            & dotnet $targetDll $suite.Value 2>&1 |
                Out-File -LiteralPath $log -Append
            $exitCode = $LASTEXITCODE
            "# completed_utc=$([DateTime]::UtcNow.ToString('O')); exit_code=$exitCode" |
                Add-Content -LiteralPath $log
            if ($exitCode -ne 0) {
                throw "Evidence suite '$($suite.Key)' run $run failed with exit code $exitCode. See '$log'."
            }
        }
    }

    $files = Get-ChildItem -LiteralPath $evidenceRoot -Filter '*.log' | Sort-Object Name | ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [ordered]@{ file = $_.Name; sha256 = $hash.Hash.ToLowerInvariant() }
    }
    $manifest = [ordered]@{
        generated_utc = [DateTime]::UtcNow.ToString('O')
        git_commit = $gitCommit
        dirty_worktree = $dirtyLines.Count -ne 0
        requested_independent_runs = $Runs
        external_gpu_baselines_included = -not [bool]$SkipExternal
        commands = [ordered]@{
            build = "dotnet build `"$project`" -c Release -f net10.0"
            suites = @($suites.GetEnumerator() | ForEach-Object {
                "dotnet `"$targetDll`" $($_.Value)"
            })
        }
        files = @($files)
    } | ConvertTo-Json -Depth 4
    $manifestPath = Join-Path $evidenceRoot 'manifest.json'
    [System.IO.File]::WriteAllText($manifestPath, $manifest + [Environment]::NewLine)
    Write-Host "Evidence capture completed: $evidenceRoot"
    Write-Host "Run both Nsight targets separately; performance-counter access is a mandatory release gate."
}
finally {
    Pop-Location
}
