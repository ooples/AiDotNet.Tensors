<#
.SYNOPSIS
    Direct-PTX row-sum reduction (#843) release-evidence runner.

.DESCRIPTION
    Runs the reduction experiment in N independent, freshly launched OS
    processes (default 3, the release-gate minimum), aggregates the
    machine-readable "reduction_evidence_json=" lines each process emits, and
    evaluates the release-gate thresholds per shape:

        - median speedup >= 1.10x vs the strongest competitor (min median)
        - P95 no worse than the competitor P95 (worst run across processes)
        - 0 managed bytes / 0 temporary device bytes / 0 local bytes per thread
        - max RELATIVE error <= tolerance (default 5e-5) -- row-sum output is
          O(columns*magnitude), so the gate runs in relative-error mode
        - at least 3 independent runs

    Exit code 0 = every promoted shape cleared the gate. Non-zero = at least one
    shape is on HOLD, or the run was incomplete. Writes an aggregated
    manifest.json. Run on the measured GPU (GA10x/SM86); not a CI step.

.EXAMPLE
    pwsh tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-reduction-evidence.ps1 -Runs 3
#>
[CmdletBinding()]
param(
    [ValidateRange(3, 20)]
    [int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-ptx-reduction-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [double]$RelativeErrorTolerance = 5e-5,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$project    = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\AiDotNet.Tensors.Benchmarks.csproj'
$targetDll  = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\bin\Release\net10.0\AiDotNet.Tensors.Benchmarks.dll'
$evidenceDir = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($evidenceDir) | Out-Null

$flag = '--direct-ptx-reduction'
$evidencePrefix = 'reduction_evidence_json='
$directMethodPrefix = 'Direct PTX'

if (-not $SkipBuild) {
    Write-Host "[build] dotnet build -c Release ..."
    & dotnet build $project -c Release -f net10.0 --nologo | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Benchmark build failed." }
}
if (-not (Test-Path $targetDll)) { throw "Benchmark DLL not found: $targetDll (build first)." }

# --- Run N independent processes -------------------------------------------
$rows = New-Object System.Collections.Generic.List[object]
for ($r = 1; $r -le $Runs; $r++) {
    $log = Join-Path $evidenceDir "run-$r.log"
    Write-Host "[run $r/$Runs] $flag 1"
    & dotnet $targetDll $flag 1 *> $log
    if ($LASTEXITCODE -ne 0) { Write-Warning "run $r exited with code $LASTEXITCODE (see $log)" }
    foreach ($line in Get-Content -LiteralPath $log) {
        if ($line.StartsWith($evidencePrefix, [StringComparison]::Ordinal)) {
            $obj = $line.Substring($evidencePrefix.Length) | ConvertFrom-Json
            $obj | Add-Member -NotePropertyName Process -NotePropertyValue $r -Force
            $rows.Add($obj)
        }
    }
}

if ($rows.Count -eq 0) { throw "No evidence rows parsed. Did the GPU run produce '$evidencePrefix' lines?" }

# --- Aggregate per shape / method ------------------------------------------
function Median([double[]]$values) {
    $s = @($values | Sort-Object)
    $n = $s.Count
    if ($n -eq 0) { return [double]::NaN }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n/2 - 1] + $s[$n/2]) / 2.0
}

$byShape = $rows | Group-Object { "{0}x{1}" -f $_.rows, $_.columns }
$manifest = New-Object System.Collections.Generic.List[object]
$anyHold = $false

foreach ($shapeGroup in $byShape) {
    $shape = $shapeGroup.Name
    $byMethod = $shapeGroup.Group | Group-Object method
    $methodAgg = @{}
    foreach ($m in $byMethod) {
        $medians = @($m.Group | ForEach-Object { [double]$_.median_us })
        $methodAgg[$m.Name] = [pscustomobject]@{
            Method       = $m.Name
            MedianOfMed  = [double](Median $medians)
            WorstP95     = [double](@($m.Group | ForEach-Object { [double]$_.p95_us } | Measure-Object -Maximum).Maximum)
            MaxManaged   = [long](@($m.Group | ForEach-Object { [long]$_.managed_bytes } | Measure-Object -Maximum).Maximum)
            MaxTemp      = [long](@($m.Group | ForEach-Object { [long]$_.temp_bytes } | Measure-Object -Maximum).Maximum)
            MaxLocal     = [long](@($m.Group | ForEach-Object { [long]$_.local_bytes } | Measure-Object -Maximum).Maximum)
            MaxRelError  = [double](@($m.Group | ForEach-Object { [double]$_.max_rel_error } | Measure-Object -Maximum).Maximum)
            Runs         = @($m.Group | Select-Object -ExpandProperty Process -Unique).Count
        }
    }

    $direct = $methodAgg.Values | Where-Object { $_.Method.StartsWith($directMethodPrefix, [StringComparison]::Ordinal) } | Select-Object -First 1
    $competitors = @($methodAgg.Values | Where-Object { -not $_.Method.StartsWith($directMethodPrefix, [StringComparison]::Ordinal) })
    if (-not $direct -or $competitors.Count -eq 0) {
        Write-Warning "[$shape] missing direct or competitor rows; skipping."
        continue
    }
    $best = $competitors | Sort-Object MedianOfMed | Select-Object -First 1

    $speedup = if ($direct.MedianOfMed -gt 0) { $best.MedianOfMed / $direct.MedianOfMed } else { [double]::NaN }
    $fail = New-Object System.Collections.Generic.List[string]
    if (-not ($speedup -ge 1.10)) { $fail.Add(("median-speedup={0:F3}x<1.100x" -f $speedup)) }
    if ($direct.WorstP95 -gt $best.WorstP95) { $fail.Add(("p95={0:F2}us>{1:F2}us" -f $direct.WorstP95, $best.WorstP95)) }
    if ($direct.MaxManaged -gt 0) { $fail.Add("managed-bytes=$($direct.MaxManaged)>0") }
    if ($direct.MaxTemp -gt 0)    { $fail.Add("temp-bytes=$($direct.MaxTemp)>0") }
    if ($direct.MaxLocal -gt 0)   { $fail.Add("local-bytes=$($direct.MaxLocal)>0") }
    if ($direct.MaxRelError -gt $RelativeErrorTolerance) { $fail.Add(("max-rel-error={0:E1}>{1:E1}" -f $direct.MaxRelError, $RelativeErrorTolerance)) }
    if ($direct.Runs -lt 3) { $fail.Add("independent-runs=$($direct.Runs)<3") }

    $passed = ($fail.Count -eq 0)
    if (-not $passed) { $anyHold = $true }
    $verdict = if ($passed) { 'PASS' } else { 'HOLD' }
    Write-Host ("[{0,-9}] {1,-4} {2:F2}x vs {3} :: {4}" -f $shape, $verdict, $speedup, $best.Method,
        ($(if ($passed) { 'all gates passed' } else { ($fail -join '; ') })))

    $manifest.Add([pscustomobject]@{
        Shape = $shape; Verdict = $verdict; Speedup = $speedup
        Direct = $direct; BestCompetitor = $best; Failures = @($fail)
    })
}

$manifestPath = Join-Path $evidenceDir 'manifest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host ""
Write-Host "Manifest: $manifestPath"
if ($anyHold) { Write-Host "Reduction release evidence: HOLD (see failures above)."; exit 1 }
Write-Host "Reduction release evidence: PASS on every promoted shape."
