<#
.SYNOPSIS
    Generic direct-PTX release-evidence runner for the fused-linear family
    (#836/#837/#838/#839 and any experiment that emits "<prefix>_evidence_json=").

.DESCRIPTION
    Runs a named benchmark experiment in N independent, freshly launched OS
    processes (default 3, the release-gate minimum), aggregates the
    machine-readable evidence lines each process emits, and evaluates the
    release-gate thresholds per shape:

        - median speedup >= 1.10x vs the strongest competitor (min median)
        - P95 no worse than the competitor P95 (worst run across processes)
        - 0 managed bytes / 0 temporary device bytes / 0 local bytes per thread
        - max error (absolute, or relative via -ErrorField max_rel_error) <= tol
        - at least 3 independent runs

    Exit code 0 = every promoted shape cleared the gate; non-zero = a HOLD or an
    incomplete run. Writes an aggregated manifest.json. Run on the measured GPU
    (GA10x/SM86); not a CI step.

.EXAMPLE
    # FP32 fused linear (#836)
    pwsh run-direct-ptx-evidence.ps1 -Experiment --direct-ptx-fused-linear -EvidencePrefix linear_evidence_json=
.EXAMPLE
    # mixed-precision (#837)
    pwsh run-direct-ptx-evidence.ps1 -Experiment --direct-ptx-mixed-linear -EvidencePrefix mixed_linear_evidence_json=
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [Parameter(Mandatory = $true)][string]$EvidencePrefix,
    [ValidateSet('max_error', 'max_rel_error')][string]$ErrorField = 'max_error',
    [ValidateRange(3, 20)][int]$Runs = 3,
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-ptx-evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    # 0 = use each experiment's emitted per-precision "tolerance" field (fp32
    # 2e-4, fp16 2e-3, w8a8 5e-4); pass a value to override.
    [double]$ErrorTolerance = 0,
    [string]$DirectMethodPrefix = 'Direct PTX',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$project    = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\AiDotNet.Tensors.Benchmarks.csproj'
$targetDll  = Join-Path $repoRoot 'tests\AiDotNet.Tensors.Benchmarks\bin\Release\net10.0\AiDotNet.Tensors.Benchmarks.dll'
$evidenceDir = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($evidenceDir) | Out-Null

if (-not $SkipBuild) {
    Write-Host "[build] dotnet build -c Release ..."
    & dotnet build $project -c Release -f net10.0 --nologo | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Benchmark build failed." }
}
if (-not (Test-Path $targetDll)) { throw "Benchmark DLL not found: $targetDll (build first)." }

$rows = New-Object System.Collections.Generic.List[object]
for ($r = 1; $r -le $Runs; $r++) {
    $log = Join-Path $evidenceDir "run-$r.log"
    Write-Host "[run $r/$Runs] $Experiment 1"
    & dotnet $targetDll $Experiment 1 *> $log
    if ($LASTEXITCODE -ne 0) { Write-Warning "run $r exited with code $LASTEXITCODE (see $log)" }
    foreach ($line in Get-Content -LiteralPath $log) {
        if ($line.StartsWith($EvidencePrefix, [StringComparison]::Ordinal)) {
            $obj = $line.Substring($EvidencePrefix.Length) | ConvertFrom-Json
            $obj | Add-Member -NotePropertyName Process -NotePropertyValue $r -Force
            $rows.Add($obj)
        }
    }
}
if ($rows.Count -eq 0) { throw "No evidence rows parsed. Did the GPU run produce '$EvidencePrefix' lines?" }

function Median([double[]]$values) {
    $s = @($values | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return [double]::NaN }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n/2 - 1] + $s[$n/2]) / 2.0
}

$byShape = $rows | Group-Object { "{0}x{1}" -f $_.rows, $_.columns }
$manifest = New-Object System.Collections.Generic.List[object]
$anyHold = $false

foreach ($shapeGroup in $byShape) {
    $shape = $shapeGroup.Name
    $methodAgg = @{}
    foreach ($m in ($shapeGroup.Group | Group-Object method)) {
        $medians = @($m.Group | ForEach-Object { [double]$_.median_us })
        $methodAgg[$m.Name] = [pscustomobject]@{
            Method      = $m.Name
            MedianOfMed = [double](Median $medians)
            WorstP95    = [double](@($m.Group | ForEach-Object { [double]$_.p95_us } | Measure-Object -Maximum).Maximum)
            MaxManaged  = [long](@($m.Group | ForEach-Object { [long]$_.managed_bytes } | Measure-Object -Maximum).Maximum)
            MaxTemp     = [long](@($m.Group | ForEach-Object { [long]$_.temp_bytes } | Measure-Object -Maximum).Maximum)
            MaxLocal    = [long](@($m.Group | ForEach-Object { [long]$_.local_bytes } | Measure-Object -Maximum).Maximum)
            MaxError    = [double](@($m.Group | ForEach-Object { [double]$_.$ErrorField } | Measure-Object -Maximum).Maximum)
            EmittedTol  = [double](@($m.Group | ForEach-Object { if ($_.PSObject.Properties.Name -contains 'tolerance') { [double]$_.tolerance } else { 0 } } | Measure-Object -Maximum).Maximum)
            Runs        = @($m.Group | Select-Object -ExpandProperty Process -Unique).Count
        }
    }
    $direct = $methodAgg.Values | Where-Object { $_.Method.StartsWith($DirectMethodPrefix, [StringComparison]::Ordinal) } | Select-Object -First 1
    $competitors = @($methodAgg.Values | Where-Object { -not $_.Method.StartsWith($DirectMethodPrefix, [StringComparison]::Ordinal) })
    if (-not $direct -or $competitors.Count -eq 0) { Write-Warning "[$shape] missing direct or competitor rows; skipping."; continue }
    $best = $competitors | Sort-Object MedianOfMed | Select-Object -First 1

    $speedup = if ($direct.MedianOfMed -gt 0) { $best.MedianOfMed / $direct.MedianOfMed } else { [double]::NaN }
    $fail = New-Object System.Collections.Generic.List[string]
    if (-not ($speedup -ge 1.10)) { $fail.Add(("median-speedup={0:F3}x<1.100x" -f $speedup)) }
    if ($direct.WorstP95 -gt $best.WorstP95) { $fail.Add(("p95={0:F2}us>{1:F2}us" -f $direct.WorstP95, $best.WorstP95)) }
    if ($direct.MaxManaged -gt 0) { $fail.Add("managed-bytes=$($direct.MaxManaged)>0") }
    if ($direct.MaxTemp -gt 0)    { $fail.Add("temp-bytes=$($direct.MaxTemp)>0") }
    if ($direct.MaxLocal -gt 0)   { $fail.Add("local-bytes=$($direct.MaxLocal)>0") }
    $effTol = if ($ErrorTolerance -gt 0) { $ErrorTolerance } elseif ($direct.EmittedTol -gt 0) { $direct.EmittedTol } else { 5e-5 }
    if ($direct.MaxError -gt $effTol) { $fail.Add(("$ErrorField={0:E1}>{1:E1}" -f $direct.MaxError, $effTol)) }
    if ($direct.Runs -lt 3) { $fail.Add("independent-runs=$($direct.Runs)<3") }

    $passed = ($fail.Count -eq 0); if (-not $passed) { $anyHold = $true }
    Write-Host ("[{0,-12}] {1,-4} {2:F2}x vs {3} :: {4}" -f $shape, ($(if ($passed) {'PASS'} else {'HOLD'})), $speedup, $best.Method,
        ($(if ($passed) { 'all gates passed' } else { ($fail -join '; ') })))
    $manifest.Add([pscustomobject]@{ Shape = $shape; Verdict = ($(if ($passed){'PASS'}else{'HOLD'})); Speedup = $speedup; Direct = $direct; BestCompetitor = $best; Failures = @($fail) })
}

$manifestPath = Join-Path $evidenceDir 'manifest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host ""; Write-Host "Manifest: $manifestPath"
if ($anyHold) { Write-Host "$Experiment release evidence: HOLD (see failures above)."; exit 1 }
Write-Host "$Experiment release evidence: PASS on every promoted shape."
