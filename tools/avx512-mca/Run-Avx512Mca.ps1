<#
.SYNOPSIS
  Models the AVX-512 direct-conv inner loops on Sapphire Rapids, Ice Lake Server and Zen 4 without AVX-512 hardware
  (#1445).

.DESCRIPTION
  1. Builds Tiles.cs, which holds verbatim copies of DirectConvAvx2's inner loops, before and after #1445. The copies
     drop [MethodImpl(AggressiveOptimization)], because crossgen2 never precompiles such methods.
  2. Compiles them with crossgen2 for x86-64-v4. The framework is put in the version bubble, so Vector512 helpers inline
     exactly as the tier-1 JIT inlines them. Each method's JIT disassembly is captured.
  3. Extracts each hot loop and runs llvm-mca per CPU.

  Caveat: LLVM's Sapphire Rapids and Ice Lake Server models issue 512-bit FMAs on port 0 only. Real server cores also
  issue them on port 5, so read the Intel rows for instruction mix only, and count the port-5 shuffles (the "inserts"
  column) against the FMAs to get the Intel figure. The Zen 4 model is usable as is.

.PARAMETER Crossgen2
  crossgen2.exe from the microsoft.netcore.app.crossgen2.<rid> NuGet package (tools/crossgen2.exe).

.PARAMETER LlvmMca
  llvm-mca.exe from an LLVM release.

.PARAMETER Framework
  A Microsoft.NETCore.App shared-framework directory of the same major version.
#>
param(
    [Parameter(Mandatory)] [string] $Crossgen2,
    [Parameter(Mandatory)] [string] $LlvmMca,
    [Parameter(Mandatory)] [string] $Framework,
    [string[]] $Methods = @('Avx2Three', 'JoinedThree', 'WideThree', 'SextetBlocked', 'SextetInterleaved'),
    [string[]] $Cpus = @('sapphirerapids', 'icelake-server', 'znver4'),
    [int] $Iterations = 200
)
$ErrorActionPreference = 'Stop'
$work = Join-Path ([IO.Path]::GetTempPath()) "avx512-mca-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
New-Item -ItemType Directory -Force $work | Out-Null
dotnet build (Join-Path $PSScriptRoot 'Avx512Mca.Tiles.csproj') -c Release -nologo -clp:ErrorsOnly -o (Join-Path $work 'bin') | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
$dll = Join-Path $work 'bin\Avx512Mca.Tiles.dll'

foreach ($method in $Methods) {
    $asm = Join-Path $work "$method.asm"
    & $Crossgen2 $dll -o (Join-Path $work "r2r_$method.dll") -r "$Framework\*.dll" --inputbubble --inputbubbleref "$Framework\*.dll" `
        --targetos windows --targetarch x64 -O --instruction-set 'x86-64-v4,vectort512' --parallelism 1 `
        --codegenopt "JitDisasm=Tiles:$method" --codegenopt "JitStdOutFile=$asm" | Out-Null
    $lines = Get-Content $asm
    # The hot loop: the block a backward conditional jump targets, up to that jump.
    $jumpIdx = -1; $labelIdx = -1
    for ($i = $lines.Count - 1; $i -ge 0 -and $jumpIdx -lt 0; $i--) {
        if ($lines[$i] -match '^\s+j(l|ne|b|le)\s+(SHORT\s+)?(G_M\d+_IG\d+)') {
            $target = $Matches[3]
            $found = [Array]::FindIndex($lines, [Predicate[string]] { param($x) $x -match "^$($target):" })
            if ($found -ge 0 -and $found -lt $i -and ($i - $found) -gt 10) { $jumpIdx = $i; $labelIdx = $found }
        }
    }
    if ($jumpIdx -lt 0) { throw "no loop found in $method" }
    $body = $lines[($labelIdx + 1)..($jumpIdx - 1)] |
        Where-Object { $_ -match '^\s+[a-z]' -and $_ -notmatch '^\s+(align|nop)\b' } |
        ForEach-Object { ($_ -replace ';.*$', '').Trim() } |
        Where-Object { $_ -ne '' }
    $source = Join-Path $work "$method.s"
    [IO.File]::WriteAllLines($source, @('.intel_syntax noprefix') + $body)
    $fma = @($body | Where-Object { $_ -match '^vfmadd' }).Count
    $shuffles = @($body | Where-Object { $_ -match '^(vinsert|vperm|vshuf)' }).Count
    "{0,-18} {1,3} instructions, {2,2} FMAs, {3} port-5 shuffles" -f $method, $body.Count, $fma, $shuffles
    foreach ($cpu in $Cpus) {
        $out = & $LlvmMca "-mcpu=$cpu" "-iterations=$Iterations" $source 2>&1
        $cycles = [double](($out | Select-String 'Total Cycles:\s+(\d+)').Matches[0].Groups[1].Value)
        "    {0,-15} {1,7:F2} cycles/iteration" -f $cpu, ($cycles / $Iterations)
    }
}
Remove-Item -Recurse -Force $work
