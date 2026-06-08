<#
.SYNOPSIS
  Packs AiDotNet.Native.CUDA as a set of <250 MB NuGet packages.

.DESCRIPTION
  nuget.org rejects any single package over 250 MB (HTTP 413). The CUDA 12 runtime
  AiDotNet.Native.CUDA ships (cudart / cuBLAS / cuBLASLt / nvRTC, ~770 MB) is far over
  that, and cuBLAS / cuBLASLt EACH exceed 250 MB on their own, so per-file packaging is
  not enough — the payload must be byte-chunked. This is the same multi-part approach
  TorchSharp uses to ship its multi-GB CUDA runtime through nuget.org.

  Layout produced (all under the 250 MB limit, all pushed to nuget.org):
    AiDotNet.Native.CUDA               - meta package: build/.targets + manifest, NO dlls,
                                         depends on every fragment package below.
    AiDotNet.Native.CUDA.Fragments.N   - one ~ChunkMiB byte-chunk of one native DLL each.

  At the CONSUMER'S build, the .targets in the meta package reassembles each DLL from its
  ordered chunks (restored via the fragment dependencies) into the build output, on
  win-x64 only. See src/AiDotNet.Native.CUDA/build/AiDotNet.Native.CUDA.targets.

.PARAMETER Version       Package version (e.g. 0.92.0).
.PARAMETER NativeDir     Folder containing the staged CUDA *.dll (default: the package's runtimes dir).
.PARAMETER OutDir        Where to drop the .nupkg files (default: out).
.PARAMETER ChunkMiB      Max chunk size in MiB (default 200, safely under nuget.org's 250 MB).
#>
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$NativeDir = "$PSScriptRoot/../src/AiDotNet.Native.CUDA/runtimes/win-x64/native",
    [string]$OutDir = "out",
    [int]$ChunkMiB = 200
)

$ErrorActionPreference = "Stop"
$chunkSize = $ChunkMiB * 1MB
$pkgRoot = Join-Path $PSScriptRoot ".." | Resolve-Path
$cudaProjDir = Join-Path $pkgRoot "src/AiDotNet.Native.CUDA"
$buildDir = Join-Path $cudaProjDir "build"
$work = Join-Path $cudaProjDir "obj/cuda-split"
$fragDir = Join-Path $work "fragments"

Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $fragDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$dlls = Get-ChildItem $NativeDir -Filter *.dll -ErrorAction SilentlyContinue | Sort-Object Name
if (-not $dlls -or $dlls.Count -eq 0) {
    throw "No CUDA DLLs found in $NativeDir - run tools/stage-cuda-redist.ps1 first."
}

# ── 1. Split each DLL into <=ChunkMiB chunks (streamed, never loads a whole DLL) ──────
$manifestLines = @()       # "<dll>|<totalBytes>|<chunkCount>"
$chunkNames = @()          # ordered list of all chunk file names
foreach ($dll in $dlls) {
    $total = $dll.Length
    $buffer = New-Object byte[] $chunkSize
    $in = [System.IO.File]::OpenRead($dll.FullName)
    try {
        $i = 0
        while (($read = $in.Read($buffer, 0, $chunkSize)) -gt 0) {
            $chunkName = "{0}.{1:D3}" -f $dll.Name, $i
            $out = [System.IO.File]::Create((Join-Path $fragDir $chunkName))
            try { $out.Write($buffer, 0, $read) } finally { $out.Dispose() }
            $chunkNames += $chunkName
            $i++
        }
    }
    finally { $in.Dispose() }
    $manifestLines += "{0}|{1}|{2}" -f $dll.Name, $total, $i
    Write-Host "  split $($dll.Name) ($([Math]::Round($total/1MB)) MB) -> $i chunk(s)"
}

# Manifest ships inside the meta package (next to the .targets) so the consumer's
# reassembly target knows the chunk order + expected size of every DLL.
$manifestPath = Join-Path $buildDir "cuda-fragments.manifest"
Set-Content -Path $manifestPath -Value $manifestLines -Encoding ASCII
Write-Host "Wrote manifest: $manifestPath ($($manifestLines.Count) DLLs, $($chunkNames.Count) chunks)"

# ── 2. Pack one fragment package per chunk (each well under 250 MB) ───────────────────
$outFull = (Resolve-Path $OutDir).Path
function Invoke-DotnetPack([string]$csproj, [switch]$NeedsFragments) {
    # The meta package depends on the freshly-packed fragment packages, which only exist
    # in $OutDir yet. Use RestoreAdditionalProjectSources to ADD $OutDir to the machine's
    # configured sources (it keeps nuget.org for the netstandard ref) — unlike `--source`,
    # which REPLACES the source list and mangles the nuget.org URL into a relative path.
    $extra = @()
    if ($NeedsFragments) { $extra = @("/p:RestoreAdditionalProjectSources=$outFull") }
    & dotnet pack $csproj -c Release -o $OutDir --nologo -v minimal "/p:PackageVersion=$Version" @extra
    if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed for $csproj" }
}

$fragmentIds = @()
$idx = 0
foreach ($chunkName in $chunkNames) {
    $idx++
    $fragId = "AiDotNet.Native.CUDA.Fragments.$idx"
    $fragmentIds += $fragId
    $fragProjDir = Join-Path $work "frag$idx"
    New-Item -ItemType Directory -Force -Path $fragProjDir | Out-Null
    $chunkAbs = (Join-Path $fragDir $chunkName)
    @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <IncludeBuildOutput>false</IncludeBuildOutput>
    <SuppressDependenciesWhenPacking>true</SuppressDependenciesWhenPacking>
    <NoWarn>NU5128;NU5100</NoWarn>
    <PackageId>$fragId</PackageId>
    <Authors>AiDotNet Contributors</Authors>
    <Description>CUDA runtime fragment $idx of $($chunkNames.Count) for AiDotNet.Native.CUDA. Internal payload chunk; reassembled into the full native DLLs by AiDotNet.Native.CUDA at build time. Do not reference directly.</Description>
    <PackageLicenseExpression>Apache-2.0</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/ooples/AiDotNet.Tensors</PackageProjectUrl>
    <PackageTags>cuda;nvidia;gpu;native;fragment;internal</PackageTags>
  </PropertyGroup>
  <ItemGroup>
    <None Include="$chunkAbs" Pack="true" PackagePath="cuda-fragments/$chunkName" />
  </ItemGroup>
</Project>
"@ | Set-Content -Path (Join-Path $fragProjDir "$fragId.csproj") -Encoding UTF8
    Invoke-DotnetPack (Join-Path $fragProjDir "$fragId.csproj")
}

# ── 3. Pack the meta package: .targets + manifest + dependency on every fragment ──────
$depXml = ($fragmentIds | ForEach-Object {
    "    <PackageReference Include=`"$_`" Version=`"$Version`" />"
}) -join "`n"

$metaProjDir = Join-Path $work "meta"
New-Item -ItemType Directory -Force -Path $metaProjDir | Out-Null
@"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <IncludeBuildOutput>false</IncludeBuildOutput>
    <NoWarn>NU5128;NU5100</NoWarn>
    <PackageId>AiDotNet.Native.CUDA</PackageId>
    <Authors>AiDotNet Contributors</Authors>
    <Company>AiDotNet</Company>
    <Description>Native CUDA 12 runtime (cudart / cuBLAS / cuBLASLt / nvRTC) for NVIDIA GPU acceleration in AiDotNet.Tensors. The ~770 MB payload is delivered as &lt;250 MB fragment packages (nuget.org's per-package limit) and reassembled into the full DLLs at build time, on win-x64.</Description>
    <PackageLicenseExpression>Apache-2.0</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/ooples/AiDotNet.Tensors</PackageProjectUrl>
    <RepositoryUrl>https://github.com/ooples/AiDotNet.Tensors</RepositoryUrl>
    <RepositoryType>git</RepositoryType>
    <PackageTags>cuda;nvidia;gpu;blas;native;cublas</PackageTags>
    <!-- Dependencies on the fragment packages MUST be preserved so a consumer that adds
         AiDotNet.Native.CUDA also restores every chunk. (Do NOT suppress.) -->
  </PropertyGroup>
  <ItemGroup>
    <None Include="$buildDir/AiDotNet.Native.CUDA.targets" Pack="true" PackagePath="build/AiDotNet.Native.CUDA.targets" />
    <None Include="$manifestPath" Pack="true" PackagePath="build/cuda-fragments.manifest" />
  </ItemGroup>
  <ItemGroup>
$depXml
  </ItemGroup>
</Project>
"@ | Set-Content -Path (Join-Path $metaProjDir "AiDotNet.Native.CUDA.csproj") -Encoding UTF8
Invoke-DotnetPack (Join-Path $metaProjDir "AiDotNet.Native.CUDA.csproj") -NeedsFragments

Write-Host "=== CUDA split packaging complete: 1 meta + $($fragmentIds.Count) fragment packages (each < $ChunkMiB MiB) ==="
