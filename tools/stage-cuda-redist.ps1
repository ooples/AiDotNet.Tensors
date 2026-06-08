<#
.SYNOPSIS
  Stages the NVIDIA CUDA runtime DLLs that AiDotNet.Native.CUDA packs.

.DESCRIPTION
  Populates src/AiDotNet.Native.CUDA/runtimes/win-x64/native with the CUDA 12
  runtime AiDotNet.Tensors' CUDA backend needs at run time:
    cudart64_12, cublas64_12, cublasLt64_12, nvrtc64_120_0(+.alt), nvrtc-builtins64_*
  These are gitignored (~770MB) and fetched on demand so the release pipeline can
  build the package without committing the binaries (industry-standard, like the
  other AiDotNet.Native.* packages).

  Source: the official NVIDIA CUDA redistributable archives (no toolkit install
  needed), parsed from redistrib_<version>.json. For local/offline use, pass
  -LocalSource <dir> to copy already-downloaded DLLs instead.

.EXAMPLE
  ./tools/stage-cuda-redist.ps1                       # download CUDA 12.6.3 redist
  ./tools/stage-cuda-redist.ps1 -LocalSource C:\cuda  # copy from a local folder
#>
param(
    [string]$CudaVersion = "12.6.3",
    [string]$OutputDir = "$PSScriptRoot/../src/AiDotNet.Native.CUDA/runtimes/win-x64/native",
    [string]$LocalSource = ""
)

$ErrorActionPreference = "Stop"
$wanted = @("cudart64_", "cublas64_", "cublasLt64_", "nvrtc64_", "nvrtc-builtins64_")
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Want([string]$name) { foreach ($w in $wanted) { if ($name -like "$w*") { return $true } } return $false }

if ($LocalSource) {
    Write-Host "Staging CUDA runtime from local source: $LocalSource"
    Get-ChildItem $LocalSource -Filter *.dll | Where-Object { Want $_.Name } | ForEach-Object {
        Copy-Item $_.FullName -Destination $OutputDir -Force
        Write-Host "  + $($_.Name)"
    }
    return
}

$base = "https://developer.download.nvidia.com/compute/cuda/redist"
$manifest = Invoke-RestMethod "$base/redistrib_$CudaVersion.json"
$components = @("cuda_cudart", "libcublas", "cuda_nvrtc")
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "cuda-redist-$CudaVersion"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($component in $components) {
    $rel = $manifest.$component."windows-x86_64".relative_path
    if (-not $rel) { Write-Warning "No windows-x86_64 archive for $component"; continue }

    $url = "$base/$rel"
    $zip = Join-Path $tmp (Split-Path $rel -Leaf)
    $extract = Join-Path $tmp $component

    Write-Host "Downloading $component -> $url"
    Invoke-WebRequest $url -OutFile $zip
    Expand-Archive $zip -DestinationPath $extract -Force

    Get-ChildItem $extract -Recurse -Filter *.dll | Where-Object { Want $_.Name } | ForEach-Object {
        Copy-Item $_.FullName -Destination $OutputDir -Force
        Write-Host "  + $($_.Name)"
    }
}

Write-Host "Staged CUDA runtime into $OutputDir"
