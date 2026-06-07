#!/usr/bin/env bash
# Stage the NVIDIA CUDA redistributable DLLs into runtimes/win-x64/native (from the curated
# cuda-redist folder — they are NOT committed to git, ~900MB) and pack AiDotNet.Native.CUDA +
# the AiDotNet.Tensors.Gpu meta-package. Run from the repo so the relative paths resolve.
#   usage: bash src/AiDotNet.Native.CUDA/stage-and-pack.sh [version] [out-dir]
set -euo pipefail
REDIST="${CUDA_REDIST:-/c/Users/cheat/cuda-redist}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NATIVE_RT="$HERE/runtimes/win-x64/native"
OUT="${2:-/c/Users/cheat/local-nuget}"
NATIVE_VER="${1:-12.5.0}"
DOTNET="${DOTNET:-/c/Users/cheat/.dotnet/dotnet.exe}"

# Files the Native.CUDA package ships (cuBLAS stack + nvRTC compiler + companions).
FILES=(cublas64_12.dll cublasLt64_12.dll cudart64_12.dll \
       nvrtc64_120_0.dll nvrtc64_120_0.alt.dll nvrtc-builtins64_129.dll)

mkdir -p "$NATIVE_RT"
for f in "${FILES[@]}"; do
  if [ ! -f "$REDIST/$f" ]; then echo "MISSING in $REDIST: $f" >&2; exit 1; fi
  cp -f "$REDIST/$f" "$NATIVE_RT/$f"
done
echo "staged ${#FILES[@]} DLLs into $NATIVE_RT"

"$DOTNET" pack "$HERE/AiDotNet.Native.CUDA.csproj" -c Release -p:Version="$NATIVE_VER" -o "$OUT"
echo "packed AiDotNet.Native.CUDA $NATIVE_VER -> $OUT"
