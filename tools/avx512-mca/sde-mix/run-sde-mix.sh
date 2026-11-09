#!/bin/bash
# Counts the instructions DirectConvAvx2's AVX-512 weight gradient ("dw") and strided input gradient ("dx") execute,
# under Intel SDE emulating Sapphire Rapids (#1445). Linux only: on Windows the runtime asks the OS which register
# state it saves, so .NET keeps AVX-512 off under SDE on a non-AVX-512 host; on Linux it reads XCR0, which SDE emulates.
#   run-sde-mix.sh <sde64> <AiDotNet.Tensors.dll built for net10.0> [dw|dx]
set -euo pipefail
SDE=$1; LIB=$2; MODE=${3:-dw}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
dotnet build "$HERE/SdeMixDriver.csproj" -c Release -v q -o "$OUT/bin" -p:TensorsDll="$LIB" >/dev/null
cp "$(dirname "$LIB")"/*.dll "$OUT/bin/"
DOTNET_EnableWriteXorExecute=0 AIDOTNET_CPU_THREADS=1 "$SDE" -spr -mix -iform -omix "$OUT/mix.txt" -- dotnet "$OUT/bin/SdeMixDriver.dll" "$MODE"
awk '/global-dynamic-counts/{p=1} p' "$OUT/mix.txt" | awk '
  $1 ~ /^VFMADD.*ZMM/ {f+=$2} $1 ~ /^VINSERTF(32X8|64X4).*ZMM/ {i+=$2}
  END {printf "zmm FMAs %d, zmm inserts %d, inserts per FMA %.4f\n", f, i, (f>0)?i/f:0}'
echo "mix: $OUT/mix.txt"
