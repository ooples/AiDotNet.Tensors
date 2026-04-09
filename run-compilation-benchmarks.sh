#!/bin/bash
# Run compilation-specific benchmarks (A/B comparisons)
# These tests are Skip'd in normal CI — this script enables and runs them.
#
# Usage:
#   ./run-compilation-benchmarks.sh              # Run all compilation benchmarks
#   ./run-compilation-benchmarks.sh --vs-pytorch # Full BDN vs PyTorch suite (~40min)
#   ./run-compilation-benchmarks.sh --vs-autograd # Autograd comparison benchmarks
#
# Output: results printed to stdout. Pipe to file for baseline:
#   ./run-compilation-benchmarks.sh > benchmark_results_$(date +%Y%m%d).txt

set -e

PROJ="tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj"
BENCH_PROJ="tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj"
TFM="net10.0"

echo "============================================"
echo "AiDotNet.Tensors Compilation Benchmark Suite"
echo "Date: $(date)"
echo "============================================"
echo ""

if [ "$1" == "--vs-pytorch" ]; then
    echo "Running BDN TensorCodec vs PyTorch benchmarks (~40 min)..."
    dotnet run --project "$BENCH_PROJ" -c Release -- --vs-tensorcodec
    exit 0
fi

if [ "$1" == "--vs-autograd" ]; then
    echo "Running BDN Autograd comparison benchmarks..."
    dotnet run --project "$BENCH_PROJ" -c Release -- --vs-autograd
    exit 0
fi

echo "Running compilation A/B benchmarks (Skip removed)..."
echo ""

# Run the baseline benchmark (full A/B matrix)
echo "=== 1. Baseline A/B: Eager vs Compiled vs Phase B ==="
dotnet test "$PROJ" --filter "FullyQualifiedName~TensorCodecBaselineBenchmark" \
    -f "$TFM" -v n -- xunit.methodDisplay=method 2>&1 | grep -E "output|Eager|Compiled|Phase|speedup|PASS|FAIL" || echo "(skipped — remove Skip to enable)"

echo ""
echo "=== 2. Multi-size elementwise/matmul benchmarks ==="
dotnet test "$PROJ" --filter "FullyQualifiedName~TensorCodecMultiSizeBenchmark" \
    -f "$TFM" -v n -- xunit.methodDisplay=method 2>&1 | grep -E "output|Eager|Compiled|speedup|PASS|FAIL" || echo "(skipped — remove Skip to enable)"

echo ""
echo "=== 3. Compilation A/B (per-optimization pass) ==="
dotnet test "$PROJ" --filter "FullyQualifiedName~CompilationABBenchmarks" \
    -f "$TFM" -v n -- xunit.methodDisplay=method 2>&1 | grep -E "output|Eager|Compiled|speedup|PASS|FAIL" || echo "(skipped — remove Skip to enable)"

echo ""
echo "=== 4. Integration tests (correctness verification) ==="
dotnet test "$PROJ" --filter "FullyQualifiedName~CompilationComponentTests" \
    -f "$TFM" -v n 2>&1 | grep -E "Passed|Failed|Total"

echo ""
echo "============================================"
echo "Benchmark suite complete."
echo "For PyTorch comparison: ./run-compilation-benchmarks.sh --vs-pytorch"
echo "============================================"
