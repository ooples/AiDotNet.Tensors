using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Collection fixture that forces DeterministicModeTests to run sequentially and
/// in isolation from any other test classes that share this collection. The tests
/// mutate a process-wide static flag (AiDotNetEngine.DeterministicMode) which
/// would otherwise race with concurrent matmul/BLAS tests under xUnit's default
/// parallel execution. Putting all BLAS-state-mutating test classes in this
/// collection serializes their execution with respect to each other and guarantees
/// the flag is stable for the duration of each test method.
/// </summary>
[CollectionDefinition("BlasGlobalState", DisableParallelization = true)]
public sealed class BlasGlobalStateCollection { }

/// <summary>
/// Tests for AiDotNetEngine.SetDeterministicMode — verifies the public API surface
/// and that enabling deterministic mode produces bit-exact reproducible results for
/// large matmuls (the hot path that uses MKL.NET in default mode).
///
/// Marked with [Collection("BlasGlobalState")] so these tests run serialized —
/// they toggle a process-wide static flag that would race with concurrent BLAS
/// calls from other parallel test classes.
/// </summary>
[Collection("BlasGlobalState")]
public class DeterministicModeTests
{
    [Fact]
    public void DeterministicMode_TogglesAndReads()
    {
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            Assert.True(AiDotNetEngine.DeterministicMode);

            AiDotNetEngine.SetDeterministicMode(false);
            Assert.False(AiDotNetEngine.DeterministicMode);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    [Fact]
    public void DeterministicMode_MatMul_BitExactAcrossRuns()
    {
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            AiDotNetEngine.SetDeterministicMode(true);

            // Large enough to clear MatrixMultiplyHelper's BLAS work threshold (work = m*k*n,
            // DefaultBlasWorkThreshold = 4096; 128*128*128 = 2,097,152 >> 4096).
            int m = 128, k = 128, n = 128;
            var rng = new Random(42);
            var a = new Tensor<float>([m, k]);
            var b = new Tensor<float>([k, n]);
            for (int i = 0; i < m * k; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < k * n; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var engine = new CpuEngine();
            var r1 = engine.TensorMatMul(a, b);
            var r2 = engine.TensorMatMul(a, b);
            var r3 = engine.TensorMatMul(a, b);

            // Bit-exact equality across three repeated calls
            for (int i = 0; i < m * n; i++)
            {
                Assert.Equal(r1[i], r2[i]);
                Assert.Equal(r1[i], r3[i]);
            }
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }

    [Fact]
    public void DeterministicMode_MatMul_NumericallyCloseToDefault()
    {
        // Sanity: deterministic and default should give the same result to reasonable FP tolerance.
        // This catches regressions where deterministic mode accidentally computes something wrong.
        bool original = AiDotNetEngine.DeterministicMode;
        try
        {
            int m = 64, k = 64, n = 64;
            var rng = new Random(123);
            var a = new Tensor<float>([m, k]);
            var b = new Tensor<float>([k, n]);
            for (int i = 0; i < m * k; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < k * n; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var engine = new CpuEngine();

            AiDotNetEngine.SetDeterministicMode(false);
            var rDefault = engine.TensorMatMul(a, b);

            AiDotNetEngine.SetDeterministicMode(true);
            var rDeterministic = engine.TensorMatMul(a, b);

            for (int i = 0; i < m * n; i++)
            {
                // 1e-4 relative tolerance — allows for accumulation-order differences
                // between MKL and the blocked fallback, but catches any real bugs.
                float expected = rDefault[i];
                float actual = rDeterministic[i];
                float tolerance = 1e-4f * Math.Max(1f, Math.Abs(expected));
                Assert.True(Math.Abs(expected - actual) <= tolerance,
                    $"At index {i}: expected {expected}, got {actual}, diff {Math.Abs(expected - actual)}");
            }
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(original);
        }
    }
}
