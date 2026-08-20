using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Foundation test for the FP16-activation-storage build (docs/fp16-activation-storage-design.md,
/// Tensors #555). Validates the primitive the whole feature depends on: a matmul executed under
/// AutocastScope(Float16) — which routes to the FP16-in / FP32-accumulate Tensor-Core GEMM
/// (GemmFp16In32fOut / cublasGemmEx) on a CUDA backend — produces results that match the FP32
/// matmul within FP16 tolerance. If this drifts, FP16 training would silently corrupt.
///
/// Backend-agnostic: on CPU (CI default) AutocastScope is a no-op and this asserts FP32==FP32
/// (still a useful guard that the autocast wrap doesn't alter results); on a CUDA backend it
/// exercises the real FP16 Tensor-Core path.
/// </summary>
public class Fp16MatMulCorrectnessTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        // Keep magnitudes modest so the FP16 dynamic range isn't the variable under test.
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, new[] { rows, cols });
    }

    [Theory]
    [InlineData(32, 48, 64)]
    [InlineData(128, 256, 64)]
    public void TensorMatMul_UnderFp16Autocast_MatchesFp32_WithinTolerance(int m, int k, int n)
    {
        var a = Rand(m, k, 1);
        var b = Rand(k, n, 2);

        // FP32 reference.
        var cFp32 = _engine.TensorMatMul(a, b);

        // FP16 autocast path (no-op on CPU; FP16 Tensor-Core GEMM on CUDA).
        Tensor<float> cFp16;
        using (new AutocastScope(PrecisionMode.Float16))
        {
            cFp16 = _engine.TensorMatMul(a, b);
        }

        Assert.Equal(cFp32.Shape.ToArray(), cFp16.Shape.ToArray());
        var ref32 = cFp32.ToArray();
        var got = cFp16.ToArray();

        // A max-relative metric is not a stable GEMM oracle: values near zero make a tiny,
        // expected FP16 quantization error look arbitrarily large. Use scale-invariant relative
        // L2 plus a K-scaled absolute ceiling, while separately rejecting non-finite output.
        double squaredError = 0;
        double squaredReference = 0;
        double maxAbsolute = 0;
        for (int i = 0; i < ref32.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]) || float.IsInfinity(got[i]),
                $"FP16 matmul produced non-finite at {i}");
            double error = got[i] - ref32[i];
            squaredError += error * error;
            squaredReference += ref32[i] * ref32[i];
            maxAbsolute = Math.Max(maxAbsolute, Math.Abs(error));
        }
        double relativeL2 = Math.Sqrt(squaredError / Math.Max(squaredReference, double.Epsilon));
        double maxAbsoluteTolerance = 3e-3 * Math.Sqrt(k);
        Assert.True(relativeL2 < 2e-3,
            $"FP16 matmul relative L2 error {relativeL2:E4} exceeded 2E-3 (m={m} k={k} n={n})");
        Assert.True(maxAbsolute < maxAbsoluteTolerance,
            $"FP16 matmul max absolute error {maxAbsolute:E4} exceeded {maxAbsoluteTolerance:E4} (m={m} k={k} n={n})");
    }
}
