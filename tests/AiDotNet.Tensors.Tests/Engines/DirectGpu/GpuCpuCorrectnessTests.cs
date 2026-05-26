// Copyright (c) AiDotNet. All rights reserved.
//
// Issue #364 regression + GPU-vs-CPU op-correctness integration suite.
//
// Root cause of #364: the gfx1012 (RDNA1) entry in the autotuned XgemmDirect
// parameter database carried an invalid VWN=4. CLBlast requires VWN to divide
// both WGD/NDIMC and WGD/NDIMB; with WGD=32 and NDIMBD=16 that quotient is 2,
// so VWN=4 made the float4 B-tile load read misaligned/out-of-bounds and the
// direct-GEMM kernel produced wrong results (64x64 ~3x off / NaN / 1e37) for
// every M,N,K that took the direct path. Fixed by VWN 4 -> 2.
//
// These tests pin GPU output to the CPU engine (the trusted managed reference)
// across a thorough shape sweep so the kernel can never silently regress again.
// They run only when a GPU is present; on CPU-only machines each case returns
// early (a no-op pass) so the suite stays portable.

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuCpuCorrectnessTests : IDisposable
{
    private readonly CpuEngine _cpu = new CpuEngine();
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _gpuReady;

    public GpuCpuCorrectnessTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuReady = _gpu.IsGpuAvailable;
        }
        catch
        {
            _gpuReady = false;
        }
    }

    public void Dispose() => _gpu?.Dispose();

    // float32 GEMM at K=512 accumulates ~7e-6 abs error; the #364 bug produced
    // errors of 0.16 / NaN / 1e37. 1e-3 cleanly separates correct from broken.
    private const float Tol = 1e-3f;

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    private static double MaxAbsErr(Tensor<float> a, Tensor<float> b)
    {
        var sa = a.ToArray();
        var sb = b.ToArray();
        Assert.Equal(sa.Length, sb.Length);
        double m = 0;
        for (int i = 0; i < sa.Length; i++)
        {
            float x = sa[i], y = sb[i];
            Assert.False(float.IsNaN(x) || float.IsInfinity(x), $"GPU produced non-finite value {x} at index {i}");
            double e = Math.Abs((double)x - y);
            if (e > m) m = e;
        }
        return m;
    }

    private void AssertGpuMatchesCpu(Tensor<float> gpu, Tensor<float> cpu, string op)
    {
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        double err = MaxAbsErr(gpu, cpu);
        Assert.True(err < Tol, $"{op}: GPU vs CPU max_abs_err {err:E3} exceeded tolerance {Tol:E3}");
    }

    // Shapes span: <128 (the #364 hot zone), tile boundaries around WGD=32,
    // exact/off-by-one multiples, non-square, degenerate 1-dims, and >=128.
    public static IEnumerable<object[]> GemmSizes() => new List<object[]>
    {
        new object[] { 1, 1, 1 },
        new object[] { 2, 2, 2 },
        new object[] { 8, 8, 8 },
        new object[] { 16, 16, 16 },
        new object[] { 17, 17, 17 },
        new object[] { 31, 31, 31 },
        new object[] { 32, 32, 32 },
        new object[] { 33, 33, 33 },
        new object[] { 63, 64, 65 },
        new object[] { 64, 64, 64 },
        new object[] { 65, 65, 65 },
        new object[] { 96, 96, 96 },
        new object[] { 127, 128, 129 },
        new object[] { 128, 128, 128 },
        new object[] { 129, 129, 129 },
        new object[] { 1, 256, 256 },
        new object[] { 256, 1, 256 },
        new object[] { 256, 256, 1 },
        new object[] { 200, 50, 120 },
        new object[] { 256, 256, 256 },
        new object[] { 300, 77, 211 },
        new object[] { 448, 448, 64 },
        new object[] { 512, 512, 512 },
    };

    [Theory]
    [MemberData(nameof(GemmSizes))]
    public void TensorMatMul_Gpu_Matches_Cpu(int m, int n, int k)
    {
        if (!_gpuReady) return;
        var a = Rand(1, m, k);
        var b = Rand(2, k, n);
        var cpu = _cpu.TensorMatMul(a, b);
        var gpu = _gpu.TensorMatMul(a, b);
        AssertGpuMatchesCpu(gpu, cpu, $"TensorMatMul[{m}x{k} * {k}x{n}]");
    }

    [Theory]
    [MemberData(nameof(GemmSizes))]
    public void TensorMatMulTransposed_Gpu_Matches_Cpu(int m, int n, int k)
    {
        if (!_gpuReady) return;
        // C[m,n] = A[m,k] * B[n,k]^T
        var a = Rand(3, m, k);
        var b = Rand(4, n, k);
        var cpu = _cpu.TensorMatMulTransposed(a, b);
        var gpu = _gpu.TensorMatMulTransposed(a, b);
        AssertGpuMatchesCpu(gpu, cpu, $"TensorMatMulTransposed[{m}x{k} * ({n}x{k})^T]");
    }

    public static IEnumerable<object[]> ElementwiseShapes() => new List<object[]>
    {
        new object[] { new[] { 1 } },
        new object[] { new[] { 63 } },
        new object[] { new[] { 64, 64 } },
        new object[] { new[] { 127, 129 } },
        new object[] { new[] { 256, 256 } },
        new object[] { new[] { 8, 16, 32 } },
    };

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorAdd_Gpu_Matches_Cpu(int[] shape)
    {
        if (!_gpuReady) return;
        var a = Rand(5, shape);
        var b = Rand(6, shape);
        AssertGpuMatchesCpu(_gpu.TensorAdd(a, b), _cpu.TensorAdd(a, b), "TensorAdd");
    }

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorSubtract_Gpu_Matches_Cpu(int[] shape)
    {
        if (!_gpuReady) return;
        var a = Rand(7, shape);
        var b = Rand(8, shape);
        AssertGpuMatchesCpu(_gpu.TensorSubtract(a, b), _cpu.TensorSubtract(a, b), "TensorSubtract");
    }

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorMultiply_Gpu_Matches_Cpu(int[] shape)
    {
        if (!_gpuReady) return;
        var a = Rand(9, shape);
        var b = Rand(10, shape);
        AssertGpuMatchesCpu(_gpu.TensorMultiply(a, b), _cpu.TensorMultiply(a, b), "TensorMultiply");
    }

    [Theory]
    [InlineData(64, 64)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    [InlineData(1, 512)]
    public void TensorTranspose_Gpu_Matches_Cpu(int r, int c)
    {
        if (!_gpuReady) return;
        var a = Rand(11, r, c);
        AssertGpuMatchesCpu(_gpu.TensorTranspose(a), _cpu.TensorTranspose(a), $"TensorTranspose[{r}x{c}]");
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    public void Softmax_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!_gpuReady) return;
        var a = Rand(12, rows, cols);
        AssertGpuMatchesCpu(_gpu.Softmax(a, axis: -1), _cpu.Softmax(a, axis: -1), $"Softmax[{rows}x{cols}]");
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    public void ReduceSum_LastAxis_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!_gpuReady) return;
        var a = Rand(13, rows, cols);
        var cpu = _cpu.ReduceSum(a, new[] { 1 }, keepDims: false);
        var gpu = _gpu.ReduceSum(a, new[] { 1 }, keepDims: false);
        AssertGpuMatchesCpu(gpu, cpu, $"ReduceSum[{rows}x{cols} axis=1]");
    }

    // ----- Element-wise UNARY ops (activations etc.) -----
    // Sizes deliberately include non-multiples-of-4 (127, 129, 1023, 1025): the float4-vectorized
    // kernels have scalar "tail" loops that historically skipped the final element at such widths
    // (that was the softmax bug). These exercise every kernel's tail path.
    public static IEnumerable<object[]> UnarySizes() => new List<object[]>
    {
        new object[] { new[] { 1 } },
        new object[] { new[] { 3 } },
        new object[] { new[] { 127 } },
        new object[] { new[] { 128 } },
        new object[] { new[] { 129 } },
        new object[] { new[] { 255 } },
        new object[] { new[] { 1023 } },
        new object[] { new[] { 1025 } },
        new object[] { new[] { 64, 127 } },
        new object[] { new[] { 33, 257 } },
    };

    // Transcendental activations use fast-math approximations on GPU; allow a wider abs tolerance
    // that still catches structural bugs (the softmax tail bug was ~1.8e-2) but tolerates fast-exp noise.
    private const float TolTranscendental = 5e-3f;

    private void AssertUnary(Func<IEngine, Tensor<float>, Tensor<float>> op, int[] shape, string name, float tol, int seed = 20)
    {
        if (!_gpuReady) return;
        var a = Rand(seed, shape);
        var gpu = op(_gpu, a);
        var cpu = op(_cpu, a);
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        double err = MaxAbsErr(gpu, cpu);
        Assert.True(err < tol, $"{name}{string.Join("x", shape)}: GPU vs CPU max_abs_err {err:E3} exceeded tolerance {tol:E3}");
    }

    [Theory] [MemberData(nameof(UnarySizes))] public void ReLU_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.ReLU(a), s, "ReLU", Tol);
    [Theory] [MemberData(nameof(UnarySizes))] public void LeakyReLU_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.LeakyReLU(a, 0.1f), s, "LeakyReLU", Tol);
    [Theory] [MemberData(nameof(UnarySizes))] public void GELU_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.GELU(a), s, "GELU", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void Sigmoid_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.Sigmoid(a), s, "Sigmoid", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void Tanh_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.Tanh(a), s, "Tanh", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void Swish_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.Swish(a), s, "Swish", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void Mish_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.Mish(a), s, "Mish", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void ELU_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.ELU(a, 1.0), s, "ELU", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void Softplus_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.Softplus(a), s, "Softplus", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void TensorExp_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.TensorExp(a), s, "TensorExp", TolTranscendental);
    [Theory] [MemberData(nameof(UnarySizes))] public void TensorDivideScalar_Gpu_Matches_Cpu(int[] s) => AssertUnary((e, a) => e.TensorDivideScalar(a, 3f), s, "TensorDivideScalar", Tol);

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorDivide_Gpu_Matches_Cpu(int[] shape)
    {
        if (!_gpuReady) return;
        var a = Rand(21, shape);
        // Denominator bounded away from zero so the ratio stays well-conditioned.
        var bd = new float[a.ToArray().Length];
        var rng = new Random(22);
        for (int i = 0; i < bd.Length; i++) bd[i] = (float)(rng.NextDouble() + 1.0); // [1,2]
        var b = new Tensor<float>(bd, shape);
        AssertGpuMatchesCpu(_gpu.TensorDivide(a, b), _cpu.TensorDivide(a, b), "TensorDivide");
    }

    // ----- Reductions over the last axis (non-aligned widths) -----
    [Theory]
    [InlineData(64, 127)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    [InlineData(33, 1025)]
    public void ReduceMax_LastAxis_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!_gpuReady) return;
        var a = Rand(23, rows, cols);
        var cpu = _cpu.ReduceMax(a, new[] { 1 }, false, out _);
        var gpu = _gpu.ReduceMax(a, new[] { 1 }, false, out _);
        AssertGpuMatchesCpu(gpu, cpu, $"ReduceMax[{rows}x{cols}]");
    }

    [Theory]
    [InlineData(64, 127)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    [InlineData(33, 1025)]
    public void ReduceMean_LastAxis_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!_gpuReady) return;
        var a = Rand(24, rows, cols);
        AssertGpuMatchesCpu(_gpu.ReduceMean(a, new[] { 1 }, false), _cpu.ReduceMean(a, new[] { 1 }, false), $"ReduceMean[{rows}x{cols}]");
    }

    // ----- Batched GEMM ([B,M,K] x [B,K,N]) incl. indirect-range sizes -----
    [Theory]
    [InlineData(2, 32, 32, 32)]
    [InlineData(4, 64, 64, 64)]
    [InlineData(3, 129, 65, 130)]
    [InlineData(2, 512, 512, 512)]
    public void BatchMatMul_Gpu_Matches_Cpu(int batch, int m, int n, int k)
    {
        if (!_gpuReady) return;
        var a = Rand(25, batch, m, k);
        var b = Rand(26, batch, k, n);
        AssertGpuMatchesCpu(_gpu.BatchMatMul(a, b), _cpu.BatchMatMul(a, b), $"BatchMatMul[{batch}x{m}x{k}*{batch}x{k}x{n}]");
    }
}
#endif
