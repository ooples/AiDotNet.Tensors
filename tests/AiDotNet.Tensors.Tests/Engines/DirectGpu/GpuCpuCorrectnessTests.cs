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
    private readonly Exception _gpuInitException;

    public GpuCpuCorrectnessTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuReady = _gpu.IsGpuAvailable;
        }
        catch (Exception ex)
        {
            // Capture so an opt-in CI gate (AIDOTNET_REQUIRE_GPU_TESTS=1) can
            // surface the underlying init failure instead of letting the suite
            // pass green as a no-op when a GPU was supposed to be available.
            _gpuInitException = ex;
            _gpuReady = false;
        }
    }

    public void Dispose() => _gpu?.Dispose();

    /// <summary>
    /// Returns whether the GPU is usable for this test. Tests should early-
    /// return when this returns <c>false</c> (CPU-only host — the suite stays
    /// portable). When <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c> is set the call
    /// instead throws, so a CI lane that's *supposed* to have a GPU surfaces
    /// init failures loudly rather than silently passing the suite.
    /// </summary>
    private bool EnsureGpuReady()
    {
        if (_gpuReady) return true;
        if (string.Equals(
                Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"),
                "1",
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but " +
                "DirectGpu initialization failed or no GPU is available.",
                _gpuInitException);
        }
        return false;
    }

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

    // Regression: LayerNorm must normalize over the trailing (gamma-length) dims for EVERY rank,
    // not just rank-2. The GPU path previously took outerSize = shape[0] / normSize = Length/shape[0],
    // which for a transformer's rank-3 [B, S, D] input normalized over S*D and read gamma/beta
    // (length D) out of bounds → garbage (measured CPU-vs-GPU max_abs_err ~70 on [2,8,48]). rank-2
    // happened to work because shape[0] equals the row count. This drove a transformer trained through
    // the AiDotNet facade to chance accuracy on the GPU engine (its [B,S,D] activations were corrupted).
    public static IEnumerable<object[]> LayerNormShapes() => new List<object[]>
    {
        new object[] { new[] { 16, 48 } },       // rank-2 (always worked)
        new object[] { new[] { 1, 8, 48 } },     // rank-3, batch 1 (the Predict shape)
        new object[] { new[] { 2, 8, 48 } },     // rank-3 [B,S,D]
        new object[] { new[] { 4, 3, 32 } },     // rank-3, non-square
        new object[] { new[] { 2, 2, 4, 16 } },  // rank-4
    };

    [Theory]
    [MemberData(nameof(LayerNormShapes))]
    public void LayerNorm_RankGe2_GpuMatchesCpu(int[] shape)
    {
        if (!EnsureGpuReady()) return;
        int d = shape[shape.Length - 1];
        var input = Rand(101, shape);
        var gamma = Rand(102, d);
        var beta = Rand(103, d);

        var cpu = _cpu.TensorLayerNorm(input, gamma, beta, 1e-5);
        var gpu = _gpu.TensorLayerNorm(input, gamma, beta, 1e-5);
        AssertGpuMatchesCpu(gpu, cpu, $"TensorLayerNorm[{string.Join(",", shape)}]");

        // The GPU-resident forward path (LayerNormGpu, used by Predict/TryForwardGpuOptimized)
        // must normalize identically for rank >= 3. It requires a GPU-resident input.
        var gpuInput = input.Gpu();
        var (gpuResident, gpuMean, gpuInvVar) = _gpu.LayerNormGpu(gpuInput, gamma, beta, 1e-5);
        AssertGpuMatchesCpu(gpuResident, cpu, $"LayerNormGpu[{string.Join(",", shape)}]");

        // Close the loop on the full tuple: the saved per-sample mean / inverse-variance (consumed by the
        // backward pass) must be correct too, not just the normalized output. Recompute them the way LayerNorm
        // does over each gamma-spanning sample — population mean and inverse std 1 / sqrt(var + eps).
        int normSize = d;
        int samples = input.Length / normSize;
        var flat = input.ToArray();
        var expMean = new float[samples];
        var expInvVar = new float[samples];
        for (int s = 0; s < samples; s++)
        {
            double sum = 0;
            for (int j = 0; j < normSize; j++) sum += flat[s * normSize + j];
            double mean = sum / normSize;
            double varAcc = 0;
            for (int j = 0; j < normSize; j++) { double diff = flat[s * normSize + j] - mean; varAcc += diff * diff; }
            expMean[s] = (float)mean;
            expInvVar[s] = (float)(1.0 / Math.Sqrt(varAcc / normSize + 1e-5));
        }
        AssertGpuMatchesCpu(gpuMean, new Tensor<float>(expMean, new[] { samples }), $"LayerNormGpu.SaveMean[{string.Join(",", shape)}]");
        AssertGpuMatchesCpu(gpuInvVar, new Tensor<float>(expInvVar, new[] { samples }), $"LayerNormGpu.SaveInvVar[{string.Join(",", shape)}]");
    }

    // Regression: the GPU training-mode Dropout must actually generate an inverted-dropout mask.
    // The CUDA path launched dropout_forward (which only APPLIES a pre-generated mask) without ever
    // filling the mask buffer and with mismatched launch args, so the mask was all-zero and 100% of
    // activations were dropped (output identically 0). That zeroed every dropout layer during training,
    // so any model with dropout > 0 could not learn on the GPU. Assert the mask actually keeps roughly
    // (1 - rate) of the units, applies the 1/keepProb inverted-dropout scale, is finite, and that
    // forward/backward are consistent (backward passes the same units through, at the same scale).
    [Theory]
    [InlineData(0.1)]
    [InlineData(0.3)]
    [InlineData(0.5)]
    public void Dropout_Training_GpuGeneratesMaskLikeCpu(double rate)
    {
        if (!EnsureGpuReady()) return;
        const int n = 8192;
        var ones = new Tensor<float>(Enumerable.Repeat(1.0f, n).ToArray(), new[] { n });

        var gpuOut = _gpu.Dropout(ones, rate, training: true, out var gpuMask);
        var og = gpuOut.ToArray();
        var mg = gpuMask.ToArray();

        int zeros = 0; double keptSum = 0; int kept = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.False(float.IsNaN(og[i]) || float.IsInfinity(og[i]), $"GPU dropout produced non-finite {og[i]}");
            Assert.Equal(og[i] == 0f, mg[i] == 0f); // mask and output agree on which units are dropped
            if (og[i] == 0f) zeros++; else { keptSum += og[i]; kept++; }
        }
        double fracDropped = (double)zeros / n;
        // The previous bug dropped 100%; a correct kernel drops ~rate. Wide band tolerates RNG variance
        // but firmly excludes the all-dropped (1.0) and no-dropped (0.0) failure modes.
        Assert.True(Math.Abs(fracDropped - rate) < 0.06,
            $"GPU dropout fraction {fracDropped:F3} not ~{rate:F3} (100%-drop bug regression?)");
        Assert.True(kept > 0, "GPU dropout dropped everything (mask never generated)");
        double invKeep = 1.0 / (1.0 - rate);
        double meanKept = keptSum / kept;
        Assert.True(Math.Abs(meanKept - invKeep) < 1e-3,
            $"GPU dropout kept-value {meanKept:F4} not the inverted-dropout scale {invKeep:F4}");

        // Backward must pass gradients through the SAME kept units at the SAME scale (gradInput = gradOut * mask).
        var gradOut = Rand(202, n);
        var gradIn = ((AiDotNet.Tensors.Engines.IEngine)_gpu).DropoutBackward(gradOut, gpuMask, rate);
        var gi = gradIn.ToArray(); var go = gradOut.ToArray();
        for (int i = 0; i < n; i++)
        {
            float expected = go[i] * mg[i];
            Assert.True(Math.Abs(gi[i] - expected) < 1e-3f,
                $"GPU dropout backward {gi[i]} != gradOut*mask {expected} at {i}");
        }

        // Inference (training: false) must be identity.
        var infer = _gpu.Dropout(ones, rate, training: false, out _);
        foreach (var v in infer.ToArray())
            Assert.True(Math.Abs(v - 1.0f) < 1e-4f, "GPU dropout in inference mode must be identity");
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
        var a = Rand(5, shape);
        var b = Rand(6, shape);
        AssertGpuMatchesCpu(_gpu.TensorAdd(a, b), _cpu.TensorAdd(a, b), "TensorAdd");
    }

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorSubtract_Gpu_Matches_Cpu(int[] shape)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(7, shape);
        var b = Rand(8, shape);
        AssertGpuMatchesCpu(_gpu.TensorSubtract(a, b), _cpu.TensorSubtract(a, b), "TensorSubtract");
    }

    [Theory]
    [MemberData(nameof(ElementwiseShapes))]
    public void TensorMultiply_Gpu_Matches_Cpu(int[] shape)
    {
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
        var a = Rand(11, r, c);
        AssertGpuMatchesCpu(_gpu.TensorTranspose(a), _cpu.TensorTranspose(a), $"TensorTranspose[{r}x{c}]");
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    public void Softmax_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(12, rows, cols);
        AssertGpuMatchesCpu(_gpu.Softmax(a, axis: -1), _cpu.Softmax(a, axis: -1), $"Softmax[{rows}x{cols}]");
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(127, 129)]
    [InlineData(256, 256)]
    public void ReduceSum_LastAxis_Gpu_Matches_Cpu(int rows, int cols)
    {
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
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
        if (!EnsureGpuReady()) return;
        var a = Rand(25, batch, m, k);
        var b = Rand(26, batch, k, n);
        AssertGpuMatchesCpu(_gpu.BatchMatMul(a, b), _cpu.BatchMatMul(a, b), $"BatchMatMul[{batch}x{m}x{k}*{batch}x{k}x{n}]");
    }

    // ----- Element-wise VECTOR ops (Add / Subtract / Multiply / Divide) -----
    // Regression guard: the deferred-download path released its input buffers
    // (the `using var bufferA/bufferB` in TryRunBinary) before the queued
    // element-wise kernel ever ran, so the kernel read freed/recycled buffers
    // and the result came back all-zeros. The sizes deliberately include values
    // that are NOT multiples of 4 (the float4 kernel's scalar-tail path) and
    // small sizes < one work-group.
    private static Vector<float> RandVec(int seed, int n)
    {
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0) + 1.5f; // keep away from 0 for divide
        return new Vector<float>(data);
    }

    private void AssertVecGpuMatchesCpu(Vector<float> gpu, Vector<float> cpu, string op)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        double m = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            float x = gpu[i], y = cpu[i];
            Assert.False(float.IsNaN(x) || float.IsInfinity(x), $"{op}: GPU produced non-finite value {x} at index {i}");
            double e = Math.Abs((double)x - y);
            if (e > m) m = e;
        }
        Assert.True(m < Tol, $"{op}: GPU vs CPU max_abs_err {m:E3} exceeded tolerance {Tol:E3}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(70)]
    [InlineData(100)]
    [InlineData(256)]
    [InlineData(257)]
    public void ElementwiseVectorOps_Gpu_Matches_Cpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var a = RandVec(101, n);
        var b = RandVec(202, n);

        // CRITICAL: dispatch through the IEngine interface. Add/Subtract/Multiply/Divide are
        // EXPLICIT interface implementations on DirectGpuTensorEngine — calling them on the
        // concrete type resolves to the inherited CpuEngine method and silently bypasses the GPU
        // path entirely. The original element-wise-returns-zeros bug only manifests via IEngine.
        IEngine g = _gpu;
        IEngine c = _cpu;

        AssertVecGpuMatchesCpu((Vector<float>)g.Add(a, b), (Vector<float>)c.Add(a, b), $"Add[{n}]");
        AssertVecGpuMatchesCpu((Vector<float>)g.Subtract(a, b), (Vector<float>)c.Subtract(a, b), $"Subtract[{n}]");
        AssertVecGpuMatchesCpu((Vector<float>)g.Multiply(a, b), (Vector<float>)c.Multiply(a, b), $"Multiply[{n}]");
        AssertVecGpuMatchesCpu((Vector<float>)g.Divide(a, b), (Vector<float>)c.Divide(a, b), $"Divide[{n}]");

        // double path too (the GAMLSS/AiModelBuilder case is double): the GPU converts
        // double->float->double, and the result is wrapped in a Vector<double> that copies.
        var ad = new Vector<double>(System.Linq.Enumerable.Range(0, n).Select(i => (double)a[i]).ToArray());
        var bd = new Vector<double>(System.Linq.Enumerable.Range(0, n).Select(i => (double)b[i]).ToArray());
        var subD = (Vector<double>)g.Subtract(ad, bd);
        var subDc = (Vector<double>)c.Subtract(ad, bd);
        for (int i = 0; i < n; i++)
            Assert.True(System.Math.Abs(subD[i] - subDc[i]) < Tol, $"Subtract<double>[{n}] idx {i}: gpu {subD[i]} vs cpu {subDc[i]}");
    }
}
#endif
