// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for issue #251 — the double-precision FusedConv2D
// fast path (Conv + Bias fused in one SIMD pass) and the 1×1 Conv
// fast path in Conv2DWithIm2ColDouble.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness guards for issue #251's two perf paths. Each test
/// compares the optimized path against a reference built from the
/// unfused ops — any numerical divergence beyond floating-point
/// tolerance means the SIMD bias fold or the 1×1 Im2Col bypass has
/// introduced a regression.
/// </summary>
public class FusedConv2DDoublePerfTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ─── FusedConv2D<double> bias fold ────────────────────────────────

    [Fact]
    public void FusedConv2D_Double_None_MatchesConvPlusBroadcastAdd()
    {
        // ResNet stage-2 shape: [1, 64, 56, 56] → 3x3 conv → 64 channels.
        int batch = 1, inC = 64, H = 56, W = 56, outC = 64;
        int kH = 3, kW = 3;

        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 1.0 / (inC * H * W), 0.5);
        var kernel = MakeTensor<double>(new[] { outC, inC, kH, kW }, 1.0 / (outC * inC * kH * kW), 0.125);
        var bias = MakeTensor<double>(new[] { outC }, 0.01, 0.01);

        // Reference: unfused sequence (same shape/activation gate as the fused path).
        var convRef = _engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
        var biasView = _engine.Reshape(bias, new[] { 1, outC, 1, 1 });
        var refOut = _engine.TensorBroadcastAdd(convRef, biasView);

        // Fused path under test.
        var fused = _engine.FusedConv2D(input, kernel, bias,
            strideH: 1, strideW: 1, padH: 1, padW: 1,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.None);

        AssertTensorEqual(refOut, fused, tolerance: 1e-9);
    }

    [Fact]
    public void FusedConv2D_Double_ReLU_MatchesConvPlusBroadcastAddPlusReLU()
    {
        // Mixed-sign bias so the ReLU gate actually fires.
        int batch = 2, inC = 8, H = 14, W = 14, outC = 16;
        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 0.03, -0.4);
        var kernel = MakeTensor<double>(new[] { outC, inC, 1, 1 }, 0.05, -0.2);
        var bias = MakeTensor<double>(new[] { outC }, 0.1, -0.5);

        var convRef = _engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var biasView = _engine.Reshape(bias, new[] { 1, outC, 1, 1 });
        var refAdd = _engine.TensorBroadcastAdd(convRef, biasView);
        var refOut = _engine.ReLU(refAdd);

        var fused = _engine.FusedConv2D(input, kernel, bias,
            strideH: 1, strideW: 1, padH: 0, padW: 0,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.ReLU);

        AssertTensorEqual(refOut, fused, tolerance: 1e-9);
    }

    [Fact]
    public void FusedConv2D_Double_NullBias_StillTakesFastPath()
    {
        // Bias-free conv — fast path must handle null bias (no add pass)
        // rather than falling back to the generic sequence.
        int batch = 1, inC = 4, H = 7, W = 7, outC = 8;
        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 0.1, 0.2);
        var kernel = MakeTensor<double>(new[] { outC, inC, 3, 3 }, 0.05, 0.3);

        var convRef = _engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

        var fused = _engine.FusedConv2D(input, kernel, bias: null,
            strideH: 1, strideW: 1, padH: 1, padW: 1,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.None);

        AssertTensorEqual(convRef, fused, tolerance: 1e-9);
    }

    // ─── 1×1 double fast path in Conv2DWithIm2ColDouble ───────────────

    [Fact]
    public void Conv2D_Double_1x1_Stride1_Pad0_Dilation1_MatchesGenericPath()
    {
        // BottleneckBlock 1×1 projection — stride=1, padding=0, dilation=1.
        int batch = 2, inC = 64, H = 28, W = 28, outC = 256;
        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 0.02, 0.7);
        var kernel = MakeTensor<double>(new[] { outC, inC, 1, 1 }, 0.03, -0.4);

        // Exercise the fast path through the IEngine surface.
        var fast = _engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

        // Build a reference using a 1×2×2 kernel padded to 1×1 semantics:
        // easiest reference is an equivalent Conv2D with a kernel that's
        // the same data but forced through a different code path — call
        // via a slightly different stride/pad combination that doesn't
        // trigger the fast path, then compare on a known identity.
        //
        // Simpler approach: treat 1×1 conv as a matmul and compare
        // element-wise. For batch=2, inC=64, outC=256, this is
        // output[n, k, h, w] = sum_c kernel[k, c, 0, 0] * input[n, c, h, w].
        int innerCount = H * W;
        for (int n = 0; n < batch; n++)
        {
            for (int k = 0; k < outC; k++)
            {
                for (int hw = 0; hw < innerCount; hw++)
                {
                    int h = hw / W, w = hw % W;
                    double expected = 0.0;
                    for (int c = 0; c < inC; c++)
                        expected += kernel[k, c, 0, 0] * input[n, c, h, w];
                    double actual = fast[n, k, h, w];
                    Assert.True(System.Math.Abs(actual - expected) < 1e-7,
                        $"1×1 conv mismatch at [{n},{k},{h},{w}]: expected {expected}, got {actual}.");
                }
            }
        }
    }

    [Fact]
    public void Conv2D_Double_1x1_NonUnitStride_TakesSlowPath()
    {
        // Stride=2 disqualifies the fast path — we should still produce
        // correct output. Regression guard: if the 1×1 gate is too
        // permissive and accepts non-unit stride, this test fails with
        // an offset-indexing mismatch on the output.
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4;
        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 0.1, 0.1);
        var kernel = MakeTensor<double>(new[] { outC, inC, 1, 1 }, 0.2, 0.3);

        var result = _engine.Conv2D(input, kernel, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 });

        int outH = H / 2, outW = W / 2;
        Assert.Equal(new[] { batch, outC, outH, outW }, result._shape);

        for (int n = 0; n < batch; n++)
        for (int k = 0; k < outC; k++)
        for (int h = 0; h < outH; h++)
        for (int w = 0; w < outW; w++)
        {
            double expected = 0.0;
            for (int c = 0; c < inC; c++)
                expected += kernel[k, c, 0, 0] * input[n, c, h * 2, w * 2];
            Assert.True(System.Math.Abs(result[n, k, h, w] - expected) < 1e-9,
                $"Strided 1×1 conv mismatch at [{n},{k},{h},{w}].");
        }
    }

    // ─── NaN-preservation guards ──────────────────────────────────────
    //
    // x86 MAXPS/MAXPD return SRC2 when SRC1 is NaN, so a naive SIMD
    // ReLU built on Avx.Max would silently zero NaN lanes — diverging
    // from the scalar `val < 0 ? 0 : val` form (where NaN < 0 is false
    // and NaN survives). The kernels in CpuFusedOperations use a
    // compare-and-AndNot form that mirrors the scalar semantics; these
    // tests are the regression guard.

    [Fact]
    public void FusedConv2D_Double_ReLU_PreservesNaNFromBias()
    {
        // Bias contains NaN in one channel — after Conv+Bias the entire
        // plane for that channel is NaN, and ReLU must leave it as NaN
        // (not silently turn it into +0.0). Size is large enough to
        // exercise the SIMD bias+ReLU path (HW = 32×32 = 1024 doubles).
        int batch = 1, inC = 4, H = 32, W = 32, outC = 8;
        var input = MakeTensor<double>(new[] { batch, inC, H, W }, 0.05, 0.1);
        var kernel = MakeTensor<double>(new[] { outC, inC, 1, 1 }, 0.1, -0.2);
        var biasData = new double[outC];
        for (int k = 0; k < outC; k++) biasData[k] = 0.1 * (k - outC / 2.0);
        biasData[3] = double.NaN;
        var bias = new Tensor<double>(biasData, new[] { outC });

        var fused = _engine.FusedConv2D(input, kernel, bias,
            strideH: 1, strideW: 1, padH: 0, padW: 0,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.ReLU);

        // Channel 3 must remain entirely NaN; other channels must not
        // contain NaN (sanity guard against accidental propagation).
        for (int n = 0; n < batch; n++)
        for (int k = 0; k < outC; k++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
        {
            double v = fused[n, k, h, w];
            if (k == 3)
            {
                Assert.True(double.IsNaN(v),
                    $"Expected NaN at channel 3 [{n},{k},{h},{w}] but got {v} — SIMD ReLU silently zeroed NaN.");
            }
            else
            {
                Assert.False(double.IsNaN(v),
                    $"Unexpected NaN propagated to channel {k} at [{n},{k},{h},{w}].");
            }
        }
    }

    [Fact]
    public void FusedConv2D_Float_ReLU_PreservesNaNFromBias()
    {
        // Single-precision counterpart — exercises the Avx512F.Max /
        // Avx.Max paths in ApplyNchwPlaneSimd (float).
        int batch = 1, inC = 4, H = 32, W = 32, outC = 8;
        var input = MakeTensor<float>(new[] { batch, inC, H, W }, 0.05, 0.1);
        var kernel = MakeTensor<float>(new[] { outC, inC, 1, 1 }, 0.1, -0.2);
        var biasData = new float[outC];
        for (int k = 0; k < outC; k++) biasData[k] = 0.1f * (k - outC / 2.0f);
        biasData[5] = float.NaN;
        var bias = new Tensor<float>(biasData, new[] { outC });

        var fused = _engine.FusedConv2D(input, kernel, bias,
            strideH: 1, strideW: 1, padH: 0, padW: 0,
            dilationH: 1, dilationW: 1,
            activation: FusedActivationType.ReLU);

        for (int n = 0; n < batch; n++)
        for (int k = 0; k < outC; k++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
        {
            float v = fused[n, k, h, w];
            if (k == 5)
                Assert.True(float.IsNaN(v),
                    $"Expected NaN at channel 5 [{n},{k},{h},{w}] but got {v}.");
            else
                Assert.False(float.IsNaN(v),
                    $"Unexpected NaN at channel {k} [{n},{k},{h},{w}].");
        }
    }

    [Fact]
    public void GetDoubleActivation_ReLU_PreservesNaN()
    {
        // The dispatch-table delegate path (used when the SIMD scalar
        // tail handles a few residual elements) must agree with the
        // scalar fallback inside the SIMD kernels.
        var fn = AiDotNet.Tensors.Helpers.CpuFusedOperations.GetDoubleActivation(FusedActivationType.ReLU);
        Assert.NotNull(fn);
        Assert.True(double.IsNaN(fn!(double.NaN)));
        Assert.Equal(0.0, fn(-1.0));
        Assert.Equal(2.5, fn(2.5));
    }

    [Fact]
    public void GetFloatActivation_ReLU_PreservesNaN()
    {
        var fn = AiDotNet.Tensors.Helpers.CpuFusedOperations.GetFloatActivation(FusedActivationType.ReLU);
        Assert.NotNull(fn);
        Assert.True(float.IsNaN(fn!(float.NaN)));
        Assert.Equal(0f, fn(-1f));
        Assert.Equal(2.5f, fn(2.5f));
    }

    // ─── helpers ──────────────────────────────────────────────────────

    private static Tensor<T> MakeTensor<T>(int[] shape, double scale, double offset)
        where T : struct
    {
        int len = 1;
        for (int i = 0; i < shape.Length; i++) len *= shape[i];
        var data = new T[len];
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < len; i++)
        {
            // Deterministic reproducible fill: small sin wave + scaled index
            double v = scale * (i * 0.017 + offset) + 0.3 * System.Math.Sin(i * 0.1);
            data[i] = ops.FromDouble(v);
        }
        return new Tensor<T>(data, shape);
    }

    private static void AssertTensorEqual<T>(Tensor<T> expected, Tensor<T> actual, double tolerance)
        where T : struct
    {
        Assert.Equal(expected._shape, actual._shape);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var expSpan = expected.AsSpan();
        var actSpan = actual.AsSpan();
        for (int i = 0; i < expSpan.Length; i++)
        {
            double ev = ops.ToDouble(expSpan[i]);
            double av = ops.ToDouble(actSpan[i]);
            Assert.True(System.Math.Abs(ev - av) < tolerance,
                $"Element {i}: expected {ev}, actual {av}, diff {System.Math.Abs(ev - av)}.");
        }
    }
}
