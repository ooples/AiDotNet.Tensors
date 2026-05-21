using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tier 1.1 correctness: the new Conv2DBackwardInputInto / Conv2DBackwardKernelInto
/// must produce bit-identical results to the allocating Conv2DBackwardInput /
/// Conv2DBackwardKernel they replace in compile-mode. Tests both overwrite
/// (accumulate=false) and accumulate (accumulate=true) paths for float and
/// double, on a typical 4D Conv shape.
/// </summary>
public class Conv2DBackwardIntoTests
{
    [Fact]
    public void Conv2DBackwardInputInto_FLOAT_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<float>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var kernel = new Tensor<float>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() - 0.5);

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<float>(new[] { batch, inC, H, W });
        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-5f,
                $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
    }

    [Fact]
    public void Conv2DBackwardInputInto_DOUBLE_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<double>(new[] { batch, inC, H, W });
        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-12,
                $"[{i}] alloc={allocating[i]:F12} into={dest[i]:F12}");
    }

    [Fact]
    public void Conv2DBackwardInputInto_DOUBLE_Accumulate_AddsToExisting()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

        // Pre-fill dest with deterministic values; accumulate should add allocating into them.
        var dest = new Tensor<double>(new[] { batch, inC, H, W });
        var preExisting = new double[dest.Length];
        var rng2 = new System.Random(7);
        for (int i = 0; i < dest.Length; i++) { preExisting[i] = rng2.NextDouble() - 0.5; dest[i] = preExisting[i]; }

        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: true);

        for (int i = 0; i < dest.Length; i++)
        {
            double expected = preExisting[i] + allocating[i];
            Assert.True(System.Math.Abs(expected - dest[i]) < 1e-12,
                $"[{i}] expected pre+alloc={expected:F12} but dest={dest[i]:F12}");
        }
    }

    [Fact]
    public void Conv2DBackwardKernelInto_FLOAT_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<float>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var input = new Tensor<float>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<float>(new[] { outC, inC, kH, kW });
        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-5f,
                $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
    }

    [Fact]
    public void Conv2DBackwardKernelInto_DOUBLE_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<double>(new[] { outC, inC, kH, kW });
        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-12,
                $"[{i}] alloc={allocating[i]:F12} into={dest[i]:F12}");
    }

    [Fact]
    public void Conv2DBackwardKernelInto_DOUBLE_Accumulate_AddsToExisting()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

        var dest = new Tensor<double>(new[] { outC, inC, kH, kW });
        var preExisting = new double[dest.Length];
        var rng2 = new System.Random(7);
        for (int i = 0; i < dest.Length; i++) { preExisting[i] = rng2.NextDouble() - 0.5; dest[i] = preExisting[i]; }

        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: true);

        for (int i = 0; i < dest.Length; i++)
        {
            double expected = preExisting[i] + allocating[i];
            Assert.True(System.Math.Abs(expected - dest[i]) < 1e-12,
                $"[{i}] expected pre+alloc={expected:F12} but dest={dest[i]:F12}");
        }
    }

    /// <summary>
    /// #415 Phase B: the new 3×3 / stride=1 / padding=1 / FP64 direct backward-input
    /// path (CpuEngine.Conv2DBackwardInput, line ~12527) must produce the same
    /// gradInput as the scalar reference computed straight from the definition.
    /// padding=1 is the exact gate that triggers the new SimdConvHelper.Conv3x3Stride1Double
    /// flipped-kernel path; the existing padding=0 tests above don't cover it.
    /// </summary>
    [Fact]
    public void Conv2DBackwardInput_DOUBLE_3x3_S1_P1_MatchesScalarReference()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 5, H = 12, W = 12, outC = 7, kH = 3, kW = 3;
        int padH = 1, padW = 1;
        int oH = H + 2 * padH - kH + 1; // = H for k=3 p=1
        int oW = W + 2 * padW - kW + 1; // = W for k=3 p=1
        var rng = new System.Random(123);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        var actual = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { padH, padW }, new[] { 1, 1 });

        // Scalar reference: gradInput[b,ic,ih,iw] = sum_{oc,kh,kw} K[oc,ic,kh,kw] * gradOut[b,oc,oh,ow]
        // where oh = ih + padH - kh, ow = iw + padW - kw, both in-bounds.
        var expected = new double[batch * inC * H * W];
        for (int b = 0; b < batch; b++)
            for (int ic = 0; ic < inC; ic++)
                for (int ih = 0; ih < H; ih++)
                    for (int iw = 0; iw < W; iw++)
                    {
                        double sum = 0.0;
                        for (int oc = 0; oc < outC; oc++)
                            for (int kh = 0; kh < kH; kh++)
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int oh = ih + padH - kh;
                                    int ow = iw + padW - kw;
                                    if (oh < 0 || oh >= oH || ow < 0 || ow >= oW) continue;
                                    double k = kernel[oc, ic, kh, kw];
                                    double g = gradOut[b, oc, oh, ow];
                                    sum += k * g;
                                }
                        expected[((b * inC + ic) * H + ih) * W + iw] = sum;
                    }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(System.Math.Abs(actual[i] - expected[i]) < 1e-11,
                $"[{i}] actual={actual[i]:F12} expected={expected[i]:F12} diff={actual[i] - expected[i]:E2}");
    }

    /// <summary>
    /// #415 Phase B-2: direct backward-kernel for 3×3 / stride=1 / padding=1
    /// FP64 must match the scalar reference computed from the
    /// Conv2DBackwardKernel definition.
    /// </summary>
    [Fact]
    public void Conv2DBackwardKernel_DOUBLE_3x3_S1_P1_MatchesScalarReference()
    {
        var engine = new CpuEngine();
        int batch = 2, inC = 5, H = 12, W = 12, outC = 7, kH = 3, kW = 3;
        int padH = 1, padW = 1;
        int oH = H + 2 * padH - kH + 1; // = H for k=3 p=1
        int oW = W + 2 * padW - kW + 1; // = W for k=3 p=1
        var rng = new System.Random(456);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var actual = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { padH, padW }, new[] { 1, 1 });

        // Scalar reference: gradKernel[oc,ic,kh,kw] = sum_{b,oh,ow}
        //   gradOut[b,oc,oh,ow] * input[b,ic,oh+kh-padH,ow+kw-padW]   (if in bounds)
        var expected = new double[outC * inC * kH * kW];
        for (int oc = 0; oc < outC; oc++)
            for (int ic = 0; ic < inC; ic++)
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++)
                    {
                        double sum = 0.0;
                        for (int b = 0; b < batch; b++)
                            for (int oh = 0; oh < oH; oh++)
                                for (int ow = 0; ow < oW; ow++)
                                {
                                    int ih = oh + kh - padH;
                                    int iw = ow + kw - padW;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    sum += gradOut[b, oc, oh, ow] * input[b, ic, ih, iw];
                                }
                        expected[((oc * inC + ic) * kH + kh) * kW + kw] = sum;
                    }

        for (int i = 0; i < actual.Length; i++)
            Assert.True(System.Math.Abs(actual[i] - expected[i]) < 1e-11,
                $"[{i}] actual={actual[i]:F12} expected={expected[i]:F12} diff={actual[i] - expected[i]:E2}");
    }
}
