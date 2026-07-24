// Copyright (c) AiDotNet. All rights reserved.
// Stage 1/2 gate for the codegen bake-off: does a GENERATED kernel match the same
// fp64 oracle the hand-written kernel is held to, and what does it cost?

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class PtxAffineEmitterTests
{
    private const int N = 2, C = 8, H = 8, W = 8;

    private static float In(int i) => (float)(((i * 37 % 97) - 48) / 64.0);
    private static float Wt(int i) => (float)(((i * 53 % 89) - 44) / 128.0);
    private static float Bs(int i) => (float)(((i * 29 % 71) - 35) / 256.0);

    /// <summary>Independent fp64 depthwise-conv reference — the semantic ground truth.</summary>
    private static double[] Oracle(float[] input, float[] weights, float[] bias)
    {
        var outp = new double[N * C * H * W];
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int oh = 0; oh < H; oh++)
                    for (int ow = 0; ow < W; ow++)
                    {
                        double acc = 0;
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                            {
                                int ih = oh + kh - 1, iw = ow + kw - 1;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                acc += (double)input[((n * C + c) * H + ih) * W + iw]
                                     * weights[(c * 3 + kh) * 3 + kw];
                            }
                        acc += bias[c];
                        outp[((n * C + c) * H + oh) * W + ow] = Math.Max(acc, 0.0);
                    }
        return outp;
    }

    private static (float[] input, float[] weights, float[] bias) MakeData()
    {
        var input = new float[N * C * H * W];
        var weights = new float[C * 3 * 3];
        var bias = new float[C];
        for (int i = 0; i < input.Length; i++) input[i] = In(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = Wt(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = Bs(i);
        return (input, weights, bias);
    }

    /// <summary>
    /// The iteration space is the single authority on the launch grid, so the
    /// derived thread count must equal the output element count exactly. This is
    /// the invariant whose hand-written violation silently zeroed half a gradient.
    /// </summary>
    [Fact]
    public void IterationSpace_TotalThreads_MatchesOutputElements()
    {
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);
        Assert.Equal(spec.Output.ElementCount, spec.Space.TotalThreads);
        Assert.Equal(9, spec.Space.ReductionTripCount);
        Assert.Equal((uint)((N * C * H * W + 255) / 256), PtxAffineEmitter.GridBlocks(spec));
    }

    /// <summary>
    /// The spec's own semantics, checked on the CPU against the fp64 oracle.
    /// Runs with no GPU at all — this is what lets the bake-off proceed while the
    /// device is busy.
    /// </summary>
    [Fact]
    public void Interpreter_MatchesFp64Oracle()
    {
        var (input, weights, bias) = MakeData();
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);

        var expected = Oracle(input, weights, bias);
        var actual = spec.Interpret(new[]
        {
            Array.ConvertAll(input, x => (double)x),
            Array.ConvertAll(weights, x => (double)x),
            Array.ConvertAll(bias, x => (double)x)
        });

        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        Assert.True(worst < 1e-12, $"interpreter deviates from oracle by {worst:E3}");
    }

    /// <summary>
    /// The out-of-range zero-padding contract: an index map that leaves the tensor
    /// must be reported out-of-bounds, and the derived predicate must agree with it.
    /// </summary>
    [Fact]
    public void DerivedBounds_RejectOutOfRangeTaps()
    {
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);
        var input = spec.Inputs[0];
        Assert.True(input.NeedsBoundsCheck);          // window map can leave the tensor
        Assert.False(spec.Inputs[1].NeedsBoundsCheck); // bare-axis weights cannot
        Assert.False(spec.Output.NeedsBoundsCheck);

        // axes: n, c, oh, ow, kh, kw — top-left corner with the top-left tap is OOB.
        input.ResolveOffset(new[] { 0, 0, 0, 0, 0, 0 }, out bool ok);
        Assert.False(ok);
        // ...and the centre tap of the same point is in range.
        input.ResolveOffset(new[] { 0, 0, 0, 0, 1, 1 }, out ok);
        Assert.True(ok);
    }

    /// <summary>
    /// The generated PTX, executed on the device, against the same fp64 oracle the
    /// hand-written kernel is tested against.
    /// </summary>
    [Fact]
    public void GeneratedPtx_MatchesFp64Oracle_OnDevice()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        var (input, weights, bias) = MakeData();
        var expected = Oracle(input, weights, bias);
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        string ptx = new PtxAffineEmitter().Emit(spec, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var module = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
            var fn = module.GetFunction(spec.Name, out _);

            using var dIn = runtime.AllocateBytes((nuint)(input.Length * sizeof(float)));
            using var dW = runtime.AllocateBytes((nuint)(weights.Length * sizeof(float)));
            using var dB = runtime.AllocateBytes((nuint)(bias.Length * sizeof(float)));
            using var dOut = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
            dIn.Upload<float>(input); dW.Upload<float>(weights); dB.Upload<float>(bias);

            LaunchFour(module, fn, dIn.Pointer, dW.Pointer, dB.Pointer, dOut.Pointer,
                       PtxAffineEmitter.GridBlocks(spec));
            runtime.Synchronize();

            var actual = new float[expected.Length];
            dOut.Download<float>(actual);

            double worst = 0; int at = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double d = Math.Abs(expected[i] - actual[i]);
                if (d > worst) { worst = d; at = i; }
            }
            if (worst > 2e-3)
                Assert.Fail($"generated kernel deviates by {worst:E3} at index {at} " +
                            $"(expected {expected[at]}, actual {actual[at]})");
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    /// <summary>
    /// Writes the generated PTX so ptxas/nvdisasm can measure it against the
    /// hand-written baseline. Static metrics need no GPU, which is what lets the
    /// bake-off run while the device is busy.
    /// </summary>
    [Fact]
    public void DumpGeneratedPtxForBakeOff()
    {
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);
        string ptx = new PtxAffineEmitter().Emit(spec, 8, 6);
        string dir = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "aidotnet-bakeoff");
        System.IO.Directory.CreateDirectory(dir);
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "generated_dwconv2d3x3.ptx"), ptx);
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "spec.txt"), spec.Describe());
        Assert.Contains(".visible .entry", ptx);
    }

    private static unsafe void LaunchFour(
        DirectPtxModule module, IntPtr fn, IntPtr a, IntPtr b, IntPtr c, IntPtr d, uint blocks)
    {
        IntPtr pa = a, pb = b, pc = c, pd = d;
        void** args = stackalloc void*[4];
        args[0] = &pa; args[1] = &pb; args[2] = &pc; args[3] = &pd;
        module.Launch(fn, blocks, 1, 1, PtxAffineEmitter.BlockThreads, 1, 1, 0, args);
    }
}
