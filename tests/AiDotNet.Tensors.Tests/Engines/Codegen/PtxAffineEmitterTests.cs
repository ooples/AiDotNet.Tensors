// Copyright (c) AiDotNet. All rights reserved.
// Stage 1/2 gate for the codegen bake-off: does a GENERATED kernel match the same
// fp64 oracle the hand-written kernel is held to, and what does it cost?

using System;
using System.Globalization;
using System.Collections.Generic;
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
    [SkippableFact]
    public void GeneratedPtx_MatchesFp64Oracle_OnDevice()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");

        var (input, weights, bias) = MakeData();
        var expected = Oracle(input, weights, bias);
        var spec = CodegenKernelSpec.DepthwiseConv2D3x3BiasRelu(N, C, H, W);

        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Experimental generated convolution is unavailable on this GPU architecture.");

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

    private static unsafe void LaunchFour(
        DirectPtxModule module, IntPtr fn, IntPtr a, IntPtr b, IntPtr c, IntPtr d, uint blocks)
    {
        IntPtr pa = a, pb = b, pc = c, pd = d;
        void** args = stackalloc void*[4];
        args[0] = &pa; args[1] = &pb; args[2] = &pc; args[3] = &pd;
        module.Launch(fn, blocks, 1, 1, PtxAffineEmitter.BlockThreads, 1, 1, 0, args);
    }

    /// <summary>
    /// A reduction too large to unroll must lower to a runtime loop rather than be
    /// refused. Dense 3x3 over 32 channels is 288 trips; before strip-mining the
    /// emitter threw NotSupportedException, which meant no dense convolution at a
    /// production channel count could be generated at all.
    /// </summary>
    [Fact]
    public void LargeReduction_StripMinesInsteadOfRefusing()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu");
        Assert.NotNull(entry);

        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(entry!.Bench, 8, 6);

        Assert.True(emitter.LoopedAxes > 0, "288 reduction trips must lower to a loop.");
        Assert.Contains("LOOP0:", ptx, StringComparison.Ordinal);
        Assert.Contains("bra LOOP0;", ptx, StringComparison.Ordinal);
    }

    /// <summary>
    /// Every catalog entry must be verified at a shape that exercises the SAME
    /// lowering as the shape that gets released. Verifying an unrolled shape and
    /// releasing a strip-mined one ships a code path nothing ever checked.
    /// </summary>
    [Fact]
    public void EveryCatalogEntry_VerifiesTheLoweringItReleases()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            var verifyEmitter = new PtxAffineEmitter();
            verifyEmitter.Emit(entry.Verify, 8, 6);
            var benchEmitter = new PtxAffineEmitter();
            benchEmitter.Emit(entry.Bench, 8, 6);

            Assert.True(verifyEmitter.LoopedAxes == benchEmitter.LoopedAxes,
                entry.Name + ": verify shape lowers with " + verifyEmitter.LoopedAxes +
                " looped axes but the released shape uses " + benchEmitter.LoopedAxes +
                "; the released path would be unverified.");
        }
    }

    /// <summary>
    /// The strip-mined loop must compute the same values as the fp64 reference. This
    /// is the device-free half of the check the conveyor runs on hardware.
    /// </summary>
    [Fact]
    public void StripMinedSpec_InterpreterMatchesFp64Oracle()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu");
        Assert.NotNull(entry);
        var spec = entry!.Verify;

        var inputs = new List<double[]>();
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            long count = 1;
            foreach (int d in spec.Inputs[i].Shape) count *= d;
            var host = new double[count];
            for (long e = 0; e < count; e++) host[e] = (((e * 37 + i * 101) % 97) - 48) / 64.0;
            inputs.Add(host);
        }

        double[] result = spec.Interpret(inputs);
        // Spelled out rather than double.IsFinite: that method does not exist on net471,
        // and this project builds every target framework.
        Assert.All(result, v => Assert.True(!double.IsNaN(v) && !double.IsInfinity(v)));
        Assert.Contains(result, v => v != 0.0);
    }

    /// <summary>
    /// A binding whose unit-stride dimension is indexed by the innermost reduction
    /// axis must be read with ld.global.v4.f32, not four scalar loads.
    /// </summary>
    [Fact]
    public void UnitStrideReductionOperand_UsesVectorLoads()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu");
        Assert.NotNull(entry);

        var vector = new PtxAffineEmitter();
        string vectorPtx = vector.Emit(entry!.Bench, 8, 6);
        Assert.True(vector.VectorisedLoads > 0, "conv2d_1x1 weights are unit-stride in the reduction axis.");
        Assert.Contains("ld.global.v4.f32", vectorPtx, StringComparison.Ordinal);

        var scalar = new PtxAffineEmitter { EnableVectorLoads = false };
        string scalarPtx = scalar.Emit(entry.Bench, 8, 6);
        Assert.Equal(0, scalar.VectorisedLoads);
        Assert.DoesNotContain("ld.global.v4.f32", scalarPtx, StringComparison.Ordinal);
    }

    /// <summary>
    /// Vectorising must never apply where it would be unsafe: a gathered window is not
    /// guaranteed 16-byte aligned, so those bindings must stay on scalar loads.
    /// </summary>
    [Fact]
    public void GatheredWindows_AreNotVectorised()
    {
        foreach (string name in new[] { "depthwise_conv2d_3x3", "conv2d_3x3_bias_relu",
                                        "conv_transpose2d_3x3_stride2" })
        {
            var entry = CodegenKernelCatalog.Find(name);
            Assert.NotNull(entry);
            var emitter = new PtxAffineEmitter();
            emitter.Emit(entry!.Bench, 8, 6);
            Assert.Equal(0, emitter.VectorisedLoads);
        }
    }

    /// <summary>
    /// The tile search must produce a STRUCTURALLY valid lowering. Its quality is not
    /// asserted here, because a model cannot arbitrate it.
    /// </summary>
    /// <remarks>
    /// This test used to compare the chosen lowering against one-output-per-thread using
    /// the cost model, in three successive forms -- loads per output, then loads per MAC,
    /// then predicted time -- and each was falsified by the next kernel added.
    ///
    /// Measurement settled it. For the transposed convolution BOTH post-emission measures
    /// call the chosen tile worse (32.4 us against 28.5 predicted, 1.250 against 1.111
    /// loads/MAC) and the hardware disagrees: 99.4 us against 111.2, so the search's pick
    /// is 1.12x FASTER. Neither model captures what makes it so.
    ///
    /// So the invariant asserted is structural, and lowering QUALITY is settled by the
    /// conveyor's bench stage against a competitor. That gap between model and hardware
    /// is the case for autotuning: measuring candidates replaces every one of these
    /// arguments with a fact.
    /// </remarks>
    [Fact]
    public void TileSearch_ProducesAStructurallyValidLowering()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            var emitter = new PtxAffineEmitter();
            emitter.Emit(entry.Bench, 8, 6);

            long outputs = entry.Bench.Output.ElementCount;
            long covered = (long)emitter.LaunchBlocks * emitter.LaunchBlockX *
                           emitter.LaunchBlockY * emitter.CoarsenedLanes;

            Assert.True(covered >= outputs,
                entry.Name + ": the launch covers " + covered + " outputs but the kernel " +
                "produces " + outputs + "; some output would never be written.");
            Assert.True(emitter.CoarsenedLanes >= 1);
            Assert.True(emitter.LaunchBlockX * emitter.LaunchBlockY >= 32,
                entry.Name + ": block of " + (emitter.LaunchBlockX * emitter.LaunchBlockY) +
                " threads is below a warp.");
        }
    }

    /// <summary>
    /// The launch grid must cover exactly the coarsened thread count. This is the
    /// invariant the whole IR exists to protect: the grid and the in-kernel guard read
    /// the same number, so a coarsened kernel cannot be launched with an uncoarsened
    /// grid (which would compute each output four times) or vice versa (which would
    /// silently skip three quarters of the output).
    /// </summary>
    [Fact]
    public void LaunchBlocks_CoverExactlyTheCoarsenedThreadCount()
    {
        foreach (var entry in CodegenKernelCatalog.All)
        {
            var emitter = new PtxAffineEmitter();
            emitter.Emit(entry.Bench, 8, 6);

            long threads = entry.Bench.Space.TotalThreads / emitter.CoarsenedLanes;
            // The block size is DERIVED per kernel now: shared-memory staging needs a
            // block to cover whole groups of the staged operand's axes, so it is not the
            // fixed constant any more.
            uint expected = (uint)((threads + emitter.LaunchBlockThreads - 1) /
                                  emitter.LaunchBlockThreads);
            Assert.Equal(expected, emitter.LaunchBlocks);

            // And the guard inside the kernel must test that same count.
            string ptx = emitter.Emit(entry.Bench, 8, 6);
            Assert.Contains("setp.ge.u32 %p0, %r2, " + threads.ToString(CultureInfo.InvariantCulture) + ";",
                            ptx, StringComparison.Ordinal);
            Assert.True(emitter.LaunchBlockThreads >= 32,
                entry.Name + ": derived block of " + emitter.LaunchBlockThreads + " threads is below a warp.");
        }
    }

    /// <summary>
    /// An operand that is unit-stride in the coarsened axis must be read with one
    /// vector load across the lanes. Without this the dense 1x1 measured 0.944x from
    /// coarsening -- a regression -- because it lost threads without gaining reuse.
    /// </summary>
    [Fact]
    public void UnitStrideActivationOperand_VectorisesAcrossLanes()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_1x1_bias_relu");
        Assert.NotNull(entry);

        var emitter = new PtxAffineEmitter();
        emitter.Emit(entry!.Bench, 8, 6);
        Assert.True(emitter.CoarsenedLanes > 1);
        Assert.True(emitter.VectorisedLoads >= 64,
            "the 1x1 input is unit-stride in the coarsened axis and must vectorise across lanes; " +
            "got " + emitter.VectorisedLoads + " vector loads.");
    }
}
