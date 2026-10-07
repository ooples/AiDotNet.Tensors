using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// A compiled CPU training plan runs each Conv2D -> TensorChannelBiasAdd [-> ReLU] chain as one forward action
/// (the conv, then one bias+ReLU pass) and one backward action (one pass for the pre-activation and bias gradients,
/// then the conv gradients). The fusion is meant to be bit-identical to the unfused chain, so these compare the
/// kernels against the separate engine ops and the fused plan against the same plan compiled with the fusion off
/// (AIDOTNET_CONV_EPILOGUE_FUSION=0), bit for bit, over several steps.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ConvBiasActivationFusionTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    private static void AssertBitEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}: element {i} expected {expected[i]:R} got {actual[i]:R}");
    }

    [Theory]
    [InlineData(2, 3, 5, 7, true)]     // spatial 35: vector body plus a scalar tail
    [InlineData(3, 4, 4, 4, true)]     // spatial 16: vector only
    [InlineData(2, 2, 1, 3, true)]     // spatial 3: scalar only
    [InlineData(2, 3, 5, 7, false)]
    public void ForwardMatchesSeparateOps(int n, int c, int h, int w, bool relu)
    {
        var engine = new CpuEngine();
        var x = Rnd(new[] { n, c, h, w }, 1);
        var b = Rnd(new[] { c }, 2);
        // Special values: a NaN sum, a -0 sum (input -b), exact zeros.
        x[0] = float.NaN; x[1] = -b[0]; x[2] = 0f;
        var expected = engine.TensorChannelBiasAdd(x, b);
        if (relu) expected = engine.ReLU(expected);
        var actual = new Tensor<float>(x._shape);
        engine.ChannelBiasActivationInto(actual, x, b, relu);
        AssertBitEqual(expected.AsSpan(), actual.AsSpan(), "forward");
    }

    [Theory]
    [InlineData(2, 3, 5, 7, true, false)]
    [InlineData(3, 4, 4, 4, true, true)]
    [InlineData(2, 2, 1, 3, false, false)]
    [InlineData(2, 3, 5, 7, false, true)]
    public void BackwardMatchesSeparateOps(int n, int c, int h, int w, bool relu, bool accumulateBias)
    {
        var engine = new CpuEngine();
        var y = Rnd(new[] { n, c, h, w }, 3);
        var dy = Rnd(new[] { n, c, h, w }, 4);
        y[0] = 0f; y[1] = -0f; y[2] = float.NaN; dy[3] = -0f; dy[4] = -0f; y[4] = 1f;
        if (relu) for (int i = 0; i < y.Length; i++) if (y[i] < 0) y[i] = 0f;   // a ReLU output is never negative

        // Reference: the ReLU backward mask, the identity gradient landing on a zeroed buffer, the bias reduction.
        var masked = new float[dy.Length];
        for (int i = 0; i < masked.Length; i++) masked[i] = relu ? (y[i] > 0 ? dy[i] : 0f) : dy[i];
        var expectedDz = new float[masked.Length];
        for (int i = 0; i < masked.Length; i++) expectedDz[i] = 0f + masked[i];
        var expectedDb = new Tensor<float>(new[] { n, c, h, w }, new Vector<float>(masked)).Sum(new[] { 0, 2, 3 });
        var priorDb = Rnd(new[] { c }, 5);
        var expectedBias = new float[c];
        for (int i = 0; i < c; i++) expectedBias[i] = accumulateBias ? priorDb[i] + expectedDb[i] : expectedDb[i];

        var dz = new Tensor<float>(y._shape);
        for (int i = 0; i < dz.Length; i++) dz[i] = 77f;   // stale contents must be fully replaced
        var db = new Tensor<float>(new[] { c });
        priorDb.AsSpan().CopyTo(db.AsWritableSpan());
        engine.ChannelBiasActivationBackwardInto(dz, db, dy, y, relu, accumulateBias);
        AssertBitEqual(expectedDz, dz.AsSpan(), "pre-activation gradient");
        AssertBitEqual(expectedBias, db.AsSpan(), "bias gradient");
    }

    /// <summary>
    /// The one-pass conv + bias [+ ReLU] (the epilogue applied in the 3x3 kernel's stores) against the convolution
    /// into a separate buffer followed by the separate epilogue, bit for bit: odd output-channel counts (the pair
    /// kernel's discarded partner), widths with a masked column tail, a NaN input (NaN sums become +0 under ReLU),
    /// a kernel the 3x3 route does not take (two-pass fallback inside the entry), and padding 0 and 2.
    /// </summary>
    [Theory]
    [InlineData(3, 1, 16, 28, 28, 3, 1, true)]    // the parity CNN's first conv
    [InlineData(2, 16, 32, 14, 14, 3, 1, true)]   // the second (14 = one full chunk + a 6-wide tail)
    [InlineData(2, 3, 5, 9, 11, 3, 1, true)]      // odd channel count, 11-wide tail
    [InlineData(2, 3, 5, 9, 11, 3, 1, false)]
    [InlineData(1, 2, 4, 7, 7, 3, 0, true)]
    [InlineData(1, 2, 4, 7, 7, 3, 2, true)]
    [InlineData(2, 3, 4, 9, 9, 5, 2, true)]       // 5x5: not the 3x3 route
    public void FusedConvForwardMatchesTwoPass(int n, int c, int o, int h, int w, int k, int pad, bool relu)
    {
        var engine = new CpuEngine();
        var x = Rnd(new[] { n, c, h, w }, 21);
        var kernel = Rnd(new[] { o, c, k, k }, 22, 0.5f);
        var b = Rnd(new[] { o }, 23, 0.5f);
        x[5] = float.NaN;
        var stride = new[] { 1, 1 }; var padding = new[] { pad, pad }; var dilation = new[] { 1, 1 };
        int oh = h + 2 * pad - k + 1, ow = w + 2 * pad - k + 1;
        var conv = new Tensor<float>(new[] { n, o, oh, ow });
        engine.Conv2DInto(conv, x, kernel, stride, padding, dilation);
        var expected = new Tensor<float>(conv._shape);
        engine.ChannelBiasActivationInto(expected, conv, b, relu);

        var actual = new Tensor<float>(conv._shape);
        for (int i = 0; i < actual.Length; i++) actual[i] = 77f;   // stale contents must be fully replaced
        engine.Conv2DBiasActivationInto(actual, x, kernel, b, relu, stride, padding, dilation);
        AssertBitEqual(expected.AsSpan(), actual.AsSpan(), "fused conv forward");
    }

    /// <summary>
    /// The pool + ReLU + bias backward in one pass against the max-pool backward into a separate activation-gradient
    /// buffer followed by the ReLU/bias backward, bit for bit. Covers the vector 2x2 path (16 and 17 windows per row),
    /// odd sizes with uncovered remainder rows/columns, a 3x3 tiling pool (scalar path), all-zero windows (ties: the
    /// first tap wins), and NaN / -inf windows, which have no winner and send their gradient to plane cell 0.
    /// </summary>
    [Theory]
    [InlineData(3, 4, 28, 28, 2, false, false)]
    [InlineData(2, 3, 9, 9, 2, false, true)]
    [InlineData(2, 3, 6, 34, 2, true, false)]
    [InlineData(2, 2, 10, 11, 3, true, true)]
    [InlineData(1, 2, 2, 2, 2, true, false)]
    public void PoolReluBiasBackwardMatchesSeparateOps(int n, int c, int h, int w, int pool, bool specials, bool accumulateBias)
    {
        var engine = new CpuEngine();
        var y = Rnd(new[] { n, c, h, w }, 31);
        for (int i = 0; i < y.Length; i++) if (y[i] < 0) y[i] = 0f;   // a ReLU output
        for (int i = 0; i < Math.Min(y.Length, 2 * w); i++) y[i] = 0f;   // whole windows of zeros (ties)
        if (specials)
        {
            // Not a ReLU output, but the kernel must still match: windows with no winner and a -0 tie.
            int plane = h * w;
            for (int i = 0; i < pool; i++)
                for (int j = 0; j < pool; j++) y[plane + i * w + j] = float.NaN;
            for (int i = 0; i < pool; i++)
                for (int j = pool; j < 2 * pool && j < w; j++) y[plane + i * w + j] = float.NegativeInfinity;
            if (2 * plane < y.Length) y[2 * plane] = -0f;
        }
        int oh = (h - pool) / pool + 1, ow = (w - pool) / pool + 1;
        var gPool = Rnd(new[] { n, c, oh, ow }, 32);
        gPool[1] = -0f;
        var priorDb = Rnd(new[] { c }, 33);

        var gradY = new Tensor<float>(y._shape);
        engine.MaxPool2DBackwardRecomputeInto(gradY, gPool, y, pool, pool, pool, pool, accumulate: false);
        var expectedDz = new Tensor<float>(y._shape);
        var expectedDb = new Tensor<float>(new[] { c });
        priorDb.AsSpan().CopyTo(expectedDb.AsWritableSpan());
        engine.ChannelBiasActivationBackwardInto(expectedDz, expectedDb, gradY, y, true, accumulateBias);

        var dz = new Tensor<float>(y._shape);
        for (int i = 0; i < dz.Length; i++) dz[i] = 77f;   // stale contents must be fully replaced
        var db = new Tensor<float>(new[] { c });
        priorDb.AsSpan().CopyTo(db.AsWritableSpan());
        engine.MaxPoolReluBiasBackwardInto(dz, db, gPool, y, pool, pool, accumulateBias);
        AssertBitEqual(expectedDz.AsSpan(), dz.AsSpan(), "pre-activation gradient");
        AssertBitEqual(expectedDb.AsSpan(), db.AsSpan(), "bias gradient");
    }

    /// <summary>
    /// The adaptive-average-pool + ReLU + bias backward in one pass against the pool backward into a separate buffer
    /// followed by the ReLU/bias backward, bit for bit: overlapping windows (14 -> 4), global pooling (-> 1x1),
    /// upsampling (3 -> 5), zero activations and bias accumulation.
    /// </summary>
    [Theory]
    [InlineData(3, 4, 14, 14, 4, 4, false)]
    [InlineData(2, 3, 7, 9, 1, 1, true)]
    [InlineData(2, 2, 3, 3, 5, 5, false)]
    [InlineData(2, 3, 5, 7, 3, 2, true)]
    public void AdaptiveAvgPoolReluBiasBackwardMatchesSeparateOps(int n, int c, int h, int w, int oh, int ow, bool accumulateBias)
    {
        var engine = new CpuEngine();
        var y = Rnd(new[] { n, c, h, w }, 41);
        for (int i = 0; i < y.Length; i++) if (y[i] < 0) y[i] = 0f;   // a ReLU output, about half zeros
        var gPool = Rnd(new[] { n, c, oh, ow }, 42);
        gPool[0] = -0f;
        var priorDb = Rnd(new[] { c }, 43);

        var gradY = new float[y.Length];
        CpuEngine.AdaptiveAvgPool2DBackwardFloat(gPool.GetDataArray(), 0, gradY, 0, n * c, h, w, oh, ow, accumulate: false);
        var gradYT = new Tensor<float>(y._shape, new Vector<float>(gradY));
        var expectedDz = new Tensor<float>(y._shape);
        var expectedDb = new Tensor<float>(new[] { c });
        priorDb.AsSpan().CopyTo(expectedDb.AsWritableSpan());
        engine.ChannelBiasActivationBackwardInto(expectedDz, expectedDb, gradYT, y, true, accumulateBias);

        var dz = new Tensor<float>(y._shape);
        for (int i = 0; i < dz.Length; i++) dz[i] = 77f;
        var db = new Tensor<float>(new[] { c });
        priorDb.AsSpan().CopyTo(db.AsWritableSpan());
        engine.AdaptiveAvgPoolReluBiasBackwardInto(dz, db, gPool, y, accumulateBias);
        AssertBitEqual(expectedDz.AsSpan(), dz.AsSpan(), "pre-activation gradient");
        AssertBitEqual(expectedDb.AsSpan(), db.AsSpan(), "bias gradient");
    }

    /// <summary>
    /// Conv -> bias -> ReLU -> AdaptiveAvgPool2D -> dense: the plan with the adaptive pool's backward folded into the
    /// conv chain's, against it kept separate (bit for bit, two steps) and against the eager tape.
    /// </summary>
    [Fact]
    public void AdaptiveAvgPoolFusedPlanMatchesUnfusedAndTape()
    {
        var priorEngine = AiDotNetEngine.Current;
        var priorFlag = Environment.GetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION");
        AiDotNetEngine.Current = new CpuEngine();
        var x = Rnd(new[] { 3, 2, 9, 9 }, 51);
        var k = Rnd(new[] { 4, 2, 3, 3 }, 52, 0.5f);
        var b = Rnd(new[] { 4 }, 53, 0.2f);
        var wt = Rnd(new[] { 16, 3 }, 54, 0.5f);
        var coef = Rnd(new[] { 3, 3 }, 55);
        var parameters = new[] { k, b, wt };
        Tensor<float> Forward(IEngine e)
        {
            var h = e.ReLU(e.TensorChannelBiasAdd(e.Conv2D(x, k, 1, 1, 1), b));
            h = e.AdaptiveAvgPool2D(h, 2, 2);
            var logits = e.TensorMatMul(e.Reshape(h, new[] { 3, 16 }), wt);
            return e.ReduceSum(e.TensorMultiply(logits, coef), null);
        }
        float[][] Run(bool fusePool, int steps)
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION", fusePool ? null : "0");
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(new CpuEngine());
                plan = scope.CompileTraining(parameters);
            }
            try
            {
                float[][] grads = Array.Empty<float[]>();
                for (int s = 0; s < steps; s++)
                {
                    plan.Step();
                    grads = Array.ConvertAll(plan.Gradients, g => g.AsSpan().ToArray());
                }
                return grads;
            }
            finally
            {
                plan.Dispose();
            }
        }
        try
        {
            float[][] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(new CpuEngine());
                var g = t.ComputeGradients(loss, parameters);
                tape = Array.ConvertAll(parameters, p => g[p].GetFlattenedData());
            }
            var firstStep = Run(fusePool: true, steps: 1);
            for (int p = 0; p < tape.Length; p++)
                for (int i = 0; i < tape[p].Length; i++)
                    Assert.True(Math.Abs(tape[p][i] - firstStep[p][i]) <= 1e-4f * (1f + Math.Abs(tape[p][i])),
                        $"parameter {p} element {i}: tape {tape[p][i]:R} plan {firstStep[p][i]:R}");
            for (int steps = 1; steps <= 2; steps++)
            {
                var separate = Run(fusePool: false, steps);
                var fused = Run(fusePool: true, steps);
                for (int p = 0; p < separate.Length; p++)
                    AssertBitEqual(separate[p], fused[p], $"after {steps} step(s), parameter {p}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION", priorFlag);
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>The compiled plan with the pool backward folded into the conv chain's, against it kept separate.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PoolFusedPlanMatchesUnfusedPool(bool sharedBias)
    {
        var priorEngine = AiDotNetEngine.Current;
        var priorFlag = Environment.GetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION");
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var net = new Net();
            for (int steps = 1; steps <= 2; steps++)
            {
                Environment.SetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION", "0");
                var separate = RunPlan(net, fusion: true, sharedBias, secondRelu: true, steps);
                Environment.SetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION", null);
                var fused = RunPlan(net, fusion: true, sharedBias, secondRelu: true, steps);
                Assert.Equal(separate.Length, fused.Length);
                for (int p = 0; p < separate.Length; p++)
                    AssertBitEqual(separate[p], fused[p], $"after {steps} step(s), parameter {p}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_POOL_BACKWARD_FUSION", priorFlag);
            AiDotNetEngine.Current = priorEngine;
        }
    }

#if !NET471 // SimdConvHelper is not compiled for net471 (AiDotNet.Tensors.csproj removes it there).
    /// <summary>A non-default 3x3 variant routes the conv off the tiled kernel; the entry must still be exact.</summary>
    [Fact]
    public void FusedConvForwardMatchesTwoPassOnLegacyVariant()
    {
        var prior = SimdConvHelper.ActiveConv3x3Variant;
        try
        {
            SimdConvHelper.ActiveConv3x3Variant = SimdConvHelper.Conv3x3Variant.Block2;
            FusedConvForwardMatchesTwoPass(2, 3, 4, 9, 11, 3, 1, true);
        }
        finally
        {
            SimdConvHelper.ActiveConv3x3Variant = prior;
        }
    }
#endif

    [Fact]
    public void BackwardIsIndependentOfThreadCount()
    {
        var engine = new CpuEngine();
        var y = Rnd(new[] { 16, 8, 14, 14 }, 6);
        var dy = Rnd(new[] { 16, 8, 14, 14 }, 7);
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var dz1 = new Tensor<float>(y._shape); var db1 = new Tensor<float>(new[] { 8 });
            engine.ChannelBiasActivationBackwardInto(dz1, db1, dy, y, true, false);
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var dz2 = new Tensor<float>(y._shape); var db2 = new Tensor<float>(new[] { 8 });
            engine.ChannelBiasActivationBackwardInto(dz2, db2, dy, y, true, false);
            AssertBitEqual(dz1.AsSpan(), dz2.AsSpan(), "dz");
            AssertBitEqual(db1.AsSpan(), db2.AsSpan(), "db");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    private sealed class Net
    {
        public Tensor<float> X = Rnd(new[] { 3, 2, 9, 9 }, 11);
        public Tensor<float> K1 = Rnd(new[] { 4, 2, 3, 3 }, 12, 0.5f);
        public Tensor<float> B1 = Rnd(new[] { 4 }, 13, 0.2f);
        public Tensor<float> K2 = Rnd(new[] { 4, 4, 3, 3 }, 14, 0.5f);
        public Tensor<float> B2 = Rnd(new[] { 4 }, 15, 0.2f);
        public Tensor<float> W = Rnd(new[] { 64, 3 }, 16, 0.5f);
        public Tensor<float> Coef = Rnd(new[] { 3, 3 }, 17);
        public Tensor<float>[] Parameters => new[] { K1, B1, K2, B2, W };

        /// <param name="sharedBias">conv2 reuses conv1's bias, so the bias has two consumers (accumulates).</param>
        /// <param name="secondRelu">conv2's bias add feeds a ReLU (conv+bias+ReLU) or not (conv+bias).</param>
        public void Forward(IEngine e, bool sharedBias, bool secondRelu)
        {
            var h = e.ReLU(e.TensorChannelBiasAdd(e.Conv2D(X, K1, 1, 1, 1), B1));
            h = e.MaxPool2DWithIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
            h = e.TensorChannelBiasAdd(e.Conv2D(h, K2, 1, 1, 1), sharedBias ? B1 : B2);
            if (secondRelu) h = e.ReLU(h);
            else h = e.TensorMultiply(h, h);
            var logits = e.TensorMatMul(e.Reshape(h, new[] { 3, 64 }), W);
            e.ReduceSum(e.TensorMultiply(logits, Coef), null);
        }
    }

    private static float[][] RunPlan(Net net, bool fusion, bool sharedBias, bool secondRelu, int steps)
    {
        var prior = Environment.GetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", fusion ? null : "0");
        ICompiledTrainingPlan<float> plan;
        try
        {
            using var scope = GraphMode.Enable();
            net.Forward(new CpuEngine(), sharedBias, secondRelu);
            plan = scope.CompileTraining(net.Parameters);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_EPILOGUE_FUSION", prior);
        }
        try
        {
            float[][] grads = Array.Empty<float[]>();
            for (int s = 0; s < steps; s++)
            {
                plan.Step();
                grads = Array.ConvertAll(plan.Gradients, g => g.AsSpan().ToArray());
            }
            return grads;
        }
        finally
        {
            plan.Dispose();
        }
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    public void FusedPlanMatchesUnfusedPlan(bool sharedBias, bool secondRelu)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var net = new Net();
            for (int steps = 1; steps <= 2; steps++)
            {
                var unfused = RunPlan(net, fusion: false, sharedBias, secondRelu, steps);
                var fused = RunPlan(net, fusion: true, sharedBias, secondRelu, steps);
                Assert.Equal(unfused.Length, fused.Length);
                for (int p = 0; p < unfused.Length; p++)
                    AssertBitEqual(unfused[p], fused[p], $"after {steps} step(s), parameter {p}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>
    /// A conv backward skips the gradient INTO an input no parameter feeds (the network input). When the caller
    /// asks for that input's gradient by listing it as a parameter, it must still be computed: through the fused
    /// conv/bias/ReLU chain and through a plain Conv2D (no bias add), each against the eager tape.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InputGradientIsComputedWhenRequested(bool withBias)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var net = new Net();
            var engine = new CpuEngine();
            var parameters = new[] { net.X, net.K1, net.B1 };
            Tensor<float> Forward(IEngine e)
            {
                var z = e.Conv2D(net.X, net.K1, 1, 1, 1);
                if (withBias) z = e.TensorChannelBiasAdd(z, net.B1);
                else z = e.TensorMultiply(z, z);
                var h = e.ReLU(z);
                return e.ReduceSum(e.TensorMultiply(h, h), null);
            }
            float[] tapeDx;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(engine);
                tapeDx = t.ComputeGradients(loss, new[] { net.X })[net.X].GetFlattenedData();
            }
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(new CpuEngine());
                plan = scope.CompileTraining(withBias ? parameters : new[] { net.X, net.K1 });
            }
            try
            {
                plan.Step();
                var dx = plan.Gradients[0].AsSpan();
                Assert.Equal(tapeDx.Length, dx.Length);
                double maxAbs = 0;
                for (int i = 0; i < dx.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(tapeDx[i]));
                Assert.True(maxAbs > 0, "the reference input gradient is all zero, so this test would prove nothing");
                for (int i = 0; i < dx.Length; i++)
                    Assert.True(Math.Abs(tapeDx[i] - dx[i]) <= 1e-4f * (1f + Math.Abs(tapeDx[i])),
                        $"dX element {i}: tape {tapeDx[i]:R} plan {dx[i]:R}");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    /// <summary>The fused plan also matches the eager tape (so the comparison above is not two wrong plans).</summary>
    [Fact]
    public void FusedPlanMatchesTape()
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var net = new Net();
            var engine = new CpuEngine();
            float[][] tape;
            using (var t = new GradientTape<float>())
            {
                var h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(net.X, net.K1, 1, 1, 1), net.B1));
                h = engine.MaxPool2DWithTensorIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
                h = engine.ReLU(engine.TensorChannelBiasAdd(engine.Conv2D(h, net.K2, 1, 1, 1), net.B2));
                var logits = engine.TensorMatMul(engine.Reshape(h, new[] { 3, 64 }), net.W);
                var loss = engine.ReduceSum(engine.TensorMultiply(logits, net.Coef), null);
                var g = t.ComputeGradients(loss, net.Parameters);
                tape = Array.ConvertAll(net.Parameters, p => g[p].GetFlattenedData());
            }
            var fused = RunPlan(net, fusion: true, sharedBias: false, secondRelu: true, steps: 1);
            for (int p = 0; p < tape.Length; p++)
                for (int i = 0; i < tape[p].Length; i++)
                    Assert.True(Math.Abs(tape[p][i] - fused[p][i]) <= 1e-4f * (1f + Math.Abs(tape[p][i])),
                        $"parameter {p} element {i}: tape {tape[p][i]:R} plan {fused[p][i]:R}");
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}