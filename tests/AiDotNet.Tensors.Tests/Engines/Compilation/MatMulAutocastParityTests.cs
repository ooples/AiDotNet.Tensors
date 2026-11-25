// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using System.Reflection;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A small dense network (Dense+tanh, a residual Dense+tanh, Dense, mean squared error) under a matmul autocast scope,
/// run as one eager tape step and as a compiled training plan on the same GPU engine. Under autocast every matrix
/// product of the forward and the backward casts its operands to 16 bits, and the sums the backward forms as products
/// with a ones vector do not. Both paths must cast the same products (the cast GEMMs cuBLAS runs are recorded per
/// path and compared), produce the same loss and gradients up to FP32 accumulation order, and differ from the FP32
/// run, which shows the cast took effect. Before the fix, the compiled plan's dense forward ran a narrow FP32 kernel
/// with a bias epilogue and its weight gradients a fused FP32 kernel, both of which ignored the scope: its loss equaled
/// the FP32 loss bit for bit.
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)]
public sealed class MatMulAutocastParityTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private const int Rows = 64, Width = 32, Out = 4;
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public MatMulAutocastParityTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private static float[] Data(int n, double a, double b, double scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(scale * Math.Sin(a * i + b));
        return v;
    }

    private sealed class Net
    {
        internal readonly Tensor<float> X, Y;
        internal readonly Tensor<float>[] Parameters;

        internal Net()
        {
            X = new Tensor<float>(new[] { Rows, Width }, new Vector<float>(Data(Rows * Width, 0.37, 0.1, 1.0))).Gpu();
            Y = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Data(Rows * Out, 0.23, 0.4, 1.0))).Gpu();
            Parameters = new[]
            {
                Param(new[] { Width, Width }, 0.71, 0.2), Param(new[] { Width }, 0.53, 0.3),
                Param(new[] { Width, Width }, 0.29, 0.5), Param(new[] { Width }, 0.47, 0.6),
                Param(new[] { Width, Out }, 0.61, 0.7), Param(new[] { Out }, 0.19, 0.8),
            };
        }

        private static Tensor<float> Param(int[] shape, double a, double b)
        {
            int n = 1;
            foreach (int d in shape) n *= d;
            // Fan-in scaled, so tanh is not saturated.
            return new Tensor<float>(shape, new Vector<float>(Data(n, a, b, 1.0 / Math.Sqrt(shape[0])))).Gpu();
        }

        // Dense(tanh) -> x + Dense(tanh)(x) -> Dense -> mean((out - y)^2), as AiDotNet's dense layers record it: a bias
        // FusedLinear without activation, then the activation.
        internal Tensor<float> Loss(IEngine e)
        {
            var p = Parameters;
            var h1 = e.Tanh(e.FusedLinear(X, p[0], p[1], FusedActivationType.None));
            var h2 = e.TensorAdd(h1, e.Tanh(e.FusedLinear(h1, p[2], p[3], FusedActivationType.None)));
            var o = e.FusedLinear(h2, p[4], p[5], FusedActivationType.None);
            var d = e.TensorSubtract(o, Y);
            return e.TensorMultiplyScalar(e.ReduceSum(e.TensorMultiply(d, d), null), 1f / (Rows * Out));
        }
    }

    private sealed record Run(float Loss, float[][] Gradients, HashSet<CudaBackend.CastGemmRecord> CastGemms);

    private static Run Eager(DirectGpuTensorEngine engine, PrecisionMode? precision)
    {
        var net = new Net();
        var casts = new HashSet<CudaBackend.CastGemmRecord>();
        var backend = (CudaBackend)engine.GetBackend();
        backend.CastGemmObserver = r => { lock (casts) casts.Add(r); };
        try
        {
            using var autocast = precision is { } p ? engine.EnterMatMulAutocast(p) : null;
            using var tape = new GradientTape<float>();
            var loss = net.Loss(engine);
            var grads = tape.ComputeGradients(loss, sources: net.Parameters);
            return new Run(loss.ToArray()[0], net.Parameters.Select(q => grads[q].ToArray()).ToArray(), casts);
        }
        finally
        {
            backend.CastGemmObserver = null;
        }
    }

    // Two plan steps: the first records and the second replays (no optimizer, so both see the same weights).
    private static (Run First, Run Second) Compiled(DirectGpuTensorEngine engine, PrecisionMode? precision)
    {
        var net = new Net();
        var backend = (CudaBackend)engine.GetBackend();
        var casts = new HashSet<CudaBackend.CastGemmRecord>();
        backend.CastGemmObserver = r => { lock (casts) casts.Add(r); };
        try
        {
            using var autocast = precision is { } p ? engine.EnterMatMulAutocast(p) : null;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                net.Loss(engine);
                plan = scope.CompileTraining(net.Parameters);
            }
            using (plan)
            {
                float first = plan.Step().ToArray()[0];
                var firstGrads = plan.Gradients.Select(g => g.ToArray()).ToArray();
                float second = plan.Step().ToArray()[0];
                var secondGrads = plan.Gradients.Select(g => g.ToArray()).ToArray();
                return (new Run(first, firstGrads, casts), new Run(second, secondGrads, casts));
            }
        }
        finally
        {
            backend.CastGemmObserver = null;
        }
    }

    private static double MaxRelative(float[][] a, float[][] b)
    {
        double worst = 0;
        for (int t = 0; t < a.Length; t++)
        {
            double scale = a[t].Max(v => Math.Abs((double)v)) + 1e-30;
            for (int i = 0; i < a[t].Length; i++) worst = Math.Max(worst, Math.Abs(a[t][i] - b[t][i]) / scale);
        }
        return worst;
    }

    private static string Describe(IEnumerable<CudaBackend.CastGemmRecord> records)
        => string.Join(", ", records.OrderBy(r => r.M).ThenBy(r => r.N).ThenBy(r => r.K).ThenBy(r => r.TransA)
            .Select(r => $"{(r.TransA ? "T" : "N")}{(r.TransB ? "T" : "N")} {r.M}x{r.N}x{r.K}"));

    [SkippableTheory]
    [InlineData(PrecisionMode.Float16)]
    [InlineData(PrecisionMode.BFloat16)]
    public void DenseStack_CompiledPlanCastsTheProductsTheEagerTapeCasts(PrecisionMode precision)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        Skip.If(engine.GetBackend() is not CudaBackend, "The cast GEMM records are the CUDA backend's.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var eagerFp32 = Eager(engine, null);
            var (compiledFp32, _) = Compiled(engine, null);
            var eager = Eager(engine, precision);
            var (compiled, replayed) = Compiled(engine, precision);
            int castType = precision == PrecisionMode.BFloat16 ? 14 : 2;   // CUDA_R_16BF, CUDA_R_16F

            _output.WriteLine($"{precision}: loss FP32 eager {eagerFp32.Loss:R} compiled {compiledFp32.Loss:R}; "
                + $"autocast eager {eager.Loss:R} compiled {compiled.Loss:R} replayed {replayed.Loss:R}");
            _output.WriteLine($"cast GEMMs eager:    {Describe(eager.CastGemms)}");
            _output.WriteLine($"cast GEMMs compiled: {Describe(compiled.CastGemms)}");
            double gradEffect = MaxRelative(eagerFp32.Gradients, eager.Gradients);
            double compiledVsEager = MaxRelative(eager.Gradients, compiled.Gradients);
            double replayedVsEager = MaxRelative(eager.Gradients, replayed.Gradients);
            _output.WriteLine($"max relative gradient difference: autocast effect {gradEffect:E3}, compiled - eager "
                + $"{compiledVsEager:E3}, replayed - eager {replayedVsEager:E3}");

            // No cast GEMM without a scope; under one, every cast is to the scope's type.
            Assert.Empty(eagerFp32.CastGemms);
            Assert.Empty(compiledFp32.CastGemms);
            Assert.All(eager.CastGemms.Concat(compiled.CastGemms), r => Assert.Equal(castType, r.CastType));
            // The same products cast on both paths (as distinct problems: the forward products, the weight and the input
            // gradients). A sum the backward forms as a ones-vector product is not a matmul and casts nothing: its K=1 or
            // M=1 problem must not appear.
            Assert.NotEmpty(eager.CastGemms);
            Assert.Equal(Describe(eager.CastGemms), Describe(compiled.CastGemms));
            Assert.DoesNotContain(eager.CastGemms.Concat(compiled.CastGemms), r => r.M == 1 || r.N == 1 || r.K == 1);

            // The cast took effect: 16-bit operands change the loss of this network.
            Assert.NotEqual(eagerFp32.Loss, eager.Loss);
            Assert.NotEqual(compiledFp32.Loss, compiled.Loss);
            // Compiled and eager agree up to FP32 accumulation order. Measured on an RTX 3090: losses within 1e-6
            // relative, gradients within ~1e-5 relative of each tensor's largest entry, against an autocast effect of
            // ~1e-3 (FP16) and ~1e-2 (BF16). The bounds leave a 10x margin and stay well below that effect.
            Assert.True(Math.Abs(compiled.Loss - eager.Loss) <= 1e-5 * Math.Abs(eager.Loss),
                $"compiled loss {compiled.Loss:R} vs eager {eager.Loss:R}");
            Assert.True(Math.Abs(replayed.Loss - eager.Loss) <= 1e-5 * Math.Abs(eager.Loss),
                $"replayed loss {replayed.Loss:R} vs eager {eager.Loss:R}");
            Assert.True(compiledVsEager <= 1e-4 && compiledVsEager < 0.1 * gradEffect,
                $"compiled gradients differ from eager by {compiledVsEager:E3} (autocast effect {gradEffect:E3})");
            Assert.True(replayedVsEager <= 1e-4 && replayedVsEager < 0.1 * gradEffect,
                $"replayed gradients differ from eager by {replayedVsEager:E3} (autocast effect {gradEffect:E3})");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static double MaxRelative(float[] a, float[] b)
    {
        double scale = a.Max(x => Math.Abs((double)x)) + 1e-30, worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]) / scale);
        return worst;
    }

    private void WithCudaEngine(Action<DirectGpuTensorEngine, CudaBackend> body)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        Skip.If(engine.GetBackend() is not CudaBackend, "The cast GEMM records are the CUDA backend's.");
        var backend = (CudaBackend)engine.GetBackend();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            body(engine, backend);
        }
        finally
        {
            backend.CastGemmObserver = null;
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// The fused linear + cross-entropy kernel computes the logits product itself. Under autocast it must read the
    /// operands as the unfused head's cast GEMM does (FP32 accumulation, FP32 softmax and loss): its loss matches the
    /// loss of the cast GEMM's logits, and the engine's head matches it too. Before the fix the kernel ignored the
    /// scope and returned the FP32 loss.
    /// </summary>
    [SkippableTheory]
    [InlineData(PrecisionMode.Float16)]
    [InlineData(PrecisionMode.BFloat16)]
    public void FusedLinearCrossEntropy_UnderAutocast_MatchesTheUnfusedCastHead(PrecisionMode precision)
    {
        WithCudaEngine((engine, backend) =>
        {
            const int n = 48, d = 32, vocab = 40;
            var hHost = Data(n * d, 0.41, 0.3, 1.0);
            var wHost = Data(d * vocab, 0.67, 0.2, 1.0 / Math.Sqrt(d));
            var bHost = Data(vocab, 0.13, 0.9, 0.1);
            var ids = Enumerable.Range(0, n).Select(i => (i * 7 + 3) % vocab).ToArray();
            using var h = backend.AllocateBuffer(hHost);
            using var w = backend.AllocateBuffer(wHost);
            using var b = backend.AllocateBuffer(bHost);
            using var t = backend.AllocateBuffer(ids.Select(i => (float)i).ToArray());   // the kernel reads its ids as floats
            using var logits = backend.AllocateBuffer(n * vocab);

            // The unfused head: logits from the GEMM entry point, the loss from them in double on the host.
            double Unfused()
            {
                backend.Gemm(h, w, logits, n, vocab, d, 1f, 0f);
                var z = backend.DownloadBuffer(logits);
                double sum = 0;
                for (int r = 0; r < n; r++)
                {
                    double max = double.NegativeInfinity;
                    for (int c = 0; c < vocab; c++) max = Math.Max(max, (double)(z[r * vocab + c] + bHost[c]));
                    double s = 0;
                    for (int c = 0; c < vocab; c++) s += Math.Exp(z[r * vocab + c] + bHost[c] - max);
                    sum += max + Math.Log(s) - (z[r * vocab + ids[r]] + bHost[ids[r]]);
                }
                return sum / n;
            }

            float fusedFp32 = backend.FusedLinearCrossEntropyIndex(h, w, b, t, n, d, vocab);
            double unfusedFp32 = Unfused();
            float fused, head;
            double unfused;
            using (engine.EnterMatMulAutocast(precision))
            {
                fused = backend.FusedLinearCrossEntropyIndex(h, w, b, t, n, d, vocab);
                unfused = Unfused();
                var hT = new Tensor<float>(new[] { n, d }, new Vector<float>(hHost)).Gpu();
                var wT = new Tensor<float>(new[] { d, vocab }, new Vector<float>(wHost)).Gpu();
                var bT = new Tensor<float>(new[] { vocab }, new Vector<float>(bHost)).Gpu();
                var idsT = new Tensor<int>(new[] { n }, new Vector<int>(ids));
                head = engine.FusedLinearCrossEntropyWithLogits(hT, wT, bT, idsT).ToArray()[0];
            }
            _output.WriteLine($"{precision}: FP32 fused {fusedFp32:R} unfused {unfusedFp32:R}; autocast fused {fused:R} "
                + $"engine {head:R} unfused {unfused:R}");

            Assert.NotEqual(unfusedFp32, unfused);   // the cast changes this head's loss
            Assert.True(Math.Abs(fusedFp32 - unfusedFp32) <= 1e-5 * Math.Abs(unfusedFp32),
                $"FP32 fused {fusedFp32:R} vs unfused {unfusedFp32:R}");
            Assert.True(Math.Abs(fused - unfused) <= 1e-5 * Math.Abs(unfused),
                $"autocast fused kernel {fused:R} vs unfused cast head {unfused:R} (FP32 {unfusedFp32:R})");
            Assert.True(Math.Abs(head - unfused) <= 1e-5 * Math.Abs(unfused),
                $"autocast engine head {head:R} vs unfused cast head {unfused:R} (FP32 {unfusedFp32:R})");
        });
    }

    /// <summary>
    /// GemmAsync and FusedGemmBiasActivationAsync on a caller's stream follow the open scope as the synchronous GEMM
    /// does: one cast GEMM each, the synchronous cast GEMM's result. Before the fix they ran FP32 cuBLAS and the FP32
    /// fused kernel under the scope.
    /// </summary>
    [SkippableTheory]
    [InlineData(PrecisionMode.Float16)]
    [InlineData(PrecisionMode.BFloat16)]
    public void AsyncGemms_UnderAutocast_CastLikeTheSynchronousGemm(PrecisionMode precision)
    {
        WithCudaEngine((engine, backend) =>
        {
            const int m = 24, n = 20, k = 36;
            var biasHost = Data(n, 0.31, 0.5, 0.2);
            using var a = backend.AllocateBuffer(Data(m * k, 0.43, 0.1, 1.0));
            using var bm = backend.AllocateBuffer(Data(k * n, 0.59, 0.7, 1.0 / Math.Sqrt(k)));
            using var bias = backend.AllocateBuffer(biasHost);
            using var c = backend.AllocateBuffer(m * n);
            using var stream = backend.CreateStream(GpuStreamType.Compute);

            float[] Async()
            {
                backend.GemmAsync(a, bm, c, m, n, k, 1f, 0f, stream);
                backend.SynchronizeStream(stream);
                return backend.DownloadBuffer(c);
            }
            float[] FusedAsync()
            {
                backend.FusedGemmBiasActivationAsync(a, bm, bias, c, m, n, k, FusedActivationType.Tanh, stream);
                backend.SynchronizeStream(stream);
                return backend.DownloadBuffer(c);
            }

            var asyncFp32 = Async();
            int casts = 0;
            backend.CastGemmObserver = _ => Interlocked.Increment(ref casts);
            float[] sync, async, fused;
            int asyncCasts, fusedCasts;
            using (engine.EnterMatMulAutocast(precision))
            {
                backend.Gemm(a, bm, c, m, n, k, 1f, 0f);
                sync = backend.DownloadBuffer(c);
                int before = casts;
                async = Async();
                asyncCasts = casts - before;
                before = casts;
                fused = FusedAsync();
                fusedCasts = casts - before;
            }
            var reference = new float[m * n];
            for (int i = 0; i < reference.Length; i++) reference[i] = (float)Math.Tanh(sync[i] + biasHost[i % n]);
            double effect = MaxRelative(asyncFp32, sync), asyncVsSync = MaxRelative(sync, async);
            double fusedVsReference = MaxRelative(reference, fused);
            _output.WriteLine($"{precision}: cast effect {effect:E3}, async - sync {asyncVsSync:E3}, fused async - "
                + $"reference {fusedVsReference:E3}, cast GEMMs async {asyncCasts} fused {fusedCasts}");

            Assert.True(effect > 0, "the cast changes this product");
            Assert.Equal(1, asyncCasts);
            Assert.Equal(1, fusedCasts);
            Assert.True(asyncVsSync <= 1e-6 && asyncVsSync < 0.01 * effect,
                $"async cast GEMM differs from the synchronous one by {asyncVsSync:E3} (cast effect {effect:E3})");
            Assert.True(fusedVsReference <= 1e-6 && fusedVsReference < 0.01 * effect,
                $"fused async GEMM differs from tanh(cast GEMM + bias) by {fusedVsReference:E3} (cast effect {effect:E3})");
        });
    }

    /// <summary>
    /// The compiled plan's optimizer update is not part of the autocast region: a GEMM issued by the update runs in FP32
    /// while the step's forward and backward products are cast. The configured update is wrapped to issue one GEMM
    /// (no built-in update rule uses one). Before the fix the scope stayed open across the update and that GEMM was cast.
    /// </summary>
    [SkippableTheory]
    [InlineData(PrecisionMode.Float16)]
    [InlineData(PrecisionMode.BFloat16)]
    public void InPlanOptimizerUpdate_RunsOutsideTheAutocastScope(PrecisionMode precision)
    {
        WithCudaEngine((engine, backend) =>
        {
            var net = new Net();
            using var a = backend.AllocateBuffer(Data(64, 0.43, 0.1, 1.0));
            using var bm = backend.AllocateBuffer(Data(64, 0.59, 0.7, 1.0));
            using var c = backend.AllocateBuffer(64);
            bool inUpdate = false;
            int updateGemms = 0, castInUpdate = 0, castOutside = 0;
            backend.CastGemmObserver = _ =>
            {
                if (inUpdate) Interlocked.Increment(ref castInUpdate);
                else Interlocked.Increment(ref castOutside);
            };
            using var autocast = engine.EnterMatMulAutocast(precision);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                net.Loss(engine);
                plan = scope.CompileTraining(net.Parameters);
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, 1e-3);
                const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
                var type = plan.GetType();
                var updateField = TestRequire.NotNull(type.GetField("_optimizerUpdate", flags), "_optimizerUpdate");
                var update = TestRequire.NotNull((Action?)updateField.GetValue(plan), "the configured update");
                // The host update runs every step: a captured update graph would not run the wrapper.
                type.GetField("_capturedOptimizerFactory", flags)?.SetValue(plan, null);
                type.GetField("_capturedOptimizer", flags)?.SetValue(plan, null);
                updateField.SetValue(plan, (Action)(() =>
                {
                    update();
                    inUpdate = true;
                    try
                    {
                        backend.Gemm(a, bm, c, 8, 8, 8, 1f, 0f);
                        updateGemms++;
                    }
                    finally
                    {
                        inUpdate = false;
                    }
                }));
                plan.Step();
                plan.Step();
            }
            _output.WriteLine($"{precision}: update GEMMs {updateGemms}, cast inside the update {castInUpdate}, "
                + $"cast outside it {castOutside}");
            Assert.True(updateGemms > 0, "the wrapped update ran");
            Assert.True(castOutside > 0, "the step's own products are cast");
            Assert.Equal(0, castInUpdate);
        });
    }
}
