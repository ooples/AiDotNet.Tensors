using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class FlashAttentionReplayTests : IDisposable
{
    private const int B = 2, H = 2, Q = 2, K = 3, D = 2;
    private const double Scale = 0.7;
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Theory]
    [InlineData(false, false, 0)]
    [InlineData(false, true, 0)]
    [InlineData(false, false, 3)]
    [InlineData(false, true, 4)]
    [InlineData(true, false, 0)]
    [InlineData(true, true, 0)]
    [InlineData(true, false, 3)]
    [InlineData(true, true, 4)]
    public void Derived_query_rebinds_without_execution_and_replays_current_output_statistics_and_gradients(
        bool useDouble, bool causal, int biasRank)
    {
        if (useDouble) CheckReplay<double>(new CpuEngine(), causal, biasRank, false);
        else CheckReplay<float>(new CpuEngine(), causal, biasRank, false);
    }

    [SkippableTheory]
    [Trait("Category", "GPU")]
    [InlineData(false, 0)]
    [InlineData(true, 3)]
    [InlineData(false, 4)]
    public void Ordinary_gpu_replay_executes_device_attention_and_matches_independent_reference(bool causal, int biasRank)
    {
        DirectGpuTensorEngine? device = null;
        try { device = new NativeIdentityAttentionEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (device)
        {
            bool available = device?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "The validation lane requires a working direct GPU backend.");
            Skip.If(!available, "No direct GPU backend available.");
            CheckReplay<float>(device!, causal, biasRank, true);
        }
    }

    private static void CheckReplay<T>(IEngine engine, bool causal, int biasRank, bool device)
    {
        AiDotNetEngine.Current = engine;
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> New(int[] shape, double phase) => new(
            Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
                .Select(i => ops.FromDouble(0.6 * Math.Sin(0.71 * i + phase))).ToArray(), shape);
        // Q is packed [B,Q,H,D], exactly the unrealized Permute producer seen by diffusion consumers.
        var q = New([B, Q, H, D], 0.2);
        var k = New([B, H, K, D], 1.1);
        var v = New([B, H, K, D], 2.3);
        var bias = biasRank == 0 ? null : New(biasRank == 3 ? [H, Q, K] : [B, H, Q, K], 0.5);
        var parameters = bias is null ? new[] { q, k, v } : new[] { q, k, v, bias };
        var weights = New([B, H, Q, D], 0.8);
        Tensor<T>? observed = null, statistics = null;
        Tensor<T> Objective()
        {
            observed = engine.FlashAttention(engine.TensorPermute(q, [0, 2, 1, 3]), k, v,
                Scale, causal, out var stats, bias);
            statistics = stats;
            GradientRecording.RetainOutput(observed);
            GradientRecording.RetainOutput(stats);
            return engine.ReduceSum(engine.TensorMultiply(observed, weights), null);
        }

        using var recording = ClosureObjectiveGuard<T>.BeginRecording();
        CompiledTrainingPlan<T> plan;
        GpuLaunchProbe.Reset();
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = Objective();
            Assert.False(Assert.IsType<LazyNode<T>>(observed!.LazySource).IsRealized);
            Assert.False(Assert.IsType<LazyNode<T>>(statistics!.LazySource).IsRealized);
            Assert.Null(scope.HostReadOp);
            Assert.Equal(0, GpuLaunchProbe.Count);
            plan = scope.CompileTraining(parameters, loss);
        }
        // Probe callbacks replace these captures with stand-ins; retain the actual compiled outputs.
        var tracedOutput = observed!;
        var tracedStats = statistics!;
        using (plan)
        {
            var guard = ClosureObjectiveGuard<T>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
            for (int step = 0; step < 3; step++)
            {
                weights = New([B, H, Q, D], 0.8 + 0.9 * step);
                if (bias is not null)
                {
                    for (int i = 0; i < bias.Length; i++) bias.SetFlat(i, ops.FromDouble(0.2 * Math.Cos(i + step)));
                    bias.IncrementVersion();
                }
                var point = parameters.Take(3).Select(t => t.ToArray().Select(ops.ToDouble).ToArray()).ToArray();
                var ws = weights.ToArray().Select(ops.ToDouble).ToArray();
                var bs = bias?.ToArray().Select(ops.ToDouble).ToArray();
                var expected = Reference(point, ws, bs, biasRank, causal);
                GpuLaunchProbe.Reset();
                Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                Assert.Equal(0, GpuLaunchProbe.Count);
                Assert.Equal(0, GpuLaunchProbe.Readbacks);
                double actualLoss = ops.ToDouble(plan.Step().ToArray()[0]);
                if (device)
                {
                    Assert.True(GpuLaunchProbe.Count > 0, "Replay must launch device work.");
                    Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, entry => entry.Contains("FlashAttention"));
                }
                double tolerance = typeof(T) == typeof(double) ? 2e-10 : 4e-5;
                Near(expected.Loss, actualLoss, tolerance);
                Check(expected.Output, tracedOutput.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
                Check(expected.Stats, tracedStats.ToArray().Select(ops.ToDouble).ToArray(), tolerance);
                for (int p = 0; p < 3; p++)
                {
                    Check(expected.Gradients[p], plan.Gradients[p].ToArray().Select(ops.ToDouble).ToArray(), tolerance);
                    Assert.Equal(step + 1L, plan.ParameterAcceptedUpdates[p]);
                    Assert.Contains(parameters[p].ToArray().Select(ops.ToDouble).Zip(point[p], (a, b) => Math.Abs(a - b)), d => d > 1e-7);
                    if (typeof(T) == typeof(double))
                        for (int i = 0; i < point[p].Length; i++)
                        {
                            const double delta = 1e-6;
                            double saved = point[p][i];
                            point[p][i] = saved + delta;
                            double plus = Reference(point, ws, bs, biasRank, causal).Loss;
                            point[p][i] = saved - delta;
                            double minus = Reference(point, ws, bs, biasRank, causal).Loss;
                            point[p][i] = saved;
                            Near((plus - minus) / (2 * delta), expected.Gradients[p][i], 2e-8);
                        }
                }
                if (bias is not null)
                {
                    Assert.Equal(0L, plan.ParameterAcceptedUpdates[3]);
                    Check(bs!, bias.ToArray().Select(ops.ToDouble).ToArray(), 0);
                }
            }
        }
    }

    [Fact]
    public void Statistics_first_realization_runs_the_forward_and_invalid_metadata_leaves_scope_usable()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var q = new Tensor<double>(Enumerable.Range(0, B * Q * H * D).Select(i => 0.1 * i).ToArray(), [B, Q, H, D]);
        var k = new Tensor<double>(Enumerable.Range(0, B * H * K * D).Select(i => Math.Cos(i)).ToArray(), [B, H, K, D]);
        var v = new Tensor<double>([B, H, K, D]);
        using var scope = GraphMode.EnableTraining(new[] { q, k, v });
        Assert.Throws<ArgumentException>(() => engine.FlashAttention(q, new Tensor<double>([B, H, K]), v, Scale, false, out _));
        Assert.Throws<ArgumentException>(() => engine.FlashAttention(q, k, new Tensor<double>([B, H, K + 1, D]), Scale, false, out _));
        Assert.Throws<ArgumentException>(() => engine.FlashAttention(q, k, v, Scale, false, out _, new Tensor<double>([H, Q, K + 1])));
        Assert.Same(scope, GraphMode.Current);
        var output = engine.FlashAttention(engine.TensorPermute(q, [0, 2, 1, 3]), k, v, Scale, false, out var stats);
        var node = Assert.IsType<LazyNode<double>>(output.LazySource);
        Assert.False(node.IsRealized);
        var expected = Reference([q.ToArray(), k.ToArray(), v.ToArray()], new double[B * H * Q * D], null, 0, false);
        Check(expected.Stats, stats.ToArray(), 2e-10);
        Assert.True(node.IsRealized);
        Check(expected.Output, output.ToArray(), 2e-10);
    }

    [Fact]
    public void Failed_forward_propagates_without_advancing_optimizer_and_next_step_refreshes_statistics()
    {
        var engine = new FailingAttentionEngine();
        AiDotNetEngine.Current = engine;
        var q = new Tensor<double>(new[] { 0.2, -0.3, 0.7, 0.1 }, [1, 1, 2, 2]);
        var k = new Tensor<double>(new[] { -0.1, 0.4, 0.2, 0.8 }, [1, 1, 2, 2]);
        var v = new Tensor<double>(new[] { 0.6, -0.2, 0.5, 0.3 }, [1, 1, 2, 2]);
        Tensor<double>[] parameters = [q, k, v];
        Tensor<double> output, stats;
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            output = engine.FlashAttention(engine.TensorAddScalar(q, 0.0), k, v, Scale, false, out stats);
            GradientRecording.RetainOutput(output);
            GradientRecording.RetainOutput(stats);
            plan = scope.CompileTraining(parameters, engine.ReduceSum(output, null));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
            var before = parameters.Select(p => p.ToArray()).ToArray();
            engine.FailNextForward = true;
            var error = Assert.Throws<InvalidOperationException>(() => plan.Step());
            Assert.Equal("Injected FlashAttention execution failure", error.Message);
            Assert.Null(GraphMode.Current);
            Assert.Equal(new long[] { 0, 0, 0 }, plan.ParameterAcceptedUpdates);
            for (int p = 0; p < parameters.Length; p++) Assert.Equal(before[p], parameters[p].ToArray());
            using (var recoveryScope = GraphMode.EnableTraining(parameters))
            {
                var recovery = engine.FlashAttention(engine.TensorAddScalar(q, 0.0), k, v, Scale, false, out _);
                Assert.NotNull(recovery.LazySource);
                Assert.Null(recoveryScope.HostReadOp);
            }
            // Independent eager evaluation supplies the pre-update expected state after the failed step.
            var referenceEngine = new CpuEngine();
            var expected = referenceEngine.FlashAttention(q, k, v, Scale, false, out var expectedStats).ToArray();
            Near(expected.Sum(), plan.Step().ToArray()[0], 2e-10);
            Check(expected, output.ToArray(), 2e-10);
            Check(expectedStats.ToArray(), stats.ToArray(), 2e-10);
            Assert.Equal(new long[] { 1, 1, 1 }, plan.ParameterAcceptedUpdates);
            Assert.Null(GraphMode.Current);
        }
    }

    private sealed class FailingAttentionEngine : CpuEngine
    {
        internal bool FailNextForward;
        public override Tensor<T> FlashAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value,
            double? scale, bool isCausal, out Tensor<T> softmaxStats, Tensor<T>? attentionBias = null)
        {
            if (!GraphMode.IsActive && FailNextForward)
            {
                FailNextForward = false;
                throw new InvalidOperationException("Injected FlashAttention execution failure");
            }
            return base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats, attentionBias);
        }
    }

    [Fact]
    public void Statistics_only_loss_has_no_differentiable_edges_or_optimizer_enrollment()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var q = new Tensor<double>(new[] { 0.2, 0.5 }, [1, 1, 1, 2]);
        var k = new Tensor<double>(new[] { 0.3, 0.4 }, [1, 1, 1, 2]);
        var v = new Tensor<double>(new[] { 0.7, 0.8 }, [1, 1, 1, 2]);
        var anchor = new Tensor<double>(new[] { 2.0 }, [1]);
        Tensor<double>[] parameters = [anchor, q, k, v];
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            engine.FlashAttention(q, k, v, Scale, false, out var stats);
            var node = Assert.IsType<LazyNode<double>>(stats.LazySource);
            Assert.Empty(node.GetDifferentiableInputs());
            plan = scope.CompileTraining(parameters,
                engine.TensorAdd(anchor, engine.ReduceSum(stats, null)));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
            Near(2 + Scale * (0.2 * 0.3 + 0.5 * 0.4), plan.Step().ToArray()[0], 2e-10);
            Assert.Equal(new long[] { 1, 0, 0, 0 }, plan.ParameterAcceptedUpdates);
        }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Cuda_whole_step_capture_contains_flash_attention_and_preserves_retained_output_lifetimes()
    {
        DirectGpuTensorEngine? device = null;
        try { device = new NativeIdentityAttentionEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (device)
        {
            bool available = device?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "The validation lane requires a working direct GPU backend.");
            Skip.If(!available, "No direct GPU backend available.");
            Skip.If(device!.GetBackend() is not CudaBackend, "Whole-step capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0", "CUDA capture disabled for this process.");
            var gpu = (NativeIdentityAttentionEngine)device;
            var backend = Assert.IsType<CudaBackend>(gpu.GetBackend());
            bool? prior = backend.DeclaredKernelsSwitch;
            try
            {
                backend.DeclaredKernelsSwitch = false;
                // Six steps include warmup, capture, and multiple native replays; retained outputs are read each step.
                var eager = CaptureRun(gpu, false);
                var captured = CaptureRun(gpu, true);
                Check(eager, captured, 7e-5);
                Assert.True(gpu.SawCapturedFlashKernel, "The native captured graph must contain flash_attention_v2.");
            }
            finally { backend.DeclaredKernelsSwitch = prior; }
        }
    }

    private static double[] CaptureRun(NativeIdentityAttentionEngine gpu, bool capture)
    {
        AiDotNetEngine.Current = gpu;
        Tensor<float> New(int[] shape, double phase) => new(
            Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (float)(0.3 * Math.Sin(i + phase))).ToArray(), shape);
        var q = New([B, Q, H, D], 0.2).Gpu();
        var k = New([B, H, K, D], 0.8).Gpu();
        var v = New([B, H, K, D], 1.7).Gpu();
        var weights = New([B, H, Q, D], 2.1).Gpu();
        Tensor<float>[] parameters = [q, k, v];
        Tensor<float> output, stats;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            output = gpu.FlashAttention(gpu.TensorPermute(q, [0, 2, 1, 3]), k, v, Scale, false, out stats);
            GradientRecording.RetainOutput(output);
            GradientRecording.RetainOutput(stats);
            plan = scope.CompileTraining(parameters, gpu.ReduceSum(gpu.TensorMultiply(output, weights), null));
        }
        var results = new System.Collections.Generic.List<double>();
        using (plan)
        {
            if (!capture) plan.DisableGraphStep();
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
            for (int step = 0; step < 6; step++)
            {
                var point = parameters.Select(p => p.ToArray().Select(x => (double)x).ToArray()).ToArray();
                var expected = Reference(point, weights.ToArray().Select(x => (double)x).ToArray(), null, 0, false);
                Near(expected.Loss, plan.Step().ToArray()[0], 7e-5);
                Check(expected.Output, output.ToArray().Select(x => (double)x).ToArray(), 7e-5);
                Check(expected.Stats, stats.ToArray().Select(x => (double)x).ToArray(), 7e-5);
                for (int p = 0; p < 3; p++)
                    Check(expected.Gradients[p], plan.Gradients[p].ToArray().Select(x => (double)x).ToArray(), 7e-5);
            }
            if (capture)
            {
                Assert.Null(plan.GraphCaptureFailureReason);
                var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(plan)!;
                Assert.NotEqual(IntPtr.Zero, exec);
            }
            results.AddRange(output.ToArray().Select(x => (double)x));
            results.AddRange(stats.ToArray().Select(x => (double)x));
            foreach (var parameter in parameters) results.AddRange(parameter.ToArray().Select(x => (double)x));
        }
        // Plan retirement must also preserve the caller-retained buffers after transient results return to the pool.
        results.AddRange(output.ToArray().Select(x => (double)x));
        results.AddRange(stats.ToArray().Select(x => (double)x));
        return results.ToArray();
    }

    private sealed class NativeIdentityAttentionEngine : DirectGpuTensorEngine
    {
        internal bool SawCapturedFlashKernel;
        public override Tensor<T> FlashAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value,
            double? scale, bool isCausal, out Tensor<T> softmaxStats, Tensor<T>? attentionBias = null)
        {
            long launchesBefore = GpuLaunchProbe.TotalEver;
            var result = base.FlashAttention(query, key, value, scale, isCausal, out softmaxStats, attentionBias);
            if (!GraphMode.IsActive)
                Assert.True(GpuLaunchProbe.TotalEver > launchesBefore, "FlashAttention itself must dispatch native device work, independently of optimizer or copy launches.");

            if (!GraphMode.IsActive && GetBackend() is CudaBackend backend)
            {
                using var context = (IDisposable)typeof(CudaBackend).GetMethod("PushContext", BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(backend, null)!;

                var stream = (IntPtr)typeof(CudaBackend).GetProperty("_stream", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(backend)!;
                int rc = StreamCaptureInfo(stream, out int status, out _, out var graph, out _, out _);
                Assert.Equal(0, rc);
                if (status == 1)
                {
                    ulong count = 0;
                    Assert.Equal(CudaResult.Success, CudaNativeBindings.cuGraphGetNodes(graph, IntPtr.Zero, ref count));
                    var nodes = new IntPtr[count];
                    Assert.Equal(CudaResult.Success, CudaNativeBindings.cuGraphGetNodesArray(graph, nodes, ref count));
                    var scratch = Marshal.AllocHGlobal(256);
                    try
                    {
                        foreach (var node in nodes)
                        {
                            Assert.Equal(CudaResult.Success, CudaNativeBindings.cuGraphNodeGetType(node, out int type));
                            if (type != 0) continue;
                            Assert.Equal(CudaResult.Success, CudaNativeBindings.cuGraphKernelNodeGetParams(node, scratch));
                            Assert.Equal(CudaResult.Success, CudaNativeBindings.cuFuncGetName(out var name, Marshal.ReadIntPtr(scratch)));
                            if (Marshal.PtrToStringAnsi(name) == "flash_attention_v2") SawCapturedFlashKernel = true;
                        }
                    }
                    finally { Marshal.FreeHGlobal(scratch); }
                }
            }
            return result;
        }
    }

    [DllImport("nvcuda", EntryPoint = "cuStreamGetCaptureInfo_v2")]
    private static extern int WindowsStreamCaptureInfo(IntPtr stream, out int status, out ulong id,
        out IntPtr graph, out IntPtr dependencies, out ulong count);
    [DllImport("libcuda", EntryPoint = "cuStreamGetCaptureInfo_v2")]
    private static extern int LinuxStreamCaptureInfo(IntPtr stream, out int status, out ulong id,
        out IntPtr graph, out IntPtr dependencies, out ulong count);
    private static int StreamCaptureInfo(IntPtr stream, out int status, out ulong id,
        out IntPtr graph, out IntPtr dependencies, out ulong count) => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        ? WindowsStreamCaptureInfo(stream, out status, out id, out graph, out dependencies, out count)
        : LinuxStreamCaptureInfo(stream, out status, out id, out graph, out dependencies, out count);

    // Independent scalar stable softmax and analytic reverse-mode equations. Q uses packed storage.
    private static (double Loss, double[] Output, double[] Stats, double[][] Gradients) Reference(
        double[][] point, double[] weights, double[]? bias, int biasRank, bool causal)
    {
        var output = new double[B * H * Q * D];
        var stats = new double[B * H * Q];
        var gradients = point.Select(p => new double[p.Length]).ToArray();
        double loss = 0;
        for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
        for (int row = 0; row < Q; row++)
        {
            int qi = ((b * Q + row) * H + h) * D;
            int oi = ((b * H + h) * Q + row) * D;
            var scores = new double[K];
            double max = double.NegativeInfinity;
            for (int col = 0; col < K; col++)
            {
                int ki = ((b * H + h) * K + col) * D;
                double score = 0;
                for (int d = 0; d < D; d++) score += point[0][qi + d] * point[1][ki + d] * Scale;
                if (bias is not null) score += bias[(((biasRank == 3 ? 0 : b) * H + h) * Q + row) * K + col];
                scores[col] = causal && col > row ? double.NegativeInfinity : score;
                max = Math.Max(max, scores[col]);
            }
            double sum = scores.Sum(s => Math.Exp(s - max));
            stats[(b * H + h) * Q + row] = max + Math.Log(sum);
            var probabilities = scores.Select(s => Math.Exp(s - max) / sum).ToArray();
            var dp = new double[K];
            for (int col = 0; col < K; col++)
            {
                int vi = ((b * H + h) * K + col) * D;
                for (int d = 0; d < D; d++)
                {
                    output[oi + d] += probabilities[col] * point[2][vi + d];
                    gradients[2][vi + d] += probabilities[col] * weights[oi + d];
                    dp[col] += weights[oi + d] * point[2][vi + d];
                }
            }
            double correction = probabilities.Zip(dp, (p, g) => p * g).Sum();
            for (int col = 0; col < K; col++)
            {
                int ki = ((b * H + h) * K + col) * D;
                double ds = probabilities[col] * (dp[col] - correction) * Scale;
                for (int d = 0; d < D; d++)
                {
                    gradients[0][qi + d] += ds * point[1][ki + d];
                    gradients[1][ki + d] += ds * point[0][qi + d];
                }
            }
            for (int d = 0; d < D; d++) loss += output[oi + d] * weights[oi + d];
        }
        return (loss, output, stats, gradients);
    }

    private static void Check(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Near(expected[i], actual[i], tolerance);
    }

    private static void Near(double expected, double actual, double tolerance) =>
        Assert.True(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, got {actual:R}; tolerance {tolerance:R}.");
}
