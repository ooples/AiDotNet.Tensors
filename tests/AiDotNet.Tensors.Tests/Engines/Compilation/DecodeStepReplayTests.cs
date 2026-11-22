using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled decode step whose in-place ops write tensors the plan also treats specially: a KV cache passed as an
/// explicit input slot (KvCacheAppend), a recurrent state written in place by GatedDeltaNetStep, a declared state
/// update, and a position leaf. Every execution, the capture call included, advances each exactly once; a cache and
/// state reset between sequences is honoured; and a captured plan keeps replaying through it. Checked against the same
/// ops run eagerly on the CPU engine over separate tensors.
/// </summary>
[Collection("DirectGpuSerial")]
public class DecodeStepReplayTests
{
    private const int Capacity = 4, Width = 4, Rows = 1;

    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static Tensor<float> T(int[] shape, Func<int, float> f)
        => new(shape, new Vector<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(f).ToArray()));

    private static void Close(float[] expect, float[] got, string what)
    {
        Assert.Equal(expect.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-4f * (1 + Math.Abs(expect[i])), $"{what}[{i}]: {got[i]} vs {expect[i]}");
    }

    private sealed class Decoder
    {
        public readonly Tensor<float> Cache = new([1, Capacity, Width]);
        public readonly Tensor<float> Position = new([1]);
        public readonly Tensor<float> State = new([Rows, Width, Width]);
        public readonly Tensor<float> Counter = new([1, Width]);
        public readonly Tensor<float> K = T([Rows, Width], i => 0.2f * MathF.Cos(i + 1));
        public readonly Tensor<float> V = T([Rows, Width], i => 0.3f * MathF.Sin(2 * i + 1));
        public readonly Tensor<float> A = T([Rows], _ => 0.9f);
        public readonly Tensor<float> B = T([Rows], _ => 0.4f);

        // One decode position: attention over the cache after appending x, plus the gated-delta readout of q, plus 0 x
        // the counter (so the declared state is read). Returns the output and the counter's next value.
        public (Tensor<float> Y, Tensor<float> Next) Step(IEngine e, Tensor<float> x, Tensor<float> q)
        {
            var mask = e.KvCacheAppend(Cache, Position, x);                                   // [Capacity]
            var keys = e.Reshape(Cache, [Capacity, Width]);
            var scores = e.TensorAdd(e.Reshape(e.TensorMatMul(keys, e.Reshape(q, [Width, 1])), [Capacity]), mask);
            var attn = e.TensorMatMul(e.Reshape(e.Softmax(scores, -1), [1, Capacity]), keys);  // [1, Width]
            var o = e.GatedDeltaNetStep(State, q, K, V, A, B, GatedDeltaRule.Standard);        // [Rows, Width]
            var y = e.TensorAdd(attn, e.TensorMultiplyScalar(o, 2f));
            y = e.TensorAdd(y, e.TensorMultiplyScalar(Counter, 0f));
            return (y, e.TensorAddScalar(Counter, 1f));
        }

        public void Reset(IEngine e)
        {
            e.TensorFill(Cache, 0f);
            e.TensorFill(Position, 0f);
            e.TensorFill(State, 0f);
            e.TensorFill(Counter, 0f);
        }
    }

    private static Tensor<float> X(int t) => T([1, Width], i => MathF.Sin(i + 3 * t) + 0.1f * t);
    private static Tensor<float> Q(int t) => T([Rows, Width], i => 0.5f * MathF.Cos(2 * i - t));

    // sequences x steps decode positions; between sequences the plan's cache, position, state and counter are reset with
    // engine ops (as a serving loop starts a new request), and the reference is reset alongside.
    private static void Run(IEngine engine, bool expectGraph, int sequences)
    {
        var cpu = new CpuEngine();
        var plan = new Decoder();
        var reference = new Decoder();
        var x = new Tensor<float>([1, Width]);
        var q = new Tensor<float>([Rows, Width]);
        CompiledInferencePlan<float> compiled;
        using (new EngineScope(engine))
        using (var scope = GraphMode.EnableInference())
        {
            var (y, next) = plan.Step(engine, x, q);
            // The cache is an explicit input slot: the caller feeds the same tensor every step.
            compiled = scope.CompileInference(y, new[] { x, q, plan.Cache }, new[] { (plan.Counter, next) });
        }
        // Compiling advanced nothing.
        Assert.Equal(0f, plan.Position.ToArray()[0]);
        Assert.All(plan.Cache.ToArray(), c => Assert.Equal(0f, c));
        Assert.All(plan.State.ToArray(), c => Assert.Equal(0f, c));
        using (compiled)
        {
            for (int s = 0; s < sequences; s++)
            {
                if (s > 0)
                {
                    plan.Reset(engine);
                    reference.Reset(cpu);
                }
                for (int t = 0; t < Capacity; t++)
                {
                    var (xt, qt) = (X(t + 7 * s), Q(t + 5 * s));
                    compiled.SetInputs(new[] { xt, qt, plan.Cache });
                    var y = compiled.Execute().ToArray();
                    Tensor<float> expect;
                    using (new EngineScope(cpu))
                    {
                        var (ey, enext) = reference.Step(cpu, xt, qt);
                        expect = ey;
                        ((ISignatureTensor)reference.Counter).CopyLeafFrom(enext);
                    }
                    string at = $"sequence {s} step {t}";
                    Close(expect.ToArray(), y, at + " y");
                    Assert.Equal(t + 1, plan.Position.ToArray()[0]);
                    Assert.Equal(t + 1, plan.Counter.ToArray()[0]);
                    Close(reference.Cache.ToArray(), plan.Cache.ToArray(), at + " cache");
                    Close(reference.State.ToArray(), plan.State.ToArray(), at + " state");
                }
            }
            if (expectGraph)
            {
                var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                var disabled = typeof(CompiledInferencePlan<float>).GetField("_graphCaptureDisabled", BindingFlags.NonPublic | BindingFlags.Instance)!;
                Assert.False((bool)disabled.GetValue(compiled)!, "graph capture was disabled");
                Assert.True(exec.GetValue(compiled) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
            }
        }
    }

    [Fact]
    public void A_cpu_plan_with_a_cache_input_and_an_in_place_state_advances_each_once_per_execution()
        => Run(new CpuEngine(), expectGraph: false, sequences: 1);

    [Fact]
    public void A_cpu_plan_follows_a_cache_and_state_reset_between_sequences()
        => Run(new CpuEngine(), expectGraph: false, sequences: 3);

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_gpu_plan_with_a_cache_input_and_an_in_place_state_advances_each_once_per_execution(bool capture)
        => RunGpu(capture, sequences: 1);

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_gpu_plan_follows_a_cache_and_state_reset_between_sequences_and_keeps_replaying(bool capture)
        => RunGpu(capture, sequences: 3);

    // Steady-state replay with nothing reading the cache, position, state or counter on the host: the pre-replay check
    // must neither download a written target to compare it nor re-upload one. A host read then costs one compare (and
    // no upload, the data being unchanged); a reset costs uploads.
    // Under the op lease: a backend lookup outside one fails the strict-lease run (AIDOTNET_GPU_LEASE_STRICT=1).
    private static bool IsCuda(DirectGpuTensorEngine? gpu)
    {
        if (gpu is null) return false;
        using (gpu.EnterGpuOp())
            return gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend;
    }

    [SkippableFact]
    public void A_steady_state_replay_with_no_host_reads_downloads_and_compares_nothing()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            Skip.IfNot(IsCuda(gpu), "Inference graph capture is CUDA-only.");
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                var plan = new Decoder();
                var x = new Tensor<float>([1, Width]);
                var q = new Tensor<float>([Rows, Width]);
                CompiledInferencePlan<float> compiled;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var (y, next) = plan.Step(gpu, x, q);
                    compiled = scope.CompileInference(y, new[] { x, q, plan.Cache }, new[] { (plan.Counter, next) });
                }
                var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                using (compiled)
                {
                    void Step(int t)
                    {
                        compiled.SetInputs(new[] { X(t), Q(t), plan.Cache });
                        compiled.Execute();
                    }
                    // Warm-up and capture (3 calls), then steady replays.
                    for (int t = 0; t < 3; t++) Step(t);
                    Assert.True(exec.GetValue(compiled) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                    long compares = compiled.WrittenTargetCompares, uploads = compiled.WrittenTargetUploads;
                    Step(3);
                    Assert.Equal(compares, compiled.WrittenTargetCompares);
                    Assert.Equal(uploads, compiled.WrittenTargetUploads);
                    Assert.Equal(4f, plan.Position.ToArray()[0]);
                    // One host read of the position: the next replay compares that one target and uploads nothing.
                    plan.Reset(gpu);
                    long resetUploads = compiled.WrittenTargetUploads;
                    Step(4);
                    Assert.True(compiled.WrittenTargetUploads > resetUploads, "a reset was not uploaded");
                    compares = compiled.WrittenTargetCompares; uploads = compiled.WrittenTargetUploads;
                    for (int t = 5; t < 7; t++) Step(t);
                    Assert.Equal(compares, compiled.WrittenTargetCompares);
                    Assert.Equal(uploads, compiled.WrittenTargetUploads);
                    Assert.Equal(3f, plan.Position.ToArray()[0]);
                    Step(7);
                    Assert.Equal(compares + 1, compiled.WrittenTargetCompares);
                    Assert.Equal(uploads, compiled.WrittenTargetUploads);
                    Assert.True(exec.GetValue(compiled) is IntPtr h2 && h2 != IntPtr.Zero, "the plan stopped replaying");
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    private static void RunGpu(bool capture, int sequences)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                Run(gpu!, capture && IsCuda(gpu), sequences);
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }
}
