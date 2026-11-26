using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// After training, every parameter of a model is a view into one flat <see cref="ParameterBuffer{T}"/> storage: the
/// first is a prefix view (offset 0, shorter than the storage), the rest are contiguous views at a nonzero offset. The
/// GPU engine's destination-aware binary ops (TensorAddInto and friends) resolved their inputs through a helper that
/// threw "GPU destination-aware tensor ops require a contiguous, dense, zero-offset canonical allocation" for any such
/// view, so a compiled stateful inference trace over a trained model failed to compile.
/// Each case is compared with the CPU engine. Skips without a GPU backend.
/// </summary>
[Collection("DirectGpuSerial")]
public class ViewInputOperandGpuTests
{
    private const int D = 16;

    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static DirectGpuTensorEngine? TryGpu()
    {
        try { return new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { return null; }
    }

    private static float[] Values(int seed, int n)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    // Three [1, D] parameters packed into one flat storage: view 0 is a prefix view, views 1 and 2 sit at offsets D, 2D.
    private static Tensor<float>[] PackedParameters()
    {
        var buffer = new ParameterBuffer<float>(new[] { new[] { 1, D }, new[] { 1, D }, new[] { 1, D } });
        var views = new[] { buffer.CreateView(0), buffer.CreateView(1), buffer.CreateView(2) };
        for (int p = 0; p < views.Length; p++)
        {
            var v = Values(10 + p, D);
            for (int i = 0; i < D; i++) views[p][0, i] = v[i] + (p == 0 ? 1.5f : 0f);
        }
        return views;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * (1 + Math.Abs(expected[i])),
                $"{what}[{i}] gpu {actual[i]} vs cpu {expected[i]}");
    }

    [SkippableTheory]
    [InlineData(1, true)]    // (a) contiguous view at storage offset D, as the left operand
    [InlineData(2, false)]   // (a) offset 2D, as the right operand
    [InlineData(0, true)]    // (b) prefix view (offset 0, shorter than its storage), left
    [InlineData(0, false)]   // (b) prefix view, right
    public void TensorAddInto_AcceptsAContiguousViewOperand(int viewIndex, bool viewOnLeft)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend available");
        using (gpu)
        {
            var views = PackedParameters();
            var view = views[viewIndex];
            Assert.True(view.IsView);
            var before = Array.ConvertAll(views, v => v.ToArray());
            var other = new Tensor<float>(new[] { 1, D }, new Vector<float>(Values(99, D)));
            var (l, r) = viewOnLeft ? (view, other) : (other, view);

            var cpuDest = new Tensor<float>(new[] { 1, D });
            ((IEngine)new CpuEngine()).TensorAddInto(cpuDest, l, r);
            var gpuDest = new Tensor<float>(new[] { 1, D });
            ((IEngine)gpu!).TensorAddInto(gpuDest, l, r);
            AssertClose(cpuDest.ToArray(), gpuDest.ToArray(), "add-into");
            // The source storage is not disturbed: the other views still read their own elements.
            var again = new Tensor<float>(new[] { 1, D });
            ((IEngine)gpu).TensorAddInto(again, l, r);
            AssertClose(cpuDest.ToArray(), again.ToArray(), "add-into repeat");
            // The op only read the view: no parameter in the shared storage changed.
            for (int p = 0; p < views.Length; p++) AssertClose(before[p], views[p].ToArray(), $"parameter {p}");
        }
    }

    // The destination is itself a view into the same packed storage as the input view: the result lands in the
    // destination's elements only, and the input view and the third parameter keep their values.
    [SkippableFact]
    public void TensorAddInto_WritesOnlyTheDestinationView_WhenItSharesTheInputViewsStorage()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend available");
        using (gpu)
        {
            var views = PackedParameters();
            var other = new Tensor<float>(new[] { 1, D }, new Vector<float>(Values(98, D)));
            var cpuViews = PackedParameters();
            ((IEngine)new CpuEngine()).TensorAddInto(cpuViews[2], cpuViews[1], other);
            ((IEngine)gpu!).TensorAddInto(views[2], views[1], other);
            for (int p = 0; p < views.Length; p++) AssertClose(cpuViews[p].ToArray(), views[p].ToArray(), $"parameter {p}");
        }
    }

    // TensorAddInto is an explicit IEngine implementation on the GPU engine: call it through IEngine.
    // A genuinely strided (transposed) input operand is handled by the same materialization path.
    [SkippableFact]
    public void TensorAddInto_AcceptsATransposedViewOperand()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend available");
        using (gpu)
        {
            var cpu = new CpuEngine();
            var src = new Tensor<float>(new[] { 4, 6 }, new Vector<float>(Values(5, 24)));
            var t = src.Transpose(new[] { 1, 0 });
            Assert.False(t.IsContiguous);
            var other = new Tensor<float>(new[] { 6, 4 }, new Vector<float>(Values(6, 24)));
            var cpuDest = new Tensor<float>(new[] { 6, 4 });
            ((IEngine)cpu).TensorAddInto(cpuDest, t, other);
            var gpuDest = new Tensor<float>(new[] { 6, 4 });
            ((IEngine)gpu!).TensorAddInto(gpuDest, t, other);
            AssertClose(cpuDest.ToArray(), gpuDest.ToArray(), "transposed add-into");
        }
    }

    // (c) The compiled stateful inference path: a step whose constants are packed-parameter views (an offset view added
    // to the input, then RMSNorm with the prefix view as gamma), the same shape as a trained model's decode step.
    private static (Tensor<float> Output, Tensor<float> Next) Step(IEngine e, Tensor<float> s, Tensor<float> x, Tensor<float>[] w)
    {
        var h = e.TensorAdd(x, w[1]);
        var n = e.RMSNorm(e.TensorAdd(h, s), w[0], 1e-6, out _);
        var next = e.TensorAdd(e.TensorMultiplyScalar(s, 0.5f), e.TensorMultiply(n, w[2]));
        return (e.TensorAdd(n, w[2]), next);
    }

    private const int Steps = 6;

    private static void RunPlan(IEngine engine, bool capture, float[][] refOut, float[][] refState)
    {
        var w = PackedParameters();
        var s = new Tensor<float>(new[] { 1, D });
        var x = new Tensor<float>(new[] { 1, D }, new Vector<float>(Values(200, D)));
        CompiledInferencePlan<float> plan;
        var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
        CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
        try
        {
            using (new EngineScope(engine))
            using (var scope = GraphMode.EnableInference())
            {
                var (o, next) = Step(engine, s, x, w);
                plan = scope.CompileInference(o, new[] { x }, new[] { (s, next) });
            }
            using (plan)
            {
                for (int t = 0; t < Steps; t++)
                {
                    plan.SetInputs(new[] { new Tensor<float>(new[] { 1, D }, new Vector<float>(Values(200 + t, D))) });
                    var output = plan.Execute().ToArray();
                    AssertClose(refOut[t], output, $"step {t} output");
                    AssertClose(refState[t], s.ToArray(), $"step {t} state");
                }
                if (capture)
                {
                    var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                        ?? throw new InvalidOperationException("_graphExec field not found.");
                    Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                }
            }
        }
        finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledStatefulInference_OverPackedParameterViews_MatchesCpu(bool capture)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "No DirectGpu backend available");
        using (gpu)
        {
            if (capture)
                Skip.IfNot(gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
            var cpu = new CpuEngine();
            var w = PackedParameters();
            var s = new Tensor<float>(new[] { 1, D });
            var refOut = new float[Steps][];
            var refState = new float[Steps][];
            for (int t = 0; t < Steps; t++)
            {
                var (o, next) = Step(cpu, s, new Tensor<float>(new[] { 1, D }, new Vector<float>(Values(200 + t, D))), w);
                refOut[t] = o.ToArray();
                s = next;
                refState[t] = s.ToArray();
            }
            RunPlan(gpu!, capture, refOut, refState);
        }
    }
}
