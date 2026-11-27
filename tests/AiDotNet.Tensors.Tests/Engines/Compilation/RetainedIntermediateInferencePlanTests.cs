using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A forward intermediate kept past the call (<see cref="TensorScope.Detach{T}"/>, <see cref="TensorScope.Keep{T}"/>)
/// is read after a compiled inference plan runs, so every replay must write it. MatMul followed by a bias add is the
/// chain <c>LinearFusionPattern</c> fuses into one node that writes only the sum: unless the plan keeps the retained
/// MatMul observable, the retained tensor would keep the first execution's values.
/// </summary>
[Collection("DirectGpuSerial")]
public class RetainedIntermediateInferencePlanTests
{
    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static readonly float[] Weights = { 1f, 2f, 3f, 4f, 5f, 6f };
    private static readonly float[] Bias = { 0.5f, -1f, 2f };

    [Fact]
    public void A_detached_intermediate_is_rewritten_on_every_replay() => Run(keep: false);

    [Fact]
    public void A_kept_intermediate_is_rewritten_on_every_replay() => Run(keep: true);

    // With inference graph capture on, the plan warms up, captures and then replays: a replay runs no binding delegate,
    // so the retained tensor's host download must be re-armed after each launch or a read shows the previous call's
    // values. Five calls cover the warmup, the capture call and replays.
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_retained_intermediate_is_rewritten_on_every_captured_replay(bool keep)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { gpu = null; }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        if (gpu is null) return;
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = true;
            try
            {
                Run(gpu, keep, Enumerable.Range(0, 5).Select(i => new[] { 1f + i, 2f - 0.5f * i }).ToArray());
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }

    private static void Run(bool keep) => Run(new CpuEngine(), keep, new[] { new[] { 1f, 2f }, new[] { 3f, 4f } });

    private static void Run(CpuEngine engine, bool keep, float[][] inputs)
    {
        var w = new Tensor<float>([2, 3], new Vector<float>(Weights));
        var bias = new Tensor<float>([3], new Vector<float>(Bias));
        Tensor<float>? saved = null;
        using (new EngineScope(engine))
        using (var cache = new CompiledModelCache<float>())
        using (var outer = TensorScope.Begin())
        {
            var plan = cache.GetOrCompileInference(new Tensor<float>([1, 2]), xi =>
            {
                var product = engine.TensorMatMul(xi, w);
                var retained = keep ? outer.Keep(product) : TensorScope.Detach(product);
                saved = retained;
                return engine.TensorBroadcastAdd(retained, bias);
            });
            Tensor<float> kept = saved ?? throw new InvalidOperationException("The forward did not run.");

            foreach (var input in inputs)
            {
                plan.SetInputs(new[] { new Tensor<float>([1, 2], new Vector<float>(input)) });
                var y = plan.Execute().ToArray();
                // x [1, 2] times W [2, 3], row-major.
                var product = Enumerable.Range(0, 3).Select(j => input[0] * Weights[j] + input[1] * Weights[3 + j]).ToArray();
                var sum = product.Select((p, j) => p + Bias[j]).ToArray();
                if (engine is not DirectGpuTensorEngine)
                {
                    Assert.Equal(sum, y);
                    Assert.Equal(product, kept.ToArray());
                    continue;
                }
                // The inputs are small integers and halves, exact in fp32 through any GEMM order; the tolerance only
                // absorbs a backend that fuses the bias add with a different rounding.
                var read = kept.ToArray();
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(sum[j], y[j], 4);
                    Assert.Equal(product[j], read[j], 4);
                }
            }
        }
    }
}
