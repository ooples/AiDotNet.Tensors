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

    private static void Run(bool keep)
    {
        var engine = new CpuEngine();
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

            foreach (var input in new[] { new[] { 1f, 2f }, new[] { 3f, 4f } })
            {
                plan.SetInputs(new[] { new Tensor<float>([1, 2], new Vector<float>(input)) });
                var y = plan.Execute().ToArray();
                // x [1, 2] times W [2, 3], row-major.
                var product = Enumerable.Range(0, 3).Select(j => input[0] * Weights[j] + input[1] * Weights[3 + j]).ToArray();
                Assert.Equal(product.Select((p, j) => p + Bias[j]).ToArray(), y);
                Assert.Equal(product, kept.ToArray());
            }
        }
    }
}
