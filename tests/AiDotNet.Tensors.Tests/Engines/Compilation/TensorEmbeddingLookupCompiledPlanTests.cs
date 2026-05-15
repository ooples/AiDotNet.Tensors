using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression tests for AiDotNet#1328 — TensorEmbeddingLookup was not
/// recording itself to GraphMode (the lazy-graph tracer used by
/// CompiledTrainingPlan), only to the eager GradientTape. As a result,
/// a Transformer trained via the fused-compiled path silently dropped
/// dL/dE — the embedding weights stayed at initialisation forever and
/// the network produced input-independent output (PPL = V exactly,
/// top-1 = 1/V).
///
/// <para>The fix wires the same lazy-recording branch every other
/// differentiable op uses (cf. Reshape / MatMul / Softmax / LayerNorm
/// in <c>CpuEngine.cs</c>). These tests pin that wiring so a future
/// op-table edit can't silently re-introduce the regression.</para>
/// </summary>
public class TensorEmbeddingLookupCompiledPlanTests
{
    /// <summary>
    /// Forward-side sanity: under <see cref="GraphMode"/>, the embedding
    /// lookup records a lazy node — the returned tensor has a non-null
    /// <see cref="Tensor{T}.LazySource"/>. Before the #1328 fix the op
    /// materialised eagerly and the returned tensor had no lazy source,
    /// so the compiled-plan tracer never saw it.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookup_UnderGraphMode_RecordsLazyNode()
    {
        var engine = new CpuEngine();
        const int vocab = 4, dim = 8, batch = 3;
        var E = new Tensor<float>([vocab, dim]);
        for (int v = 0; v < vocab; v++)
            for (int d = 0; d < dim; d++)
                E[v, d] = (v + 1) * 0.1f + d * 0.001f;

        var indices = new Tensor<int>([batch]);
        indices[0] = 0; indices[1] = 2; indices[2] = 1;

        using var scope = GraphMode.Enable();
        var output = engine.TensorEmbeddingLookup<float, int>(E, indices);

        Assert.NotNull(output.LazySource);
        Assert.Equal(new[] { batch, dim }, output._shape);
    }

    /// <summary>
    /// End-to-end: a compiled training plan whose forward includes an
    /// embedding lookup actually computes a non-zero gradient on the
    /// embedding table when <c>plan.Step()</c> runs. Before the #1328
    /// fix the gradient buffer for E stayed at its initialisation
    /// value (zero) regardless of input, because the lazy graph had no
    /// node for the embedding op and the compiler produced no backward
    /// step for it.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookup_InCompiledPlan_ProducesNonZeroEmbeddingGradient()
    {
        var engine = new CpuEngine();
        const int vocab = 6, dim = 4, batch = 3;
        var E = new Tensor<float>([vocab, dim]);
        var eSpan = E.AsWritableSpan();
        for (int i = 0; i < eSpan.Length; i++) eSpan[i] = 0.01f * (i + 1);

        var indices = new Tensor<int>([batch]);
        indices[0] = 0; indices[1] = 2; indices[2] = 4;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorEmbeddingLookup<float, int>(E, indices);
            // Loss = sum(output). The plan seeds dL/dOutput as a ones-tensor
            // matching output.Shape, so the scatter-add backward puts a 1
            // in every selected embedding row's coords and 0 elsewhere.
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { E });
        }

        using (plan)
        {
            plan.Step();

            // Gradients are indexed parallel to the parameters array we
            // passed to CompileTraining — slot 0 is E.
            var gradE = plan.Gradients[0];
            Assert.NotNull(gradE);
            Assert.Equal(E._shape, gradE._shape);

            var gSpan = gradE.AsSpan();
            // Rows 0, 2, 4 selected → grad rows should be all-1.
            // Rows 1, 3, 5 un-selected → grad rows should stay all-0.
            var selected = new System.Collections.Generic.HashSet<int> { 0, 2, 4 };
            for (int row = 0; row < vocab; row++)
            {
                float expected = selected.Contains(row) ? 1f : 0f;
                for (int col = 0; col < dim; col++)
                {
                    float actual = gSpan[row * dim + col];
                    Assert.Equal(expected, actual, 5);
                }
            }
        }
    }

    /// <summary>
    /// Per-row independence: distinct index batches in the same plan
    /// produce DIFFERENT gradient patterns on E. Pre-fix the gradient
    /// was always all-zero, so this assertion would trivially pass on
    /// the broken path (false negative). Pinning a SPECIFIC selection
    /// shape catches both classes of regression: dropped-gradient AND
    /// always-same-gradient kernels.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookup_InCompiledPlan_DistinctIndexBatches_ProduceDistinctGradients()
    {
        var engine = new CpuEngine();
        const int vocab = 5, dim = 2;

        // Plan #1: pick rows {1, 3}
        var E1 = new Tensor<float>([vocab, dim]);
        var idx1 = new Tensor<int>([2]); idx1[0] = 1; idx1[1] = 3;
        ICompiledTrainingPlan<float> plan1;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorEmbeddingLookup<float, int>(E1, idx1);
            engine.ReduceSum(output, null);
            plan1 = scope.CompileTraining(new[] { E1 });
        }
        plan1.Step();
        var g1 = plan1.Gradients[0].AsSpan().ToArray();
        plan1.Dispose();

        // Plan #2: pick rows {0, 4}
        var E2 = new Tensor<float>([vocab, dim]);
        var idx2 = new Tensor<int>([2]); idx2[0] = 0; idx2[1] = 4;
        ICompiledTrainingPlan<float> plan2;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorEmbeddingLookup<float, int>(E2, idx2);
            engine.ReduceSum(output, null);
            plan2 = scope.CompileTraining(new[] { E2 });
        }
        plan2.Step();
        var g2 = plan2.Gradients[0].AsSpan().ToArray();
        plan2.Dispose();

        // Plan #1 selected rows {1, 3} so g1[row 1] == 1, g1[row 3] == 1, rest 0.
        // Plan #2 selected rows {0, 4} so g2[row 0] == 1, g2[row 4] == 1, rest 0.
        // The two patterns are necessarily distinct.
        Assert.Equal(1f, g1[1 * dim], 5);
        Assert.Equal(1f, g1[3 * dim], 5);
        Assert.Equal(0f, g1[0 * dim], 5);

        Assert.Equal(1f, g2[0 * dim], 5);
        Assert.Equal(1f, g2[4 * dim], 5);
        Assert.Equal(0f, g2[3 * dim], 5);
    }
}
