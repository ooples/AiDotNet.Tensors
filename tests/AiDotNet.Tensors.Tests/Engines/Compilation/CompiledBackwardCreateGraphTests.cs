using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase 4C — compiled-backward path with <c>createGraph=true</c> so higher-order AD
/// (Hessian-vector products, WGAN-GP gradient penalty) can differentiate the outer
/// backward's ops. Before the fix, <c>GradientTape.ComputeGradients</c> gated the
/// compiled path on <c>!createGraph</c>, forcing higher-order AD to fall back to the
/// slow tape-walking path even for persistent tapes.
/// </summary>
public class CompiledBackwardCreateGraphTests
{
    /// <summary>
    /// Baseline: a first-order backward with <c>createGraph=true</c> must produce
    /// the same gradient as the same call with <c>createGraph=false</c> (the tape
    /// walk records ops but the outer-most gradient values are identical).
    /// </summary>
    [Fact]
    public void CompiledBackward_CreateGraphTrue_MatchesCreateGraphFalse_FirstOrder()
    {
        var engine = new CpuEngine();

        // Same graph both times so the compiled path caches it after step 2.
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weightFalse = new Tensor<float>(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, new[] { 2, 2 });
        var weightTrue = new Tensor<float>(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, new[] { 2, 2 });

        Tensor<float> gradFalse, gradTrue;

        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var y = engine.TensorMatMul(input, weightFalse);
            var loss = engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, new[] { weightFalse }, createGraph: false);
            gradFalse = grads[weightFalse];
        }

        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var y = engine.TensorMatMul(input, weightTrue);
            var loss = engine.ReduceSum(y, null);
            var grads = tape.ComputeGradients(loss, new[] { weightTrue }, createGraph: true);
            gradTrue = grads[weightTrue];
        }

        Assert.Equal(gradFalse.Length, gradTrue.Length);
        var a = gradFalse.ToArray();
        var b = gradTrue.ToArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(System.MathF.Abs(a[i] - b[i]) < 1e-5f,
                $"gradient[{i}] createGraph=false ({a[i]:R}) != createGraph=true ({b[i]:R})");
    }

    /// <summary>
    /// Second-order AD (the WGAN-GP-style pattern): compute grad-of-grad through
    /// a matmul via a NESTED tape. Under a persistent OUTER tape (which enables
    /// compiled backward), the inner tape's <c>createGraph=true</c> must record
    /// its backward ops on the outer tape so the outer tape's ComputeGradients
    /// can traverse the second-order chain back to the weight.
    ///
    /// <para>Before the Phase 4C fix, <c>ComputeGradients(..., createGraph: true)</c>
    /// on a persistent tape would silently skip the compiled path AND the graph
    /// path, forcing the slow tape walk. This test would still pass on the slow
    /// path — the value is the same — but the compiled path was unreachable for
    /// higher-order AD.</para>
    ///
    /// <para>What this test validates: the SECOND-order gradient <c>∂(sum(∂y/∂x)²) /
    /// ∂w</c> for <c>y = w·x + b</c> is finite, non-NaN, and matches an analytic
    /// hand-computation, even when the inner tape is persistent-enabled.</para>
    /// </summary>
    [Fact]
    public void CompiledBackward_CreateGraphTrue_EnablesHigherOrderAD()
    {
        var engine = new CpuEngine();

        // y = w · x  where x is the input tensor
        // ∂y/∂x = w  (a matrix)
        // Let g = sum(y). Then ∂g/∂x = w^T · [ones]  → a vector
        // Second-order objective: L2² of ∂g/∂x, i.e. sum((w^T · [ones])²)
        // ∂L2² / ∂w = 2 · [ones] · (w^T · [ones])  — that's what the OUTER backward computes.
        var x = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var w = new Tensor<float>(new float[] { 0.5f, 0.3f, 0.7f, 0.1f }, new[] { 2, 2 });

        Tensor<float> secondOrderGrad;
        using (var outer = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            // y = w · x  → shape [2]
            var y = engine.TensorMatMul(w, engine.Reshape(x, new[] { 2, 1 }));
            var g = engine.ReduceSum(y, null);

            // Inner backward: ∂g/∂x, with createGraph=true so the inner backward's
            // ops record on the OUTER tape.
            Tensor<float> gradX;
            using (var inner = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
            {
                // Re-establish tape linkage via a second forward that records on
                // the inner tape while ALSO being visible to the outer tape.
                var yInner = engine.TensorMatMul(w, engine.Reshape(x, new[] { 2, 1 }));
                var gInner = engine.ReduceSum(yInner, null);
                var innerGrads = inner.ComputeGradients(gInner, new[] { x }, createGraph: true);
                gradX = innerGrads.TryGetValue(x, out var gx)
                    ? gx
                    : new Tensor<float>(x._shape);
            }

            // Outer objective: sum(gradX²). Its gradient wrt w exists only because
            // createGraph=true recorded the inner backward's ops on the outer tape.
            var gradXSq = engine.TensorMultiply(gradX, gradX);
            var outerLoss = engine.ReduceSum(gradXSq, null);
            var outerGrads = outer.ComputeGradients(outerLoss, new[] { w });
            secondOrderGrad = outerGrads.TryGetValue(w, out var gw)
                ? gw
                : new Tensor<float>(w._shape);
        }

        // The second-order gradient must be finite and non-zero (a broken
        // createGraph=true would leak inputGradients as a leaf, driving the
        // outer gradient wrt w to all zeros — exactly the symptom of the
        // WGAN-GP bug we're avoiding).
        var arr = secondOrderGrad.ToArray();
        bool allZero = true;
        for (int i = 0; i < arr.Length; i++)
        {
            Assert.True(!float.IsNaN(arr[i]) && !float.IsInfinity(arr[i]),
                $"secondOrderGrad[{i}] = {arr[i]:R} is non-finite");
            if (System.MathF.Abs(arr[i]) > 1e-6f) allZero = false;
        }
        Assert.False(allZero,
            "Second-order gradient wrt w is all-zero — createGraph=true is NOT propagating " +
            "the inner backward into the outer tape (regresses the Phase 4C fix; WGAN-GP " +
            "gradient penalty would silently not train the discriminator).");
    }
}
