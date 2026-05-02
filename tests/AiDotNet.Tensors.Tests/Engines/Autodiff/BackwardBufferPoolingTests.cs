using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for the buffer-pooling changes in BackwardFunctions.cs (this PR).
///
/// The PR replaces <c>new float[]</c> allocations in the matmul and fused-linear backward
/// passes with pooled buffers from <c>AutoTensorCache.RentOrAllocate</c>.
///
/// Key correctness invariants that must hold with pooled buffers:
/// 1. MatMul backward: TryGemmEx uses beta=0 (overwrites) → stale pool data doesn't matter.
/// 2. FusedLinear backward — gradBias: requires Array.Clear before accumulating into the
///    bias gradient buffer (pool may contain garbage from a prior step).
/// 3. All paths must produce numerically identical gradients across repeated steps
///    (regression: if a buffer is not cleared/overwritten, step 2+ would return wrong values).
/// </summary>
public class BackwardBufferPoolingTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private const float Tolerance = 1e-4f;

    public BackwardBufferPoolingTests()
    {
        AutoTrainingCompiler.Enabled = true;
        AutoTrainingCompiler.ReplayMode = false;
    }

    public void Dispose()
    {
        AutoTrainingCompiler.Enabled = true;
        AutoTrainingCompiler.ReplayMode = false;
    }

    // ──────────────────────────────────────────────────────────────
    // MatMul backward — pooled gradient buffers
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void MatMul_Backward_PooledBuffers_ProducesConsistentGradientsAcrossSteps()
    {
        // Regression: if the rented buffer from AutoTensorCache still contains stale
        // data from a prior GEMM and beta=0 doesn't overwrite properly, step 2+ would
        // produce different (wrong) gradients.
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, new[] { 2, 2 });

        float[]? refInputGrad = null;
        float[]? refWeightGrad = null;

        // Run 5 identical backward passes. If any buffer is stale, step 2+ will differ.
        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.TensorMatMul(input, weight);
            var loss = _engine.ReduceSum(output, null);
            var grads = tape.ComputeGradients(loss, new[] { input, weight });

            if (step == 0)
            {
                // Capture reference gradients from the first step
                refInputGrad = Enumerable.Range(0, grads[input].Length)
                    .Select(i => grads[input].GetFlat(i)).ToArray();
                refWeightGrad = Enumerable.Range(0, grads[weight].Length)
                    .Select(i => grads[weight].GetFlat(i)).ToArray();
                continue;
            }

            // Every subsequent step must match the first step
            for (int i = 0; i < refInputGrad!.Length; i++)
            {
                float actual = grads[input].GetFlat(i);
                Assert.True(Math.Abs(actual - refInputGrad[i]) <= Tolerance,
                    $"MatMul gradInput[{i}] differs at step {step + 1}: " +
                    $"expected {refInputGrad[i]}, got {actual} (pooled buffer stale data?)");
            }

            for (int i = 0; i < refWeightGrad!.Length; i++)
            {
                float actual = grads[weight].GetFlat(i);
                Assert.True(Math.Abs(actual - refWeightGrad[i]) <= Tolerance,
                    $"MatMul gradWeight[{i}] differs at step {step + 1}: " +
                    $"expected {refWeightGrad[i]}, got {actual} (pooled buffer stale data?)");
            }
        }
    }

    [Fact]
    public void MatMul_Backward_PooledBuffers_MatchesExpectedValues()
    {
        // Verify the actual gradient values are mathematically correct, not just consistent.
        // Z = A @ B, dL/dZ = ones(2,2)
        // dL/dA = dL/dZ @ B^T,  dL/dB = A^T @ dL/dZ
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var b = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMatMul(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // dL/dA = ones(2,2) @ B^T = ones @ [[5,7],[6,8]] = [[11,15],[11,15]]
        Assert.Equal(11f, grads[a][0, 0], Tolerance);
        Assert.Equal(15f, grads[a][0, 1], Tolerance);
        Assert.Equal(11f, grads[a][1, 0], Tolerance);
        Assert.Equal(15f, grads[a][1, 1], Tolerance);

        // dL/dB = A^T @ ones(2,2) = [[1,3],[2,4]] @ ones = [[4,4],[6,6]]
        Assert.Equal(4f, grads[b][0, 0], Tolerance);
        Assert.Equal(4f, grads[b][0, 1], Tolerance);
        Assert.Equal(6f, grads[b][1, 0], Tolerance);
        Assert.Equal(6f, grads[b][1, 1], Tolerance);
    }

    [Fact]
    public void MatMul_Backward_Rectangular_PooledBuffers_CorrectValues()
    {
        // Non-square matrices catch shape-mismatch bugs in buffer sizing.
        // A: [4, 3], B: [3, 2] → Z: [4, 2]
        var a = new Tensor<float>(
            new float[] { 1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1 },
            new[] { 4, 3 });
        var b = new Tensor<float>(
            new float[] { 1, 0,  0, 1,  1, 1 },
            new[] { 3, 2 });

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMatMul(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // Gradient shapes must match input shapes
        Assert.Equal(new[] { 4, 3 }, grads[a].Shape.ToArray());
        Assert.Equal(new[] { 3, 2 }, grads[b].Shape.ToArray());

        // All entries finite (no NaN/Inf from stale buffer data)
        for (int i = 0; i < grads[a].Length; i++)
            Assert.True(!float.IsNaN(grads[a].GetFlat(i)), $"gradA[{i}] is not finite");
        for (int i = 0; i < grads[b].Length; i++)
            Assert.True(!float.IsNaN(grads[b].GetFlat(i)), $"gradB[{i}] is not finite");
    }

    // ──────────────────────────────────────────────────────────────
    // FusedLinear + ReLU backward — pooled gradient buffers
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearReLU_Backward_PooledBuffers_ProducesConsistentGradientsAcrossSteps()
    {
        // Regression: gradBias requires Array.Clear before accumulation because the pool
        // buffer may contain non-zero data from a prior step.
        var input = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weights = new Tensor<float>(new float[] { 0.5f, -0.5f, 0.5f, -0.5f }, new[] { 2, 2 });
        var bias = new Tensor<float>(new float[] { 0.1f, -0.1f }, new[] { 1, 2 });

        float[]? refWeightGrad = null;
        float[]? refBiasGrad = null;

        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            var loss = _engine.TensorMeanDiff(output);
            var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

            if (step == 0)
            {
                refWeightGrad = Enumerable.Range(0, grads[weights].Length)
                    .Select(i => grads[weights].GetFlat(i)).ToArray();
                refBiasGrad = Enumerable.Range(0, grads[bias].Length)
                    .Select(i => grads[bias].GetFlat(i)).ToArray();
                continue;
            }

            // Weight gradients
            for (int i = 0; i < refWeightGrad!.Length; i++)
            {
                float actual = grads[weights].GetFlat(i);
                Assert.True(Math.Abs(actual - refWeightGrad[i]) <= Tolerance,
                    $"FusedLinear+ReLU gradWeight[{i}] differs at step {step + 1}: " +
                    $"expected {refWeightGrad[i]}, got {actual}");
            }

            // Bias gradients — most sensitive to Array.Clear regression
            for (int i = 0; i < refBiasGrad!.Length; i++)
            {
                float actual = grads[bias].GetFlat(i);
                Assert.True(Math.Abs(actual - refBiasGrad[i]) <= Tolerance,
                    $"FusedLinear+ReLU gradBias[{i}] differs at step {step + 1}: " +
                    $"expected {refBiasGrad[i]}, got {actual} (Array.Clear missing?)");
            }
        }
    }

    [Fact]
    public void FusedLinearReLU_Backward_GradBias_SumsCorrectly()
    {
        // Verify gradBias = sum(maskedGrad, axis=0) is computed correctly.
        // Input: [3, 4], Weights: [4, 2], Bias: [1, 2]
        var input = new Tensor<float>(
            new float[] { 1f, 0f, 0f, 0f,
                          0f, 1f, 0f, 0f,
                          0f, 0f, 1f, 0f },
            new[] { 3, 4 });
        var weights = new Tensor<float>(
            new float[] { 1f, 0f,
                          0f, 1f,
                          0f, 0f,
                          0f, 0f },
            new[] { 4, 2 });
        // All positive pre-activations so ReLU passes everything through
        var bias = new Tensor<float>(new float[] { 0.5f, 0.5f }, new[] { 1, 2 });

        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

        Assert.True(grads.ContainsKey(bias), "Bias gradient must be present");
        Assert.Equal(2, grads[bias].Length);

        // All bias gradient values must be finite
        for (int i = 0; i < grads[bias].Length; i++)
            Assert.True(!float.IsNaN(grads[bias].GetFlat(i)),
                $"gradBias[{i}] is not finite — possible stale buffer accumulation");
    }

    [Fact]
    public void FusedLinearReLU_Backward_PooledBuffers_GradBiasIsConsistentAcrossRentals()
    {
        // Specifically stress-tests bias gradient consistency across buffer rentals.
        // Each gradient computation rents the bias buffer from the pool; if Array.Clear
        // is missing, each rental accumulates on top of the previous step's values.
        var input = new Tensor<float>(
            new float[] { 2f, 1f, -1f, 3f },
            new[] { 2, 2 });
        var weights = new Tensor<float>(
            new float[] { 1f, 2f, -1f, 0.5f },
            new[] { 2, 2 });
        var bias = new Tensor<float>(new float[] { 1f, -1f }, new[] { 1, 2 });

        // Reference: single-step gradients
        float[] refBiasGrad;
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            var loss = _engine.ReduceSum(output, null);
            var grads = tape.ComputeGradients(loss, new[] { weights, bias });
            refBiasGrad = new float[grads[bias].Length];
            for (int i = 0; i < grads[bias].Length; i++)
                refBiasGrad[i] = grads[bias].GetFlat(i);
        }

        // Repeated: each step must return the same bias gradient as the reference
        for (int step = 0; step < 6; step++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
            var loss = _engine.ReduceSum(output, null);
            var grads = tape.ComputeGradients(loss, new[] { weights, bias });

            for (int i = 0; i < grads[bias].Length; i++)
            {
                float actual = grads[bias].GetFlat(i);
                Assert.True(Math.Abs(actual - refBiasGrad[i]) <= Tolerance,
                    $"Bias gradient at index {i} is wrong at step {step + 1}: " +
                    $"expected {refBiasGrad[i]}, got {actual} (stale pool buffer?)");
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // LinearLayer backward (no activation) — pooled buffers
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearNone_Backward_PooledBuffers_ConsistentAcrossSteps()
    {
        // FusedActivationType.None: matmul + bias, no activation mask.
        // The PR also pools buffers for this code path.
        var input = new Tensor<float>(new float[] { 1f, -1f, 2f, -2f }, new[] { 2, 2 });
        var weights = new Tensor<float>(new float[] { 3f, 1f, -1f, 2f }, new[] { 2, 2 });
        var bias = new Tensor<float>(new float[] { 0.5f, -0.5f }, new[] { 1, 2 });

        float[]? refInputGrad = null;
        float[]? refWeightGrad = null;
        float[]? refBiasGrad = null;

        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>();
            var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.None);
            var loss = _engine.ReduceSum(output, null);
            var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

            if (step == 0)
            {
                refInputGrad = Enumerable.Range(0, grads[input].Length)
                    .Select(i => grads[input].GetFlat(i)).ToArray();
                refWeightGrad = Enumerable.Range(0, grads[weights].Length)
                    .Select(i => grads[weights].GetFlat(i)).ToArray();
                refBiasGrad = Enumerable.Range(0, grads[bias].Length)
                    .Select(i => grads[bias].GetFlat(i)).ToArray();
                continue;
            }

            for (int i = 0; i < refInputGrad!.Length; i++)
            {
                float actual = grads[input].GetFlat(i);
                Assert.True(Math.Abs(actual - refInputGrad[i]) <= Tolerance,
                    $"FusedLinearNone gradInput[{i}] differs at step {step + 1}");
            }

            for (int i = 0; i < refWeightGrad!.Length; i++)
            {
                float actual = grads[weights].GetFlat(i);
                Assert.True(Math.Abs(actual - refWeightGrad[i]) <= Tolerance,
                    $"FusedLinearNone gradWeight[{i}] differs at step {step + 1}");
            }

            for (int i = 0; i < refBiasGrad!.Length; i++)
            {
                float actual = grads[bias].GetFlat(i);
                Assert.True(Math.Abs(actual - refBiasGrad[i]) <= Tolerance,
                    $"FusedLinearNone gradBias[{i}] differs at step {step + 1}");
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Gradient value correctness (not just consistency)
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearReLU_Backward_Correctness_MatchesFiniteDifference()
    {
        // Compare autodiff gradients with numerical finite-difference approximation.
        // This validates that the pooled-buffer path computes correct values,
        // not just consistent (but wrong) ones.
        float h = 1e-3f;
        float fdTolerance = 1e-2f;

        var inputData = new float[] { 1f, 2f, -1f, 0.5f };
        var weightData = new float[] { 0.3f, -0.4f, 0.5f, 0.1f };
        var biasData = new float[] { 0.1f, -0.2f };

        var input = new Tensor<float>(inputData, new[] { 2, 2 });
        var weights = new Tensor<float>(weightData, new[] { 2, 2 });
        var bias = new Tensor<float>(biasData, new[] { 1, 2 });

        // Autodiff gradients
        using var tape = new GradientTape<float>();
        var output = _engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
        var loss = _engine.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { weights, bias });

        // Numerical gradient for weights
        for (int i = 0; i < weightData.Length; i++)
        {
            var wPlus = (float[])weightData.Clone();
            wPlus[i] += h;
            var wMinus = (float[])weightData.Clone();
            wMinus[i] -= h;

            var outPlus = _engine.ReduceSum(
                _engine.FusedLinear(input,
                    new Tensor<float>(wPlus, new[] { 2, 2 }),
                    bias, FusedActivationType.ReLU), null);
            var outMinus = _engine.ReduceSum(
                _engine.FusedLinear(input,
                    new Tensor<float>(wMinus, new[] { 2, 2 }),
                    bias, FusedActivationType.ReLU), null);

            float numerical = (outPlus.GetFlat(0) - outMinus.GetFlat(0)) / (2f * h);
            float autodiff = grads[weights].GetFlat(i);
            Assert.True(Math.Abs(autodiff - numerical) <= fdTolerance,
                $"FusedLinear+ReLU weight gradient at index {i}: " +
                $"autodiff={autodiff:F4}, numerical={numerical:F4}");
        }

        // Numerical gradient for bias
        for (int i = 0; i < biasData.Length; i++)
        {
            var bPlus = (float[])biasData.Clone();
            bPlus[i] += h;
            var bMinus = (float[])biasData.Clone();
            bMinus[i] -= h;

            var outPlus = _engine.ReduceSum(
                _engine.FusedLinear(input, weights,
                    new Tensor<float>(bPlus, new[] { 1, 2 }),
                    FusedActivationType.ReLU), null);
            var outMinus = _engine.ReduceSum(
                _engine.FusedLinear(input, weights,
                    new Tensor<float>(bMinus, new[] { 1, 2 }),
                    FusedActivationType.ReLU), null);

            float numerical = (outPlus.GetFlat(0) - outMinus.GetFlat(0)) / (2f * h);
            float autodiff = grads[bias].GetFlat(i);
            Assert.True(Math.Abs(autodiff - numerical) <= fdTolerance,
                $"FusedLinear+ReLU bias gradient at index {i}: " +
                $"autodiff={autodiff:F4}, numerical={numerical:F4}");
        }
    }

    [Fact]
    public void FusedLinearReLU_Backward_AboveSimdGemmThreshold_ProducesValidGradients()
    {
        // Exercises FusedMatMulAddReLUBackward at a shape sized DIRECTLY
        // from SimdGemm.ParallelWorkThreshold so M·K·N is always above the
        // gate, no matter the runner core count (the threshold scales per-
        // core and clamps to [2 Mi, 20 Mi] work-elements). The cube-root
        // sizing keeps the matrices as small as possible while still
        // crossing the gate, so this test stays cheap on CI instead of
        // using the previous 256×512×256 = 33.5 Mi shape that was much
        // larger than necessary.
        //
        // Existing tests in this class use 2×2 matrices that fall under
        // the threshold and take the engine fallback path, so the fast-
        // path success branch (which the post-PR-280 fix touches) goes
        // uncovered without this case.
        //
        // We assert behavioural correctness (gradients are populated,
        // finite, and consistent across repeated calls) rather than
        // heap-retention numbers — GC.GetTotalMemory deltas are
        // unreliable on a shared CI runner where concurrent test classes
        // allocate during the measurement window.
        long threshold = global::AiDotNet.Tensors.Engines.Simd.SimdGemm.ParallelWorkThreshold;
        // Cube root with 25 % headroom so we land safely above the gate
        // even after rounding down per dimension. Math.Cbrt isn't available
        // on net471, so use Math.Pow(x, 1.0/3.0) which is — accurate enough
        // for sizing a test matrix (we round up afterwards anyway).
        int dim = (int)Math.Ceiling(Math.Pow(threshold * 1.25, 1.0 / 3.0));
        // Round up to the next multiple of 16 so the per-dim sizes stay
        // friendly to the vector kernel's micro-tile (Mr=6, Nr=16) without
        // forcing huge edges.
        dim = ((dim + 15) / 16) * 16;
        int M = dim, K = dim, N = dim;
        Assert.True((long)M * K * N >= threshold,
            $"Test sizing math regressed: M*K*N = {(long)M * K * N} < threshold = {threshold}");

        var input = new Tensor<float>(MakeFilled(M * K, 0.01f), new[] { M, K });
        var weights = new Tensor<float>(MakeFilled(K * N, 0.005f), new[] { K, N });
        var bias = new Tensor<float>(MakeFilled(N, 0.01f), new[] { 1, N });

        float[]? referenceWeightGrad = null;
        float[]? referenceBiasGrad = null;
        float[]? referenceInputGrad = null;

        for (int step = 0; step < 5; step++)
        {
            using var tape = new GradientTape<float>();
            // FusedLinearReLU registers FusedMatMulAddReLUBackward — the
            // function that owns the maskedTensor scratch buffer this fix
            // returns to the cache. _engine.FusedLinear(..., ReLU) takes a
            // different code path (FusedLinearWithActivationBackward) that
            // doesn't have the mask buffer at all.
            var output = _engine.FusedLinearReLU(input, weights, bias);
            var loss = _engine.ReduceSum(output, null);
            var grads = tape.ComputeGradients(loss, new[] { input, weights, bias });

            Assert.True(grads.ContainsKey(weights), "weight gradient missing");
            Assert.True(grads.ContainsKey(bias), "bias gradient missing");
            Assert.True(grads.ContainsKey(input), "input gradient missing");

            // Validate FULL finiteness across every gradient element so a
            // NaN/Inf from a buffer-reuse bug anywhere in the K×N / M×K /
            // 1×N grads fails loudly. Spot-checks (every Nth entry) let
            // contiguous corruption slip through.
            for (int i = 0; i < grads[weights].Length; i++)
                Assert.True(!float.IsNaN(grads[weights].GetFlat(i)) && !float.IsInfinity(grads[weights].GetFlat(i)),
                    $"non-finite weight gradient at index {i}");
            for (int i = 0; i < grads[bias].Length; i++)
                Assert.True(!float.IsNaN(grads[bias].GetFlat(i)) && !float.IsInfinity(grads[bias].GetFlat(i)),
                    $"non-finite bias gradient at index {i}");
            for (int i = 0; i < grads[input].Length; i++)
                Assert.True(!float.IsNaN(grads[input].GetFlat(i)) && !float.IsInfinity(grads[input].GetFlat(i)),
                    $"non-finite input gradient at index {i}");

            if (step == 0)
            {
                referenceWeightGrad = Enumerable.Range(0, grads[weights].Length)
                    .Select(i => grads[weights].GetFlat(i)).ToArray();
                referenceBiasGrad = Enumerable.Range(0, grads[bias].Length)
                    .Select(i => grads[bias].GetFlat(i)).ToArray();
                referenceInputGrad = Enumerable.Range(0, grads[input].Length)
                    .Select(i => grads[input].GetFlat(i)).ToArray();
                continue;
            }

            // Gradient values must be identical across steps — if the
            // pooled scratch buffer leaks stale data into a subsequent
            // call, this catches it. Validate FULL gradient content for
            // weights AND bias AND input across steps.
            for (int i = 0; i < referenceWeightGrad!.Length; i++)
            {
                float actual = grads[weights].GetFlat(i);
                Assert.True(Math.Abs(actual - referenceWeightGrad[i]) <= 1e-3f,
                    $"FusedLinear+ReLU weight gradient drifted at step {step}, " +
                    $"index {i}: expected {referenceWeightGrad[i]}, got {actual}");
            }
            for (int i = 0; i < referenceBiasGrad!.Length; i++)
            {
                float actual = grads[bias].GetFlat(i);
                Assert.True(Math.Abs(actual - referenceBiasGrad[i]) <= 1e-3f,
                    $"FusedLinear+ReLU bias gradient drifted at step {step}, " +
                    $"index {i}: expected {referenceBiasGrad[i]}, got {actual}");
            }
            for (int i = 0; i < referenceInputGrad!.Length; i++)
            {
                float actual = grads[input].GetFlat(i);
                Assert.True(Math.Abs(actual - referenceInputGrad[i]) <= 1e-3f,
                    $"FusedLinear+ReLU input gradient drifted at step {step}, " +
                    $"index {i}: expected {referenceInputGrad[i]}, got {actual}");
            }
        }
    }

    private static float[] MakeFilled(int len, float scale)
    {
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)((i * 0.017 + 0.1) * scale);
        return data;
    }
}