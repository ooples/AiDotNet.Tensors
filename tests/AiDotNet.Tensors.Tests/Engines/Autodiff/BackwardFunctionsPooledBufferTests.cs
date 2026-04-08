using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for the pooled-buffer changes introduced in BackwardFunctions.cs.
///
/// Previously the MatMul, FusedLinear, and FusedReLU backward passes allocated
/// fresh <c>float[]</c> arrays on every call.  The PR replaces those with
/// <see cref="Helpers.AutoTensorCache.RentOrAllocate{T}"/> pooled tensors.
///
/// Pooled tensors may contain stale data from a previous call, so each caller
/// is responsible for ensuring the buffer is correctly initialised before use
/// (either via BLAS beta=0 or explicit <c>Array.Clear</c>).  These tests verify
/// that the gradient values produced by consecutive backward passes are not
/// contaminated by data left in the pool from the preceding pass.
/// </summary>
public class BackwardFunctionsPooledBufferTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // ──────────────────────────────────────────────────────────────
    // MatMul backward — pooled gradATensor / gradBTensor
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void MatMulBackward_PooledBuffers_GradientMatchesOnFirstAndSubsequentCalls()
    {
        // The first call exercises potentially-fresh buffers; subsequent calls
        // exercise the REUSE path.  All results must agree with the analytical
        // gradient: for loss = sum(A @ B), dA = ones @ B^T, dB = A^T @ ones.
        int M = 8, K = 6, N = 4;
        var aData = CreateSequential(M * K);   // fixed values → deterministic gradients
        var bData = CreateSequential(K * N);
        var a = new Tensor<float>(aData, [M, K]);
        var b = new Tensor<float>(bData, [K, N]);

        // Compute reference gradients on the first call
        Dictionary<Tensor<float>, Tensor<float>> refGrads;
        using (var tape = new GradientTape<float>())
        {
            var result = _engine.TensorMatMul(a, b);
            var loss = _engine.ReduceSum(result, null);
            refGrads = tape.ComputeGradients(loss, new[] { a, b });
        }

        // Run 4 more backward passes; each must produce the same gradients.
        for (int run = 0; run < 4; run++)
        {
            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                var result = _engine.TensorMatMul(a, b);
                var loss = _engine.ReduceSum(result, null);
                grads = tape.ComputeGradients(loss, new[] { a, b });
            }

            Assert.Equal(refGrads[a].Length, grads[a].Length);
            Assert.Equal(refGrads[b].Length, grads[b].Length);

            for (int i = 0; i < refGrads[a].Length; i++)
                Assert.True(
                    Math.Abs((double)refGrads[a][i] - (double)grads[a][i]) < 1e-4,
                    $"run {run}: gradA[{i}] mismatch (pooled buffer stale data?)");

            for (int i = 0; i < refGrads[b].Length; i++)
                Assert.True(
                    Math.Abs((double)refGrads[b][i] - (double)grads[b][i]) < 1e-4,
                    $"run {run}: gradB[{i}] mismatch (pooled buffer stale data?)");
        }
    }

    [Fact]
    public void MatMulBackward_PooledBuffers_DifferentInputsEachStep_CorrectGradients()
    {
        // When inputs change between runs, the pooled buffer from the previous run
        // must not contaminate the new gradient.  The BLAS call uses beta=0 which
        // should overwrite the buffer entirely.
        int M = 4, K = 4, N = 4;
        var rng = new Random(77);

        for (int run = 0; run < 5; run++)
        {
            // Fresh random inputs each run
            var a = CreateRandom([M, K], rng);
            var b = CreateRandom([K, N], rng);

            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                var result = _engine.TensorMatMul(a, b);
                var loss = _engine.ReduceSum(result, null);
                grads = tape.ComputeGradients(loss, new[] { a, b });
            }

            // Analytical: loss = sum(A@B) → dA[i,k] = sum_n B[k,n], dB[k,n] = sum_m A[m,k]
            var bArr = b.GetDataArray();
            var aArr = a.GetDataArray();

            // Spot-check: dA[0,0] = sum over n of B[0,n]  (first row of B)
            float dA00 = 0f;
            for (int n = 0; n < N; n++)
                dA00 += bArr[0 * N + n];
            Assert.True(
                Math.Abs((double)dA00 - (double)grads[a][0]) < 1e-3,
                $"run {run}: dA[0,0] expected={dA00} actual={grads[a][0]}");

            // Spot-check: dB[0,0] = sum over m of A[m,0]  (first column of A)
            float dB00 = 0f;
            for (int m = 0; m < M; m++)
                dB00 += aArr[m * K + 0];
            Assert.True(
                Math.Abs((double)dB00 - (double)grads[b][0]) < 1e-3,
                $"run {run}: dB[0,0] expected={dB00} actual={grads[b][0]}");
        }
    }

    // ──────────────────────────────────────────────────────────────
    // FusedLinear backward — pooled inputGrad / weightGrad / biasGrad
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearBackward_PooledBuffers_ConsistentResultsAcrossMultipleCalls()
    {
        // FusedLinear backward uses Array.Clear on the bias buffer.  Verify that
        // the bias gradient does not accumulate across pooled-buffer reuse.
        int batchSize = 4, inF = 8, outF = 6;
        var rng = new Random(13);
        var input = CreateRandom([batchSize, inF], rng);
        var weight = CreateRandom([inF, outF], rng);
        var bias = CreateRandom([outF], rng);

        // Reference on first call
        Dictionary<Tensor<float>, Tensor<float>> refGrads;
        using (var tape = new GradientTape<float>())
        {
            var output = _engine.FusedLinearReLU(input, weight, bias);
            var loss = _engine.ReduceSum(output, null);
            refGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
        }

        // Repeat and verify
        for (int run = 0; run < 4; run++)
        {
            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                var output = _engine.FusedLinearReLU(input, weight, bias);
                var loss = _engine.ReduceSum(output, null);
                grads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            // Bias gradient is most sensitive to stale pooled data
            for (int j = 0; j < outF; j++)
                Assert.True(
                    Math.Abs((double)refGrads[bias][j] - (double)grads[bias][j]) < 1e-4,
                    $"run {run}: bias grad[{j}] mismatch — possible Array.Clear omission");

            for (int i = 0; i < refGrads[weight].Length; i++)
                Assert.True(
                    Math.Abs((double)refGrads[weight][i] - (double)grads[weight][i]) < 1e-4,
                    $"run {run}: weight grad[{i}] mismatch");

            for (int i = 0; i < refGrads[input].Length; i++)
                Assert.True(
                    Math.Abs((double)refGrads[input][i] - (double)grads[input][i]) < 1e-4,
                    $"run {run}: input grad[{i}] mismatch");
        }
    }

    [Fact]
    public void FusedLinearBackward_BiasGrad_NotAccumulatedAcrossPooledReuse()
    {
        // Critical regression test: bias gradient must equal sum(maskedGrad, axis=0)
        // for EACH backward call independently, not accumulate across calls.
        // If Array.Clear is missing from the bias buffer, the gradient would grow
        // on each subsequent reuse of the same pooled buffer.
        int batchSize = 8, inF = 6, outF = 4;
        var rng = new Random(55);
        var input = CreateRandom([batchSize, inF], rng);
        var weight = CreateRandom([inF, outF], rng);
        var bias = new Tensor<float>(new float[outF], [outF]); // zeros

        float[]? firstBiasGrad = null;

        for (int run = 0; run < 5; run++)
        {
            Dictionary<Tensor<float>, Tensor<float>> grads;
            using (var tape = new GradientTape<float>())
            {
                var output = _engine.FusedLinearReLU(input, weight, bias);
                var loss = _engine.ReduceSum(output, null);
                grads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            var biasGrad = grads[bias].GetDataArray();

            if (firstBiasGrad is null)
            {
                firstBiasGrad = (float[])biasGrad.Clone();
                continue;
            }

            // All runs must produce the identical bias gradient — not k * firstBiasGrad
            for (int j = 0; j < outF; j++)
                Assert.True(
                    Math.Abs((double)firstBiasGrad[j] - (double)biasGrad[j]) < 1e-4,
                    $"run {run}: bias grad[{j}] expected={firstBiasGrad[j]:R} actual={biasGrad[j]:R} — " +
                    "bias gradient is accumulating across pooled buffer reuse (missing Array.Clear)");
        }
    }

    // ──────────────────────────────────────────────────────────────
    // FusedLinearReLU backward — pooled maskedGrad / gradInput / gradWeight / gradBias
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearReLUBackward_PooledBuffers_MatchesUnfusedEquivalent_MultipleRuns()
    {
        // Verify that pooled buffers do not corrupt the gradient when the fused path
        // is exercised across repeated calls.  Compare against unfused path.
        int batchSize = 4, inF = 8, outF = 6;
        var rng = new Random(99);
        var input = CreateRandom([batchSize, inF], rng);
        var weight = CreateRandom([inF, outF], rng);
        var bias = CreateRandom([outF], rng);

        for (int run = 0; run < 4; run++)
        {
            // Unfused reference
            Dictionary<Tensor<float>, Tensor<float>> unfusedGrads;
            using (var tape = new GradientTape<float>())
            {
                var linear = _engine.TensorMatMul(input, weight);
                var biased = _engine.TensorBroadcastAdd(linear, bias);
                var activated = _engine.ReLU(biased);
                var loss = _engine.ReduceSum(activated, [0, 1], keepDims: false);
                unfusedGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            // Fused path (uses pooled buffers internally)
            Dictionary<Tensor<float>, Tensor<float>> fusedGrads;
            using (var tape = new GradientTape<float>())
            {
                var output = _engine.FusedLinearReLU(input, weight, bias);
                var loss = _engine.ReduceSum(output, [0, 1], keepDims: false);
                fusedGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            foreach (var param in new[] { input, weight, bias })
            {
                Assert.Equal(unfusedGrads[param].Length, fusedGrads[param].Length);
                for (int i = 0; i < unfusedGrads[param].Length; i++)
                    Assert.Equal((double)unfusedGrads[param][i], (double)fusedGrads[param][i], precision: 4);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // DifferentiableOps: forward recording always happens even in ReplayMode
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void DifferentiableOps_ForwardRecording_HappensEvenWhenReplayModeIsTrue()
    {
        // Per the PR comment in DifferentiableOps.cs:
        //   "we always record during forward passes even when a compiled backward exists"
        // This ensures GradFn is set and the tape is populated as a fallback.
        bool savedReplay = AutoTrainingCompiler.ReplayMode;
        try
        {
            AutoTrainingCompiler.ReplayMode = true;

            var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, [2, 2]);
            var b = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, [2, 2]);

            using var tape = new GradientTape<float>();
            int countBefore = tape.EntryCount;
            _engine.TensorAdd(a, b);

            // Recording must have happened despite ReplayMode=true
            Assert.True(tape.EntryCount > countBefore,
                "Forward pass must record to tape even when ReplayMode=true " +
                "(needed for GradFn and tape-based fallback path)");
        }
        finally
        {
            AutoTrainingCompiler.ReplayMode = savedReplay;
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Regression: MatMul backward gradient values (sanity check)
    // ──────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(2, 3, 2)]
    [InlineData(4, 8, 4)]
    [InlineData(8, 16, 4)]
    public void MatMulBackward_GradientValues_CorrectForVariousSizes(int M, int K, int N)
    {
        // For loss = sum(A @ B), the analytical gradients are:
        //   dA[i, k] = sum_j B[k, j]  (sum of k-th row of B)
        //   dB[k, j] = sum_i A[i, k]  (sum of k-th column of A)
        var rng = new Random(M * 100 + K * 10 + N);
        var a = CreateRandom([M, K], rng);
        var b = CreateRandom([K, N], rng);

        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var result = _engine.TensorMatMul(a, b);
            var loss = _engine.ReduceSum(result, null);
            grads = tape.ComputeGradients(loss, new[] { a, b });
        }

        var aArr = a.GetDataArray();
        var bArr = b.GetDataArray();

        // Verify dA
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < K; k++)
            {
                float expected = 0f;
                for (int n = 0; n < N; n++)
                    expected += bArr[k * N + n];
                Assert.Equal((double)expected, (double)grads[a][i * K + k], precision: 3);
            }
        }

        // Verify dB
        for (int k = 0; k < K; k++)
        {
            for (int n = 0; n < N; n++)
            {
                float expected = 0f;
                for (int m = 0; m < M; m++)
                    expected += aArr[m * K + k];
                Assert.Equal((double)expected, (double)grads[b][k * N + n], precision: 3);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // FusedLinear (no activation) backward — pooled gradients
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void FusedLinearNoActivation_Backward_MatchesManualEquivalent_MultipleRuns()
    {
        // FusedLinear without activation (Identity) should match manual MatMul+BroadcastAdd.
        int batchSize = 4, inF = 6, outF = 4;
        var rng = new Random(31);
        var input = CreateRandom([batchSize, inF], rng);
        var weight = CreateRandom([inF, outF], rng);
        var bias = CreateRandom([outF], rng);

        for (int run = 0; run < 3; run++)
        {
            // Manual reference
            Dictionary<Tensor<float>, Tensor<float>> refGrads;
            using (var tape = new GradientTape<float>())
            {
                var linear = _engine.TensorMatMul(input, weight);
                var biased = _engine.TensorBroadcastAdd(linear, bias);
                var loss = _engine.ReduceSum(biased, null);
                refGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            // Fused path (no activation)
            Dictionary<Tensor<float>, Tensor<float>> fusedGrads;
            using (var tape = new GradientTape<float>())
            {
                var output = _engine.FusedLinear(input, weight, bias, FusedActivationType.None);
                var loss = _engine.ReduceSum(output, null);
                fusedGrads = tape.ComputeGradients(loss, new[] { input, weight, bias });
            }

            foreach (var param in new[] { input, weight, bias })
            {
                for (int i = 0; i < refGrads[param].Length; i++)
                    Assert.Equal((double)refGrads[param][i], (double)fusedGrads[param][i], precision: 4);
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────

    private static float[] CreateSequential(int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = (i + 1) * 0.1f;
        return data;
    }

    private static Tensor<float> CreateRandom(int[] shape, Random rng)
    {
        int len = 1;
        foreach (int d in shape) len *= d;
        var data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }
}