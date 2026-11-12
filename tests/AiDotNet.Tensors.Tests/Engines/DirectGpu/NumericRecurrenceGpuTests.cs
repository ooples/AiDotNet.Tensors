#if NETCOREAPP
// GpuCpuCorrectnessFixture (GpuCpuCorrectnessTests.cs) is compiled only off .NET Framework.
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Ref = AiDotNet.Tensors.Tests.Engines.NumericRecurrenceOpsTests;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The device numeric recurrences (INumericRecurrenceBackend, implemented by every backend) through the GPU engine
/// against the CPU engine and the reference values of <see cref="NumericRecurrenceOpsTests"/>: the tridiagonal solve and
/// all four of its gradients, the Fritsch-Carlson limiter, the CTC lattice, posteriors and loss gradient (whose backward
/// runs the device posteriors), and the assignment. No case may fall back to the host.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class NumericRecurrenceGpuTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public NumericRecurrenceGpuTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    private static void AssertNoHostFallback(string op) =>
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains(op));

    private static double[] D(Tensor<float> t) => t.ToArray().Select(v => (double)v).ToArray();

    [SkippableFact]
    public void TridiagonalSolve_AndGradients_MatchCpu()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        int batch = 4, n = 23;
        var lo = Ref.Random(31, 1, 0, batch, n - 1); var up = Ref.Random(32, 1, 0, batch, n - 1);
        var dg = Ref.Random(33, 1, 4, batch, n); var rh = Ref.Random(34, 1, 0, batch, n);
        var w = Ref.Random(35, 1, 0, batch, n);
        Dictionary<Tensor<float>, Tensor<float>> gpuGrads, cpuGrads;
        Tensor<float> xGpu, xCpu;
        using (var tape = new GradientTape<float>())
        {
            xGpu = _fixture.Gpu.TensorTridiagonalSolve(lo, dg, up, rh);
            var loss = _fixture.Gpu.ReduceSum(_fixture.Gpu.TensorMultiply(xGpu, w), null);
            gpuGrads = tape.ComputeGradients(loss, new[] { lo, dg, up, rh });
        }
        using (var tape = new GradientTape<float>())
        {
            xCpu = _fixture.Cpu.TensorTridiagonalSolve(lo, dg, up, rh);
            var loss = _fixture.Cpu.ReduceSum(_fixture.Cpu.TensorMultiply(xCpu, w), null);
            cpuGrads = tape.ComputeGradients(loss, new[] { lo, dg, up, rh });
        }
        Ref.AssertClose(D(xCpu), xGpu, 1e-4, "x");
        foreach (var (input, name) in new[] { (lo, "dlower"), (dg, "ddiag"), (up, "dupper"), (rh, "drhs") })
            Ref.AssertClose(D(cpuGrads[input]), gpuGrads[input], 1e-4, name);
        AssertNoHostFallback("TridiagonalSolve");
    }

    [SkippableFact]
    public void FritschCarlsonLimit_MatchesScalarRecurrence()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        int batch = 3, rows = 5, n = 41;
        var m = Ref.Random(41, 3, 0, batch, rows, n); var d = Ref.Random(42, 1, 0, batch, rows, n - 1);
        var r = _fixture.Gpu.TensorFritschCarlsonLimit(m, d);
        Assert.Equal(new[] { batch, rows, n }, r.Shape.ToArray());
        Ref.AssertClose(Ref.ScalarFritschCarlson(m.ToArray(), d.ToArray(), batch * rows, n), r, 1e-5, "m");
        AssertNoHostFallback("FritschCarlsonLimit");
    }

    private static Tensor<float> CtcLogProbsF() => new Tensor<float>(Ref.CtcLogProbs().ToArray().Select(v => (float)v).ToArray(), new[] { 4, 2, 3 });

    [SkippableFact]
    public void CtcForwardBackward_AndPosteriors_MatchTorch()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        var lp = CtcLogProbsF();
        var (alpha, beta) = _fixture.Gpu.TensorCTCForwardBackward(lp, Ref.CtcTargets, Ref.CtcInputLengths, Ref.CtcTargetLengths, 0);
        Ref.AssertClose(Ref.TorchLogAlpha, alpha, 1e-5, "alpha");
        var (_, cpuBeta) = _fixture.Cpu.TensorCTCForwardBackward(lp, Ref.CtcTargets, Ref.CtcInputLengths, Ref.CtcTargetLengths, 0);
        Ref.AssertClose(cpuBeta.ToArray().Select(v => (double)v).ToArray(), beta, 1e-5, "beta");
        var gamma = _fixture.Gpu.TensorCTCLabelPosteriors(lp, Ref.CtcTargets, Ref.CtcInputLengths, Ref.CtcTargetLengths, 0);
        Ref.AssertClose(Ref.TorchPosteriors, gamma, 1e-5, "gamma");
        AssertNoHostFallback("CTCForwardBackward");
        AssertNoHostFallback("CTCLabelPosteriors");
    }

    // The loss's backward is -gamma from TensorCTCLabelPosteriors, so on the GPU engine it runs on the device.
    [SkippableFact]
    public void CtcLossGradient_IsMinusTorchPosteriors()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        var lp = CtcLogProbsF();
        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var loss = _fixture.Gpu.TensorCTCLoss(lp, Ref.CtcTargets, Ref.CtcInputLengths, Ref.CtcTargetLengths, 0);
            grads = tape.ComputeGradients(_fixture.Gpu.ReduceSum(loss, null), new[] { lp });
        }
        Ref.AssertClose(Ref.TorchPosteriors.Select(v => -v).ToArray(), grads[lp], 1e-5, "dlp");
        AssertNoHostFallback("CTCLabelPosteriors");
    }

    [SkippableFact]
    public void LinearSumAssignment_MatchesCpu_AndScipyExample()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        var doc = new Tensor<float>(new[] { 4f, 1, 3, 2, 0, 5, 3, 2, 2 }, new[] { 3, 3 });
        Assert.Equal(new[] { 1, 0, 2 }, _fixture.Gpu.TensorLinearSumAssignment(doc).ToArray());
        var cost = Ref.Random(51, 5, 0, 5, 12, 12);
        var gpu = _fixture.Gpu.TensorLinearSumAssignment(cost);
        Assert.Equal(new[] { 5, 12 }, gpu.Shape.ToArray());
        Assert.Equal(_fixture.Cpu.TensorLinearSumAssignment(cost).ToArray(), gpu.ToArray());
        AssertNoHostFallback("LinearSumAssignment");
    }

    [SkippableFact]
    public void LinearSumAssignment_ExtremeCosts_DoNotOverflowTheAugmentingPath()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        float h = float.MaxValue;
        var finite = new Tensor<float>(new[] { h, -h, -h, h }, new[] { 2, 2 });
        Assert.Equal(new[] { 1, 0 }, _fixture.Gpu.TensorLinearSumAssignment(finite).ToArray());
        var forced = new Tensor<float>(new[] { float.NaN, float.NaN, h, -h }, new[] { 2, 2 });
        Assert.Equal(new[] { 0, 1 }, _fixture.Gpu.TensorLinearSumAssignment(forced).ToArray());
        AssertNoHostFallback("LinearSumAssignment");
    }

    // The limiter and the CTC posteriors on random batched inputs (ragged lengths, repeated labels) against the CPU
    // engine itself, not only the scalar / PyTorch references above.
    [SkippableFact]
    public void FritschCarlsonLimit_AndCtcPosteriors_MatchTheCpuEngine()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        int batch = 4, rows = 3, n = 67;
        var m = Ref.Random(61, 3, 0, batch, rows, n);
        var d = Ref.Random(62, 1, 0, batch, rows, n - 1);
        GpuLaunchProbe.Reset();
        var gpuLimit = _fixture.Gpu.TensorFritschCarlsonLimit(m, d);
        AssertNoHostFallback("FritschCarlsonLimit");
        Ref.AssertClose(D(_fixture.Cpu.TensorFritschCarlsonLimit(m, d)), gpuLimit, 1e-5, "limited slopes");

        int T = 23, N = 3, C = 5;
        var raw = Ref.Random(63, 2, 0, T, N, C).ToArray();
        var lp = new float[raw.Length];
        for (int row = 0; row < T * N; row++)
        {
            double sum = 0;
            for (int c = 0; c < C; c++) sum += System.Math.Exp(raw[row * C + c]);
            for (int c = 0; c < C; c++) lp[row * C + c] = (float)(raw[row * C + c] - System.Math.Log(sum));
        }
        var logProbs = new Tensor<float>(lp, new[] { T, N, C });
        // Concatenated targets (the PyTorch 1-D layout): 6, 4 and 5 labels, with repeats that force a blank between.
        var targets = new Tensor<int>(new[] { 1, 2, 2, 3, 4, 1, 3, 3, 3, 1, 4, 2, 1, 2, 4 }, new[] { 15 });
        var inputLengths = new[] { 23, 17, 20 };
        var targetLengths = new[] { 6, 4, 5 };
        GpuLaunchProbe.Reset();
        var gpuGamma = _fixture.Gpu.TensorCTCLabelPosteriors(logProbs, targets, inputLengths, targetLengths, 0);
        AssertNoHostFallback("CTCLabelPosteriors");
        var cpuGamma = _fixture.Cpu.TensorCTCLabelPosteriors(logProbs, targets, inputLengths, targetLengths, 0);
        Assert.Equal(cpuGamma.Shape.ToArray(), gpuGamma.Shape.ToArray());
        Ref.AssertClose(D(cpuGamma), gpuGamma, 1e-4, "posteriors");
    }

    // Sizes past what shared memory holds (1024 lattice positions, 400 assignment rows) still run on the device.
    [SkippableFact]
    public void LargeProblems_StayOnTheDevice()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        // The fallback record is process-wide: clear what earlier tests left before this test's device work.
        GpuLaunchProbe.Reset();
        var cost = Ref.Random(52, 5, 0, 450, 450);
        Assert.Equal(_fixture.Cpu.TensorLinearSumAssignment(cost).ToArray(), _fixture.Gpu.TensorLinearSumAssignment(cost).ToArray());

        int T = 1100, U = 520, C = 3;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(53);
        var labels = Enumerable.Range(0, U).Select(_ => 1 + rng.Next(C - 1)).ToArray();
        var raw = Ref.Random(54, 1, 0, T, 1, C).ToArray();
        var lp = new float[raw.Length];
        for (int t = 0; t < T; t++)
        {
            double sum = 0;
            for (int c = 0; c < C; c++) sum += System.Math.Exp(raw[t * C + c]);
            for (int c = 0; c < C; c++) lp[t * C + c] = (float)(raw[t * C + c] - System.Math.Log(sum));
        }
        var lpT = new Tensor<float>(lp, new[] { T, 1, C });
        var tg = new Tensor<int>(labels, new[] { U });
        var (alpha, _) = _fixture.Gpu.TensorCTCForwardBackward(lpT, tg, new[] { T }, new[] { U }, 0);
        var (cpuAlpha, _) = _fixture.Cpu.TensorCTCForwardBackward(lpT, tg, new[] { T }, new[] { U }, 0);
        Assert.Equal(new[] { 1, T, 2 * U + 1 }, alpha.Shape.ToArray());
        var expected = cpuAlpha.ToArray().Select(v => (double)v).ToArray();
        var actual = alpha.ToArray();
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsNegativeInfinity(expected[i])) Assert.True(float.IsNegativeInfinity(actual[i]), $"alpha[{i}]");
            else Assert.True(System.Math.Abs(expected[i] - actual[i]) <= 1e-4 * System.Math.Max(1, System.Math.Abs(expected[i])),
                $"alpha[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
        }
        AssertNoHostFallback("LinearSumAssignment");
        AssertNoHostFallback("CTCForwardBackward");
    }
}
#endif
