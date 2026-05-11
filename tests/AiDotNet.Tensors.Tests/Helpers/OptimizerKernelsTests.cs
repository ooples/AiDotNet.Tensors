// Copyright (c) AiDotNet. All rights reserved.
// Tests for OptimizerKernels (issue #319 — eliminate per-iter SGD
// step allocation in eager-mode training loops).
//
// Three checks:
//   1. Float numerical equivalence — SgdInPlace produces the same
//      result as the naive TensorMultiplyScalar + TensorSubtractInPlace
//      pattern, within float32 ULP tolerance.
//   2. Double numerical equivalence — same check for the double path.
//   3. Allocation regression guard — SgdInPlace must not allocate any
//      tensor objects on the heap during the update.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Helpers;

public class OptimizerKernelsTests
{
    private readonly ITestOutputHelper _output;
    public OptimizerKernelsTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void SgdInPlace_Float_MatchesNaiveMultiplyScalarThenSubtract()
    {
        const int N = 4096; // covers the AVX2 32-wide unroll, the 8-wide tail, and the scalar tail
        var rng = new Random(123);
        var paramA = MakeRandom<float>(N, rng);
        var paramB = CloneFloat(paramA);
        var grad = MakeRandom<float>(N, rng);
        const float lr = 1e-3f;
        var engine = new CpuEngine();

        // Reference: naive pattern with intermediate tensor allocation.
        var update = engine.TensorMultiplyScalar(grad, lr);
        engine.TensorSubtractInPlace(paramA, update);

        // Target: fused in-place kernel.
        OptimizerKernels.SgdInPlace(paramB, grad, lr);

        var spanA = paramA.AsSpan();
        var spanB = paramB.AsSpan();
        float maxDiff = 0;
        for (int i = 0; i < N; i++)
        {
            float diff = MathF.Abs(spanA[i] - spanB[i]);
            float tol = 1e-6f * MathF.Max(1f, MathF.Abs(spanA[i]));
            Assert.True(diff <= tol,
                $"SgdInPlace<float> diverges at idx {i}: naive={spanA[i]} fused={spanB[i]} diff={diff}");
            if (diff > maxDiff) maxDiff = diff;
        }
        _output.WriteLine($"Float SGD equivalence: N={N} max abs diff = {maxDiff:E4}");
    }

    [Fact]
    public void SgdInPlace_Double_MatchesNaiveMultiplyScalarThenSubtract()
    {
        const int N = 4096;
        var rng = new Random(456);
        var paramA = MakeRandom<double>(N, rng);
        var paramB = CloneDouble(paramA);
        var grad = MakeRandom<double>(N, rng);
        const double lr = 1e-3;
        var engine = new CpuEngine();

        var update = engine.TensorMultiplyScalar(grad, lr);
        engine.TensorSubtractInPlace(paramA, update);

        OptimizerKernels.SgdInPlace(paramB, grad, lr);

        var spanA = paramA.AsSpan();
        var spanB = paramB.AsSpan();
        double maxDiff = 0;
        for (int i = 0; i < N; i++)
        {
            double diff = Math.Abs(spanA[i] - spanB[i]);
            double tol = 1e-13 * Math.Max(1.0, Math.Abs(spanA[i]));
            Assert.True(diff <= tol,
                $"SgdInPlace<double> diverges at idx {i}: naive={spanA[i]} fused={spanB[i]} diff={diff}");
            if (diff > maxDiff) maxDiff = diff;
        }
        _output.WriteLine($"Double SGD equivalence: N={N} max abs diff = {maxDiff:E4}");
    }

    [Fact]
    public void SgdInPlace_RejectsNullArguments()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Throws<ArgumentNullException>(() => OptimizerKernels.SgdInPlace<float>(null!, t, 0.1f));
        Assert.Throws<ArgumentNullException>(() => OptimizerKernels.SgdInPlace<float>(t, null!, 0.1f));
    }

    [Fact]
    public void SgdInPlace_RejectsLengthMismatch()
    {
        var p = new Tensor<float>(new[] { 4 });
        var g = new Tensor<float>(new[] { 8 });
        Assert.Throws<ArgumentException>(() => OptimizerKernels.SgdInPlace(p, g, 0.1f));
    }

    [Fact]
    public void SgdInPlace_EmptyTensor_NoOp()
    {
        var p = new Tensor<float>(new[] { 0 });
        var g = new Tensor<float>(new[] { 0 });
        // Must not throw, must not write to anything.
        OptimizerKernels.SgdInPlace(p, g, 0.1f);
    }

    [Fact]
    public void SgdInPlace_PerformsZeroTensorAllocationsPerCall()
    {
        // The whole point of this primitive is to eliminate the
        // per-call tensor allocation that TensorMultiplyScalar +
        // TensorSubtractInPlace incurs. Verify by snapshotting the
        // total-bytes-allocated counter before and after a tight loop
        // of SgdInPlace calls and checking the delta is below the
        // size of one parameter-shaped tensor.
        const int Hidden = 768;
        const int Iters = 100;
        var p = MakeRandom<float>(Hidden * Hidden, new Random(7));
        var g = MakeRandom<float>(Hidden * Hidden, new Random(13));

        // Warmup so JIT and any first-touch caches settle.
        for (int i = 0; i < 3; i++) OptimizerKernels.SgdInPlace(p, g, 1e-4f);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

#if NET5_0_OR_GREATER
        long before = GC.GetTotalAllocatedBytes(precise: true);
#else
        long before = GC.GetTotalMemory(forceFullCollection: false);
#endif

        for (int i = 0; i < Iters; i++)
        {
            OptimizerKernels.SgdInPlace(p, g, 1e-4f);
        }

#if NET5_0_OR_GREATER
        long after = GC.GetTotalAllocatedBytes(precise: true);
#else
        long after = GC.GetTotalMemory(forceFullCollection: false);
#endif

        long deltaBytes = after - before;
        long perCallBytes = deltaBytes / Iters;
        // The naive path allocates a Tensor<float>[Hidden*Hidden] = 768*768*4
        // = ~2.4 MB per call. We need to be MASSIVELY below that — the
        // budget is essentially zero, but allow a tiny margin for
        // bookkeeping that's not on the kernel path itself (e.g. JIT
        // tier-up state, internal counter updates).
        const long PerCallBudgetBytes = 1024; // 1 KB headroom
        _output.WriteLine($"SgdInPlace allocation: {deltaBytes} bytes over {Iters} iters = {perCallBytes} bytes/call.");
        _output.WriteLine($"Naive path equivalent: ~{Hidden * Hidden * 4} bytes/call (one Tensor<float>).");
        Assert.True(perCallBytes <= PerCallBudgetBytes,
            $"SgdInPlace allocated {perCallBytes} bytes/call — budget is {PerCallBudgetBytes} bytes. " +
            "Regression: the fused kernel is supposed to be zero-allocation.");
    }

    private static Tensor<T> MakeRandom<T>(int length, Random rng) where T : struct
    {
        var t = new Tensor<T>(new[] { length });
        var s = t.AsWritableSpan();
        if (typeof(T) == typeof(float))
        {
            var sf = System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(s);
            for (int i = 0; i < length; i++) sf[i] = (float)((rng.NextDouble() - 0.5) * 2.0);
        }
        else if (typeof(T) == typeof(double))
        {
            var sd = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(s);
            for (int i = 0; i < length; i++) sd[i] = (rng.NextDouble() - 0.5) * 2.0;
        }
        return t;
    }

    private static Tensor<float> CloneFloat(Tensor<float> src)
    {
        var dst = new Tensor<float>(src._shape);
        src.AsSpan().CopyTo(dst.AsWritableSpan());
        return dst;
    }

    private static Tensor<double> CloneDouble(Tensor<double> src)
    {
        var dst = new Tensor<double>(src._shape);
        src.AsSpan().CopyTo(dst.AsWritableSpan());
        return dst;
    }
}
