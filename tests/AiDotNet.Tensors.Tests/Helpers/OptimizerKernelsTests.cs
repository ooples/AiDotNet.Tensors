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
    public void AdamInPlace_Float_MatchesReferenceFormula()
    {
        const int N = 4096;
        var rng = new Random(1001);
        var paramKernel = MakeRandom<float>(N, rng);
        var paramRef = CloneFloat(paramKernel);
        var grad = MakeRandom<float>(N, rng);
        var mKernel = new Tensor<float>(new[] { N });
        var vKernel = new Tensor<float>(new[] { N });
        var mRef = new Tensor<float>(new[] { N });
        var vRef = new Tensor<float>(new[] { N });

        const float lr = 1e-3f;
        const float b1 = 0.9f;
        const float b2 = 0.999f;
        const float eps = 1e-8f;

        // Run 5 steps of both and compare.
        for (int step = 1; step <= 5; step++)
        {
            OptimizerKernels.AdamInPlace(paramKernel, grad, mKernel, vKernel, lr, b1, b2, eps, step);
            AdamReference(paramRef, grad, mRef, vRef, lr, b1, b2, eps, step);

            var spanK = paramKernel.AsSpan();
            var spanR = paramRef.AsSpan();
            float maxDiff = 0;
            for (int i = 0; i < N; i++)
            {
                float diff = MathF.Abs(spanK[i] - spanR[i]);
                float tol = 1e-5f * MathF.Max(1f, MathF.Abs(spanR[i]));
                Assert.True(diff <= tol,
                    $"Step {step}: AdamInPlace<float> diverges at idx {i}: kernel={spanK[i]} ref={spanR[i]} diff={diff}");
                if (diff > maxDiff) maxDiff = diff;
            }
            _output.WriteLine($"Adam step {step}: max abs diff = {maxDiff:E4}");
        }
    }

    [Fact]
    public void AdamWInPlace_Float_AppliesDecoupledWeightDecay()
    {
        // Verify AdamW differs from Adam ONLY by the (1 - weightDecay*lr)
        // pre-multiply on param. With weightDecay=0 the two paths must
        // be byte-identical.
        const int N = 1024;
        var rng = new Random(2002);
        var p1 = MakeRandom<float>(N, rng);
        var p2 = CloneFloat(p1);
        var grad = MakeRandom<float>(N, rng);
        var m1 = new Tensor<float>(new[] { N });
        var v1 = new Tensor<float>(new[] { N });
        var m2 = new Tensor<float>(new[] { N });
        var v2 = new Tensor<float>(new[] { N });

        OptimizerKernels.AdamInPlace(p1, grad, m1, v1, 1e-3f, 0.9f, 0.999f, 1e-8f, 1);
        OptimizerKernels.AdamWInPlace(p2, grad, m2, v2, 1e-3f, 0.9f, 0.999f, 1e-8f, weightDecay: 0f, step: 1);

        var s1 = p1.AsSpan();
        var s2 = p2.AsSpan();
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(s1[i], s2[i]);
        }

        // With weightDecay > 0, AdamW must produce strictly different output.
        var p3 = MakeRandom<float>(N, rng);
        var p4 = CloneFloat(p3);
        var m3 = new Tensor<float>(new[] { N });
        var v3 = new Tensor<float>(new[] { N });
        var m4 = new Tensor<float>(new[] { N });
        var v4 = new Tensor<float>(new[] { N });
        OptimizerKernels.AdamInPlace(p3, grad, m3, v3, 1e-3f, 0.9f, 0.999f, 1e-8f, 1);
        OptimizerKernels.AdamWInPlace(p4, grad, m4, v4, 1e-3f, 0.9f, 0.999f, 1e-8f, weightDecay: 0.1f, step: 1);

        var s3 = p3.AsSpan();
        var s4 = p4.AsSpan();
        int differences = 0;
        for (int i = 0; i < N; i++) if (s3[i] != s4[i]) differences++;
        Assert.True(differences > N / 2,
            $"AdamW with weightDecay=0.1 should differ from Adam on most elements; observed {differences}/{N}.");
    }

    [Fact]
    public void AdamInPlace_RejectsBadArguments()
    {
        var p = new Tensor<float>(new[] { 4 });
        var g = new Tensor<float>(new[] { 4 });
        var m = new Tensor<float>(new[] { 4 });
        var v = new Tensor<float>(new[] { 4 });

        Assert.Throws<ArgumentNullException>(() =>
            OptimizerKernels.AdamInPlace<float>(null!, g, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, 1));
        Assert.Throws<ArgumentNullException>(() =>
            OptimizerKernels.AdamInPlace<float>(p, null!, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, 1));
        Assert.Throws<ArgumentNullException>(() =>
            OptimizerKernels.AdamInPlace<float>(p, g, null!, v, 1e-3f, 0.9f, 0.999f, 1e-8f, 1));
        Assert.Throws<ArgumentNullException>(() =>
            OptimizerKernels.AdamInPlace<float>(p, g, m, null!, 1e-3f, 0.9f, 0.999f, 1e-8f, 1));

        var wrongLen = new Tensor<float>(new[] { 8 });
        Assert.Throws<ArgumentException>(() =>
            OptimizerKernels.AdamInPlace(p, wrongLen, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, 1));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            OptimizerKernels.AdamInPlace(p, g, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, step: 0));
    }

    [Fact]
    public void AdamInPlace_PerformsZeroTensorAllocationsPerCall()
    {
        const int Hidden = 768;
        const int Iters = 100;
        var rng = new Random(7);
        var p = MakeRandom<float>(Hidden * Hidden, rng);
        var g = MakeRandom<float>(Hidden * Hidden, rng);
        var m = new Tensor<float>(new[] { Hidden * Hidden });
        var v = new Tensor<float>(new[] { Hidden * Hidden });

        for (int i = 0; i < 3; i++)
            OptimizerKernels.AdamInPlace(p, g, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, i + 1);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

#if NET5_0_OR_GREATER
        long before = GC.GetTotalAllocatedBytes(precise: true);
#else
        long before = GC.GetTotalMemory(forceFullCollection: false);
#endif

        for (int i = 0; i < Iters; i++)
            OptimizerKernels.AdamInPlace(p, g, m, v, 1e-3f, 0.9f, 0.999f, 1e-8f, 4 + i);

#if NET5_0_OR_GREATER
        long after = GC.GetTotalAllocatedBytes(precise: true);
#else
        long after = GC.GetTotalMemory(forceFullCollection: false);
#endif

        long perCallBytes = (after - before) / Iters;
        _output.WriteLine($"AdamInPlace allocation: {after - before} bytes over {Iters} iters = {perCallBytes} bytes/call.");
#if NET5_0_OR_GREATER
        Assert.True(perCallBytes <= 1024,
            $"AdamInPlace allocated {perCallBytes} bytes/call — budget 1024 bytes. " +
            "Regression: the fused Adam kernel is supposed to be zero-allocation.");
#endif
    }

    // Reference Adam implementation — straight transcription of the
    // canonical update for cross-checking the SIMD kernel. Never the
    // hot path; only used by the equivalence test.
    private static void AdamReference(
        Tensor<float> param, Tensor<float> grad,
        Tensor<float> m, Tensor<float> v,
        float lr, float beta1, float beta2, float eps, int step)
    {
        var p = param.AsWritableSpan();
        var g = grad.AsSpan();
        var ms = m.AsWritableSpan();
        var vs = v.AsWritableSpan();
        float bc1 = 1f - MathF.Pow(beta1, step);
        float bc2 = 1f - MathF.Pow(beta2, step);
        for (int i = 0; i < p.Length; i++)
        {
            ms[i] = beta1 * ms[i] + (1f - beta1) * g[i];
            vs[i] = beta2 * vs[i] + (1f - beta2) * g[i] * g[i];
            float mHat = ms[i] / bc1;
            float vHat = vs[i] / bc2;
            p[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
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
        _output.WriteLine($"SgdInPlace allocation: {deltaBytes} bytes over {Iters} iters = {perCallBytes} bytes/call.");
        _output.WriteLine($"Naive path equivalent: ~{Hidden * Hidden * 4} bytes/call (one Tensor<float>).");
#if NET5_0_OR_GREATER
        // GC.GetTotalAllocatedBytes(precise: true) measures bytes
        // *allocated* — monotonic, unaffected by intervening GCs.
        // Assert tightly on this platform: kernel must allocate well
        // below one tensor's worth of bytes. 1 KB headroom covers
        // JIT tier-up state and internal counter updates.
        const long PerCallBudgetBytes = 1024;
        Assert.True(perCallBytes <= PerCallBudgetBytes,
            $"SgdInPlace allocated {perCallBytes} bytes/call — budget is {PerCallBudgetBytes} bytes. " +
            "Regression: the fused kernel is supposed to be zero-allocation.");
#else
        // On net471 the only available signal is GC.GetTotalMemory,
        // which reports current heap size — sensitive to background
        // GCs running between the two snapshots. Skip the strict
        // assertion here and just verify the kernel succeeded; the
        // diagnostic line above lets a reviewer eyeball it.
        // The numerical-equivalence test and net5+ allocation-budget
        // test together prove the contract.
#endif
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
