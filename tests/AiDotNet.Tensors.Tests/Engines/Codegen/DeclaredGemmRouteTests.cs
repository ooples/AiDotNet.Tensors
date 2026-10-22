// Copyright (c) AiDotNet. All rights reserved.
// The declared GEMM route (#890 phase 4c): Gemm, MatMulTransposed, MatMulTransposedA and BatchedGemm at alpha 1,
// beta 0 offer declared schedules to the GEMM tuned slot, whose reference is each entry point's own path. Every
// schedule is checked against an fp64 reference written out here, the entry points with the route on and off, the
// programs under graph capture, and the GEMMs as resident-program stages.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredGemmRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        // IsCudaAvailable, not CudaNativeBindings.IsAvailable: the backend also reports unavailable when NVRTC is
        // missing, and guarding on the driver alone turns that host into a throw instead of a skip.
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA and NVRTC were detected, but the CUDA backend failed to initialize.");
        }
        // Disposed before the skip: Skip throws, and the caller's `using` only takes ownership once this returns, so
        // skipping here would leak the retained primary context and the main stream for the rest of the run.
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is not released on this architecture (see AIDOTNET_DECLARED_KERNELS_ARCH).");
        }
        return backend;
    }

    // Values that are not dyadic, so a reordered fp32 sum differs from the exact one and the bound is exercised.
    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 97.0 + 0.003 * (i % 7));
        return values;
    }

    private static DeclaredFamily FamilyOf(string layout) => layout switch
    {
        "ta" => DeclaredFamily.GemmTransposedA,
        "tb" => DeclaredFamily.GemmTransposedB,
        _ => DeclaredFamily.Gemm,
    };

    // op(A)[i, r] * op(B)[r, j] summed in fp64, per batch, row-major [batch, m, n].
    private static double[] Reference(DeclaredFamily family, float[] a, float[] b, int batch, int m, int n, int k)
    {
        bool transA = family == DeclaredFamily.GemmTransposedA, transB = family == DeclaredFamily.GemmTransposedB;
        var c = new double[batch * m * n];
        for (int g = 0; g < batch; g++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int r = 0; r < k; r++)
                        sum += (double)a[g * m * k + (transA ? r * m + i : i * k + r)] *
                            b[g * k * n + (transB ? j * k + r : r * n + j)];
                    c[(g * m + i) * n + j] = sum;
                }
        return c;
    }

    // fp32 summation error of a K-long dot product, relative to max(1, |reference|).
    private static double Bound(int k) => (16 + Math.Log(Math.Max(2, k), 2)) / (1 << 24) * 8;

    private static double MaxError(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), "NaN at " + i);
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        }
        return worst;
    }

    private static int ScheduleIndex(string name)
    {
        var schedules = CudaBackend.DeclaredGemmSchedules;
        for (int s = 0; s < schedules.Count; s++)
            if (schedules[s].Name == name) return s;
        throw new ArgumentException("No declared GEMM schedule " + name);
    }

    private static void Entry(CudaBackend backend, DeclaredFamily family, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int batch, int m, int n, int k)
    {
        if (batch > 1) backend.BatchedGemm(a, b, c, m, n, k, batch);
        else if (family == DeclaredFamily.GemmTransposedA) backend.MatMulTransposedA(a, b, c, m, n, k);
        else if (family == DeclaredFamily.GemmTransposedB) backend.MatMulTransposed(a, b, c, m, n, k);
        else backend.Gemm(a, b, c, m, n, k);
    }

    /// <summary>
    /// Every schedule that lowers the shape writes every output within the fp32 bound of the fp64 reference, on ragged
    /// shapes in each layout and batched; the unsplit implicit-GEMM tiles lower all of them.
    /// </summary>
    [SkippableTheory]
    [InlineData("nn", 1, 5, 33, 17)]
    [InlineData("nn", 1, 130, 70, 45)]
    [InlineData("tb", 1, 96, 40, 33)]
    [InlineData("ta", 1, 64, 48, 1100)]
    [InlineData("ta", 1, 16, 10, 4096)]
    [InlineData("nn", 3, 40, 24, 20)]
    // Vector gathers (operands contiguous along a multiple of 4) with ragged tile edges and a partial last chunk.
    [InlineData("nn", 1, 100, 68, 36)]
    [InlineData("tb", 1, 72, 36, 52)]
    [InlineData("nn", 2, 36, 20, 24)]
    public void EveryScheduleMatchesTheReference(string layout, int batch, int m, int n, int k)
    {
        using var backend = OpenBackendOrSkip();
        var family = FamilyOf(layout);
        float[] a = Fill(batch * m * k, 1), b = Fill(batch * k * n, 2);
        double[] expected = Reference(family, a, b, batch, m, n, k);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(batch * m * n);
        var lowered = new List<string>();
        var schedules = CudaBackend.DeclaredGemmSchedules;
        for (int s = 0; s < schedules.Count; s++)
        {
            backend.Fill(bc, float.NaN, batch * m * n);
            if (!backend.TryDeclaredGemmSchedule(family, ba, bb, bc, batch, m, n, k, s)) continue;
            backend.Synchronize();
            double error = MaxError(expected, backend.DownloadBuffer(bc));
            Assert.True(error <= Bound(k), $"{schedules[s].Name}: error {error:E3} over {Bound(k):E3}");
            lowered.Add(schedules[s].Name);
        }
        Assert.Contains("implicit-gemm.m64n64", lowered);
        Assert.Contains("implicit-gemm.m128n64", lowered);
    }

    /// <summary>
    /// The four entry points with the route on: the shape goes through the GEMM slot (one decision recorded, whichever
    /// candidate won), and the result is within the fp32 bound of fp64, as is the route-off result.
    /// </summary>
    [SkippableTheory]
    [InlineData("nn", 1, 512, 512, 16)]
    [InlineData("nn", 1, 33, 7, 5)]
    [InlineData("tb", 1, 256, 64, 64)]
    [InlineData("ta", 1, 16, 10, 4096)]
    [InlineData("nn", 4, 64, 64, 32)]
    public void TheEntryPointsRouteThroughTheGemmSlot(string layout, int batch, int m, int n, int k)
    {
        using var backend = OpenBackendOrSkip();
        TunedKernelMode? previousMode = TunedKernelPolicy.ModeOverride;
        bool previousPersist = TunedKernelPolicy.PersistDecisions;
        double previousBudget = TunedKernelPolicy.BudgetMilliseconds;
        TunedKernelPolicy.ModeOverride = TunedKernelMode.Tune;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.BudgetMilliseconds = 60_000;
        try
        {
            var family = FamilyOf(layout);
            float[] a = Fill(batch * m * k, 3), b = Fill(batch * k * n, 4);
            double[] expected = Reference(family, a, b, batch, m, n, k);
            using var ba = backend.AllocateBuffer(a);
            using var bb = backend.AllocateBuffer(b);
            using var bc = backend.AllocateBuffer(batch * m * n);
            foreach (bool declared in new[] { false, true })
            {
                backend.DeclaredKernelsSwitch = declared;
                try
                {
                    backend.Fill(bc, float.NaN, batch * m * n);
                    Entry(backend, family, ba, bb, bc, batch, m, n, k);
                    backend.Synchronize();
                    double error = MaxError(expected, backend.DownloadBuffer(bc));
                    Assert.True(error <= Bound(k), $"route {(declared ? "on" : "off")}: error {error:E3}");
                }
                finally
                {
                    backend.DeclaredKernelsSwitch = null;
                }
            }
            int layoutCode = family == DeclaredFamily.GemmTransposedA ? 1 : family == DeclaredFamily.GemmTransposedB ? 2 : 0;
            var decision = Assert.Single(backend.GemmSlot.CachedDecisions, d =>
                d.Shape[0] == layoutCode && d.Shape[1] == batch && d.Shape[2] == m && d.Shape[3] == n && d.Shape[4] == k);
            Assert.Equal(TunedKernelOp.Gemm, decision.Op);
        }
        finally
        {
            TunedKernelPolicy.ModeOverride = previousMode;
            TunedKernelPolicy.PersistDecisions = previousPersist;
            TunedKernelPolicy.BudgetMilliseconds = previousBudget;
        }
    }

    /// <summary>
    /// With the slot serving a shape by a declared schedule (forced here by the slot's override), the fused GEMM + bias
    /// + ReLU call runs that schedule's fused program and stays bit-identical to Gemm, BiasAdd and Relu run separately
    /// -- the IGemmBiasActivationBackend contract -- including a NaN row the ReLU passes through.
    /// </summary>
    [SkippableTheory]
    [InlineData("implicit-gemm.m64n64", 2048, 64, 64, true)]
    // A split schedule too: there the bias and the ReLU are the COMBINE pass's epilogue
    // (CodegenSplitReduction keeps only the product operands on the partial), so the bit-identity claim rests on a
    // different placement than the unsplit cases above and was not covered. The shape is the one
    // ADeclaredGemmCapturesAndReplaysBitIdentically already lowers under this schedule.
    [InlineData("implicit-gemm.split8v.m64n64", 64, 48, 2048, true)]
    [InlineData("implicit-gemm.m128n64", 300, 4, 8, true)]
    [InlineData("implicit-gemm.m128n64", 512, 512, 16, false)]
    public void TheFusedBiasActivationStaysBitIdenticalUnderADeclaredSchedule(string scheduleName, int m, int n, int k,
        bool relu)
    {
        using var backend = OpenBackendOrSkip();
        bool previousPersist = TunedKernelPolicy.PersistDecisions;
        TunedKernelPolicy.PersistDecisions = false;
        TunedKernelPolicy.SetOverride(TunedKernelOp.Gemm, "ptx.gemm." + scheduleName);
        try
        {
            float[] a = Fill(m * k, 9), w = Fill(k * n, 10), bias = Fill(n, 11);
            a[k + 1] = float.NaN;   // row 1 of the output is NaN throughout
            using var aBuf = backend.AllocateBuffer(a);
            using var wBuf = backend.AllocateBuffer(w);
            using var biasBuf = backend.AllocateBuffer(bias);
            using var fused = backend.AllocateBuffer(m * n);
            using var separate = backend.AllocateBuffer(m * n);
            var activation = relu ? AiDotNet.Tensors.Engines.FusedActivationType.ReLU
                : AiDotNet.Tensors.Engines.FusedActivationType.None;

            long before = backend.DeclaredProgramLaunches;
            Assert.True(((IGemmBiasActivationBackend)backend)
                .TryGemmBiasActivation(aBuf, wBuf, biasBuf, fused, m, n, k, activation), $"declined {m}x{n}x{k}");
            backend.Gemm(aBuf, wBuf, separate, m, n, k);
            Assert.Equal(2, backend.DeclaredProgramLaunches - before);   // the fused program, then the GEMM's
            backend.BiasAdd(separate, biasBuf, separate, m, n);
            if (relu) backend.Relu(separate, separate, m * n);

            var got = backend.DownloadBuffer(fused);
            var expected = backend.DownloadBuffer(separate);
            Assert.True(float.IsNaN(got[n]), "the NaN row did not stay NaN");
            for (int i = 0; i < got.Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(got[i]) ==
                    TestHelpers.MathCompat.SingleToInt32Bits(expected[i]),
                    $"element {i}: fused {got[i]} != gemm + bias{(relu ? " + relu" : "")} {expected[i]}");
        }
        finally
        {
            TunedKernelPolicy.SetOverride(TunedKernelOp.Gemm, null);
            TunedKernelPolicy.PersistDecisions = previousPersist;
        }
    }

    /// <summary>
    /// A declared GEMM first met inside a capture launches nothing (the caller's own path serves it); once built outside,
    /// it captures, and the replay is bit-identical to the eager launch.
    /// </summary>
    [SkippableTheory]
    [InlineData("implicit-gemm.m64n64")]
    [InlineData("implicit-gemm.split8v.m64n64")]
    public void ADeclaredGemmCapturesAndReplaysBitIdentically(string scheduleName)
    {
        using var backend = OpenBackendOrSkip();
        const int m = 64, n = 48, k = 2048;
        int schedule = ScheduleIndex(scheduleName);
        float[] a = Fill(k * m, 5), b = Fill(k * n, 6);
        using var ba = backend.AllocateBuffer(a);
        using var bb = backend.AllocateBuffer(b);
        using var bc = backend.AllocateBuffer(m * n);
        bool ran = true;
        IntPtr unbuilt = backend.CaptureGraph(() =>
            ran = backend.TryDeclaredGemmSchedule(DeclaredFamily.GemmTransposedA, ba, bb, bc, 1, m, n, k, schedule));
        backend.DestroyCapturedGraph(unbuilt);
        Assert.False(ran, "a program first met under capture must not be built there");

        Assert.True(backend.TryDeclaredGemmSchedule(DeclaredFamily.GemmTransposedA, ba, bb, bc, 1, m, n, k, schedule));
        backend.Synchronize();
        float[] eager = backend.DownloadBuffer(bc);
        Assert.True(MaxError(Reference(DeclaredFamily.GemmTransposedA, a, b, 1, m, n, k), eager) <= Bound(k));
        backend.Fill(bc, float.NaN, m * n);
        IntPtr graph = backend.CaptureGraph(() =>
            Assert.True(backend.TryDeclaredGemmSchedule(DeclaredFamily.GemmTransposedA, ba, bb, bc, 1, m, n, k, schedule)));
        try
        {
            backend.LaunchCapturedGraph(graph);
            backend.Synchronize();
            float[] replayed = backend.DownloadBuffer(bc);
            for (int i = 0; i < eager.Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(eager[i]) ==
                    TestHelpers.MathCompat.SingleToInt32Bits(replayed[i]), $"[{i}]: eager {eager[i]} replay {replayed[i]}");
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    /// <summary>
    /// A GEMM chain -- h = x W1, y = h W2^T, dW = x^T y -- as one resident program is bit-identical to the same stages
    /// launched separately, in one cooperative launch.
    /// </summary>
    [SkippableTheory]
    [InlineData("implicit-gemm.m64n64", 64, 96, 80)]
    [InlineData("implicit-gemm.m64n64", 33, 17, 5)]
    [InlineData("planner", 33, 17, 5)]
    public void GemmStagesRunAsOneResidentLaunchBitIdenticalToSeparateLaunches(string scheduleName, int rows, int inner,
        int outer)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.SupportsResidentProgram, "A declared resident program needs a cooperative launch.");
        int schedule = ScheduleIndex(scheduleName);
        float[] x = Fill(rows * inner, 7), w1 = Fill(inner * outer, 8), w2 = Fill(inner * outer, 9);
        using var barrier = backend.AllocateResidentBarrier();
        var results = new float[2][][];
        for (int mode = 0; mode < 2; mode++)
        {
            using var bx = backend.AllocateBuffer(x);
            using var bw1 = backend.AllocateBuffer(w1);
            using var bw2 = backend.AllocateBuffer(w2);
            using var h = backend.AllocateBuffer(rows * outer);
            using var y = backend.AllocateBuffer(rows * inner);
            using var dw = backend.AllocateBuffer(inner * inner);
            var stages = new[]
            {
                backend.GemmStage(DeclaredFamily.Gemm, bx, bw1, h, 1, rows, outer, inner, schedule),
                backend.GemmStage(DeclaredFamily.GemmTransposedB, h, bw2, y, 1, rows, inner, outer, schedule),
                backend.GemmStage(DeclaredFamily.GemmTransposedA, bx, y, dw, 1, inner, inner, rows, schedule),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? 3 : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[] { backend.DownloadBuffer(h), backend.DownloadBuffer(y), backend.DownloadBuffer(dw) };
        }
        double[] hRef = Reference(DeclaredFamily.Gemm, x, w1, 1, rows, outer, inner);
        Assert.True(MaxError(hRef, results[0][0]) <= Bound(inner));
        for (int o = 0; o < results[0].Length; o++)
            for (int i = 0; i < results[0][o].Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(results[0][o][i]) ==
                    TestHelpers.MathCompat.SingleToInt32Bits(results[1][o][i]),
                    $"buffer {o}[{i}]: separate {results[0][o][i]} resident {results[1][o][i]}");
    }
}
