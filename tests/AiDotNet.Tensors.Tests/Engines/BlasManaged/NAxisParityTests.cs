using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasMgd = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using PB = AiDotNet.Tensors.Engines.BlasManaged.PackBothStrategy;
using MachineKernelGemm = AiDotNet.Tensors.Engines.BlasManaged.MachineKernelGemm;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The N-axis private-B parallel path (wide-N moderate-K float, panel kernel) must be
/// BIT-IDENTICAL to the M-axis shared-B path — same per-element K-reduction order, jc blocks
/// own disjoint C columns. Toggled via <c>PackBothStrategy.s_disableNAxis</c>.
/// </summary>
// Pinned to the serial GEMM-state collection: this mutates process-wide BlasManaged knobs
// (PB.s_disableNAxis is read by the production GEMM dispatch), so it must not run concurrently
// with sibling tests that issue GEMMs.
[Collection("BlasManaged-Stats-Serial")]
public class NAxisParityTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5);
        return a;
    }

    [SkippableTheory]
    [InlineData(384, 4096, 512)]   // wide-N, m mr-aligned → N-axis fires
    [InlineData(6, 2048, 777)]     // thin-M
    [InlineData(384, 6000, 1024)]  // non-pow2 N, larger K (n >= k, enough N-blocks)
    [InlineData(48, 8192, 256)]    // very wide N
    [InlineData(768, 4096, 512)]   // larger mr-aligned M, wide N
    public void NAxis_BitIdentical_ToMAxis(int m, int n, int k)
    {
        // The N-axis private-B path requires the FP32 6x16 machine-code panel kernel.
        // Exercise PackBothStrategy directly with the matching tile so this route-coverage
        // test is independent of the host's top-level BlasManaged tile heuristic.
        Skip.IfNot(MachineKernelGemm.IsFp32PanelAvailable,
            "FP32 machine-code panel kernel unavailable — N-axis path cannot run on this platform.");

        var prior = CpuParallelSettings.MaxDegreeOfParallelism;
        var priorDisable = PB.s_disableNAxis;
        var priorMachineKernel = MachineKernelGemm.Enabled;
        var priorPanelKernel = MachineKernelGemm.EnablePanelKernel;
        var priorDeterministic = BlasProvider.IsDeterministicMode;
        var priorThreadDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        CpuParallelSettings.MaxDegreeOfParallelism = 4;
        var a = Rand(m * k, 1);
        var b = Rand(k * n, 2);
        var cN = new float[m * n];
        var cM = new float[m * n];
        long nAxisRunsForThisCase;
        try
        {
            MachineKernelGemm.Enabled = true;
            MachineKernelGemm.EnablePanelKernel = true;
            BlasProvider.SetThreadLocalDeterministicMode(null);
            BlasProvider.SetDeterministicMode(false);

            var opts = new BlasOptions<float> { NumThreads = 4 };
            const int kc = 256;
            int mr = MachineKernelGemm.Fp32Mr;
            int nr = MachineKernelGemm.Fp32Nr;

            PB.s_disableNAxis = true;
            PB.Run<float>(a, k, false, b, n, false, cM, n, m, n, k, m, n, kc, mr, nr, opts);

            PB.s_disableNAxis = false;
            long before = System.Threading.Interlocked.Read(ref PB.s_nAxisRunCount);
            PB.Run<float>(a, k, false, b, n, false, cN, n, m, n, k, m, n, kc, mr, nr, opts);
            nAxisRunsForThisCase = System.Threading.Interlocked.Read(ref PB.s_nAxisRunCount) - before;
        }
        finally
        {
            PB.s_disableNAxis = priorDisable;
            MachineKernelGemm.Enabled = priorMachineKernel;
            MachineKernelGemm.EnablePanelKernel = priorPanelKernel;
            BlasProvider.SetDeterministicMode(priorDeterministic);
            BlasProvider.SetThreadLocalDeterministicMode(priorThreadDeterministic);
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }

        // Guard against a vacuous M-vs-M pass: the N-axis path MUST have actually executed for this shape.
        Assert.True(nAxisRunsForThisCase > 0,
            $"N-axis path did not run for m={m} n={n} k={k} — comparison would be vacuous (M-axis vs M-axis).");

        for (int i = 0; i < cM.Length; i++)
            if (cM[i] != cN[i])
                Assert.Fail($"N-axis diverged at [{i / n},{i % n}] m={m} n={n} k={k}: Maxis={cM[i]:R} Naxis={cN[i]:R}");
    }

    [SkippableFact]
    public void Avx512PackBothTile_DoesNotEnterSixBySixteenNAxisPanelPath()
    {
        Skip.IfNot(MachineKernelGemm.IsFp32PanelAvailable && Avx512Fp32_16x16.IsSupported,
            "Requires an AVX-512 FP32 PackBoth host with the FP32 panel kernel available.");

        var prior = CpuParallelSettings.MaxDegreeOfParallelism;
        var priorDisable = PB.s_disableNAxis;
        var priorDisableGoto = BlasMgd.s_disableGotoGemm;
        var priorMachineKernel = MachineKernelGemm.Enabled;
        var priorPanelKernel = MachineKernelGemm.EnablePanelKernel;
        var priorDeterministic = BlasProvider.IsDeterministicMode;
        var priorThreadDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
        CpuParallelSettings.MaxDegreeOfParallelism = 4;
        try
        {
            MachineKernelGemm.Enabled = true;
            MachineKernelGemm.EnablePanelKernel = true;
            BlasProvider.SetThreadLocalDeterministicMode(null);
            BlasProvider.SetDeterministicMode(false);
            BlasMgd.s_disableGotoGemm = true;
            PB.s_disableNAxis = false;

            int m = 384, n = 4096, k = 512;
            var a = Rand(m * k, 11);
            var b = Rand(k * n, 12);
            var c = new float[m * n];
            long before = System.Threading.Interlocked.Read(ref PB.s_nAxisRunCount);

            BlasMgd.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k,
                new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 4 });

            long nAxisRuns = System.Threading.Interlocked.Read(ref PB.s_nAxisRunCount) - before;
            Assert.Equal(0, nAxisRuns);
        }
        finally
        {
            PB.s_disableNAxis = priorDisable;
            BlasMgd.s_disableGotoGemm = priorDisableGoto;
            MachineKernelGemm.Enabled = priorMachineKernel;
            MachineKernelGemm.EnablePanelKernel = priorPanelKernel;
            BlasProvider.SetDeterministicMode(priorDeterministic);
            BlasProvider.SetThreadLocalDeterministicMode(priorThreadDeterministic);
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }
}
