using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// The FP32 6×16 machine-code PANEL kernel (loops the Nr=16 tiles internally in one call)
/// must be BIT-IDENTICAL to invoking the proven single-tile managed kernel njr times — same
/// packed layouts, same FMA reduction order, disjoint C columns. Validates the hand-emitted
/// outer N-loop (register reset of A/kc, natural B-stripe advance, C += 64 bytes/tile).
/// </summary>
public class MachineKernelPanelTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5);
        return a;
    }

    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(4, 1)]
    [InlineData(1, 8)]
    [InlineData(7, 5)]
    [InlineData(256, 16)]
    [InlineData(3, 257)]
    public void Panel_BitIdentical_ToSingleTileLoop(int kc, int njr)
    {
        Skip.IfNot(MachineKernelGemm.IsFp32PanelAvailable && Avx2Fp32_6x16.IsSupported,
            "FP32 panel machine kernel / AVX2 unavailable on this host.");

        const int Mr = 6, Nr = 16;
        // packedA: [kc, Mr] row-major. packedB: [njr stripes, kc, Nr].
        var packedA = Rand(kc * Mr, 11);
        var packedB = Rand(njr * kc * Nr, 22);
        int n = njr * Nr;
        int ldc = n;

        // Reference: loop the single-tile managed kernel over the njr stripes.
        var cRef = new float[Mr * ldc]; // zeroed (fresh C)
        for (int jr = 0; jr < njr; jr++)
        {
            Avx2Fp32_6x16.Run(
                packedA.AsSpan(),
                packedB.AsSpan(jr * kc * Nr, kc * Nr),
                cRef.AsSpan(jr * Nr),   // tile at column jr*16, row-stride ldc
                ldc, kc);
        }

        // Panel: one call processes all njr tiles.
        var cPanel = new float[Mr * ldc];
        MachineKernelGemm.RunPanelFp32(packedA, packedB, cPanel, ldc, kc, njr);

        for (int i = 0; i < cRef.Length; i++)
            if (cRef[i] != cPanel[i])
                Assert.Fail($"panel diverged at [{i / ldc},{i % ldc}] kc={kc} njr={njr}: ref={cRef[i]:R} panel={cPanel[i]:R}");
    }
}
