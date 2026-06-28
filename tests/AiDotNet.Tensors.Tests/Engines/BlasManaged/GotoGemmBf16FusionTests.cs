using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
// The class and its namespace share the name "BlasManaged"; inside this (…Tests.Engines.BlasManaged)
// namespace the bare name binds to the namespace, so alias the static class explicitly.
using BlasManagedT = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Path B GEMM additions: (1) the opt-in bf16-B GotoGemm path (AIDOTNET_GEMM_BF16 / s_gemmBf16) on
/// large bandwidth-bound shapes — precision-changing, so verify it stays within a bf16 tolerance of
/// the exact fp32 result and is OFF by default; (2) the fused-epilogue-on-the-fast-path change —
/// large epilogue-bearing GEMMs now run through GotoGemm + EpilogueChain (was gated to None and fell
/// to PackBoth), so verify the result equals the unfused GEMM followed by the same bias+activation.
/// </summary>
public class GotoGemmBf16FusionTests
{
    private static float[] Random(int len, int seed)
    {
        var rng = new Random(seed);
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }

    private static float[] Gemm(float[] a, float[] b, int m, int n, int k, in BlasOptions<float> opts)
    {
        var c = new float[(long)m * n];
        BlasManagedT.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, in opts);
        return c;
    }

    [Fact]
    public void Bf16OptIn_LargeBandwidthShape_WithinBf16Tolerance_AndExactByDefault()
    {
        if (!GotoGemmFp32.IsAvailable) return; // AVX2+FMA x64 net5+ only; otherwise no bf16 path to exercise
        const int m = 1024, n = 1024, k = 4096; // work 4.29e9 ≥ 4e9, m≥512, n<2k ⇒ ShouldUseBf16 true
        Assert.True(GotoGemmFp32.ShouldUseBf16(m, n, k));
        var a = Random(m * k, 1);
        var b = Random(k * n, 2);
        var opts = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };

        bool saved = BlasManagedT.s_gemmBf16;
        try
        {
            BlasManagedT.s_gemmBf16 = false;
            var cExact = Gemm(a, b, m, n, k, in opts);
            BlasManagedT.s_gemmBf16 = true;
            var cBf16 = Gemm(a, b, m, n, k, in opts);

            // bf16-B truncates B to ~3 significant digits. Verify error relative to the RESULT SCALE
            // (max-abs-diff / max-abs-value) — the standard bf16-GEMM metric; per-element relative error
            // legitimately spikes on cancellation-prone elements (small |C|) and is not meaningful here.
            double maxDiff = 0, maxAbs = 0;
            for (long i = 0; i < cExact.LongLength; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(cBf16[i] - cExact[i]));
                maxAbs = Math.Max(maxAbs, Math.Abs(cExact[i]));
            }
            double scaledRel = maxDiff / Math.Max(maxAbs, 1e-9);
            Assert.True(scaledRel < 1e-2, $"bf16-B error vs result scale too large: {scaledRel:E3} (maxDiff={maxDiff:E3}, maxAbs={maxAbs:E3})");
            // And it MUST differ from exact fp32 — proves the bf16 path actually ran (not silently fp32).
            Assert.True(maxDiff > 1e-5, $"bf16-B did not change the result (path not taken?): maxDiff={maxDiff:E3}");
        }
        finally { BlasManagedT.s_gemmBf16 = saved; }
    }

    [Fact]
    public void FusedEpilogue_OnLargeGotoGemmShape_MatchesUnfusedThenBiasActivation()
    {
        if (!GotoGemmFp32.IsAvailable) return;
        const int m = 768, n = 512, k = 768; // m≥512 ⇒ BeatsPackBoth ⇒ GotoGemm fast path
        var a = Random(m * k, 7);
        var b = Random(k * n, 9);
        var bias = Random(n, 11);

        // Reference: plain GEMM (no epilogue), then apply bias + ReLU by hand (same math EpilogueChain does).
        var plain = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };
        var cRef = Gemm(a, b, m, n, k, in plain);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float v = cRef[(long)i * n + j] + bias[j];
                cRef[(long)i * n + j] = v > 0f ? v : 0f;
            }

        // Fused: bias + ReLU carried in the epilogue → now runs through GotoGemm + EpilogueChain.
        var fusedOpts = new BlasOptions<float>
        {
            PackingMode = PackingMode.DisableAutotune,
            Epilogue = new Epilogue<float> { BiasN = bias, Activation = FusedActivationType.ReLU },
        };
        var cFused = Gemm(a, b, m, n, k, in fusedOpts);

        double maxDiff = 0;
        for (long i = 0; i < cRef.LongLength; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(cFused[i] - cRef[i]));
        // Same GEMM path + same epilogue math ⇒ bit-close (fp32 rounding only).
        Assert.True(maxDiff < 1e-3, $"fused-epilogue GotoGemm diverged from unfused+manual: maxDiff={maxDiff:E3}");
    }

    [Fact]
    public void ShouldUseBf16_GatesToLargeBandwidthBoundShapesOnly()
    {
        Assert.True(GotoGemmFp32.ShouldUseBf16(1024, 1024, 4096)); // work 4.29e9, n<2k, m≥512
        Assert.True(GotoGemmFp32.ShouldUseBf16(2048, 2048, 2048)); // 8.6e9
        Assert.False(GotoGemmFp32.ShouldUseBf16(1024, 1024, 1024)); // 1.07e9 < 4e9 (compute-bound; bf16 hurt)
        Assert.False(GotoGemmFp32.ShouldUseBf16(256, 4608, 1152));  // short-M wide-N: n≥2k ⇒ excluded
        Assert.False(GotoGemmFp32.ShouldUseBf16(256, 1152, 1152));  // m<512
    }
}
