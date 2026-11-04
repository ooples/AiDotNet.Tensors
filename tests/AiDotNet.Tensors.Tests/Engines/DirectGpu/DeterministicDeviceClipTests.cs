using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using N = AiDotNet.Tensors.Engines.Gpu.DeterministicGradientNorm;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The device global-norm clip on every backend matches <see cref="DeterministicGradientNorm"/> bit for bit: the
/// record (sum, norm, coefficient, non-finite counts) and every clipped gradient element, including overflow-range
/// norms, subnormal gradients and non-finite elements.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeterministicDeviceClipTests
{
    private static float[] Random(int n, int seed, int minExp, int maxExp, double subnormalShare = 0)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (rng.NextDouble() < subnormalShare)
                a[i] = N.UInt32BitsToSingle((uint)rng.Next(1, 0x7FFFFF) | ((uint)rng.Next(2) << 31));
            else
                a[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(2, rng.Next(minExp, maxExp)));
        }
        return a;
    }

    public static IEnumerable<object[]> Cases()
    {
        yield return new object[] { "pythagoras", new[] { new[] { 3f, 4f } }, 1f };
        yield return new object[] { "1e20", new[] { new[] { 3e20f, -4e20f } }, 1f };
        yield return new object[] { "past-float-range", new[] { new[] { 2.4e38f, 3.2e38f } }, 1f };
        yield return new object[] { "subnormal", new[] { Random(5000, 1, -149, -126, 0.7) }, 1e-40f };
        yield return new object[] { "mixed", new[] { Random(10000, 2, -30, 30), Random(4097, 3, -60, 10, 0.01), Random(3, 4, -5, 5), Random(70000, 5, -20, 64) }, 0.5f };
        yield return new object[] { "unclipped", new[] { Random(1000, 6, -20, -10) }, 10f };
        yield return new object[] { "zeros", new[] { new float[9000] }, 1f };
        yield return new object[] { "nan", new[] { new[] { 1f, 2f }, new[] { 3f, float.NaN, float.PositiveInfinity } }, 1f };
        yield return new object[] { "inf", new[] { Random(5000, 7, -3, 3), new[] { float.NegativeInfinity, 1f } }, 1f };
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void DeviceClip_MatchesTheIntegerReferenceBitForBit(string name, float[][] host, float maxNorm)
    {
        foreach (var policy in new[] { NonFiniteGradientPolicy.LeaveUnscaled, NonFiniteGradientPolicy.Propagate,
                     NonFiniteGradientPolicy.ZeroNonFinite, NonFiniteGradientPolicy.Error })
            DeviceClip(name, host, maxNorm, policy);
    }

    private static void DeviceClip(string name, float[][] host, float maxNorm, NonFiniteGradientPolicy policy)
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var gradients = host.Select(h => gpu.TensorAdd(new Tensor<float>(new[] { h.Length }, new Vector<float>(h)),
                new Tensor<float>(new[] { h.Length }))).ToArray();
            // The reference reads what the device holds (an engine op may already have flushed subnormals).
            var before = gradients.Select(g => g.ToArray()).ToArray();
            bool nonFinite = before.Any(t => t.Any(x => float.IsNaN(x) || float.IsInfinity(x)));
            if (policy == NonFiniteGradientPolicy.Error && nonFinite)
            {
                var ex = Assert.Throws<NonFiniteGradientException>(() => GpuOptimizer.TryClipGradNorm(gradients, maxNorm, policy));
                Assert.Contains("gradient " + Array.FindIndex(before, t => t.Any(x => float.IsNaN(x) || float.IsInfinity(x))), ex.Message);
                for (int t = 0; t < gradients.Length; t++)
                    Assert.Equal(before[t].Select(N.SingleToUInt32Bits), gradients[t].ToArray().Select(N.SingleToUInt32Bits));
                return;
            }
            uint[] expected = N.Compute(before, maxNorm, policy == NonFiniteGradientPolicy.Error ? NonFiniteGradientPolicy.LeaveUnscaled : policy);
            bool zero = policy == NonFiniteGradientPolicy.ZeroNonFinite;

            using var norm = GpuOptimizer.TryClipGradNorm(gradients, maxNorm, policy);
            Assert.NotNull(norm);
            Assert.Equal(BitConverter.DoubleToInt64Bits(Double(expected)), BitConverter.DoubleToInt64Bits(norm!.TotalNormSquared));
            Assert.Equal(expected[N.WordNorm], N.SingleToUInt32Bits(norm.Norm));
            Assert.Equal(expected[N.WordScale], N.SingleToUInt32Bits(norm.ClipCoefficient));
            Assert.Equal(expected[N.WordNanCount], (uint)norm.NanCount);
            Assert.Equal(expected[N.WordInfCount], (uint)norm.InfinityCount);
            Assert.Equal(expected[N.WordFirstNonFiniteTensor] == uint.MaxValue ? -1 : (int)expected[N.WordFirstNonFiniteTensor],
                norm.FirstNonFiniteGradient);
            Assert.Equal(expected[N.WordNonFinite] == 0, norm.IsFinite);

            uint scale = expected[N.WordScale];
            for (int t = 0; t < gradients.Length; t++)
            {
                var after = gradients[t].ToArray();
                for (int i = 0; i < after.Length; i++)
                {
                    uint want = N.ScaleElement(N.SingleToUInt32Bits(before[t][i]), scale, zero);
                    uint got = N.SingleToUInt32Bits(after[i]);
                    if (want != got)
                        Assert.Fail($"{name}/{policy}: gradient {t}[{i}] = {before[t][i]:R} * {N.UInt32BitsToSingle(scale):R}: device 0x{got:X8}, reference 0x{want:X8}");
                }
            }
        }
        finally { AiDotNetEngine.Current = saved; }
    }

    /// <summary>More chunks than the fused launch has blocks (each block grid-strides), run three times: the last block's
    /// ticket must reset, so every launch combines all partials once. Then the record-only path (no coefficient).</summary>
    [SkippableFact]
    public void FusedNormClip_GridStrideAndRepeatedLaunches_MatchTheReference()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "GPU engine unavailable.");
        var saved = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            // Five 4M-element tensors and a 4097-element one: 5122 chunks. (Each tensor stays under 65535 * 256
            // elements, so an elementwise op keeps it on the device even where the X workgroup count is 65535.)
            var host = Enumerable.Range(0, 5).Select(k => Random(4 << 20, 8 + k, -40, 40)).Append(Random(4097, 13, -10, 10)).ToArray();
            for (int run = 0; run < 3; run++)
            {
                var gradients = host.Select(h => gpu.TensorAdd(new Tensor<float>(new[] { h.Length }, new Vector<float>(h)),
                    new Tensor<float>(new[] { h.Length }))).ToArray();
                var before = gradients.Select(g => g.ToArray()).ToArray();
                uint[] expected = N.Compute(before, 0.5f);
                using (var sum = GpuOptimizer.TryGradientSumOfSquares(gradients))
                {
                    Assert.NotNull(sum);
                    Assert.Equal(BitConverter.DoubleToInt64Bits(Double(expected)), BitConverter.DoubleToInt64Bits(sum!.TotalNormSquared));
                    Assert.Equal(expected[N.WordNorm], N.SingleToUInt32Bits(sum.Norm));
                }
                using var norm = GpuOptimizer.TryClipGradNorm(gradients, 0.5f);
                Assert.NotNull(norm);
                Assert.Equal(expected[N.WordNorm], N.SingleToUInt32Bits(norm!.Norm));
                Assert.Equal(expected[N.WordScale], N.SingleToUInt32Bits(norm.ClipCoefficient));
                for (int t = 0; t < gradients.Length; t++)
                {
                    var after = gradients[t].ToArray();
                    for (int i = 0; i < after.Length; i += after.Length > 4097 ? 4099 : 1)
                        Assert.Equal(N.Mul(N.SingleToUInt32Bits(before[t][i]), expected[N.WordScale]), N.SingleToUInt32Bits(after[i]));
                }
            }
        }
        finally { AiDotNetEngine.Current = saved; }
    }

    private static double Double(uint[] r) =>
        BitConverter.Int64BitsToDouble((long)(((ulong)r[N.WordSumSquaresHi] << 32) | r[N.WordSumSquaresLo]));
}

#if NET7_0_OR_GREATER
/// <summary>The WebGPU port of the deterministic clip, on Dawn (it only runs where AIDOTNET_WEBGPU_NODE_DIR is set):
/// the same record and scaled bits as <see cref="DeterministicGradientNorm"/>.</summary>
[Collection("DirectGpuSerial")]
public sealed class DeterministicWebGpuClipTests
{
    [SkippableTheory]
    [MemberData(nameof(DeterministicDeviceClipTests.Cases), MemberType = typeof(DeterministicDeviceClipTests))]
    public void WebGpuClip_MatchesTheIntegerReferenceBitForBit(string name, float[][] host, float maxNorm)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        CheckBackend(backend, name, host, maxNorm);
    }

    internal static void CheckBackend(IDirectGpuBackend backend, string name, float[][] host, float maxNorm)
    {
        var multi = (IMultiTensorKernels)backend;
        foreach (var policy in new[] { NonFiniteGradientPolicy.LeaveUnscaled, NonFiniteGradientPolicy.Propagate,
                     NonFiniteGradientPolicy.ZeroNonFinite })
        {
            bool zero = policy == NonFiniteGradientPolicy.ZeroNonFinite;
            var buffers = host.Select(h => backend.AllocateBuffer(h)).ToArray();
            var sizes = host.Select(h => h.Length).ToArray();
            using var record = backend.AllocateBuffer(N.RecordWords);
            using var scale = backend.AllocateBuffer(1);
            try
            {
                multi.MultiTensorSumOfSquares(buffers, sizes, record);
                multi.ClipScaleFromRecord(record, maxNorm, scale, policy);
                multi.MultiTensorClipScale(buffers, sizes, scale, zero);
                uint[] expected = N.Compute(host, maxNorm, policy);
                uint[] got = MultiTensorArgs.RecordFromSlots(backend.DownloadBuffer(record));
                for (int w = 0; w <= N.WordMaxNorm; w++)
                    Assert.True(expected[w] == got[w], $"{name}/{policy}: record word {w}: device 0x{got[w]:X8}, reference 0x{expected[w]:X8}");
                for (int t = 0; t < buffers.Length; t++)
                {
                    var after = backend.DownloadBuffer(buffers[t]);
                    for (int i = 0; i < host[t].Length; i++)
                    {
                        uint want = N.ScaleElement(N.SingleToUInt32Bits(host[t][i]), expected[N.WordScale], zero);
                        uint gotBits = N.SingleToUInt32Bits(after[i]);
                        if (want != gotBits) Assert.Fail($"{name}/{policy}: {t}[{i}]: device 0x{gotBits:X8}, reference 0x{want:X8}");
                    }
                }
            }
            finally { foreach (var b in buffers) b.Dispose(); }
        }
    }
}
#endif
