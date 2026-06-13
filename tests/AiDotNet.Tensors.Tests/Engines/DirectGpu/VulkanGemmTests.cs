// Copyright (c) AiDotNet. All rights reserved.
// Tests for the real Vulkan GPU GEMM (replacing the CPU download/loop/upload
// fallback) and the FP16 IGpuHalfPrecisionBackend path (issue #560).

// System.Half / BitConverter.Int16BitsToHalf (the FP16 oracle) are .NET 6+.
#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Verifies the Vulkan GPU GEMM on real hardware: the FP32 path (now a GLSL
/// compute shader, previously a CPU fallback) and the FP16 mixed-precision path
/// (<see cref="IGpuHalfPrecisionBackend.GemmFp16In32fOut"/> / <see cref="IGpuHalfPrecisionBackend.Hgemm"/>).
/// <para>
/// The FP16 oracle downloads the packed-FP16 operands and decodes each half with
/// <see cref="BitConverter.Int16BitsToHalf"/> — the same IEEE binary16 decode the
/// GPU's <c>unpackHalf2x16</c> performs — so the only residual error is FP32
/// accumulation order. Honors <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c> (throws instead
/// of skipping when a GPU is required). Requires libshaderc for the FP16 path.
/// </para>
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class VulkanGemmTests
{
    private readonly VulkanBackend _backend;
    private readonly bool _ready;
    private readonly bool _fp16Ready;

    public VulkanGemmTests()
    {
        try
        {
            _backend = VulkanBackend.Instance;
            // Both the FP32 and FP16 GPU GEMM paths are runtime-compiled GLSL
            // compute shaders, so they only actually run on the GPU when
            // libshaderc is present. Without it Gemm transparently falls back to
            // the CPU loop — gating on IsGlslCompilerAvailable keeps these tests
            // honest (they verify the GPU path or skip, never silently CPU-pass).
            _ready = _backend.Initialize() && _backend.IsGlslCompilerAvailable;
            _fp16Ready = _ready && _backend.SupportsHgemm;
        }
        catch
        {
            _ready = false;
            _fp16Ready = false;
        }
    }

    private bool EnsureReady(bool needFp16)
    {
        bool ok = needFp16 ? _fp16Ready : _ready;
        if (ok) return true;
        if (string.Equals(
                Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"),
                "1",
                StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the Vulkan GPU GEMM " +
                "was unavailable (Vulkan device + libshaderc for runtime GLSL→SPIR-V are required).");
        }

        return false;
    }

    private static float[] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return data;
    }

    private static float[] CpuReference(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int l = 0; l < k; l++)
                    acc += a[i * k + l] * b[l * n + j];
                c[i * n + j] = acc;
            }
        }

        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            double tol = absTol + relTol * Math.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"GEMM mismatch at [{i}]: expected {expected[i]}, got {actual[i]} (|diff|={diff}, tol={tol}).");
        }
    }

    // Decode the exact half values held in a packed-FP16 buffer (two halves per
    // float word, little-endian) — matches the GPU's unpackHalf2x16 exactly.
    private float[] DecodePackedHalf(IGpuBuffer fp16Buffer, int count)
    {
        var packed = _backend.DownloadBuffer(fp16Buffer);
        var bytes = new byte[packed.Length * 4];
        Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
        var values = new float[count];
        for (int e = 0; e < count; e++)
        {
            ushort bits = (ushort)(bytes[e * 2] | (bytes[e * 2 + 1] << 8));
            values[e] = (float)BitConverter.Int16BitsToHalf(unchecked((short)bits));
        }

        return values;
    }

    // ---- FP32 GPU GEMM (now a real compute shader, not a CPU fallback) ----

    [SkippableTheory]
    [InlineData(16, 16, 16)]
    [InlineData(64, 48, 96)]
    [InlineData(37, 53, 71)]
    [InlineData(128, 128, 256)]
    public void Gemm_Fp32_MatchesCpuReference(int m, int n, int k)
    {
        Skip.If(!EnsureReady(needFp16: false), "Vulkan not available on this system.");

        var a = RandomMatrix(m, k, seed: 11 + m + k);
        var b = RandomMatrix(k, n, seed: 22 + n + k);
        var expected = CpuReference(a, b, m, n, k);

        using var aBuf = _backend.AllocateBuffer(a);
        using var bBuf = _backend.AllocateBuffer(b);
        using var cBuf = _backend.AllocateBuffer(m * n);
        _backend.Gemm(aBuf, bBuf, cBuf, m, n, k);
        var actual = _backend.DownloadBuffer(cBuf);

        AssertClose(expected, actual, absTol: 1e-3, relTol: 1e-4);
    }

    // ---- FP16 mixed-precision GEMM ----

    [SkippableTheory]
    [InlineData(16, 16, 16)]
    [InlineData(64, 48, 96)]
    [InlineData(37, 53, 71)]
    [InlineData(128, 128, 256)]
    public void GemmFp16In32fOut_MatchesCpuReference(int m, int n, int k)
    {
        Skip.If(!EnsureReady(needFp16: true), "Vulkan FP16 GEMM (libshaderc) not available on this system.");

        var a = RandomMatrix(m, k, seed: 1234 + m + k);
        var b = RandomMatrix(k, n, seed: 5678 + n + k);

        using var aFp32 = _backend.AllocateBuffer(a);
        using var bFp32 = _backend.AllocateBuffer(b);
        using var aFp16 = _backend.AllocateBuffer((m * k + 1) / 2);
        using var bFp16 = _backend.AllocateBuffer((k * n + 1) / 2);
        _backend.ConvertToFp16(aFp32, aFp16, m * k);
        _backend.ConvertToFp16(bFp32, bFp16, k * n);

        // Exact operands the GPU will multiply (decoded from the packed buffers).
        var aTrunc = DecodePackedHalf(aFp16, m * k);
        var bTrunc = DecodePackedHalf(bFp16, k * n);
        var expected = CpuReference(aTrunc, bTrunc, m, n, k);

        using var cBuf = _backend.AllocateBuffer(m * n);
        ((IGpuHalfPrecisionBackend)_backend).GemmFp16In32fOut(aFp16, bFp16, cBuf, m, n, k);
        var actual = _backend.DownloadBuffer(cBuf);

        AssertClose(expected, actual, absTol: 1e-2, relTol: 1e-2);
    }

    [SkippableFact]
    public void GemmFp16In32fOut_RejectsNonPositiveDimensions()
    {
        Skip.If(!EnsureReady(needFp16: true), "Vulkan FP16 GEMM not available on this system.");

        using var dummy = _backend.AllocateBuffer(4);
        var half = (IGpuHalfPrecisionBackend)_backend;
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 0, 4, 4));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 4, -1, 4));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => half.GemmFp16In32fOut(dummy, dummy, dummy, 4, 4, 0));
    }
}

#endif
