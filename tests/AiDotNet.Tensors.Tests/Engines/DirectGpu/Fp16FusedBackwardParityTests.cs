// Copyright (c) AiDotNet. All rights reserved.
// Correctness tests for the multi-backend fused FP16 matmul BACKWARD (#633):
// IGpuHalfPrecisionBackend.MatMulBackwardFp16Fused computes both gradients of
// C[M,N] = A[M,K]·B[K,N] — gradA[M,K] = gradC·Bᵀ and gradB[K,N] = Aᵀ·gradC — in
// two FP16-in/FP32-accumulate GEMMs with no materialized transpose. These tests
// validate the GENERALIZATION off CUDA onto every backend by checking each
// available backend's fused backward against a CPU reference computed from the
// SAME FP16-rounded inputs, so the only residual error is FP32 accumulation order.

// System.Half (the FP16 oracle) only exists on .NET 5+, and the FP16 GPU path is
// exercised only on the modern TFM anyway; net471 builds skip this file.
#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Validates <see cref="IGpuHalfPrecisionBackend.MatMulBackwardFp16Fused"/> on every GPU backend that ships it.
/// Each test instantiates one backend, skips when its fused backward is unavailable on this system (unless
/// <c>AIDOTNET_REQUIRE_GPU_TESTS=1</c>, which turns the skip into a loud failure), and compares the fused
/// gradients to a CPU oracle computed from FP16-rounded inputs. This box (AMD RDNA1) runs OpenCL + Vulkan;
/// the CUDA/HIP/Metal/WebGpu cases exercise the same contract on hardware that has them.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class Fp16FusedBackwardParityTests
{
    /// <summary>The backends whose fused FP16 backward is validated here. Each maps to a parameterless ctor.</summary>
    public enum BackendKind { OpenCL, Vulkan }

    public static TheoryData<BackendKind> Backends => new() { BackendKind.OpenCL, BackendKind.Vulkan };

    /// <summary>
    /// Acquires a ready backend, honoring each backend's lifecycle: OpenCL is a disposable per-test instance;
    /// Vulkan is a shared singleton (<see cref="VulkanBackend.Instance"/>) that must be <c>Initialize()</c>d and
    /// must NOT be disposed. <paramref name="dispose"/> is the correct teardown for the acquired backend (a no-op
    /// for the Vulkan singleton). Returns null when the backend's fused FP16 backward is unavailable here.
    /// </summary>
    private static (IGpuHalfPrecisionBackend? half, IDirectGpuBackend? backend, Action dispose, Exception? error) TryCreate(BackendKind kind)
    {
        try
        {
            switch (kind)
            {
                case BackendKind.OpenCL:
                {
                    var b = new OpenClBackend();
                    if (b is IGpuHalfPrecisionBackend half && b.IsAvailable && half.SupportsFp16FusedBackward)
                        return (half, b, b.Dispose, null);
                    b.Dispose();
                    return (null, null, () => { }, null);
                }
                case BackendKind.Vulkan:
                {
                    var b = VulkanBackend.Instance;
                    if (b.Initialize() && ((IGpuHalfPrecisionBackend)b).SupportsFp16FusedBackward)
                        return ((IGpuHalfPrecisionBackend)b, b, () => { }, null); // singleton: do not dispose
                    return (null, null, () => { }, null);
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind));
            }
        }
        catch (Exception ex)
        {
            return (null, null, () => { }, ex);
        }
    }

    private static bool RequireGpu =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);

    // Round-trip through IEEE-754 binary16 exactly as the GPU does, so the CPU oracle and the GPU kernel
    // start from identical FP16 inputs.
    private static float ToFp16AndBack(float value) => (float)(System.Half)value;

    private static float[] RandomMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1)
        return data;
    }

    // gradA[M,K] = gradC[M,N] · Bᵀ : gradA[m,kk] = Σ_n gradC[m,n]·B[kk,n]  (B stored [K,N]).
    private static float[] CpuGradA(float[] gradC, float[] b, int m, int n, int k)
    {
        var g = new float[m * k];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                float acc = 0f;
                for (int j = 0; j < n; j++)
                    acc += ToFp16AndBack(gradC[i * n + j]) * ToFp16AndBack(b[p * n + j]);
                g[i * k + p] = acc;
            }
        return g;
    }

    // gradB[K,N] = Aᵀ · gradC[M,N] : gradB[kk,n] = Σ_m A[m,kk]·gradC[m,n]  (A stored [M,K]).
    private static float[] CpuGradB(float[] a, float[] gradC, int m, int n, int k)
    {
        var g = new float[k * n];
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int i = 0; i < m; i++)
                    acc += ToFp16AndBack(a[i * k + p]) * ToFp16AndBack(gradC[i * n + j]);
                g[p * n + j] = acc;
            }
        return g;
    }

    // Upload an FP32 matrix and convert to an FP16 device buffer (the same ConvertToFp16 the engine's
    // ResolveToFp16 makes). Over-allocates the FP16 buffer as floats (4 bytes ≥ the 2 bytes/elem it needs),
    // matching the existing FP16 GEMM tests.
    private static IGpuBuffer UploadFp16(IDirectGpuBackend backend, float[] data, int elems)
    {
        using var fp32 = backend.AllocateBuffer(data);
        var fp16 = backend.AllocateBuffer(elems);
        backend.ConvertToFp16(fp32, fp16, elems);
        return fp16;
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol, string what)
    {
        // actual may be longer than expected: AllocateBuffer pools/rents, so the result buffer can have
        // capacity > the requested element count, and DownloadBuffer returns that capacity. The valid result
        // is the first expected.Length contiguous (row-major) elements; the pooled tail is stale.
        Assert.True(actual.Length >= expected.Length,
            $"{what}: downloaded {actual.Length} elements, fewer than the expected {expected.Length}.");
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(expected[i] - actual[i]);
            double tol = absTol + relTol * Math.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"{what} mismatch at [{i}]: expected {expected[i]}, got {actual[i]} (|diff|={diff}, tol={tol}).");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void MatMulBackwardFp16Fused_MatchesCpuReference(BackendKind kind)
    {
        var (half, backend, dispose, error) = TryCreate(kind);
        if (backend is null || half is null)
        {
            if (RequireGpu)
                throw new InvalidOperationException(
                    $"GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the {kind} backend's " +
                    "fused FP16 backward was unavailable.", error);
            Skip.If(true, $"{kind} fused FP16 backward not available on this system.");
            return;
        }

        try
        {
            // A few shapes incl. ragged (out-of-range tile guards) and a deeper K.
            foreach (var (m, n, k) in new[] { (16, 16, 16), (64, 48, 96), (37, 53, 71), (128, 96, 160) })
            {
                var gradC = RandomMatrix(m, n, seed: 11 + m + n);
                var a = RandomMatrix(m, k, seed: 23 + m + k);
                var b = RandomMatrix(k, n, seed: 37 + k + n);

                var expectedGradA = CpuGradA(gradC, b, m, n, k);
                var expectedGradB = CpuGradB(a, gradC, m, n, k);

                var hGradC = UploadFp16(backend, gradC, m * n);
                var hA = UploadFp16(backend, a, m * k);
                var hB = UploadFp16(backend, b, k * n);
                using var gradAOut = backend.AllocateBuffer(m * k);
                using var gradBOut = backend.AllocateBuffer(k * n);
                try
                {
                    half.MatMulBackwardFp16Fused(hGradC, hA, hB, gradAOut, gradBOut, m, n, k, gradOutHalf: false);
                    var actualGradA = backend.DownloadBuffer(gradAOut);
                    var actualGradB = backend.DownloadBuffer(gradBOut);
                    AssertClose(expectedGradA, actualGradA, absTol: 1e-2, relTol: 1e-2, $"{kind} gradA [{m}x{n}x{k}]");
                    AssertClose(expectedGradB, actualGradB, absTol: 1e-2, relTol: 1e-2, $"{kind} gradB [{m}x{n}x{k}]");
                }
                finally
                {
                    hGradC.Dispose();
                    hA.Dispose();
                    hB.Dispose();
                }
            }
        }
        finally
        {
            dispose();
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void MatMulBackwardFp16Fused_RejectsNonPositiveDimensions(BackendKind kind)
    {
        var (half, backend, dispose, error) = TryCreate(kind);
        if (backend is null || half is null)
        {
            if (RequireGpu)
                throw new InvalidOperationException(
                    $"GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the {kind} backend's " +
                    "fused FP16 backward was unavailable.", error);
            Skip.If(true, $"{kind} fused FP16 backward not available on this system.");
            return;
        }

        try
        {
            using var dummy = backend.AllocateBuffer(16);
            Assert.Throws<ArgumentOutOfRangeException>(
                () => half.MatMulBackwardFp16Fused(dummy, dummy, dummy, dummy, dummy, 0, 4, 4, false));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => half.MatMulBackwardFp16Fused(dummy, dummy, dummy, dummy, dummy, 4, -1, 4, false));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => half.MatMulBackwardFp16Fused(dummy, dummy, dummy, dummy, dummy, 4, 4, 0, false));
        }
        finally
        {
            dispose();
        }
    }
}

#endif
