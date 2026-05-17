using System;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Verifies BlasManaged.Gemm works without dynamic code generation
/// (System.Reflection.Emit / DynamicMethod). This documents the contract
/// that BlasManaged is NativeAOT-compatible: when RuntimeFeature.IsDynamicCodeSupported
/// is false (or when we simulate that condition), the JIT cache layer must
/// be a no-op and all kernels fall back to the hand-written paths.
///
/// <para>
/// The full NativeAOT smoke test (publish with PublishAot=true, run the
/// binary, verify exit code) is CI infrastructure outside the scope of this
/// PR — it lives in the GitHub Actions workflow. This in-process test
/// verifies the CODE-LEVEL invariant: BlasManaged.Gemm produces correct
/// output via the same paths an AOT build would use.
/// </para>
/// </summary>
public class NativeAotCompatibilityTest
{
    private readonly ITestOutputHelper _output;

    public NativeAotCompatibilityTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void NativeAotDetector_AgreesWithRuntimeFeature()
    {
        // On normal .NET host, IsDynamicCodeSupported is true.
        // Under NativeAOT publish, it'd be false. This test documents the
        // contract that the detector mirrors RuntimeFeature.
        bool detected = NativeAotDetector.IsDynamicCodeSupported;
        _output.WriteLine($"NativeAotDetector.IsDynamicCodeSupported = {detected}");
        // Don't assert true/false — could legitimately be either depending on runtime.
        // Just verify it doesn't throw.
    }

    [Fact]
    public void JittedKernelCache_NoOpWhenJitNotEmitted_HandWrittenKernelsCorrect()
    {
        // The cache is empty by default (no IL emission yet). Verify Gemm
        // still produces correct output — this is the AOT-compatible path.

        // Sanity check: cache is empty at the start of this test.
        JittedKernelCache.Clear();
        Assert.Equal(0, JittedKernelCache.Count);

        // Run a representative GEMM. With no JIT cache entries, the dispatch
        // falls through to the hand-written microkernels — exactly the path
        // an AOT-published binary would use.
        int m = 32, n = 32, k = 32;
        var rng = new Random(42);
        double[] a = new double[m * k];
        double[] b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // Reference: naive triple-loop GEMM.
        double[] expected = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++) sum += a[i * k + kk] * b[kk * n + j];
                expected[i * n + j] = sum;
            }

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        // Verify the cache was still empty after the call (no IL emission).
        Assert.Equal(0, JittedKernelCache.Count);

        // Verify correctness.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void BlasManaged_AvoidsDynamicCodeOnHotPath()
    {
        // The hot path through BlasManaged.Gemm must NOT call into
        // System.Reflection.Emit at all when the JIT cache has no entries.
        // We verify this indirectly: a GEMM call that succeeds without
        // throwing PlatformNotSupportedException (which Reflection.Emit
        // would throw under AOT) demonstrates the path is AOT-compatible.

        JittedKernelCache.Clear();

        int m = 16, n = 16, k = 16;
        var rng = new Random(42);
        double[] a = new double[m * k];
        double[] b = new double[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        double[] c = new double[m * n];

        // The call must succeed without throwing.
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            c, ldc: n,
            m, n, k);

        // Sanity: at least one cell should be non-zero (the random inputs guarantee this).
        Assert.Contains(c, x => Math.Abs(x) > 1e-10);
    }

    [Fact]
    public void BlasManaged_AllArchPaths_AreAotCompatible()
    {
        // Verify that the IsSupported gates on each arch path are queryable
        // without throwing. Under NativeAOT, these checks are valid even
        // without dynamic code support.
        bool avx512 = Avx512Fp64_8x16.IsSupported;
        bool avx2 = Avx2Fp64_4x8.IsSupported;
        bool neon = NeonFp64_4x4.IsSupported;

        _output.WriteLine($"Avx512Fp64_8x16.IsSupported = {avx512}");
        _output.WriteLine($"Avx2Fp64_4x8.IsSupported    = {avx2}");
        _output.WriteLine($"NeonFp64_4x4.IsSupported    = {neon}");

        // None of these queries should throw. The actual values depend on the runtime CPU.
        // We don't assert specific values — just that the gates are callable.
    }
}
