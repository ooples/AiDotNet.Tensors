using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): end-to-end verification that the machine-code 6×8 microkernel
/// path wired into <see cref="BlasManagedLib.Gemm{T}"/> produces correct results
/// for qualifying FP64 shapes (Auto pack-mode), matching both the managed
/// ForcePackBoth strategy and a naive triple-loop reference.
/// </summary>
[Collection("BlasManaged-Perf-Serial")]
public class MachineKernelGemmTests
{
    private readonly ITestOutputHelper _output;
    public MachineKernelGemmTests(ITestOutputHelper output) { _output = output; }

    private static bool IsX64Windows =>
        RuntimeInformation.OSArchitecture == Architecture.X64
        && RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    private static void ReferenceGemm(
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb,
        Span<double> c, int ldc, int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[i * lda + p] * b[p * ldb + j];
                c[i * ldc + j] = sum;
            }
    }

    [Theory]
    [InlineData(6, 8, 4)]       // single tile, single Kc
    [InlineData(6, 8, 256)]     // single tile, exactly one Kc block
    [InlineData(6, 8, 500)]     // single tile, multi-Kc (crosses the 256 block boundary)
    [InlineData(48, 64, 128)]   // many tiles
    [InlineData(192, 768, 768)] // ViT-class
    [InlineData(384, 1024, 512)]
    [InlineData(12, 1024, 333)] // odd K, wide N
    public void Gemm_MachineKernel_Auto_Matches_PackBoth_And_Reference(int M, int N, int K)
    {
        // On non-x64-Windows the machine path no-ops (managed fallback); the test
        // still validates the dispatch wiring is correct (Auto == ForcePackBoth).
        Assert.True(M % 6 == 0 && N % 8 == 0, "test shapes must qualify (m%6==0, n%8==0)");

        var rng = new Random(1234 + M + N + K);
        var a = new double[M * K];
        var b = new double[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        var cAuto = new double[M * N];     // Auto → machine kernel (when qualified)
        var cPackBoth = new double[M * N]; // managed reference strategy
        var cRef = new double[M * N];

        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cAuto, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.Auto });
        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cPackBoth, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });
        ReferenceGemm(a, K, b, N, cRef, N, M, N, K);

        double maxVsPackBoth = 0, maxVsRef = 0, scale = 0;
        for (int i = 0; i < cAuto.Length; i++)
        {
            maxVsPackBoth = Math.Max(maxVsPackBoth, Math.Abs(cAuto[i] - cPackBoth[i]));
            maxVsRef = Math.Max(maxVsRef, Math.Abs(cAuto[i] - cRef[i]));
            scale = Math.Max(scale, Math.Abs(cRef[i]));
        }
        // Both paths do FMA in K-order; differences are pure float-association
        // (block boundaries), so the tolerance is tiny relative to the magnitude.
        double tol = 1e-9 * Math.Max(1.0, scale) * K;
        Assert.True(maxVsPackBoth < tol,
            $"M={M} N={N} K={K}: Auto vs PackBoth drift {maxVsPackBoth:G6} > {tol:G6}");
        Assert.True(maxVsRef < tol,
            $"M={M} N={N} K={K}: Auto vs reference drift {maxVsRef:G6} > {tol:G6}");
    }

    [Fact]
    public void Gemm_MachineKernel_Disabled_StillCorrect()
    {
        // Flipping the master switch off must produce identical results (managed path).
        const int M = 48, N = 64, K = 256;
        var rng = new Random(7);
        var a = new double[M * K];
        var b = new double[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var cOn = new double[M * N];
        var cOff = new double[M * N];

        MachineKernelGemm.Enabled = true;
        BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cOn, N, M, N, K,
            new BlasOptions<double> { PackingMode = PackingMode.Auto });
        try
        {
            MachineKernelGemm.Enabled = false;
            BlasManagedLib.Gemm<double>(a, K, false, b, N, false, cOff, N, M, N, K,
                new BlasOptions<double> { PackingMode = PackingMode.Auto });
        }
        finally { MachineKernelGemm.Enabled = true; }

        double maxDelta = 0, scale = 0;
        for (int i = 0; i < cOn.Length; i++)
        {
            maxDelta = Math.Max(maxDelta, Math.Abs(cOn[i] - cOff[i]));
            scale = Math.Max(scale, Math.Abs(cOff[i]));
        }
        Assert.True(maxDelta < 1e-9 * Math.Max(1.0, scale) * K,
            $"machine-on vs machine-off drift {maxDelta:G6}");
    }

    [Fact]
    public void Gemm_MachineKernel_EndToEnd_Perf()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;
        if (!IsX64Windows || !Avx2.IsSupported || !Fma.IsSupported) return;

        const int M = 1536, N = 1536, K = 1536; // aligned (m%6==0, n%8==0), big enough to be compute-bound
        var rng = new Random(42);
        var a = new double[M * K];
        var b = new double[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        var c = new double[M * N];
        double gflopWork = 2.0 * M * N * K / 1e9;

        double Run(PackingMode mode, bool machine)
        {
            MachineKernelGemm.Enabled = machine;
            var opt = new BlasOptions<double> { PackingMode = mode };
            for (int i = 0; i < 2; i++) BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K, opt);
            double best = double.MaxValue;
            for (int w = 0; w < 5; w++)
            {
                var sw = Stopwatch.StartNew();
                BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K, opt);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
            }
            return gflopWork / best;
        }

        try
        {
            double managed = Run(PackingMode.Auto, machine: false);
            double machine = Run(PackingMode.Auto, machine: true);
            _output.WriteLine($"end-to-end DGEMM {M}x{N}x{K} (multithreaded): " +
                $"managed-strategy {managed:F1} GFLOPS, machine-kernel {machine:F1} GFLOPS " +
                $"({machine / managed:F2}×)");
        }
        finally { MachineKernelGemm.Enabled = true; }
    }
}
