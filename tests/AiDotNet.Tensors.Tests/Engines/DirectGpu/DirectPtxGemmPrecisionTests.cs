#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The generated GEMM's non-FP32 precisions (#1265, the cuBLAS-free fallback): FP16 operands with FP32 or FP16 output,
/// and FP64, each against an fp64 oracle over every layout, odd and edge shapes, alpha/beta, batches, a NaN-filled C with
/// beta = 0 and K = 0, with zero local memory. They are register-staged and load scalars.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DirectPtxGemmPrecisionTests
{
    private static readonly PtxGemmLayout[] Layouts =
    {
        new(false, false), new(true, false), new(false, true), new(true, true),
    };

    // Scalar, register-staged: small and large tiles, single and double buffered, split and swizzled.
    private static readonly PtxGemmConfiguration[] Configurations =
    {
        new(32, 32, 8, 4, 4, 1, 0, false),
        new(64, 128, 8, 4, 8, 1, 4, true),
        new(128, 128, 8, 8, 8, 1, 4, true, true, 8),
    };

    private static readonly (int M, int N, int K)[] Shapes =
    {
        (1, 1, 1), (3, 5, 7), (127, 255, 129), (64, 64, 64), (1, 300, 200), (300, 1, 200), (40, 44, 300),
    };

    private static readonly PtxGemmPrecision[] Precisions =
    {
        PtxGemmPrecision.Fp16InFp32Out, PtxGemmPrecision.Fp16InFp16Out, PtxGemmPrecision.Fp64,
    };

    private static DirectPtxRuntime RequireRuntime()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7)
        {
            runtime.Dispose();
            Skip.If(true, "Needs compute capability 7.0 or newer.");
        }
        return runtime;
    }

    [Fact]
    public void NonFp32Precisions_RequireScalarRegisterStagedConfigurations()
    {
        foreach (var precision in Precisions)
        {
            foreach (var config in Configurations) Assert.Null(precision.Invalidity(config));
            Assert.NotNull(precision.Invalidity(new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 4, true)));                 // vectors
            Assert.NotNull(precision.Invalidity(new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 4, false, false, 1, 1, 2))); // cp.async
            foreach (var layout in Layouts)
            {
                string ptx = PtxSgemmGenerator.Emit(Configurations[2], layout, 8, 6, precision);
                Assert.Contains(".entry " + PtxSgemmGenerator.EntryPoint, ptx);
                Assert.Contains(precision == PtxGemmPrecision.Fp64 ? "fma.rn.f64" : "fma.rn.f32", ptx);
            }
        }
        // FP64 doubles the shared tile: a configuration at the FP32 limit no longer fits.
        Assert.NotNull(PtxGemmPrecision.Fp64.Invalidity(new PtxGemmConfiguration(256, 128, 16, 8, 8, 1, 4, true)));
        Assert.Throws<ArgumentException>(() => PtxSgemmGenerator.Emit(
            new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 4, true), Layouts[0], 8, 6, PtxGemmPrecision.Fp64));
    }

    // Inputs exactly representable in the operand type, so the oracle sees what the kernel sees.
    private static double[] Values(int count, int seed, PtxGemmPrecision precision)
    {
        var random = new Random(seed);
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            double value = random.NextDouble() * 2 - 1;
            values[i] = precision == PtxGemmPrecision.Fp64 ? value : (double)(Half)value;
        }
        return values;
    }

    private static void Upload(DirectPtxBuffer buffer, double[] values, int bytes)
    {
        if (bytes == 8) buffer.Upload<double>(values);
        else if (bytes == 4) buffer.Upload<float>(values.Select(v => (float)v).ToArray());
        else buffer.Upload<Half>(values.Select(v => (Half)v).ToArray());
    }

    private static double[] Download(DirectPtxBuffer buffer, int count, int bytes)
    {
        if (bytes == 8)
        {
            var values = new double[count];
            buffer.Download<double>(values);
            return values;
        }
        if (bytes == 4)
        {
            var values = new float[count];
            buffer.Download<float>(values);
            return values.Select(v => (double)v).ToArray();
        }
        var halves = new Half[count];
        buffer.Download<Half>(halves);
        return halves.Select(v => (double)v).ToArray();
    }

    private static void Check(DirectPtxRuntime runtime, PtxSgemmKernel kernel, int batch, int m, int n, int k,
        double alpha, double beta, bool poisonC)
    {
        var layout = kernel.Layout;
        PtxGemmPrecision precision = kernel.Precision;
        int ob = precision.OperandBytes(), outb = precision.OutputBytes();
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n, ldc = n;
        double[] a = Values(Math.Max(1, batch * m * k), m * 31 + k, precision);
        double[] b = Values(Math.Max(1, batch * k * n), n * 17 + k, precision);
        double[] c = poisonC
            ? Enumerable.Repeat(double.NaN, batch * m * n).ToArray()
            : Values(batch * m * n, m + n, outb == 2 ? PtxGemmPrecision.Fp16InFp16Out : PtxGemmPrecision.Fp64);
        using var da = runtime.AllocateBytes((nuint)(a.Length * ob));
        using var db = runtime.AllocateBytes((nuint)(b.Length * ob));
        using var dc = runtime.AllocateBytes((nuint)(c.Length * outb));
        Upload(da, a, ob);
        Upload(db, b, ob);
        Upload(dc, c, outb);
        // The oracle sees C as stored (an FP32 C rounds the FP64 test values).
        double[] cStored = Download(dc, c.Length, outb);
        if (precision == PtxGemmPrecision.Fp64)
            kernel.LaunchDouble(da.Pointer, db.Pointer, dc.Pointer, m, n, k, lda, ldb, ldc, alpha, beta, batch,
                (long)m * k, (long)k * n, (long)m * n);
        else
            kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, lda, ldb, ldc, (float)alpha, (float)beta, batch,
                (long)m * k, (long)k * n, (long)m * n);
        runtime.Synchronize();
        double[] actual = Download(dc, c.Length, outb);

        // FP32 arithmetic over K terms; an FP16 output also rounds once (2^-11 relative); FP64 is near-exact.
        double tolerance = precision switch
        {
            PtxGemmPrecision.Fp64 => 1e-12,
            PtxGemmPrecision.Fp16InFp16Out => 1e-3,
            _ => 2e-5,
        };
        string what = $"{kernel.Configuration.Name} {layout.Name} {precision} b{batch} M{m} N{n} K{k} alpha{alpha} beta{beta}";
        for (int z = 0; z < batch; z++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0, magnitude = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double av = layout.TransposeA ? a[z * m * k + p * lda + i] : a[z * m * k + i * lda + p];
                        double bv = layout.TransposeB ? b[z * k * n + j * ldb + p] : b[z * k * n + p * ldb + j];
                        sum += av * bv;
                        magnitude += Math.Abs(av * bv);
                    }
                    int index = z * m * n + i * n + j;
                    double expected = alpha * sum + (beta == 0 ? 0 : beta * cStored[index]);
                    double scale = Math.Abs(alpha) * magnitude + Math.Abs(beta * (beta == 0 ? 0 : cStored[index])) + 1e-30;
                    if (precision == PtxGemmPrecision.Fp16InFp16Out) scale = Math.Max(scale, Math.Abs(expected));
                    Assert.True(Math.Abs(actual[index] - expected) <= tolerance * Math.Max(scale, 1e-3),
                        $"{what}: [{index}] got {actual[index]}, expected {expected}");
                }
    }

    [SkippableFact]
    public void EveryPrecision_MatchesTheOracle_OverLayoutsShapesScalarsAndBatches()
    {
        using var runtime = RequireRuntime();
        foreach (var precision in Precisions)
            foreach (var config in Configurations)
                foreach (var layout in Layouts)
                {
                    using var kernel = new PtxSgemmKernel(runtime, config, layout, precision);
                    Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
                    foreach (var (m, n, k) in Shapes)
                    {
                        Check(runtime, kernel, 1, m, n, k, 1, 0, poisonC: false);
                        Check(runtime, kernel, 1, m, n, k, 0.5, 2, poisonC: false);
                    }
                    Check(runtime, kernel, 3, 40, 36, 52, -1.5, 0.25, poisonC: false);
                    Check(runtime, kernel, 1, 64, 64, 64, 1, 0, poisonC: true);   // beta = 0 never reads C
                    Check(runtime, kernel, 1, 64, 64, 0, 1, 3, poisonC: false);   // K = 0: C = beta * C
                }
    }

    [SkippableFact]
    public void Fp32KernelRefusesLaunchDouble_AndFp64RefusesLaunch()
    {
        using var runtime = RequireRuntime();
        using var fp32 = new PtxSgemmKernel(runtime, Configurations[0], Layouts[0]);
        using var fp64 = new PtxSgemmKernel(runtime, Configurations[0], Layouts[0], PtxGemmPrecision.Fp64);
        Assert.Throws<InvalidOperationException>(() =>
            fp32.LaunchDouble(IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, 1, 1, 1, 1, 1, 1, 0));
        Assert.Throws<InvalidOperationException>(() =>
            fp64.Launch(IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, 1, 1, 1, 1, 1, 1f, 0f));
    }
}
#endif
