#if NETCOREAPP
using System;
using System.Linq;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1265: cuBLAS is optional. With cuBLAS forced off, every public GEMM entry point that used to call cuBLAS runs the
/// generated kernels instead and reproduces its cuBLAS result (taken on the same backend with cuBLAS on): FP32 in every
/// transpose form, batched, on a caller's stream, and FP16 / BF16 / FP64. The dispatch counter proves the fallback, not
/// cuBLAS, produced each result.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CublasOptionalTests
{
    private readonly ITestOutputHelper _output;

    public CublasOptionalTests(ITestOutputHelper output) => _output = output;

    private static CudaBackend RequireBackend()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable || !backend.IsCublasAvailable)
        {
            backend.Dispose();
            Skip.If(true, "Needs CUDA with cuBLAS installed, to compare the fallback against.");
        }
        return backend;
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    // Halves (or doubles) packed bit-for-bit into a float buffer, padded to whole floats.
    private static IGpuBuffer HalfBuffer(CudaBackend backend, float[] values)
    {
        var halves = values.Select(v => (Half)v).ToArray();
        var packed = new float[(halves.Length + 1) / 2];
        MemoryMarshal.Cast<Half, byte>(halves.AsSpan()).CopyTo(MemoryMarshal.Cast<float, byte>(packed.AsSpan()));
        return backend.AllocateBuffer(packed);
    }

    private static float[] DownloadHalves(CudaBackend backend, IGpuBuffer buffer, int count) =>
        MemoryMarshal.Cast<float, Half>(backend.DownloadBuffer(buffer)).Slice(0, count).ToArray().Select(h => (float)h).ToArray();

    private static IGpuBuffer DoubleBuffer(CudaBackend backend, double[] values) =>
        backend.AllocateBuffer(MemoryMarshal.Cast<double, float>(values).ToArray());

    private static double[] DownloadDoubles(CudaBackend backend, IGpuBuffer buffer, int count) =>
        MemoryMarshal.Cast<float, double>(backend.DownloadBuffer(buffer)).Slice(0, count).ToArray();

    private static double Relative(float[] expected, float[] actual) =>
        Relative(expected.Select(v => (double)v).ToArray(), actual.Select(v => (double)v).ToArray());

    private static double Relative(double[] expected, double[] actual)
    {
        double largest = 1e-300, worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            largest = Math.Max(largest, Math.Abs(expected[i]));
            double difference = Math.Abs(actual[i] - expected[i]);
            worst = Math.Max(worst, double.IsNaN(difference) ? double.PositiveInfinity : difference);
        }
        return worst / largest;
    }

    // Runs `run` with cuBLAS on, then off, from the same initial output, and returns both outputs; asserts the second
    // went through the fallback.
    private (T[] Cublas, T[] Fallback) Both<T>(CudaBackend backend, Func<IGpuBuffer> freshOutput,
        Action<IGpuBuffer> run, Func<IGpuBuffer, T[]> read, string what)
    {
        backend.CublasEnabledOverride = true;
        T[] cublas;
        using (var c = freshOutput())
        {
            run(c);
            cublas = read(c);
        }
        backend.CublasEnabledOverride = false;
        long before = backend.FallbackGemmDispatchCount;
        T[] fallback;
        using (var c = freshOutput())
        {
            run(c);
            fallback = read(c);
        }
        Assert.True(backend.FallbackGemmDispatchCount > before, $"{what}: the fallback did not run");
        backend.CublasEnabledOverride = null;
        return (cublas, fallback);
    }

    [SkippableFact]
    public void Fp32Entries_ReproduceCublas_WithoutIt()
    {
        using var backend = RequireBackend();
        bool previous = AiDotNetEngine.DeterministicMode;
        try
        {
            // Strict FP32 on both sides, so the fallback is compared like for like.
            AiDotNetEngine.SetDeterministicMode(true);
            foreach (var (m, n, k) in new[] { (67, 129, 95), (128, 256, 64), (1, 300, 33) })
            {
                float[] hostC = Values(m * n, 9);
                using var a = backend.AllocateBuffer(Values(m * k, 1));
                using var b = backend.AllocateBuffer(Values(k * n, 2));
                using var bT = backend.AllocateBuffer(Values(n * k, 3));   // B stored N x K for MatMulTransposed
                using var aT = backend.AllocateBuffer(Values(k * m, 4));   // A stored K x M for MatMulTransposedA
                Func<IGpuBuffer> output = () => backend.AllocateBuffer(hostC);
                float[] Read(IGpuBuffer c) => backend.DownloadBuffer(c);
                void Expect((float[] Cublas, float[] Fallback) r, string what)
                {
                    double relative = Relative(r.Cublas, r.Fallback);
                    _output.WriteLine($"{what} M{m} N{n} K{k}: {relative:G3}");
                    Assert.True(relative <= 1e-4, $"{what} M{m} N{n} K{k}: fallback differs from cuBLAS by {relative:G3}");
                }

                backend.GemmRoutingEnabled = false;   // keep routing out of it: this compares cuBLAS and the fallback
                Expect(Both(backend, output, c => backend.Gemm(a, b, c, m, n, k, 0.5f, 2f), Read, "Gemm"), "Gemm");
                Expect(Both(backend, output, c => backend.MatMulTransposed(a, bT, c, m, n, k, 1.5f, -1f), Read, "MatMulTransposed"),
                    "MatMulTransposed");
                Expect(Both(backend, output, c => backend.MatMulTransposedA(aT, b, c, m, n, k), Read, "MatMulTransposedA"),
                    "MatMulTransposedA");
                Expect(Both(backend, output, c => backend.GemmAsync(a, b, c, m, n, k, 1f, 0.5f, backend.DefaultStream), c =>
                {
                    backend.Synchronize();
                    return Read(c);
                }, "GemmAsync"), "GemmAsync");
            }

            const int batch = 3, bm = 40, bn = 36, bk = 52;
            using var ba = backend.AllocateBuffer(Values(batch * bm * bk, 5));
            using var bb = backend.AllocateBuffer(Values(batch * bk * bn, 6));
            var batched = Both(backend, () => backend.AllocateBuffer(Values(batch * bm * bn, 7)),
                c => backend.BatchedGemm(ba, bb, c, bm, bn, bk, batch, -1.5f, 0.25f), c => backend.DownloadBuffer(c), "BatchedGemm");
            Assert.True(Relative(batched.Cublas, batched.Fallback) <= 1e-4);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previous);
        }
    }

    [SkippableFact]
    public void UseCublasFalse_RunsTheGeneratedKernels_EvenWithCublasInstalled()
    {
        using var backend = RequireBackend();
        backend.GemmRoutingEnabled = false;
        const int m = 33, n = 65, k = 17;
        using var a = backend.AllocateBuffer(Values(m * k, 1));
        using var b = backend.AllocateBuffer(Values(k * n, 2));
        using var c = backend.AllocateBuffer(m * n);
        var prior = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        try
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
                new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { UseCublas = false });
            long before = backend.FallbackGemmDispatchCount;
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(before + 1, backend.FallbackGemmDispatchCount);

            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
                new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { UseCublas = true });
            backend.Gemm(a, b, c, m, n, k);
            Assert.Equal(before + 1, backend.FallbackGemmDispatchCount);   // cuBLAS again
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(prior);
        }
    }

    [SkippableFact]
    public void HalfAndDoubleEntries_ReproduceCublas_WithoutIt()
    {
        using var backend = RequireBackend();
        Skip.IfNot(backend.SupportsHgemm, "Needs FP16 GEMM support.");
        const int m = 48, n = 72, k = 40;
        float[] hostA = Values(m * k, 1), hostB = Values(k * n, 2);
        using var a16 = HalfBuffer(backend, hostA);
        using var b16 = HalfBuffer(backend, hostB);

        // FP16 in / FP32 out: both accumulate in FP32.
        var fp32Out = Both(backend, () => backend.AllocateBuffer(m * n),
            c => backend.GemmFp16In32fOut(a16, b16, c, m, n, k), c => backend.DownloadBuffer(c), "GemmFp16In32fOut");
        Assert.True(Relative(fp32Out.Cublas, fp32Out.Fallback) <= 1e-4);

        var gemmFp16 = Both(backend, () => backend.AllocateBuffer(Values(m * n, 3)),
            c => backend.GemmFp16(a16, b16, c, m, n, k, 0.5f, 2f), c => backend.DownloadBuffer(c), "GemmFp16");
        Assert.True(Relative(gemmFp16.Cublas, gemmFp16.Fallback) <= 1e-4);

        // FP16 out: one FP16 rounding each side (cuBLAS's Hgemm also accumulates in FP16).
        var hgemm = Both(backend, () => HalfBuffer(backend, new float[m * n]),
            c => backend.Hgemm(a16, b16, c, m, n, k), c => DownloadHalves(backend, c, m * n), "Hgemm");
        double hgemmRelative = Relative(hgemm.Cublas, hgemm.Fallback);
        _output.WriteLine($"Hgemm: {hgemmRelative:G3}");
        Assert.True(hgemmRelative <= 1e-2);

        var halfOut = Both(backend, () => HalfBuffer(backend, Values(m * n, 4)),
            c => backend.GemmFp16HalfOut(a16, b16, c, m, n, k, 0.5f, 2f), c => DownloadHalves(backend, c, m * n), "GemmFp16HalfOut");
        Assert.True(Relative(halfOut.Cublas, halfOut.Fallback) <= 5e-3);

        // FP64 batched fanout.
        const int batch = 2;
        var random = new Random(11);
        double[] da = Enumerable.Range(0, batch * m * k).Select(_ => random.NextDouble() - 0.5).ToArray();
        double[] db = Enumerable.Range(0, batch * k * n).Select(_ => random.NextDouble() - 0.5).ToArray();
        using var a64 = DoubleBuffer(backend, da);
        using var b64 = DoubleBuffer(backend, db);
        using var streamPool = new AiDotNet.Tensors.Engines.Gpu.GpuStreamPool(backend);
        using var scheduler = new AiDotNet.Tensors.Engines.Gpu.GpuStreamScheduler(streamPool);
        var dgemm = Both(backend, () => DoubleBuffer(backend, new double[batch * m * n]),
            c => backend.BatchedDgemmFanout(a64, b64, c, m, n, k, batch, scheduler, 1.25, 0),
            c => DownloadDoubles(backend, c, batch * m * n), "BatchedDgemmFanout");
        double dgemmRelative = Relative(dgemm.Cublas, dgemm.Fallback);
        _output.WriteLine($"BatchedDgemmFanout: {dgemmRelative:G3}");
        Assert.True(dgemmRelative <= 1e-12);
    }
}
#endif
