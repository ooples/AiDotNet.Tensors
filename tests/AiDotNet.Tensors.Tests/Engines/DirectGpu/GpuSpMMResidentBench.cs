using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #515: the managed CSR SpMM kernels against cuSPARSE with everything resident on the device, which is what the
/// perf bar is about. <see cref="GpuSpMMBenchHarness"/> times the host-array convenience calls, which allocate,
/// upload, run and download on every call, so on these shapes it measures transfers and allocation and reads
/// 0.0 GF/s for both sides. Here the operands are uploaded once, each arm runs a batch of launches between two
/// device synchronizations, the two arms alternate over several rounds, and each arm keeps its fastest round.
/// Gated like the other harness on <c>AIDOTNET_PERF_RUNNER=1</c>: it needs a quiet machine.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuSpMMResidentBench
{
    private readonly ITestOutputHelper _out;
    public GpuSpMMResidentBench(ITestOutputHelper output) => _out = output;

    [DllImport("nvcuda")] private static extern int cuCtxSynchronize();

    private const int Launches = 200, Rounds = 7;

    [SkippableFact]
    public void Bench_ResidentKernels_CustomVsCuSparse()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine to bench GPU SpMM.");
        Skip.IfNot(CudaSparseBackend.IsAvailable, "CUDA is not available.");
        Skip.IfNot(CuSparseBackend.IsAvailable, "cuSPARSE is not available.");

        using var backend = CudaBackend.CreateOrThrow();
        var rng = new Random(515);
        _out.WriteLine($"{"shape",-30}{"nnz",9}{"kernel",8}{"custom us",11}{"cuSPARSE us",13}{"custom GF/s",13}{"cuSPARSE GF/s",15}{"speedup",9}");
        int wins = 0, compared = 0;
        double maxLoss = 1.0;

        foreach (var s in SpecializedShapeCatalog.SpMM)
        {
            var (rowPtr, colIdx, nnz) = BuildCsr(s.Rows, s.Cols, s.DensityPercent / 100.0, rng);
            var values = new float[nnz];
            var b = new float[s.Cols * s.N];
            for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            using var dValues = backend.AllocateBuffer(values);
            using var dB = backend.AllocateBuffer(b);
            using var dOut = backend.AllocateBuffer(s.Rows * s.N);
            using var dOut2 = backend.AllocateBuffer(s.Rows * s.N);
            using var dRowPtr = Ints(backend, rowPtr);
            using var dColIdx = Ints(backend, colIdx);

            // The kernel CudaSparseBackend.SpMM picks for this width.
            string kernel = s.N % 4 == 0 ? "vec4" : s.N <= 64 ? "warp" : "scalar";
            Action custom = kernel switch
            {
                "vec4" => () => backend.CsrSpMMVec4(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
                "warp" => () => backend.CsrSpMMWarp(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
                _ => () => backend.CsrSpMM(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
            };

            using var cu = new CuSparseSpMM(dValues, dColIdx, dRowPtr, dB, dOut2, s.Rows, s.Cols, s.N, nnz, backend);

            // Both arms must compute the same product before either is timed.
            custom();
            cu.Run();
            Sync(backend);
            float[] a = backend.DownloadBuffer(dOut), c = backend.DownloadBuffer(dOut2);
            for (int i = 0; i < s.Rows * s.N; i++)
                Assert.True(Math.Abs(a[i] - c[i]) <= 1e-3f * Math.Max(1f, Math.Abs(c[i])), $"{s.Name}[{i}]: custom {a[i]}, cuSPARSE {c[i]}");

            double customUs = double.MaxValue, cuUs = double.MaxValue;
            for (int round = 0; round < Rounds; round++)
            {
                customUs = Math.Min(customUs, BatchMicroseconds(backend, custom));
                cuUs = Math.Min(cuUs, BatchMicroseconds(backend, cu.Run));
            }
            double flop = 2.0 * nnz * s.N;
            double speedup = cuUs / customUs;
            compared++;
            if (speedup >= 1.0) wins++; else maxLoss = Math.Max(maxLoss, 1.0 / speedup);
            _out.WriteLine($"{s.Name,-30}{nnz,9}{kernel,8}{customUs,11:F2}{cuUs,13:F2}{flop / (customUs * 1e3),13:F1}{flop / (cuUs * 1e3),15:F1}{speedup,8:F2}x");
        }

        int winRate = compared == 0 ? 0 : wins * 100 / compared;
        _out.WriteLine($"=> custom wins {wins}/{compared} = {winRate}%; max loss {maxLoss:F2}x");
        Assert.True(winRate >= AiDotNet.Tensors.Tests.Engines.BlasManaged.SpecializedPerfBar.GpuSpMMMinWinRatePercent,
            $"GPU SpMM win rate {winRate}% is below the frozen bar {AiDotNet.Tensors.Tests.Engines.BlasManaged.SpecializedPerfBar.GpuSpMMMinWinRatePercent}%.");
        Assert.True(maxLoss <= AiDotNet.Tensors.Tests.Engines.BlasManaged.SpecializedPerfBar.GpuSpMMMaxLossMultiple,
            $"GPU SpMM max loss {maxLoss:F2}x is above the frozen bar {AiDotNet.Tensors.Tests.Engines.BlasManaged.SpecializedPerfBar.GpuSpMMMaxLossMultiple}x.");
    }

    private static void Sync(CudaBackend backend)
    {
        backend.Synchronize();
        Assert.Equal(0, cuCtxSynchronize());   // cuSPARSE's own stream too
    }

    /// <summary>Microseconds per launch over <see cref="Launches"/> back-to-back launches between two syncs.</summary>
    private static double BatchMicroseconds(CudaBackend backend, Action launch)
    {
        Sync(backend);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Launches; i++) launch();
        Sync(backend);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Launches;
    }

    private static IGpuBuffer Ints(CudaBackend backend, int[] values)
    {
        var bytes = new byte[values.Length * sizeof(int)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        var buffer = backend.AllocateByteBuffer(bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static (int[] RowPtr, int[] ColIdx, int Nnz) BuildCsr(int rows, int cols, double density, Random rng)
    {
        var rowPtr = new int[rows + 1];
        var columns = new System.Collections.Generic.List<int>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
                if (rng.NextDouble() < density) columns.Add(c);
            rowPtr[r + 1] = columns.Count;
        }
        return (rowPtr, columns.ToArray(), columns.Count);
    }

    /// <summary>One cuSPARSE SpMM plan on resident buffers: descriptors and workspace made once, then run many times.</summary>
    private sealed class CuSparseSpMM : IDisposable
    {
        private readonly IntPtr _handle, _a, _b, _c, _alpha, _beta;
        private readonly IGpuBuffer? _workspace;

        public CuSparseSpMM(IGpuBuffer values, IGpuBuffer colIdx, IGpuBuffer rowPtr, IGpuBuffer b, IGpuBuffer c,
            int rows, int cols, int n, int nnz, CudaBackend backend)
        {
            Check(CuSparseNative.cusparseCreate(out _handle));
            Check(CuSparseNative.cusparseCreateCsr(out _a, rows, cols, nnz, rowPtr.Handle, colIdx.Handle, values.Handle,
                CuSparseNative.IndexType.I32, CuSparseNative.IndexType.I32, 0, CuSparseNative.DataType.R32F));
            Check(CuSparseNative.cusparseCreateDnMat(out _b, cols, n, n, b.Handle, CuSparseNative.DataType.R32F, CuSparseNative.DnMatDescr_Order.RowMajor));
            Check(CuSparseNative.cusparseCreateDnMat(out _c, rows, n, n, c.Handle, CuSparseNative.DataType.R32F, CuSparseNative.DnMatDescr_Order.RowMajor));
            _alpha = Marshal.AllocHGlobal(sizeof(float));
            _beta = Marshal.AllocHGlobal(sizeof(float));
            Marshal.StructureToPtr(1.0f, _alpha, false);
            Marshal.StructureToPtr(0.0f, _beta, false);
            Check(CuSparseNative.cusparseSpMM_bufferSize(_handle, CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
                _alpha, _a, _b, _beta, _c, CuSparseNative.DataType.R32F, CuSparseNative.SpMMAlg.Default, out ulong size));
            if (size > 0) _workspace = backend.AllocateByteBuffer((int)size);
        }

        public void Run() => Check(CuSparseNative.cusparseSpMM(_handle, CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
            _alpha, _a, _b, _beta, _c, CuSparseNative.DataType.R32F, CuSparseNative.SpMMAlg.Default, _workspace?.Handle ?? IntPtr.Zero));

        private static void Check(CuSparseNative.Status s)
        {
            if (s != CuSparseNative.Status.Success) throw new InvalidOperationException($"cuSPARSE status {s}");
        }

        public void Dispose()
        {
            CuSparseNative.cusparseDestroyDnMat(_c);
            CuSparseNative.cusparseDestroyDnMat(_b);
            CuSparseNative.cusparseDestroySpMat(_a);
            CuSparseNative.cusparseDestroy(_handle);
            _workspace?.Dispose();
            Marshal.FreeHGlobal(_alpha);
            Marshal.FreeHGlobal(_beta);
        }
    }
}
