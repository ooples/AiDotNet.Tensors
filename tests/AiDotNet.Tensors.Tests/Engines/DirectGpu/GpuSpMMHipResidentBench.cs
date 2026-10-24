using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Tests.Engines.BlasManaged;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #515, the HIP half: the managed CSR SpMM kernels against rocSPARSE on an AMD GPU, everything resident on the
/// device. The mirror of <see cref="GpuSpMMResidentBench"/>, held to the same frozen bar: operands uploaded once,
/// both arms cross-checked first, each arm timed over a batch of launches between two device synchronizations, the
/// arms alternating over several rounds and each keeping its fastest round. Gated on <c>AIDOTNET_PERF_RUNNER=1</c>:
/// it needs a quiet machine.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuSpMMHipResidentBench
{
    private readonly ITestOutputHelper _out;
    public GpuSpMMHipResidentBench(ITestOutputHelper output) => _out = output;

    private const int Launches = 200, Rounds = 7;

    [SkippableFact]
    public void Bench_ResidentKernels_CustomVsRocSparse()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1",
            "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine to bench GPU SpMM.");
        Skip.IfNot(RocSparseNative.IsAvailable, "rocSPARSE is not available.");
        var backend = HipCustomSparseBackend.Shared;
        Skip.If(backend is null, "No HIP backend with the managed sparse kernels on this host.");
        if (backend is null) return;

        var rng = new Random(515);
        _out.WriteLine($"device: {backend.DeviceName}");
        _out.WriteLine($"{"shape",-30}{"nnz",9}{"kernel",8}{"custom us",11}{"rocSPARSE us",14}{"custom GF/s",13}{"rocSPARSE GF/s",16}{"speedup",9}");
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

            // The kernel HipCustomSparseBackend.SpMM picks for this width.
            string kernel = s.N % 4 == 0 ? "vec4" : s.N <= 64 ? "warp" : "scalar";
            Action custom = kernel switch
            {
                "vec4" => () => backend.CsrSpMMVec4(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
                "warp" => () => backend.CsrSpMMWarp(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
                _ => () => backend.CsrSpMM(dValues, dColIdx, dRowPtr, dB, dOut, s.Rows, s.Cols, s.N, nnz),
            };

            using var roc = new RocSparseSpMM(dValues, dColIdx, dRowPtr, dB, dOut2, s.Rows, s.Cols, s.N, nnz, backend);

            // Both arms must compute the same product before either is timed.
            custom();
            roc.Run();
            Sync(backend);
            float[] a = backend.DownloadBuffer(dOut), c = backend.DownloadBuffer(dOut2);
            // rocSPARSE reports success but writes nothing on a GPU its build has no code object for (the Windows 6.4
            // build has none for gfx1012, the RX 5500 XT): there is no baseline to time against there.
            Skip.If(Array.TrueForAll(c, v => v == 0f) && !Array.TrueForAll(a, v => v == 0f),
                $"rocSPARSE produced no output on {backend.DeviceName}: its build has no code for this GPU's architecture.");
            for (int i = 0; i < s.Rows * s.N; i++)
                Assert.True(Math.Abs(a[i] - c[i]) <= 1e-3f * Math.Max(1f, Math.Abs(c[i])), $"{s.Name}[{i}]: custom {a[i]}, rocSPARSE {c[i]}");

            double customUs = double.MaxValue, rocUs = double.MaxValue;
            for (int round = 0; round < Rounds; round++)
            {
                customUs = Math.Min(customUs, BatchMicroseconds(backend, custom));
                rocUs = Math.Min(rocUs, BatchMicroseconds(backend, roc.Run));
            }
            double flop = 2.0 * nnz * s.N;
            double speedup = rocUs / customUs;
            compared++;
            if (speedup >= 1.0) wins++; else maxLoss = Math.Max(maxLoss, 1.0 / speedup);
            _out.WriteLine($"{s.Name,-30}{nnz,9}{kernel,8}{customUs,11:F2}{rocUs,14:F2}{flop / (customUs * 1e3),13:F1}{flop / (rocUs * 1e3),16:F1}{speedup,8:F2}x");
        }

        int winRate = compared == 0 ? 0 : wins * 100 / compared;
        _out.WriteLine($"=> custom wins {wins}/{compared} = {winRate}%; max loss {maxLoss:F2}x");
        Assert.True(winRate >= SpecializedPerfBar.GpuSpMMMinWinRatePercent,
            $"HIP SpMM win rate {winRate}% is below the frozen bar {SpecializedPerfBar.GpuSpMMMinWinRatePercent}%.");
        Assert.True(maxLoss <= SpecializedPerfBar.GpuSpMMMaxLossMultiple,
            $"HIP SpMM max loss {maxLoss:F2}x is above the frozen bar {SpecializedPerfBar.GpuSpMMMaxLossMultiple}x.");
    }

    // rocSPARSE runs on the null stream (no handle stream is set), the managed kernels on the backend's: the device
    // synchronization covers both.
    private static void Sync(HipBackend backend)
    {
        backend.Synchronize();
        Assert.Equal(HipError.Success, HipNativeBindings.hipDeviceSynchronize());
    }

    /// <summary>Microseconds per launch over <see cref="Launches"/> back-to-back launches between two syncs.</summary>
    private static double BatchMicroseconds(HipBackend backend, Action launch)
    {
        Sync(backend);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Launches; i++) launch();
        Sync(backend);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / Launches;
    }

    private static IGpuBuffer Ints(HipBackend backend, int[] values)
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

    /// <summary>One rocSPARSE SpMM plan on resident buffers: descriptors and workspace made once, then run many times.</summary>
    private sealed class RocSparseSpMM : IDisposable
    {
        private readonly IntPtr _handle, _a, _b, _c, _alpha, _beta;
        private readonly IGpuBuffer? _workspace;
        private ulong _workspaceSize;

        public RocSparseSpMM(IGpuBuffer values, IGpuBuffer colIdx, IGpuBuffer rowPtr, IGpuBuffer b, IGpuBuffer c,
            int rows, int cols, int n, int nnz, HipBackend backend)
        {
            Check(RocSparseNative.rocsparse_create_handle(out _handle));
            Check(RocSparseNative.rocsparse_create_csr_descr(out _a, rows, cols, nnz, rowPtr.Handle, colIdx.Handle, values.Handle,
                RocSparseNative.IndexType.I32, RocSparseNative.IndexType.I32, RocSparseNative.IndexBase.Zero, RocSparseNative.DataType.R32F));
            Check(RocSparseNative.rocsparse_create_dnmat_descr(out _b, cols, n, n, b.Handle, RocSparseNative.DataType.R32F, RocSparseNative.Order.Row));
            Check(RocSparseNative.rocsparse_create_dnmat_descr(out _c, rows, n, n, c.Handle, RocSparseNative.DataType.R32F, RocSparseNative.Order.Row));
            _alpha = Marshal.AllocHGlobal(sizeof(float));
            _beta = Marshal.AllocHGlobal(sizeof(float));
            Marshal.StructureToPtr(1.0f, _alpha, false);
            Marshal.StructureToPtr(0.0f, _beta, false);
            // All three stages, in order: size the workspace, preprocess once, then compute on every Run.
            Check(RocSparseNative.rocsparse_spmm(_handle, RocSparseNative.Operation.NonTranspose, RocSparseNative.Operation.NonTranspose,
                _alpha, _a, _b, _beta, _c, RocSparseNative.DataType.R32F, RocSparseNative.SpMMAlg.Default,
                RocSparseNative.SpMMStage.BufferSize, ref _workspaceSize, IntPtr.Zero));
            if (_workspaceSize > 0) _workspace = backend.AllocateByteBuffer((int)_workspaceSize);
            Check(RocSparseNative.rocsparse_spmm(_handle, RocSparseNative.Operation.NonTranspose, RocSparseNative.Operation.NonTranspose,
                _alpha, _a, _b, _beta, _c, RocSparseNative.DataType.R32F, RocSparseNative.SpMMAlg.Default,
                RocSparseNative.SpMMStage.Preprocess, ref _workspaceSize, _workspace?.Handle ?? IntPtr.Zero));
        }

        public void Run() => Check(RocSparseNative.rocsparse_spmm(_handle, RocSparseNative.Operation.NonTranspose, RocSparseNative.Operation.NonTranspose,
            _alpha, _a, _b, _beta, _c, RocSparseNative.DataType.R32F, RocSparseNative.SpMMAlg.Default,
            RocSparseNative.SpMMStage.Compute, ref _workspaceSize, _workspace?.Handle ?? IntPtr.Zero));

        private static void Check(RocSparseNative.Status s)
        {
            if (s != RocSparseNative.Status.Success) throw new InvalidOperationException($"rocSPARSE status {s}");
        }

        public void Dispose()
        {
            RocSparseNative.rocsparse_destroy_dnmat_descr(_c);
            RocSparseNative.rocsparse_destroy_dnmat_descr(_b);
            RocSparseNative.rocsparse_destroy_spmat_descr(_a);
            RocSparseNative.rocsparse_destroy_handle(_handle);
            _workspace?.Dispose();
            Marshal.FreeHGlobal(_alpha);
            Marshal.FreeHGlobal(_beta);
        }
    }
}
