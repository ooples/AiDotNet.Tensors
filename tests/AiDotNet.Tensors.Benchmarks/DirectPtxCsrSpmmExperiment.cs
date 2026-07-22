using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Resident apples-to-apples evidence harness for issue #852.</summary>
internal static class DirectPtxCsrSpmmExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerSample = 20;
    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private sealed record Evidence(
        string Status, int Run, string Method, Distribution DeviceUs, Distribution EndToEndUs,
        double Gflops, double GigabytesPerSecond, long ManagedBytesPerCall,
        long TemporaryDeviceBytes, double MaximumError, int Registers, int SharedBytes,
        int LocalBytes, int ActiveBlocksPerSm, string Environment);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        Console.WriteLine(
            $"CSR SpMM M1024/K1024/N64/nnz16384: {independentRuns} runs, " +
            $"{Warmups} warmups + {Samples} samples, {LaunchesPerSample} launches/device sample.");
        for (int run = 1; run <= independentRuns; run++)
        {
            RunAiDotNetCells(run);
            RunCuSparseCell(run);
        }
        Console.WriteLine(
            "Run BaselineRunners/py/run_direct_ptx_csr_spmm_competitors.py in the same idle-GPU window " +
            "for PyTorch sparse-CSR eager/graph evidence. Promotion remains HOLD until all three clean " +
            "runs and Nsight zero-spill evidence are attached.");
    }

    private static void RunAiDotNetCells(int run)
    {
        (int[] rowPointers, int[] columns, float[] values, float[] dense, double[] expected) = Inputs();
        bool? prior = DirectPtxFeatureGate.SparseGraphExperimentOverride;
        try
        {
            DirectPtxFeatureGate.SparseGraphExperimentOverride = true;
            using var backend = new CudaBackend();
            if (!backend.IsDirectPtxSparseGraphEnabled)
                throw new InvalidOperationException("Set AIDOTNET_DIRECT_PTX_SPARSE_GRAPH=1 on an SM86 GPU.");
            using var valuesBuffer = backend.AllocateBuffer(values);
            using var columnsBuffer = backend.AllocateBuffer(Reinterpret(columns));
            using var rowPointersBuffer = backend.AllocateBuffer(Reinterpret(rowPointers));
            using var denseBuffer = backend.AllocateBuffer(dense);
            using var outputBuffer = backend.AllocateBuffer(expected.Length);
            if (!backend.PrewarmDirectPtxCsrSpmmVec4F32(1024, 1024, 64, 16384))
                throw new InvalidOperationException(backend.DirectPtxLastError);

            void Direct() => backend.CsrSpMMVec4(
                valuesBuffer, columnsBuffer, rowPointersBuffer, denseBuffer, outputBuffer,
                1024, 1024, 64, 16384);
            void Baseline()
            {
                DirectPtxFeatureGate.SparseGraphExperimentOverride = false;
                try
                {
                    backend.CsrSpMMVec4(
                        valuesBuffer, columnsBuffer, rowPointersBuffer, denseBuffer, outputBuffer,
                        1024, 1024, 64, 16384);
                }
                finally { DirectPtxFeatureGate.SparseGraphExperimentOverride = true; }
            }

            Direct();
            backend.Synchronize();
            double directError = MaximumError(backend.DownloadBuffer(outputBuffer), expected);
            Baseline();
            backend.Synchronize();
            double baselineError = MaximumError(backend.DownloadBuffer(outputBuffer), expected);
            Distribution directDevice = MeasureDevice(backend, Direct);
            Distribution baselineDevice = MeasureDevice(backend, Baseline);
            Distribution directE2e = MeasureEndToEnd(backend, Direct);
            Distribution baselineE2e = MeasureEndToEnd(backend, Baseline);
            long directBytes = MeasureAllocation(backend, Direct);
            long baselineBytes = MeasureAllocation(backend, Baseline);
            if (!backend.TryGetDirectPtxCsrSpmmAudit(1024, 1024, 64, 16384, out var audit))
                throw new InvalidOperationException("Missing CSR SpMM PTX audit.");
            string environment = $"gpu={backend.DeviceName};sm=8.6;runtime={Environment.Version};os={Environment.OSVersion}";
            Print(new Evidence("hardware-unverified", run, "Direct PTX CSR SpMM vec4",
                directDevice, directE2e, Gflops(directDevice.Median), Bandwidth(directDevice.Median),
                directBytes, 0, directError, audit.Function.RegistersPerThread,
                audit.Function.StaticSharedBytes, audit.Function.LocalBytesPerThread,
                audit.ActiveBlocksPerMultiprocessor, environment));
            Print(new Evidence("hardware-unverified", run, "AiDotNet NVRTC csr_spmm_vec4",
                baselineDevice, baselineE2e, Gflops(baselineDevice.Median), Bandwidth(baselineDevice.Median),
                baselineBytes, 0, baselineError, -1, -1, -1, -1, environment));
        }
        finally { DirectPtxFeatureGate.SparseGraphExperimentOverride = prior; }
    }

    private static void RunCuSparseCell(int run)
    {
        using var runtime = new DirectPtxRuntime();
        bool cuSparseAvailable;
        using (runtime.Enter()) cuSparseAvailable = CuSparseNative.IsAvailable;
        if (!cuSparseAvailable)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                status = "unsupported",
                run,
                method = "NVIDIA cuSPARSE resident",
                reason = "cusparse64_12 is unavailable"
            }));
            return;
        }

        var inputs = Inputs();
        using var values = runtime.AllocateBytes((nuint)(inputs.Values.Length * sizeof(float)));
        using var columns = runtime.AllocateBytes((nuint)(inputs.Columns.Length * sizeof(int)));
        using var rows = runtime.AllocateBytes((nuint)(inputs.RowPointers.Length * sizeof(int)));
        using var dense = runtime.AllocateBytes((nuint)(inputs.Dense.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(1024 * 64 * sizeof(float)));
        values.Upload<float>(inputs.Values);
        columns.Upload<int>(inputs.Columns);
        rows.Upload<int>(inputs.RowPointers);
        dense.Upload<float>(inputs.Dense);
        IntPtr handle = IntPtr.Zero, sparse = IntPtr.Zero, matrixB = IntPtr.Zero, matrixC = IntPtr.Zero;
        IntPtr alpha = Marshal.AllocHGlobal(sizeof(float));
        IntPtr beta = Marshal.AllocHGlobal(sizeof(float));
        DirectPtxBuffer? workspace = null;
        try
        {
            Marshal.WriteInt32(alpha, BitConverter.SingleToInt32Bits(1f));
            Marshal.WriteInt32(beta, 0);
            using (runtime.Enter())
            {
                Check(CuSparseNative.cusparseCreate(out handle), "cusparseCreate");
                Check(CuSparseNative.cusparseCreateCsr(out sparse, 1024, 1024, 16384,
                    rows.Pointer, columns.Pointer, values.Pointer,
                    CuSparseNative.IndexType.I32, CuSparseNative.IndexType.I32, 0,
                    CuSparseNative.DataType.R32F), "cusparseCreateCsr");
                Check(CuSparseNative.cusparseCreateDnMat(out matrixB, 1024, 64, 64,
                    dense.Pointer, CuSparseNative.DataType.R32F,
                    CuSparseNative.DnMatDescr_Order.RowMajor), "cusparseCreateDnMat(B)");
                Check(CuSparseNative.cusparseCreateDnMat(out matrixC, 1024, 64, 64,
                    output.Pointer, CuSparseNative.DataType.R32F,
                    CuSparseNative.DnMatDescr_Order.RowMajor), "cusparseCreateDnMat(C)");
                Check(CuSparseNative.cusparseSpMM_bufferSize(handle,
                    CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
                    alpha, sparse, matrixB, beta, matrixC, CuSparseNative.DataType.R32F,
                    CuSparseNative.SpMMAlg.Default, out ulong workspaceBytes),
                    "cusparseSpMM_bufferSize");
                if (workspaceBytes > 0) workspace = runtime.AllocateBytes((nuint)workspaceBytes);
            }

            void Launch()
            {
                using var _ = runtime.Enter();
                Check(CuSparseNative.cusparseSpMM(handle,
                    CuSparseNative.Operation.NonTranspose, CuSparseNative.Operation.NonTranspose,
                    alpha, sparse, matrixB, beta, matrixC, CuSparseNative.DataType.R32F,
                    CuSparseNative.SpMMAlg.Default, workspace?.Pointer ?? IntPtr.Zero),
                    "cusparseSpMM");
            }

            float[] deviceMs = runtime.MeasureKernelSamples(
                Launch, Warmups, Samples, LaunchesPerSample);
            Distribution deviceUs = Summarize(deviceMs.Select(value => (double)value * 1000.0).ToArray());
            var e2eValues = new double[Samples];
            double scale = 1_000_000.0 / Stopwatch.Frequency;
            for (int i = 0; i < Warmups; i++) Launch();
            runtime.Synchronize();
            for (int i = 0; i < Samples; i++)
            {
                long started = Stopwatch.GetTimestamp();
                Launch();
                runtime.Synchronize();
                e2eValues[i] = (Stopwatch.GetTimestamp() - started) * scale;
            }
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < Samples; i++) Launch();
            long managed = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
            runtime.Synchronize();
            var actual = new float[1024 * 64];
            output.Download<float>(actual);
            string environment =
                $"gpu={runtime.DeviceName};sm={runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor};" +
                $"driver={runtime.DriverVersion};runtime={Environment.Version};os={Environment.OSVersion}";
            Print(new Evidence("hardware-unverified", run, "NVIDIA cuSPARSE resident",
                deviceUs, Summarize(e2eValues), Gflops(deviceUs.Median), Bandwidth(deviceUs.Median),
                managed, workspace is null ? 0 : checked((long)workspace.ByteLength),
                MaximumError(actual, inputs.Expected), -1, -1, -1, -1, environment));
        }
        finally
        {
            using (runtime.Enter())
            {
                if (matrixC != IntPtr.Zero) CuSparseNative.cusparseDestroyDnMat(matrixC);
                if (matrixB != IntPtr.Zero) CuSparseNative.cusparseDestroyDnMat(matrixB);
                if (sparse != IntPtr.Zero) CuSparseNative.cusparseDestroySpMat(sparse);
                if (handle != IntPtr.Zero) CuSparseNative.cusparseDestroy(handle);
            }
            workspace?.Dispose();
            Marshal.FreeHGlobal(alpha);
            Marshal.FreeHGlobal(beta);
        }
    }

    private static void Check(CuSparseNative.Status status, string operation)
    {
        if (status != CuSparseNative.Status.Success)
            throw new InvalidOperationException($"{operation} failed with cuSPARSE status {status}.");
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(true);
        using IGpuEvent stop = backend.CreateEvent(true);
        for (int sample = 0; sample < Samples; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < LaunchesPerSample; i++) launch();
            backend.RecordEvent(stop, backend.DefaultStream);
            stop.Synchronize();
            values[sample] = backend.GetEventElapsedTime(start, stop) * 1000.0 / LaunchesPerSample;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
        var values = new double[Samples];
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < Samples; i++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            backend.Synchronize();
            values[i] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < 8; i++) launch();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) launch();
        long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
        backend.Synchronize();
        return bytes / Samples;
    }

    internal static (int[] RowPointers, int[] Columns, float[] Values, float[] Dense, double[] Expected) Inputs()
    {
        int[] rows = Enumerable.Range(0, 1025).Select(row => row * 16).ToArray();
        int[] columns = new int[16384];
        float[] values = new float[16384];
        for (int row = 0; row < 1024; row++)
        for (int item = 0; item < 16; item++)
        {
            int index = row * 16 + item;
            columns[index] = (row * 17 + item * 13) & 1023;
            values[index] = (item - 7.5f) / 32f;
        }
        float[] dense = Enumerable.Range(0, 1024 * 64)
            .Select(index => ((index * 19) % 101 - 50) / 128f).ToArray();
        var expected = new double[1024 * 64];
        for (int row = 0; row < 1024; row++)
        for (int col = 0; col < 64; col++)
        for (int p = rows[row]; p < rows[row + 1]; p++)
            expected[row * 64 + col] += (double)values[p] * dense[columns[p] * 64 + col];
        return (rows, columns, values, dense, expected);
    }

    private static float[] Reinterpret(int[] source)
    {
        var result = new float[source.Length];
        Buffer.BlockCopy(source, 0, result, 0, source.Length * sizeof(int));
        return result;
    }

    private static double MaximumError(float[] actual, double[] expected) =>
        actual.Select((value, index) => Math.Abs(value - expected[index])).Max();

    private static Distribution Summarize(double[] source)
    {
        double[] sorted = source.OrderBy(value => value).ToArray();
        double Percentile(double p) => sorted[(int)Math.Ceiling(p * sorted.Length) - 1];
        return new(source.Average(), Percentile(0.5), Percentile(0.95), Percentile(0.99));
    }

    private static double Gflops(double medianUs) => (2.0 * 16384 * 64) / (medianUs * 1000.0);
    private static double Bandwidth(double medianUs)
    {
        const double estimatedBytes = 16.0 * 16384 * (4 + 4 + 16) + 1024.0 * 64 * 4;
        return estimatedBytes / (medianUs * 1000.0);
    }

    private static void Print(Evidence evidence) =>
        Console.WriteLine(JsonSerializer.Serialize(evidence));
}
