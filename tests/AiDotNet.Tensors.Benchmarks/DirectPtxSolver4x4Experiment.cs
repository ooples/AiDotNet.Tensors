using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Issue #853 resident NVIDIA solver-family evidence generator.</summary>
internal static class DirectPtxSolver4x4Experiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int LaunchesPerSample = 10;
    private static readonly int[] Batches = [1024, 4096, 16384, 65536];
    private static readonly string[] Operations =
        [
            "cholesky", "lu-factor", "qr", "eigh", "eigh-lower", "svd", "lu-solve",
            "ldl-factor", "ldl-solve", "solve", "tri-lower", "tri-upper",
            "chol-backward", "solve-backward"
        ];

    private readonly record struct Distribution(double Mean, double Median, double P95, double P99);
    private readonly record struct Result(
        int Run, string Operation, int Batch, string Method,
        Distribution Device, Distribution EndToEnd, double Gflops, double GbPerSecond,
        long ManagedBytes, long TemporaryDeviceBytes, double MaximumError,
        int Registers, int SharedBytes, int LocalBytes, int ActiveBlocksPerSm);

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        Console.WriteLine(
            $"Direct PTX solver family: {independentRuns} clean run(s), {Warmups} warmups + " +
            $"{Samples} samples, {LaunchesPerSample} launches/device sample.");
        var results = new List<Result>();
        for (int run = 1; run <= independentRuns; run++)
        {
            using var backend = new CudaBackend();
            foreach (string operation in Operations)
            foreach (int batch in Batches)
                RunCell(backend, run, operation, batch, results);
        }
        Print(results);
        Console.WriteLine(
            "HOLD: PyTorch eager/graph and Nsight JSON/CSV must be joined before the release gate can pass.");
        Environment.ExitCode = 2;
    }

    private static void RunCell(
        CudaBackend backend,
        int run,
        string operation,
        int batch,
        List<Result> results)
    {
        float[] matrixHost = Matrices(batch, MatrixFor(operation));
        float[] vectorHost = Enumerable.Range(0, batch).SelectMany(_ => new[] { 1f, 2f, 3f, 4f }).ToArray();
        float[] gradVectorHost = Enumerable.Range(0, batch).SelectMany(_ => new[] { .5f, 1f, 1.5f, 2f }).ToArray();
        float[] pivotBits = new float[batch * 4];
        int[] identityPivots = Enumerable.Range(0, batch).SelectMany(_ => new[] { 0, 1, 2, 3 }).ToArray();
        Buffer.BlockCopy(identityPivots, 0, pivotBits, 0, identityPivots.Length * sizeof(int));
        using var input = backend.AllocateBuffer(matrixHost);
        using var matrix1 = backend.AllocateBuffer(new float[matrixHost.Length]);
        using var matrix2 = backend.AllocateBuffer(new float[matrixHost.Length]);
        using var vector = backend.AllocateBuffer(new float[vectorHost.Length]);
        using var rhs = backend.AllocateBuffer(vectorHost);
        using var pivots = backend.AllocateBuffer(pivotBits);
        using var info = backend.AllocateBuffer(new float[batch]);
        using var gradMatrix = backend.AllocateBuffer(Matrices(batch, Identity));
        using var gradVector = backend.AllocateBuffer(gradVectorHost);

        Action direct = operation switch
        {
            "cholesky" => () => backend.LinalgCholesky(input, matrix1, info, batch, 4, false),
            "lu-factor" => () => backend.LinalgLuFactor(input, matrix1, pivots, batch, 4, 4),
            "qr" => () => backend.LinalgQrReduced(input, matrix1, matrix2, batch, 4, 4),
            "eigh" => () => backend.LinalgEigh(input, vector, matrix2, batch, 4),
            "eigh-lower" => () => ((IExtendedLinalgBackend)backend).LinalgEighSymmetric(
                input, vector, matrix2, batch, 4, upper: false),
            "svd" => () => ((IExtendedLinalgBackend)backend).LinalgSvdReduced(
                input, matrix1, vector, matrix2, batch, 4, 4),
            "lu-solve" => () => ((IExtendedLinalgBackend)backend).LinalgLuSolve(
                input, pivots, rhs, vector, batch, 4, 1, true),
            "ldl-factor" => () => ((IExtendedLinalgBackend)backend).LinalgLdlFactor(
                input, matrix1, pivots, batch, 4, upper: false),
            "ldl-solve" => () => ((IExtendedLinalgBackend)backend).LinalgLdlSolve(
                input, pivots, rhs, vector, batch, 4, upper: false, rhsIsVector: true),
            "solve" => () => ((IExtendedLinalgBackend)backend).LinalgSolveVector(
                input, rhs, vector, info, batch, 4),
            "tri-lower" => () => ((IExtendedLinalgBackend)backend).LinalgTriangularSolveVector(
                input, rhs, vector, batch, 4, upper: false, unitDiagonal: false),
            "tri-upper" => () => ((IExtendedLinalgBackend)backend).LinalgTriangularSolveVector(
                input, rhs, vector, batch, 4, upper: true, unitDiagonal: false),
            "chol-backward" => () => ((IExtendedLinalgBackend)backend).LinalgCholeskyBackwardLower(
                input, gradMatrix, matrix1, batch, 4),
            _ => () => ((IExtendedLinalgBackend)backend).LinalgSolveBackwardVector(
                input, rhs, gradVector, matrix1, vector, batch, 4)
        };
        Action? current = operation switch
        {
            "cholesky" => () => backend.LinalgCholesky(input, matrix1, info, batch, 4, false),
            "lu-factor" => () => backend.LinalgLuFactor(input, matrix1, pivots, batch, 4, 4),
            "qr" => () => backend.LinalgQrReduced(input, matrix1, matrix2, batch, 4, 4),
            "eigh" => () => backend.LinalgEigh(input, vector, matrix2, batch, 4),
            _ => null
        };

        Enable(operation, true);
        // First uncaptured call performs the bounded cold-path tune when enabled.
        long dispatchBefore = DispatchCount(backend, operation);
        direct();
        long dispatchAfter = DispatchCount(backend, operation);
        if (dispatchAfter != dispatchBefore + 1 || backend.DirectPtxLastError is not null)
            throw new InvalidOperationException(
                $"Direct PTX routing was not exclusive for {operation}/B={batch}: " +
                $"expected one dispatch, observed {dispatchAfter - dispatchBefore}; " +
                $"last error={backend.DirectPtxLastError ?? "none"}.");
        Prewarm(backend, operation, batch);
        direct(); backend.Synchronize();
        double error = Error(backend, operation, batch, matrixHost, vectorHost,
            matrix1, matrix2, vector, pivots, info);
        DirectPtxKernelAudit audit = Audit(backend, operation, batch);
        Distribution directDevice = MeasureDevice(backend, direct);
        Distribution directE2e = MeasureEndToEnd(backend, direct);
        long directBytes = MeasureAllocation(backend, direct);
        IntPtr graph = backend.CaptureGraph(direct);
        if (graph == IntPtr.Zero) throw new InvalidOperationException($"Capture failed for {operation}/B={batch}.");
        try
        {
            Action graphLaunch = () => backend.EnqueueCapturedGraph(graph);
            Distribution graphDevice = MeasureDevice(backend, graphLaunch);
            Distribution graphE2e = MeasureEndToEnd(backend, graphLaunch);
            results.Add(Create(run, operation, batch, "Direct PTX CUDA graph", graphDevice,
                graphE2e, MeasureAllocation(backend, graphLaunch), 0, error, audit));
        }
        finally { backend.DestroyCapturedGraph(graph); }
        results.Add(Create(run, operation, batch, "Direct PTX resident", directDevice,
            directE2e, directBytes, 0, error, audit));

        if (current is not null)
        {
            Enable(operation, false);
            long establishedBefore = DispatchCount(backend, operation);
            current(); backend.Synchronize();
            if (DispatchCount(backend, operation) != establishedBefore)
                throw new InvalidOperationException(
                    $"Established baseline unexpectedly routed through direct PTX for {operation}/B={batch}.");
            Distribution currentDevice = MeasureDevice(backend, current);
            Distribution currentE2e = MeasureEndToEnd(backend, current);
            results.Add(Create(run, operation, batch, "AiDotNet CUDA established", currentDevice,
                currentE2e, MeasureAllocation(backend, current), 0,
                Error(backend, operation, batch, matrixHost, vectorHost,
                    matrix1, matrix2, vector, pivots, info), null));
        }
        Enable(operation, null);
    }

    private static long DispatchCount(CudaBackend backend, string operation) =>
        operation == "cholesky"
            ? backend.DirectPtxCholesky4x4DispatchCount
            : backend.DirectPtxSolver4x4DispatchCount;

    private static Result Create(
        int run, string operation, int batch, string method,
        Distribution device, Distribution e2e, long managedBytes, long tempBytes,
        double error, DirectPtxKernelAudit? audit)
    {
        double flops = EstimatedFlops(operation, batch);
        double seconds = device.Median * 1e-6;
        double bytes = BytesMoved(operation, batch);
        return new Result(run, operation, batch, method, device, e2e,
            flops / seconds / 1e9, bytes / seconds / 1e9,
            managedBytes, tempBytes, error,
            audit?.Function.RegistersPerThread ?? -1,
            audit?.Function.StaticSharedBytes ?? -1,
            audit?.Function.LocalBytesPerThread ?? -1,
            audit?.ActiveBlocksPerMultiprocessor ?? -1);
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(true);
        using IGpuEvent stop = backend.CreateEvent(true);
        for (int i = 0; i < Samples; i++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int launch = 0; launch < LaunchesPerSample; launch++) action();
            backend.RecordEvent(stop, backend.DefaultStream);
            stop.Synchronize();
            values[i] = backend.GetEventElapsedTime(start, stop) * 1000.0 / LaunchesPerSample;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var values = new double[Samples];
        double scale = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < Samples; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action(); backend.Synchronize();
            values[i] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) action();
        long result = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return result;
    }

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, .50),
            Percentile(values, .95), Percentile(values, .99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position, upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void Enable(string operation, bool? value)
    {
        if (operation == "cholesky") DirectPtxFeatureGate.Cholesky4x4ExperimentOverride = value;
        else DirectPtxFeatureGate.Solver4x4ExperimentOverride = value;
    }

    private static void Prewarm(CudaBackend backend, string operation, int batch)
    {
        bool ok = operation == "cholesky"
            ? backend.PrewarmDirectPtxCholesky4x4(batch)
            : backend.PrewarmDirectPtxSolver4x4(operation switch
            {
                "lu-factor" => DirectPtxSolver4x4Operation.LuFactor,
                "qr" => DirectPtxSolver4x4Operation.QrReduced,
                "eigh" => DirectPtxSolver4x4Operation.EighUpper,
                "eigh-lower" => DirectPtxSolver4x4Operation.EighLower,
                "svd" => DirectPtxSolver4x4Operation.SvdReduced,
                "lu-solve" => DirectPtxSolver4x4Operation.LuSolveVector,
                "ldl-factor" => DirectPtxSolver4x4Operation.LdlFactorLower,
                "ldl-solve" => DirectPtxSolver4x4Operation.LdlSolveVectorLower,
                "solve" => DirectPtxSolver4x4Operation.GeneralSolveVector,
                "tri-lower" => DirectPtxSolver4x4Operation.TriangularSolveVectorLower,
                "tri-upper" => DirectPtxSolver4x4Operation.TriangularSolveVectorUpper,
                "chol-backward" => DirectPtxSolver4x4Operation.CholeskyBackwardLower,
                _ => DirectPtxSolver4x4Operation.SolveBackwardVector
            }, batch);
        if (!ok) throw new InvalidOperationException($"Prewarm failed: {backend.DirectPtxLastError}");
    }

    private static DirectPtxKernelAudit Audit(CudaBackend backend, string operation, int batch)
    {
        bool found;
        DirectPtxKernelAudit audit;
        if (operation == "cholesky") found = backend.TryGetDirectPtxCholesky4x4Audit(batch, out audit);
        else found = backend.TryGetDirectPtxSolver4x4Audit(operation switch
        {
            "lu-factor" => DirectPtxSolver4x4Operation.LuFactor,
            "qr" => DirectPtxSolver4x4Operation.QrReduced,
            "eigh" => DirectPtxSolver4x4Operation.EighUpper,
            "eigh-lower" => DirectPtxSolver4x4Operation.EighLower,
            "svd" => DirectPtxSolver4x4Operation.SvdReduced,
            "lu-solve" => DirectPtxSolver4x4Operation.LuSolveVector,
            "ldl-factor" => DirectPtxSolver4x4Operation.LdlFactorLower,
            "ldl-solve" => DirectPtxSolver4x4Operation.LdlSolveVectorLower,
            "solve" => DirectPtxSolver4x4Operation.GeneralSolveVector,
            "tri-lower" => DirectPtxSolver4x4Operation.TriangularSolveVectorLower,
            "tri-upper" => DirectPtxSolver4x4Operation.TriangularSolveVectorUpper,
            "chol-backward" => DirectPtxSolver4x4Operation.CholeskyBackwardLower,
            _ => DirectPtxSolver4x4Operation.SolveBackwardVector
        }, batch, out audit);
        return found ? audit : throw new InvalidOperationException("Direct PTX audit missing.");
    }

    private static double Error(
        CudaBackend backend, string operation, int batch, float[] input, float[] rhs,
        IGpuBuffer matrix1, IGpuBuffer matrix2, IGpuBuffer vector, IGpuBuffer pivots, IGpuBuffer info)
    {
        // Operation-specific residuals are emitted by the external harness as
        // well; this resident check deliberately samples the first 16 matrices.
        float[] a = backend.DownloadBuffer(matrix1);
        if (operation is "lu-solve" or "ldl-solve")
        {
            float[] x = backend.DownloadBuffer(vector);
            return x.Take(64).Zip(rhs.Take(64), (left, right) => Math.Abs(left - right)).Max();
        }
        if (operation is "solve" or "tri-lower" or "tri-upper")
            return SolveError(input, rhs, backend.DownloadBuffer(vector), Math.Min(batch, 16));
        if (operation == "cholesky") return ReconstructionError(input, a, null, 16, triangular: true);
        if (operation == "lu-factor")
            return LuFactorError(input, a, backend.DownloadBuffer(pivots), Math.Min(batch, 16));
        if (operation == "ldl-factor")
            return LdlFactorError(input, a, backend.DownloadBuffer(pivots), Math.Min(batch, 16));
        if (operation == "qr") return ReconstructionError(input, a, backend.DownloadBuffer(matrix2), 16, false);
        if (operation is "eigh" or "eigh-lower")
            return EighError(input, backend.DownloadBuffer(vector), backend.DownloadBuffer(matrix2),
                Math.Min(batch, 16));
        if (operation == "svd") return SvdError(input, a, backend.DownloadBuffer(vector), backend.DownloadBuffer(matrix2), 16);
        if (operation == "chol-backward")
            return CholeskyBackwardDiagonalError(input, a, Math.Min(batch, 16));
        if (operation == "solve-backward")
            return SolveBackwardIdentityError(rhs, backend.DownloadBuffer(matrix1),
                backend.DownloadBuffer(vector), Math.Min(batch, 16));
        throw new ArgumentOutOfRangeException(nameof(operation), operation, "Unknown solver benchmark operation.");
    }

    private static double ReconstructionError(
        float[] expected, float[] left, float[]? right, int matrices, bool triangular)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            double sum = 0;
            for (int k = 0; k < 4; k++)
                sum += left[b * 16 + row * 4 + k] *
                    (triangular ? left[b * 16 + col * 4 + k] : right![b * 16 + k * 4 + col]);
            maximum = Math.Max(maximum, Math.Abs(sum - expected[b * 16 + row * 4 + col]));
        }
        return maximum;
    }

    private static double SvdError(float[] expected, float[] u, float[] s, float[] vh, int matrices)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            double sum = 0;
            for (int k = 0; k < 4; k++) sum += u[b * 16 + row * 4 + k] * s[b * 4 + k] * vh[b * 16 + k * 4 + col];
            maximum = Math.Max(maximum, Math.Abs(sum - expected[b * 16 + row * 4 + col]));
        }
        return maximum;
    }

    private static double LuFactorError(float[] input, float[] lu, float[] pivotBits, int matrices)
    {
        var pivots = new int[pivotBits.Length];
        Buffer.BlockCopy(pivotBits, 0, pivots, 0, pivotBits.Length * sizeof(float));
        double maximum = 0;
        var permuted = new float[16];
        for (int b = 0; b < matrices; b++)
        {
            Array.Copy(input, b * 16, permuted, 0, 16);
            for (int step = 0; step < 4; step++)
            {
                int pivot = pivots[b * 4 + step];
                if ((uint)pivot >= 4u) return double.PositiveInfinity;
                if (pivot == step) continue;
                for (int col = 0; col < 4; col++)
                    (permuted[step * 4 + col], permuted[pivot * 4 + col]) =
                        (permuted[pivot * 4 + col], permuted[step * 4 + col]);
            }

            for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                double sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    double left = row == k ? 1 : row > k ? lu[b * 16 + row * 4 + k] : 0;
                    double right = k <= col ? lu[b * 16 + k * 4 + col] : 0;
                    sum += left * right;
                }
                maximum = Math.Max(maximum, Math.Abs(sum - permuted[row * 4 + col]));
            }
        }
        return maximum;
    }

    private static double EighError(float[] input, float[] values, float[] vectors, int matrices)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            double av = 0;
            double orthogonal = 0;
            for (int k = 0; k < 4; k++)
            {
                float symmetric = row <= k
                    ? input[b * 16 + row * 4 + k]
                    : input[b * 16 + k * 4 + row];
                av += symmetric * vectors[b * 16 + k * 4 + col];
                orthogonal += vectors[b * 16 + k * 4 + row] * vectors[b * 16 + k * 4 + col];
            }
            double vd = vectors[b * 16 + row * 4 + col] * values[b * 4 + col];
            maximum = Math.Max(maximum, Math.Abs(av - vd));
            maximum = Math.Max(maximum, Math.Abs(orthogonal - (row == col ? 1 : 0)));
        }
        return maximum;
    }

    private static double LdlFactorError(float[] input, float[] ld, float[] pivotBits, int matrices)
    {
        var pivots = new int[pivotBits.Length];
        Buffer.BlockCopy(pivotBits, 0, pivots, 0, pivotBits.Length * sizeof(float));
        double maximum = 0;
        var permuted = new float[16];
        for (int b = 0; b < matrices; b++)
        {
            Array.Copy(input, b * 16, permuted, 0, 16);
            for (int step = 0; step < 4; step++)
            {
                int pivot = pivots[b * 4 + step];
                if ((uint)pivot >= 4u) return double.PositiveInfinity;
                for (int k = 0; pivot != step && k < 4; k++)
                {
                    (permuted[step * 4 + k], permuted[pivot * 4 + k]) =
                        (permuted[pivot * 4 + k], permuted[step * 4 + k]);
                }
                for (int k = 0; pivot != step && k < 4; k++)
                {
                    (permuted[k * 4 + step], permuted[k * 4 + pivot]) =
                        (permuted[k * 4 + pivot], permuted[k * 4 + step]);
                }
            }
            for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                double sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    double left = row == k ? 1 : row > k ? ld[b * 16 + row * 4 + k] : 0;
                    double right = col == k ? 1 : col > k ? ld[b * 16 + col * 4 + k] : 0;
                    sum += left * ld[b * 16 + k * 4 + k] * right;
                }
                maximum = Math.Max(maximum, Math.Abs(sum - permuted[row * 4 + col]));
            }
        }
        return maximum;
    }

    private static double SolveError(float[] matrix, float[] rhs, float[] solution, int matrices)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        for (int row = 0; row < 4; row++)
        {
            double value = 0;
            for (int col = 0; col < 4; col++)
                value += matrix[b * 16 + row * 4 + col] * solution[b * 4 + col];
            maximum = Math.Max(maximum, Math.Abs(value - rhs[b * 4 + row]));
        }
        return maximum;
    }

    private static double CholeskyBackwardDiagonalError(float[] factor, float[] gradient, int matrices)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            double expected = row == col ? .5 / factor[b * 16 + row * 5] : 0;
            maximum = Math.Max(maximum, Math.Abs(gradient[b * 16 + row * 4 + col] - expected));
        }
        return maximum;
    }

    private static double SolveBackwardIdentityError(
        float[] solution, float[] gradInput, float[] gradRhs, int matrices)
    {
        double maximum = 0;
        for (int b = 0; b < matrices; b++)
        {
            for (int row = 0; row < 4; row++)
            {
                double expectedGradRhs = .5 * (row + 1);
                maximum = Math.Max(maximum, Math.Abs(gradRhs[b * 4 + row] - expectedGradRhs));
                for (int col = 0; col < 4; col++)
                {
                    double expected = -expectedGradRhs * solution[b * 4 + col];
                    maximum = Math.Max(maximum,
                        Math.Abs(gradInput[b * 16 + row * 4 + col] - expected));
                }
            }
        }
        return maximum;
    }

    private static void Print(IReadOnlyList<Result> results)
    {
        Console.WriteLine(
            $"{"Run",3} {"Operation",11} {"Batch",7} {"Method",25} {"dev mean",9} {"dev med",9} " +
            $"{"dev P95",9} {"dev P99",9} {"E2E mean",9} {"E2E med",9} {"E2E P95",9} {"E2E P99",9} " +
            $"{"GFLOPS",9} {"GB/s",9} {"alloc B",8} {"temp B",8} " +
            $"{"max err",10} {"regs",5} {"shared",7} {"local",5} {"occ",4}");
        foreach (Result row in results)
        {
            Console.WriteLine($"{row.Run,3} {row.Operation,11} {row.Batch,7} {row.Method,25} " +
                $"{row.Device.Mean,9:F2} {row.Device.Median,9:F2} {row.Device.P95,9:F2} {row.Device.P99,9:F2} " +
                $"{row.EndToEnd.Mean,9:F2} {row.EndToEnd.Median,9:F2} {row.EndToEnd.P95,9:F2} {row.EndToEnd.P99,9:F2} " +
                $"{row.Gflops,9:F2} {row.GbPerSecond,9:F2} {row.ManagedBytes,8} {row.TemporaryDeviceBytes,8} " +
                $"{row.MaximumError,10:G4} {Dash(row.Registers),5} {Dash(row.SharedBytes),7} " +
                $"{Dash(row.LocalBytes),5} {Dash(row.ActiveBlocksPerSm),4}");
            Console.WriteLine("solver_evidence_json=" + JsonSerializer.Serialize(row));
        }
    }

    private static string Dash(int value) => value < 0 ? "-" : value.ToString();
    private static double EstimatedFlops(string operation, int batch) => operation switch
    {
        "cholesky" => batch * (64.0 / 3.0),
        "lu-factor" => batch * (128.0 / 3.0),
        "qr" => batch * (256.0 / 3.0),
        "eigh" or "eigh-lower" => batch * 8 * 6 * 96.0,
        "svd" => batch * (10 * 6 * 96.0 + 256.0),
        "ldl-factor" => batch * (64.0 / 3.0),
        "tri-lower" or "tri-upper" => batch * 32.0,
        "chol-backward" => batch * 384.0,
        "solve-backward" => batch * 192.0,
        _ => batch * 64.0
    };
    private static double BytesMoved(string operation, int batch) => operation switch
    {
        "cholesky" => batch * (64 + 64 + 4.0),
        "lu-factor" => batch * (64 + 64 + 16.0),
        "qr" => batch * (64 + 64 + 64.0),
        "eigh" or "eigh-lower" => batch * (64 + 16 + 64.0),
        "svd" => batch * (64 + 64 + 16 + 64.0),
        "ldl-factor" => batch * (64 + 64 + 16.0),
        "solve" => batch * (64 + 16 + 16 + 4.0),
        "tri-lower" or "tri-upper" => batch * (64 + 16 + 16.0),
        "chol-backward" => batch * (64 + 64 + 64.0),
        "solve-backward" => batch * (64 + 16 + 16 + 64 + 16.0),
        _ => batch * (64 + 16 + 16 + 16.0)
    };

    private static float[] MatrixFor(string operation) => operation switch
    {
        "lu-solve" or "ldl-solve" or "solve-backward" => Identity,
        "chol-backward" => CholeskyDiagonal,
        "tri-lower" => LowerTriangular,
        "tri-upper" => UpperTriangular,
        _ => Spd
    };

    private static float[] Matrices(int batch, float[] matrix)
    {
        var result = new float[batch * 16];
        for (int i = 0; i < batch; i++) Array.Copy(matrix, 0, result, i * 16, 16);
        return result;
    }

    private static readonly float[] Spd =
        [9, 1, 2, .5f, 1, 8, .25f, 1, 2, .25f, 7, .75f, .5f, 1, .75f, 6];
    private static readonly float[] Identity =
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    private static readonly float[] CholeskyDiagonal =
        [3, 0, 0, 0, 0, 2.5f, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1.5f];
    private static readonly float[] LowerTriangular =
        [3, 0, 0, 0, 1, 4, 0, 0, .5f, 1, 5, 0, .25f, .5f, 1, 6];
    private static readonly float[] UpperTriangular =
        [3, 1, .5f, .25f, 0, 4, 1, .5f, 0, 0, 5, 1, 0, 0, 0, 6];
}
