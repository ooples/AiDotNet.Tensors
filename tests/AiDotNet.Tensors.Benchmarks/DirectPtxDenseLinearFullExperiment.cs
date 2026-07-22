using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #836 resident, apples-to-apples championship matrix. Every route uses
/// preallocated device buffers; JIT, module creation, graph construction, and
/// host/device copies are outside the timed region.
/// </summary>
internal static class DirectPtxDenseLinearFullExperiment
{
    private const int Warmups = 30;
    private const int Samples = 101;
    private const int DeviceLaunches = 10;
    private const double Tolerance = 2e-4;
    private static string _deviceFingerprint = "n/a";

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    private readonly record struct Resources(
        int Registers, int StaticShared, int Local, int ActiveBlocks,
        string Blueprint, string DeviceFingerprint, string PtxSha256)
    {
        internal static Resources None => new(
            -1, -1, -1, -1, "n/a",
            DirectPtxDenseLinearFullExperiment._deviceFingerprint, "n/a");
    }

    internal static void Run(
        int independentRuns = 3,
        bool includePython = true,
        string? onlyOperation = null)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        AddPackagedNativeRuntimeToPath();
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.FusedLinearExperimentOverride;
        bool previousDeterminism = AiDotNetEngine.DeterministicMode;
        try
        {
            GpuBenchmarkEnvironment.RequireIdleGpu("dense-linear-start");
            GpuBenchmarkEnvironment.PrintSnapshot("dense-linear-start");
            PrintEnvironmentFingerprint(independentRuns);
            Console.WriteLine(
                $"Issue #836 dense-linear matrix: {independentRuns} clean run(s), " +
                $"{Warmups} warmups + {Samples} samples/cell, {DeviceLaunches} launches/device sample.");
            Console.WriteLine(
                "Resident operands and caller-owned outputs; managed allocation is measured after prewarm.");
            PrintHeader();
            for (int run = 1; run <= independentRuns; run++)
            {
                GpuBenchmarkEnvironment.RequireNoForeignCompute(
                    $"dense-linear-run-{run}-start");
                using var backend = new CudaBackend();
                using var cublasLt = new CuBlasLtMatmul();
                if (run == 1) Console.WriteLine($"GPU: {backend.DeviceName}");
                if (Includes(onlyOperation, "decode-gelu")) RunDecode(backend, cublasLt, run);
                if (Includes(onlyOperation, "gemm-fp32")) RunGemm(backend, run);
                if (Includes(onlyOperation, "fused-gelu")) RunFusedGelu(backend, cublasLt, run);
                if (Includes(onlyOperation, "batched-gemm")) RunBatchedGemm(backend, run);
                if (Includes(onlyOperation, "gemm-fp16")) RunFp16Gemm(backend, run);
                if (Includes(onlyOperation, "lora")) RunLoRA(backend, run);
                if (Includes(onlyOperation, "linear-ce-index")) RunCrossEntropy(backend, run);
                if (Includes(onlyOperation, "linear-backward-relu")) RunLinearBackward(backend, run);
                if (onlyOperation is null || onlyOperation is "dot" or "outer" or "batched-dot" or "strided-dot")
                    RunDenseVectors(backend, run, onlyOperation);
            }
            if (includePython)
            {
                GpuBenchmarkEnvironment.RequireIdleGpu("dense-linear-pytorch-start");
                RunPython(independentRuns, onlyOperation);
            }
            GpuBenchmarkEnvironment.RequireNoForeignCompute("dense-linear-end");
            GpuBenchmarkEnvironment.PrintSnapshot("dense-linear-end");
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previousDeterminism);
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
        }
    }

    private static void AddPackagedNativeRuntimeToPath()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return;
        string nativeRuntime = Path.Combine(
            AppContext.BaseDirectory, "runtimes", "win-x64", "native");
        if (!Directory.Exists(nativeRuntime)) return;
        string current = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        if (current.Split(Path.PathSeparator).Any(path =>
            string.Equals(path, nativeRuntime, StringComparison.OrdinalIgnoreCase)))
            return;
        Environment.SetEnvironmentVariable(
            "PATH", nativeRuntime + Path.PathSeparator + current);
    }

    private static void PrintEnvironmentFingerprint(int independentRuns)
    {
        using var runtime = new DirectPtxRuntime();
        _deviceFingerprint = runtime.DeviceFingerprint;
        Console.WriteLine("dense_linear_environment_json=" + JsonSerializer.Serialize(new
        {
            os = RuntimeInformation.OSDescription,
            framework = RuntimeInformation.FrameworkDescription,
            os_architecture = RuntimeInformation.OSArchitecture.ToString(),
            process_architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            processor_count = Environment.ProcessorCount,
            processor_identifier = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "n/a",
            server_gc = System.Runtime.GCSettings.IsServerGC,
            gpu = runtime.DeviceName,
            gpu_ordinal = runtime.DeviceOrdinal,
            gpu_uuid = runtime.DeviceUuid,
            compute_capability = $"{runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor}",
            max_threads_per_sm = runtime.MaxThreadsPerMultiprocessor,
            cuda_driver_version = runtime.DriverVersion,
            device_fingerprint = runtime.DeviceFingerprint,
            warmups = Warmups,
            samples = Samples,
            launches_per_device_sample = DeviceLaunches,
            independent_runs = independentRuns
        }));
    }

    private static void RunDecode(CudaBackend backend, CuBlasLtMatmul cublasLt, int run)
    {
        const int m = 1, k = 512, n = 2048;
        var random = new Random(20263600 + run);
        float[] inputHost = Values(random, m * k);
        float[] weightHost = Values(random, k * n, 0.0625f);
        float[] outputMajorHost = Transpose(weightHost, k, n);
        float[] biasHost = Values(random, n, 0.0625f);
        float[] expected = GemmOracle(inputHost, weightHost, biasHost, m, k, n, Gelu);
        using var input = backend.AllocateBuffer(inputHost);
        using var weight = backend.AllocateBuffer(weightHost);
        using var outputMajor = backend.AllocateBuffer(outputMajorHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var current = backend.AllocateBuffer(m * n);
        using var vendor = backend.AllocateBuffer(m * n);
        using var direct = backend.AllocateBuffer(m * n);

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "decode-gelu", "M1 K512 N2048",
            "AiDotNet NVRTC", 2d * m * k * n,
            () => backend.FusedLinearGELU(input, weight, bias, current, m, k, n),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);

        BenchmarkRoute(backend, run, "decode-gelu", "M1 K512 N2048",
            "NVIDIA cuBLASLt", 2d * m * k * n,
            () => cublasLt.MatmulFused(
                weight.Handle, n, k, false, input.Handle, m, false,
                IntPtr.Zero, vendor.Handle, bias.Handle, CublasLtEpilogue.GELUBias,
                stream: backend.DefaultStream.Handle,
                dtype: CublasDataType.Float32, computeType: CublasComputeType.Float32),
            () => MaximumError(backend.DownloadBuffer(vendor), expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFusedLinearGeluM1(k, n), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFusedLinearGeluM1(
                input, outputMajor, bias, direct, k, n))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFusedLinearAudit(k, n, out var audit));
        BenchmarkRoute(backend, run, "decode-gelu", "M1 K512 N2048",
            "Direct PTX", 2d * m * k * n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunGemm(CudaBackend backend, int run)
    {
        const int m = 64, k = 256, n = 256;
        var random = new Random(20263700 + run);
        float[] leftHost = Values(random, m * k);
        float[] rightHost = Values(random, k * n, 0.0625f);
        float[] expected = GemmOracle(leftHost, rightHost, null, m, k, n, Identity);
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(m * n);
        using var direct = backend.AllocateBuffer(m * n);

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "gemm-fp32", "M64 K256 N256",
            "NVIDIA cuBLAS", 2d * m * k * n,
            () => backend.Gemm(left, right, current, m, n, k),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxGemmTiled(
            m, k, n, DirectPtxLinearWeightLayout.InputMajor), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxGemmTiled(
                left, right, direct, m, k, n, DirectPtxLinearWeightLayout.InputMajor))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxGemmTiledAudit(
            m, k, n, DirectPtxLinearWeightLayout.InputMajor, 1, out var audit));
        BenchmarkRoute(backend, run, "gemm-fp32", "M64 K256 N256",
            "Direct PTX", 2d * m * k * n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunFusedGelu(CudaBackend backend, CuBlasLtMatmul cublasLt, int run)
    {
        const int m = 64, k = 256, n = 256;
        var random = new Random(20263800 + run);
        float[] inputHost = Values(random, m * k);
        float[] weightHost = Values(random, k * n, 0.0625f);
        float[] biasHost = Values(random, n, 0.0625f);
        float[] expected = GemmOracle(inputHost, weightHost, biasHost, m, k, n, Gelu);
        using var input = backend.AllocateBuffer(inputHost);
        using var weight = backend.AllocateBuffer(weightHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var current = backend.AllocateBuffer(m * n);
        using var vendor = backend.AllocateBuffer(m * n);
        using var direct = backend.AllocateBuffer(m * n);

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "fused-gelu", "M64 K256 N256",
            "AiDotNet NVRTC", 2d * m * k * n,
            () => backend.FusedLinearGELU(input, weight, bias, current, m, k, n),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);
        BenchmarkRoute(backend, run, "fused-gelu", "M64 K256 N256",
            "NVIDIA cuBLASLt", 2d * m * k * n,
            () => cublasLt.MatmulFused(
                weight.Handle, n, k, false, input.Handle, m, false,
                IntPtr.Zero, vendor.Handle, bias.Handle, CublasLtEpilogue.GELUBias,
                stream: backend.DefaultStream.Handle,
                dtype: CublasDataType.Float32, computeType: CublasComputeType.Float32),
            () => MaximumError(backend.DownloadBuffer(vendor), expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFusedLinearTiled(
            m, k, n, DirectPtxLinearActivation.GeluTanh,
            DirectPtxLinearWeightLayout.InputMajor), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFusedLinearTiled(
                input, weight, bias, direct, m, k, n,
                DirectPtxLinearActivation.GeluTanh,
                DirectPtxLinearWeightLayout.InputMajor))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFusedLinearTiledAudit(
            m, k, n, DirectPtxLinearActivation.GeluTanh,
            DirectPtxLinearWeightLayout.InputMajor, out var audit));
        BenchmarkRoute(backend, run, "fused-gelu", "M64 K256 N256",
            "Direct PTX", 2d * m * k * n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunBatchedGemm(CudaBackend backend, int run)
    {
        const int batch = 4, m = 64, k = 256, n = 256;
        var random = new Random(20263900 + run);
        float[] leftHost = Values(random, batch * m * k);
        float[] rightHost = Values(random, batch * k * n, 0.0625f);
        float[] expected = BatchedGemmOracle(leftHost, rightHost, batch, m, k, n);
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(batch * m * n);
        using var direct = backend.AllocateBuffer(batch * m * n);

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "batched-gemm", "B4 M64 K256 N256",
            "NVIDIA cuBLAS", 2d * batch * m * k * n,
            () => backend.BatchedGemm(left, right, current, m, n, k, batch),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxGemmTiled(
            m, k, n, DirectPtxLinearWeightLayout.InputMajor, batch), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxGemmTiled(
                left, right, direct, m, k, n,
                DirectPtxLinearWeightLayout.InputMajor, batch))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxGemmTiledAudit(
            m, k, n, DirectPtxLinearWeightLayout.InputMajor, batch, out var audit));
        BenchmarkRoute(backend, run, "batched-gemm", "B4 M64 K256 N256",
            "Direct PTX", 2d * batch * m * k * n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunFp16Gemm(CudaBackend backend, int run)
    {
        const int m = 16, k = 32, n = 16;
        var random = new Random(20264000 + run);
        ushort[] leftHost = HalfValues(random, m * k);
        ushort[] rightHost = HalfValues(random, k * n, 0.0625f);
        float[] expected = HalfGemmOracle(leftHost, rightHost, m, k, n);
        using var left = backend.AllocateByteBuffer(leftHost.Length * sizeof(ushort));
        using var right = backend.AllocateByteBuffer(rightHost.Length * sizeof(ushort));
        using var current = backend.AllocateBuffer(m * n);
        using var direct = backend.AllocateBuffer(m * n);
        backend.UploadBytes(left, Bytes(leftHost));
        backend.UploadBytes(right, Bytes(rightHost));

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "gemm-fp16", "M16 K32 N16",
            "NVIDIA cuBLAS GemmEx", 2d * m * k * n,
            () => backend.GemmFp16In32fOut(left, right, current, m, n, k),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFp16Gemm(m, n, k), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFp16Gemm(left, right, direct, m, n, k))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFp16GemmAudit(
            m, n, k, 1, false, false,
            DirectPtx16BitInputType.Float16, DirectPtxGemmOutputType.Float32,
            false, out var audit));
        BenchmarkRoute(backend, run, "gemm-fp16", "M16 K32 N16",
            "Direct PTX", 2d * m * k * n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunLoRA(CudaBackend backend, int run)
    {
        const int batch = 8, inputFeatures = 256, rank = 8, outputFeatures = 256;
        const float scaling = 0.125f;
        var random = new Random(20264100 + run);
        float[] inputHost = Values(random, batch * inputFeatures);
        float[] baseHost = Values(random, batch * outputFeatures);
        float[] aHost = Values(random, inputFeatures * rank, 0.0625f);
        float[] bHost = Values(random, rank * outputFeatures, 0.0625f);
        float[] expected = LoRAOracle(
            inputHost, baseHost, aHost, bHost,
            batch, inputFeatures, rank, outputFeatures, scaling);
        using var input = backend.AllocateBuffer(inputHost);
        using var baseOutput = backend.AllocateBuffer(baseHost);
        using var a = backend.AllocateBuffer(aHost);
        using var b = backend.AllocateBuffer(bHost);
        using var current = backend.AllocateBuffer(batch * outputFeatures);
        using var direct = backend.AllocateBuffer(batch * outputFeatures);

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "lora", "B8 I256 R8 O256",
            "AiDotNet NVRTC", 4d * batch * inputFeatures * rank,
            () => backend.FusedLoRAForward(
                input, baseOutput, a, b, current,
                batch, inputFeatures, rank, outputFeatures, scaling),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: false);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFusedLoRA(
            batch, inputFeatures, rank, outputFeatures, scaling), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFusedLoRA(
                input, baseOutput, a, b, direct,
                batch, inputFeatures, rank, outputFeatures, scaling))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFusedLoRAAudit(
            batch, inputFeatures, rank, outputFeatures, scaling, out var audit));
        BenchmarkRoute(backend, run, "lora", "B8 I256 R8 O256",
            "Direct PTX", 4d * batch * inputFeatures * rank, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunCrossEntropy(CudaBackend backend, int run)
    {
        const int rows = 4, hiddenDimension = 16, vocabulary = 32;
        var random = new Random(20264200 + run);
        float[] hiddenHost = Values(random, rows * hiddenDimension);
        float[] weightHost = Values(random, hiddenDimension * vocabulary, 0.0625f);
        float[] biasHost = Values(random, vocabulary, 0.03125f);
        float[] targetHost = [1f, 7f, 15f, 31f];
        float expected = CrossEntropyOracle(
            hiddenHost, weightHost, biasHost, targetHost,
            rows, hiddenDimension, vocabulary);
        using var hidden = backend.AllocateBuffer(hiddenHost);
        using var weight = backend.AllocateBuffer(weightHost);
        using var bias = backend.AllocateBuffer(biasHost);
        using var target = backend.AllocateBuffer(targetHost);
        using var current = backend.AllocateBuffer(1);
        using var direct = backend.AllocateBuffer(1);

        AiDotNetEngine.SetDeterministicMode(false);
        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "linear-ce-index", "B4 K16 V32",
            "AiDotNet NVRTC", 4d * rows * hiddenDimension * vocabulary,
            () => backend.FusedLinearCrossEntropyIndex(
                hidden, weight, bias, target, current,
                rows, hiddenDimension, vocabulary),
            () => Math.Abs(backend.DownloadBuffer(current)[0] - expected),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFusedLinearCrossEntropy(
            DirectPtxCrossEntropyTarget.Index,
            rows, hiddenDimension, vocabulary), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFusedLinearCrossEntropy(
                DirectPtxCrossEntropyTarget.Index,
                hidden, weight, bias, target, direct,
                rows, hiddenDimension, vocabulary))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFusedLinearCrossEntropyAudit(
            DirectPtxCrossEntropyTarget.Index,
            rows, hiddenDimension, vocabulary, out var audit));
        BenchmarkRoute(backend, run, "linear-ce-index", "B4 K16 V32",
            "Direct PTX", 4d * rows * hiddenDimension * vocabulary, Launch,
            () => Math.Abs(backend.DownloadBuffer(direct)[0] - expected),
            Resource(audit), capture: true);
    }

    private static void RunLinearBackward(CudaBackend backend, int run)
    {
        const int m = 64, k = 256, n = 256;
        var random = new Random(20264300 + run);
        float[] gradHost = Values(random, m * n);
        float[] inputHost = Values(random, m * k);
        float[] weightHost = Values(random, k * n, 0.0625f);
        float[] savedHost = Values(random, m * n, 0.25f);
        (float[] expectedInput, float[] expectedWeight, float[] expectedBias) =
            ReluBackwardOracle(gradHost, inputHost, weightHost, savedHost, m, k, n);
        using var grad = backend.AllocateBuffer(gradHost);
        using var input = backend.AllocateBuffer(inputHost);
        using var weight = backend.AllocateBuffer(weightHost);
        using var saved = backend.AllocateBuffer(savedHost);
        using var currentInput = backend.AllocateBuffer(m * k);
        using var currentWeight = backend.AllocateBuffer(k * n);
        using var currentBias = backend.AllocateBuffer(n);
        using var directInput = backend.AllocateBuffer(m * k);
        using var directWeight = backend.AllocateBuffer(k * n);
        using var directBias = backend.AllocateBuffer(n);

        double Error(IGpuBuffer di, IGpuBuffer dw, IGpuBuffer db) => Math.Max(
            MaximumError(backend.DownloadBuffer(di), expectedInput), Math.Max(
            MaximumError(backend.DownloadBuffer(dw), expectedWeight),
            MaximumError(backend.DownloadBuffer(db), expectedBias)));

        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "linear-backward-relu", "M64 K256 N256",
            "AiDotNet NVRTC", 4d * m * n * k + m * n,
            () => backend.FusedLinearReLUBackward(
                grad, input, weight, saved,
                currentInput, currentWeight, currentBias, m, k, n),
            () => Error(currentInput, currentWeight, currentBias),
            Resources.None, capture: true);

        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxFusedLinearBackward(
            m, k, n, DirectPtxLinearActivation.Relu), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxFusedLinearBackward(
                grad, input, weight, saved,
                directInput, directWeight, directBias,
                m, k, n, DirectPtxLinearActivation.Relu))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxFusedLinearBackwardAudits(
            m, k, n, DirectPtxLinearActivation.Relu, out var audits));
        BenchmarkRoute(backend, run, "linear-backward-relu", "M64 K256 N256",
            "Direct PTX", 4d * m * n * k + m * n, Launch,
            () => Error(directInput, directWeight, directBias),
            Resource(audits), capture: true);
    }

    private static void RunDenseVectors(CudaBackend backend, int run, string? onlyOperation)
    {
        var random = new Random(20264400 + run);
        if (Includes(onlyOperation, "dot")) RunDot(backend, run, random);
        if (Includes(onlyOperation, "outer")) RunOuter(backend, run, random);
        if (Includes(onlyOperation, "batched-dot")) RunBatchedDot(backend, run, random);
        if (Includes(onlyOperation, "strided-dot")) RunStridedDot(backend, run, random);
    }

    private static bool Includes(string? onlyOperation, string operation) =>
        onlyOperation is null || string.Equals(onlyOperation, operation, StringComparison.Ordinal);

    private static void RunDot(CudaBackend backend, int run, Random random)
    {
        const int length = 4096;
        float[] leftHost = Values(random, length);
        float[] rightHost = Values(random, length);
        float expected = DotOracle(leftHost, rightHost, 0, 1);
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(1);
        using var direct = backend.AllocateBuffer(1);
        AiDotNetEngine.SetDeterministicMode(false);
        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "dot", "K4096", "AiDotNet NVRTC",
            2d * length, () =>
            {
                backend.Fill(current, 0f, 1);
                backend.DotProduct(left, right, current, length);
            },
            () => Math.Abs(backend.DownloadBuffer(current)[0] - expected),
            Resources.None, capture: true, allowMissingEstablishedKernel: true);
        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxDenseVector(
            DirectPtxDenseVectorOperation.Dot, length), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxDotProduct(left, right, direct, length))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxDenseVectorAudit(
            DirectPtxDenseVectorOperation.Dot, length, 1, out var audit));
        BenchmarkRoute(backend, run, "dot", "K4096", "Direct PTX",
            2d * length, Launch,
            () => Math.Abs(backend.DownloadBuffer(direct)[0] - expected),
            Resource(audit), capture: true);
        AiDotNetEngine.SetDeterministicMode(false);
    }

    private static void RunOuter(CudaBackend backend, int run, Random random)
    {
        const int m = 64, n = 128;
        float[] leftHost = Values(random, m);
        float[] rightHost = Values(random, n);
        float[] expected = OuterOracle(leftHost, rightHost);
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(m * n);
        using var direct = backend.AllocateBuffer(m * n);
        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "outer", "M64 N128", "AiDotNet NVRTC",
            m * (double)n, () => backend.OuterProduct(left, right, current, m, n),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true);
        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxDenseVector(
            DirectPtxDenseVectorOperation.Outer, m, n), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxOuterProduct(left, right, direct, m, n))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxDenseVectorAudit(
            DirectPtxDenseVectorOperation.Outer, m, n, out var audit));
        BenchmarkRoute(backend, run, "outer", "M64 N128", "Direct PTX",
            m * (double)n, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunBatchedDot(CudaBackend backend, int run, Random random)
    {
        const int batch = 4, dimension = 512;
        float[] leftHost = Values(random, batch * dimension);
        float[] rightHost = Values(random, batch * dimension);
        var expected = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            double sum = 0;
            int offset = b * dimension;
            for (int i = 0; i < dimension; i++)
                sum += leftHost[offset + i] * (double)rightHost[offset + i];
            expected[b] = (float)sum;
        }
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(batch);
        using var direct = backend.AllocateBuffer(batch);
        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "batched-dot", "B4 K512", "AiDotNet NVRTC",
            2d * batch * dimension,
            () => backend.BatchedDotProduct(left, right, current, batch, dimension),
            () => MaximumError(backend.DownloadBuffer(current), expected),
            Resources.None, capture: true, allowMissingEstablishedKernel: true);
        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxBatchedVector(
            DirectPtxBatchedVectorOperation.Dot, batch, dimension), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxBatchDotProduct(
                left, right, direct, batch, dimension))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxBatchedVectorAudit(
            DirectPtxBatchedVectorOperation.Dot, batch, dimension, 1, out var audit));
        BenchmarkRoute(backend, run, "batched-dot", "B4 K512", "Direct PTX",
            2d * batch * dimension, Launch,
            () => MaximumError(backend.DownloadBuffer(direct), expected),
            Resource(audit), capture: true);
    }

    private static void RunStridedDot(CudaBackend backend, int run, Random random)
    {
        const int length = 512, offset = 511, step = -1;
        float[] leftHost = Values(random, length);
        float[] rightHost = Values(random, length);
        float expected = DotOracle(leftHost, rightHost, offset, step);
        using var left = backend.AllocateBuffer(leftHost);
        using var right = backend.AllocateBuffer(rightHost);
        using var current = backend.AllocateBuffer(1);
        using var direct = backend.AllocateBuffer(1);
        AiDotNetEngine.SetDeterministicMode(false);
        DirectPtxFeatureGate.TestOverride = false;
        BenchmarkRoute(backend, run, "strided-dot", "A512 B512 reverse",
            "AiDotNet NVRTC", 2d * length,
            () =>
            {
                backend.Fill(current, 0f, 1);
                backend.StridedDotProduct(
                    left, right, current, length, length, offset, step);
            },
            () => Math.Abs(backend.DownloadBuffer(current)[0] - expected),
            Resources.None, capture: true, allowMissingEstablishedKernel: true);
        EnableDirectExperiment();
        Require(backend.PrewarmDirectPtxStridedDot(length, length, offset, step), backend);
        void Launch()
        {
            if (!backend.TryDirectPtxStridedDotProduct(
                left, right, direct, length, length, offset, step))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }
        Require(backend.TryGetDirectPtxStridedDotAudit(
            length, length, offset, step, out var audit));
        BenchmarkRoute(backend, run, "strided-dot", "A512 B512 reverse",
            "Direct PTX", 2d * length, Launch,
            () => Math.Abs(backend.DownloadBuffer(direct)[0] - expected),
            Resource(audit), capture: true);
        AiDotNetEngine.SetDeterministicMode(false);
    }

    private static void BenchmarkRoute(
        CudaBackend backend,
        int run,
        string operation,
        string shape,
        string method,
        double work,
        Action launch,
        Func<double> error,
        Resources resources,
        bool capture,
        bool allowMissingEstablishedKernel = false)
    {
        try
        {
            launch();
        }
        catch (InvalidOperationException ex) when (
            allowMissingEstablishedKernel &&
            ex.Message.StartsWith("CUDA kernel not found:", StringComparison.Ordinal))
        {
            Console.WriteLine("dense_linear_status_json=" + JsonSerializer.Serialize(new
            {
                run,
                operation,
                shape,
                method,
                status = "unsupported",
                reason = ex.Message
            }));
            return;
        }
        backend.Synchronize();
        double maxError = error();
        Distribution device = MeasureDevice(backend, launch);
        Distribution endToEnd = MeasureEndToEnd(backend, launch);
        long allocation = MeasureAllocation(backend, launch);
        Print(run, operation, shape, method, work, device, endToEnd,
            allocation, temporaryBytes: 0, maxError, resources);

        if (!capture) return;
        IntPtr graph = backend.CaptureGraph(launch);
        if (graph == IntPtr.Zero)
        {
            Console.WriteLine("dense_linear_graph_status_json=" + JsonSerializer.Serialize(new
            {
                run,
                operation,
                shape,
                method,
                status = "unsupported",
                reason = "CUDA graph capture returned a null executable graph"
            }));
            return;
        }
        try
        {
            // Queue replay on the measured stream. LaunchCapturedGraph performs
            // a context-wide host barrier, which cannot be enclosed by CUDA
            // events without measuring an idle host gap.
            void GraphLaunch() => backend.EnqueueCapturedGraph(graph);
            GraphLaunch();
            backend.Synchronize();
            double graphError = error();
            Distribution graphDevice = MeasureDevice(backend, GraphLaunch);
            Distribution graphEndToEnd = MeasureEndToEnd(backend, GraphLaunch);
            long graphAllocation = MeasureAllocation(backend, GraphLaunch);
            Print(run, operation, shape, method + " graph", work,
                graphDevice, graphEndToEnd, graphAllocation,
                temporaryBytes: 0, graphError, resources);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
    }

    private static Distribution MeasureDevice(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
        var values = new double[Samples];
        using IGpuEvent start = backend.CreateEvent(enableTiming: true);
        using IGpuEvent end = backend.CreateEvent(enableTiming: true);
        for (int sample = 0; sample < values.Length; sample++)
        {
            backend.RecordEvent(start, backend.DefaultStream);
            for (int i = 0; i < DeviceLaunches; i++) launch();
            backend.RecordEvent(end, backend.DefaultStream);
            end.Synchronize();
            values[sample] = backend.GetEventElapsedTime(start, end) * 1_000d / DeviceLaunches;
        }
        return Summarize(values);
    }

    private static Distribution MeasureEndToEnd(CudaBackend backend, Action launch)
    {
        for (int i = 0; i < Warmups; i++) launch();
        backend.Synchronize();
        var values = new double[Samples];
        double scale = 1_000_000d / Stopwatch.Frequency;
        for (int sample = 0; sample < values.Length; sample++)
        {
            long start = Stopwatch.GetTimestamp();
            launch();
            backend.Synchronize();
            values[sample] = (Stopwatch.GetTimestamp() - start) * scale;
        }
        return Summarize(values);
    }

    private static long MeasureAllocation(CudaBackend backend, Action launch)
    {
        // Cross the tiered-JIT promotion threshold before opening the exact
        // per-thread allocation window. Otherwise a one-time delegate/code
        // promotion can be misreported as a few bytes per hot launch.
        for (int i = 0; i < 64; i++) launch();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++) launch();
        long result = (GC.GetAllocatedBytesForCurrentThread() - before) / Samples;
        backend.Synchronize();
        return result;
    }

    private static Distribution Summarize(double[] values)
    {
        Array.Sort(values);
        return new Distribution(values.Average(), Percentile(values, 0.5),
            Percentile(values, 0.95), Percentile(values, 0.99));
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static void PrintHeader()
    {
        Console.WriteLine(
            "run operation            shape                  method                     " +
            "dev mean/med/p95/p99 us         e2e mean/med/p95/p99 us       " +
            "TFLOPS GFLOPS allocB tmpB error     R/S/L/B");
        Console.WriteLine(new string('-', 190));
    }

    private static void Print(
        int run, string operation, string shape, string method, double work,
        Distribution device, Distribution endToEnd,
        long allocation, long temporaryBytes, double maxError, Resources resources)
    {
        double gflops = work / (device.Median * 1e-6) / 1e9;
        Console.WriteLine(
            $"{run,3} {operation,-20} {shape,-22} {method,-26} " +
            $"{device.Mean,6:F2}/{device.Median,6:F2}/{device.P95,6:F2}/{device.P99,6:F2}  " +
            $"{endToEnd.Mean,6:F2}/{endToEnd.Median,6:F2}/{endToEnd.P95,6:F2}/{endToEnd.P99,6:F2}  " +
            $"{gflops / 1_000d,6:F3} {gflops,7:F2} {allocation,6} {temporaryBytes,4} " +
            $"{maxError,8:E1} {resources.Registers,2}/{resources.StaticShared,4}/" +
            $"{resources.Local,2}/{resources.ActiveBlocks,2}");

        Console.WriteLine("dense_linear_evidence_json=" + JsonSerializer.Serialize(new
        {
            run,
            operation,
            shape,
            method,
            work_flops = work,
            device_mean_us = device.Mean,
            device_median_us = device.Median,
            device_p95_us = device.P95,
            device_p99_us = device.P99,
            e2e_mean_us = endToEnd.Mean,
            e2e_median_us = endToEnd.Median,
            e2e_p95_us = endToEnd.P95,
            e2e_p99_us = endToEnd.P99,
            tflops = gflops / 1_000d,
            gflops,
            managed_bytes = allocation,
            temporary_device_bytes = temporaryBytes,
            max_error = maxError,
            tolerance = Tolerance,
            registers = resources.Registers,
            static_shared_bytes = resources.StaticShared,
            local_bytes_per_thread = resources.Local,
            active_blocks_per_sm = resources.ActiveBlocks,
            blueprint = resources.Blueprint,
            device_fingerprint = resources.DeviceFingerprint,
            ptx_sha256 = resources.PtxSha256
        }));
    }

    private static Resources Resource(DirectPtxKernelAudit audit) => new(
        audit.Function.RegistersPerThread,
        audit.Function.StaticSharedBytes,
        audit.Function.LocalBytesPerThread,
        audit.ActiveBlocksPerMultiprocessor,
        audit.BlueprintId,
        audit.DeviceFingerprint,
        audit.PtxSha256);

    private static Resources Resource(IReadOnlyList<DirectPtxKernelAudit> audits)
    {
        if (audits.Count == 0) return Resources.None;
        return new Resources(
            audits.Max(audit => audit.Function.RegistersPerThread),
            audits.Max(audit => audit.Function.StaticSharedBytes),
            audits.Max(audit => audit.Function.LocalBytesPerThread),
            audits.Min(audit => audit.ActiveBlocksPerMultiprocessor),
            string.Join("+", audits.Select(audit => audit.BlueprintId)),
            audits[0].DeviceFingerprint,
            string.Join("+", audits.Select(audit => audit.PtxSha256)));
    }

    private static void RunPython(int runs, string? onlyOperation)
    {
        string script = Path.Combine(
            AppContext.BaseDirectory, "BaselineRunners", "py",
            "run_direct_ptx_dense_linear_full_competitors.py");
        if (!File.Exists(script))
            script = Path.Combine(AppContext.BaseDirectory,
                "run_direct_ptx_dense_linear_full_competitors.py");
        if (!File.Exists(script))
            throw new FileNotFoundException("The issue #836 PyTorch harness was not copied.", script);
        var start = new ProcessStartInfo
        {
            FileName = Environment.GetEnvironmentVariable("PYTHON") ?? "python",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--runs");
        start.ArgumentList.Add(runs.ToString(CultureInfo.InvariantCulture));
        start.ArgumentList.Add("--json-lines");
        if (onlyOperation is not null)
        {
            start.ArgumentList.Add("--only");
            start.ArgumentList.Add(onlyOperation);
        }
        using Process process = Process.Start(start) ??
            throw new InvalidOperationException("Could not start the PyTorch CUDA baseline.");
        while (process.StandardOutput.ReadLine() is { } line)
            Console.WriteLine("dense_linear_external_json=" + line);
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"PyTorch CUDA baseline exited {process.ExitCode}: {stderr}");
    }

    private static void EnableDirectExperiment()
    {
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = true;
    }

    private static void Require(bool condition, CudaBackend backend)
    {
        if (!condition) throw new InvalidOperationException(backend.DirectPtxLastError);
    }

    private static void Require(bool condition)
    {
        if (!condition) throw new InvalidOperationException("Direct PTX audit was not found.");
    }

    private static float[] Values(Random random, int count, float scale = 0.125f) =>
        Enumerable.Range(0, count)
            .Select(_ => (random.NextSingle() * 2f - 1f) * scale).ToArray();

    private static ushort[] HalfValues(Random random, int count, float scale = 0.125f) =>
        Enumerable.Range(0, count)
            .Select(_ => BitConverter.HalfToUInt16Bits(
                (Half)((random.NextSingle() * 2f - 1f) * scale))).ToArray();

    private static byte[] Bytes(ushort[] values)
    {
        var bytes = new byte[values.Length * sizeof(ushort)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float Half(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);
    private static float Identity(float value) => value;
    private static float Gelu(float value) => 0.5f * value *
        (1f + MathF.Tanh(0.7978845608f * (value + 0.044715f * value * value * value)));

    private static float[] GemmOracle(
        float[] left, float[] right, float[]? bias,
        int m, int k, int n, Func<float, float> activation)
    {
        var output = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
        {
            double sum = bias is null ? 0 : bias[column];
            for (int inner = 0; inner < k; inner++)
                sum += left[row * k + inner] * (double)right[inner * n + column];
            output[row * n + column] = activation((float)sum);
        }
        return output;
    }

    private static float[] BatchedGemmOracle(
        float[] left, float[] right, int batch, int m, int k, int n)
    {
        var output = new float[batch * m * n];
        for (int b = 0; b < batch; b++)
        for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
        {
            double sum = 0;
            for (int inner = 0; inner < k; inner++)
                sum += left[(b * m + row) * k + inner] *
                    (double)right[(b * k + inner) * n + column];
            output[(b * m + row) * n + column] = (float)sum;
        }
        return output;
    }

    private static float[] HalfGemmOracle(
        ushort[] left, ushort[] right, int m, int k, int n)
    {
        var output = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
        {
            float sum = 0;
            for (int inner = 0; inner < k; inner++)
                sum += Half(left[row * k + inner]) * Half(right[inner * n + column]);
            output[row * n + column] = sum;
        }
        return output;
    }

    private static float[] LoRAOracle(
        float[] input, float[] baseOutput, float[] a, float[] b,
        int batch, int inputFeatures, int rank, int outputFeatures, float scaling)
    {
        var output = new float[baseOutput.Length];
        for (int row = 0; row < batch; row++)
        for (int column = 0; column < outputFeatures; column++)
        {
            double update = 0;
            for (int r = 0; r < rank; r++)
            {
                double projection = 0;
                for (int k = 0; k < inputFeatures; k++)
                    projection += input[row * inputFeatures + k] * (double)a[k * rank + r];
                update += projection * b[r * outputFeatures + column];
            }
            output[row * outputFeatures + column] =
                baseOutput[row * outputFeatures + column] + (float)(scaling * update);
        }
        return output;
    }

    private static float CrossEntropyOracle(
        float[] hidden, float[] weight, float[] bias, float[] targets,
        int rows, int hiddenDimension, int vocabulary)
    {
        double total = 0;
        var logits = new double[vocabulary];
        for (int row = 0; row < rows; row++)
        {
            double maximum = double.NegativeInfinity;
            for (int column = 0; column < vocabulary; column++)
            {
                double value = bias[column];
                for (int k = 0; k < hiddenDimension; k++)
                    value += hidden[row * hiddenDimension + k] *
                        (double)weight[k * vocabulary + column];
                logits[column] = value;
                maximum = Math.Max(maximum, value);
            }
            double sum = 0;
            for (int column = 0; column < vocabulary; column++)
                sum += Math.Exp(logits[column] - maximum);
            total += maximum + Math.Log(sum) - logits[(int)targets[row]];
        }
        return (float)(total / rows);
    }

    private static (float[] GradInput, float[] GradWeight, float[] GradBias)
        ReluBackwardOracle(
            float[] grad, float[] input, float[] weight, float[] saved,
            int m, int k, int n)
    {
        var masked = new double[m * n];
        for (int i = 0; i < masked.Length; i++) masked[i] = saved[i] > 0 ? grad[i] : 0;
        var gradInput = new float[m * k];
        var gradWeight = new float[k * n];
        var gradBias = new float[n];
        for (int row = 0; row < m; row++)
        for (int inner = 0; inner < k; inner++)
        {
            double sum = 0;
            for (int column = 0; column < n; column++)
                sum += masked[row * n + column] * weight[inner * n + column];
            gradInput[row * k + inner] = (float)sum;
        }
        for (int inner = 0; inner < k; inner++)
        for (int column = 0; column < n; column++)
        {
            double sum = 0;
            for (int row = 0; row < m; row++)
                sum += input[row * k + inner] * masked[row * n + column];
            gradWeight[inner * n + column] = (float)sum;
        }
        for (int column = 0; column < n; column++)
        {
            double sum = 0;
            for (int row = 0; row < m; row++) sum += masked[row * n + column];
            gradBias[column] = (float)sum;
        }
        return (gradInput, gradWeight, gradBias);
    }

    private static float[] OuterOracle(float[] left, float[] right)
    {
        var output = new float[left.Length * right.Length];
        for (int row = 0; row < left.Length; row++)
        for (int column = 0; column < right.Length; column++)
            output[row * right.Length + column] = left[row] * right[column];
        return output;
    }

    private static float DotOracle(
        float[] left, float[] right, int rightOffset, int rightStep,
        int length = -1)
    {
        int count = length < 0 ? left.Length : length;
        double sum = 0;
        for (int i = 0; i < count; i++)
            sum += left[i] * (double)right[rightOffset + i * rightStep];
        return (float)sum;
    }

    private static float[] Transpose(float[] inputMajor, int k, int n)
    {
        var outputMajor = new float[inputMajor.Length];
        for (int inner = 0; inner < k; inner++)
        for (int output = 0; output < n; output++)
            outputMajor[output * k + inner] = inputMajor[inner * n + output];
        return outputMajor;
    }

    private static double MaximumError(float[] actual, float[] expected)
    {
        if (actual.Length != expected.Length) return double.PositiveInfinity;
        double maximum = 0;
        for (int i = 0; i < actual.Length; i++)
            maximum = Math.Max(maximum, Math.Abs(actual[i] - expected[i]));
        return maximum;
    }
}
