using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>The operand arrangement a #1261 grid case times.</summary>
internal enum GemmGridVariant
{
    /// <summary>Row-major <c>C[M,N] = A[M,K] @ B[K,N]</c>.</summary>
    Standard,

    /// <summary><c>C = A @ B^T</c> with B stored <c>[N,K]</c> (linear-layer weights, attention <c>Q K^T</c>).</summary>
    TransposeB,

    /// <summary><c>C = A^T @ B</c> with A stored <c>[K,M]</c> (the weight gradient <c>X^T dY</c>).</summary>
    TransposeA,

    /// <summary><c>C = 0.5 A @ B + 0.75 C</c>: alpha and beta other than 1 and 0.</summary>
    AlphaBeta,

    /// <summary>Strided-batched <c>C[b] = A[b] @ B[b]</c>.</summary>
    StridedBatched,
}

/// <summary>One shape of the #1261 grid. <see cref="Batch"/> is 1 for every variant but strided-batched.</summary>
internal readonly record struct GemmGridCase(string Name, GemmGridVariant Variant, int Batch, int M, int K, int N)
{
    internal double Work => 2d * Batch * M * K * N;

    internal string Shape
    {
        get
        {
            string core = $"M{M} K{K} N{N}";
            return Variant switch
            {
                GemmGridVariant.Standard => core,
                GemmGridVariant.TransposeB => core + " A@B^T",
                GemmGridVariant.TransposeA => core + " A^T@B",
                GemmGridVariant.AlphaBeta => core + " a=0.5 b=0.75",
                GemmGridVariant.StridedBatched => $"B{Batch} " + core,
                _ => throw new ArgumentOutOfRangeException(nameof(Variant), Variant, null),
            };
        }
    }
}

/// <summary>
/// #1261 (phase A of #377): where CUDA GEMM stands before any kernel work. Every CUDA GEMM runs on cuBLAS today
/// (both direct-PTX GEMM kernels report <c>IsPromotedShape =&gt; false</c>), so this times, per shape:
/// <list type="bullet">
/// <item>cuBLAS as production configures it: TF32 tensor-op math on Ampere+ outside deterministic mode (on older
/// GPUs, where TF32 does not exist, the default FP32 math);</item>
/// <item>cuBLAS in strict FP32 (<c>CUBLAS_PEDANTIC_MATH</c>, what deterministic mode selects): the fair bar for
/// an FP32 kernel, and the precision PyTorch's default <c>float32_matmul_precision("highest")</c> uses;</item>
/// <item>cuBLASLt with FP32 compute (its own heuristic, no workspace), through <see cref="CuBlasLtMatmul"/>;</item>
/// <item>each direct-PTX GEMM kernel where its shape contract admits the shape, else an explicit unsupported row
/// that says why;</item>
/// <item>PyTorch (eager, CUDA graph, torch.compile) through the shared Python lane.</item>
/// </list>
/// Opt-in only (<c>--only=gemm-grid</c>, or <c>--only=gemm-grid-NAME</c> for one shape): the large shapes make it far
/// longer than the default matrix. <c>Profiling/summarize_cuda_gemm_grid.py</c> turns the evidence of several process
/// launches into the median/spread table.
/// </summary>
internal static partial class DirectPtxDenseLinearFullExperiment
{
    internal const string GemmGridOperation = "gemm-grid";
    private const string GemmGridPrefix = GemmGridOperation + "-";

    // Mirrors GEMM_GRID in BaselineRunners/py/run_direct_ptx_dense_linear_full_competitors.py; the summarizer flags
    // any operation one side reports and the other does not.
    internal static readonly GemmGridCase[] GemmGrid =
    {
        // Square, 64 to 8192.
        Plain("sq64", 64, 64, 64), Plain("sq128", 128, 128, 128), Plain("sq256", 256, 256, 256),
        Plain("sq512", 512, 512, 512), Plain("sq1024", 1024, 1024, 1024), Plain("sq2048", 2048, 2048, 2048),
        Plain("sq4096", 4096, 4096, 4096), Plain("sq8192", 8192, 8192, 8192),
        // Skinny (decode, M = 1..64) and tall-and-thin (N = 16) against a 4096x4096 weight.
        Plain("skinny-m4", 4, 4096, 4096), Plain("skinny-m64", 64, 4096, 4096), Plain("tall-n16", 4096, 4096, 16),
        // LLM projections (Llama-7B): QKV/O 4096x4096, MLP up 4096x11008, MLP down 11008x4096, at M in {1,16,128,2048}.
        Plain("llm-qkv-m1", 1, 4096, 4096), Plain("llm-qkv-m16", 16, 4096, 4096),
        Plain("llm-qkv-m128", 128, 4096, 4096), Plain("llm-qkv-m2048", 2048, 4096, 4096),
        Plain("llm-up-m1", 1, 4096, 11008), Plain("llm-up-m16", 16, 4096, 11008),
        Plain("llm-up-m128", 128, 4096, 11008), Plain("llm-up-m2048", 2048, 4096, 11008),
        Plain("llm-down-m1", 1, 11008, 4096), Plain("llm-down-m16", 16, 11008, 4096),
        Plain("llm-down-m128", 128, 11008, 4096), Plain("llm-down-m2048", 2048, 11008, 4096),
        // Vision im2col (ResNet-50, batch 1): M = output pixels, K = Cin*kh*kw, N = Cout.
        Plain("vision-stem-112x112-c64", 12544, 147, 64), Plain("vision-56x56-c64", 3136, 576, 64),
        Plain("vision-28x28-c128", 784, 1152, 128), Plain("vision-14x14-c256", 196, 2304, 256),
        Plain("vision-7x7-c512", 49, 4608, 512),
        // Sizes no tile divides.
        Plain("odd-1000", 1000, 1000, 1000), Plain("odd-127x513x255", 127, 513, 255),
        // Strided-batched: attention scores and context per head, and a mid-size batch both PTX kernels admit.
        new("batched-attn-qk-b32", GemmGridVariant.StridedBatched, 32, 512, 64, 512),
        new("batched-attn-pv-b32", GemmGridVariant.StridedBatched, 32, 512, 512, 64),
        new("batched-b16-256", GemmGridVariant.StridedBatched, 16, 256, 256, 256),
        // Transposes.
        new("nt-1024", GemmGridVariant.TransposeB, 1, 1024, 1024, 1024),
        new("nt-llm-m2048", GemmGridVariant.TransposeB, 1, 2048, 4096, 4096),
        new("tn-1024", GemmGridVariant.TransposeA, 1, 1024, 1024, 1024),
        new("tn-wgrad-4096", GemmGridVariant.TransposeA, 1, 4096, 2048, 4096),
        // alpha/beta other than 1/0.
        new("alphabeta-1024", GemmGridVariant.AlphaBeta, 1, 1024, 1024, 1024),
        new("alphabeta-llm-m2048", GemmGridVariant.AlphaBeta, 1, 2048, 4096, 4096),
    };

    private const float GridAlpha = 0.5f;
    private const float GridBeta = 0.75f;

    // Checked output elements per lane: an fp64 oracle over the whole 8192^3 output would take minutes on the host.
    private const int GridErrorSamples = 512;
    private const double Fp32GridTolerance = 2e-4;
    private const double Tf32GridTolerance = 1e-2;

    // A device sample spans about 2e10 FLOP (~5 ms at 4 TFLOP/s), capped at DeviceLaunches launches, so small shapes
    // are not timer-bound and the largest do not take minutes. The 1,000-operation CUDA-graph lane runs only up to
    // 0.2 GFLOP: above that it is tens of seconds of sustained load that pushes the GPU past the evidence temperature
    // ceiling and adds nothing, because event timing over several launches already amortizes launch overhead there.
    private const double GridFlopsPerDeviceSample = 2e10;
    private const double GridGraphWorkCeiling = 2e8;

    // The cooldown target before every route. The default suits a desktop card that idles in the 40s; a laptop GPU
    // that idles near 55 C needs a higher target, set through this variable (the evidence ceiling is unchanged).
    private const string CooldownVariable = "AIDOTNET_GEMM_GRID_COOLDOWN_CELSIUS";
    private const int DefaultCooldownCelsius = 55;

    private static GemmGridCase Plain(string name, int m, int k, int n) =>
        new(name, GemmGridVariant.Standard, 1, m, k, n);

    /// <summary>True when <paramref name="onlyOperation"/> selects the whole grid or one of its shapes.</summary>
    internal static bool SelectsGemmGrid(string? onlyOperation) =>
        onlyOperation is not null &&
        (onlyOperation == GemmGridOperation || onlyOperation.StartsWith(GemmGridPrefix, StringComparison.Ordinal));

    private static void RunGemmGrid(CudaBackend backend, CuBlasLtMatmul cublasLt, int run, string onlyOperation)
    {
        GemmGridCase[] cases = onlyOperation == GemmGridOperation
            ? GemmGrid
            : GemmGrid.Where(c => GemmGridPrefix + c.Name == onlyOperation).ToArray();
        if (cases.Length == 0)
            throw new ArgumentException(
                $"Unknown GEMM grid shape '{onlyOperation}'. Known: {string.Join(", ", GemmGrid.Select(c => c.Name))}.");

        int cooldownCelsius = ReadCooldownCelsius();
        using var directRuntime = new DirectPtxRuntime(backend.CudaContextHandle, () => backend.DefaultStream.Handle);
        int major = directRuntime.ComputeCapabilityMajor, minor = directRuntime.ComputeCapabilityMinor;
        var device = new GridDevice(major >= 8, DirectPtxArchitecture.HasValidatedFusedLinear(major, minor),
            $"sm_{major}{minor}");

        foreach (GemmGridCase gridCase in cases)
            RunGemmGridCase(backend, cublasLt, directRuntime, run, gridCase, cooldownCelsius, device);
    }

    private readonly record struct GridDevice(bool HasTf32, bool DirectGateOpen, string Architecture);

    private static void RunGemmGridCase(
        CudaBackend backend, CuBlasLtMatmul cublasLt, DirectPtxRuntime directRuntime, int run, GemmGridCase gridCase,
        int cooldownCelsius, GridDevice device)
    {
        var (name, variant, batch, m, k, n) = gridCase;
        var random = new Random(20261261 + run * 101 + m * 7 + n + batch * 13 + (int)variant * 1009);
        // Logical operands A[b,M,K], B[b,K,N] and C0[M,N]; the device copies are laid out per variant below.
        float[] leftHost = Values(random, batch * m * k);
        float[] rightHost = Values(random, batch * k * n, 0.0625f);
        float[] initialOutput = variant == GemmGridVariant.AlphaBeta ? Values(random, m * n) : new float[batch * m * n];
        var samples = SampleOutputs(random, batch, m, n);
        double[] oracle = SampledOracle(gridCase, leftHost, rightHost, initialOutput, samples);
        string shape = gridCase.Shape;
        string operation = GemmGridPrefix + name;
        int launches = (int)Math.Clamp(Math.Ceiling(GridFlopsPerDeviceSample / gridCase.Work), 1, DeviceLaunches);
        bool capture = gridCase.Work <= GridGraphWorkCeiling;

        float[] leftDevice = variant == GemmGridVariant.TransposeA ? Transpose(leftHost, m, k) : leftHost;
        float[] rightDevice = variant == GemmGridVariant.TransposeB ? Transpose(rightHost, k, n) : rightHost;
        using var left = backend.AllocateBuffer(leftDevice);
        using var right = backend.AllocateBuffer(rightDevice);

        void Route(string method, Action<IGpuBuffer> launch, Resources resources, double tolerance)
        {
            // A fresh output per lane: alpha/beta reads C, so every lane must start from the same C0.
            using IGpuBuffer output = backend.AllocateBuffer(initialOutput);
            GpuBenchmarkEnvironment.WaitForCooldown(operation, cooldownCelsius);
            BenchmarkRoute(backend, run, operation, shape, method, gridCase.Work, () => launch(output),
                () => SampledRelativeError(backend.DownloadBuffer(output), samples, oracle, m, n), resources, capture,
                tolerance: tolerance, deviceLaunches: launches);
            PrintGridClock(run, operation, method);
        }

        DirectPtxFeatureGate.TestOverride = false;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = false;
        AiDotNetEngine.SetDeterministicMode(false);
        Route(device.HasTf32 ? "cuBLAS TF32 (prod default)" : "cuBLAS default math (prod default)",
            output => CublasLaunch(backend, gridCase, left, right, output), Resources.None,
            device.HasTf32 ? Tf32GridTolerance : Fp32GridTolerance);

        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            Route("cuBLAS FP32 strict", output => CublasLaunch(backend, gridCase, left, right, output),
                Resources.None, Fp32GridTolerance);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(false);
        }

        string? cublasLtGap = CublasLtUnsupportedReason(gridCase);
        if (cublasLtGap is null)
            Route("cuBLASLt FP32", output => CublasLtLaunch(backend, cublasLt, gridCase, left, right, output),
                Resources.None, Fp32GridTolerance);
        else
            PrintUnsupported(run, operation, shape, "cuBLASLt FP32", cublasLtGap);

        RunTiledLane(backend, run, gridCase, operation, left, right, device, Route);
        RunGemm64Lane(backend, directRuntime, run, gridCase, operation, left, right, device, Route);
    }

    private static void CublasLaunch(
        CudaBackend backend, GemmGridCase gridCase, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output)
    {
        var (_, variant, batch, m, k, n) = gridCase;
        switch (variant)
        {
            case GemmGridVariant.Standard: backend.Gemm(left, right, output, m, n, k); break;
            case GemmGridVariant.TransposeB: backend.MatMulTransposed(left, right, output, m, n, k); break;
            case GemmGridVariant.TransposeA: backend.MatMulTransposedA(left, right, output, m, n, k); break;
            case GemmGridVariant.AlphaBeta: backend.Gemm(left, right, output, m, n, k, GridAlpha, GridBeta); break;
            case GemmGridVariant.StridedBatched: backend.BatchedGemm(left, right, output, m, n, k, batch); break;
            default: throw new ArgumentOutOfRangeException(nameof(gridCase), variant, null);
        }
    }

    /// <summary>
    /// Why <see cref="CuBlasLtMatmul"/> cannot time <paramref name="gridCase"/>, or null when it can. The wrapper sets
    /// no batch count or stride, and it describes a transposed operand with its rows and columns unswapped
    /// (<c>rows=m, cols=k, ld=k</c> rather than <c>rows=k, cols=m</c>), which is only right when the two are equal.
    /// </summary>
    private static string? CublasLtUnsupportedReason(GemmGridCase gridCase) => gridCase.Variant switch
    {
        GemmGridVariant.StridedBatched =>
            "CuBlasLtMatmul exposes no batch count or batch stride (cuBLASLt itself supports strided batches)",
        GemmGridVariant.TransposeB when gridCase.N != gridCase.K =>
            "CuBlasLtMatmul's transposed-operand layout is only correct when that operand is square",
        GemmGridVariant.TransposeA when gridCase.M != gridCase.K =>
            "CuBlasLtMatmul's transposed-operand layout is only correct when that operand is square",
        _ => null,
    };

    // Row-major C[M,N] = op(A) op(B) is column-major C^T[N,M] = op(B)^T op(A)^T, so B is cuBLASLt's first operand.
    private static void CublasLtLaunch(
        CudaBackend backend, CuBlasLtMatmul cublasLt, GemmGridCase gridCase,
        IGpuBuffer left, IGpuBuffer right, IGpuBuffer output)
    {
        var (_, variant, _, m, k, n) = gridCase;
        bool alphaBeta = variant == GemmGridVariant.AlphaBeta;
        cublasLt.MatmulFused(
            right.Handle, n, k, variant == GemmGridVariant.TransposeB,
            left.Handle, m, variant == GemmGridVariant.TransposeA,
            alphaBeta ? output.Handle : IntPtr.Zero, output.Handle, IntPtr.Zero, CublasLtEpilogue.Default,
            alpha: alphaBeta ? GridAlpha : 1f, beta: alphaBeta ? GridBeta : 0f,
            stream: backend.DefaultStream.Handle,
            dtype: CublasDataType.Float32, computeType: CublasComputeType.Float32);
    }

    private static void RunTiledLane(
        CudaBackend backend, int run, GemmGridCase gridCase, string operation,
        IGpuBuffer left, IGpuBuffer right, GridDevice device,
        Action<string, Action<IGpuBuffer>, Resources, double> route)
    {
        const string method = "Direct PTX tiled 16x16x32";
        var (_, variant, batch, m, k, n) = gridCase;
        string shape = gridCase.Shape;
        DirectPtxLinearWeightLayout? layout = variant switch
        {
            GemmGridVariant.Standard or GemmGridVariant.StridedBatched => DirectPtxLinearWeightLayout.InputMajor,
            GemmGridVariant.TransposeB => DirectPtxLinearWeightLayout.OutputMajor,
            _ => null,
        };
        if (layout is not { } weightLayout)
        {
            PrintUnsupported(run, operation, shape, method, variant == GemmGridVariant.AlphaBeta
                ? "the kernel has no alpha/beta epilogue"
                : "the kernel has no transposed-A operand");
            return;
        }
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) || batch > 64)
        {
            PrintUnsupported(run, operation, shape, method,
                "shape outside the kernel's contract (M in 64..2048, K/N in 256..4096, powers of two, batch <= 64)");
            return;
        }
        if (!device.DirectGateOpen)
        {
            PrintUnsupported(run, operation, shape, method,
                $"validated only on sm_86, and its PTX uses cp.async (sm_80+), so it cannot run on {device.Architecture}");
            return;
        }

        EnableDirectExperiment();
        if (!backend.PrewarmDirectPtxGemmTiled(m, k, n, weightLayout, batch) ||
            !backend.TryGetDirectPtxGemmTiledAudit(m, k, n, weightLayout, batch, out var audit))
        {
            PrintUnsupported(run, operation, shape, method, backend.DirectPtxLastError ?? "prewarm refused the shape");
            return;
        }
        route(method, output =>
        {
            if (!backend.TryDirectPtxGemmTiled(left, right, output, m, k, n, weightLayout, batch))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }, Resource(audit), Fp32GridTolerance);
    }

    private static void RunGemm64Lane(
        CudaBackend backend, DirectPtxRuntime directRuntime, int run, GemmGridCase gridCase, string operation,
        IGpuBuffer left, IGpuBuffer right, GridDevice device,
        Action<string, Action<IGpuBuffer>, Resources, double> route)
    {
        const string method = "Direct PTX gemm 64x64x8";
        var (_, variant, _, m, k, n) = gridCase;
        string shape = gridCase.Shape;
        if (variant != GemmGridVariant.Standard)
        {
            PrintUnsupported(run, operation, shape, method,
                "the kernel computes only C = A @ B (no batch, transpose or alpha/beta)");
            return;
        }
        if (!PtxGemmKernel.IsSupportedShape(m, k, n))
        {
            PrintUnsupported(run, operation, shape, method,
                "shape outside the kernel's contract (M in 64..2048, K/N in 256..4096, powers of two)");
            return;
        }

        if (device.DirectGateOpen)
        {
            EnableDirectExperiment();
            if (!backend.PrewarmDirectPtxGemm(m, k, n) || !backend.TryGetDirectPtxGemmAudit(m, k, n, out var audit))
            {
                PrintUnsupported(run, operation, shape, method, backend.DirectPtxLastError ?? "prewarm refused the shape");
                return;
            }
            route(method, output =>
            {
                if (!backend.TryDirectPtxGemm(left, right, output, m, k, n))
                    throw new InvalidOperationException(backend.DirectPtxLastError);
            }, Resource(audit), Fp32GridTolerance);
            return;
        }

        // Outside sm_86 the library's architecture gate refuses this kernel. Its PTX uses no sm_80+ instruction, so
        // the grid loads the identical emitted PTX for this device directly: measurement only, never a dispatch path.
        string ptx = PtxGemmKernel.EmitPtx(
            directRuntime.ComputeCapabilityMajor, directRuntime.ComputeCapabilityMinor, m, k, n);
        using DirectPtxModule module = directRuntime.LoadModule(ptx);
        IntPtr function = module.GetFunction(PtxGemmKernel.EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, PtxGemmKernel.BlockThreads);
        var resources = new Resources(
            info.RegistersPerThread, info.StaticSharedBytes, 0, info.ConstBytes, info.LocalBytesPerThread,
            info.MaxThreadsPerBlock, info.PtxVersion, info.BinaryVersion, PtxGemmKernel.BlockThreads, activeBlocks,
            $"gemm-64x64x8 loaded directly on {device.Architecture} (the library gate admits sm_86 only)",
            directRuntime.DeviceFingerprint, Sha256Hex(ptx), "benchmark-loaded", "n/a", "n/a", "n/a", "n/a");
        route(method, output => LaunchGemm64(module, function, left, right, output, m, n), resources,
            Fp32GridTolerance);
    }

    private static unsafe void LaunchGemm64(
        DirectPtxModule module, IntPtr function, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int m, int n)
    {
        IntPtr a = left.Handle, b = right.Handle, c = output.Handle;
        void** arguments = stackalloc void*[3];
        arguments[0] = &a;
        arguments[1] = &b;
        arguments[2] = &c;
        module.Launch(function, (uint)(n / PtxGemmKernel.BlockN), (uint)(m / PtxGemmKernel.BlockM), 1,
            PtxGemmKernel.BlockThreads, 1, 1, 0, arguments);
    }

    private static int ReadCooldownCelsius()
    {
        string? configured = Environment.GetEnvironmentVariable(CooldownVariable);
        if (string.IsNullOrWhiteSpace(configured)) return DefaultCooldownCelsius;
        if (!int.TryParse(configured, out int celsius) || celsius is < 30 or > 75)
            throw new ArgumentOutOfRangeException(CooldownVariable, configured,
                "The GEMM grid cooldown target must be an integer from 30 to 75 C (75 C is the evidence ceiling).");
        return celsius;
    }

    // The SM clock right after a route, so the summary can state % of peak at the clock the GPU actually ran.
    private static void PrintGridClock(int run, string operation, string method)
        => Console.WriteLine("gemm_grid_clock_json=" + JsonSerializer.Serialize(new
        {
            run,
            operation,
            method,
            sm_clock_mhz = GpuBenchmarkEnvironment.SampleSmClockMhz(),
        }));

    private static void PrintUnsupported(int run, string operation, string shape, string method, string reason)
        => Console.WriteLine("dense_linear_status_json=" + JsonSerializer.Serialize(new
        {
            run,
            operation,
            shape,
            method,
            status = "unsupported",
            reason,
        }));

    private static string Sha256Hex(string text)
    {
        using SHA256 sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    }

    // Fixed corners of the first and last batch plus random interior points, so edge tiles are always checked.
    private static (int Batch, int Row, int Column)[] SampleOutputs(Random random, int batch, int m, int n)
    {
        var points = new List<(int, int, int)>
        {
            (0, 0, 0), (0, 0, n - 1), (0, m - 1, 0), (batch - 1, m - 1, n - 1),
        };
        long total = (long)batch * m * n;
        while (points.Count < Math.Min(GridErrorSamples, total))
            points.Add((random.Next(batch), random.Next(m), random.Next(n)));
        return points.ToArray();
    }

    private static double[] SampledOracle(
        GemmGridCase gridCase, float[] left, float[] right, float[] initialOutput,
        (int Batch, int Row, int Column)[] samples)
    {
        var (_, variant, _, m, k, n) = gridCase;
        var expected = new double[samples.Length];
        for (int s = 0; s < samples.Length; s++)
        {
            var (b, row, column) = samples[s];
            long leftBase = (long)b * m * k + (long)row * k;
            long rightBase = (long)b * k * n + column;
            double sum = 0;
            for (int inner = 0; inner < k; inner++)
                sum += left[leftBase + inner] * (double)right[rightBase + (long)inner * n];
            expected[s] = variant == GemmGridVariant.AlphaBeta
                ? GridAlpha * sum + GridBeta * (double)initialOutput[(long)row * n + column]
                : sum;
        }
        return expected;
    }

    // Max absolute error over the samples, relative to the largest sampled magnitude.
    private static double SampledRelativeError(
        float[] actual, (int Batch, int Row, int Column)[] samples, double[] expected, int m, int n)
    {
        double scale = 1e-30, error = 0;
        for (int s = 0; s < samples.Length; s++)
        {
            var (b, row, column) = samples[s];
            scale = Math.Max(scale, Math.Abs(expected[s]));
            error = Math.Max(error, Math.Abs(actual[(long)b * m * n + (long)row * n + column] - expected[s]));
        }
        return error / scale;
    }
}
