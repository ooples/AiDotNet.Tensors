using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1261 (phase A of #377): where CUDA GEMM stands before any kernel work. Every CUDA GEMM runs on cuBLAS today
/// (both direct-PTX GEMM kernels report <c>IsPromotedShape =&gt; false</c>), so this times, per shape:
/// <list type="bullet">
/// <item>cuBLAS as production configures it: TF32 tensor-op math on Ampere+ outside deterministic mode;</item>
/// <item>cuBLAS in strict FP32 (<c>CUBLAS_PEDANTIC_MATH</c>, what deterministic mode selects): the fair bar for
/// an FP32 kernel, and the precision PyTorch's default <c>float32_matmul_precision("highest")</c> uses;</item>
/// <item>each direct-PTX GEMM kernel where its shape contract admits the shape, else an explicit unsupported row;</item>
/// <item>PyTorch (eager, CUDA graph, torch.compile) through the shared Python lane.</item>
/// </list>
/// Opt-in only (<c>--only=gemm-grid</c>): the large shapes make it far longer than the default matrix.
/// </summary>
internal static partial class DirectPtxDenseLinearFullExperiment
{
    internal const string GemmGridOperation = "gemm-grid";

    // (name, M, K, N): square, decode / skinny, LLM projections, vision im2col, and odd sizes no tile divides.
    internal static readonly (string Name, int M, int K, int N)[] GemmGrid =
    {
        ("sq128", 128, 128, 128), ("sq256", 256, 256, 256), ("sq512", 512, 512, 512),
        ("sq1024", 1024, 1024, 1024), ("sq2048", 2048, 2048, 2048), ("sq4096", 4096, 4096, 4096),
        ("decode-m1-4096", 1, 4096, 4096), ("decode-m16-4096", 16, 4096, 4096),
        ("decode-m1-mlp-up", 1, 4096, 11008), ("decode-m16-mlp-up", 16, 4096, 11008),
        ("prefill-m128-qkv", 128, 4096, 4096), ("prefill-m2048-qkv", 2048, 4096, 4096),
        ("prefill-m2048-mlp-up", 2048, 4096, 11008), ("prefill-m2048-mlp-down", 2048, 11008, 4096),
        ("vision-56x56-c64", 3136, 576, 64), ("vision-28x28-c128", 784, 1152, 128),
        ("odd-1000", 1000, 1000, 1000), ("odd-127x513x255", 127, 513, 255),
    };

    // Checked output elements per lane: an fp64 oracle over the whole 4096^3 output would take minutes on the host.
    private const int GridErrorSamples = 512;
    private const double Fp32GridTolerance = 2e-4;
    private const double Tf32GridTolerance = 1e-2;

    private static void RunGemmGrid(CudaBackend backend, int run)
    {
        foreach (var (name, m, k, n) in GemmGrid)
        {
            var random = new Random(20261261 + run * 101 + m * 7 + n);
            float[] leftHost = Values(random, m * k);
            float[] rightHost = Values(random, k * n, 0.0625f);
            var samples = SampleOutputs(random, m, n);
            double[] oracle = SampledOracle(leftHost, rightHost, k, n, samples);
            double work = 2d * m * k * n;
            string shape = $"M{m} K{k} N{n}";
            string operation = GemmGridOperation + "-" + name;
            // Enough launches per device sample to span ~1 ms, so small shapes are not timer-bound and the
            // largest do not take minutes; graphs only where 200 back-to-back operations stay short.
            int launches = (int)Math.Clamp(Math.Ceiling(2e10 / work), 1, DeviceLaunches);
            bool capture = work <= 2e9;
            double RelativeError(IGpuBuffer output) => SampledRelativeError(backend.DownloadBuffer(output), samples, oracle, n);

            using var left = backend.AllocateBuffer(leftHost);
            using var right = backend.AllocateBuffer(rightHost);
            using var output = backend.AllocateBuffer(m * n);

            DirectPtxFeatureGate.TestOverride = false;
            DirectPtxFeatureGate.FusedLinearExperimentOverride = false;
            AiDotNetEngine.SetDeterministicMode(false);
            BenchmarkRoute(backend, run, operation, shape, "cuBLAS TF32 (prod default)", work,
                () => backend.Gemm(left, right, output, m, n, k), () => RelativeError(output),
                Resources.None, capture, tolerance: Tf32GridTolerance, deviceLaunches: launches);

            AiDotNetEngine.SetDeterministicMode(true);
            try
            {
                BenchmarkRoute(backend, run, operation, shape, "cuBLAS FP32 strict", work,
                    () => backend.Gemm(left, right, output, m, n, k), () => RelativeError(output),
                    Resources.None, capture, tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            finally
            {
                AiDotNetEngine.SetDeterministicMode(false);
            }

            EnableDirectExperiment();
            if (PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) &&
                backend.PrewarmDirectPtxGemmTiled(m, k, n, DirectPtxLinearWeightLayout.InputMajor) &&
                backend.TryGetDirectPtxGemmTiledAudit(m, k, n, DirectPtxLinearWeightLayout.InputMajor, 1, out var tiledAudit))
            {
                BenchmarkRoute(backend, run, operation, shape, "Direct PTX tiled 16x16x32", work, () =>
                    {
                        if (!backend.TryDirectPtxGemmTiled(left, right, output, m, k, n, DirectPtxLinearWeightLayout.InputMajor))
                            throw new InvalidOperationException(backend.DirectPtxLastError);
                    }, () => RelativeError(output), Resource(tiledAudit), capture,
                    tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            else
            {
                PrintUnsupported(run, operation, shape, "Direct PTX tiled 16x16x32", backend.DirectPtxLastError);
            }

            if (PtxGemmKernel.IsSupportedShape(m, k, n) && backend.PrewarmDirectPtxGemm(m, k, n) &&
                backend.TryGetDirectPtxGemmAudit(m, k, n, out var gemmAudit))
            {
                BenchmarkRoute(backend, run, operation, shape, "Direct PTX gemm 64x64x8", work, () =>
                    {
                        if (!backend.TryDirectPtxGemm(left, right, output, m, k, n))
                            throw new InvalidOperationException(backend.DirectPtxLastError);
                    }, () => RelativeError(output), Resource(gemmAudit), capture,
                    tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            else
            {
                PrintUnsupported(run, operation, shape, "Direct PTX gemm 64x64x8", backend.DirectPtxLastError);
            }
        }
    }

    private static void PrintUnsupported(int run, string operation, string shape, string method, string? reason)
        => Console.WriteLine("dense_linear_status_json=" + JsonSerializer.Serialize(new
        {
            run,
            operation,
            shape,
            method,
            status = "unsupported",
            reason = string.IsNullOrEmpty(reason) ? "shape outside the kernel's contract" : reason,
        }));

    // Fixed corners plus random interior points, so edge tiles are always checked.
    private static (int Row, int Column)[] SampleOutputs(Random random, int m, int n)
    {
        var points = new List<(int, int)> { (0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1) };
        while (points.Count < Math.Min(GridErrorSamples, m * n)) points.Add((random.Next(m), random.Next(n)));
        return points.ToArray();
    }

    private static double[] SampledOracle(float[] left, float[] right, int k, int n, (int Row, int Column)[] samples)
    {
        var expected = new double[samples.Length];
        for (int s = 0; s < samples.Length; s++)
        {
            var (row, column) = samples[s];
            double sum = 0;
            for (int inner = 0; inner < k; inner++) sum += left[row * k + inner] * (double)right[inner * n + column];
            expected[s] = sum;
        }
        return expected;
    }

    // Max absolute error over the samples, relative to the largest sampled magnitude.
    private static double SampledRelativeError(float[] actual, (int Row, int Column)[] samples, double[] expected, int n)
    {
        double scale = 1e-30, error = 0;
        for (int s = 0; s < samples.Length; s++)
        {
            scale = Math.Max(scale, Math.Abs(expected[s]));
            error = Math.Max(error, Math.Abs(actual[samples[s].Row * n + samples[s].Column] - expected[s]));
        }
        return error / scale;
    }
}
