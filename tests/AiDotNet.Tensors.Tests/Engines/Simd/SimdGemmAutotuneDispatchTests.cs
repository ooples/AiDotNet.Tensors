using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;

public class SimdGemmAutotuneDispatchTests : IDisposable
{
    private readonly string _tmpCache;
    private readonly string? _prevEnv;

    public SimdGemmAutotuneDispatchTests()
    {
        _prevEnv = Environment.GetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH");
        _tmpCache = Path.Combine(Path.GetTempPath(), "aidotnet-gemm-dispatch-" + Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH", _tmpCache);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_AUTOTUNE_CACHE_PATH", _prevEnv);
        try { if (Directory.Exists(_tmpCache)) Directory.Delete(_tmpCache, recursive: true); } catch { }
    }

    [Fact]
    public void Sgemm_UsesCachedWinner_WhenAvailable()
    {
        // Store a "parallel" winner for a specific shape, run Sgemm at
        // that shape, and verify the output matches a known-good
        // reference (correctness is the assertion — the dispatch
        // pathway is an implementation detail we can't instrument at
        // the kernel call site without adding a hook).
        int M = 8, N = 4, K = 6;
        AutotuneCache.Store(BuiltInCatalog.SGEMM, new ShapeProfile(M, N, K),
            new KernelChoice { Variant = "parallel", MeasuredGflops = 42.0 });

        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        var rng = new Random(1);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);

        SimdGemm.Sgemm(a, b, c, M, K, N);

        // Reference: naive O(MKN) matmul.
        var reference = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int p = 0; p < K; p++) acc += a[i * K + p] * b[p * N + j];
                reference[i * N + j] = acc;
            }
        for (int i = 0; i < c.Length; i++)
            Assert.Equal(reference[i], c[i], 3);
    }

    [Fact]
    public void Sgemm_NoCachedWinner_FallsBackToDefault()
    {
        // Fresh temp cache → no entry → must fall back to
        // SimdGemm.UseParallelGemm default and produce the same correct
        // result.
        int M = 7, N = 3, K = 5;
        var a = new float[M * K];
        var b = new float[K * N];
        var c = new float[M * N];
        var rng = new Random(2);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble());
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble());

        SimdGemm.Sgemm(a, b, c, M, K, N);

        var reference = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int p = 0; p < K; p++) acc += a[i * K + p] * b[p * N + j];
                reference[i * N + j] = acc;
            }
        for (int i = 0; i < c.Length; i++)
            Assert.Equal(reference[i], c[i], 3);
    }
}
