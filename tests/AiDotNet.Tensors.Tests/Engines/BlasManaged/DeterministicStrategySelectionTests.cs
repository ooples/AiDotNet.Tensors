// Regression test for the deterministic-mode autotune-strategy bug.
//
// BlasManaged.Gemm picks its packing strategy from a TIMING-MEASURED, disk-persisted
// autotune cache (Dispatcher.SelectStrategy -> BlasManagedAutotune.TryLookupStrategy,
// populated by BackgroundAutotuner.Measure, which times ForceStreaming / ForcePackAOnly /
// ForcePackBoth and persists the fastest). Those strategies are NOT reduction-order
// equivalent, so whichever one the autotuner happened to clock fastest changed the
// RESULT BITS of the very same GEMM -- including while deterministic mode was on, which
// AiModelBuilder documents as producing "bitwise-identical results across runs on the
// same hardware".
//
// Each test redirects AIDOTNET_AUTOTUNE_CACHE_PATH at a private temp directory. That is
// load-bearing, not hygiene: against the real ~/.aidotnet/autotune cache these assertions
// would pass or fail purely according to what the background autotuner had already
// measured on the host -- the warm-cache dependence that hid this bug in the first place.
// A private root makes every case start from a guaranteed-empty cache.

using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// Toggles process-global deterministic mode and the on-disk autotune cache; serialize
// against the other BlasManaged bit-exactness tests.
[Collection("BlasManaged-Stats-Serial")]
public sealed class DeterministicStrategySelectionTests : IDisposable
{
    private const string EnvVarCachePath = "AIDOTNET_AUTOTUNE_CACHE_PATH";

    private readonly string _cacheRoot;
    private readonly string? _priorCachePath;
    private readonly bool _priorDeterministic;
    private readonly bool? _priorThreadLocal;
    private readonly bool _priorBackgroundAutotuner;

    public DeterministicStrategySelectionTests()
    {
        _priorCachePath = Environment.GetEnvironmentVariable(EnvVarCachePath);
        _priorDeterministic = BlasProvider.IsDeterministicMode;
        _priorThreadLocal = BlasProvider.GetThreadLocalDeterministicMode();
        _priorBackgroundAutotuner = BackgroundAutotuner.Enabled;

        _cacheRoot = Path.Combine(Path.GetTempPath(), "aidotnet-determinism-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_cacheRoot);
        Environment.SetEnvironmentVariable(EnvVarCachePath, _cacheRoot);

        // The background autotuner would race these cases by persisting a timed winner
        // mid-test; the point here is what a PRESENT entry does, written deterministically.
        BackgroundAutotuner.Enabled = false;
        BlasProvider.SetThreadLocalDeterministicMode(null);
        BlasProvider.SetDeterministicMode(true);
        ClearCache();
    }

    public void Dispose()
    {
        ClearCache();
        Environment.SetEnvironmentVariable(EnvVarCachePath, _priorCachePath);
        BackgroundAutotuner.Enabled = _priorBackgroundAutotuner;
        BlasProvider.SetDeterministicMode(_priorDeterministic);
        BlasProvider.SetThreadLocalDeterministicMode(_priorThreadLocal);
        BlasManagedAutotune.ClearStrategyMemo();
        try { if (Directory.Exists(_cacheRoot)) Directory.Delete(_cacheRoot, recursive: true); } catch { }
    }

    public static TheoryData<int, int, int, bool> AffectedShapes => new()
    {
        // Transposed-B shapes reach Dispatcher.SelectStrategy: the non-transposed
        // machine-code and GotoGemm fast paths decline transB, so strategy selection
        // (and therefore the learned cache) actually decides the kernel for these.
        { 48, 256, 64, true },
        { 48, 512, 128, true },
        { 48, 1024, 256, true },
        { 96, 1024, 512, true },
        { 192, 1024, 256, true },
        { 384, 1024, 128, true },
    };

    [Theory]
    [MemberData(nameof(AffectedShapes))]
    public void DeterministicMode_PersistedStrategy_DoesNotChangeResultBits(int m, int n, int k, bool transB)
    {
        var (a, b) = MakeOperands(m, n, k, transA: false, transB, seed: 12345);

        ClearCache();
        float[] coldCache = Gemm(a, b, m, n, k, transB, PackingMode.Auto);

        foreach (var candidate in new[]
                 {
                     PackingMode.ForceStreaming,
                     PackingMode.ForcePackAOnly,
                     PackingMode.ForcePackBoth,
                 })
        {
            ClearCache();
            StoreStrategy(m, n, k, transB, candidate);

            float[] warmCache = Gemm(a, b, m, n, k, transB, PackingMode.Auto);

            int differing = CountDifferingBits(coldCache, warmCache);
            Assert.True(
                differing == 0,
                $"Deterministic mode must not let a persisted autotune entry change result bits, but a "
                + $"'{candidate}' entry changed {differing}/{coldCache.Length} elements of the "
                + $"{m}x{n}x{k} (transB={transB}) GEMM. Deterministic mode is documented as producing "
                + "bitwise-identical results across runs on the same hardware, and the autotune cache "
                + "is populated from wall-clock timings, so it is not reproducible input.");
        }
    }

    [Theory]
    [MemberData(nameof(AffectedShapes))]
    public void DeterministicMode_StrategySelection_IgnoresPersistedEntry(int m, int n, int k, bool transB)
    {
        ClearCache();
        PackingMode cold = SelectStrategy(m, n, k, transB);

        foreach (var candidate in new[]
                 {
                     PackingMode.ForceStreaming,
                     PackingMode.ForcePackAOnly,
                     PackingMode.ForcePackBoth,
                 })
        {
            ClearCache();
            StoreStrategy(m, n, k, transB, candidate);

            PackingMode warm = SelectStrategy(m, n, k, transB);
            Assert.True(
                cold == warm,
                $"Deterministic strategy selection must be a pure function of shape and hardware, but a "
                + $"persisted '{candidate}' entry changed the choice from {cold} to {warm} for "
                + $"{m}x{n}x{k} (transB={transB}).");
        }
    }

    [Fact]
    public void FastMode_StillConsultsThePersistedStrategyCache()
    {
        // The fix must not disable learned routing for callers who did NOT ask for
        // determinism -- fast mode keeps the autotuner's benefit.
        BlasProvider.SetDeterministicMode(false);

        const int m = 48, n = 1024, k = 256;
        ClearCache();
        PackingMode cold = SelectStrategy(m, n, k, transB: true);

        PackingMode other = cold == PackingMode.ForceStreaming
            ? PackingMode.ForcePackBoth
            : PackingMode.ForceStreaming;

        ClearCache();
        StoreStrategy(m, n, k, transB: true, other);
        Assert.Equal(other, SelectStrategy(m, n, k, transB: true));
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static PackingMode SelectStrategy(int m, int n, int k, bool transB) =>
        Dispatcher.SelectStrategy<float>(
            m, n, k, transA: false, transB, new BlasOptions<float> { PackingMode = PackingMode.Auto });

    private static void StoreStrategy(int m, int n, int k, bool transB, PackingMode mode)
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(
            m, n, k, transA: false, transB, mr: 0, nr: 0, hasEpilogue: false,
            isDeterministic: BlasProvider.IsDeterministicMode);
        BlasManagedAutotune.StoreStrategy(
            shape, mode, ParallelismAxis.M,
            mc: 64, nc: 64, kc: 64,
            threadCount: Environment.ProcessorCount, BlasKernelVersion.Current);
    }

    private void ClearCache()
    {
        BlasManagedAutotune.ClearStrategyMemo();
        try
        {
            foreach (string dir in Directory.GetDirectories(_cacheRoot)) Directory.Delete(dir, recursive: true);
            foreach (string file in Directory.GetFiles(_cacheRoot)) File.Delete(file);
        }
        catch
        {
            // Best effort: a stale handle must not fail the test outright.
        }
        BlasManagedAutotune.ClearStrategyMemo();
    }

    private static float[] Gemm(float[] a, float[] b, int m, int n, int k, bool transB, PackingMode mode)
    {
        var c = new float[m * n];
        BlasManagedLib.Gemm<float>(
            a, k, false,
            b, transB ? k : n, transB,
            c, n, m, n, k,
            new BlasOptions<float> { PackingMode = mode });
        return c;
    }

    private static (float[] a, float[] b) MakeOperands(int m, int n, int k, bool transA, bool transB, int seed)
    {
        var rng = new Random(seed);
        var a = new float[transA ? k * m : m * k];
        var b = new float[transB ? n * k : k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        return (a, b);
    }

    private static int CountDifferingBits(float[] x, float[] y)
    {
        int differing = 0;
        for (int i = 0; i < x.Length; i++)
        {
            // Bitwise, not value-equality: the contract is bit-identical, and .Equals
            // would treat +0.0 == -0.0 and mishandle NaN payloads, masking a drift.
            if (FloatBits(x[i]) != FloatBits(y[i])) differing++;
        }
        return differing;
    }

    // BitConverter.SingleToInt32Bits is net5+; this test project also targets net471,
    // where GetBytes -> ToInt32 is the equivalent (matches DeterministicParallelGemmContractTests).
    private static int FloatBits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);
}
