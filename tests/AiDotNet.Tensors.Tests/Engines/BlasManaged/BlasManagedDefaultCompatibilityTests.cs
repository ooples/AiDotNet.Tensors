using System;
using System.IO;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class BlasManagedDefaultCompatibilityTests : IDisposable
{
    private const string CacheVariable = "AIDOTNET_AUTOTUNE_CACHE_PATH";
    private readonly string? _previousCache = Environment.GetEnvironmentVariable(CacheVariable);
    private readonly bool? _previousDeterministic = BlasProvider.GetThreadLocalDeterministicMode();
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "aidotnet-plan-" + Guid.NewGuid().ToString("N"));

    public BlasManagedDefaultCompatibilityTests()
    {
        Environment.SetEnvironmentVariable(CacheVariable, _cache);
        BlasProvider.SetThreadLocalDeterministicMode(false);
        BlasManagedAutotune.ClearStrategyMemo();
    }

    [Theory]
    [InlineData(PackingMode.Auto, false)]
    [InlineData(PackingMode.DisableAutotune, false)]
    [InlineData(PackingMode.Auto, true)]
    [InlineData(PackingMode.DisableAutotune, true)]
    public void PackBothDefault_PreservesPackAOnlyCompatibleOptions(PackingMode mode, bool nAxis)
    {
        const int m = 512, n = 256, k = 128, lda = m + 1, ldb = n + 1;
        var shape = BlasManagedAutotune.EncodeShape<double>(m, n, k, true, false, 0, 0, false, false);
        // Emulate the AVX-512 ThinK default on every host, including AVX2 and framework builds.
        BlasManagedAutotune.StoreStrategy(shape, PackingMode.ForcePackBoth,
            ParallelismAxis.M, 64, 128, 64, 0, BlasKernelVersion.Current);
        var auto = default(BlasOptions<double>);
        Assert.Equal(PackingMode.ForcePackBoth, Dispatcher.SelectStrategy<double>(m, n, k, true, false, in auto));
        // Padded strides bypass the contiguous thin-M shortcut and exercise default strategy selection.
        var a = new double[k * lda];
        var b = new double[k * ldb];
        for (int i = 0; i < a.Length; i++) a[i] = Math.Sin(i * 0.17);
        for (int i = 0; i < b.Length; i++) b[i] = Math.Cos(i * 0.13);
        var expected = new double[m * n];
        var actual = new double[m * n];
        ParallelismAxis? axis = nAxis ? ParallelismAxis.N : null;
        int block = nAxis ? 0 : 64;
        BlasManagedLib.Gemm<double>(a, lda, true, b, ldb, false, expected, n, m, n, k,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackAOnly, Mc = block, Kc = block, ParallelismAxis = axis });
        BlasManagedLib.Gemm<double>(a, lda, true, b, ldb, false, actual, n, m, n, k,
            new BlasOptions<double> { PackingMode = mode, Mc = block, Kc = block, ParallelismAxis = axis });
        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        for (int row = 0; row < m; row += 97)
            for (int col = 0; col < n; col += 13)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a[p * lda + row] * b[p * ldb + col];
                Assert.True(Math.Abs(actual[row * n + col] - sum) <= 1e-10 * (1 + Math.Abs(sum)));
            }
    }

    public void Dispose()
    {
        BlasManagedAutotune.ClearStrategyMemo();
        Environment.SetEnvironmentVariable(CacheVariable, _previousCache);
        BlasProvider.SetThreadLocalDeterministicMode(_previousDeterministic);
        if (Directory.Exists(_cache)) Directory.Delete(_cache, recursive: true);
    }
}
