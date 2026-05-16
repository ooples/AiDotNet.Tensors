using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ScalarKernelTests
{
    [Fact]
    public void Gemm_DispatchesToScalarPath_SmallShape_MatchesNaive()
    {
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void Gemm_DispatchesToStreaming_ForSmallK()
    {
        // K=8 is below the Streaming threshold (K<32 OR M*N<1024).
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Theory]
    [InlineData(PackingMode.ForcePackBoth)]
    [InlineData(PackingMode.ForcePackAOnly)]
    [InlineData(PackingMode.ForceStreaming)]
    public void Gemm_AllForcedStrategies_ProduceMatchingResults(PackingMode mode)
    {
        int m = 16, n = 16, k = 16;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        var options = new BlasOptions<double> { PackingMode = mode };
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void Gemm_L2ShapeStandIn_TransA_MatchesNaive()
    {
        // Stand-in for ConvTranspose2D L2-shape (M=4096, N=16, K=512 transA=true).
        // Test at smaller dims to keep the suite fast; full perf gate is Phase L.
        int m = 32, n = 16, k = 64;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: true, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, k, m, transA: true, b, k, n, transB: false);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: m, transA: true,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        // Phase B perf is not the target — correctness only. Phase L gates perf.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void PackingMode_HasFiveValues()
    {
        var values = Enum.GetValues(typeof(PackingMode));
        Assert.Equal(5, values.Length);
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.Auto));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForcePackBoth));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForcePackAOnly));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForceStreaming));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.DisableAutotune));
    }

    [Fact]
    public void BlasOptions_DefaultValues_AreSafe()
    {
        var options = new BlasOptions<double>();
        Assert.Equal(PackingMode.Auto, options.PackingMode);
        Assert.Equal(FusedActivationType.None, options.Epilogue.Activation);
        Assert.True(options.Epilogue.BiasN.IsEmpty);
        Assert.True(options.Epilogue.SkipMxN.IsEmpty);
        Assert.Equal(0u, options.Epilogue.DropoutMask);
        Assert.Equal(0.0, options.Epilogue.OutputScale);  // T=double; default(T)==0.0
        Assert.Equal(0, options.NumThreads);
        Assert.Equal(0UL, options.AutotuneKey);
        Assert.Equal(0L, options.MaxJitCacheBytes);
        Assert.Null(options.PackedA);
        Assert.Null(options.PackedB);
        Assert.True(options.Workspace.IsEmpty);
    }

    [Fact]
    public void BlasOptions_CanSetAllFields()
    {
        Span<byte> ws = new byte[64];
        Span<double> bias = new double[4];
        var options = new BlasOptions<double>
        {
            PackingMode = PackingMode.ForcePackBoth,
            Epilogue = new Epilogue<double>
            {
                BiasN = bias,
                Activation = FusedActivationType.ReLU,
                OutputScale = 2.0,
            },
            Workspace = ws,
            NumThreads = -1,
            AutotuneKey = 42UL,
            MaxJitCacheBytes = 1024L * 1024,
        };
        Assert.Equal(PackingMode.ForcePackBoth, options.PackingMode);
        Assert.Equal(FusedActivationType.ReLU, options.Epilogue.Activation);
        Assert.Equal(2.0, options.Epilogue.OutputScale);
        Assert.Equal(-1, options.NumThreads);
        Assert.Equal(42UL, options.AutotuneKey);
        Assert.Equal(1024L * 1024, options.MaxJitCacheBytes);
        Assert.False(options.Workspace.IsEmpty);
        Assert.False(options.Epilogue.BiasN.IsEmpty);
    }

    [Fact]
    public void KernelKey_EqualKeysHaveEqualHashCodes()
    {
        var k1 = new KernelKey
        {
            M = 4, N = 4, K = 4, Lda = 4, Ldb = 4, Ldc = 4,
            TransA = true, TransB = false,
            Packing = PackingMode.ForcePackBoth,
            EpilogueFlags = 0,
            ElemType = typeof(double),
            Arch = CpuArch.Avx512,
        };
        var k2 = k1;
        Assert.Equal(k1, k2);
        Assert.Equal(k1.GetHashCode(), k2.GetHashCode());
        Assert.True(k1.Equals(k2));

        // Verify by-value equality with an independently-constructed key,
        // not just struct copy. Catches Equals/GetHashCode bugs that skip
        // fields (which struct-copy comparison wouldn't surface).
        var k3 = new KernelKey
        {
            M = 4, N = 4, K = 4, Lda = 4, Ldb = 4, Ldc = 4,
            TransA = true, TransB = false,
            Packing = PackingMode.ForcePackBoth,
            EpilogueFlags = 0,
            ElemType = typeof(double),
            Arch = CpuArch.Avx512,
        };
        Assert.Equal(k1, k3);
        Assert.Equal(k1.GetHashCode(), k3.GetHashCode());
    }

    [Fact]
    public void KernelKey_DifferentKeysAreNotEqual()
    {
        var k1 = new KernelKey
        {
            M = 4, N = 4, K = 4, Lda = 4, Ldb = 4, Ldc = 4,
            TransA = true, TransB = false,
            Packing = PackingMode.ForcePackBoth,
            EpilogueFlags = 0,
            ElemType = typeof(double),
            Arch = CpuArch.Avx512,
        };
        var k2 = k1 with { M = 8 };
        Assert.NotEqual(k1, k2);
        Assert.False(k1.Equals(k2));
    }

    [Fact]
    public void BlasManagedStats_DefaultIsZero()
    {
        var stats = new BlasManagedStats();
        Assert.Equal(0L, stats.AutotuneHits);
        Assert.Equal(0L, stats.AutotuneMisses);
        Assert.Equal(0L, stats.JitEmissions);
        Assert.Equal(0L, stats.JitCacheHits);
        Assert.Equal(0L, stats.PackCacheHits);
        Assert.Equal(0L, stats.PackCacheMisses);
        Assert.Equal(0L, stats.PackCacheBytes);
    }

    [Fact]
    public void ScalarFp64_4x4_Computes_4x4_Tile_From_Packed_Inputs()
    {
        // Packed-A vpanel layout: [Kc × Mr]. Mr-contiguous within each k-slice.
        //   packedA[k*Mr + row] = A[row, k]
        // For a single stripe with Mr=4 rows and Kc=2 K-steps:
        //   k=0 slice: A[0,0]=1, A[1,0]=3, A[2,0]=5, A[3,0]=7
        //   k=1 slice: A[0,1]=2, A[1,1]=4, A[2,1]=6, A[3,1]=8
        double[] packedA = { 1, 3, 5, 7,   2, 4, 6, 8 };

        // Packed-B layout: Kc=2 × Nr=4 cols, col-contiguous within k.
        //   packedB[k*Nr + col] = B[k, col]
        // For Kc=2:
        //   k=0: [1, 0, 0, 0]
        //   k=1: [0, 1, 0, 0]
        double[] packedB = { 1, 0, 0, 0,   0, 1, 0, 0 };

        double[] c = new double[4 * 4];  // C tile, ldc = 4, pre-zeroed by new[]
        int ldc = 4;

        ScalarFp64_4x4.Run(packedA, packedB, c, ldc, kc: 2);

        // C[row, col] = sum_k A[row, k] · B[k, col]
        // A = [[1,2],[3,4],[5,6],[7,8]],  B = [[1,0,0,0],[0,1,0,0]]
        // → C = [[1,2,0,0],[3,4,0,0],[5,6,0,0],[7,8,0,0]]
        double[] expected = { 1, 2, 0, 0,   3, 4, 0, 0,   5, 6, 0, 0,   7, 8, 0, 0 };
        // FP64 has 15–17 significant decimal digits; precision: 12 leaves margin for FMA reordering.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], c[i], precision: 12);
    }

    [Fact]
    public void ScalarFp32_4x4_Computes_4x4_Tile_From_Packed_Inputs()
    {
        // Packed-A vpanel layout: [Kc × Mr]. Mr-contiguous within each k-slice.
        //   packedA[k*Mr + row] = A[row, k]
        // For a single stripe with Mr=4 rows and Kc=2 K-steps:
        //   k=0 slice: A[0,0]=1, A[1,0]=3, A[2,0]=5, A[3,0]=7
        //   k=1 slice: A[0,1]=2, A[1,1]=4, A[2,1]=6, A[3,1]=8
        float[] packedA = { 1f, 3f, 5f, 7f,   2f, 4f, 6f, 8f };

        // Packed-B layout: [Kc × Nr] row-major. Nr-contiguous within each k.
        //   packedB[k*Nr + col] = B[k, col]
        float[] packedB = { 1f, 0f, 0f, 0f,   0f, 1f, 0f, 0f };

        float[] c = new float[4 * 4];  // pre-zeroed by new[]
        int ldc = 4;

        ScalarFp32_4x4.Run(packedA, packedB, c, ldc, kc: 2);

        // C[row, col] = sum_k A[row, k] · B[k, col]
        // A = [[1,2],[3,4],[5,6],[7,8]], B = [[1,0,0,0],[0,1,0,0]]
        // → C = [[1,2,0,0],[3,4,0,0],[5,6,0,0],[7,8,0,0]]
        float[] expected = { 1f, 2f, 0f, 0f,   3f, 4f, 0f, 0f,   5f, 6f, 0f, 0f,   7f, 8f, 0f, 0f };
        // FP32 has ~7 significant decimal digits; precision: 6 leaves margin for FMA reordering.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], c[i], precision: 6);
    }

    [Fact]
    public void PackA_NonTransposed_FP64_MatchesLogicalLayout()
    {
        // Logical A is 8×4 row-major [M=8, K=4]:
        //   row 0: [1.0, 2.0, 3.0, 4.0]
        //   row 1: [5.0, 6.0, 7.0, 8.0]
        //   ...
        //   row 7: [29.0, 30.0, 31.0, 32.0]
        int m = 8, k = 4;
        double[] a = new double[m * k];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < k; c++)
                a[r * k + c] = r * k + c + 1;  // 1..32

        // Pack into 2 stripes (Mc=8, Mr=4) × Kc=4 × Mr=4 = 32 elements.
        int mc = 8, kc = 4, mr = 4;
        double[] packed = new double[mc * kc];
        ScalarPack.PackA<double>(a, lda: k, transA: false, packed, mc, kc, mr);

        // For each packed cell, the value must equal the corresponding logical A cell.
        for (int stripe = 0; stripe < mc / mr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int packedIndex = stripe * kc * mr + kk * mr + row;
                    int logicalRow = stripe * mr + row;
                    int logicalCol = kk;
                    double expected = a[logicalRow * k + logicalCol];
                    Assert.Equal(expected, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackA_Transposed_FP64_MatchesLogicalLayout()
    {
        // Logical A is 8×4 but stored transposed [K=4, M=8] row-major:
        //   memory row 0 (k=0): [A[0,0], A[1,0], A[2,0], ..., A[7,0]]
        //   memory row 1 (k=1): [A[0,1], A[1,1], A[2,1], ..., A[7,1]]
        //   memory row 2 (k=2): ...
        //   memory row 3 (k=3): ...
        // So a[k * M + m] = logical A[m, k].
        int m = 8, k = 4;
        double[] a = new double[k * m];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < k; c++)
                a[c * m + r] = r * k + c + 1;  // logical A[r, c] = r*k + c + 1

        int mc = 8, kc = 4, mr = 4;
        double[] packed = new double[mc * kc];
        ScalarPack.PackA<double>(a, lda: m, transA: true, packed, mc, kc, mr);

        // Same verification as the non-transposed test: each packed cell must equal
        // its corresponding logical A cell. The pack routine handles the
        // memory-layout difference transparently.
        for (int stripe = 0; stripe < mc / mr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int packedIndex = stripe * kc * mr + kk * mr + row;
                    int logicalRow = stripe * mr + row;
                    int logicalCol = kk;
                    double expectedLogical = logicalRow * k + logicalCol + 1;
                    Assert.Equal(expectedLogical, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackA_NonTransposed_FP32_MatchesLogicalLayout()
    {
        int m = 8, k = 4;
        float[] a = new float[m * k];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < k; c++)
                a[r * k + c] = r * k + c + 1f;

        int mc = 8, kc = 4, mr = 4;
        float[] packed = new float[mc * kc];
        ScalarPack.PackA<float>(a, lda: k, transA: false, packed, mc, kc, mr);

        for (int stripe = 0; stripe < mc / mr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int packedIndex = stripe * kc * mr + kk * mr + row;
                    int logicalRow = stripe * mr + row;
                    int logicalCol = kk;
                    float expected = a[logicalRow * k + logicalCol];
                    Assert.Equal(expected, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackA_Transposed_FP32_MatchesLogicalLayout()
    {
        // Logical A is 8×4 but stored transposed [K=4, M=8] row-major:
        //   a[k * M + m] = logical A[m, k].
        int m = 8, k = 4;
        float[] a = new float[k * m];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < k; c++)
                a[c * m + r] = r * k + c + 1f;  // logical A[r, c] = r*k + c + 1

        int mc = 8, kc = 4, mr = 4;
        float[] packed = new float[mc * kc];
        ScalarPack.PackA<float>(a, lda: m, transA: true, packed, mc, kc, mr);

        for (int stripe = 0; stripe < mc / mr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int packedIndex = stripe * kc * mr + kk * mr + row;
                    int logicalRow = stripe * mr + row;
                    int logicalCol = kk;
                    float expectedLogical = logicalRow * k + logicalCol + 1f;
                    Assert.Equal(expectedLogical, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackB_NonTransposed_FP64_MatchesLogicalLayout()
    {
        // Logical B is 4×8 row-major [K=4, N=8]:
        //   row 0: [1, 2, 3, 4, 5, 6, 7, 8]
        //   row 1: [9, 10, ..., 16]
        //   ...
        int k = 4, n = 8;
        double[] b = new double[k * n];
        for (int r = 0; r < k; r++)
            for (int c = 0; c < n; c++)
                b[r * n + c] = r * n + c + 1;

        int kc = 4, nc = 8, nr = 4;
        double[] packed = new double[kc * nc];
        ScalarPack.PackB<double>(b, ldb: n, transB: false, packed, nc, kc, nr);

        for (int stripe = 0; stripe < nc / nr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int col = 0; col < nr; col++)
                {
                    int packedIndex = stripe * kc * nr + kk * nr + col;
                    int logicalRow = kk;
                    int logicalCol = stripe * nr + col;
                    double expected = b[logicalRow * n + logicalCol];
                    Assert.Equal(expected, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackB_Transposed_FP64_MatchesLogicalLayout()
    {
        // Logical B is 4×8 but stored transposed [N=8, K=4] row-major:
        //   b[c * K + r] = logical B[r, c].
        int k = 4, n = 8;
        double[] b = new double[n * k];
        for (int r = 0; r < k; r++)
            for (int c = 0; c < n; c++)
                b[c * k + r] = r * n + c + 1;

        int kc = 4, nc = 8, nr = 4;
        double[] packed = new double[kc * nc];
        ScalarPack.PackB<double>(b, ldb: k, transB: true, packed, nc, kc, nr);

        for (int stripe = 0; stripe < nc / nr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int col = 0; col < nr; col++)
                {
                    int packedIndex = stripe * kc * nr + kk * nr + col;
                    int logicalRow = kk;
                    int logicalCol = stripe * nr + col;
                    double expectedLogical = logicalRow * n + logicalCol + 1;
                    Assert.Equal(expectedLogical, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackB_NonTransposed_FP32_MatchesLogicalLayout()
    {
        int k = 4, n = 8;
        float[] b = new float[k * n];
        for (int r = 0; r < k; r++)
            for (int c = 0; c < n; c++)
                b[r * n + c] = r * n + c + 1f;

        int kc = 4, nc = 8, nr = 4;
        float[] packed = new float[kc * nc];
        ScalarPack.PackB<float>(b, ldb: n, transB: false, packed, nc, kc, nr);

        for (int stripe = 0; stripe < nc / nr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int col = 0; col < nr; col++)
                {
                    int packedIndex = stripe * kc * nr + kk * nr + col;
                    int logicalRow = kk;
                    int logicalCol = stripe * nr + col;
                    float expected = b[logicalRow * n + logicalCol];
                    Assert.Equal(expected, packed[packedIndex]);
                }
            }
        }
    }

    [Fact]
    public void PackB_Transposed_FP32_MatchesLogicalLayout()
    {
        int k = 4, n = 8;
        float[] b = new float[n * k];
        for (int r = 0; r < k; r++)
            for (int c = 0; c < n; c++)
                b[c * k + r] = r * n + c + 1f;

        int kc = 4, nc = 8, nr = 4;
        float[] packed = new float[kc * nc];
        ScalarPack.PackB<float>(b, ldb: k, transB: true, packed, nc, kc, nr);

        for (int stripe = 0; stripe < nc / nr; stripe++)
        {
            for (int kk = 0; kk < kc; kk++)
            {
                for (int col = 0; col < nr; col++)
                {
                    int packedIndex = stripe * kc * nr + kk * nr + col;
                    int logicalRow = kk;
                    int logicalCol = stripe * nr + col;
                    float expectedLogical = logicalRow * n + logicalCol + 1f;
                    Assert.Equal(expectedLogical, packed[packedIndex]);
                }
            }
        }
    }

    // ── B5 helpers ────────────────────────────────────────────────────────────

    private static (double[] a, double[] b) GenerateRandomMatrices(int m, int n, int k, bool transA, bool transB, int seed)
    {
        var rng = new Random(seed);
        // a holds either [M, K] (transA=false, length m*k) or [K, M] (transA=true, length k*m).
        int aLen = m * k;  // same length either way
        var a = new double[aLen];
        for (int i = 0; i < aLen; i++) a[i] = rng.NextDouble() * 2 - 1;  // [-1, 1)
        // b holds either [K, N] (transB=false) or [N, K] (transB=true).
        int bLen = k * n;
        var b = new double[bLen];
        for (int i = 0; i < bLen; i++) b[i] = rng.NextDouble() * 2 - 1;
        return (a, b);
    }

    private static double[] NaiveGemm(
        double[] a, int aRows, int aCols, bool transA,
        double[] b, int bRows, int bCols, bool transB)
    {
        // Logical shapes: op(A) is m×k, op(B) is k×n; result is m×n.
        // When transA=true: aRows is K, aCols is M (logical M = aCols, K = aRows).
        // When transA=false: aRows is M, aCols is K.
        int m = transA ? aCols : aRows;
        int kA = transA ? aRows : aCols;
        int kB = transB ? bCols : bRows;
        int n = transB ? bRows : bCols;
        if (kA != kB) throw new ArgumentException($"Inner dim mismatch: A K={kA}, B K={kB}");
        int k = kA;

        var c = new double[m * n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                {
                    double aval = transA ? a[kk * aCols + i] : a[i * aCols + kk];
                    double bval = transB ? b[j * bCols + kk] : b[kk * bCols + j];
                    sum += aval * bval;
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    [Theory]
    [InlineData(8, 8, 8, false, false)]   // NN
    [InlineData(8, 8, 8, true, false)]    // TN
    [InlineData(8, 8, 8, false, true)]    // NT
    [InlineData(8, 8, 8, true, true)]     // TT
    [InlineData(16, 4, 8, false, false)]  // rectangular
    [InlineData(4, 16, 8, false, false)]
    [InlineData(8, 8, 4, false, false)]   // small K
    [InlineData(8, 8, 16, false, false)]  // larger K
    public void Streaming_MatchesNaiveReference_FP64(int m, int n, int k, bool transA, bool transB)
    {
        int aCols = transA ? m : k;
        int lda = aCols;
        int bCols = transB ? k : n;
        int ldb = bCols;
        int aRows = transA ? k : m;
        int bRows = transB ? n : k;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, bRows, bCols, transB);

        double[] actual = new double[m * n];
        StreamingStrategy.Run<double>(
            a, lda, transA,
            b, ldb, transB,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void Streaming_MatchesNaiveReference_FP32_SquareNoTrans()
    {
        int m = 8, n = 8, k = 8;
        var rng = new Random(42);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference: naive triple-loop FP32 GEMM (no trans).
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) sum += a[i * k + kk] * b[kk * n + j];
                expected[i * n + j] = sum;
            }

        float[] actual = new float[m * n];
        StreamingStrategy.Run<float>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 4);  // FP32 K=8 accumulation
    }

    [Theory]
    [InlineData(8, 8, 8, false)]     // square, no transA
    [InlineData(8, 8, 8, true)]      // transA
    [InlineData(16, 4, 8, false)]    // multi-iter M
    [InlineData(4, 16, 8, false)]    // multi-iter N (uses jc loop)
    [InlineData(8, 8, 16, false)]    // larger K
    public void PackAOnly_MatchesNaiveReference_FP64(int m, int n, int k, bool transA)
    {
        // PackAOnly is transB=false only (Phase B limitation).
        int aCols = transA ? m : k;
        int lda = aCols;
        int ldb = n;  // B is [K, N] row-major

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB: false, seed: 42);
        int aRows = transA ? k : m;
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, k, n, transB: false);

        double[] actual = new double[m * n];
        PackAOnlyStrategy.Run<double>(
            a, lda, transA,
            b, ldb,
            actual, ldc: n,
            m, n, k,
            mc: 8, kc: 8,
            mr: 4, nr: 4);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Theory]
    [InlineData(8, 8, 8, false, false)]    // square, no trans
    [InlineData(8, 8, 8, true, false)]     // transA
    [InlineData(8, 8, 8, false, true)]     // transB
    [InlineData(8, 8, 8, true, true)]      // both
    [InlineData(16, 4, 8, false, false)]   // rectangular, multi-iter on M
    [InlineData(4, 16, 8, false, false)]   // rectangular, multi-iter on N
    public void PackBoth_MatchesNaiveReference(int m, int n, int k, bool transA, bool transB)
    {
        // For transA=false: a stored [m, k] with lda=k.
        // For transA=true:  a stored [k, m] with lda=m.
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int lda = aCols;
        // For transB=false: b stored [k, n] with ldb=n.
        // For transB=true:  b stored [n, k] with ldb=k.
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        int ldb = bCols;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, bRows, bCols, transB);

        double[] actual = new double[m * n];

        PackBothStrategy.Run<double>(
            a, lda, transA,
            b, ldb, transB,
            actual, ldc: n,
            m, n, k,
            mc: 8, nc: 8, kc: 8,
            mr: 4, nr: 4);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }
}
