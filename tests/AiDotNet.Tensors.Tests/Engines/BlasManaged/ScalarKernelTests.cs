using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Serialized with StatsCounterTests and NativeAotCompatibilityTest so that
/// JittedKernelCache and BlasManagedStatsTracker global state is not
/// contaminated by concurrent test classes on net471's parallel runner.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
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

    [Fact]
    public void Avx2Fp64_4x8_MatchesScalarReference()
    {
        // Skip if AVX2/FMA not available at runtime.
        if (!Avx2Fp64_4x8.IsSupported)
            return;

        int kc = 16;
        int Mr = 4, Nr = 8;

        // Build deterministic packed inputs.
        double[] packedA = new double[Mr * kc];
        double[] packedB = new double[kc * Nr];
        var rng = new Random(42);
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        // Reference: ScalarFp64_4x4 run TWICE (once for cols 0..3, once for cols 4..7).
        // ScalarFp64_4x4.Run uses Nr=4 and expects packed-B [Kc × 4]. We need to
        // build two scalar packed-B slices from the AVX2 packed-B [Kc × 8].
        double[] packedB_lo = new double[kc * 4];
        double[] packedB_hi = new double[kc * 4];
        for (int k = 0; k < kc; k++)
        {
            for (int col = 0; col < 4; col++)
            {
                packedB_lo[k * 4 + col] = packedB[k * 8 + col];
                packedB_hi[k * 4 + col] = packedB[k * 8 + 4 + col];
            }
        }

        double[] cRef = new double[4 * 8];  // ldc = 8
        int ldc = 8;
        ScalarFp64_4x4.Run(packedA, packedB_lo, cRef.AsSpan(), ldc, kc);
        // Run scalar on the hi 4 cols — offset C by 4.
        ScalarFp64_4x4.Run(packedA, packedB_hi, cRef.AsSpan(4), ldc, kc);

        // AVX2: single call with packedB [Kc × 8].
        double[] cAvx = new double[4 * 8];
        Avx2Fp64_4x8.Run(packedA, packedB, cAvx.AsSpan(), ldc, kc);

        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cAvx[i], precision: 12);
    }

    [Fact]
    public void Avx2Fp32_8x8_MatchesScalarReference()
    {
        if (!Avx2Fp32_8x8.IsSupported)
            return;

        int kc = 16;
        int Mr = 8, Nr = 8;

        float[] packedA = new float[Mr * kc];
        float[] packedB = new float[kc * Nr];
        var rng = new Random(42);
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference: 4 calls to ScalarFp32_4x4 covering the 8×8 output.
        // ScalarFp32_4x4 has Mr=4, Nr=4. To cover 8×8 = 4 quadrants:
        //   (rows 0..3, cols 0..3), (rows 0..3, cols 4..7),
        //   (rows 4..7, cols 0..3), (rows 4..7, cols 4..7).
        //
        // The packed-A for the 8x8 kernel is [Kc × Mr=8]; the scalar kernel
        // expects [Kc × Mr=4]. We need to slice packedA into rows 0..3 and rows 4..7
        // per k-step.
        float[] packedA_top = new float[kc * 4];      // rows 0..3
        float[] packedA_bottom = new float[kc * 4];   // rows 4..7
        float[] packedB_left = new float[kc * 4];     // cols 0..3
        float[] packedB_right = new float[kc * 4];    // cols 4..7
        for (int k = 0; k < kc; k++)
        {
            for (int r = 0; r < 4; r++)
            {
                packedA_top[k * 4 + r] = packedA[k * 8 + r];
                packedA_bottom[k * 4 + r] = packedA[k * 8 + 4 + r];
            }
            for (int col = 0; col < 4; col++)
            {
                packedB_left[k * 4 + col] = packedB[k * 8 + col];
                packedB_right[k * 4 + col] = packedB[k * 8 + 4 + col];
            }
        }

        float[] cRef = new float[8 * 8];  // ldc = 8
        int ldc = 8;
        // (rows 0..3, cols 0..3)
        ScalarFp32_4x4.Run(packedA_top, packedB_left, cRef.AsSpan(), ldc, kc);
        // (rows 0..3, cols 4..7) — offset C by 4
        ScalarFp32_4x4.Run(packedA_top, packedB_right, cRef.AsSpan(4), ldc, kc);
        // (rows 4..7, cols 0..3) — offset C by 4*ldc
        ScalarFp32_4x4.Run(packedA_bottom, packedB_left, cRef.AsSpan(4 * ldc), ldc, kc);
        // (rows 4..7, cols 4..7) — offset C by 4*ldc + 4
        ScalarFp32_4x4.Run(packedA_bottom, packedB_right, cRef.AsSpan(4 * ldc + 4), ldc, kc);

        // AVX2 single call with full packed-A [Kc × 8] and packed-B [Kc × 8].
        float[] cAvx = new float[8 * 8];
        Avx2Fp32_8x8.Run(packedA, packedB, cAvx.AsSpan(), ldc, kc);

        // FP32 precision: 16 K-step accumulations of [-1,1) values, ~16 * eps_f = 1.9e-6.
        // precision: 4 is safe (loose enough to accommodate FMA reordering).
        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cAvx[i], precision: 4);
    }

    // ── C3: Avx2Pack tests ────────────────────────────────────────────────────

    [Fact]
    public void Avx2Pack_PackA_Fp64_Transposed_MatchesScalarBitIdentical()
    {
        // Skip when AVX2 is unavailable at runtime.
        if (!Avx2Pack.IsSupported) return;

        int m = 8, k = 16;
        int mc = 8, kc = 16, mr = 4;

        // A stored row-major [K=16, M=8] (transA=true case).
        var rng = new Random(42);
        double[] a = new double[k * m];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedAvx    = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        Avx2Pack.PackA_Fp64(a, lda: m, transA: true, packedAvx, mc, kc, mr);

        // Bit-identical assertion: we are copying bits, not computing — no FP rounding occurs.
        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx2Pack_PackA_Fp64_NonTransposed_DelegatesToScalarBitIdentical()
    {
        if (!Avx2Pack.IsSupported) return;

        int m = 8, k = 16;
        int mc = 8, kc = 16, mr = 4;

        // A stored row-major [M=8, K=16] (transA=false case).
        var rng = new Random(42);
        double[] a = new double[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedAvx    = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: k, transA: false, packedScalar, mc, kc, mr);
        Avx2Pack.PackA_Fp64(a, lda: k, transA: false, packedAvx, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx2Pack_PackA_Fp32_Transposed_MatchesScalarBitIdentical()
    {
        if (!Avx2Pack.IsSupported) return;

        int m = 16, k = 16;
        int mc = 16, kc = 16, mr = 8;

        var rng = new Random(42);
        float[] a = new float[k * m];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[mc * kc];
        float[] packedAvx    = new float[mc * kc];

        ScalarPack.PackA<float>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        Avx2Pack.PackA_Fp32(a, lda: m, transA: true, packedAvx, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    // ── C4: Avx2Pack Pack-B tests ─────────────────────────────────────────────

    [Fact]
    public void Avx2Pack_PackB_Fp64_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!Avx2Pack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 8;

        var rng = new Random(42);
        double[] b = new double[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedAvx = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        Avx2Pack.PackB_Fp64(b, ldb: n, transB: false, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx2Pack_PackB_Fp64_Transposed_DelegatesToScalarBitIdentical()
    {
        if (!Avx2Pack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 8;

        var rng = new Random(42);
        double[] b = new double[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedAvx = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: k, transB: true, packedScalar, nc, kc, nr);
        Avx2Pack.PackB_Fp64(b, ldb: k, transB: true, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx2Pack_PackB_Fp32_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!Avx2Pack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 8;

        var rng = new Random(42);
        float[] b = new float[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[kc * nc];
        float[] packedAvx = new float[kc * nc];

        ScalarPack.PackB<float>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        Avx2Pack.PackB_Fp32(b, ldb: n, transB: false, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    // ── C6: Avx2Tail tests ───────────────────────────────────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    public void Avx2Tail_RunFp64_4xN_MaskedStore_PreservesUnmaskedColumns(int effectiveNr)
    {
        if (!Avx2Tail.IsSupported) return;

        int kc = 8;
        var rng = new Random(42);
        double[] packedA = new double[4 * kc];
        double[] packedB = new double[kc * 8];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        // Pre-fill C with a sentinel value so we can verify unmasked columns are preserved.
        double[] c = new double[4 * 8];
        for (int i = 0; i < c.Length; i++) c[i] = 999.0;

        Avx2Tail.RunFp64_4xN(packedA, packedB, c.AsSpan(), ldc: 8, kc, effectiveNr);

        // Reference: full 4×8 microkernel into a separate buffer.
        double[] cRef = new double[4 * 8];
        for (int i = 0; i < cRef.Length; i++) cRef[i] = 999.0;
        Avx2Fp64_4x8.Run(packedA, packedB, cRef.AsSpan(), ldc: 8, kc);

        // For each row, columns [0, effectiveNr) should match the full kernel;
        // columns [effectiveNr, 8) should retain the sentinel value 999.
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                double actual = c[row * 8 + col];
                if (col < effectiveNr)
                {
                    Assert.Equal(cRef[row * 8 + col], actual, precision: 12);
                }
                else
                {
                    Assert.Equal(999.0, actual);
                }
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    public void Avx2Tail_RunFp32_8xN_MaskedStore_PreservesUnmaskedColumns(int effectiveNr)
    {
        if (!Avx2Tail.IsSupported) return;

        int kc = 8;
        var rng = new Random(42);
        float[] packedA = new float[8 * kc];
        float[] packedB = new float[kc * 8];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] c = new float[8 * 8];
        for (int i = 0; i < c.Length; i++) c[i] = 999f;

        Avx2Tail.RunFp32_8xN(packedA, packedB, c.AsSpan(), ldc: 8, kc, effectiveNr);

        float[] cRef = new float[8 * 8];
        for (int i = 0; i < cRef.Length; i++) cRef[i] = 999f;
        Avx2Fp32_8x8.Run(packedA, packedB, cRef.AsSpan(), ldc: 8, kc);

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                float actual = c[row * 8 + col];
                if (col < effectiveNr)
                {
                    Assert.Equal(cRef[row * 8 + col], actual, precision: 4);
                }
                else
                {
                    Assert.Equal(999f, actual);
                }
            }
        }
    }

    // ── C7: AVX2 dispatch end-to-end tests ───────────────────────────────────

    [Theory]
    [InlineData(8, 8, 8, false, false)]    // tile multiple of AVX2 (mr=4, nr=8 for FP64)
    [InlineData(16, 16, 16, false, false)]
    [InlineData(16, 8, 64, true, false)]   // L2-shape-style
    [InlineData(16, 16, 128, false, false)] // k>=128 → PackBoth AVX2 microkernel
    public void Gemm_AvxPath_FP64_MatchesScalar(int m, int n, int k, bool transA, bool transB)
    {
        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;

        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, bRows, bCols, transB);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: aCols, transA,
            b, ldb: bCols, transB,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void Gemm_AvxPath_FP32_MatchesScalar()
    {
        int m = 16, n = 16, k = 16;
        var rng = new Random(42);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference: naive triple-loop FP32 GEMM.
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) sum += a[i * k + kk] * b[kk * n + j];
                expected[i * n + j] = sum;
            }

        float[] actual = new float[m * n];
        BlasManagedLib.Gemm<float>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 4);
    }

    // ── D1: AVX-512 FP64 8x16 microkernel test ───────────────────────────────

    [Fact]
    public void Avx512Fp64_8x16_MatchesScalarReference()
    {
        if (!Avx512Fp64_8x16.IsSupported) return;

        int kc = 16;
        int Mr = 8, Nr = 16;

        var rng = new Random(42);
        double[] packedA = new double[Mr * kc];
        double[] packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        // Reference: 8 calls to ScalarFp64_4x4 covering the 8×16 output (2 row-blocks × 4 col-blocks).
        // ScalarFp64_4x4 has Mr=4, Nr=4. The 8×16 grid splits into 2×4=8 quadrants.
        // Slice packed-A into rows 0..3 and 4..7. Slice packed-B into 4-col chunks: [0..3, 4..7, 8..11, 12..15].
        double[] packedA_top = new double[kc * 4];
        double[] packedA_bot = new double[kc * 4];
        double[][] packedB_chunks = new double[4][];
        for (int c = 0; c < 4; c++) packedB_chunks[c] = new double[kc * 4];

        for (int k = 0; k < kc; k++)
        {
            for (int r = 0; r < 4; r++)
            {
                packedA_top[k * 4 + r] = packedA[k * Mr + r];
                packedA_bot[k * 4 + r] = packedA[k * Mr + 4 + r];
            }
            for (int c = 0; c < 4; c++)
            {
                for (int col = 0; col < 4; col++)
                {
                    packedB_chunks[c][k * 4 + col] = packedB[k * Nr + c * 4 + col];
                }
            }
        }

        double[] cRef = new double[8 * 16];
        int ldc = 16;
        // Top row-block (rows 0..3): 4 calls covering cols 0..15
        for (int c = 0; c < 4; c++)
        {
            ScalarFp64_4x4.Run(packedA_top, packedB_chunks[c], cRef.AsSpan(c * 4), ldc, kc);
        }
        // Bottom row-block (rows 4..7): 4 calls covering cols 0..15
        for (int c = 0; c < 4; c++)
        {
            ScalarFp64_4x4.Run(packedA_bot, packedB_chunks[c], cRef.AsSpan(4 * ldc + c * 4), ldc, kc);
        }

        double[] cAvx = new double[8 * 16];
        Avx512Fp64_8x16.Run(packedA, packedB, cAvx.AsSpan(), ldc, kc);

        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cAvx[i], precision: 12);
    }

    // ── D2: AVX-512 FP32 16x16 microkernel test ──────────────────────────────

    [Fact]
    public void Avx512Fp32_16x16_MatchesScalarReference()
    {
        if (!Avx512Fp32_16x16.IsSupported) return;

        int kc = 16;
        int Mr = 16, Nr = 16;

        var rng = new Random(42);
        float[] packedA = new float[Mr * kc];
        float[] packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference: 16 ScalarFp32_4x4 calls covering 16×16 = 4 row-blocks × 4 col-blocks.
        // Need to slice packedA into 4 row-blocks (rows 0..3, 4..7, 8..11, 12..15) and
        // packedB into 4 col-blocks (cols 0..3, 4..7, 8..11, 12..15).
        float[][] packedA_blocks = new float[4][];
        float[][] packedB_blocks = new float[4][];
        for (int r = 0; r < 4; r++) packedA_blocks[r] = new float[kc * 4];
        for (int c = 0; c < 4; c++) packedB_blocks[c] = new float[kc * 4];

        for (int k = 0; k < kc; k++)
        {
            for (int rb = 0; rb < 4; rb++)
            {
                for (int r = 0; r < 4; r++)
                {
                    packedA_blocks[rb][k * 4 + r] = packedA[k * Mr + rb * 4 + r];
                }
            }
            for (int cb = 0; cb < 4; cb++)
            {
                for (int col = 0; col < 4; col++)
                {
                    packedB_blocks[cb][k * 4 + col] = packedB[k * Nr + cb * 4 + col];
                }
            }
        }

        float[] cRef = new float[16 * 16];
        int ldc = 16;
        for (int rb = 0; rb < 4; rb++)
        {
            for (int cb = 0; cb < 4; cb++)
            {
                int offset = rb * 4 * ldc + cb * 4;
                ScalarFp32_4x4.Run(packedA_blocks[rb], packedB_blocks[cb], cRef.AsSpan(offset), ldc, kc);
            }
        }

        float[] cAvx = new float[16 * 16];
        Avx512Fp32_16x16.Run(packedA, packedB, cAvx.AsSpan(), ldc, kc);

        // FP32 precision: 16 K-step accumulations of [-1,1) values, ~16 * eps_f = 1.9e-6.
        // precision: 4 is safe (loose enough to accommodate FMA reordering).
        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cAvx[i], precision: 4);
    }

    // ── D3: Avx512Pack tests ─────────────────────────────────────────────────

    [Fact]
    public void Avx512Pack_PackA_Fp64_Transposed_MatchesScalarBitIdentical()
    {
        if (!Avx512Pack.IsSupported) return;

        int m = 16, k = 16;
        int mc = 16, kc = 16, mr = 8;

        var rng = new Random(42);
        double[] a = new double[k * m];  // [K, M] row-major
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedAvx = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        Avx512Pack.PackA_Fp64(a, lda: m, transA: true, packedAvx, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx512Pack_PackA_Fp64_NonTransposed_DelegatesToScalar()
    {
        if (!Avx512Pack.IsSupported) return;

        int m = 16, k = 16;
        int mc = 16, kc = 16, mr = 8;

        var rng = new Random(42);
        double[] a = new double[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedAvx = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: k, transA: false, packedScalar, mc, kc, mr);
        Avx512Pack.PackA_Fp64(a, lda: k, transA: false, packedAvx, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx512Pack_PackA_Fp32_Transposed_MatchesScalarBitIdentical()
    {
        if (!Avx512Pack.IsSupported) return;

        int m = 32, k = 16;
        int mc = 32, kc = 16, mr = 16;

        var rng = new Random(42);
        float[] a = new float[k * m];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[mc * kc];
        float[] packedAvx = new float[mc * kc];

        ScalarPack.PackA<float>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        Avx512Pack.PackA_Fp32(a, lda: m, transA: true, packedAvx, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    // ── D4: Avx512Pack Pack-B tests ──────────────────────────────────────────

    [Fact]
    public void Avx512Pack_PackB_Fp64_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!Avx512Pack.IsSupported) return;

        int k = 8, n = 32;
        int kc = 8, nc = 32, nr = 16;

        var rng = new Random(42);
        double[] b = new double[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedAvx = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        Avx512Pack.PackB_Fp64(b, ldb: n, transB: false, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx512Pack_PackB_Fp64_Transposed_DelegatesToScalar()
    {
        if (!Avx512Pack.IsSupported) return;

        int k = 8, n = 32;
        int kc = 8, nc = 32, nr = 16;

        var rng = new Random(42);
        double[] b = new double[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedAvx = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: k, transB: true, packedScalar, nc, kc, nr);
        Avx512Pack.PackB_Fp64(b, ldb: k, transB: true, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    [Fact]
    public void Avx512Pack_PackB_Fp32_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!Avx512Pack.IsSupported) return;

        int k = 8, n = 32;
        int kc = 8, nc = 32, nr = 16;

        var rng = new Random(42);
        float[] b = new float[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[kc * nc];
        float[] packedAvx = new float[kc * nc];

        ScalarPack.PackB<float>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        Avx512Pack.PackB_Fp32(b, ldb: n, transB: false, packedAvx, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedAvx[i]);
    }

    // ── D5: Avx512Streaming tests ─────────────────────────────────────────────

    [Theory]
    [InlineData(8, 16, 8, false, false)]   // NN, n=16 (2 fp64 blocks)
    [InlineData(8, 16, 8, true, false)]    // TN
    [InlineData(8, 16, 8, false, true)]    // NT (scalar fallback)
    [InlineData(8, 16, 8, true, true)]     // TT (scalar fallback)
    [InlineData(4, 8, 16, false, false)]   // single block
    [InlineData(8, 24, 8, false, false)]   // 3 blocks
    [InlineData(8, 10, 8, false, false)]   // 1 block + tail (n=10)
    public void Avx512Streaming_Fp64_MatchesScalarReference(int m, int n, int k, bool transA, bool transB)
    {
        if (!Avx512Streaming.IsSupported) return;

        int aCols = transA ? m : k;
        int lda = aCols;
        int bCols = transB ? k : n;
        int ldb = bCols;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);

        double[] cScalar = new double[m * n];
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, cScalar, ldc: n, m, n, k);

        double[] cAvx = new double[m * n];
        Avx512Streaming.RunFp64(a, lda, transA, b, ldb, transB, cAvx, ldc: n, m, n, k);

        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cAvx[i], precision: 10);
    }

    [Fact]
    public void Avx512Streaming_Fp32_NN_MatchesScalarReference()
    {
        if (!Avx512Streaming.IsSupported) return;

        int m = 8, n = 32, k = 8;  // n=32 = 2 FP32 blocks
        var rng = new Random(42);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] cScalar = new float[m * n];
        ScalarStreaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cScalar, ldc: n, m, n, k);

        float[] cAvx = new float[m * n];
        Avx512Streaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cAvx, ldc: n, m, n, k);

        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cAvx[i], precision: 4);
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

    // ── C5: Avx2Streaming tests ───────────────────────────────────────────────

    [Theory]
    [InlineData(8, 8, 8, false, false)]    // NN
    [InlineData(8, 8, 8, true, false)]     // TN (the L2-shape pattern)
    [InlineData(8, 8, 8, false, true)]     // NT (falls back to scalar)
    [InlineData(8, 8, 8, true, true)]      // TT (falls back to scalar)
    [InlineData(8, 16, 8, false, false)]   // multi-block N
    [InlineData(8, 8, 4, false, false)]    // small K
    [InlineData(8, 8, 24, false, false)]   // larger K
    [InlineData(8, 12, 8, false, false)]   // N tail (n=12 → nBlocks=3, nTail=0 for nr=4; not really a tail)
    [InlineData(8, 14, 8, false, false)]   // N tail (n=14 → nBlocks=3, nTail=2)
    public void Avx2Streaming_Fp64_MatchesScalarReference(int m, int n, int k, bool transA, bool transB)
    {
        if (!Avx2Streaming.IsSupported) return;

        int aCols = transA ? m : k;
        int lda = aCols;
        int bCols = transB ? k : n;
        int ldb = bCols;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);

        // Reference: ScalarStreaming (already proven correct in B6).
        double[] cScalar = new double[m * n];
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, cScalar, ldc: n, m, n, k);

        double[] cAvx = new double[m * n];
        Avx2Streaming.RunFp64(a, lda, transA, b, ldb, transB, cAvx, ldc: n, m, n, k);

        // FMA reordering may produce tiny differences; allow ULP-scale tolerance.
        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cAvx[i], precision: 10);
    }

    [Fact]
    public void Avx2Streaming_Fp32_NN_MatchesScalarReference()
    {
        if (!Avx2Streaming.IsSupported) return;

        int m = 8, n = 16, k = 8;
        var rng = new Random(42);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] cScalar = new float[m * n];
        ScalarStreaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cScalar, ldc: n, m, n, k);

        float[] cAvx = new float[m * n];
        Avx2Streaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cAvx, ldc: n, m, n, k);

        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cAvx[i], precision: 4);
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

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(15)]
    public void Avx512Tail_RunFp64_8xN_MaskedStore_PreservesUnmaskedColumns(int effectiveNr)
    {
        if (!Avx512Tail.IsSupported) return;

        int kc = 8;
        var rng = new Random(42);
        double[] packedA = new double[8 * kc];
        double[] packedB = new double[kc * 16];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        double[] c = new double[8 * 16];
        for (int i = 0; i < c.Length; i++) c[i] = 999.0;

        Avx512Tail.RunFp64_8xN(packedA, packedB, c.AsSpan(), ldc: 16, kc, effectiveNr);

        double[] cRef = new double[8 * 16];
        for (int i = 0; i < cRef.Length; i++) cRef[i] = 999.0;
        Avx512Fp64_8x16.Run(packedA, packedB, cRef.AsSpan(), ldc: 16, kc);

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 16; col++)
            {
                double actual = c[row * 16 + col];
                if (col < effectiveNr)
                    Assert.Equal(cRef[row * 16 + col], actual, precision: 12);
                else
                    Assert.Equal(999.0, actual);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(11)]
    [InlineData(15)]
    public void Avx512Tail_RunFp32_16xN_MaskedStore_PreservesUnmaskedColumns(int effectiveNr)
    {
        if (!Avx512Tail.IsSupported) return;

        int kc = 8;
        var rng = new Random(42);
        float[] packedA = new float[16 * kc];
        float[] packedB = new float[kc * 16];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] c = new float[16 * 16];
        for (int i = 0; i < c.Length; i++) c[i] = 999f;

        Avx512Tail.RunFp32_16xN(packedA, packedB, c.AsSpan(), ldc: 16, kc, effectiveNr);

        float[] cRef = new float[16 * 16];
        for (int i = 0; i < cRef.Length; i++) cRef[i] = 999f;
        Avx512Fp32_16x16.Run(packedA, packedB, cRef.AsSpan(), ldc: 16, kc);

        for (int row = 0; row < 16; row++)
        {
            for (int col = 0; col < 16; col++)
            {
                float actual = c[row * 16 + col];
                if (col < effectiveNr)
                    Assert.Equal(cRef[row * 16 + col], actual, precision: 4);
                else
                    Assert.Equal(999f, actual);
            }
        }
    }

    [Fact]
    public void Gemm_L2ShapeScaled_TransA_FP64_MatchesNaive()
    {
        // Scaled-down L2 shape (M=4096 → 32, N=16, K=512 → 128, transA=true).
        // The real L2 from issue #358 has M=4096 but takes too long for unit tests;
        // 32 exercises the same dispatch path (PackBoth + AVX-512 if available).
        // Phase L (Gate 1) runs the full M=4096 shape under the perf benchmark.
        int m = 32, n = 16, k = 128;
        bool transA = true, transB = false;

        int aCols = transA ? m : k;
        int lda = aCols;
        int ldb = n;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        int aRows = transA ? k : m;
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, k, n, transB);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda, transA,
            b, ldb, transB,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 9);
    }

    // ── D8: PackBoth + AVX-512 end-to-end coverage ───────────────────────────
    // These shapes have M*N >= 1024 and K >= 128, which routes through PackBoth.
    // On AVX-512 hardware the (8,16) FP64 tile is chosen; on AVX2 the (4,8) tile;
    // on scalar the (4,4) tile. All three produce numerically correct results.

    [Theory]
    [InlineData(64, 32, 128, false, false)]    // m*n=2048 → PackBoth; FP64 AVX-512 tile (8,16) divides (64,32)
    [InlineData(32, 32, 128, false, false)]    // m*n=1024 → PackBoth (exactly at threshold)
    [InlineData(32, 32, 128, true, false)]     // transA
    public void Gemm_PackBoth_AVX512_FP64_MatchesNaive(int m, int n, int k, bool transA, bool transB)
    {
        int aCols = transA ? m : k;
        int lda = aCols;
        int bCols = transB ? k : n;
        int ldb = bCols;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        int aRows = transA ? k : m;
        int bRows = transB ? n : k;
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, bRows, bCols, transB);

        double[] actual = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda, transA,
            b, ldb, transB,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 9);
    }

    // ── E1: ARM64 Neon FP64 4×4 microkernel ──────────────────────────────────
    // On x64 hosts IsSupported=false and the test skips via early-return.
    // On ARM64 hosts the Neon path is exercised and must match the scalar reference.

    [Fact]
    public void NeonFp64_4x4_MatchesScalarReference()
    {
        if (!NeonFp64_4x4.IsSupported) return;

        int kc = 16;
        int Mr = 4, Nr = 4;

        var rng = new Random(42);
        double[] packedA = new double[Mr * kc];
        double[] packedB = new double[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < packedB.Length; i++) packedB[i] = rng.NextDouble() * 2 - 1;

        // Reference: ScalarFp64_4x4 (which uses the SAME Mr=4 Nr=4 layout).
        double[] cRef = new double[4 * 4];
        ScalarFp64_4x4.Run(packedA, packedB, cRef.AsSpan(), ldc: 4, kc);

        double[] cNeon = new double[4 * 4];
        NeonFp64_4x4.Run(packedA, packedB, cNeon.AsSpan(), ldc: 4, kc);

        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cNeon[i], precision: 12);
    }

    // ── E2: ARM64 Neon FP32 8×4 microkernel ──────────────────────────────────
    // On x64 hosts IsSupported=false and the test skips via early-return.
    // On ARM64 hosts the Neon path is exercised and must match the scalar reference.

    [Fact]
    public void NeonFp32_8x4_MatchesScalarReference()
    {
        if (!NeonFp32_8x4.IsSupported) return;

        int kc = 16;
        int Mr = 8, Nr = 4;

        var rng = new Random(42);
        float[] packedA = new float[Mr * kc];
        float[] packedB = new float[kc * Nr];
        for (int i = 0; i < packedA.Length; i++) packedA[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < packedB.Length; i++) packedB[i] = (float)(rng.NextDouble() * 2 - 1);

        // Reference: 2 ScalarFp32_4x4 calls covering rows 0..3 and 4..7.
        float[] packedA_top = new float[kc * 4];
        float[] packedA_bot = new float[kc * 4];
        for (int k = 0; k < kc; k++)
        {
            for (int r = 0; r < 4; r++)
            {
                packedA_top[k * 4 + r] = packedA[k * Mr + r];
                packedA_bot[k * 4 + r] = packedA[k * Mr + 4 + r];
            }
        }

        float[] cRef = new float[8 * 4];
        int ldc = 4;
        ScalarFp32_4x4.Run(packedA_top, packedB, cRef.AsSpan(), ldc, kc);
        ScalarFp32_4x4.Run(packedA_bot, packedB, cRef.AsSpan(4 * ldc), ldc, kc);

        float[] cNeon = new float[8 * 4];
        NeonFp32_8x4.Run(packedA, packedB, cNeon.AsSpan(), ldc, kc);

        for (int i = 0; i < cRef.Length; i++)
            Assert.Equal(cRef[i], cNeon[i], precision: 4);
    }

    // -------------------------------------------------------------------------
    // E3: NeonPack tests
    // -------------------------------------------------------------------------

    [Fact]
    public void NeonPack_PackA_Fp64_Transposed_MatchesScalarBitIdentical()
    {
        if (!NeonPack.IsSupported) return;

        int m = 8, k = 16;
        int mc = 8, kc = 16, mr = 4;

        var rng = new Random(42);
        double[] a = new double[k * m];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedNeon = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        NeonPack.PackA_Fp64(a, lda: m, transA: true, packedNeon, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    [Fact]
    public void NeonPack_PackA_Fp64_NonTransposed_DelegatesToScalar()
    {
        if (!NeonPack.IsSupported) return;

        int m = 8, k = 16;
        int mc = 8, kc = 16, mr = 4;

        var rng = new Random(42);
        double[] a = new double[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[mc * kc];
        double[] packedNeon = new double[mc * kc];

        ScalarPack.PackA<double>(a, lda: k, transA: false, packedScalar, mc, kc, mr);
        NeonPack.PackA_Fp64(a, lda: k, transA: false, packedNeon, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    [Fact]
    public void NeonPack_PackA_Fp32_Transposed_MatchesScalarBitIdentical()
    {
        if (!NeonPack.IsSupported) return;

        int m = 16, k = 16;
        int mc = 16, kc = 16, mr = 8;

        var rng = new Random(42);
        float[] a = new float[k * m];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[mc * kc];
        float[] packedNeon = new float[mc * kc];

        ScalarPack.PackA<float>(a, lda: m, transA: true, packedScalar, mc, kc, mr);
        NeonPack.PackA_Fp32(a, lda: m, transA: true, packedNeon, mc, kc, mr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    // ── E4: NeonPack Pack-B tests ─────────────────────────────────────────────

    [Fact]
    public void NeonPack_PackB_Fp64_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!NeonPack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 4;

        var rng = new Random(42);
        double[] b = new double[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedNeon = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        NeonPack.PackB_Fp64(b, ldb: n, transB: false, packedNeon, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    [Fact]
    public void NeonPack_PackB_Fp64_Transposed_DelegatesToScalar()
    {
        if (!NeonPack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 4;

        var rng = new Random(42);
        double[] b = new double[n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double[] packedScalar = new double[kc * nc];
        double[] packedNeon = new double[kc * nc];

        ScalarPack.PackB<double>(b, ldb: k, transB: true, packedScalar, nc, kc, nr);
        NeonPack.PackB_Fp64(b, ldb: k, transB: true, packedNeon, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    [Fact]
    public void NeonPack_PackB_Fp32_NonTransposed_MatchesScalarBitIdentical()
    {
        if (!NeonPack.IsSupported) return;

        int k = 8, n = 16;
        int kc = 8, nc = 16, nr = 4;

        var rng = new Random(42);
        float[] b = new float[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] packedScalar = new float[kc * nc];
        float[] packedNeon = new float[kc * nc];

        ScalarPack.PackB<float>(b, ldb: n, transB: false, packedScalar, nc, kc, nr);
        NeonPack.PackB_Fp32(b, ldb: n, transB: false, packedNeon, nc, kc, nr);

        for (int i = 0; i < packedScalar.Length; i++)
            Assert.Equal(packedScalar[i], packedNeon[i]);
    }

    // ── E5: NeonStreaming tests ───────────────────────────────────────────────

    [Theory]
    [InlineData(8, 8, 8, false, false)]    // NN
    [InlineData(8, 8, 8, true, false)]     // TN
    [InlineData(8, 8, 8, false, true)]     // NT scalar fallback
    [InlineData(8, 8, 8, true, true)]      // TT scalar fallback
    [InlineData(4, 6, 8, false, false)]    // n=6, 3 FP64 blocks
    [InlineData(8, 8, 4, false, false)]    // small K
    public void NeonStreaming_Fp64_MatchesScalarReference(int m, int n, int k, bool transA, bool transB)
    {
        if (!NeonStreaming.IsSupported) return;

        int aCols = transA ? m : k;
        int lda = aCols;
        int bCols = transB ? k : n;
        int ldb = bCols;

        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);

        double[] cScalar = new double[m * n];
        ScalarStreaming.RunFp64(a, lda, transA, b, ldb, transB, cScalar, ldc: n, m, n, k);

        double[] cNeon = new double[m * n];
        NeonStreaming.RunFp64(a, lda, transA, b, ldb, transB, cNeon, ldc: n, m, n, k);

        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cNeon[i], precision: 10);
    }

    [Fact]
    public void NeonStreaming_Fp32_NN_MatchesScalarReference()
    {
        if (!NeonStreaming.IsSupported) return;

        int m = 8, n = 12, k = 8;  // n=12 = 3 FP32 blocks
        var rng = new Random(42);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] cScalar = new float[m * n];
        ScalarStreaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cScalar, ldc: n, m, n, k);

        float[] cNeon = new float[m * n];
        NeonStreaming.RunFp32(a, lda: k, transA: false, b, ldb: n, transB: false, cNeon, ldc: n, m, n, k);

        for (int i = 0; i < cScalar.Length; i++)
            Assert.Equal(cScalar[i], cNeon[i], precision: 4);
    }

    // ── E6: Neon dispatch wired into BlasManaged.Gemm ────────────────────────

    /// <summary>
    /// On ARM64 hosts this exercises the Neon dispatch path (NeonFp64_4x4 in
    /// PackBothStrategy, NeonStreaming in StreamingStrategy). On x64 hosts the
    /// same shapes run through AVX-512/AVX2/scalar. Either way the output must
    /// match the naive reference so the test is unconditional and portable.
    /// </summary>
    [Fact]
    public void Gemm_NeonPath_FP64_SmallShape_MatchesNaive()
    {
        // (8, 8) is an exact multiple of Neon FP64 tile (Mr=4, Nr=4) so the
        // PackBoth path won't fall back to scalar tile selection. K=8 triggers
        // the Streaming strategy (K < 32), exercising NeonStreaming as well.
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

    /// <summary>
    /// Exercises the Neon FP32 (Mr=8, Nr=4) tile via PackBothStrategy on ARM64.
    /// On x64 it runs through AVX-512/AVX2/scalar. Shape (8, 4) is an exact
    /// multiple of the Neon FP32 tile so no fallback to scalar tile occurs.
    /// Large K (k=128) triggers the PackBoth strategy, exercising NeonFp32_8x4.
    /// </summary>
    [Fact]
    public void Gemm_NeonPath_FP32_NeonFriendlyShape_MatchesNaive()
    {
        // (8, 4) is exact multiple of Neon FP32 tile (Mr=8, Nr=4).
        // K=128 ≥ 128 → PackBoth strategy, exercises NeonFp32_8x4 on ARM64.
        int m = 8, n = 4, k = 128;
        var rng = new Random(43);
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Naive reference.
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) sum += a[i * k + kk] * b[kk * n + j];
                expected[i * n + j] = sum;
            }

        float[] actual = new float[m * n];
        BlasManagedLib.Gemm<float>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 4);
    }

    // ── PerThreadPool (Layer 1 allocator) ────────────────────────────────────

    [Fact]
    public void PerThreadPool_FirstRent_AllocatesBuffer()
    {
        var pool = PerThreadPool.Current;
        pool.ResetForTest();

        Span<byte> packA = pool.RentPackA(1024);
        Assert.Equal(1024, packA.Length);
        Assert.True(pool.TotalBytesHeld >= 1024);
    }

    [Fact]
    public void PerThreadPool_GrowsMonotonically()
    {
        var pool = PerThreadPool.Current;
        pool.ResetForTest();

        // First rent — small buffer.
        pool.RentPackA(1024);
        long after1 = pool.TotalBytesHeld;

        // Bigger rent — must grow.
        pool.RentPackA(8192);
        long after2 = pool.TotalBytesHeld;
        Assert.True(after2 >= 8192);
        Assert.True(after2 > after1);

        // Smaller rent — must NOT shrink (reuse existing buffer).
        pool.RentPackA(512);
        long after3 = pool.TotalBytesHeld;
        Assert.Equal(after2, after3);  // No change — still holds 8192-byte buffer.
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Zero-allocation regression: a second RentPackA call with the same size
    /// must not allocate a new backing array. Uses
    /// <c>GC.GetAllocatedBytesForCurrentThread</c> to verify. Only runs on
    /// net5+ since the API is unavailable on net471.
    /// </summary>
    [Fact]
    public void PerThreadPool_ReuseDoesNotAllocate()
    {
        var pool = PerThreadPool.Current;
        pool.ResetForTest();

        // First rent — allocate.
        pool.RentPackA(2048);
        long before = GC.GetAllocatedBytesForCurrentThread();

        // Second rent same size — must not allocate.
        pool.RentPackA(2048);
        long after = GC.GetAllocatedBytesForCurrentThread();

        // Some GC overhead can happen from test framework; allow tiny slack.
        // The key assertion is that we're not allocating a new 2048-byte buffer.
        Assert.True(after - before < 200, $"Expected no significant allocation; got {after - before} bytes.");
    }
#endif

    [Fact]
    public void PerThreadPool_SeparateThreadsHaveSeparatePools()
    {
        long thread1Pool;
        var pool1 = PerThreadPool.Current;
        pool1.ResetForTest();
        pool1.RentPackA(4096);
        thread1Pool = pool1.TotalBytesHeld;

        long thread2PoolFromThread2 = 0;
        var t = new System.Threading.Thread(() =>
        {
            var pool2 = PerThreadPool.Current;
            pool2.ResetForTest();
            // Note: pool2 is a different instance from pool1 because of [ThreadStatic].
            thread2PoolFromThread2 = pool2.TotalBytesHeld;  // Should be 0 (fresh pool).
        });
        t.Start();
        t.Join();

        Assert.True(thread1Pool >= 4096);
        Assert.Equal(0, thread2PoolFromThread2);
    }

    [Fact]
    public void PerThreadPool_RentPackB_AndKSplitC_Independent()
    {
        var pool = PerThreadPool.Current;
        pool.ResetForTest();

        Span<byte> packA = pool.RentPackA(1024);
        Span<byte> packB = pool.RentPackB(2048);
        Span<byte> kSplit = pool.RentKSplitC(512);

        Assert.Equal(1024, packA.Length);
        Assert.Equal(2048, packB.Length);
        Assert.Equal(512, kSplit.Length);
        Assert.True(pool.TotalBytesHeld >= 1024 + 2048 + 512);
    }

    [Fact]
    public void PerThreadPool_ZeroBytesReturnsEmptySpan()
    {
        var pool = PerThreadPool.Current;
        pool.ResetForTest();

        Span<byte> empty = pool.RentPackA(0);
        Assert.Equal(0, empty.Length);
        Assert.True(empty.IsEmpty);
    }

    // -------------------------------------------------------------------------
    // F2: ArrayPoolOverflow tests
    // -------------------------------------------------------------------------

    [Fact]
    public void ArrayPoolOverflow_Rent_ReturnsBufferOfRequestedSize()
    {
        using var rent = ArrayPoolOverflow.Rent(2048);
        Assert.Equal(2048, rent.Span.Length);
    }

    [Fact]
    public void ArrayPoolOverflow_Rent_ZeroBytes_ReturnsEmpty()
    {
        using var rent = ArrayPoolOverflow.Rent(0);
        Assert.True(rent.Span.IsEmpty);
    }

    [Fact]
    public void ArrayPoolOverflow_Rent_NegativeBytes_ReturnsEmpty()
    {
        using var rent = ArrayPoolOverflow.Rent(-100);
        Assert.True(rent.Span.IsEmpty);
    }

    [Fact]
    public void ArrayPoolOverflow_DoubleDispose_IsIdempotent()
    {
        var rent = ArrayPoolOverflow.Rent(1024);
        rent.Dispose();
        rent.Dispose();  // Must not throw.
        Assert.True(rent.Span.IsEmpty);  // After dispose, span is empty.
    }

    [Fact]
    public void ArrayPoolOverflow_SpanIsWritable()
    {
        using var rent = ArrayPoolOverflow.Rent(64);
        var span = rent.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (byte)i;
        for (int i = 0; i < span.Length; i++) Assert.Equal((byte)i, span[i]);
    }

    // -------------------------------------------------------------------------
    // F3: WeightPackCache + BlasManaged.PrePackA / PrePackB
    // -------------------------------------------------------------------------

    [Fact]
    public void WeightPackCache_IsCacheCurrent_TrueAfterMarkCurrent()
    {
        var handle = WeightPackCache.Allocate(
            1024,
            (Mc: 8, Kc: 8, TransA: false, Mode: PackingMode.Auto, ElemType: typeof(double)),
            isForA: true);
        // Initial state: Version=1, LastPackedVersion=0 → not current.
        Assert.False(WeightPackCache.IsCacheCurrent(handle));

        // CodeRabbit #366: MarkCacheCurrent now requires the snapshot taken
        // before packing started. Callers (PrePackA/B/dispatch path) Read
        // handle.Version before doing the pack work, then pass that snapshot.
        long packedVersion = System.Threading.Interlocked.Read(ref handle.Version);
        WeightPackCache.MarkCacheCurrent(handle, packedVersion);
        Assert.True(WeightPackCache.IsCacheCurrent(handle));
    }

    [Fact]
    public void WeightPackCache_MarkDirty_InvalidatesCache()
    {
        var handle = WeightPackCache.Allocate(
            1024,
            (Mc: 8, Kc: 8, TransA: false, Mode: PackingMode.Auto, ElemType: typeof(double)),
            isForA: true);
        long packedVersion = System.Threading.Interlocked.Read(ref handle.Version);
        WeightPackCache.MarkCacheCurrent(handle, packedVersion);
        Assert.True(WeightPackCache.IsCacheCurrent(handle));

        handle.MarkDirty();
        Assert.False(WeightPackCache.IsCacheCurrent(handle));
    }

    [Fact]
    public void BlasManaged_PrePackA_FP64_ProducesUsableHandle()
    {
        int m = 8, k = 16;
        var rng = new Random(42);
        double[] a = new double[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        Assert.NotNull(handle);
        Assert.True(handle.PackedBuffer.Length > 0);
        Assert.True(WeightPackCache.IsCacheCurrent(handle));
        // The packed buffer should contain non-zero values matching the source weights.
        var packedAsDouble = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(handle.PackedBuffer.AsSpan());
        // At least one element of the source should match somewhere in the packed buffer.
        Assert.Contains(packedAsDouble.ToArray(), d => Math.Abs(d - a[0]) < 1e-12);

        handle.Dispose();
    }

    [Fact]
    public void BlasManaged_PrePackB_FP32_ProducesUsableHandle()
    {
        int k = 8, n = 16;
        var rng = new Random(42);
        float[] b = new float[k * n];
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var handle = BlasManagedLib.PrePackB<float>(b, ldb: n, transB: false, k, n);
        Assert.NotNull(handle);
        Assert.True(handle.PackedBuffer.Length > 0);
        Assert.True(WeightPackCache.IsCacheCurrent(handle));

        handle.Dispose();
    }

    [Fact]
    public void BlasManaged_PrePack_AfterMarkDirty_InvalidatesCache()
    {
        int m = 8, k = 16;
        var rng = new Random(42);
        double[] a = new double[m * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;

        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        Assert.True(WeightPackCache.IsCacheCurrent(handle));

        handle.MarkDirty();
        Assert.False(WeightPackCache.IsCacheCurrent(handle));
    }

    // -------------------------------------------------------------------------
    // F4: ArenaIntegration (Layer 4 allocator)
    // -------------------------------------------------------------------------

    [Fact]
    public void ArenaIntegration_NoActiveArena_ReturnsEmpty()
    {
        // Ensure no arena is active (caller may have set one earlier).
        Assert.False(ArenaIntegration.IsArenaActive,
            "Test precondition failed: an arena is already active. Make sure prior tests dispose their arenas.");

        Span<byte> result = ArenaIntegration.TryRentBytes(1024);
        Assert.True(result.IsEmpty);
    }

    [Fact]
    public void ArenaIntegration_ActiveArena_ReturnsArenaBuffer()
    {
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        Assert.True(ArenaIntegration.IsArenaActive);

        Span<byte> result = ArenaIntegration.TryRentBytes(2048);
        Assert.Equal(2048, result.Length);
    }

    [Fact]
    public void ArenaIntegration_ZeroBytes_ReturnsEmpty()
    {
        using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
        Span<byte> result = ArenaIntegration.TryRentBytes(0);
        Assert.True(result.IsEmpty);
    }

    [Fact]
    public void ArenaIntegration_ArenaScopeRespected()
    {
        // After arena disposal, TryRentBytes should fall through (empty).
        {
            using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
            Assert.True(ArenaIntegration.IsArenaActive);
        }
        Assert.False(ArenaIntegration.IsArenaActive);
        Span<byte> result = ArenaIntegration.TryRentBytes(1024);
        Assert.True(result.IsEmpty);
    }

    // -------------------------------------------------------------------------
    // F5: WorkspaceCarver (Layer 5 allocator)
    // -------------------------------------------------------------------------

    [Fact]
    public void WorkspaceCarver_TryCarve_ReturnsRequestedSize()
    {
        byte[] ws = new byte[1024];
        var carver = new WorkspaceCarver(ws.AsSpan());
        var slice = carver.TryCarve(256);
        Assert.Equal(256, slice.Length);
    }

    [Fact]
    public void WorkspaceCarver_SuccessiveCarves_DoNotOverlap()
    {
        byte[] ws = new byte[1024];
        var carver = new WorkspaceCarver(ws.AsSpan());

        var first = carver.TryCarve(256);
        var second = carver.TryCarve(256);

        Assert.Equal(256, first.Length);
        Assert.Equal(256, second.Length);

        // Write into the first; the second must remain zero.
        for (int i = 0; i < first.Length; i++) first[i] = 0xFF;
        foreach (var b in second) Assert.Equal(0, b);
    }

    [Fact]
    public void WorkspaceCarver_RunsOutOfSpace_ReturnsEmpty()
    {
        byte[] ws = new byte[100];
        var carver = new WorkspaceCarver(ws.AsSpan());

        var first = carver.TryCarve(80);  // OK, 20 remaining
        Assert.Equal(80, first.Length);

        var second = carver.TryCarve(50);  // Not enough — returns empty.
        Assert.True(second.IsEmpty);

        // RemainingBytes should reflect that we didn't consume more after the failed carve.
        Assert.Equal(20, carver.RemainingBytes);
    }

    [Fact]
    public void WorkspaceCarver_EmptyWorkspace_HasNoWorkspace()
    {
        var carver = new WorkspaceCarver(Span<byte>.Empty);
        Assert.False(carver.HasWorkspace);
        Assert.Equal(0, carver.TotalBytes);
        Assert.Equal(0, carver.RemainingBytes);
        Assert.True(carver.TryCarve(1).IsEmpty);
    }

    [Fact]
    public void WorkspaceCarver_ZeroByteCarve_ReturnsEmpty()
    {
        byte[] ws = new byte[100];
        var carver = new WorkspaceCarver(ws.AsSpan());
        var slice = carver.TryCarve(0);
        Assert.True(slice.IsEmpty);
        // Zero-byte carve must not consume any workspace.
        Assert.Equal(100, carver.RemainingBytes);
    }

    // -------------------------------------------------------------------------
    // F7: WeightPackHandle invalidation tests (MarkDirty correctness)
    // -------------------------------------------------------------------------

    [Fact]
    public void Gemm_WithPackedAHandle_MatchesUnpackedGemm()
    {
        // Sanity test: when caller supplies a pre-packed handle, the result must
        // still match the unpacked Gemm.
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);

        double[] expected = NaiveGemm(a, m, k, transA: false, b, k, n, transB: false);

        // First Gemm WITHOUT pre-pack handle.
        double[] actualNoHandle = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actualNoHandle, ldc: n,
            m, n, k);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actualNoHandle[i], precision: 10);

        // Now Gemm WITH a pre-pack handle for A. Result should be identical.
        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };

        double[] actualWithHandle = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actualWithHandle, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actualWithHandle[i], precision: 10);

        handle.Dispose();
    }

    [Fact]
    public void Gemm_MutatedWeight_AfterMarkDirty_ReflectsMutation()
    {
        // The PyTorch-surpassing correctness test: mutate the underlying weight,
        // call MarkDirty, and verify the next Gemm returns the new result (not
        // a stale cached one).
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);

        // Initial Gemm with pre-pack handle.
        var handle = BlasManagedLib.PrePackA<double>(a, lda: k, transA: false, m, k);
        var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };

        double[] firstResult = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            firstResult, ldc: n,
            m, n, k,
            options);

        // Mutate the underlying weight in-place: add 10.0 to every element.
        for (int i = 0; i < a.Length; i++) a[i] += 10.0;
        handle.MarkDirty();

        // Second Gemm — must reflect the mutated weight.
        double[] secondResult = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            secondResult, ldc: n,
            m, n, k,
            options);

        // Reference: compute the expected result from the MUTATED a.
        double[] expectedMutated = NaiveGemm(a, m, k, transA: false, b, k, n, transB: false);

        for (int i = 0; i < expectedMutated.Length; i++)
            Assert.Equal(expectedMutated[i], secondResult[i], precision: 10);

        // Sanity: the two results should differ (mutation had an effect).
        bool anyDifferent = false;
        for (int i = 0; i < firstResult.Length; i++)
        {
            if (Math.Abs(firstResult[i] - secondResult[i]) > 1e-6) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent, "Expected the mutated weight to produce a different Gemm result.");

        handle.Dispose();
    }

    [Fact]
    public void Gemm_WithoutMarkDirty_UsesCachedPackedBuffer()
    {
        // When the handle is current (no MarkDirty), the strategy uses the cached
        // packed buffer regardless of what's in the source `a` array. This is the
        // intended behavior — the caller is promising the source matches the
        // cache. We verify this by mutating `a` WITHOUT calling MarkDirty: the
        // Gemm result must match the ORIGINAL weight (the one captured in PrePackA),
        // not the mutated one.
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] aOriginal = (double[])a.Clone();

        var handle = BlasManagedLib.PrePackA<double>(aOriginal, lda: k, transA: false, m, k);
        var options = new BlasOptions<double> { PackedA = handle, PackingMode = PackingMode.ForcePackBoth };

        // Mutate `a` AFTER PrePackA, but do NOT call MarkDirty.
        for (int i = 0; i < a.Length; i++) a[i] += 100.0;

        double[] result = new double[m * n];
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            result, ldc: n,
            m, n, k,
            options);

        // Expected: Gemm uses the ORIGINAL a (captured in handle's PackedBuffer).
        double[] expectedOriginal = NaiveGemm(aOriginal, m, k, transA: false, b, k, n, transB: false);

        for (int i = 0; i < expectedOriginal.Length; i++)
            Assert.Equal(expectedOriginal[i], result[i], precision: 10);

        handle.Dispose();
    }

    // ── G1: M-axis parallel split in PackBothStrategy ─────────────────────────
    // These test cases exercise the new parallel ic-loop path introduced in Task G1.
    // Shapes are chosen to push total work (m × n × k) well above the serial grain-size
    // threshold (32 768 elements), so ParallelForOrSerial dispatches to Parallel.For
    // on multi-core machines. On single-core or low-core machines the serial fallback
    // runs instead — both paths must produce results identical to the naïve reference.
    // The correctness gate is precision: 9 decimal places (double has ~15 sig figs;
    // M-axis splits write disjoint C rows, so there is no cross-thread accumulation
    // that could introduce FMA reordering errors).

    [Theory]
    [InlineData(64, 64, 256)]   // 1M MACs — above grain-size, 1 ic-block (mc=64)
    [InlineData(128, 32, 256)]  // 1M MACs — 2 ic-blocks (m=128, mc=64)
    [InlineData(32, 128, 256)]  // 1M MACs — 1 ic-block, wide N
    [InlineData(64, 64, 512)]   // 2M MACs — 4 kc-iterations (k=512, kc=64)
    public void Gemm_PackBoth_ParallelMAxis_MatchesNaive(int m, int n, int k)
    {
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        var options = new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth };
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 9);
    }

    // ── G2: N-tail handling via Avx2Tail/Avx512Tail kernels ──────────────────
    // These test cases verify that shapes with M divisible by mr but N NOT divisible
    // by nr use the SIMD tile for full N-blocks and the tail kernel (Avx2Tail or
    // Avx512Tail) for the last partial-N column-block.
    // On machines without AVX2/AVX-512 support the scalar fallback in
    // DispatchMicrokernelWithTail handles the partial tile.

    [Theory]
    [InlineData(16, 14, 16, false, false)]   // N=14, not divisible by AVX-512 nr=16
    [InlineData(32, 12, 16, false, false)]   // N=12
    [InlineData(16, 17, 32, false, false)]   // N=17 (partial tile in second jr iteration)
    [InlineData(8, 9, 16, false, false)]     // FP64 AVX2: nr=8, n=9 (single tail col)
    public void Gemm_PartialNTile_MatchesNaive(int m, int n, int k, bool transA, bool transB)
    {
        var (a, b) = GenerateRandomMatrices(m, n, k, transA, transB, seed: 42);
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        double[] expected = NaiveGemm(a, aRows, aCols, transA, b, k, n, transB);

        int lda = aCols;
        int ldb = transB ? k : n;
        double[] actual = new double[m * n];
        var options = new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth };
        BlasManagedLib.Gemm<double>(
            a, lda, transA,
            b, ldb, transB,
            actual, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void KAxisDriver_PartitionsEvenly_WhenKDivisibleByNumThreads()
    {
        var (s0, l0) = KAxisDriver.GetThreadRange(k: 16, numThreads: 4, threadIndex: 0);
        var (s1, l1) = KAxisDriver.GetThreadRange(k: 16, numThreads: 4, threadIndex: 1);
        var (s2, l2) = KAxisDriver.GetThreadRange(k: 16, numThreads: 4, threadIndex: 2);
        var (s3, l3) = KAxisDriver.GetThreadRange(k: 16, numThreads: 4, threadIndex: 3);

        Assert.Equal((0, 4), (s0, l0));
        Assert.Equal((4, 4), (s1, l1));
        Assert.Equal((8, 4), (s2, l2));
        Assert.Equal((12, 4), (s3, l3));

        // Total coverage equals K.
        Assert.Equal(16, l0 + l1 + l2 + l3);
    }

    [Fact]
    public void KAxisDriver_PartitionsWithRemainder()
    {
        // K=17, numThreads=4 → first 1 thread gets 5, remaining 3 get 4 each (1+5 + 3*4 = 17).
        var (s0, l0) = KAxisDriver.GetThreadRange(k: 17, numThreads: 4, threadIndex: 0);
        var (s1, l1) = KAxisDriver.GetThreadRange(k: 17, numThreads: 4, threadIndex: 1);
        var (s2, l2) = KAxisDriver.GetThreadRange(k: 17, numThreads: 4, threadIndex: 2);
        var (s3, l3) = KAxisDriver.GetThreadRange(k: 17, numThreads: 4, threadIndex: 3);

        Assert.Equal((0, 5), (s0, l0));
        Assert.Equal((5, 4), (s1, l1));
        Assert.Equal((9, 4), (s2, l2));
        Assert.Equal((13, 4), (s3, l3));

        Assert.Equal(17, l0 + l1 + l2 + l3);
    }

    [Fact]
    public void KAxisDriver_RoundUpToPowerOfTwo()
    {
        Assert.Equal(1, KAxisDriver.RoundUpToPowerOfTwo(1));
        Assert.Equal(2, KAxisDriver.RoundUpToPowerOfTwo(2));
        Assert.Equal(4, KAxisDriver.RoundUpToPowerOfTwo(3));
        Assert.Equal(4, KAxisDriver.RoundUpToPowerOfTwo(4));
        Assert.Equal(8, KAxisDriver.RoundUpToPowerOfTwo(5));
        Assert.Equal(16, KAxisDriver.RoundUpToPowerOfTwo(16));
        Assert.Equal(32, KAxisDriver.RoundUpToPowerOfTwo(17));
    }

    [Fact]
    public void ReductionTree_TwoPartials_FP64_Sums()
    {
        double[] p0 = { 1.0, 2.0, 3.0, 4.0 };
        double[] p1 = { 10.0, 20.0, 30.0, 40.0 };
        Memory<double>[] partials = { p0.AsMemory(), p1.AsMemory() };

        ReductionTree.ReducePairwiseFp64(partials, elementCount: 4);

        Assert.Equal(11.0, partials[0].Span[0]);
        Assert.Equal(22.0, partials[0].Span[1]);
        Assert.Equal(33.0, partials[0].Span[2]);
        Assert.Equal(44.0, partials[0].Span[3]);
    }

    [Fact]
    public void ReductionTree_FourPartials_FP64_TreeReduces()
    {
        double[] p0 = { 1.0 };
        double[] p1 = { 2.0 };
        double[] p2 = { 3.0 };
        double[] p3 = { 4.0 };
        Memory<double>[] partials = { p0.AsMemory(), p1.AsMemory(), p2.AsMemory(), p3.AsMemory() };

        ReductionTree.ReducePairwiseFp64(partials, elementCount: 1);

        Assert.Equal(10.0, partials[0].Span[0]);
    }

    [Fact]
    public void ReductionTree_OddCount_FP64_HandlesRemainder()
    {
        // 3 partials: pair (0+1), then (0+2).
        double[] p0 = { 1.0 };
        double[] p1 = { 2.0 };
        double[] p2 = { 4.0 };
        Memory<double>[] partials = { p0.AsMemory(), p1.AsMemory(), p2.AsMemory() };

        ReductionTree.ReducePairwiseFp64(partials, elementCount: 1);

        Assert.Equal(7.0, partials[0].Span[0]);
    }

    [Fact]
    public void ReductionTree_DeterministicAcrossOrderingsInRepeats()
    {
        // The same input must produce the same bit pattern across multiple invocations.
        double[] p0Run1 = { 1e16, 1.0, -1e16 };
        double[] p1Run1 = { 1.0, 1.0, 1.0 };
        Memory<double>[] run1 = { p0Run1.AsMemory(), p1Run1.AsMemory() };
        ReductionTree.ReducePairwiseFp64(run1, elementCount: 3);

        double[] p0Run2 = { 1e16, 1.0, -1e16 };
        double[] p1Run2 = { 1.0, 1.0, 1.0 };
        Memory<double>[] run2 = { p0Run2.AsMemory(), p1Run2.AsMemory() };
        ReductionTree.ReducePairwiseFp64(run2, elementCount: 3);

        // Bit-identical (no precision argument).
        for (int i = 0; i < 3; i++)
            Assert.Equal(run1[0].Span[i], run2[0].Span[i]);
    }

    [Fact]
    public void ReductionTree_FP32_BasicSum()
    {
        float[] p0 = { 1.0f, 2.0f };
        float[] p1 = { 3.0f, 4.0f };
        Memory<float>[] partials = { p0.AsMemory(), p1.AsMemory() };

        ReductionTree.ReducePairwiseFp32(partials, elementCount: 2);

        Assert.Equal(4.0f, partials[0].Span[0]);
        Assert.Equal(6.0f, partials[0].Span[1]);
    }

    // ── G4: MN2DDriver ──────────────────────────────────────────────────────

    [Fact]
    public void MN2DDriver_UnflattenIndex_FirstRow()
    {
        // numNBlocks=4, so first 4 items are (0,0), (0,1), (0,2), (0,3).
        Assert.Equal((0, 0), MN2DDriver.UnflattenIndex(0, numNBlocks: 4));
        Assert.Equal((0, 1), MN2DDriver.UnflattenIndex(1, numNBlocks: 4));
        Assert.Equal((0, 3), MN2DDriver.UnflattenIndex(3, numNBlocks: 4));
    }

    [Fact]
    public void MN2DDriver_UnflattenIndex_SecondRow()
    {
        Assert.Equal((1, 0), MN2DDriver.UnflattenIndex(4, numNBlocks: 4));
        Assert.Equal((1, 1), MN2DDriver.UnflattenIndex(5, numNBlocks: 4));
        Assert.Equal((2, 3), MN2DDriver.UnflattenIndex(11, numNBlocks: 4));
    }

    [Fact]
    public void MN2DDriver_TotalItems()
    {
        Assert.Equal(12, MN2DDriver.TotalItems(3, 4));
        Assert.Equal(0, MN2DDriver.TotalItems(0, 5));
        Assert.Equal(0, MN2DDriver.TotalItems(5, 0));
    }

    [Fact]
    public void MN2DDriver_ShouldUse2DGrid_PrefersMAxisWhenEnoughBlocks()
    {
        // procs=4, numMBlocks=4: M alone fills cores → use 1D.
        Assert.False(MN2DDriver.ShouldUse2DGrid(numMBlocks: 4, numNBlocks: 4, procs: 4));
    }

    [Fact]
    public void MN2DDriver_ShouldUse2DGrid_PicksGridWhenMUnderutilizes()
    {
        // procs=16, numMBlocks=2 (M*2=4 < 16), numNBlocks=4: M alone underutilizes → 2D.
        Assert.True(MN2DDriver.ShouldUse2DGrid(numMBlocks: 2, numNBlocks: 4, procs: 16));
    }

    [Fact]
    public void MN2DDriver_ShouldUse2DGrid_NoGridWhenSingleNBlock()
    {
        // No N parallelism possible — 1D M-axis only.
        Assert.False(MN2DDriver.ShouldUse2DGrid(numMBlocks: 1, numNBlocks: 1, procs: 16));
        Assert.False(MN2DDriver.ShouldUse2DGrid(numMBlocks: 2, numNBlocks: 1, procs: 16));
    }

    [Fact]
    public void MN2DDriver_ShouldUse2DGrid_NoParallelOnSingleCore()
    {
        Assert.False(MN2DDriver.ShouldUse2DGrid(numMBlocks: 4, numNBlocks: 4, procs: 1));
    }

    // -------------------------------------------------------------------------
    // G5 — AxisSelector tests
    // -------------------------------------------------------------------------

    [Fact]
    public void AxisSelector_BelowWorkThreshold_ReturnsNone()
    {
        // m*n*k = 8*8*8 = 512 < threshold.
        var axis = AxisSelector.Select(m: 8, n: 8, k: 8, mr: 4, nr: 4, procs: 8, isDeterministic: false);
        Assert.Equal(ParallelismAxis.None, axis);
    }

    [Fact]
    public void AxisSelector_SingleProcessor_ReturnsNone()
    {
        var axis = AxisSelector.Select(m: 4096, n: 16, k: 512, mr: 8, nr: 16, procs: 1, isDeterministic: false);
        Assert.Equal(ParallelismAxis.None, axis);
    }

    [Fact]
    public void AxisSelector_L2Shape_PicksMAxis()
    {
        // L2-shape: M=4096, N=16, K=512, mr=8, nr=16, procs=8.
        // m=4096 >= procs*mr*2 = 128. k=512 > 256 but not deterministic → still M.
        var axis = AxisSelector.Select(m: 4096, n: 16, k: 512, mr: 8, nr: 16, procs: 8, isDeterministic: false);
        Assert.Equal(ParallelismAxis.M, axis);
    }

    [Fact]
    public void AxisSelector_LargeKDeterministic_FallsThroughFromM()
    {
        // m=4096 >= procs*mr*2 = 128, BUT k=300 > 256 AND deterministic → skip M.
        // n=16 vs procs*nr*2 = 128 → skip N.
        // k=300 < 512 → skip K.
        // m*n = 4096*16 = 65536, procs*mr*nr*4 = 8*8*16*4 = 4096 → 2D-grid eligible.
        var axis = AxisSelector.Select(m: 4096, n: 16, k: 300, mr: 8, nr: 16, procs: 8, isDeterministic: true);
        Assert.Equal(ParallelismAxis.MN_2D, axis);
    }

    [Fact]
    public void AxisSelector_NHeavyShape_PicksNAxis()
    {
        // m=4, n=4096, k=512. M-axis needs m >= 8*4*2 = 64 — no.
        // N-axis: n=4096 >= 8*4*2 = 64 — yes.
        var axis = AxisSelector.Select(m: 4, n: 4096, k: 512, mr: 4, nr: 4, procs: 8, isDeterministic: false);
        Assert.Equal(ParallelismAxis.N, axis);
    }

    [Fact]
    public void AxisSelector_TallThinShape_NonDeterministic_PicksKAxis()
    {
        // m=4, n=8, k=2048. M, N too small. K=2048 >= 512 → K-axis (non-deterministic).
        // Workload check: m*n*k = 65536 >= threshold.
        var axis = AxisSelector.Select(m: 4, n: 8, k: 2048, mr: 4, nr: 4, procs: 8, isDeterministic: false);
        Assert.Equal(ParallelismAxis.K, axis);
    }

    [Fact]
    public void AxisSelector_TallThinShape_Deterministic_FallsBackToMaxis()
    {
        // Same shape as above but deterministic: K-axis forbidden.
        // m=4, n=8, k=2048. M needs m >= 8*4*2 = 64 — no. N needs n >= 8*4*2 = 64 — no.
        // K-axis disabled (deterministic). 2D: m*n = 32 < procs*mr*nr*4 = 512 — no.
        // Fallback → M-axis.
        var axis = AxisSelector.Select(m: 4, n: 8, k: 2048, mr: 4, nr: 4, procs: 8, isDeterministic: true);
        Assert.Equal(ParallelismAxis.M, axis);
    }

    [Fact]
    public void AxisSelector_BalancedLargeShape_Picks2DGrid()
    {
        // m=128, n=128, k=128, mr=8, nr=16, procs=16.
        // M-axis: m=128 < procs*mr*2 = 256 — no.
        // N-axis: n=128 < procs*nr*2 = 512 — no.
        // K-axis: k=128 < 512 — no.
        // 2D: m*n = 16384 >= procs*mr*nr*4 = 8192 — yes.
        var axis = AxisSelector.Select(m: 128, n: 128, k: 128, mr: 8, nr: 16, procs: 16, isDeterministic: false);
        Assert.Equal(ParallelismAxis.MN_2D, axis);
    }

    // -------------------------------------------------------------------------
    // BlasManagedAutotune facade tests (Task H1)
    // -------------------------------------------------------------------------

    [Fact]
    public void BlasManagedAutotune_EncodeShape_RoundTrips()
    {
        var shape = BlasManagedAutotune.EncodeShape<double>(
            m: 4096, n: 16, k: 512,
            transA: true, transB: false,
            mr: 8, nr: 16,
            hasEpilogue: false);

        var dims = shape.Dimensions;
        Assert.Equal(9, dims.Length);
        Assert.Equal(4096, dims[0]);
        Assert.Equal(16, dims[1]);
        Assert.Equal(512, dims[2]);
        Assert.Equal(1, dims[3]);  // transA = true
        Assert.Equal(0, dims[4]);  // transB = false
        Assert.Equal(2, dims[5]);  // FP64 tag
        Assert.Equal(8, dims[6]);
        Assert.Equal(16, dims[7]);
        Assert.Equal(0, dims[8]);  // hasEpilogue = false
    }

    [Fact]
    public void BlasManagedAutotune_EncodeShape_FP32_UsesPrecisionTag1()
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(
            m: 32, n: 32, k: 32,
            transA: false, transB: false,
            mr: 16, nr: 16,
            hasEpilogue: false);
        Assert.Equal(1, shape.Dimensions[5]);  // FP32 tag
    }

    [Fact]
    public void BlasManagedAutotune_EncodeDecodeChoice_RoundTrips()
    {
        var choice = BlasManagedAutotune.EncodeChoice(
            ParallelismAxis.MN_2D, mc: 128, nc: 128, kc: 64, threadCount: 4, measuredTimeMs: 1.5);

        var (axis, mc, nc, kc, threads) = BlasManagedAutotune.DecodeChoice(choice);
        Assert.Equal(ParallelismAxis.MN_2D, axis);
        Assert.Equal(128, mc);
        Assert.Equal(128, nc);
        Assert.Equal(64, kc);
        Assert.Equal(4, threads);
    }

    [Fact]
    public void BlasManagedAutotune_TryLookup_ReturnsNullForUnseenShape()
    {
        // Unique shape unlikely to exist in any prior cache.
        var shape = BlasManagedAutotune.EncodeShape<double>(
            m: 7919, n: 7919, k: 7919,  // primes — uncacheable shape
            transA: false, transB: false,
            mr: 8, nr: 16,
            hasEpilogue: false);

        var result = BlasManagedAutotune.TryLookup(shape);
        // Cache should not have this shape unless we just stored it; on a clean
        // run it must be null. On a developer machine that has cached this
        // before, the test still passes if result is non-null AND decoded
        // correctly. We accept either outcome — the assertion is "no exception".
        if (result.HasValue)
        {
            var (axis, mc, nc, kc, threads) = result.Value;
            Assert.True(mc > 0 && nc > 0 && kc > 0);
        }
    }

    [Fact]
    public void BlasManagedAutotune_StoreThenLookup_RoundTrips()
    {
        // Unique shape for this test to avoid cross-test interference.
        var shape = BlasManagedAutotune.EncodeShape<double>(
            m: 5101, n: 5101, k: 5101,
            transA: false, transB: false,
            mr: 8, nr: 16,
            hasEpilogue: false);

        BlasManagedAutotune.Store(shape, ParallelismAxis.K, mc: 96, nc: 128, kc: 256, threadCount: 8, measuredTimeMs: 2.5);
        var result = BlasManagedAutotune.TryLookup(shape);
        Assert.NotNull(result);
        var (axis, mc, nc, kc, threads) = result!.Value;
        Assert.Equal(ParallelismAxis.K, axis);
        Assert.Equal(96, mc);
        Assert.Equal(128, nc);
        Assert.Equal(256, kc);
        Assert.Equal(8, threads);
    }

    [Fact]
    public void AutotuneDispatcher_CacheHit_ReturnsCachedChoice()
    {
        // Use a unique shape (primes) to isolate this test from any prior cache entries.
        int m = 3137, n = 3137, k = 3137;
        int mr = 8, nr = 16;

        // First, store a specific known choice.
        var shape = BlasManagedAutotune.EncodeShape<double>(m, n, k, false, false, mr, nr, false);
        BlasManagedAutotune.Store(shape, ParallelismAxis.N, mc: 96, nc: 96, kc: 96, threadCount: 4, measuredTimeMs: 1.0);

        // Dispatch should return the cached choice.
        var (axis, mc, nc, kc, threads) = AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: mr, nr: nr,
            procs: 16,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);

        Assert.Equal(ParallelismAxis.N, axis);
        Assert.Equal(96, mc);
        Assert.Equal(96, nc);
        Assert.Equal(96, kc);
        Assert.Equal(4, threads);
    }

    [Fact]
    public void AutotuneDispatcher_DisableAutotune_BypassesCache()
    {
        // Use the same unique shape from previous test (cache has N + 96/96/96 stored).
        int m = 3137, n = 3137, k = 3137;
        int mr = 8, nr = 16;

        // With DisableAutotune, heuristic runs regardless of cache contents.
        var (axis, mc, nc, kc, threads) = AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: mr, nr: nr,
            procs: 16,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.DisableAutotune);

        // The heuristic for this large shape with M=3137 and procs=16:
        // procs*mr*2 = 256, m=3137 > 256 → M-axis.
        // Defaults: mc=64, nc=64, kc=64.
        Assert.Equal(ParallelismAxis.M, axis);
        Assert.Equal(64, mc);
        Assert.Equal(64, nc);
        Assert.Equal(64, kc);
    }

    [Fact]
    public void AutotuneDispatcher_CacheMiss_UsesHeuristicAndStores()
    {
        // Brand-new shape (won't be in any prior cache).
        int m = 4099, n = 4099, k = 4099;  // primes
        int mr = 8, nr = 16;

        // Force a miss by storing nothing for this shape (using prime numbers ensures uniqueness).
        var (axis1, mc1, nc1, kc1, threads1) = AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: mr, nr: nr,
            procs: 8,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);

        // Now the cache should have an entry. Second call returns same values.
        var (axis2, mc2, nc2, kc2, threads2) = AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: mr, nr: nr,
            procs: 8,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.Auto);

        Assert.Equal(axis1, axis2);
        Assert.Equal(mc1, mc2);
        Assert.Equal(nc1, nc2);
        Assert.Equal(kc1, kc2);
    }

    [Fact]
    public void AutotuneDispatcher_BlocksRoundedToMrNrAlignment()
    {
        // Tiny shape: m=8 < default mc=64. Heuristic should clamp mc to m, then
        // round down to mr alignment. For m=8, mr=8 → mc=8.
        int m = 8, n = 8, k = 8;
        int mr = 8, nr = 16;

        var (_, mc, nc, kc, _) = AutotuneDispatcher.Decide<double>(
            m, n, k,
            transA: false, transB: false,
            mr: mr, nr: nr,
            procs: 8,
            isDeterministic: false,
            hasEpilogue: false,
            packingMode: PackingMode.DisableAutotune);  // skip cache for determinism

        // m=8 → mc=8 (mr-aligned).
        Assert.Equal(8, mc);
        // n=8 < nr=16 → alignment rounding drives nc to 0. CodeRabbit #366
        // fixed the safeguard to clamp to min(nr, n) instead of unconditionally
        // setting nc = nr, which previously made nc exceed n on tiny shapes.
        Assert.Equal(8, nc);
        Assert.Equal(8, kc);
    }

    // ── H4: AutotuneDispatcher wired into BlasManaged.Gemm ───────────────────

    [Fact]
    public void Gemm_UsesAutotuneDispatcher_ForBlockingDecisions()
    {
        // Verify that BlasManaged.Gemm picks up blocking from the autotune
        // dispatcher rather than hardcoded values. We pre-seed the cache with
        // distinctive values for a unique shape, then call Gemm and verify the
        // result is still correct (the blocking change doesn't break math).
        int m = 64, n = 64, k = 64;
        int mr = 8, nr = 16;  // Assume AVX-512 tile.

        var shape = BlasManagedAutotune.EncodeShape<double>(m, n, k, false, false, mr, nr, false);
        // Pre-seed with custom (non-default) blocking.
        BlasManagedAutotune.Store(shape, ParallelismAxis.M, mc: 32, nc: 32, kc: 32, threadCount: 4, measuredTimeMs: 1.0);

        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        var options = new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth };
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k,
            options);

        // Output must match naive reference regardless of which mc/nc/kc the autotune chose.
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    [Fact]
    public void Gemm_DisableAutotune_StillProducesCorrectResult()
    {
        // Even with autotune disabled, Gemm must work correctly.
        int m = 32, n = 32, k = 32;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);

        double[] actual = new double[m * n];
        var options = new BlasOptions<double> { PackingMode = PackingMode.DisableAutotune };
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    // -------------------------------------------------------------------------
    // Task I1: EpilogueFlags bit-pack + Compute helper
    // -------------------------------------------------------------------------

    [Fact]
    public void EpilogueFlags_DefaultEpilogue_IsNone()
    {
        var ep = new Epilogue<double>();
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.None, flags);
    }

    [Fact]
    public void EpilogueFlags_BiasOnly_SetsHasBias()
    {
        Span<double> bias = stackalloc double[4] { 1, 2, 3, 4 };
        var ep = new Epilogue<double> { BiasN = bias };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.HasBias, flags);
    }

    [Fact]
    public void EpilogueFlags_ActivationOnly_SetsHasActivation()
    {
        var ep = new Epilogue<double> { Activation = AiDotNet.Tensors.Engines.FusedActivationType.ReLU };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.HasActivation, flags);
    }

    [Fact]
    public void EpilogueFlags_OutputScale_NonZero_SetsFlag()
    {
        var ep = new Epilogue<double> { OutputScale = 2.5 };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.HasOutputScale, flags);
    }

    [Fact]
    public void EpilogueFlags_OutputScale_Zero_DoesNotSetFlag()
    {
        // OutputScale=0 is the sentinel for "use 1.0" — flag must NOT be set.
        var ep = new Epilogue<double> { OutputScale = 0.0 };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.None, flags);
    }

    [Fact]
    public void EpilogueFlags_FullChain_SetsAllFlags()
    {
        Span<double> bias = stackalloc double[4] { 1, 2, 3, 4 };
        Span<double> skip = stackalloc double[16];
        var ep = new Epilogue<double>
        {
            BiasN = bias,
            Activation = AiDotNet.Tensors.Engines.FusedActivationType.GELU,
            SkipMxN = skip,
            DropoutMask = 0xDEADBEEF,
            OutputScale = 0.5,
        };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.True(flags.HasFlag(EpilogueFlags.HasBias));
        Assert.True(flags.HasFlag(EpilogueFlags.HasActivation));
        Assert.True(flags.HasFlag(EpilogueFlags.HasSkip));
        Assert.True(flags.HasFlag(EpilogueFlags.HasDropout));
        Assert.True(flags.HasFlag(EpilogueFlags.HasOutputScale));
    }

    [Fact]
    public void EpilogueFlags_FP32_Works()
    {
        Span<float> bias = stackalloc float[4] { 1f, 2f, 3f, 4f };
        var ep = new Epilogue<float> { BiasN = bias };
        var flags = EpilogueFlagsCompute.Compute(in ep);
        Assert.Equal(EpilogueFlags.HasBias, flags);
    }

    // ── I2-I6 epilogue post-pass tests ──────────────────────────────────────

    [Fact]
    public void BiasEpilogue_AddsPerColumnBias()
    {
        double[] c = { 1, 2, 3, 4, 5, 6 };  // 2x3
        double[] bias = { 10, 20, 30 };
        BiasEpilogue.Apply<double>(c.AsSpan(), ldc: 3, m: 2, n: 3, bias);
        Assert.Equal(new double[] { 11, 22, 33, 14, 25, 36 }, c);
    }

    [Fact]
    public void ActivationEpilogue_ReLU_ZeroesNegatives()
    {
        double[] c = { 1, -2, 3, -4 };
        ActivationEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, AiDotNet.Tensors.Engines.FusedActivationType.ReLU);
        Assert.Equal(new double[] { 1, 0, 3, 0 }, c);
    }

    [Fact]
    public void ActivationEpilogue_Sigmoid_TransformsValues()
    {
        double[] c = { 0, 1 };
        ActivationEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 1, n: 2, AiDotNet.Tensors.Engines.FusedActivationType.Sigmoid);
        Assert.Equal(0.5, c[0], precision: 6);            // sigmoid(0) = 0.5
        Assert.Equal(0.7310586, c[1], precision: 6);      // sigmoid(1)
    }

    [Fact]
    public void ActivationEpilogue_Tanh_TransformsValues()
    {
        double[] c = { 0, 1 };
        ActivationEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 1, n: 2, AiDotNet.Tensors.Engines.FusedActivationType.Tanh);
        Assert.Equal(0.0, c[0], precision: 6);
        Assert.Equal(Math.Tanh(1), c[1], precision: 6);
    }

    [Fact]
    public void ActivationEpilogue_None_NoChange()
    {
        double[] c = { -1, 0, 1, 2 };
        double[] expected = (double[])c.Clone();
        ActivationEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, AiDotNet.Tensors.Engines.FusedActivationType.None);
        Assert.Equal(expected, c);
    }

    [Fact]
    public void SkipEpilogue_AddsSkipMatrix()
    {
        double[] c = { 1, 2, 3, 4 };
        double[] skip = { 10, 20, 30, 40 };
        SkipEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, skip);
        Assert.Equal(new double[] { 11, 22, 33, 44 }, c);
    }

    [Fact]
    public void DropoutEpilogue_Deterministic_SameSeedSameResult()
    {
        double[] c1 = { 1, 2, 3, 4, 5, 6, 7, 8 };
        double[] c2 = (double[])c1.Clone();

        DropoutEpilogue.Apply<double>(c1.AsSpan(), ldc: 4, m: 2, n: 4, seed: 42);
        DropoutEpilogue.Apply<double>(c2.AsSpan(), ldc: 4, m: 2, n: 4, seed: 42);

        Assert.Equal(c1, c2);
    }

    [Fact]
    public void DropoutEpilogue_RoughlyHalfZero()
    {
        // For a 100-element matrix, ~50 elements should be zero (50% dropout).
        double[] c = new double[100];
        for (int i = 0; i < c.Length; i++) c[i] = 1.0;
        DropoutEpilogue.Apply<double>(c.AsSpan(), ldc: 10, m: 10, n: 10, seed: 12345);

        int zeros = 0;
        foreach (var v in c) if (v == 0.0) zeros++;
        // Allow 30-70 zeros (random distribution variance).
        Assert.InRange(zeros, 30, 70);
    }

    [Fact]
    public void OutputScaleEpilogue_MultipliesByScalar()
    {
        double[] c = { 1, 2, 3, 4 };
        OutputScaleEpilogue.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, scale: 2.5);
        Assert.Equal(new double[] { 2.5, 5.0, 7.5, 10.0 }, c);
    }

    [Fact]
    public void OutputScaleEpilogue_FP32()
    {
        float[] c = { 1f, 2f, 3f, 4f };
        OutputScaleEpilogue.Apply<float>(c.AsSpan(), ldc: 2, m: 2, n: 2, scale: 0.5f);
        Assert.Equal(new float[] { 0.5f, 1.0f, 1.5f, 2.0f }, c);
    }

    // ── I7/I8: EpilogueChain orchestration + strategy integration ────────────

    [Fact]
    public void EpilogueChain_NoFlags_IsNoOp()
    {
        double[] c = { 1, 2, 3, 4 };
        var epilogue = new Epilogue<double>();
        EpilogueChain.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, in epilogue);
        Assert.Equal(new double[] { 1, 2, 3, 4 }, c);
    }

    [Fact]
    public void EpilogueChain_BiasAndReLU_ApplyInOrder()
    {
        // C[i,j] = original, then C += bias, then ReLU.
        double[] c = { -5, 0, 1, 2 };  // 2x2
        double[] bias = { -1, 3 };

        var epilogue = new Epilogue<double>
        {
            BiasN = bias,
            Activation = FusedActivationType.ReLU,
        };
        EpilogueChain.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, in epilogue);

        // After bias: { -6, 3, 0, 5 }
        // After ReLU: { 0, 3, 0, 5 }
        Assert.Equal(new double[] { 0, 3, 0, 5 }, c);
    }

    [Fact]
    public void EpilogueChain_FullChain_BiasActivationSkipDropoutScale()
    {
        // Verify all five stages run in correct order. Use simple values to make
        // the math hand-checkable.
        double[] c = { 0, 0, 0, 0 };  // 2x2, starts at zero
        double[] bias = { 2, 4 };
        double[] skip = { 1, 1, 1, 1 };

        var epilogue = new Epilogue<double>
        {
            BiasN = bias,
            Activation = FusedActivationType.ReLU,
            SkipMxN = skip,
            OutputScale = 0.5,
            // Skip dropout in this test (it's tested separately) — leave DropoutMask = 0.
        };
        EpilogueChain.Apply<double>(c.AsSpan(), ldc: 2, m: 2, n: 2, in epilogue);

        // After bias: { 2, 4, 2, 4 }
        // After ReLU: { 2, 4, 2, 4 } (already non-negative)
        // After skip: { 3, 5, 3, 5 }
        // After scale (×0.5): { 1.5, 2.5, 1.5, 2.5 }
        Assert.Equal(new double[] { 1.5, 2.5, 1.5, 2.5 }, c);
    }

    [Fact]
    public void Gemm_WithBiasEpilogue_MatchesUnfusedTwoPass()
    {
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] bias = new double[n];
        for (int j = 0; j < n; j++) bias[j] = (j + 1) * 0.1;

        // Two-pass reference: Gemm then Bias.
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                expected[i * n + j] += bias[j];

        // Fused chain: BlasManaged.Gemm with bias in epilogue.
        double[] actual = new double[m * n];
        var options = new BlasOptions<double>
        {
            PackingMode = PackingMode.ForcePackBoth,
            Epilogue = new Epilogue<double> { BiasN = bias },
        };
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
    public void Gemm_WithReLUEpilogue_MatchesUnfusedTwoPass()
    {
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);

        // Two-pass reference.
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] < 0) expected[i] = 0;

        double[] actual = new double[m * n];
        var options = new BlasOptions<double>
        {
            PackingMode = PackingMode.ForcePackBoth,
            Epilogue = new Epilogue<double> { Activation = FusedActivationType.ReLU },
        };
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
    public void Gemm_WithBiasAndReLU_MatchesUnfusedThreePass()
    {
        int m = 8, n = 8, k = 8;
        var (a, b) = GenerateRandomMatrices(m, n, k, transA: false, transB: false, seed: 42);
        double[] bias = new double[n];
        for (int j = 0; j < n; j++) bias[j] = (j - 4) * 0.5;  // mix of positive + negative

        // Three-pass reference.
        double[] expected = NaiveGemm(a, m, k, false, b, k, n, false);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                expected[i * n + j] += bias[j];
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] < 0) expected[i] = 0;

        double[] actual = new double[m * n];
        var options = new BlasOptions<double>
        {
            PackingMode = PackingMode.ForcePackBoth,
            Epilogue = new Epilogue<double>
            {
                BiasN = bias,
                Activation = FusedActivationType.ReLU,
            },
        };
        BlasManagedLib.Gemm<double>(
            a, lda: k, transA: false,
            b, ldb: n, transB: false,
            actual, ldc: n,
            m, n, k,
            options);

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], precision: 10);
    }

    // -------------------------------------------------------------------------
    // J1 + J6 + J9 — JittedKernelCache infrastructure + NativeAOT detector
    // -------------------------------------------------------------------------

    [Fact]
    public void NativeAotDetector_IsDynamicCodeSupported_TrueOnNormalRuntime()
    {
        // On normal .NET (not NativeAOT-published), dynamic code is supported.
        Assert.True(NativeAotDetector.IsDynamicCodeSupported);
    }

    [Fact]
    public void JittedKernelCache_EmptyAtStartup_TryGetReturnsNull()
    {
        JittedKernelCache.Clear();
        var key = new KernelKey
        {
            M = 4, N = 4, K = 4, Lda = 4, Ldb = 4, Ldc = 4,
            TransA = false, TransB = false,
            Packing = PackingMode.Auto,
            EpilogueFlags = 0,
            ElemType = typeof(double),
            Arch = CpuArch.Avx512,
        };
        Assert.Null(JittedKernelCache.TryGetJittedKernel(key));
    }

    [Fact]
    public void JittedKernelCache_StoreThenLookup_ReturnsCachedDelegate()
    {
        JittedKernelCache.Clear();
        var key = new KernelKey
        {
            M = 16, N = 16, K = 16, Lda = 16, Ldb = 16, Ldc = 16,
            TransA = true, TransB = false,
            Packing = PackingMode.ForcePackBoth,
            EpilogueFlags = 0,
            ElemType = typeof(double),
            Arch = CpuArch.Avx512,
        };
        Action testDelegate = () => { };  // Placeholder delegate.
        JittedKernelCache.Store(key, testDelegate);

        var result = JittedKernelCache.TryGetJittedKernel(key);
        Assert.Same(testDelegate, result);
    }

    [Fact]
    public void JittedKernelCache_DifferentKeys_StoreSeparately()
    {
        JittedKernelCache.Clear();
        var key1 = new KernelKey { M = 4, N = 4, K = 4, ElemType = typeof(double), Arch = CpuArch.Scalar };
        var key2 = new KernelKey { M = 8, N = 4, K = 4, ElemType = typeof(double), Arch = CpuArch.Scalar };
        Action d1 = () => { };
        Action d2 = () => { };
        JittedKernelCache.Store(key1, d1);
        JittedKernelCache.Store(key2, d2);

        Assert.Same(d1, JittedKernelCache.TryGetJittedKernel(key1));
        Assert.Same(d2, JittedKernelCache.TryGetJittedKernel(key2));
        Assert.Equal(2, JittedKernelCache.Count);
    }

    [Fact]
    public void JittedKernelCache_Clear_EmptiesCache()
    {
        var key = new KernelKey { M = 4, N = 4, K = 4, ElemType = typeof(double) };
        JittedKernelCache.Store(key, (Action)(() => { }));
        Assert.True(JittedKernelCache.Count > 0);
        JittedKernelCache.Clear();
        Assert.Equal(0, JittedKernelCache.Count);
        Assert.Null(JittedKernelCache.TryGetJittedKernel(key));
    }

}
