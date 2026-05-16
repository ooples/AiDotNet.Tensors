using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ScalarKernelTests
{
    [Fact]
    public void Gemm_StubExistsButNotImplemented_ThrowsNotImplemented()
    {
        // Span<T> and readonly ref struct cannot be captured by lambdas (CS8175),
        // so we call Gemm directly and catch the expected NotImplementedException.
        var threw = false;
        try
        {
            double[] cArr = new double[4];
            var options = new BlasOptions<double>();
            BlasManagedLib.Gemm<double>(
                new ReadOnlySpan<double>(new double[2]), 2, false,
                new ReadOnlySpan<double>(new double[2]), 2, false,
                cArr.AsSpan(), 2, 1, 2, 2, options);
        }
        catch (NotImplementedException)
        {
            threw = true;
        }
        Assert.True(threw, "Expected NotImplementedException from BlasManaged.Gemm stub.");
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
}
