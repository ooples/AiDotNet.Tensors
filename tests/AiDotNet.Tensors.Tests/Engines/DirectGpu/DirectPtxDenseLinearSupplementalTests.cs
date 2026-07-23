#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

// Supplemental #836 tests retained while integrating the independently-authored
// dense-linear kernel families on the same PR branch.
public partial class DirectPtxWmmaTests
{
    [Fact]
    public void LinkerLogNormalization_IsDeterministicAndPreservesValidBarrierCounts()
    {
        const string input =
            "used -1206059006 barriers\n" +
            "used 0 barriers\n" +
            "used 16 barriers\n" +
            "used 1452736514 barriers\n";
        Assert.Equal(
            "used unavailable barriers\n" +
            "used 0 barriers\n" +
            "used 16 barriers\n" +
            "used unavailable barriers\n",
            DirectPtxCubinArtifactCache.NormalizeLinkerInfoLog(input));
    }

    [Fact]
    public void DenseLinearAutotuner_UsesMedianAndDeterministicNearTiePolicy()
    {
        Assert.Equal(0.010, DirectPtxDenseLinearAutotuner.MedianMilliseconds(
            [0.025f, 0.009f, 0.010f]), precision: 6);
        Assert.Equal(0.015, DirectPtxDenseLinearAutotuner.MedianMilliseconds(
            [0.020f, 0.010f]), precision: 6);

        Assert.Equal(64, DirectPtxDenseLinearAutotuner.SelectWinner(
            n64MedianMilliseconds: 0.01029,
            n32MedianMilliseconds: 0.01000));
        Assert.Equal(32, DirectPtxDenseLinearAutotuner.SelectWinner(
            n64MedianMilliseconds: 0.01031,
            n32MedianMilliseconds: 0.01000));
        Assert.Equal(64, DirectPtxDenseLinearAutotuner.SelectWinner(
            n64MedianMilliseconds: 0.00900,
            n32MedianMilliseconds: 0.01000));
    }

    [Fact]
    public void DenseLinearAutotuner_RejectsInvalidMeasurements()
    {
        Assert.Throws<ArgumentException>(() =>
            DirectPtxDenseLinearAutotuner.MedianMilliseconds([]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DirectPtxDenseLinearAutotuner.MedianMilliseconds([0f]));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DirectPtxDenseLinearAutotuner.SelectWinner(double.NaN, 0.01));
    }

    [Fact]
    public void CubinSourceIdentity_IsCanonicalAcrossNewlineStyles()
    {
        Assert.Equal(3, DirectPtxCubinArtifactCache.PipelineVersion);
        const string lf = ".version 7.1\n.target sm_86\n";
        const string crlf = ".version 7.1\r\n.target sm_86\r\n";
        const string cr = ".version 7.1\r.target sm_86\r";
        Assert.Equal(lf, DirectPtxCubinArtifactCache.CanonicalizePtx(crlf));
        Assert.Equal(lf, DirectPtxCubinArtifactCache.CanonicalizePtx(cr));
        Assert.Equal(
            DirectPtxCubinArtifactCache.ComputePtxSha256(lf),
            DirectPtxCubinArtifactCache.ComputePtxSha256(crlf));
        Assert.Equal(
            DirectPtxCubinArtifactCache.ComputeSourceKey(lf, 8, 6),
            DirectPtxCubinArtifactCache.ComputeSourceKey(crlf, 8, 6));
        Assert.Equal(
            "cd5c1a81a390b30a00e2ebb41f3b4ab79950317c763550ec5553543ee08da009",
            DirectPtxCubinArtifactCache.ComputeSourceKey(lf, 8, 6));
    }

    [Fact]
    public void EmbeddedDenseLinearArtifacts_IncludeHashedCubinsAndLinkerLogs()
    {
        System.Reflection.Assembly assembly = typeof(DirectPtxRuntime).Assembly;
        string[] resources = assembly.GetManifestResourceNames();
        string manifestName = Assert.Single(resources.Where(name =>
            name.EndsWith(
                ".Artifacts.sm86.dense-linear-cubins.tsv",
                StringComparison.Ordinal)));
        using System.IO.Stream manifestStream =
            assembly.GetManifestResourceStream(manifestName)!;
        Assert.NotNull(manifestStream);
        using var reader = new System.IO.StreamReader(manifestStream);
        int rows = 0;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (line.Length == 0 || line[0] == '#' ||
                line.StartsWith("blueprint-id", StringComparison.Ordinal))
                continue;
            string[] columns = line.Split('\t');
            Assert.Equal(6, columns.Length);
            string cubinName = Assert.Single(resources.Where(name =>
                name.EndsWith("." + columns[4], StringComparison.Ordinal)));
            string linkerName = Assert.Single(resources.Where(name =>
                name.EndsWith(
                    "." + columns[2] + ".linker.txt",
                    StringComparison.Ordinal)));
            using System.IO.Stream cubin =
                assembly.GetManifestResourceStream(cubinName)!;
            using System.IO.Stream linker =
                assembly.GetManifestResourceStream(linkerName)!;
            Assert.Equal(columns[3], Hash(cubin));
            Assert.Equal(columns[5], Hash(linker));
            rows++;
        }
        Assert.Equal(16, rows);

        static string Hash(System.IO.Stream stream)
        {
            using System.Security.Cryptography.SHA256 sha =
                System.Security.Cryptography.SHA256.Create();
            return PtxCompat.ToHexString(sha.ComputeHash(stream))
                .ToLowerInvariant();
        }
    }

    [SkippableFact]
    public void BackendGemmTiledInputMajor_IsPublicAndMatchesOracle()
    {
        const int m = 64, k = 256, n = 256;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.FusedLinearExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFusedLinearEnabled,
                "Requires the measured SM86 fused-linear architecture domain.");
            var random = RandomHelper.CreateSeededRandom(20262365);
            float[] aHost = Values(random, m * k, 0.125f);
            float[] bHost = Values(random, k * n, 0.0625f);
            var expected = new float[m * n];
            for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
            {
                double acc = 0;
                for (int kk = 0; kk < k; kk++)
                    acc += aHost[row * k + kk] * (double)bHost[kk * n + col];
                expected[row * n + col] = (float)acc;
            }

            using var a = backend.AllocateBuffer(aHost);
            using var b = backend.AllocateBuffer(bHost);
            using var output = backend.AllocateBuffer(m * n);
            long before = backend.DirectPtxFusedLinearTiledDispatchCount;
            backend.Gemm(a, b, output, m, n, k);
            backend.Synchronize();
            Assert.Equal(before + 1, backend.DirectPtxFusedLinearTiledDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 2e-3f,
                "public direct PTX GEMM input-major");
        }
        finally
        {
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
            DirectPtxFeatureGate.TestOverride = previousGate;
        }
    }

    [Fact]
    public void BatchedVectorAndCrossEntropyEmitters_HavePointerOnlyExactAbis()
    {
        string batchDot = PtxBatchedVectorKernel.EmitPtx(
            8, 6, DirectPtxBatchedVectorOperation.Dot, batch: 8, m: 512);
        Assert.Contains(PtxBatchedVectorKernel.DotEntryPoint, batchDot, StringComparison.Ordinal);
        Assert.Equal(3, Count(batchDot, ".param .u64"));
        Assert.Contains(".maxntid 256, 1, 1", batchDot, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", batchDot, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", batchDot, StringComparison.Ordinal);
        Assert.Equal(2, Count(batchDot, "ld.global.nc.v4.f32"));
        Assert.Equal(4, Count(batchDot, "fma.rn.f32"));
        Assert.Equal(5, Count(batchDot, "shfl.sync.down.b32"));
        Assert.DoesNotContain(".param .u32", batchDot, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", batchDot, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", batchDot, StringComparison.OrdinalIgnoreCase);

        string batchOuter = PtxBatchedVectorKernel.EmitPtx(
            8, 6, DirectPtxBatchedVectorOperation.Outer, batch: 4, m: 32, n: 64);
        Assert.Contains(PtxBatchedVectorKernel.OuterEntryPoint, batchOuter, StringComparison.Ordinal);
        Assert.Equal(3, Count(batchOuter, ".param .u64"));
        Assert.Equal(1, Count(batchOuter, "st.global.f32"));
        Assert.DoesNotContain(".local", batchOuter, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", batchOuter, StringComparison.OrdinalIgnoreCase);

        string strided = PtxStridedDotKernel.EmitPtx(
            8, 6, aSize: 512, bSize: 512, bOffset: 511, bStep: -1);
        Assert.Contains(PtxStridedDotKernel.EntryPoint, strided, StringComparison.Ordinal);
        Assert.Equal(3, Count(strided, ".param .u64"));
        Assert.DoesNotContain(".param .u32", strided, StringComparison.Ordinal);
        Assert.Contains("valid-i=[0,511]", strided, StringComparison.Ordinal);
        Assert.Contains(".shared .align 16 .b8 partial[32]", strided, StringComparison.Ordinal);
        Assert.Equal(1, Count(strided, "bar.sync 0"));
        Assert.Equal(10, Count(strided, "shfl.sync.down.b32"));
        Assert.DoesNotContain(".local", strided, StringComparison.Ordinal);
        Assert.Equal((0, 511), PtxStridedDotKernel.ValidInterval(512, 512, 511, -1));
        Assert.Equal((3, 10), PtxStridedDotKernel.ValidInterval(16, 8, -3, 1));

        string index = PtxFusedLinearCrossEntropyKernel.EmitPtx(
            8, 6, DirectPtxCrossEntropyTarget.Index, rows: 4,
            hiddenDimension: 16, vocabulary: 32);
        Assert.Contains(PtxFusedLinearCrossEntropyKernel.IndexEntryPoint, index, StringComparison.Ordinal);
        Assert.Equal(5, Count(index, ".param .u64"));
        Assert.Contains("atom.global.add.f32", index, StringComparison.Ordinal);
        Assert.Contains(".maxntid 32, 1, 1", index, StringComparison.Ordinal);
        Assert.Contains(".shared .align 16 .b8 scratch[256]", index, StringComparison.Ordinal);
        Assert.DoesNotContain("st.global.f32", index, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", index, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", index, StringComparison.OrdinalIgnoreCase);

        string dense = PtxFusedLinearCrossEntropyKernel.EmitPtx(
            8, 6, DirectPtxCrossEntropyTarget.Dense, rows: 4,
            hiddenDimension: 16, vocabulary: 32);
        Assert.Contains(PtxFusedLinearCrossEntropyKernel.DenseEntryPoint, dense, StringComparison.Ordinal);
        Assert.Equal(5, Count(dense, ".param .u64"));
        Assert.DoesNotContain(".local", dense, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", dense, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DenseVectorEmitters_HavePointerOnlyExactAbis()
    {
        string dot = PtxDenseVectorKernel.EmitPtx(
            8, 6, DirectPtxDenseVectorOperation.Dot, 4096);
        Assert.Contains(PtxDenseVectorKernel.DotEntryPoint, dot, StringComparison.Ordinal);
        Assert.Equal(3, Count(dot, ".param .u64"));
        Assert.Contains(".shared .align 16 .b8 partial[32]", dot, StringComparison.Ordinal);
        Assert.Equal(2, Count(dot, "ld.global.nc.v4.f32"));
        Assert.Equal(4, Count(dot, "fma.rn.f32"));
        Assert.Equal(1, Count(dot, "bar.sync 0")); // publish eight warp partials
        Assert.Equal(10, Count(dot, "shfl.sync.down.b32"));
        Assert.DoesNotContain(".local", dot, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", dot, StringComparison.OrdinalIgnoreCase);

        string outer = PtxDenseVectorKernel.EmitPtx(
            8, 6, DirectPtxDenseVectorOperation.Outer, 64, 128);
        Assert.Contains(PtxDenseVectorKernel.OuterEntryPoint, outer, StringComparison.Ordinal);
        Assert.Equal(3, Count(outer, ".param .u64"));
        Assert.Equal(1, Count(outer, "st.global.v4.f32"));
        Assert.Contains("ld.global.nc.v4.f32", outer, StringComparison.Ordinal);
        Assert.Contains("st.global.v4.f32", outer, StringComparison.Ordinal);
        Assert.DoesNotContain("div.u32", outer, StringComparison.Ordinal);
        Assert.DoesNotContain("rem.u32", outer, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", outer, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", outer, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", outer, StringComparison.OrdinalIgnoreCase);
    }

    [SkippableFact]
    public void DriverOnlyBatchedVectorKernels_MatchOracleAndHaveZeroLocalBytes()
    {
        const int batch = 4, dimension = 512, m = 16, n = 32;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in batched-vector specializations are measured on GA10x/SM86.");
        var random = RandomHelper.CreateSeededRandom(20262369);

        float[] dotLeft = Values(random, batch * dimension, 0.125f);
        float[] dotRight = Values(random, batch * dimension, 0.125f);
        var dotExpected = new float[batch];
        for (int b = 0; b < batch; b++)
        {
            double sum = 0;
            for (int i = 0; i < dimension; i++)
                sum += dotLeft[b * dimension + i] * (double)dotRight[b * dimension + i];
            dotExpected[b] = (float)sum;
        }
        using (var kernel = new PtxBatchedVectorKernel(
            runtime, DirectPtxBatchedVectorOperation.Dot, batch, dimension))
        using (var left = runtime.AllocateBytes((nuint)(dotLeft.Length * sizeof(float))))
        using (var right = runtime.AllocateBytes((nuint)(dotRight.Length * sizeof(float))))
        using (var output = runtime.AllocateBytes((nuint)(batch * sizeof(float))))
        {
            Assert.Equal(128, kernel.BlockThreads);
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            left.Upload<float>(dotLeft);
            right.Upload<float>(dotRight);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[batch];
            output.Download<float>(actual);
            AssertVectorClose(actual, dotExpected, 2e-5f, "batched dot direct PTX");
        }

        float[] outerLeft = Values(random, batch * m, 0.125f);
        float[] outerRight = Values(random, batch * n, 0.125f);
        using (var kernel = new PtxBatchedVectorKernel(
            runtime, DirectPtxBatchedVectorOperation.Outer, batch, m, n))
        using (var left = runtime.AllocateBytes((nuint)(outerLeft.Length * sizeof(float))))
        using (var right = runtime.AllocateBytes((nuint)(outerRight.Length * sizeof(float))))
        using (var output = runtime.AllocateBytes((nuint)(batch * m * n * sizeof(float))))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            left.Upload<float>(outerLeft);
            right.Upload<float>(outerRight);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[batch * m * n];
            output.Download<float>(actual);
            for (int b = 0; b < batch; b++)
            for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
                Assert.InRange(Math.Abs(
                    actual[(b * m + row) * n + col] -
                    outerLeft[b * m + row] * outerRight[b * n + col]), 0, 1e-7);
        }

        const int stridedSize = 512;
        float[] stridedLeft = Values(random, stridedSize, 0.125f);
        float[] stridedRight = Values(random, stridedSize, 0.125f);
        double stridedExpected = 0;
        for (int i = 0; i < stridedSize; i++)
            stridedExpected += stridedLeft[i] * (double)stridedRight[stridedSize - 1 - i];
        using (var kernel = new PtxStridedDotKernel(
            runtime, stridedSize, stridedSize, stridedSize - 1, -1))
        using (var left = runtime.AllocateBytes((nuint)(stridedSize * sizeof(float))))
        using (var right = runtime.AllocateBytes((nuint)(stridedSize * sizeof(float))))
        using (var output = runtime.AllocateBytes(sizeof(float)))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            left.Upload<float>(stridedLeft);
            right.Upload<float>(stridedRight);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[1];
            output.Download<float>(actual);
            Assert.InRange(Math.Abs(actual[0] - stridedExpected), 0, 2e-5);
        }
    }

    [SkippableFact]
    public void DriverOnlyDenseVectorKernels_MatchOracleAndHaveZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in dense-vector specializations are measured on GA10x/SM86.");
        var random = RandomHelper.CreateSeededRandom(20262366);

        const int length = 4096;
        float[] leftHost = Values(random, length, 0.125f);
        float[] rightHost = Values(random, length, 0.125f);
        double expectedDot = 0;
        for (int i = 0; i < length; i++) expectedDot += leftHost[i] * (double)rightHost[i];
        using (var kernel = new PtxDenseVectorKernel(
            runtime, DirectPtxDenseVectorOperation.Dot, length))
        using (var left = runtime.AllocateBytes((nuint)(length * sizeof(float))))
        using (var right = runtime.AllocateBytes((nuint)(length * sizeof(float))))
        using (var output = runtime.AllocateBytes(sizeof(float)))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            left.Upload<float>(leftHost);
            right.Upload<float>(rightHost);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[1];
            output.Download<float>(actual);
            Assert.InRange(Math.Abs(actual[0] - expectedDot), 0, 2e-5);
        }

        const int m = 64, n = 128;
        float[] outerLeft = Values(random, m, 0.125f);
        float[] outerRight = Values(random, n, 0.125f);
        using (var kernel = new PtxDenseVectorKernel(
            runtime, DirectPtxDenseVectorOperation.Outer, m, n))
        using (var left = runtime.AllocateBytes((nuint)(m * sizeof(float))))
        using (var right = runtime.AllocateBytes((nuint)(n * sizeof(float))))
        using (var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float))))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            left.Upload<float>(outerLeft);
            right.Upload<float>(outerRight);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[m * n];
            output.Download<float>(actual);
            for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
                Assert.InRange(
                    Math.Abs(actual[row * n + col] - outerLeft[row] * outerRight[col]),
                    0, 1e-7);
        }
    }

    [SkippableFact]
    public void DriverOnlyFp16GemmVariants_MatchOracleAndHaveZeroLocalBytes()
    {
        const int m = 16, n = 16, k = 32;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in 16-bit GEMM specializations are measured on GA10x/SM86.");
        var random = RandomHelper.CreateSeededRandom(20262371);
        ushort[] leftHost = RandomHalfRange(random, m * k, 0.125f);
        ushort[] rightHost = RandomHalfRange(random, k * n, 0.125f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            float sum = 0;
            for (int inner = 0; inner < k; inner++)
                sum += Half(leftHost[row * k + inner]) * Half(rightHost[inner * n + col]);
            expected[row * n + col] = sum;
        }

        using var left = runtime.AllocateBytes((nuint)(leftHost.Length * sizeof(ushort)));
        using var right = runtime.AllocateBytes((nuint)(rightHost.Length * sizeof(ushort)));
        left.Upload<ushort>(leftHost);
        right.Upload<ushort>(rightHost);

        using (var kernel = new PtxFp16GemmKernel(runtime, m, n, k))
        using (var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float))))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            Assert.Equal(DirectPtxModuleImageKind.EmbeddedCubin, kernel.Audit.ImageKind);
            Assert.NotEmpty(kernel.Audit.CubinSha256);
            Assert.NotEmpty(kernel.Audit.CubinSourceKey);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[expected.Length];
            output.Download<float>(actual);
            AssertVectorClose(actual, expected, 2e-5f, "FP16-input FP32-output GEMM");
        }

        using (var kernel = new PtxFp16GemmKernel(
            runtime, m, n, k,
            outputType: DirectPtxGemmOutputType.Float16))
        using (var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(ushort))))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(left, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(right, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actualBits = new ushort[expected.Length];
            output.Download<ushort>(actualBits);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(Half(actualBits[i]) - (float)(Half)expected[i]), 0, 1e-7);
        }

        ushort[] leftBf16 = BFloat16Values(random, m * k);
        ushort[] rightBf16 = BFloat16Values(random, k * n);
        var expectedBf16 = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int inner = 0; inner < k; inner++)
                expectedBf16[row * n + col] +=
                    BFloat16(leftBf16[row * k + inner]) * BFloat16(rightBf16[inner * n + col]);
        using var leftB = runtime.AllocateBytes((nuint)(leftBf16.Length * sizeof(ushort)));
        using var rightB = runtime.AllocateBytes((nuint)(rightBf16.Length * sizeof(ushort)));
        using var outputB = runtime.AllocateBytes((nuint)(expectedBf16.Length * sizeof(float)));
        leftB.Upload<ushort>(leftBf16);
        rightB.Upload<ushort>(rightBf16);
        using (var kernel = new PtxFp16GemmKernel(
            runtime, m, n, k, inputType: DirectPtx16BitInputType.BFloat16))
        {
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(leftB, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(rightB, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(outputB, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[expectedBf16.Length];
            outputB.Download<float>(actual);
            AssertVectorClose(actual, expectedBf16, 2e-5f, "BF16-input FP32-output GEMM");
        }
    }

    [SkippableTheory]
    [InlineData(1)] // ReLU
    [InlineData(3)] // Sigmoid
    [InlineData(4)] // Tanh
    [InlineData(2)] // GELU-tanh
    [InlineData(5)] // Swish
    public void DriverOnlyFusedLinearBackwardActivations_MatchOracleAndHaveZeroLocalBytes(
        int activationValue)
    {
        const int m = 64, k = 256, n = 256;
        var activation = (DirectPtxLinearActivation)activationValue;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in backward specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLinearBackwardKernel(
            runtime, m, k, n, activation);
        Assert.All(kernel.Audits, audit => Assert.Equal(0, audit.Function.LocalBytesPerThread));

        var random = RandomHelper.CreateSeededRandom(20262368);
        float[] gradOutputHost = Values(random, m * n, 0.125f);
        float[] inputHost = Values(random, m * k, 0.125f);
        float[] weightsHost = Values(random, k * n, 0.0625f);
        float[] preActivationHost = Values(random, m * n, 0.25f);
        float[] savedHost = preActivationHost
            .Select(value => activation switch
            {
                DirectPtxLinearActivation.Sigmoid =>
                    (float)(1.0 / (1.0 + Math.Exp(-value))),
                DirectPtxLinearActivation.Tanh => (float)Math.Tanh(value),
                _ => value
            })
            .ToArray();
        float[] masked = new float[m * n];
        for (int i = 0; i < masked.Length; i++)
        {
            double x = preActivationHost[i];
            double derivative = activation switch
            {
                DirectPtxLinearActivation.Relu => x > 0 ? 1.0 : 0.0,
                DirectPtxLinearActivation.Sigmoid => savedHost[i] * (1.0 - savedHost[i]),
                DirectPtxLinearActivation.Tanh => 1.0 - savedHost[i] * savedHost[i],
                DirectPtxLinearActivation.GeluTanh => GeluTanhDerivative(x),
                DirectPtxLinearActivation.Swish =>
                    Sigmoid(x) + x * Sigmoid(x) * (1.0 - Sigmoid(x)),
                _ => throw new ArgumentOutOfRangeException(nameof(activation))
            };
            masked[i] = (float)(gradOutputHost[i] * derivative);
        }
        var expectedInput = new float[m * k];
        var expectedWeight = new float[k * n];
        var expectedBias = new float[n];
        for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            float g = masked[row * n + col];
            expectedBias[col] += g;
            for (int kk = 0; kk < k; kk++)
            {
                expectedInput[row * k + kk] += g * weightsHost[kk * n + col];
                expectedWeight[kk * n + col] += inputHost[row * k + kk] * g;
            }
        }

        using var gradOutput = runtime.AllocateBytes((nuint)(gradOutputHost.Length * sizeof(float)));
        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(weightsHost.Length * sizeof(float)));
        using var saved = runtime.AllocateBytes((nuint)(savedHost.Length * sizeof(float)));
        using var gradInput = runtime.AllocateBytes((nuint)(expectedInput.Length * sizeof(float)));
        using var gradWeight = runtime.AllocateBytes((nuint)(expectedWeight.Length * sizeof(float)));
        using var gradBias = runtime.AllocateBytes((nuint)(expectedBias.Length * sizeof(float)));
        gradOutput.Upload<float>(gradOutputHost);
        input.Upload<float>(inputHost);
        weights.Upload<float>(weightsHost);
        saved.Upload<float>(savedHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(saved, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(gradInput, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(gradWeight, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradBias, kernel.Blueprint.Tensors[6]));
        runtime.Synchronize();
        var actualInput = new float[expectedInput.Length];
        var actualWeight = new float[expectedWeight.Length];
        var actualBias = new float[expectedBias.Length];
        gradInput.Download<float>(actualInput);
        gradWeight.Download<float>(actualWeight);
        gradBias.Download<float>(actualBias);
        AssertVectorClose(actualInput, expectedInput, 3e-4f, $"fused {activation} backward dInput");
        AssertVectorClose(actualWeight, expectedWeight, 3e-4f, $"fused {activation} backward dWeight");
        AssertVectorClose(actualBias, expectedBias, 3e-4f, $"fused {activation} backward dBias");
    }

    [SkippableFact]
    public void DriverOnlyFusedLinearCrossEntropy_MatchesStableOracleAndHasZeroLocalBytes()
    {
        const int rows = 4, hiddenDimension = 16, vocabulary = 32;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in fused CE specializations are measured on GA10x/SM86.");
        var random = RandomHelper.CreateSeededRandom(20262370);
        float[] hiddenHost = Values(random, rows * hiddenDimension, 0.125f);
        float[] weightHost = Values(random, hiddenDimension * vocabulary, 0.0625f);
        float[] biasHost = Values(random, vocabulary, 0.03125f);
        float[] indexTarget = [1f, 7f, 15f, 31f];
        var denseTarget = new float[rows * vocabulary];
        for (int row = 0; row < rows; row++)
            denseTarget[row * vocabulary + (int)indexTarget[row]] = 1f;

        double expected = 0;
        var logits = new double[vocabulary];
        for (int row = 0; row < rows; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < vocabulary; col++)
            {
                double value = biasHost[col];
                for (int k = 0; k < hiddenDimension; k++)
                    value += hiddenHost[row * hiddenDimension + k] *
                        (double)weightHost[k * vocabulary + col];
                logits[col] = value;
                max = Math.Max(max, value);
            }
            double sum = 0;
            for (int col = 0; col < vocabulary; col++) sum += Math.Exp(logits[col] - max);
            expected += max + Math.Log(sum) - logits[(int)indexTarget[row]];
        }
        expected /= rows;

        using var hidden = runtime.AllocateBytes((nuint)(hiddenHost.Length * sizeof(float)));
        using var weight = runtime.AllocateBytes((nuint)(weightHost.Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var targetIndex = runtime.AllocateBytes((nuint)(indexTarget.Length * sizeof(float)));
        using var targetDense = runtime.AllocateBytes((nuint)(denseTarget.Length * sizeof(float)));
        using var loss = runtime.AllocateBytes(sizeof(float));
        hidden.Upload<float>(hiddenHost);
        weight.Upload<float>(weightHost);
        bias.Upload<float>(biasHost);
        targetIndex.Upload<float>(indexTarget);
        targetDense.Upload<float>(denseTarget);

        foreach (DirectPtxCrossEntropyTarget kind in new[]
        {
            DirectPtxCrossEntropyTarget.Index,
            DirectPtxCrossEntropyTarget.Dense
        })
        {
            using var kernel = new PtxFusedLinearCrossEntropyKernel(
                runtime, kind, rows, hiddenDimension, vocabulary);
            Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
            Assert.Equal(kernel.StaticSharedBytes,
                kernel.Audit.Function.StaticSharedBytes);
            loss.Upload<float>(new float[1]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(hidden, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(weight, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(
                    kind == DirectPtxCrossEntropyTarget.Index ? targetIndex : targetDense,
                    kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(loss, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[1];
            loss.Download<float>(actual);
            Assert.InRange(Math.Abs(actual[0] - expected), 0, 2e-4);
        }
    }

    [SkippableFact]
    public void DriverOnlyFusedLinearTiledInputMajor_MatchesOracleAndHasZeroLocalBytes()
    {
        const int m = 64, k = 256, n = 256;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in tiled fused-linear specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLinearTiledKernel(
            runtime, m, k, n, DirectPtxLinearActivation.GeluTanh,
            DirectPtxLinearWeightLayout.InputMajor);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(DirectPtxPhysicalLayout.LinearWeightInputMajor,
            kernel.Blueprint.Tensors[1].Layout);

        var random = RandomHelper.CreateSeededRandom(20262364);
        float[] inputHost = Values(random, m * k, 0.125f);
        float[] weightsHost = Values(random, k * n, 0.0625f);
        float[] biasHost = Values(random, n, 0.0625f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            double acc = biasHost[col];
            for (int kk = 0; kk < k; kk++)
                acc += inputHost[row * k + kk] * (double)weightsHost[kk * n + col];
            expected[row * n + col] = (float)ApplyTiledActivation(
                acc, DirectPtxLinearActivation.GeluTanh);
        }

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(weightsHost.Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<float>(inputHost);
        weights.Upload<float>(weightsHost);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, "tiled fused linear input-major");
    }

    [SkippableFact]
    public void DriverOnlyFusedLoRA_MatchesOracleAndHasZeroLocalBytes()
    {
        const int batch = 8, inputFeatures = 256, rank = 8, outputFeatures = 256;
        const float scaling = 0.125f;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in fused-LoRA specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLoRAKernel(
            runtime, batch, inputFeatures, rank, outputFeatures, scaling);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(
            rank * sizeof(float) + 256,
            kernel.Audit.Function.StaticSharedBytes);

        var random = RandomHelper.CreateSeededRandom(20262367);
        float[] inputHost = Values(random, batch * inputFeatures, 0.125f);
        float[] baseHost = Values(random, batch * outputFeatures, 0.125f);
        float[] aHost = Values(random, inputFeatures * rank, 0.0625f);
        float[] bHost = Values(random, rank * outputFeatures, 0.0625f);
        var expected = new float[baseHost.Length];
        var projection = new double[rank];
        for (int row = 0; row < batch; row++)
        {
            Array.Clear(projection, 0, projection.Length);
            for (int r = 0; r < rank; r++)
                for (int i = 0; i < inputFeatures; i++)
                    projection[r] += inputHost[row * inputFeatures + i] * (double)aHost[i * rank + r];
            for (int col = 0; col < outputFeatures; col++)
            {
                double delta = 0;
                for (int r = 0; r < rank; r++)
                    delta += projection[r] * bHost[r * outputFeatures + col];
                expected[row * outputFeatures + col] =
                    baseHost[row * outputFeatures + col] + scaling * (float)delta;
            }
        }

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var baseOutput = runtime.AllocateBytes((nuint)(baseHost.Length * sizeof(float)));
        using var loraA = runtime.AllocateBytes((nuint)(aHost.Length * sizeof(float)));
        using var loraB = runtime.AllocateBytes((nuint)(bHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<float>(inputHost);
        baseOutput.Upload<float>(baseHost);
        loraA.Upload<float>(aHost);
        loraB.Upload<float>(bHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(baseOutput, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(loraA, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(loraB, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-4f, "fused LoRA direct PTX");
    }

    [Fact]
    public void Fp16GemmEmitter_BakesDtypeTransposeBatchAndConversionIntoPointerOnlyAbi()
    {
        string fp16 = PtxFp16GemmKernel.EmitPtx(
            8, 6, m: 16, n: 32, k: 64, batch: 4,
            transposeA: true, transposeB: false,
            outputType: DirectPtxGemmOutputType.Float16);
        Assert.Contains(PtxFp16GemmKernel.EntryPoint, fp16, StringComparison.Ordinal);
        Assert.Equal(3, Count(fp16, ".param .u64"));
        Assert.DoesNotContain(".param .u32", fp16, StringComparison.Ordinal);
        Assert.Contains("mov.u32 %r5, %ctaid.z", fp16, StringComparison.Ordinal);
        Assert.Contains("cvt.f32.f16", fp16, StringComparison.Ordinal);
        Assert.Contains("cvt.rn.f16.f32", fp16, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", fp16, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", fp16, StringComparison.OrdinalIgnoreCase);

        string bf16 = PtxFp16GemmKernel.EmitPtx(
            8, 6, m: 16, n: 32, k: 64,
            inputType: DirectPtx16BitInputType.BFloat16);
        Assert.Contains("shl.b32", bf16, StringComparison.Ordinal);
        Assert.Contains("st.global.f32", bf16, StringComparison.Ordinal);
        Assert.DoesNotContain("cvt.f32.f16", bf16, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", bf16, StringComparison.Ordinal);
    }

    [Fact]
    public void ExactFp16GemmEmitter_UsesTwoWarpAsyncTensorCorePipeline()
    {
        string ptx = PtxFp16GemmKernel.EmitPtx(8, 6, m: 16, n: 16, k: 32);
        Assert.Contains(".maxntid 64, 1, 1", ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, "cp.async.ca.shared.global"));
        Assert.Equal(2, Count(ptx, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        Assert.Equal(2, Count(ptx, "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
        Assert.Equal(2, Count(
            ptx, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        Assert.Equal(2, Count(ptx, "st.global.v2.f32"));
        Assert.DoesNotContain("GEMM_K_LOOP", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Fp16TensorCoreM16Emitter_UsesAsyncMmaAndRegisterOnlyEpilogue()
    {
        string k512Ptx = PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
            8, 6, inputFeatures: 512, outputFeatures: 2_048);
        Assert.Contains(
            PtxFusedLinearGeluFp16M16Kernel.EntryPoint, k512Ptx,
            StringComparison.Ordinal);
        Assert.Equal(4, Count(k512Ptx, ".param .u64"));
        Assert.Equal(24,
            Count(k512Ptx, "cp.async.ca.shared.global") +
            Count(k512Ptx, "cp.async.cg.shared.global"));
        Assert.Equal(32, Count(
            k512Ptx, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        Assert.Equal(32, Count(
            k512Ptx, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        Assert.Equal(32, Count(
            k512Ptx, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"));
        Assert.Contains(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1}, [%rd13+6144]",
            k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%b0,%b1}, [%rd13+5120]",
            k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ldmatrix.sync.aligned.m8n8.x2.trans", k512Ptx,
            StringComparison.Ordinal);
        Assert.Contains(
            "cp.async.cg.shared.global [%rd13+3072]",
            k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "cp.async.cg.shared.global [%rd13+2560]",
            k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ld.shared.b32", k512Ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(k512Ptx, "st.global.v2.f32"));
        Assert.DoesNotContain(".param .u32", k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", k512Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", k512Ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(20_480,
            PtxFusedLinearGeluFp16M16Kernel.GetStaticSharedBytes(512));
        string nblock32Ptx = PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
            8, 6, inputFeatures: 512, outputFeatures: 2_048,
            outputsPerBlock: 32);
        Assert.Contains("mul.lo.u32 %r4, %r3, 32", nblock32Ptx,
            StringComparison.Ordinal);
        Assert.Contains(
            "cp.async.cg.shared.global [%rd13+2560]",
            nblock32Ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "cp.async.cg.shared.global [%rd13+2304]",
            nblock32Ptx, StringComparison.Ordinal);
        Assert.Equal(12_288,
            PtxFusedLinearGeluFp16M16Kernel.GetStaticSharedBytes(512, 32));
        Assert.Equal(new[] { 64, 32 },
            DirectPtxDenseLinearAutotuner.Candidates(512, 2_048));

        string k1024Ptx = PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
            8, 6, inputFeatures: 1_024, outputFeatures: 4_096);
        Assert.Equal(48,
            Count(k1024Ptx, "cp.async.ca.shared.global") +
            Count(k1024Ptx, "cp.async.cg.shared.global"));
        Assert.Equal(64, Count(
            k1024Ptx, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        Assert.Equal(64, Count(
            k1024Ptx, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        Assert.Equal(64, Count(
            k1024Ptx, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"));
        Assert.DoesNotContain("ld.shared.b32", k1024Ptx, StringComparison.Ordinal);
        Assert.Equal(40_960,
            PtxFusedLinearGeluFp16M16Kernel.GetStaticSharedBytes(1_024));
        Assert.False(
            PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(1_024, 4_096));
    }

    [SkippableTheory]
    [InlineData(512, 2_048, 32)]
    [InlineData(512, 2_048, 64)]
    [InlineData(1_024, 4_096, 32)]
    [InlineData(1_024, 4_096, 64)]
    public void DriverOnlyFp16TensorCoreM16_MatchesOracleAndLoadsExactCubin(
        int k,
        int n,
        int outputsPerBlock)
    {
        const int m = PtxFusedLinearGeluFp16M16Kernel.Rows;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in Tensor Core specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLinearGeluFp16M16Kernel(
            runtime, k, n, outputsPerBlock);
        Assert.Equal(DirectPtxModuleImageKind.EmbeddedCubin, kernel.Audit.ImageKind);
        Assert.False(string.IsNullOrWhiteSpace(kernel.Audit.CubinSha256));
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20267223);
        ushort[] inputHost = RandomHalfRange(random, m * k, 0.125f);
        ushort[] weightHost = RandomHalfRange(random, n * k, 0.0625f);
        float[] biasHost = Values(random, n, 0.03125f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
        {
            float sum = biasHost[column];
            for (int inner = 0; inner < k; inner++)
                sum += Half(inputHost[row * k + inner]) *
                    Half(weightHost[column * k + inner]);
            expected[row * n + column] = (float)ApplyTiledActivation(
                sum, DirectPtxLinearActivation.GeluTanh);
        }

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(ushort)));
        using var weights = runtime.AllocateBytes((nuint)(weightHost.Length * sizeof(ushort)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<ushort>(inputHost);
        weights.Upload<ushort>(weightHost);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, "FP16 Tensor Core fused linear");
    }

    [SkippableFact]
    public void DriverOnlyFp16TensorCoreM16_PreservesEveryRowAndKFragment()
    {
        const int m = PtxFusedLinearGeluFp16M16Kernel.Rows;
        const int k = 512;
        const int n = 2_048;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in Tensor Core specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLinearGeluFp16M16Kernel(runtime, k, n);

        var inputHost = new ushort[m * k];
        var weightHost = new ushort[n * k];
        var biasHost = new float[n];
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        for (int inner = 0; inner < k; inner++)
        {
            float value = ((row * 17 + inner * 13) % 61 - 30) / 64f;
            inputHost[row * k + inner] = BitConverter.HalfToUInt16Bits((Half)value);
        }
        ushort one = BitConverter.HalfToUInt16Bits((Half)1f);
        for (int column = 0; column < n; column++)
        {
            int selectedInner = column % k;
            weightHost[column * k + selectedInner] = one;
            for (int row = 0; row < m; row++)
            {
                float value = Half(inputHost[row * k + selectedInner]);
                expected[row * n + column] = (float)ApplyTiledActivation(
                    value, DirectPtxLinearActivation.GeluTanh);
            }
        }

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(ushort)));
        using var weights = runtime.AllocateBytes((nuint)(weightHost.Length * sizeof(ushort)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        input.Upload<ushort>(inputHost);
        weights.Upload<ushort>(weightHost);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, "FP16 Tensor Core row/K fragment mapping");

        Array.Clear(inputHost, 0, inputHost.Length);
        Array.Clear(weightHost, 0, weightHost.Length);
        for (int column = 0; column < n; column++)
            biasHost[column] = (column % 61 - 30) / 64f;
        for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++)
            expected[row * n + column] = (float)ApplyTiledActivation(
                biasHost[column], DirectPtxLinearActivation.GeluTanh);
        input.Upload<ushort>(inputHost);
        weights.Upload<ushort>(weightHost);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, "FP16 Tensor Core bias mapping");
    }

    [Fact]
    public void FusedLinearBackwardEmitter_HasSinglePointerOnlyOutputKernel()
    {
        string ptx = PtxFusedLinearBackwardKernel.EmitPtx(
            8, 6, 64, 256, 256, DirectPtxLinearActivation.GeluTanh);
        Assert.Contains(PtxFusedLinearBackwardKernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, ".visible .entry"));
        Assert.Equal(7, Count(ptx, ".param .u64"));
        Assert.Contains("GRAD_INPUT_PATH:", ptx, StringComparison.Ordinal);
        Assert.Contains("@%p5 add.rn.f32 %f11, %f11, %f1", ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Contains(".shared .align 16 .b8 tile_a[1024]", ptx, StringComparison.Ordinal);
        Assert.Contains(".shared .align 16 .b8 tile_b[1024]", ptx, StringComparison.Ordinal);
        Assert.Contains("ld.shared.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("bar.sync 0", ptx, StringComparison.Ordinal);
        Assert.Contains("@!%p5 bra KERNEL_DONE", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("@!%p5 bra.uni", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(64)]
    public void FusedLoRAEmitter_HasPointerOnlyFusedDataflow(int rank)
    {
        string ptx = PtxFusedLoRAKernel.EmitPtx(
            8, 6, batch: 8, inputFeatures: 256, rank,
            outputFeatures: 256, scaling: 0.125f);
        Assert.Contains(PtxFusedLoRAKernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .f32", ptx, StringComparison.Ordinal);
        Assert.Contains(
            $".shared .align 16 .b8 projection[{rank * sizeof(float)}]",
            ptx, StringComparison.Ordinal);
        Assert.Contains(".shared .align 16 .b8 projection_partials[256]", ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, "shfl.sync.down.b32"));
        Assert.Contains("add.u32 %r2, %r2, 32", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    private static ushort[] BFloat16Values(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < result.Length; i++)
        {
            float value = random.Next(-16, 17) / 128f;
            result[i] = (ushort)((uint)BitConverter.SingleToInt32Bits(value) >> 16);
        }
        return result;
    }

    private static float BFloat16(ushort bits) =>
        BitConverter.Int32BitsToSingle(bits << 16);

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double GeluTanhDerivative(double x)
    {
        const double sqrtTwoOverPi = 0.7978845608;
        const double coefficient = 0.044715;
        double x2 = x * x;
        double t = Math.Tanh(sqrtTwoOverPi * (x + coefficient * x * x2));
        return 0.5 * (1.0 + t) +
            0.5 * x * (1.0 - t * t) * sqrtTwoOverPi *
            (1.0 + 3.0 * coefficient * x2);
    }
}
#endif
