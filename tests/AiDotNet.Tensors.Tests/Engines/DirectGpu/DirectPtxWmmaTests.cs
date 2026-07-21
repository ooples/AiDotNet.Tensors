#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class DirectPtxWmmaTests
{
    [SkippableTheory]
    [InlineData(16, false)]
    [InlineData(16, true)]
    [InlineData(32, false)]
    [InlineData(32, true)]
    [InlineData(64, false)]
    [InlineData(64, true)]
    [InlineData(128, false)]
    [InlineData(128, true)]
    public void DriverOnlyWarpSoftmax_MatchesCpuReference(int sequence, bool isCausal)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");

        const int batchHeads = 3;
        using var runtime = new DirectPtxRuntime();
        using var kernel = new PtxAttentionSoftmax32Kernel(runtime, batchHeads, isCausal, sequence);
        using var scoresDevice = runtime.AllocateBytes(kernel.ScoreBytes);
        using var probabilitiesDevice = runtime.AllocateBytes(kernel.ProbabilityBytes);
        var random = new Random(isCausal ? 20260730 : 20260729);
        var scores = new float[batchHeads * sequence * sequence];
        for (int i = 0; i < scores.Length; i++) scores[i] = (float)(random.NextDouble() * 4 - 2);
        scoresDevice.Upload<float>(scores);

        kernel.Launch(scoresDevice, probabilitiesDevice);
        runtime.Synchronize();
        var actualBits = new ushort[scores.Length];
        probabilitiesDevice.Download<ushort>(actualBits);

        float maxAbsoluteError = 0;
        for (int row = 0; row < batchHeads * sequence; row++)
        {
            int queryRow = row % sequence;
            int lastColumn = isCausal ? queryRow : sequence - 1;
            int offset = row * sequence;
            float maximum = float.NegativeInfinity;
            for (int column = 0; column <= lastColumn; column++)
                maximum = MathF.Max(maximum, scores[offset + column]);
            float sum = 0;
            for (int column = 0; column <= lastColumn; column++)
                sum += MathF.Exp(scores[offset + column] - maximum);
            for (int column = 0; column < sequence; column++)
            {
                float expected = column <= lastColumn
                    ? MathF.Exp(scores[offset + column] - maximum) / sum
                    : 0f;
                float actual = (float)BitConverter.UInt16BitsToHalf(actualBits[offset + column]);
                maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(expected - actual));
            }
        }

        Assert.True(maxAbsoluteError <= 0.001f,
            $"Warp softmax max absolute error {maxAbsoluteError:G9} (causal={isCausal}).");
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DriverOnlyFusedAttention32x16_MatchesCpuReference(bool isCausal)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");

        const int batchHeads = 3, sequence = 32, dimension = 16;
        const float scale = 0.25f;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 7, "Requires Tensor Core compute capability 7.0+.");
        using var kernel = new PtxWmmaFusedAttention32x16Kernel(runtime, batchHeads, isCausal, scale);
        using var qDevice = runtime.AllocateBytes(kernel.QBytes);
        using var kDevice = runtime.AllocateBytes(kernel.KBytes);
        using var vDevice = runtime.AllocateBytes(kernel.VBytes);
        using var outputDevice = runtime.AllocateBytes(kernel.OutputBytes);

        var random = new Random(isCausal ? 20260726 : 20260725);
        ushort[] q = RandomHalf(random, batchHeads * sequence * dimension);
        ushort[] k = RandomHalf(random, batchHeads * sequence * dimension);
        ushort[] v = RandomHalf(random, batchHeads * sequence * dimension);
        var actual = new float[batchHeads * sequence * dimension];
        qDevice.Upload<ushort>(q);
        kDevice.Upload<ushort>(k);
        vDevice.Upload<ushort>(v);
        kernel.Launch(qDevice, kDevice, vDevice, outputDevice);
        runtime.Synchronize();
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int bh = 0; bh < batchHeads; bh++)
        for (int row = 0; row < sequence; row++)
        {
            var scores = new float[sequence];
            float rowMax = float.NegativeInfinity;
            int lastKey = isCausal ? row : sequence - 1;
            for (int col = 0; col <= lastKey; col++)
            {
                float score = 0;
                for (int inner = 0; inner < dimension; inner++)
                {
                    float qValue = (float)BitConverter.UInt16BitsToHalf(
                        q[(bh * sequence + row) * dimension + inner]);
                    float kValue = (float)BitConverter.UInt16BitsToHalf(
                        k[(bh * sequence + col) * dimension + inner]);
                    score += qValue * kValue;
                }
                scores[col] = score * scale;
                rowMax = MathF.Max(rowMax, scores[col]);
            }

            float sum = 0;
            for (int col = 0; col <= lastKey; col++)
            {
                scores[col] = MathF.Exp(scores[col] - rowMax);
                sum += scores[col];
            }
            for (int outputColumn = 0; outputColumn < dimension; outputColumn++)
            {
                float expected = 0;
                for (int col = 0; col <= lastKey; col++)
                {
                    float vValue = (float)BitConverter.UInt16BitsToHalf(
                        v[(bh * sequence + col) * dimension + outputColumn]);
                    expected += (scores[col] / sum) * vValue;
                }
                float got = actual[(bh * sequence + row) * dimension + outputColumn];
                maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
            }
        }

        Assert.True(maxAbsoluteError <= 0.003f,
            $"Fused PTX attention max absolute error {maxAbsoluteError:G9} (causal={isCausal}).");
    }

    [SkippableFact]
    public void DriverOnlyWmmaBatchedQk_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");

        const int batch = 2, m = 32, n = 32, k = 64;
        float scale = 1.0f / MathF.Sqrt(k);
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 7, "Requires Tensor Core compute capability 7.0+.");

        using var kernel = new PtxWmmaBatchedQkKernel(runtime, m, n, k, batch, scale);
        using var qDevice = runtime.AllocateBytes(kernel.ABytes);
        using var kDevice = runtime.AllocateBytes(kernel.BBytes);
        using var scoresDevice = runtime.AllocateBytes(kernel.CBytes);

        var random = new Random(20260720);
        var q = RandomHalf(random, batch * m * k);
        var keys = RandomHalf(random, batch * n * k);
        var actual = new float[batch * m * n];
        qDevice.Upload<ushort>(q);
        kDevice.Upload<ushort>(keys);

        kernel.Launch(qDevice, kDevice, scoresDevice);
        runtime.Synchronize();
        scoresDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        float maxRelativeError = 0;
        for (int b = 0; b < batch; b++)
        for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
        {
            float expected = 0;
            for (int inner = 0; inner < k; inner++)
            {
                float qValue = (float)BitConverter.UInt16BitsToHalf(q[(b * m + row) * k + inner]);
                float kValue = (float)BitConverter.UInt16BitsToHalf(keys[(b * n + col) * k + inner]);
                expected += qValue * kValue;
            }
            expected *= scale;

            float got = actual[(b * m + row) * n + col];
            float absolute = MathF.Abs(got - expected);
            float relative = absolute / MathF.Max(1e-5f, MathF.Abs(expected));
            maxAbsoluteError = MathF.Max(maxAbsoluteError, absolute);
            maxRelativeError = MathF.Max(maxRelativeError, relative);
        }

        Assert.True(maxAbsoluteError <= 0.003f,
            $"PTX WMMA max absolute error {maxAbsoluteError:G9}, max relative {maxRelativeError:G9}");
    }

    [Fact]
    public void PtxEmitter_BakesShapeScaleAndTensorCoreInstructions()
    {
        string ptx = PtxWmmaBatchedQkKernel.EmitPtx(8, 6, 32, 32, 64, 0.125f);
        Assert.Contains(".target sm_86", ptx);
        Assert.Equal(4, Count(ptx, "wmma.mma.sync"));
        Assert.Contains("0f3E000000", ptx);
        Assert.Contains(".shared .align 16", ptx);
        Assert.Contains("ld.global.v4.u32", ptx);
        Assert.DoesNotContain(".param .u32 m", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("nvrtc", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void FusedAttentionEmitter_UsesTwoTensorCoreStagesAndNoScoreOutput()
    {
        string ptx = PtxWmmaFusedAttention32x16Kernel.EmitPtx(8, 6, isCausal: true, scale: 0.25f);
        Assert.Contains(".target sm_86", ptx);
        Assert.Equal(4, Count(ptx, "wmma.mma.sync"));
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Contains("wmma.mma.sync.aligned.row.col", ptx);
        Assert.Contains("wmma.mma.sync.aligned.row.row", ptx);
        Assert.DoesNotContain("scores_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("probabilities_ptr", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void SoftmaxEmitter_UsesWarpReductionAndCompileTimeCausalMask()
    {
        string ptx = PtxAttentionSoftmax32Kernel.EmitPtx(8, 6, isCausal: true);
        Assert.Contains(".target sm_86", ptx);
        Assert.Equal(10, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Contains("setp.gt.u32", ptx);
        Assert.DoesNotContain(".param .u32 rows", ptx, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void OnlineEmitter_UsesAsyncDoubleBufferingOnlineStateAndFusedEpilogue(
        bool isCausal, bool fuseLayerNormGelu)
    {
        string ptx = PtxOnlineFusedAttention128x64Kernel.EmitPtx(
            8, 6, isCausal, fuseLayerNormGelu, 0.125f, 1e-5f);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains("cp.async.ca.shared.global", ptx);
        Assert.Contains("cp.async.commit_group", ptx);
        Assert.Contains("cp.async.wait_group", ptx);
        Assert.Contains("mma.sync.aligned.m16n8k16", ptx);
        Assert.Contains("// ---- online K/V tile 7 ----", ptx);
        Assert.DoesNotContain("scores_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("probabilities_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        if (fuseLayerNormGelu)
        {
            Assert.Contains("Head-local LayerNorm(D=64) + tanh-GELU epilogue", ptx);
            Assert.Contains("tanh.approx.f32", ptx);
        }
        else
        {
            Assert.DoesNotContain("tanh.approx.f32", ptx);
        }
    }

    [Fact]
    public void OnlineInferenceEmitter_OmitsStatsAndSkipsFutureCausalTiles()
    {
        string ptx = PtxOnlineFusedAttention128x64Kernel.EmitPtx(
            8, 6, isCausal: true, fuseLayerNormGelu: false,
            0.125f, 1e-5f, 128, emitSoftmaxStats: false);

        Assert.Contains("SKIP_CAUSAL_TILE_7", ptx);
        Assert.Contains("@%p6 bra SKIP_CAUSAL_TILE_7", ptx);
        Assert.DoesNotContain("lg2.approx.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("st.global.f32 [%rd18]", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlineFamilyEmitter_BakesRectangularGqaMappingWithoutStrideParameters()
    {
        string ptx = PtxOnlineFusedAttention128x64Kernel.EmitFamilyPtx(
            8, 6, queryHeads: 8, keyValueHeads: 2,
            isCausal: false, fuseLayerNormGelu: false,
            scale: 0.125f, epsilon: 1e-5f,
            querySequence: 32, keyValueSequence: 128,
            emitSoftmaxStats: true, warpsPerBlock: 2);

        Assert.Contains("div.u32 %r24, %r2, 8", ptx);
        Assert.Contains("div.u32 %r25, %r25, 4", ptx);
        Assert.Contains("mad.lo.u32 %r26, %r24, 2, %r25", ptx);
        Assert.Contains("// ---- online K/V tile 7 ----", ptx);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OnlineFamilyEmitter_BakesBottomRightCausalOffsetWithoutRuntimeParameters()
    {
        string ptx = PtxOnlineFusedAttention128x64Kernel.EmitFamilyPtx(
            8, 6, queryHeads: 8, keyValueHeads: 2,
            isCausal: true, fuseLayerNormGelu: false,
            scale: 0.125f, epsilon: 1e-5f,
            querySequence: 32, keyValueSequence: 64,
            emitSoftmaxStats: false, warpsPerBlock: 2,
            causalQueryOffset: 32);

        Assert.Contains("add.u32 %r8, %r8, 32", ptx);
        Assert.Contains("add.u32 %r27, %r1, 2", ptx);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void CanonicalPhysicalView_RejectsBadExtentAlignmentAndDtypeExtent()
    {
        using var aligned = new SyntheticGpuBuffer((IntPtr)0x1000, 1024);
        DirectPtxTensorView view = DirectPtxTensorView.CreateBhsd(
            aligned, DirectPtxPhysicalType.Float16, 1024);
        Assert.Equal((IntPtr)0x1000, view.Pointer);

        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.CreateBhsd(
            aligned, DirectPtxPhysicalType.Float16, 2048));
        using var unaligned = new SyntheticGpuBuffer((IntPtr)0x1004, 1024);
        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.CreateBhsd(
            unaligned, DirectPtxPhysicalType.Float16, 1024));
        using var oddExtent = new SyntheticGpuBuffer((IntPtr)0x1000, 1023);
        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.CreateBhsd(
            oddExtent, DirectPtxPhysicalType.Float32, 1020));
    }

    [Fact]
    public void DirectPtxFeatureGate_TestOverrideIsFailClosed()
    {
        bool? previous = DirectPtxFeatureGate.TestOverride;
        try
        {
            DirectPtxFeatureGate.TestOverride = false;
            Assert.False(DirectPtxFeatureGate.IsEnabled);
            DirectPtxFeatureGate.TestOverride = true;
            Assert.True(DirectPtxFeatureGate.IsEnabled);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableTheory]
    [InlineData(16, false, false, 1)]
    [InlineData(16, true, true, 1)]
    [InlineData(32, false, false, 1)]
    [InlineData(32, true, true, 2)]
    [InlineData(64, false, false, 2)]
    [InlineData(64, true, true, 4)]
    [InlineData(128, false, false, 4)]
    [InlineData(128, false, true, 8)]
    [InlineData(128, true, false, 4)]
    [InlineData(128, true, true, 8)]
    public void DriverOnlyOnlineAttention_MatchesOracleAndHasZeroLocalBytes(
        int sequence, bool isCausal, bool fuseLayerNormGelu, int warpsPerBlock)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        const int dimension = 64;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ComputeCapabilityMajor >= 8, "Requires SM80+ cp.async and mma.m16n8k16.");
        using var current = runtime.Enter();
        using var kernel = new PtxOnlineFusedAttention128x64Kernel(
            runtime, 1, isCausal, fuseLayerNormGelu, 0.125f, 1e-5f, sequence,
            emitSoftmaxStats: true, warpsPerBlock);
        Assert.Equal(0, kernel.FunctionInfo.LocalBytesPerThread);
        Assert.InRange(kernel.FunctionInfo.RegistersPerThread, 1, 255);

        using var qDevice = runtime.AllocateBytes(kernel.QBytes);
        using var kDevice = runtime.AllocateBytes(kernel.KBytes);
        using var vDevice = runtime.AllocateBytes(kernel.VBytes);
        using var gammaDevice = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
        using var betaDevice = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
        using var outputDevice = runtime.AllocateBytes(kernel.OutputBytes);
        using var statsDevice = runtime.AllocateBytes(kernel.StatsBytes);
        var random = new Random(20260720);
        ushort[] q = RandomHalfRange(random, sequence * dimension, 0.25f);
        ushort[] k = RandomHalfRange(random, sequence * dimension, 0.25f);
        ushort[] v = RandomHalfRange(random, sequence * dimension, 0.25f);
        float[] gamma = Enumerable.Range(0, dimension).Select(i => 0.75f + i / 256f).ToArray();
        float[] beta = Enumerable.Range(0, dimension).Select(i => (i - 32) / 512f).ToArray();
        qDevice.Upload<ushort>(q);
        kDevice.Upload<ushort>(k);
        vDevice.Upload<ushort>(v);
        gammaDevice.Upload<float>(gamma);
        betaDevice.Upload<float>(beta);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(qDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(kDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(vDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(gammaDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(betaDevice, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(statsDevice, kernel.Blueprint.Tensors[6]));
        runtime.Synchronize();
        var actual = new float[sequence * dimension];
        outputDevice.Download<float>(actual);

        float maxError = ValidateOnlineFirstHead(
            actual, q, k, v, gamma, beta, sequence, isCausal, fuseLayerNormGelu);
        Assert.True(maxError <= (fuseLayerNormGelu ? 0.025f : 0.012f),
            $"Online PTX max abs error {maxError:G9}.");
    }

    [SkippableTheory]
    [InlineData(2, 4, 2, 32, 64, false, 0)]
    [InlineData(2, 8, 1, 32, 64, true, 0)]
    [InlineData(2, 8, 2, 32, 64, true, 32)]
    [InlineData(1, 4, 4, 128, 32, true, 0)]
    public void DriverOnlyOnlineAttentionFamily_MatchesRectangularGqaOracleAndHasZeroLocalBytes(
        int batch, int queryHeads, int keyValueHeads,
        int querySequence, int keyValueSequence, bool isCausal, int causalQueryOffset)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        const int dimension = 64;
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in v3 attention family is validated on Ampere.");
        using var current = runtime.Enter();
        using var kernel = new PtxOnlineFusedAttention128x64Kernel(
            runtime, batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            isCausal, false, 0.125f, 1e-5f, emitSoftmaxStats: true,
            causalQueryOffset: causalQueryOffset);
        Assert.Equal(0, kernel.FunctionInfo.LocalBytesPerThread);
        Assert.Equal((long)batch * queryHeads * querySequence * dimension * sizeof(float),
            (long)kernel.OutputBytes);
        Assert.Equal((long)batch * keyValueHeads * keyValueSequence * dimension * sizeof(ushort),
            (long)kernel.KBytes);

        using var qDevice = runtime.AllocateBytes(kernel.QBytes);
        using var kDevice = runtime.AllocateBytes(kernel.KBytes);
        using var vDevice = runtime.AllocateBytes(kernel.VBytes);
        using var outputDevice = runtime.AllocateBytes(kernel.OutputBytes);
        using var statsDevice = runtime.AllocateBytes(kernel.StatsBytes);
        var random = new Random(
            20260800 + batch * 1000 + queryHeads * 100 + keyValueHeads * 10 +
            querySequence + keyValueSequence + (isCausal ? 1 : 0));
        ushort[] q = RandomHalfRange(
            random, batch * queryHeads * querySequence * dimension, 0.25f);
        ushort[] k = RandomHalfRange(
            random, batch * keyValueHeads * keyValueSequence * dimension, 0.25f);
        ushort[] v = RandomHalfRange(
            random, batch * keyValueHeads * keyValueSequence * dimension, 0.25f);
        qDevice.Upload<ushort>(q);
        kDevice.Upload<ushort>(k);
        vDevice.Upload<ushort>(v);

        DirectPtxTensorView qView = DirectPtxTensorView.CreateOwned(qDevice, kernel.Blueprint.Tensors[0]);
        DirectPtxTensorView kView = DirectPtxTensorView.CreateOwned(kDevice, kernel.Blueprint.Tensors[1]);
        DirectPtxTensorView vView = DirectPtxTensorView.CreateOwned(vDevice, kernel.Blueprint.Tensors[2]);
        DirectPtxTensorView outputView = DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[5]);
        DirectPtxTensorView statsView = DirectPtxTensorView.CreateOwned(statsDevice, kernel.Blueprint.Tensors[6]);
        kernel.Launch(qView, kView, vView, default, default, outputView, statsView);
        runtime.Synchronize();
        long launchBefore = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 32; i++)
            kernel.Launch(qView, kView, vView, default, default, outputView, statsView);
        long launchAllocated = GC.GetAllocatedBytesForCurrentThread() - launchBefore;
        runtime.Synchronize();
        Assert.True(launchAllocated == 0, $"raw module launch allocated {launchAllocated} bytes");
        var actual = new float[batch * queryHeads * querySequence * dimension];
        var actualStats = new float[batch * queryHeads * querySequence];
        outputDevice.Download<float>(actual);
        statsDevice.Download<float>(actualStats);

        (float outputError, float statsError) = ValidateOnlineFamily(
            actual, actualStats, q, k, v, batch, queryHeads, keyValueHeads,
            querySequence, keyValueSequence, isCausal, causalQueryOffset);
        Assert.True(outputError <= 0.012f,
            $"Rectangular GQA PTX max output error {outputError:G9}.");
        Assert.True(statsError <= 0.02f,
            $"Rectangular GQA PTX max LSE error {statsError:G9}.");
        Assert.Contains(queryHeads == keyValueHeads ? "mha" : keyValueHeads == 1 ? "mqa" : "gqa",
            kernel.Blueprint.Semantics["head-mapping"]);
    }

    [SkippableFact]
    public void CudaBackend_OptInUsesBorrowedContextAndCarriesZeroSpillAudit()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires an SM80+ CUDA backend.");
            const int elements = 128 * 64;
            float[] host = Enumerable.Range(0, elements).Select(i => (i % 29 - 14) / 128f).ToArray();
            using var qFloat = backend.AllocateBuffer(host);
            using var kFloat = backend.AllocateBuffer(host);
            using var vFloat = backend.AllocateBuffer(host);
            using var qHalf = backend.AllocateByteBuffer(elements * 2);
            using var kHalf = backend.AllocateByteBuffer(elements * 2);
            using var vHalf = backend.AllocateByteBuffer(elements * 2);
            using var output = backend.AllocateBuffer(elements);
            using var stats = backend.AllocateBuffer(128);
            backend.ConvertToFp16Native(qFloat, qHalf, elements);
            backend.ConvertToFp16Native(kFloat, kHalf, elements);
            backend.ConvertToFp16Native(vFloat, vHalf, elements);

            Assert.True(backend.TryDirectPtxOnlineAttention(
                qHalf, kHalf, vHalf, output, stats, 1, 0.125f, isCausal: false),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.All(backend.DownloadBuffer(output), value => Assert.True(float.IsFinite(value)));
            Assert.True(backend.TryGetDirectPtxAttentionAudit(
                1, 0.125f, false, false, 1e-5f, 128,
                out DirectPtxFunctionInfo info, out _));
            Assert.Equal(0, info.LocalBytesPerThread);
            Assert.True(backend.TryGetDirectPtxAttentionKernelAudit(
                1, 0.125f, false, false, 1e-5f, 128,
                out DirectPtxKernelAudit audit));
            Assert.Contains("online-attention-d64-v3-Ampere-q16-k16-w", audit.BlueprintId);
            Assert.Equal(64, audit.PtxSha256.Length);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableFact]
    public void TensorEngine_FlashAttentionRoutesCanonicalShapeThroughFeatureGate()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var engine = new DirectGpuTensorEngine();
            Skip.IfNot(engine.IsGpuAvailable, "CUDA tensor engine is unavailable.");
            var backend = Assert.IsType<CudaBackend>(engine.GetBackend());
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires SM80+.");
            const int elements = 128 * 64;
            var q = new Tensor<float>(
                Enumerable.Range(0, elements).Select(i => (i % 17 - 8) / 128f).ToArray(),
                [1, 1, 128, 64]);
            var k = new Tensor<float>(
                Enumerable.Range(0, elements).Select(i => (i % 19 - 9) / 128f).ToArray(),
                [1, 1, 128, 64]);
            var v = new Tensor<float>(
                Enumerable.Range(0, elements).Select(i => (i % 23 - 11) / 128f).ToArray(),
                [1, 1, 128, 64]);
            long before = backend.DirectPtxAttentionDispatchCount;

            Tensor<float> result = engine.FlashAttention(
                q, k, v, 0.125, false, out Tensor<float> stats);

            Assert.True(float.IsFinite(result.GetFlat(0)));
            Assert.True(float.IsFinite(stats.GetFlat(0)));
            Assert.Equal(before + 1, backend.DirectPtxAttentionDispatchCount);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void PhysicalAbi_SeparatesLogicalPhysicalLayoutAndAlignment()
    {
        var logical = new DirectPtxExtent(2, 3);
        var physical = new DirectPtxExtent(2, 8);
        var contract = new DirectPtxTensorContract(
            "padded", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
            logical, physical, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact);
        using var exact = new SyntheticGpuBuffer((IntPtr)0x2000, 32);
        DirectPtxTensorView view = DirectPtxTensorView.Create(exact, contract);
        Assert.Equal(logical, view.LogicalExtent);
        Assert.Equal(physical, view.PhysicalExtent);
        Assert.Equal(DirectPtxPhysicalLayout.RowMajor2D, view.Layout);
        Assert.Equal((nuint)32, view.ByteLength);

        using var oversized = new SyntheticGpuBuffer((IntPtr)0x2000, 48);
        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.Create(oversized, contract));
        using var misaligned = new SyntheticGpuBuffer((IntPtr)0x2008, 32);
        Assert.Throws<ArgumentException>(() => DirectPtxTensorView.Create(misaligned, contract));
    }

    [Theory]
    [InlineData(8, 0, (int)DirectPtxArchitectureFamily.Ampere)]
    [InlineData(8, 6, (int)DirectPtxArchitectureFamily.Ampere)]
    [InlineData(8, 9, (int)DirectPtxArchitectureFamily.Ada)]
    [InlineData(9, 0, (int)DirectPtxArchitectureFamily.Hopper)]
    [InlineData(10, 0, (int)DirectPtxArchitectureFamily.Blackwell)]
    [InlineData(7, 5, (int)DirectPtxArchitectureFamily.Unsupported)]
    public void ArchitectureFamilies_AreSeparateDispatchDomains(
        int major, int minor, int expectedValue)
    {
        var expected = (DirectPtxArchitectureFamily)expectedValue;
        Assert.Equal(expected, DirectPtxArchitecture.Classify(major, minor));
        Assert.Equal(expected == DirectPtxArchitectureFamily.Ampere,
            DirectPtxArchitecture.HasValidatedOnlineAttention(expected));
    }

    [Fact]
    public void KernelCache_IsBoundedLruAndDisposesEvictions()
    {
        using var cache = new DirectPtxKernelCache<int, TrackingDisposable>(2);
        TrackingDisposable one = cache.GetOrAdd(1, () => new TrackingDisposable());
        TrackingDisposable two = cache.GetOrAdd(2, () => new TrackingDisposable());
        Assert.True(cache.TryGetValue(1, out _)); // 2 becomes least-recently used
        TrackingDisposable three = cache.GetOrAdd(3, () => new TrackingDisposable());
        Assert.True(two.IsDisposed);
        Assert.False(one.IsDisposed);
        Assert.False(three.IsDisposed);
        Assert.Equal(2, cache.Count);

        var plans = new DirectPtxPlanCache<int, string>(2);
        plans.Set(1, "one");
        plans.Set(2, "two");
        Assert.True(plans.TryGetValue(1, out _));
        plans.Set(3, "three");
        Assert.False(plans.TryGetValue(2, out _));
        Assert.True(plans.TryGetValue(3, out string value));
        Assert.Equal("three", value);
    }

    [Fact]
    public void AttentionAutotuner_ExposesArchitectureIndependentShapeCandidates()
    {
        Assert.Equal(new[] { 1 }, DirectPtxAttentionAutotuner.Candidates(16));
        Assert.Equal(new[] { 2, 1 }, DirectPtxAttentionAutotuner.Candidates(32));
        Assert.Equal(new[] { 4, 2 }, DirectPtxAttentionAutotuner.Candidates(64));
        Assert.Equal(new[] { 8, 4 }, DirectPtxAttentionAutotuner.Candidates(128));
        Assert.Throws<ArgumentOutOfRangeException>(() => DirectPtxAttentionAutotuner.Candidates(96));
    }

    [Fact]
    public void AttentionEligibility_AdmitsOnlyTheProvenSemanticDomain()
    {
        var supported = new DirectPtxAttentionRequest(
            DirectPtxArchitectureFamily.Ampere, DirectPtxPhysicalType.Float16,
            DirectPtxPhysicalLayout.Bhsd, 2, 8, 8, 128, 128, 64,
            DirectPtxAttentionMaskKind.CausalTopLeft, DirectPtxAttentionPhase.Inference,
            0, false, false);
        Assert.True(DirectPtxAttentionEligibility.Evaluate(supported).IsEligible);

        Assert.Equal("dtype-not-fp16", DirectPtxAttentionEligibility.Evaluate(
            supported with { InputType = DirectPtxPhysicalType.BFloat16 }).Reason);
        Assert.True(DirectPtxAttentionEligibility.Evaluate(
            supported with { KeyValueSequence = 64, KeyValueHeads = 2 }).IsEligible);
        Assert.Equal("query-heads-not-divisible-by-kv-heads", DirectPtxAttentionEligibility.Evaluate(
            supported with { KeyValueHeads = 3 }).Reason);
        Assert.Equal("query-sequence-bucket-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { QuerySequence = 48 }).Reason);
        Assert.Equal("kv-sequence-bucket-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { KeyValueSequence = 48 }).Reason);
        Assert.Equal("mask-kind-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { Mask = DirectPtxAttentionMaskKind.AdditiveBias }).Reason);
        Assert.Equal("dropout-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { DropoutProbability = 0.1f }).Reason);
        Assert.Equal("training-or-backward-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { Phase = DirectPtxAttentionPhase.Backward }).Reason);
        Assert.Equal("ragged-layout-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { IsRagged = true }).Reason);
        Assert.Equal("paged-kv-not-implemented", DirectPtxAttentionEligibility.Evaluate(
            supported with { UsesPagedKv = true }).Reason);
    }

    [Fact]
    public void AttentionCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        string[] expected =
        [
            "ScaledDotProductAttention", "ScaledDotProductAttentionBackward",
            "FlashAttention", "FlashAttentionV2", "FlashAttentionBackward",
            "GroupedQueryAttention", "GroupedQueryAttentionBackward", "FlashDecode",
            "PagedAttentionDecode", "PagedAttentionPrefill",
            "PagedAttentionDecodeGqa", "PagedAttentionPrefillGqa"
        ];
        string[] actual = DirectPtxAttentionCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(
            expected.OrderBy(name => name, StringComparer.Ordinal).ToArray(), actual);
        Assert.Equal(actual.Length, actual.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxAttentionCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxAttentionCoverageManifest.Get("UnassignedAttentionApi"));
    }

    [Fact]
    public void NsightEvidence_RequiresEveryExecutedSpillCounterToBeZero()
    {
        string path = System.IO.Path.GetTempFileName();
        try
        {
            System.IO.File.WriteAllText(path,
                "\"sass__inst_executed_register_spilling\",\"0\"\n" +
                "\"sass__inst_executed_local_loads\",\"0\"\n" +
                "\"sass__inst_executed_local_stores\",\"0\"\n" +
                "\"derived__local_spilling_requests\",\"0\"\n");
            DirectPtxProfilerEvidence zero = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.True(zero.ProvesZeroExecutedSpills);

            System.IO.File.WriteAllText(path,
                "\"sass__inst_executed_register_spilling\",\"4\"\n");
            DirectPtxProfilerEvidence spilling = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.False(spilling.ProvesZeroExecutedSpills);
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    [Fact]
    public void ReleaseGate_RequiresSpeedTailAllocationAccuracySpillsAndRepetition()
    {
        DirectPtxReleaseGatePolicy policy = DirectPtxReleaseGatePolicy.ProductionDefault;
        var competitor = new DirectPtxPerformanceEvidence(
            100, 120, 64, 4096, 0, -1, 3);
        var passing = new DirectPtxPerformanceEvidence(
            80, 110, 0, 0, 1e-6, 0, 3);
        Assert.True(policy.Evaluate(passing, competitor).Passed);

        DirectPtxReleaseDecision failing = policy.Evaluate(
            passing with
            {
                MedianMicroseconds = 95,
                P95Microseconds = 140,
                ManagedBytesPerCall = 8,
                TemporaryDeviceBytes = 4,
                MaxAbsoluteError = 1e-3,
                LocalBytesPerThread = 16,
                IndependentRuns = 1
            }, competitor);
        Assert.False(failing.Passed);
        Assert.Equal(7, failing.Failures.Count);
    }

    [Fact]
    public void ResidualRmsNormEmitter_IsRegisterReducedAndStrideFree()
    {
        string ptx = PtxFusedResidualRmsNormD64Kernel.EmitPtx(8, 6, 1e-5f);
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Contains("sqrt.rn.f32", ptx);
        Assert.Contains("rmsnorm", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void DriverOnlyResidualRmsNorm_MatchesReferenceAndHasZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in residual RMSNorm specialization is validated on Ampere.");
        const int rows = 37, dimension = 64;
        using var kernel = new PtxFusedResidualRmsNormD64Kernel(runtime, rows);
        Assert.Equal(0, kernel.FunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.FunctionInfo.StaticSharedBytes);
        Assert.Equal(64, kernel.Blueprint.Tensors[0].LogicalExtent.D1);
        Assert.Equal(64, kernel.Audit.PtxSha256.Length);

        using var input = runtime.AllocateBytes(kernel.InputBytes);
        using var residual = runtime.AllocateBytes(kernel.InputBytes);
        using var gamma = runtime.AllocateBytes(PtxFusedResidualRmsNormD64Kernel.GammaBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        using var rms = runtime.AllocateBytes(kernel.RmsBytes);
        var random = new Random(20260721);
        float[] x = Enumerable.Range(0, rows * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 2f).ToArray();
        float[] r = Enumerable.Range(0, rows * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] g = Enumerable.Range(0, dimension).Select(i => 0.75f + i / 256f).ToArray();
        input.Upload<float>(x);
        residual.Upload<float>(r);
        gamma.Upload<float>(g);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(rms, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[x.Length];
        var actualRms = new float[rows];
        output.Download<float>(actual);
        rms.Download<float>(actualRms);

        float maxError = 0;
        for (int row = 0; row < rows; row++)
        {
            float sumSquares = 0;
            for (int d = 0; d < dimension; d++)
            {
                float value = x[row * dimension + d] + r[row * dimension + d];
                sumSquares += value * value;
            }
            float expectedRms = MathF.Sqrt(sumSquares / dimension + 1e-5f);
            maxError = MathF.Max(maxError, MathF.Abs(actualRms[row] - expectedRms));
            for (int d = 0; d < dimension; d++)
            {
                float expected = (x[row * dimension + d] + r[row * dimension + d]) /
                    expectedRms * g[d];
                maxError = MathF.Max(maxError,
                    MathF.Abs(actual[row * dimension + d] - expected));
            }
        }
        Assert.True(maxError < 5e-5f, $"Residual RMSNorm max error {maxError:G9}.");
    }

    [SkippableFact]
    public void BackendResidualRmsNorm_PrewarmedKernelIsCudaGraphCapturable()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxResidualRmsNormEnabled, "Requires an Ampere CUDA backend.");
            const int rows = 32, elements = rows * 64;
            using var input = backend.AllocateBuffer(Enumerable.Repeat(0.25f, elements).ToArray());
            using var residual = backend.AllocateBuffer(Enumerable.Repeat(0.5f, elements).ToArray());
            using var gamma = backend.AllocateBuffer(Enumerable.Repeat(1f, 64).ToArray());
            using var output = backend.AllocateBuffer(elements);
            using var rms = backend.AllocateBuffer(rows);
            Assert.True(backend.PrewarmDirectPtxFusedResidualRmsNorm(rows), backend.DirectPtxLastError);

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxFusedResidualRmsNorm(
                    input, residual, gamma, output, rms, rows), backend.DirectPtxLastError));
            Assert.NotEqual(IntPtr.Zero, graph);
            try
            {
                backend.LaunchCapturedGraph(graph);
                Assert.All(backend.DownloadBuffer(output), value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableFact]
    public void BackendAttention_PrewarmedKernelIsCudaGraphCapturable()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires an Ampere CUDA backend.");
            const int sequence = 16, elements = sequence * 64;
            float[] host = Enumerable.Range(0, elements).Select(i => (i % 13 - 6) / 128f).ToArray();
            using var qFloat = backend.AllocateBuffer(host);
            using var kFloat = backend.AllocateBuffer(host);
            using var vFloat = backend.AllocateBuffer(host);
            using var q = backend.AllocateByteBuffer(elements * 2);
            using var k = backend.AllocateByteBuffer(elements * 2);
            using var v = backend.AllocateByteBuffer(elements * 2);
            using var output = backend.AllocateBuffer(elements);
            using var stats = backend.AllocateBuffer(sequence);
            backend.ConvertToFp16Native(qFloat, q, elements);
            backend.ConvertToFp16Native(kFloat, k, elements);
            backend.ConvertToFp16Native(vFloat, v, elements);
            Assert.True(backend.PrewarmDirectPtxAttention(
                1, sequence, false, false, 0.125f), backend.DirectPtxLastError);

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxOnlineAttention(
                    q, k, v, output, stats, 1, 0.125f, false, sequence),
                    backend.DirectPtxLastError));
            Assert.NotEqual(IntPtr.Zero, graph);
            try
            {
                backend.LaunchCapturedGraph(graph);
                Assert.All(backend.DownloadBuffer(output), value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableFact]
    public void BackendAttentionFamily_PrewarmedRectangularMqaKernelIsCudaGraphCapturable()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires an Ampere CUDA backend.");
            const int batch = 1, queryHeads = 4, keyValueHeads = 1;
            const int querySequence = 16, keyValueSequence = 32, dimension = 64;
            int queryElements = batch * queryHeads * querySequence * dimension;
            int keyValueElements = batch * keyValueHeads * keyValueSequence * dimension;
            using var qFloat = backend.AllocateBuffer(Enumerable.Repeat(0.03125f, queryElements).ToArray());
            using var kFloat = backend.AllocateBuffer(Enumerable.Repeat(0.015625f, keyValueElements).ToArray());
            using var vFloat = backend.AllocateBuffer(Enumerable.Repeat(0.0625f, keyValueElements).ToArray());
            using var q = backend.AllocateByteBuffer(queryElements * sizeof(ushort));
            using var k = backend.AllocateByteBuffer(keyValueElements * sizeof(ushort));
            using var v = backend.AllocateByteBuffer(keyValueElements * sizeof(ushort));
            using var output = backend.AllocateBuffer(queryElements);
            backend.ConvertToFp16Native(qFloat, q, queryElements);
            backend.ConvertToFp16Native(kFloat, k, keyValueElements);
            backend.ConvertToFp16Native(vFloat, v, keyValueElements);
            Assert.True(backend.PrewarmDirectPtxAttentionFamily(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                false, false, 0.125f, emitSoftmaxStats: false),
                backend.DirectPtxLastError);

            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxOnlineAttentionFamily(
                    q, k, v, output, null,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    0.125f, isCausal: false, emitSoftmaxStats: false));
            backend.Synchronize();
            long captureBefore = GC.GetAllocatedBytesForCurrentThread();
            bool captureStatus = false;
            for (int i = 0; i < 32; i++) captureStatus |= backend.IsStreamCapturing();
            long captureAllocated = GC.GetAllocatedBytesForCurrentThread() - captureBefore;
            Assert.False(captureStatus);
            Assert.True(captureAllocated == 0, $"capture check allocated {captureAllocated} bytes");

            long gateBefore = GC.GetAllocatedBytesForCurrentThread();
            bool gateStatus = true;
            for (int i = 0; i < 32; i++) gateStatus &= backend.IsDirectPtxAttentionEnabled;
            long gateAllocated = GC.GetAllocatedBytesForCurrentThread() - gateBefore;
            Assert.True(gateStatus);
            Assert.True(gateAllocated == 0, $"feature gate allocated {gateAllocated} bytes");

            long cacheBefore = GC.GetAllocatedBytesForCurrentThread();
            bool cacheStatus = true;
            for (int i = 0; i < 32; i++)
                cacheStatus &= backend.TryGetDirectPtxAttentionKernelAuditFamily(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    0.125f, false, false, 1e-5f, false, 0, out _);
            long cacheAllocated = GC.GetAllocatedBytesForCurrentThread() - cacheBefore;
            Assert.True(cacheStatus);
            Assert.True(cacheAllocated == 0, $"cache audit allocated {cacheAllocated} bytes");

            var eligibilityRequest = new DirectPtxAttentionRequest(
                DirectPtxArchitectureFamily.Ampere, DirectPtxPhysicalType.Float16,
                DirectPtxPhysicalLayout.Bhsd, batch, queryHeads, keyValueHeads,
                querySequence, keyValueSequence, 64, DirectPtxAttentionMaskKind.None,
                DirectPtxAttentionPhase.Inference, 0, false, false);
            long eligibilityBefore = GC.GetAllocatedBytesForCurrentThread();
            bool eligibilityStatus = true;
            for (int i = 0; i < 32; i++)
                eligibilityStatus &= DirectPtxAttentionEligibility.Evaluate(eligibilityRequest).IsEligible;
            long eligibilityAllocated = GC.GetAllocatedBytesForCurrentThread() - eligibilityBefore;
            Assert.True(eligibilityStatus);
            Assert.True(eligibilityAllocated == 0, $"eligibility allocated {eligibilityAllocated} bytes");

            long contextBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++) backend.EnsureContextCurrent();
            long contextAllocated = GC.GetAllocatedBytesForCurrentThread() - contextBefore;
            Assert.True(contextAllocated == 0, $"context assertion allocated {contextAllocated} bytes");

            var qContract = new DirectPtxTensorContract(
                "q", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                new DirectPtxExtent(batch, queryHeads, querySequence, 64),
                new DirectPtxExtent(batch, queryHeads, querySequence, 64),
                16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact);
            var kContract = new DirectPtxTensorContract(
                "k", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                new DirectPtxExtent(batch, keyValueHeads, keyValueSequence, 64),
                new DirectPtxExtent(batch, keyValueHeads, keyValueSequence, 64),
                16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact);
            var outputContract = new DirectPtxTensorContract(
                "o", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Bhsd,
                new DirectPtxExtent(batch, queryHeads, querySequence, 64),
                new DirectPtxExtent(batch, queryHeads, querySequence, 64),
                16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact);
            long viewBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
            {
                _ = DirectPtxTensorView.Create(q, qContract);
                _ = DirectPtxTensorView.Create(k, kContract);
                _ = DirectPtxTensorView.Create(v, kContract);
                _ = DirectPtxTensorView.Create(output, outputContract);
            }
            long viewAllocated = GC.GetAllocatedBytesForCurrentThread() - viewBefore;
            Assert.True(viewAllocated == 0, $"physical views allocated {viewAllocated} bytes");

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxOnlineAttentionFamily(
                    q, k, v, output, null,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    0.125f, isCausal: false, emitSoftmaxStats: false);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            IntPtr graph = backend.CaptureGraph(() =>
                Assert.True(backend.TryDirectPtxOnlineAttentionFamily(
                    q, k, v, output, null,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    0.125f, isCausal: false, emitSoftmaxStats: false), backend.DirectPtxLastError));
            Assert.NotEqual(IntPtr.Zero, graph);
            try
            {
                backend.LaunchCapturedGraph(graph);
                Assert.All(backend.DownloadBuffer(output), value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    private static ushort[] RandomHalfRange(Random random, int length, float magnitude)
    {
        var result = new ushort[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.HalfToUInt16Bits(
                (Half)((random.NextSingle() - 0.5f) * 2f * magnitude));
        return result;
    }

    private static (float OutputError, float StatsError) ValidateOnlineFamily(
        float[] actual,
        float[] actualStats,
        ushort[] q,
        ushort[] k,
        ushort[] v,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool causal,
        int causalQueryOffset = 0)
    {
        const int dimension = 64;
        const float scale = 0.125f;
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        var scores = new float[keyValueSequence];
        float maxOutputError = 0;
        float maxStatsError = 0;
        for (int b = 0; b < batch; b++)
        for (int queryHead = 0; queryHead < queryHeads; queryHead++)
        {
            int keyValueHead = queryHead / queriesPerKeyValue;
            int flatQueryHead = b * queryHeads + queryHead;
            int flatKeyValueHead = b * keyValueHeads + keyValueHead;
            for (int row = 0; row < querySequence; row++)
            {
                int lastKey = causal
                    ? Math.Min(row + causalQueryOffset, keyValueSequence - 1)
                    : keyValueSequence - 1;
                float maximum = float.NegativeInfinity;
                int queryBase = (flatQueryHead * querySequence + row) * dimension;
                for (int column = 0; column <= lastKey; column++)
                {
                    int keyBase = (flatKeyValueHead * keyValueSequence + column) * dimension;
                    float score = 0;
                    for (int d = 0; d < dimension; d++)
                        score += Half(q[queryBase + d]) * Half(k[keyBase + d]);
                    scores[column] = score * scale;
                    maximum = MathF.Max(maximum, scores[column]);
                }

                float sum = 0;
                for (int column = 0; column <= lastKey; column++)
                {
                    scores[column] = MathF.Exp(scores[column] - maximum);
                    sum += scores[column];
                }
                for (int d = 0; d < dimension; d++)
                {
                    float expected = 0;
                    for (int column = 0; column <= lastKey; column++)
                    {
                        int valueBase = (flatKeyValueHead * keyValueSequence + column) * dimension;
                        expected += scores[column] / sum * Half(v[valueBase + d]);
                    }
                    maxOutputError = MathF.Max(maxOutputError,
                        MathF.Abs(actual[queryBase + d] - expected));
                }
                float expectedStats = maximum + MathF.Log(sum);
                maxStatsError = MathF.Max(maxStatsError,
                    MathF.Abs(actualStats[flatQueryHead * querySequence + row] - expectedStats));
            }
        }
        return (maxOutputError, maxStatsError);
    }

    private static float ValidateOnlineFirstHead(
        float[] actual, ushort[] q, ushort[] k, ushort[] v,
        float[] gamma, float[] beta, int sequence, bool causal, bool epilogue)
    {
        const int dimension = 64;
        var scores = new float[sequence];
        var rowOutput = new float[dimension];
        float maxError = 0;
        for (int row = 0; row < sequence; row++)
        {
            int lastKey = causal ? row : sequence - 1;
            float maximum = float.NegativeInfinity;
            for (int column = 0; column <= lastKey; column++)
            {
                float score = 0;
                for (int d = 0; d < dimension; d++)
                    score += Half(q[row * dimension + d]) * Half(k[column * dimension + d]);
                scores[column] = score * 0.125f;
                maximum = MathF.Max(maximum, scores[column]);
            }
            float sum = 0;
            for (int column = 0; column <= lastKey; column++)
            {
                scores[column] = MathF.Exp(scores[column] - maximum);
                sum += scores[column];
            }
            for (int d = 0; d < dimension; d++)
            {
                float value = 0;
                for (int column = 0; column <= lastKey; column++)
                    value += scores[column] / sum * Half(v[column * dimension + d]);
                rowOutput[d] = value;
            }
            if (epilogue)
            {
                float mean = rowOutput.Average();
                float variance = rowOutput.Select(x => (x - mean) * (x - mean)).Average();
                float inverseStd = 1f / MathF.Sqrt(variance + 1e-5f);
                for (int d = 0; d < dimension; d++)
                {
                    float x = (rowOutput[d] - mean) * inverseStd * gamma[d] + beta[d];
                    rowOutput[d] = 0.5f * x *
                        (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                }
            }
            for (int d = 0; d < dimension; d++)
                maxError = MathF.Max(maxError, MathF.Abs(actual[row * dimension + d] - rowOutput[d]));
        }
        return maxError;
    }

    private static float Half(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < length; i++)
        {
            var value = (Half)(random.NextDouble() * 2.0 - 1.0);
            result[i] = BitConverter.HalfToUInt16Bits(value);
        }
        return result;
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }

    private sealed class SyntheticGpuBuffer : IGpuBuffer
    {
        internal SyntheticGpuBuffer(IntPtr handle, long bytes)
        {
            Handle = handle;
            SizeInBytes = bytes;
        }

        public int Size => checked((int)(SizeInBytes / sizeof(float)));
        public long SizeInBytes { get; }
        public IntPtr Handle { get; }
        public void Dispose() { }
    }

    private sealed class TrackingDisposable : IDisposable
    {
        internal bool IsDisposed { get; private set; }
        public void Dispose() => IsDisposed = true;
    }
}
#endif
