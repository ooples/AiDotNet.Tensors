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
        Skip.IfNot(DirectPtxArchitecture.HasValidatedOnlineAttention(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the validated SM86 cp.async/mma specialization.");
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
    [InlineData(1, 4, 2, 128, 64, true, -64)]
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
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires a validated Ampere CUDA backend.");
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
            // Cross the tiered-JIT promotion threshold before asserting the
            // resident steady-state contract; JIT bookkeeping is not dispatch
            // allocation and otherwise makes this assertion process-order dependent.
            for (int i = 0; i < 64; i++)
            {
                Assert.True(backend.TryDirectPtxOnlineAttention(
                    qHalf, kHalf, vHalf, output, stats, 1, 0.125f, isCausal: false),
                    backend.DirectPtxLastError);
            }
            backend.Synchronize();
            long allocationBefore = GC.GetAllocatedBytesForCurrentThread();
            bool everyLaunchSucceeded = true;
            for (int i = 0; i < 32; i++)
            {
                everyLaunchSucceeded &= backend.TryDirectPtxOnlineAttention(
                    qHalf, kHalf, vHalf, output, stats, 1, 0.125f, isCausal: false);
            }
            long launchAllocation =
                GC.GetAllocatedBytesForCurrentThread() - allocationBefore;
            backend.Synchronize();
            long synchronizationBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++) backend.Synchronize();
            long synchronizationAllocation =
                GC.GetAllocatedBytesForCurrentThread() - synchronizationBefore;
            Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
            Assert.Equal(0, launchAllocation);
            Assert.Equal(0, synchronizationAllocation);
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
            Skip.IfNot(backend.IsDirectPtxAttentionEnabled, "Requires a validated Ampere CUDA backend.");
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

            var paddedQuery = new Tensor<float>(
                Enumerable.Range(0, 129 * 64).Select(i => (i % 17 - 8) / 128f).ToArray(),
                [1, 1, 129, 64]);
            Tensor<float> offsetQuery = paddedQuery.Slice(axis: 2, start: 1, end: 129);
            long beforeFallback = backend.DirectPtxAttentionDispatchCount;
            Tensor<float> fallbackResult = engine.FlashAttention(
                offsetQuery, k, v, 0.125, false, out Tensor<float> fallbackStats);

            Assert.True(float.IsFinite(fallbackResult.GetFlat(0)));
            Assert.True(float.IsFinite(fallbackStats.GetFlat(0)));
            Assert.Equal(beforeFallback, backend.DirectPtxAttentionDispatchCount);
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
        Assert.Equal(major == 8 && minor == 6,
            DirectPtxArchitecture.HasValidatedOnlineAttention(major, minor));
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
        var duplicate = new TrackingDisposable();
        Assert.Same(one, cache.AddOrGetExisting(1, duplicate));
        Assert.True(duplicate.IsDisposed);
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
    public void KernelCache_GraphPinnedEntriesAreNeverEvicted()
    {
        using var cache = new DirectPtxKernelCache<int, TrackingDisposable>(2);
        TrackingDisposable one = cache.GetOrAdd(1, () => new TrackingDisposable());
        TrackingDisposable two = cache.GetOrAdd(2, () => new TrackingDisposable());
        Assert.True(cache.Pin(1));
        TrackingDisposable three = cache.GetOrAdd(3, () => new TrackingDisposable());

        Assert.False(one.IsDisposed);
        Assert.True(two.IsDisposed);
        Assert.False(three.IsDisposed);
        Assert.Equal(1, cache.PinnedCount);
        Assert.True(cache.Pin(3));

        var rejected = new TrackingDisposable();
        Assert.Throws<InvalidOperationException>(() => cache.AddOrGetExisting(4, rejected));
        Assert.True(rejected.IsDisposed);
        Assert.Equal(2, cache.Count);
        Assert.Equal(2, cache.PinnedCount);

        bool factoryCalled = false;
        Assert.Throws<InvalidOperationException>(() => cache.GetOrAdd(5, () =>
        {
            factoryCalled = true;
            return new TrackingDisposable();
        }));
        Assert.False(factoryCalled);
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
            DirectPtxArchitectureFamily.Ampere, 8, 6, DirectPtxPhysicalType.Float16,
            DirectPtxPhysicalLayout.Bhsd, 2, 8, 8, 128, 128, 64,
            DirectPtxAttentionMaskKind.CausalTopLeft, DirectPtxAttentionPhase.Inference,
            0, false, false);
        Assert.True(DirectPtxAttentionEligibility.Evaluate(supported).IsEligible);

        Assert.Equal("sm-version-not-validated", DirectPtxAttentionEligibility.Evaluate(
            supported with { ComputeCapabilityMinor = 0 }).Reason);

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

    [Theory]
    [InlineData(16, true)]
    [InlineData(32, true)]
    [InlineData(64, true)]
    [InlineData(128, true)]
    [InlineData(0, false)]
    [InlineData(15, false)]
    [InlineData(96, false)]
    [InlineData(256, false)]
    public void OnlineAttentionSequenceAdmission_UsesCanonicalShapeDefinition(
        int sequenceLength, bool expected) =>
        Assert.Equal(expected,
            PtxOnlineFusedAttention128x64Kernel.IsSupportedSequenceLength(sequenceLength));

    [Fact]
    public void CanonicalDenseAdmission_RejectsOffsetViews()
    {
        var canonical = new Tensor<float>(new float[8], [2, 4]);
        Tensor<float> offsetView = canonical.Slice(axis: 0, start: 1, end: 2);
        var definition = typeof(DirectGpuTensorEngine).GetMethod(
            "IsCanonicalDenseAllocation",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!
            .MakeGenericMethod(typeof(float));

        Assert.True((bool)definition.Invoke(null, [canonical])!);
        Assert.False((bool)definition.Invoke(null, [offsetView])!);
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
                "\"sass__inst_executed_register_spilling.sum\",\"0\"\n" +
                "\"sass__inst_executed_local_loads\",\"0\"\n" +
                "\"sass__inst_executed_local_stores.sum\",\"0\"\n");
            DirectPtxProfilerEvidence zero = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.True(zero.ProvesZeroExecutedSpills);

            System.IO.File.WriteAllText(path,
                "\"sass__inst_executed_register_spilling\",\"4\"\n" +
                "\"sass__inst_executed_local_loads\",\"0\"\n" +
                "\"sass__inst_executed_local_stores\",\"0\"\n");
            DirectPtxProfilerEvidence spilling = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.False(spilling.ProvesZeroExecutedSpills);

            System.IO.File.WriteAllText(path,
                "\"sass__inst_executed_register_spilling\",\"0\"\n" +
                "\"sass__inst_executed_local_loads\",\"0\"\n");
            DirectPtxProfilerEvidence incomplete = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.False(incomplete.ProvesZeroExecutedSpills);

            System.IO.File.WriteAllText(path,
                "==PROF== Connected\n" +
                "\"ID\",\"Kernel Name\",\"smsp__sass_inst_executed_op_local.sum\",\"smsp__sass_inst_executed_op_local_ld.sum\",\"smsp__sass_inst_executed_op_local_st.sum\"\n" +
                "\"\",\"\",\"inst\",\"inst\",\"inst\"\n" +
                "\"0\",\"kernel_a\",\"0\",\"0\",\"0\"\n" +
                "\"1\",\"kernel_b\",\"0\",\"0\",\"0\"\n");
            DirectPtxProfilerEvidence modern = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.True(modern.ProvesZeroExecutedSpills);
            Assert.Equal(3, modern.ObservedMetricGroups);

            System.IO.File.WriteAllText(path,
                "\"ID\",\"Kernel Name\",\"smsp__sass_inst_executed_op_local.sum\",\"smsp__sass_inst_executed_op_local_ld.sum\",\"smsp__sass_inst_executed_op_local_st.sum\"\n" +
                "\"0\",\"kernel_a\",\"8\",\"8\",\"0\"\n");
            DirectPtxProfilerEvidence modernSpilling = DirectPtxProfilerEvidence.FromNcuCsv(path);
            Assert.False(modernSpilling.ProvesZeroExecutedSpills);
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

    [Fact]
    public void DecodeEmitters_AreWarpResidentStrideFreeAndSeparatePagedAbi()
    {
        string dense = PtxFusedDecodeAttentionD64Kernel.EmitPtx(
            8, 6, isPaged: false, queryHeads: 8, keyValueHeads: 2,
            sequenceLength: 32, blockSize: 0, poolBlocks: 0, scale: 0.125f);
        Assert.Contains(PtxFusedDecodeAttentionD64Kernel.DenseEntryPoint, dense);
        Assert.Contains("shfl.sync.bfly.b32", dense);
        Assert.Contains("st.global.v2.f32", dense);
        Assert.DoesNotContain(".shared", dense, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", dense, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", dense, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", dense, StringComparison.OrdinalIgnoreCase);

        string paged = PtxFusedDecodeAttentionD64Kernel.EmitPtx(
            8, 6, isPaged: true, queryHeads: 8, keyValueHeads: 1,
            sequenceLength: 32, blockSize: 16, poolBlocks: 4, scale: 0.125f);
        Assert.Contains(PtxFusedDecodeAttentionD64Kernel.PagedEntryPoint, paged);
        Assert.Contains("ld.param.u64 %rd3, [block_table_ptr]", paged);
        Assert.Equal(2, Count(paged, "ld.global.u32 %r6"));
        Assert.DoesNotContain(PtxFusedDecodeAttentionD64Kernel.DenseEntryPoint, paged);
    }

    [Fact]
    public void DecodeEmitter_RejectsEveryUnsupportedSpecializationDomain()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, false, 8, 2, 17, 0, 0, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, false, 3, 2, 32, 0, 0, 0.125f));
        Assert.Throws<ArgumentException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, false, 8, 2, 32, 16, 2, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, true, 8, 2, 32, 8, 4, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, true, 8, 2, 64, 16, 2, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedDecodeAttentionD64Kernel.EmitPtx(
                8, 6, false, 8, 2, 32, 0, 0, float.NaN));
    }

    [Fact]
    public void PagedPrefillEmitter_IsCausalWarpResidentAndStrideFree()
    {
        string ptx = PtxFusedPagedPrefillAttentionD64Kernel.EmitPtx(
            8, 6, queryHeads: 8, keyValueHeads: 2,
            queryCount: 4, startPosition: 12,
            blockSize: 16, poolBlocks: 4, scale: 0.125f);
        Assert.Contains(PtxFusedPagedPrefillAttentionD64Kernel.EntryPoint, ptx);
        Assert.Contains("mov.u32 %r9, %ctaid.y", ptx);
        Assert.Equal(16, Count(ptx, "setp.ge.u32 %p0"));
        Assert.Equal(1, Count(ptx, "ld.global.u32 %r6"));
        Assert.Contains("shfl.sync.bfly.b32", ptx);
        Assert.Contains("st.global.v2.f32", ptx);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedPagedPrefillAttentionD64Kernel.EmitPtx(
                8, 6, 8, 2, 3, 12, 16, 4, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedPagedPrefillAttentionD64Kernel.EmitPtx(
                8, 6, 8, 2, 32, 112, 16, 10, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedPagedPrefillAttentionD64Kernel.EmitPtx(
                8, 6, 8, 2, 4, 12, 8, 4, 0.125f));
    }

    [SkippableFact]
    public void DriverOnlyPagedPrefillD64_MatchesOracleAndHasZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in paged-prefill family is validated on Ampere.");
        const int heads = 8, kvHeads = 2, queries = 4, start = 12;
        const int dimension = 64, blockSize = 16, poolBlocks = 4;
        var random = new Random(20260843);
        float[] q = Enumerable.Range(0, queries * heads * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] k = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] v = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        int[] table = [3];
        using var kernel = new PtxFusedPagedPrefillAttentionD64Kernel(
            runtime, heads, kvHeads, queries, start, blockSize, poolBlocks, 0.125f);
        Assert.Equal(0, kernel.FunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.FunctionInfo.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 8);
        Assert.Equal(3, kernel.Blueprint.Tensors[0].LogicalExtent.Rank);
        Assert.Equal(4, kernel.Blueprint.Tensors[1].LogicalExtent.Rank);
        using var qDevice = runtime.AllocateBytes(kernel.QueryBytes);
        using var kDevice = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var vDevice = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var tableDevice = runtime.AllocateBytes(kernel.BlockTableBytes);
        using var outputDevice = runtime.AllocateBytes(kernel.OutputBytes);
        qDevice.Upload<float>(q); kDevice.Upload<float>(k); vDevice.Upload<float>(v);
        tableDevice.Upload<int>(table);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(qDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(kDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(vDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(tableDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[q.Length];
        outputDevice.Download<float>(actual);
        AssertDecodeClose(actual, PagedPrefillOracle(
            q, k, v, table, heads, kvHeads, queries, start, blockSize, 0.125f));
    }

    [SkippableFact]
    public void BackendPagedPrefillD64_IsZeroAllocationCapturableAndRoutesPublicGqa()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxPagedPrefillEnabled, "Requires an Ampere CUDA backend.");
            const int heads = 8, kvHeads = 2, queries = 4, start = 12;
            const int dimension = 64, blockSize = 16, poolBlocks = 4;
            var random = new Random(20260844);
            float[] qHost = Enumerable.Range(0, queries * heads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] kHost = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] vHost = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            int[] tableHost = [3];
            using var q = backend.AllocateBuffer(qHost);
            using var k = backend.AllocateBuffer(kHost);
            using var v = backend.AllocateBuffer(vHost);
            using var table = backend.AllocateIntBuffer(tableHost);
            using var output = backend.AllocateBuffer(qHost.Length);
            Assert.True(backend.PrewarmDirectPtxPagedPrefillD64(
                heads, kvHeads, queries, start, blockSize, poolBlocks, 0.125f),
                backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxPagedPrefillD64(
                    q, k, v, table, output, heads, kvHeads,
                    queries, start, blockSize, 0.125f), backend.DirectPtxLastError);
            backend.Synchronize();
            long beforeAllocation = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxPagedPrefillD64(
                    q, k, v, table, output, heads, kvHeads,
                    queries, start, blockSize, 0.125f);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - beforeAllocation;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxPagedPrefillD64(
                    q, k, v, table, output, heads, kvHeads,
                    queries, start, blockSize, 0.125f));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }

            long beforePublic = backend.DirectPtxPagedPrefillDispatchCount;
            using IGpuBuffer publicOutput = backend.PagedAttentionPrefillGqa(
                q, k, v, table, heads, kvHeads, dimension,
                blockSize, queries, start, 0.125f);
            Assert.Equal(beforePublic + 1, backend.DirectPtxPagedPrefillDispatchCount);
            float[] expected = PagedPrefillOracle(
                qHost, kHost, vHost, tableHost,
                heads, kvHeads, queries, start, blockSize, 0.125f);
            AssertDecodeClose(backend.DownloadBuffer(publicOutput), expected);
            Assert.True(backend.TryGetDirectPtxPagedPrefillAudit(
                heads, kvHeads, queries, start, blockSize, poolBlocks, 0.125f,
                out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DriverOnlyDecodeD64_MatchesOracleAndHasZeroLocalBytes(bool paged)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in decode family is validated on Ampere.");
        const int heads = 4, kvHeads = 2, sequence = 32, dimension = 64, blockSize = 16;
        int poolBlocks = paged ? 4 : 0;
        int keyValueElements = paged
            ? poolBlocks * blockSize * kvHeads * dimension
            : sequence * kvHeads * dimension;
        var random = new Random(paged ? 20260841 : 20260840);
        float[] q = Enumerable.Range(0, heads * dimension)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] k = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] v = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        int[] table = [3, 1];
        using var kernel = new PtxFusedDecodeAttentionD64Kernel(
            runtime, paged, heads, kvHeads, sequence,
            paged ? blockSize : 0, poolBlocks, 0.125f);
        Assert.Equal(0, kernel.FunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.FunctionInfo.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 8);
        Assert.Equal(paged ? 48 : 40, kernel.Blueprint.ResourceBudget.MaxRegistersPerThread);
        Assert.Equal(paged ? 4 : 3, kernel.Blueprint.Tensors[1].LogicalExtent.Rank);
        Assert.Equal(64, kernel.Audit.PtxSha256.Length);
        using var qDevice = runtime.AllocateBytes(kernel.QueryBytes);
        using var kDevice = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var vDevice = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var outputDevice = runtime.AllocateBytes(kernel.OutputBytes);
        qDevice.Upload<float>(q); kDevice.Upload<float>(k); vDevice.Upload<float>(v);
        if (paged)
        {
            using var tableDevice = runtime.AllocateBytes(kernel.BlockTableBytes);
            tableDevice.Upload<int>(table);
            kernel.LaunchPaged(
                DirectPtxTensorView.CreateOwned(qDevice, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(kDevice, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(vDevice, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(tableDevice, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[4]));
        }
        else
        {
            kernel.LaunchDense(
                DirectPtxTensorView.CreateOwned(qDevice, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(kDevice, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(vDevice, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[3]));
        }
        runtime.Synchronize();
        var actual = new float[q.Length];
        outputDevice.Download<float>(actual);
        float[] expected = DecodeOracle(
            q, k, v, heads, kvHeads, sequence,
            paged ? blockSize : 0, paged ? table : null, 0.125f);
        float error = 0;
        for (int i = 0; i < actual.Length; i++)
            error = MathF.Max(error, MathF.Abs(actual[i] - expected[i]));
        Assert.True(error < 2e-5f, $"Direct PTX decode max error {error:G9}.");
    }

    [SkippableFact]
    public void BackendDecodeD64_IsZeroAllocationCapturableAndRoutesDenseAndPagedApis()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFlashDecodeEnabled, "Requires an Ampere CUDA backend.");
            const int heads = 4, kvHeads = 2, sequence = 32, dimension = 64, blockSize = 16;
            const int poolBlocks = 4;
            var random = new Random(20260842);
            float[] qHost = Enumerable.Range(0, heads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] denseK = Enumerable.Range(0, sequence * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] denseV = Enumerable.Range(0, sequence * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            using var q = backend.AllocateBuffer(qHost);
            using var k = backend.AllocateBuffer(denseK);
            using var v = backend.AllocateBuffer(denseV);
            using var output = backend.AllocateBuffer(qHost.Length);
            Assert.True(backend.PrewarmDirectPtxDecodeD64(
                false, heads, kvHeads, sequence, 0, 0, 0.125f), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFlashDecodeD64(
                    q, k, v, output, heads, kvHeads, sequence, 0.125f));
            backend.Synchronize();
            long beforeAllocation = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFlashDecodeD64(
                    q, k, v, output, heads, kvHeads, sequence, 0.125f);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - beforeAllocation;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            Assert.True(backend.TryGetDirectPtxDecodeAudit(
                false, heads, kvHeads, sequence, 0, 0, 0.125f,
                out DirectPtxKernelAudit denseAudit));
            Assert.Equal(0, denseAudit.Function.LocalBytesPerThread);
            Assert.False(backend.TryDirectPtxFlashDecodeD64(
                q, k, v, output, heads, kvHeads, 48, 0.125f));
            Assert.Equal("decode-sequence-bucket-not-implemented", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxFlashDecodeD64(
                q, k, v, q, heads, kvHeads, sequence, 0.125f));
            Assert.Contains("alias", backend.DirectPtxLastError!, StringComparison.OrdinalIgnoreCase);

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxFlashDecodeD64(
                    q, k, v, output, heads, kvHeads, sequence, 0.125f));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }

            long beforePublicDense = backend.DirectPtxDecodeDispatchCount;
            using IGpuBuffer publicDense = backend.FlashDecode(
                q, k, v, heads, kvHeads, dimension, sequence, 0.125f);
            Assert.Equal(beforePublicDense + 1, backend.DirectPtxDecodeDispatchCount);
            float[] expectedDense = DecodeOracle(
                qHost, denseK, denseV, heads, kvHeads, sequence, 0, null, 0.125f);
            AssertDecodeClose(backend.DownloadBuffer(publicDense), expectedDense);
            long beforeExplicitSplit = backend.DirectPtxDecodeDispatchCount;
            using IGpuBuffer publicSplitFallback = backend.FlashDecode(
                q, k, v, heads, kvHeads, dimension, sequence, 0.125f, splits: 2);
            Assert.Equal(beforeExplicitSplit, backend.DirectPtxDecodeDispatchCount);
            AssertDecodeClose(backend.DownloadBuffer(publicSplitFallback), expectedDense);

            float[] pagedK = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] pagedV = Enumerable.Range(0, poolBlocks * blockSize * kvHeads * dimension)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            int[] tableHost = [3, 1];
            using var keyPages = backend.AllocateBuffer(pagedK);
            using var valuePages = backend.AllocateBuffer(pagedV);
            using var table = backend.AllocateIntBuffer(tableHost);
            using var pagedOutput = backend.AllocateBuffer(qHost.Length);
            Assert.False(backend.TryDirectPtxPagedDecodeD64(
                q, keyPages, valuePages, table, pagedOutput,
                heads, kvHeads, 8, sequence, 0.125f));
            Assert.Equal("paged-decode-block-size-not-implemented", backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxDecodeD64(
                true, heads, kvHeads, sequence, blockSize, poolBlocks, 0.125f),
                backend.DirectPtxLastError);
            bool pagedCaptureLaunch = true;
            IntPtr pagedGraph = backend.CaptureGraph(() =>
                pagedCaptureLaunch &= backend.TryDirectPtxPagedDecodeD64(
                    q, keyPages, valuePages, table, pagedOutput,
                    heads, kvHeads, blockSize, sequence, 0.125f));
            Assert.True(pagedCaptureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(pagedGraph); }
            finally { backend.DestroyCapturedGraph(pagedGraph); }
            long beforePublicPaged = backend.DirectPtxDecodeDispatchCount;
            using IGpuBuffer publicPaged = backend.PagedAttentionDecodeGqa(
                q, keyPages, valuePages, table,
                heads, kvHeads, dimension, blockSize, sequence, 0.125f);
            Assert.Equal(beforePublicPaged + 1, backend.DirectPtxDecodeDispatchCount);
            float[] expectedPaged = DecodeOracle(
                qHost, pagedK, pagedV, heads, kvHeads, sequence,
                blockSize, tableHost, 0.125f);
            AssertDecodeClose(backend.DownloadBuffer(publicPaged), expectedPaged);
            Assert.True(backend.TryGetDirectPtxDecodeAudit(
                true, heads, kvHeads, sequence, blockSize, poolBlocks, 0.125f,
                out DirectPtxKernelAudit pagedAudit));
            Assert.Equal(0, pagedAudit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void AttentionBackwardEmitter_IsDeterministicScratchFreeAndStrideFree()
    {
        string ptx = PtxFusedAttentionBackwardD64Kernel.EmitPtx(
            8, 6, 1, 8, 2, 16, 32, 0.125f);

        Assert.Contains(PtxFusedAttentionBackwardD64Kernel.GradQueryEntryPoint, ptx);
        Assert.Contains(PtxFusedAttentionBackwardD64Kernel.GradKeyValueEntryPoint, ptx);
        Assert.Contains(PtxFusedAttentionBackwardD64Kernel.RowDeltaEntryPoint, ptx);
        Assert.Contains("grad_query_workspace_ptr", ptx);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        string biasPtx = PtxFlashAttentionBackwardD64Kernel.EmitPtx(
            8, 6, 2, 4, 16, 32, 0.125f, isCausal: false,
            biasBatchStride: 4 * 16 * 32);
        Assert.Contains("attention_bias_ptr", biasPtx);
        Assert.DoesNotContain("has_bias", biasPtx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedAttentionBackwardD64Kernel.EmitPtx(
                8, 6, 1, 8, 3, 16, 16, 0.125f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedAttentionBackwardD64Kernel.EmitPtx(
                8, 6, 1, 8, 2, 8, 16, 0.125f));
    }

    [SkippableTheory]
    [InlineData(4, 4, 16, 16)]
    [InlineData(4, 2, 16, 32)]
    [InlineData(4, 1, 32, 16)]
    public void DriverOnlyAttentionBackwardD64_MatchesOracleAndHasZeroLocalBytes(
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in attention-backward specialization is validated on Ampere.");
        const int batch = 1, dimension = 64;
        const float scale = 0.125f;
        using var kernel = new PtxFusedAttentionBackwardD64Kernel(
            runtime, batch, queryHeads, keyValueHeads,
            querySequence, keyValueSequence, scale);
        Assert.Equal(0, kernel.GradQueryFunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.GradKeyValueFunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.RowDeltaFunctionInfo.LocalBytesPerThread);
        Assert.Equal(0, kernel.GradQueryFunctionInfo.StaticSharedBytes);
        Assert.Equal(0, kernel.GradKeyValueFunctionInfo.StaticSharedBytes);
        Assert.Equal(0, kernel.RowDeltaFunctionInfo.StaticSharedBytes);

        int queryElements = batch * queryHeads * querySequence * dimension;
        int keyValueElements = batch * keyValueHeads * keyValueSequence * dimension;
        int probabilityElements = batch * queryHeads * querySequence * keyValueSequence;
        var random = new Random(
            20260800 + queryHeads * 100 + keyValueHeads * 10 + querySequence + keyValueSequence);
        float[] gradOutputHost = Enumerable.Range(0, queryElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] queryHost = Enumerable.Range(0, queryElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] keyHost = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] valueHost = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] probabilitiesHost = AttentionProbabilities(
            queryHost, keyHost, batch, queryHeads, keyValueHeads,
            querySequence, keyValueSequence, scale);
        (float[] expectedQ, float[] expectedK, float[] expectedV) = AttentionBackwardOracle(
            gradOutputHost, queryHost, keyHost, valueHost, probabilitiesHost,
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale);

        using var gradOutput = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var query = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var key = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var value = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var probabilities = runtime.AllocateBytes((nuint)(probabilityElements * sizeof(float)));
        using var gradQuery = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var gradKey = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var gradValue = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        gradOutput.Upload<float>(gradOutputHost);
        query.Upload<float>(queryHost);
        key.Upload<float>(keyHost);
        value.Upload<float>(valueHost);
        probabilities.Upload<float>(probabilitiesHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(probabilities, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[7]));
        runtime.Synchronize();
        var actualQ = new float[queryElements];
        var actualK = new float[keyValueElements];
        var actualV = new float[keyValueElements];
        gradQuery.Download<float>(actualQ);
        gradKey.Download<float>(actualK);
        gradValue.Download<float>(actualV);
        AssertAttentionGradientClose(actualQ, expectedQ, "dQ");
        AssertAttentionGradientClose(actualK, expectedK, "dK");
        AssertAttentionGradientClose(actualV, expectedV, "dV");
    }

    [SkippableFact]
    public void BackendAttentionBackwardD64_IsZeroAllocationCapturableAndRoutesPublicGqa()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxAttentionBackwardEnabled,
                "Requires an Ampere CUDA backend.");
            const int batch = 1, queryHeads = 4, keyValueHeads = 2;
            const int querySequence = 16, keyValueSequence = 32, dimension = 64;
            const float scale = 0.125f;
            int queryElements = batch * queryHeads * querySequence * dimension;
            int keyValueElements = batch * keyValueHeads * keyValueSequence * dimension;
            var random = new Random(20260817);
            float[] gradOutputHost = Enumerable.Range(0, queryElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] queryHost = Enumerable.Range(0, queryElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] keyHost = Enumerable.Range(0, keyValueElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] valueHost = Enumerable.Range(0, keyValueElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] probabilitiesHost = AttentionProbabilities(
                queryHost, keyHost, batch, queryHeads, keyValueHeads,
                querySequence, keyValueSequence, scale);
            (float[] expectedQ, float[] expectedK, float[] expectedV) = AttentionBackwardOracle(
                gradOutputHost, queryHost, keyHost, valueHost, probabilitiesHost,
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale);

            using var gradOutput = backend.AllocateBuffer(gradOutputHost);
            using var query = backend.AllocateBuffer(queryHost);
            using var key = backend.AllocateBuffer(keyHost);
            using var value = backend.AllocateBuffer(valueHost);
            using var probabilities = backend.AllocateBuffer(probabilitiesHost);
            using var gradQuery = backend.AllocateBuffer(queryElements);
            using var gradKey = backend.AllocateBuffer(keyValueElements);
            using var gradValue = backend.AllocateBuffer(keyValueElements);
            Assert.False(backend.TryDirectPtxAttentionBackwardD64(
                gradOutput, query, key, value, probabilities,
                gradQuery, gradKey, gradValue,
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                32, scale));
            Assert.Equal("attention-backward-head-dimension-not-implemented", backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxAttentionBackwardD64(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale),
                backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxAttentionBackwardD64(
                    gradOutput, query, key, value, probabilities,
                    gradQuery, gradKey, gradValue,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    dimension, scale), backend.DirectPtxLastError);
            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxAttentionBackwardD64(
                    gradOutput, query, key, value, probabilities,
                    gradQuery, gradKey, gradValue,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    dimension, scale);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertAttentionGradientClose(backend.DownloadBuffer(gradQuery), expectedQ, "dQ");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradKey), expectedK, "dK");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradValue), expectedV, "dV");

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxAttentionBackwardD64(
                    gradOutput, query, key, value, probabilities,
                    gradQuery, gradKey, gradValue,
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    dimension, scale));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }

            using var publicGradQuery = backend.AllocateBuffer(queryElements);
            using var publicGradKey = backend.AllocateBuffer(keyValueElements);
            using var publicGradValue = backend.AllocateBuffer(keyValueElements);
            long dispatchBefore = backend.DirectPtxAttentionBackwardDispatchCount;
            backend.GroupedQueryAttentionBackward(
                gradOutput, query, key, value, probabilities,
                publicGradQuery, publicGradKey, publicGradValue,
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                dimension, scale, queryHeads / keyValueHeads);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxAttentionBackwardDispatchCount);
            AssertAttentionGradientClose(backend.DownloadBuffer(publicGradQuery), expectedQ, "public dQ");
            AssertAttentionGradientClose(backend.DownloadBuffer(publicGradKey), expectedK, "public dK");
            AssertAttentionGradientClose(backend.DownloadBuffer(publicGradValue), expectedV, "public dV");
            Assert.True(backend.TryGetDirectPtxAttentionBackwardAudits(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence, scale,
                out DirectPtxKernelAudit rowDeltaAudit,
                out DirectPtxKernelAudit gradQueryAudit,
                out DirectPtxKernelAudit gradKeyValueAudit));
            Assert.Equal(0, rowDeltaAudit.Function.LocalBytesPerThread);
            Assert.Equal(0, gradQueryAudit.Function.LocalBytesPerThread);
            Assert.Equal(0, gradKeyValueAudit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void FlashAttentionBackwardEmitter_RecomputesWithoutScratchAtomicsOrStrides()
    {
        string ptx = PtxFlashAttentionBackwardD64Kernel.EmitPtx(
            8, 6, 1, 4, 16, 32, 0.125f, isCausal: true);

        Assert.Contains(PtxFlashAttentionBackwardD64Kernel.GradQueryEntryPoint, ptx);
        Assert.Contains(PtxFlashAttentionBackwardD64Kernel.GradKeyValueEntryPoint, ptx);
        Assert.Contains("softmax_stats_ptr", ptx);
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.DoesNotContain("probabilities_ptr", ptx);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFlashAttentionBackwardD64Kernel.EmitPtx(
                8, 6, 1, 4, 8, 16, 0.125f, isCausal: false));
    }

    [SkippableTheory]
    [InlineData(4, 16, 16, false, false)]
    [InlineData(4, 16, 32, true, false)]
    [InlineData(4, 32, 16, false, false)]
    [InlineData(4, 16, 32, false, true)]
    public void DriverOnlyFlashAttentionBackwardD64_MatchesOracleAndHasZeroLocalBytes(
        int heads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        bool hasBias)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in FlashAttention-backward specialization is validated on Ampere.");
        const int batch = 1, dimension = 64;
        const float scale = 0.125f;
        int biasBatchStride = hasBias ? 0 : -1;
        using var kernel = new PtxFlashAttentionBackwardD64Kernel(
            runtime, batch, heads, querySequence, keyValueSequence, scale, isCausal,
            biasBatchStride);
        Assert.Equal(0, kernel.GradQueryAudit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.GradKeyValueAudit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.GradQueryAudit.Function.StaticSharedBytes);
        Assert.Equal(0, kernel.GradKeyValueAudit.Function.StaticSharedBytes);

        int queryElements = batch * heads * querySequence * dimension;
        int keyValueElements = batch * heads * keyValueSequence * dimension;
        var random = new Random(
            20260820 + heads * 100 + querySequence * 10 + keyValueSequence + (isCausal ? 1 : 0));
        float[] gradOutputHost = Enumerable.Range(0, queryElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] queryHost = Enumerable.Range(0, queryElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] keyHost = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[] valueHost = Enumerable.Range(0, keyValueElements)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
        float[]? biasHost = hasBias
            ? Enumerable.Range(0, heads * querySequence * keyValueSequence)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.25f).ToArray()
            : null;
        (float[] probabilities, float[] outputHost, float[] statsHost) = FlashForwardReference(
            queryHost, keyHost, valueHost, batch, heads,
            querySequence, keyValueSequence, scale, isCausal, biasHost);
        (float[] expectedQ, float[] expectedK, float[] expectedV) = AttentionBackwardOracle(
            gradOutputHost, queryHost, keyHost, valueHost, probabilities,
            batch, heads, heads, querySequence, keyValueSequence, scale);

        using var gradOutput = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var query = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var key = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var value = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var stats = runtime.AllocateBytes((nuint)(statsHost.Length * sizeof(float)));
        using var gradQuery = runtime.AllocateBytes((nuint)(queryElements * sizeof(float)));
        using var gradKey = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using var gradValue = runtime.AllocateBytes((nuint)(keyValueElements * sizeof(float)));
        using DirectPtxBuffer? bias = hasBias
            ? runtime.AllocateBytes((nuint)(biasHost!.Length * sizeof(float)))
            : null;
        gradOutput.Upload<float>(gradOutputHost);
        query.Upload<float>(queryHost);
        key.Upload<float>(keyHost);
        value.Upload<float>(valueHost);
        output.Upload<float>(outputHost);
        stats.Upload<float>(statsHost);
        bias?.Upload<float>(biasHost!);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(stats, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[7]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[8]),
            bias is null
                ? null
                : DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[9]));
        runtime.Synchronize();
        var actualQ = new float[queryElements];
        var actualK = new float[keyValueElements];
        var actualV = new float[keyValueElements];
        gradQuery.Download<float>(actualQ);
        gradKey.Download<float>(actualK);
        gradValue.Download<float>(actualV);
        AssertAttentionGradientClose(actualQ, expectedQ, "Flash dQ");
        AssertAttentionGradientClose(actualK, expectedK, "Flash dK");
        AssertAttentionGradientClose(actualV, expectedV, "Flash dV");
    }

    [SkippableFact]
    public void BackendFlashAttentionBackwardD64_IsZeroAllocationCapturableAndRoutesPublicApi()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFlashAttentionBackwardEnabled,
                "Requires an Ampere CUDA backend.");
            const int batch = 1, heads = 4, querySequence = 16, keyValueSequence = 32;
            const int dimension = 64;
            const float scale = 0.125f;
            const bool isCausal = true;
            int queryElements = batch * heads * querySequence * dimension;
            int keyValueElements = batch * heads * keyValueSequence * dimension;
            var random = new Random(20260821);
            float[] gradOutputHost = Enumerable.Range(0, queryElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] queryHost = Enumerable.Range(0, queryElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] keyHost = Enumerable.Range(0, keyValueElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            float[] valueHost = Enumerable.Range(0, keyValueElements)
                .Select(_ => (random.NextSingle() - 0.5f) * 0.5f).ToArray();
            (float[] probabilities, float[] outputHost, float[] statsHost) = FlashForwardReference(
                queryHost, keyHost, valueHost, batch, heads,
                querySequence, keyValueSequence, scale, isCausal);
            (float[] expectedQ, float[] expectedK, float[] expectedV) = AttentionBackwardOracle(
                gradOutputHost, queryHost, keyHost, valueHost, probabilities,
                batch, heads, heads, querySequence, keyValueSequence, scale);

            using var gradOutput = backend.AllocateBuffer(gradOutputHost);
            using var query = backend.AllocateBuffer(queryHost);
            using var key = backend.AllocateBuffer(keyHost);
            using var value = backend.AllocateBuffer(valueHost);
            using var output = backend.AllocateBuffer(outputHost);
            using var stats = backend.AllocateBuffer(statsHost);
            using var gradQuery = backend.AllocateBuffer(queryElements);
            using var gradKey = backend.AllocateBuffer(keyValueElements);
            using var gradValue = backend.AllocateBuffer(keyValueElements);
            using var badBias = backend.AllocateBuffer(
                batch * heads * querySequence * keyValueSequence + 1);
            Assert.False(backend.TryDirectPtxFlashAttentionBackwardD64(
                gradOutput, query, key, value, output, stats,
                gradQuery, gradKey, gradValue,
                batch, heads, querySequence, keyValueSequence,
                dimension, scale, isCausal, badBias));
            Assert.Equal(
                "flash-attention-backward-bias-physical-extent-mismatch",
                backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxFlashAttentionBackwardD64(
                batch, heads, querySequence, keyValueSequence, scale, isCausal),
                backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, attentionBias: null),
                    backend.DirectPtxLastError);

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, attentionBias: null);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertAttentionGradientClose(backend.DownloadBuffer(gradQuery), expectedQ, "backend Flash dQ");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradKey), expectedK, "backend Flash dK");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradValue), expectedV, "backend Flash dV");

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, attentionBias: null));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }

            using var publicGradQuery = backend.AllocateBuffer(queryElements);
            using var publicGradKey = backend.AllocateBuffer(keyValueElements);
            using var publicGradValue = backend.AllocateBuffer(keyValueElements);
            long dispatchBefore = backend.DirectPtxFlashAttentionBackwardDispatchCount;
            backend.FlashAttentionBackward(
                gradOutput, query, key, value, output, stats,
                publicGradQuery, publicGradKey, publicGradValue,
                batch, heads, querySequence, keyValueSequence,
                dimension, scale, isCausal);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxFlashAttentionBackwardDispatchCount);
            AssertAttentionGradientClose(
                backend.DownloadBuffer(publicGradQuery), expectedQ, "public Flash dQ");
            AssertAttentionGradientClose(
                backend.DownloadBuffer(publicGradKey), expectedK, "public Flash dK");
            AssertAttentionGradientClose(
                backend.DownloadBuffer(publicGradValue), expectedV, "public Flash dV");
            Assert.True(backend.TryGetDirectPtxFlashAttentionBackwardAudits(
                batch, heads, querySequence, keyValueSequence, scale, isCausal,
                out DirectPtxKernelAudit gradQueryAudit,
                out DirectPtxKernelAudit gradKeyValueAudit));
            Assert.Equal(0, gradQueryAudit.Function.LocalBytesPerThread);
            Assert.Equal(0, gradKeyValueAudit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(4 * 16 * 32)]
    public void BackendFlashAttentionBackwardBias_BroadcastAndPerBatchMatchOracle(
        int biasBatchStride)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFlashAttentionBackwardEnabled,
                "Requires an Ampere CUDA backend.");
            const int batch = 2, heads = 4, querySequence = 16, keyValueSequence = 32;
            const int dimension = 64;
            const float scale = 0.125f;
            const bool isCausal = false;
            int queryElements = batch * heads * querySequence * dimension;
            int keyValueElements = batch * heads * keyValueSequence * dimension;
            int biasElements = biasBatchStride == 0
                ? heads * querySequence * keyValueSequence
                : batch * biasBatchStride;
            var random = new Random(20260822 + biasBatchStride);
            float[] gradOutputHost = Values(random, queryElements);
            float[] queryHost = Values(random, queryElements);
            float[] keyHost = Values(random, keyValueElements);
            float[] valueHost = Values(random, keyValueElements);
            float[] biasHost = Values(random, biasElements);
            (float[] probabilities, float[] outputHost, float[] statsHost) =
                FlashForwardReference(
                    queryHost, keyHost, valueHost, batch, heads,
                    querySequence, keyValueSequence, scale, isCausal,
                    biasHost, biasBatchStride);
            (float[] expectedQ, float[] expectedK, float[] expectedV) = AttentionBackwardOracle(
                gradOutputHost, queryHost, keyHost, valueHost, probabilities,
                batch, heads, heads, querySequence, keyValueSequence, scale);

            using var gradOutput = backend.AllocateBuffer(gradOutputHost);
            using var query = backend.AllocateBuffer(queryHost);
            using var key = backend.AllocateBuffer(keyHost);
            using var value = backend.AllocateBuffer(valueHost);
            using var output = backend.AllocateBuffer(outputHost);
            using var stats = backend.AllocateBuffer(statsHost);
            using var bias = backend.AllocateBuffer(biasHost);
            using var gradQuery = backend.AllocateBuffer(queryElements);
            using var gradKey = backend.AllocateBuffer(keyValueElements);
            using var gradValue = backend.AllocateBuffer(keyValueElements);

            Assert.True(backend.PrewarmDirectPtxFlashAttentionBackwardD64(
                batch, heads, querySequence, keyValueSequence, scale, isCausal,
                biasBatchStride), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, bias, biasBatchStride),
                    backend.DirectPtxLastError);
            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, bias, biasBatchStride);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertAttentionGradientClose(backend.DownloadBuffer(gradQuery), expectedQ, "biased dQ");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradKey), expectedK, "biased dK");
            AssertAttentionGradientClose(backend.DownloadBuffer(gradValue), expectedV, "biased dV");

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxFlashAttentionBackwardD64(
                    gradOutput, query, key, value, output, stats,
                    gradQuery, gradKey, gradValue,
                    batch, heads, querySequence, keyValueSequence,
                    dimension, scale, isCausal, bias, biasBatchStride));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }

            long dispatchBefore = backend.DirectPtxFlashAttentionBackwardDispatchCount;
            backend.FlashAttentionBackward(
                gradOutput, query, key, value, output, stats,
                gradQuery, gradKey, gradValue,
                batch, heads, querySequence, keyValueSequence,
                dimension, scale, isCausal, bias, biasBatchStride);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxFlashAttentionBackwardDispatchCount);
            Assert.True(backend.TryGetDirectPtxFlashAttentionBackwardAudits(
                batch, heads, querySequence, keyValueSequence, scale, isCausal,
                out DirectPtxKernelAudit gradQueryAudit,
                out DirectPtxKernelAudit gradKeyValueAudit,
                biasBatchStride));
            Assert.Equal(0, gradQueryAudit.Function.LocalBytesPerThread);
            Assert.Equal(0, gradKeyValueAudit.Function.LocalBytesPerThread);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void QkvRopeCacheEmitter_BakesCanonicalAbiAndEliminatesIntermediates()
    {
        string ptx = PtxFusedQkvRopeCacheD64Kernel.EmitPtx(
            8, 6, heads: 8, cacheCapacity: 64, position: 17);

        Assert.Contains(PtxFusedQkvRopeCacheD64Kernel.EntryPoint, ptx);
        Assert.Contains(".target sm_86", ptx);
        Assert.Contains("st.global.v2.f32 [%rd20]", ptx);
        Assert.Contains("st.global.v2.f32 [%rd22]", ptx);
        Assert.Contains("st.global.v2.f32 [%rd23]", ptx);
        Assert.Contains("shfl.sync.bfly.b32", ptx);
        Assert.Contains("@%p1 bra QKV_RETURN;", ptx);
        Assert.DoesNotContain("@%p1 bra.uni", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("shape", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("q_projection_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("k_projection_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("v_projection_ptr", ptx, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedQkvRopeCacheD64Kernel.EmitPtx(8, 6, 6, 64, 17));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedQkvRopeCacheD64Kernel.EmitPtx(8, 6, 8, 24, 17));
    }

    [Theory]
    [InlineData(8, 6, true)]
    [InlineData(8, 0, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, false)]
    [InlineData(9, 0, false)]
    [InlineData(10, 0, false)]
    public void QkvRopeCacheArchitectureMatrix_FailsClosedOutsideSm86(
        int major,
        int minor,
        bool expected)
    {
        Assert.Equal(expected,
            DirectPtxArchitecture.HasValidatedQkvRopeCache(major, minor));
    }

    [Fact]
    public void QkvRopeCacheCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        string[] expected =
        [
            "CudaBackend.QkvProjectionRoPECacheD64",
            "IDirectGpuBackend.MatMulTransposed",
            "DirectGpuTensorEngine.ApplyRoPEInterleavedGpu",
            "IEngine.ApplyRoPEInterleaved",
            "KVCache<T>.Append",
            "DevicePagedKVCache.Append",
            "Tensor reshape/transpose/slice QKV handoff",
            "RecordingGpuBackend.RopeInterleaved",
            "Attention ABI handoff"
        ];
        string[] actual = DirectPtxQkvRopeCacheCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(expected.OrderBy(name => name, StringComparer.Ordinal).ToArray(), actual);
        Assert.Equal(actual.Length, actual.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxQkvRopeCacheCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxQkvRopeCacheCoverageManifest.Get("UnassignedQkvApi"));
    }

    [SkippableTheory]
    [InlineData(4, 16, 0)]
    [InlineData(8, 64, 17)]
    [InlineData(16, 128, 127)]
    public void DriverOnlyQkvRopeCacheD64_MatchesOracleAndHasZeroSpills(
        int heads,
        int cacheCapacity,
        int position)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedQkvRopeCache(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in QKV/RoPE/cache specialization is validated on SM86.");
        using var kernel = new PtxFusedQkvRopeCacheD64Kernel(
            runtime, heads, cacheCapacity, position);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 8);

        int modelDimension = heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
        int projectionElements = 3 * modelDimension;
        int cacheElements = cacheCapacity * modelDimension;
        var random = new Random(20260830 + heads * 100 + cacheCapacity + position);
        float[] inputHost = Values(random, modelDimension, 0.125f);
        float[] weightsHost = Values(random, projectionElements * modelDimension, 0.0625f);
        float[] biasHost = Values(random, projectionElements, 0.0625f);
        (float[] cosineHost, float[] sineHost) = RopeTables(cacheCapacity);
        float[] keyCacheHost = Values(random, cacheElements, 0.125f);
        float[] valueCacheHost = Values(random, cacheElements, 0.125f);
        (float[] expectedQ, float[] expectedK, float[] expectedV) = QkvRopeCacheOracle(
            inputHost, weightsHost, biasHost, cosineHost, sineHost,
            keyCacheHost, valueCacheHost, heads, cacheCapacity, position);

        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(weightsHost.Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var cosine = runtime.AllocateBytes((nuint)(cosineHost.Length * sizeof(float)));
        using var sine = runtime.AllocateBytes((nuint)(sineHost.Length * sizeof(float)));
        using var query = runtime.AllocateBytes((nuint)(modelDimension * sizeof(float)));
        using var keyCache = runtime.AllocateBytes((nuint)(cacheElements * sizeof(float)));
        using var valueCache = runtime.AllocateBytes((nuint)(cacheElements * sizeof(float)));
        input.Upload<float>(inputHost);
        weights.Upload<float>(weightsHost);
        bias.Upload<float>(biasHost);
        cosine.Upload<float>(cosineHost);
        sine.Upload<float>(sineHost);
        keyCache.Upload<float>(keyCacheHost);
        valueCache.Upload<float>(valueCacheHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(cosine, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(sine, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(keyCache, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(valueCache, kernel.Blueprint.Tensors[7]));
        runtime.Synchronize();
        var actualQ = new float[modelDimension];
        var actualK = new float[cacheElements];
        var actualV = new float[cacheElements];
        query.Download<float>(actualQ);
        keyCache.Download<float>(actualK);
        valueCache.Download<float>(actualV);
        AssertVectorClose(actualQ, expectedQ, 2e-5f, "fused Q");
        AssertVectorClose(actualK, expectedK, 2e-5f, "fused K cache");
        AssertVectorClose(actualV, expectedV, 2e-5f, "fused V cache");
    }

    [SkippableFact]
    public void BackendQkvRopeCacheD64_IsExactZeroAllocationAndCapturable()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxQkvRopeCacheEnabled,
                "Requires an Ampere CUDA backend.");
            const int heads = 4, cacheCapacity = 32, position = 11;
            int modelDimension = heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
            int projectionElements = 3 * modelDimension;
            int cacheElements = cacheCapacity * modelDimension;
            var random = new Random(20260831);
            float[] inputHost = Values(random, modelDimension, 0.125f);
            float[] weightsHost = Values(random, projectionElements * modelDimension, 0.0625f);
            float[] biasHost = Values(random, projectionElements, 0.0625f);
            (float[] cosineHost, float[] sineHost) = RopeTables(cacheCapacity);
            float[] keyCacheHost = Values(random, cacheElements, 0.125f);
            float[] valueCacheHost = Values(random, cacheElements, 0.125f);
            (float[] expectedQ, float[] expectedK, float[] expectedV) = QkvRopeCacheOracle(
                inputHost, weightsHost, biasHost, cosineHost, sineHost,
                keyCacheHost, valueCacheHost, heads, cacheCapacity, position);
            using var input = backend.AllocateBuffer(inputHost);
            using var weights = backend.AllocateBuffer(weightsHost);
            using var bias = backend.AllocateBuffer(biasHost);
            using var cosine = backend.AllocateBuffer(cosineHost);
            using var sine = backend.AllocateBuffer(sineHost);
            using var query = backend.AllocateBuffer(modelDimension);
            using var keyCache = backend.AllocateBuffer(keyCacheHost);
            using var valueCache = backend.AllocateBuffer(valueCacheHost);
            using var wrongQuery = backend.AllocateBuffer(modelDimension + 1);

            DirectPtxFeatureGate.TestOverride = false;
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-feature-disabled", backend.DirectPtxLastError);
            Assert.False(backend.PrewarmDirectPtxQkvRopeCacheD64(
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-feature-disabled", backend.DirectPtxLastError);
            long disabledDispatch = backend.DirectPtxQkvRopeCacheDispatchCount;
            backend.QkvProjectionRoPECacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position);
            backend.Synchronize();
            Assert.Equal(disabledDispatch, backend.DirectPtxQkvRopeCacheDispatchCount);
            Assert.Equal("qkv-rope-cache-feature-disabled", backend.DirectPtxLastError);
            AssertVectorClose(backend.DownloadBuffer(query), expectedQ, 2e-5f, "disabled fallback Q");
            DirectPtxFeatureGate.TestOverride = true;

            Assert.Throws<ArgumentNullException>(() => backend.QkvProjectionRoPECacheD64(
                null!, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position));

            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                6, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-head-count-not-implemented", backend.DirectPtxLastError);
            Assert.False(backend.PrewarmDirectPtxQkvRopeCacheD64(
                6, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-head-count-not-implemented", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, 24, position));
            Assert.Equal("qkv-rope-cache-capacity-not-implemented", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, -1));
            Assert.Equal("qkv-rope-cache-position-out-of-range", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, cacheCapacity));
            Assert.Equal("qkv-rope-cache-position-out-of-range", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                null!, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-null-buffer", backend.DirectPtxLastError);

            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, wrongQuery, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-physical-extent-mismatch", backend.DirectPtxLastError);
            using var zeroPointer = new SyntheticGpuBuffer(
                IntPtr.Zero, input.SizeInBytes);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                zeroPointer, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-invalid-device-pointer", backend.DirectPtxLastError);
            using var misaligned = new SyntheticGpuBuffer(
                new IntPtr(input.Handle.ToInt64() + 4), input.SizeInBytes);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                misaligned, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-alignment-mismatch", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxQkvRopeCacheD64(
                input, weights, bias, cosine, sine, input, keyCache, valueCache,
                heads, cacheCapacity, position));
            Assert.Equal("qkv-rope-cache-alias-not-supported", backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxQkvRopeCacheD64(
                heads, cacheCapacity, position), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxQkvRopeCacheD64(
                    input, weights, bias, cosine, sine, query, keyCache, valueCache,
                    heads, cacheCapacity, position), backend.DirectPtxLastError);

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxQkvRopeCacheD64(
                    input, weights, bias, cosine, sine, query, keyCache, valueCache,
                    heads, cacheCapacity, position);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            long publicBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++)
                backend.QkvProjectionRoPECacheD64(
                    input, weights, bias, cosine, sine, query, keyCache, valueCache,
                    heads, cacheCapacity, position);
            long publicAllocated = GC.GetAllocatedBytesForCurrentThread() - publicBefore;
            backend.Synchronize();
            Assert.Equal(0, publicAllocated);
            AssertVectorClose(backend.DownloadBuffer(query), expectedQ, 2e-5f, "backend Q");
            AssertVectorClose(backend.DownloadBuffer(keyCache), expectedK, 2e-5f, "backend K cache");
            AssertVectorClose(backend.DownloadBuffer(valueCache), expectedV, 2e-5f, "backend V cache");

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxQkvRopeCacheD64(
                    input, weights, bias, cosine, sine, query, keyCache, valueCache,
                    heads, cacheCapacity, position));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxQkvRopeCachePinnedKernelCount);
            try
            {
                int warmed = 0;
                foreach (int otherHeads in new[] { 4, 8, 16 })
                foreach (int otherCapacity in new[] { 16, 32, 64, 128 })
                for (int otherPosition = 0; otherPosition < otherCapacity; otherPosition++)
                {
                    if (otherHeads == heads && otherCapacity == cacheCapacity &&
                        otherPosition == position)
                        continue;
                    Assert.True(backend.PrewarmDirectPtxQkvRopeCacheD64(
                        otherHeads, otherCapacity, otherPosition), backend.DirectPtxLastError);
                    if (++warmed >= backend.DirectPtxQkvRopeCacheKernelCapacity)
                        goto CacheFilled;
                }

            CacheFilled:
                Assert.True(backend.TryGetDirectPtxQkvRopeCacheAudit(
                    heads, cacheCapacity, position, out _));
                for (int i = 0; i < 8; i++)
                    backend.EnqueueCapturedGraph(graph);
                backend.Synchronize();
                long graphBefore = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 32; i++)
                    backend.EnqueueCapturedGraph(graph);
                long graphAllocated =
                    GC.GetAllocatedBytesForCurrentThread() - graphBefore;
                backend.Synchronize();
                Assert.Equal(0, graphAllocated);
            }
            finally { backend.DestroyCapturedGraph(graph); }

            Assert.True(backend.TryGetDirectPtxQkvRopeCacheAudit(
                heads, cacheCapacity, position, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(0, audit.Function.StaticSharedBytes);
            Assert.True(audit.ActiveBlocksPerMultiprocessor >= 8);
            Assert.True(backend.DirectPtxQkvRopeCacheDispatchCount >= 41);
            long dispatchBefore = backend.DirectPtxQkvRopeCacheDispatchCount;
            backend.QkvProjectionRoPECacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxQkvRopeCacheDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(query), expectedQ, 2e-5f, "public Q");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [SkippableFact]
    public void BackendQkvRopeCacheD64_UnsupportedBucketsUseExistingGpuFallback()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxQkvRopeCacheEnabled,
                "Requires an Ampere CUDA backend.");
            const int heads = 6, cacheCapacity = 24, position = 23;
            int modelDimension = heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
            int projectionElements = 3 * modelDimension;
            int cacheElements = cacheCapacity * modelDimension;
            var random = new Random(20260901);
            float[] inputHost = Values(random, modelDimension, 0.125f);
            float[] weightsHost = Values(random, projectionElements * modelDimension, 0.0625f);
            float[] biasHost = Values(random, projectionElements, 0.0625f);
            (float[] cosineHost, float[] sineHost) = RopeTables(cacheCapacity);
            float[] keyCacheHost = Values(random, cacheElements, 0.125f);
            float[] valueCacheHost = Values(random, cacheElements, 0.125f);
            (float[] expectedQ, float[] expectedK, float[] expectedV) = QkvRopeCacheOracle(
                inputHost, weightsHost, biasHost, cosineHost, sineHost,
                keyCacheHost, valueCacheHost, heads, cacheCapacity, position);
            using var input = backend.AllocateBuffer(inputHost);
            using var weights = backend.AllocateBuffer(weightsHost);
            using var bias = backend.AllocateBuffer(biasHost);
            using var cosine = backend.AllocateBuffer(cosineHost);
            using var sine = backend.AllocateBuffer(sineHost);
            using var query = backend.AllocateBuffer(modelDimension);
            using var keyCache = backend.AllocateBuffer(keyCacheHost);
            using var valueCache = backend.AllocateBuffer(valueCacheHost);

            long dispatchBefore = backend.DirectPtxQkvRopeCacheDispatchCount;
            backend.QkvProjectionRoPECacheD64(
                input, weights, bias, cosine, sine, query, keyCache, valueCache,
                heads, cacheCapacity, position);
            backend.Synchronize();

            Assert.Equal(dispatchBefore, backend.DirectPtxQkvRopeCacheDispatchCount);
            Assert.Equal("qkv-rope-cache-head-count-not-implemented", backend.DirectPtxLastError);
            AssertVectorClose(backend.DownloadBuffer(query), expectedQ, 2e-5f, "fallback Q");
            AssertVectorClose(backend.DownloadBuffer(keyCache), expectedK, 2e-5f, "fallback K cache");
            AssertVectorClose(backend.DownloadBuffer(valueCache), expectedV, 2e-5f, "fallback V cache");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void FusedLinearGeluEmitter_BakesShapeAndHasPointerOnlyAbi()
    {
        string ptx = PtxFusedLinearGeluM1Kernel.EmitPtx(
            8, 6, inputFeatures: 512, outputFeatures: 2048);

        Assert.Contains(PtxFusedLinearGeluM1Kernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Contains("mul.wide.u32 %rd6, %r4, 2048", ptx);
        Assert.Contains("setp.lt.u32 %p0, %r5, 512", ptx);
        Assert.Contains("tanh.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("batch", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluM1Kernel.EmitPtx(8, 6, 384, 2048));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluM1Kernel.EmitPtx(8, 6, 512, 768));
        Assert.False(PtxFusedLinearGeluM1Kernel.IsPromotedShape(256, 256));
        Assert.True(PtxFusedLinearGeluM1Kernel.IsPromotedShape(512, 2048));
        Assert.False(PtxFusedLinearGeluM1Kernel.IsPromotedShape(1024, 4096));
    }

    [Fact]
    public void Fp16FusedLinearGeluEmitter_BakesHalf2AbiAndNoStrideChecks()
    {
        string ptx = PtxFusedLinearGeluFp16M1Kernel.EmitPtx(
            8, 6, inputFeatures: 512, outputFeatures: 2048);

        Assert.Contains(PtxFusedLinearGeluFp16M1Kernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Contains("mul.wide.u32 %rd6, %r4, 1024", ptx);
        Assert.Contains("ld.global.nc.b32", ptx);
        Assert.Contains("cvt.f32.f16", ptx);
        Assert.Contains("setp.lt.u32 %p0, %r5, 512", ptx);
        Assert.Contains("tanh.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("batch", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluFp16M1Kernel.EmitPtx(8, 6, 384, 2048));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluFp16M1Kernel.EmitPtx(8, 6, 512, 768));
        Assert.False(PtxFusedLinearGeluFp16M1Kernel.IsPromotedShape(256, 256));
        Assert.True(PtxFusedLinearGeluFp16M1Kernel.IsPromotedShape(512, 2048));
        Assert.True(PtxFusedLinearGeluFp16M1Kernel.IsPromotedShape(1024, 4096));
    }

    [Fact]
    public void Fp16TensorCoreFusedLinearGeluEmitter_BakesAsyncM16AbiAndNoStrideChecks()
    {
        string ptx = PtxFusedLinearGeluFp16M16Kernel.EmitPtx(
            8, 6, inputFeatures: 512, outputFeatures: 2048);

        Assert.Contains(PtxFusedLinearGeluFp16M16Kernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Equal(8, Count(ptx, "cp.async.ca.shared.global"));
        Assert.Equal(16, Count(ptx, "cp.async.cg.shared.global"));
        Assert.Equal(8, Count(ptx, "cp.async.commit_group"));
        Assert.Equal(32, Count(ptx, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        Assert.Contains(
            $"smem[{PtxFusedLinearGeluFp16M16Kernel.GetStaticSharedBytes(512)}]", ptx);
        Assert.Contains("ld.global.nc.v2.f32", ptx);
        Assert.Contains("tanh.approx.f32", ptx);
        Assert.Equal(2, Count(ptx, "st.global.v2.f32"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("batch", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluFp16M16Kernel.EmitPtx(8, 6, 256, 2048));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedLinearGeluFp16M16Kernel.EmitPtx(8, 6, 512, 768));
        Assert.True(PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(512, 2048));
        Assert.True(PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(1024, 4096));
    }

    [SkippableFact]
    public void DriverOnlyFp16TensorCoreFusedLinearGeluM16_MatchesOracleAndHasZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in FP16 Tensor Core fused-linear specialization is validated on Ampere.");
        const int rows = PtxFusedLinearGeluFp16M16Kernel.Rows;
        const int inputFeatures = 512, outputFeatures = 2048;
        using var kernel = new PtxFusedLinearGeluFp16M16Kernel(
            runtime, inputFeatures, outputFeatures);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxFusedLinearGeluFp16M16Kernel.GetStaticSharedBytes(inputFeatures),
            kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 4);
        Assert.True(kernel.Audit.Function.RegistersPerThread <= 48);
        Assert.Equal(DirectPtxPhysicalLayout.RowMajor2D, kernel.Blueprint.Tensors[0].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.LinearWeightOutputMajor,
            kernel.Blueprint.Tensors[1].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.RowMajor2D, kernel.Blueprint.Tensors[3].Layout);

        var random = new Random(20260872);
        ushort[] inputBits = RandomHalfRange(random, rows * inputFeatures, 0.125f);
        ushort[] weightBits = RandomHalfRange(
            random, inputFeatures * outputFeatures, 0.0625f);
        float[] inputHost = inputBits.Select(Half).ToArray();
        float[] weightHost = weightBits.Select(Half).ToArray();
        float[] biasHost = Values(random, outputFeatures, 0.0625f);
        float[] expected = FusedLinearGeluOracle(
            inputHost, weightHost, biasHost, rows, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes((nuint)(inputBits.Length * sizeof(ushort)));
        using var weights = runtime.AllocateBytes((nuint)(weightBits.Length * sizeof(ushort)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(rows * outputFeatures * sizeof(float)));
        input.Upload<ushort>(inputBits);
        weights.Upload<ushort>(weightBits);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[rows * outputFeatures];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f,
            "direct FP16 Tensor Core fused linear GELU M16");
    }

    [SkippableFact]
    public void BackendFp16TensorCoreFusedLinearGeluM16_IsZeroAllocationCapturableAndFailClosed()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride = false;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxMixedLinearEnabled, "Requires an Ampere CUDA backend.");
            const int rows = PtxFusedLinearGeluFp16M16Kernel.Rows;
            const int inputFeatures = 512, outputFeatures = 2048;
            var random = new Random(20260873);
            ushort[] inputBits = RandomHalfRange(random, rows * inputFeatures, 0.125f);
            ushort[] weightBits = RandomHalfRange(
                random, inputFeatures * outputFeatures, 0.0625f);
            float[] inputHost = inputBits.Select(Half).ToArray();
            float[] weightHost = weightBits.Select(Half).ToArray();
            float[] biasHost = Values(random, outputFeatures, 0.0625f);
            float[] expected = FusedLinearGeluOracle(
                inputHost, weightHost, biasHost, rows, inputFeatures, outputFeatures);
            using var inputFloat = backend.AllocateBuffer(inputHost);
            using var weightFloat = backend.AllocateBuffer(weightHost);
            using var inputHalf = backend.AllocateByteBuffer(
                rows * inputFeatures * sizeof(ushort));
            using var weightHalf = backend.AllocateByteBuffer(
                inputFeatures * outputFeatures * sizeof(ushort));
            using var bias = backend.AllocateBuffer(biasHost);
            using var output = backend.AllocateBuffer(rows * outputFeatures);
            using var wrongOutput = backend.AllocateBuffer(rows * outputFeatures + 1);
            backend.ConvertToFp16Native(inputFloat, inputHalf, rows * inputFeatures);
            backend.ConvertToFp16Native(
                weightFloat, weightHalf, inputFeatures * outputFeatures);

            Assert.False(backend.TryDirectPtxFusedLinearGeluFp16M16(
                inputHalf, weightHalf, bias, output, 256, 2048));
            Assert.Equal("mixed-linear-m16-shape-not-implemented", backend.DirectPtxLastError);
            DirectPtxFeatureGate.TestOverride = false;
            long fallbackDispatchBefore = backend.DirectPtxMixedLinearDispatchCount;
            backend.FusedLinearGELUFp16TransposedM16(
                inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            backend.Synchronize();
            Assert.Equal(fallbackDispatchBefore, backend.DirectPtxMixedLinearDispatchCount);
            AssertVectorClose(
                backend.DownloadBuffer(output), expected, 2e-3f,
                "cuBLAS fallback FP16 Tensor Core fused linear GELU M16");

            DirectPtxFeatureGate.TestOverride = true;
            Assert.False(backend.TryDirectPtxFusedLinearGeluFp16M16(
                inputHalf, weightHalf, bias, wrongOutput, inputFeatures, outputFeatures));
            Assert.Equal("mixed-linear-m16-physical-extent-mismatch", backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxFusedLinearGeluFp16M16(
                inputFeatures, outputFeatures), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFusedLinearGeluFp16M16(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures),
                    backend.DirectPtxLastError);

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFusedLinearGeluFp16M16(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertVectorClose(
                backend.DownloadBuffer(output), expected, 2e-3f,
                "backend FP16 Tensor Core fused linear GELU M16");

            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxFusedLinearGeluFp16M16(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures));
            Assert.True(captured, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
            Assert.True(backend.TryGetDirectPtxMixedLinearM16Audit(
                inputFeatures, outputFeatures, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);

            long dispatchBefore = backend.DirectPtxMixedLinearDispatchCount;
            backend.FusedLinearGELUFp16TransposedM16(
                inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxMixedLinearDispatchCount);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void DriverOnlyFp16FusedLinearGeluM1_MatchesRoundedOracleAndHasZeroLocalBytes()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in FP16 fused-linear specialization is validated on Ampere.");
        const int inputFeatures = 512, outputFeatures = 2048;
        using var kernel = new PtxFusedLinearGeluFp16M1Kernel(
            runtime, inputFeatures, outputFeatures);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 8);
        Assert.Equal(DirectPtxPhysicalType.Float16, kernel.Blueprint.Tensors[0].PhysicalType);
        Assert.Equal(DirectPtxPhysicalType.Float16, kernel.Blueprint.Tensors[1].PhysicalType);
        Assert.Equal(DirectPtxPhysicalType.Float32, kernel.Blueprint.Tensors[3].PhysicalType);

        var random = new Random(20260870);
        ushort[] inputBits = RandomHalfRange(random, inputFeatures, 0.125f);
        ushort[] weightBits = RandomHalfRange(
            random, inputFeatures * outputFeatures, 0.0625f);
        float[] inputHost = inputBits.Select(Half).ToArray();
        float[] weightHost = weightBits.Select(Half).ToArray();
        float[] biasHost = Values(random, outputFeatures, 0.0625f);
        float[] expected = FusedLinearGeluOracle(
            inputHost, weightHost, biasHost, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes((nuint)(inputBits.Length * sizeof(ushort)));
        using var weights = runtime.AllocateBytes((nuint)(weightBits.Length * sizeof(ushort)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(outputFeatures * sizeof(float)));
        input.Upload<ushort>(inputBits);
        weights.Upload<ushort>(weightBits);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[outputFeatures];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 4e-4f, "direct FP16 fused linear GELU");
    }

    [SkippableFact]
    public void BackendFp16FusedLinearGeluM1_IsZeroAllocationCapturableAndFailClosed()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxMixedLinearEnabled, "Requires an Ampere CUDA backend.");
            const int inputFeatures = 512, outputFeatures = 2048;
            var random = new Random(20260871);
            ushort[] inputBits = RandomHalfRange(random, inputFeatures, 0.125f);
            ushort[] weightBits = RandomHalfRange(
                random, inputFeatures * outputFeatures, 0.0625f);
            float[] inputHost = inputBits.Select(Half).ToArray();
            float[] weightHost = weightBits.Select(Half).ToArray();
            float[] biasHost = Values(random, outputFeatures, 0.0625f);
            float[] expected = FusedLinearGeluOracle(
                inputHost, weightHost, biasHost, inputFeatures, outputFeatures);
            using var inputFloat = backend.AllocateBuffer(inputHost);
            using var weightFloat = backend.AllocateBuffer(weightHost);
            using var inputHalf = backend.AllocateByteBuffer(inputFeatures * sizeof(ushort));
            using var weightHalf = backend.AllocateByteBuffer(
                inputFeatures * outputFeatures * sizeof(ushort));
            using var bias = backend.AllocateBuffer(biasHost);
            using var output = backend.AllocateBuffer(outputFeatures);
            using var wrongOutput = backend.AllocateBuffer(outputFeatures + 1);
            backend.ConvertToFp16Native(inputFloat, inputHalf, inputFeatures);
            backend.ConvertToFp16Native(
                weightFloat, weightHalf, inputFeatures * outputFeatures);

            DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride = false;
            Assert.False(backend.TryDirectPtxFusedLinearGeluFp16M1(
                inputHalf, weightHalf, bias, output, 256, 256));
            Assert.Equal("mixed-linear-performance-gate-not-met", backend.DirectPtxLastError);
            DirectPtxFeatureGate.TestOverride = false;
            long fallbackDispatchBefore = backend.DirectPtxMixedLinearDispatchCount;
            backend.FusedLinearGELUFp16TransposedM1(
                inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            backend.Synchronize();
            Assert.Equal(fallbackDispatchBefore, backend.DirectPtxMixedLinearDispatchCount);
            AssertVectorClose(
                backend.DownloadBuffer(output), expected, 4e-4f,
                "cuBLAS fallback FP16 fused linear GELU");
            DirectPtxFeatureGate.TestOverride = true;
            DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride = true;
            Assert.False(backend.TryDirectPtxFusedLinearGeluFp16M1(
                inputHalf, weightHalf, bias, wrongOutput, inputFeatures, outputFeatures));
            Assert.Equal("mixed-linear-physical-extent-mismatch", backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxFusedLinearGeluFp16M1(
                inputFeatures, outputFeatures), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFusedLinearGeluFp16M1(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures),
                    backend.DirectPtxLastError);

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFusedLinearGeluFp16M1(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertVectorClose(
                backend.DownloadBuffer(output), expected, 4e-4f,
                "backend FP16 fused linear GELU");

            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxFusedLinearGeluFp16M1(
                    inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures));
            Assert.True(captured, backend.DirectPtxLastError);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
            Assert.True(backend.TryGetDirectPtxMixedLinearAudit(
                inputFeatures, outputFeatures, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);

            long dispatchBefore = backend.DirectPtxMixedLinearDispatchCount;
            backend.FusedLinearGELUFp16TransposedM1(
                inputHalf, weightHalf, bias, output, inputFeatures, outputFeatures);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxMixedLinearDispatchCount);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.MixedPrecisionLinearExperimentOverride = previousExperiment;
        }
    }

    [Fact]
    public void DenseLinearCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(39, DirectPtxDenseLinearCoverageManifest.All.Count);
        string[] names = DirectPtxDenseLinearCoverageManifest.All
            .Select(cell => cell.Api).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxDenseLinearCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(DirectPtxDenseLinearCoverageStatus.ExperimentalDirectPtx,
            DirectPtxDenseLinearCoverageManifest.Get(
                "CudaBackend.FusedLinearGELUTransposedM1").Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxDenseLinearCoverageManifest.Get("UnassignedDenseLinearApi"));
    }

    [Theory]
    [InlineData(8, 6, true)]
    [InlineData(8, 0, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, false)]
    [InlineData(9, 0, false)]
    [InlineData(10, 0, false)]
    public void FusedLinearArchitectureMatrix_FailsClosedOutsideSm86(
        int major,
        int minor,
        bool expected)
    {
        Assert.Equal(expected,
            DirectPtxArchitecture.HasValidatedFusedLinear(major, minor));
    }

    [Fact]
    public void QuantizedMixedSparseCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(15, DirectPtxQuantizedMixedSparseCoverageManifest.All.Count);
        string[] names = DirectPtxQuantizedMixedSparseCoverageManifest.All
            .Select(cell => cell.Api).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxQuantizedMixedSparseCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(DirectPtxQuantizedMixedSparseCoverageStatus.ExperimentalDirectPtx,
            DirectPtxQuantizedMixedSparseCoverageManifest.Get(
                "CudaBackend.FusedLinearGELUFp16TransposedM1").Status);
        Assert.Equal(DirectPtxQuantizedMixedSparseCoverageStatus.ExperimentalDirectPtx,
            DirectPtxQuantizedMixedSparseCoverageManifest.Get(
                "CudaBackend.FusedLinearGELUFp16TransposedM16").Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxQuantizedMixedSparseCoverageManifest.Get("UnassignedMixedApi"));
    }

    [SkippableTheory]
    [InlineData(256, 256)]
    [InlineData(512, 2048)]
    [InlineData(1024, 4096)]
    public void DriverOnlyFusedLinearGeluM1_MatchesOracleAndHasZeroLocalBytes(
        int inputFeatures,
        int outputFeatures)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in fused-linear GELU specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedLinearGeluM1Kernel(
            runtime, inputFeatures, outputFeatures);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 8);
        Assert.Equal(4, kernel.Blueprint.Tensors.Count);
        Assert.Equal(DirectPtxPhysicalLayout.LinearWeightOutputMajor,
            kernel.Blueprint.Tensors[1].Layout);

        var random = new Random(20260860 + inputFeatures + outputFeatures);
        float[] inputHost = Values(random, inputFeatures, 0.125f);
        float[] weightsHost = Values(random, inputFeatures * outputFeatures, 0.0625f);
        float[] biasHost = Values(random, outputFeatures, 0.0625f);
        float[] expected = FusedLinearGeluOracle(
            inputHost, weightsHost, biasHost, inputFeatures, outputFeatures);
        using var input = runtime.AllocateBytes((nuint)(inputHost.Length * sizeof(float)));
        using var weights = runtime.AllocateBytes((nuint)(weightsHost.Length * sizeof(float)));
        using var bias = runtime.AllocateBytes((nuint)(biasHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(outputFeatures * sizeof(float)));
        input.Upload<float>(inputHost);
        weights.Upload<float>(weightsHost);
        bias.Upload<float>(biasHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[outputFeatures];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-4f, "direct fused linear GELU");
    }

    [SkippableFact]
    public void BackendFusedLinearGeluM1_IsExactZeroAllocationCapturableAndPublic()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFusedLinearEnabled,
                "Requires an Ampere CUDA backend.");
            const int inputFeatures = 512, outputFeatures = 2048;
            var random = new Random(20260861);
            float[] inputHost = Values(random, inputFeatures, 0.125f);
            float[] weightsHost = Values(random, inputFeatures * outputFeatures, 0.0625f);
            float[] biasHost = Values(random, outputFeatures, 0.0625f);
            float[] expected = FusedLinearGeluOracle(
                inputHost, weightsHost, biasHost, inputFeatures, outputFeatures);
            using var input = backend.AllocateBuffer(inputHost);
            using var weights = backend.AllocateBuffer(weightsHost);
            using var bias = backend.AllocateBuffer(biasHost);
            using var output = backend.AllocateBuffer(outputFeatures);
            using var wrongOutput = backend.AllocateBuffer(outputFeatures + 1);

            Assert.False(PtxFusedLinearGeluM1Kernel.IsPromotedShape(1024, 4096));
            Assert.False(backend.TryDirectPtxFusedLinearGeluM1(
                input, weights, bias, output, 1024, 4096));
            Assert.Equal("fused-linear-performance-gate-not-met", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxFusedLinearGeluM1(
                input, weights, bias, wrongOutput, inputFeatures, outputFeatures));
            Assert.Equal("fused-linear-physical-extent-mismatch", backend.DirectPtxLastError);
            Assert.False(backend.TryDirectPtxFusedLinearGeluM1(
                input, weights, bias, bias, inputFeatures, outputFeatures));
            Assert.Contains("alias", backend.DirectPtxLastError, StringComparison.OrdinalIgnoreCase);
            Assert.True(backend.PrewarmDirectPtxFusedLinearGeluM1(
                inputFeatures, outputFeatures), backend.DirectPtxLastError);
            for (int i = 0; i < 8; i++)
                Assert.True(backend.TryDirectPtxFusedLinearGeluM1(
                    input, weights, bias, output, inputFeatures, outputFeatures),
                    backend.DirectPtxLastError);

            long before = GC.GetAllocatedBytesForCurrentThread();
            bool allLaunched = true;
            for (int i = 0; i < 32; i++)
                allLaunched &= backend.TryDirectPtxFusedLinearGeluM1(
                    input, weights, bias, output, inputFeatures, outputFeatures);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            backend.Synchronize();
            Assert.True(allLaunched, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 2e-4f,
                "backend fused linear GELU");

            bool captureLaunch = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch &= backend.TryDirectPtxFusedLinearGeluM1(
                    input, weights, bias, output, inputFeatures, outputFeatures));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxFusedLinearPinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();

            Assert.True(backend.TryGetDirectPtxFusedLinearAudit(
                inputFeatures, outputFeatures, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            long dispatchBefore = backend.DirectPtxFusedLinearDispatchCount;
            backend.FusedLinearGELUTransposedM1(
                input, weights, bias, output, inputFeatures, outputFeatures);
            backend.Synchronize();
            Assert.Equal(dispatchBefore + 1, backend.DirectPtxFusedLinearDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 2e-4f,
                "public fused linear GELU");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
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

            Assert.True(backend.TryDirectPtxFusedResidualRmsNorm(
                input, residual, gamma, output, rms, rows), backend.DirectPtxLastError);
            for (int i = 0; i < 64; i++)
            {
                Assert.True(backend.TryDirectPtxFusedResidualRmsNorm(
                    input, residual, gamma, output, rms, rows), backend.DirectPtxLastError);
            }
            backend.Synchronize();
            long allocationBefore = GC.GetAllocatedBytesForCurrentThread();
            bool everyLaunchSucceeded = true;
            for (int i = 0; i < 32; i++)
            {
                everyLaunchSucceeded &= backend.TryDirectPtxFusedResidualRmsNorm(
                    input, residual, gamma, output, rms, rows);
            }
            long launchAllocation =
                GC.GetAllocatedBytesForCurrentThread() - allocationBefore;
            backend.Synchronize();
            long synchronizationBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 32; i++) backend.Synchronize();
            long synchronizationAllocation =
                GC.GetAllocatedBytesForCurrentThread() - synchronizationBefore;
            Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
            Assert.Equal(0, launchAllocation);
            Assert.Equal(0, synchronizationAllocation);

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
                DirectPtxArchitectureFamily.Ampere, 8, 6, DirectPtxPhysicalType.Float16,
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

    private static (float[] Probabilities, float[] Output, float[] Stats) FlashForwardReference(
        float[] query,
        float[] key,
        float[] value,
        int batch,
        int heads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        float[]? attentionBias = null,
        int biasBatchStride = 0)
    {
        const int dimension = 64;
        float[] probabilities = AttentionProbabilities(
            query, key, batch, heads, heads,
            querySequence, keyValueSequence, scale, isCausal,
            attentionBias, biasBatchStride);
        var output = new float[query.Length];
        var stats = new float[batch * heads * querySequence];
        for (int b = 0; b < batch; b++)
        for (int h = 0; h < heads; h++)
        for (int qi = 0; qi < querySequence; qi++)
        {
            int row = (b * heads + h) * querySequence + qi;
            int queryBase = row * dimension;
            int probabilityBase = row * keyValueSequence;
            float maximum = float.NegativeInfinity;
            float sum = 0;
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                if (isCausal && ki > qi) continue;
                int keyBase = ((b * heads + h) * keyValueSequence + ki) * dimension;
                float score = 0;
                for (int d = 0; d < dimension; d++)
                    score += query[queryBase + d] * key[keyBase + d];
                score *= scale;
                if (attentionBias is not null)
                    score += attentionBias[
                        b * biasBatchStride + (h * querySequence + qi) * keyValueSequence + ki];
                maximum = MathF.Max(maximum, score);
            }
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                if (isCausal && ki > qi) continue;
                float probability = probabilities[probabilityBase + ki];
                sum += probability;
                int valueBase = ((b * heads + h) * keyValueSequence + ki) * dimension;
                for (int d = 0; d < dimension; d++)
                    output[queryBase + d] += probability * value[valueBase + d];
            }
            // The normalized probabilities sum to one; recover the exact LSE
            // from any admitted score/probability pair without another exp sum.
            int firstKey = 0;
            int firstKeyBase = ((b * heads + h) * keyValueSequence + firstKey) * dimension;
            float firstScore = 0;
            for (int d = 0; d < dimension; d++)
                firstScore += query[queryBase + d] * key[firstKeyBase + d];
            firstScore *= scale;
            if (attentionBias is not null)
                firstScore += attentionBias[
                    b * biasBatchStride + (h * querySequence + qi) * keyValueSequence + firstKey];
            stats[row] = firstScore - MathF.Log(probabilities[probabilityBase + firstKey]);
            Assert.InRange(sum, 0.9999f, 1.0001f);
            Assert.True(float.IsFinite(maximum));
        }
        return (probabilities, output, stats);
    }

    private static float[] AttentionProbabilities(
        float[] query,
        float[] key,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal = false,
        float[]? attentionBias = null,
        int biasBatchStride = 0)
    {
        const int dimension = 64;
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        var probabilities = new float[batch * queryHeads * querySequence * keyValueSequence];
        for (int b = 0; b < batch; b++)
        for (int qh = 0; qh < queryHeads; qh++)
        for (int qi = 0; qi < querySequence; qi++)
        {
            int kvh = qh / queriesPerKeyValue;
            int queryBase = ((b * queryHeads + qh) * querySequence + qi) * dimension;
            int probabilityBase = ((b * queryHeads + qh) * querySequence + qi) * keyValueSequence;
            float maximum = float.NegativeInfinity;
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                if (isCausal && ki > qi)
                {
                    probabilities[probabilityBase + ki] = float.NegativeInfinity;
                    continue;
                }
                int keyBase = ((b * keyValueHeads + kvh) * keyValueSequence + ki) * dimension;
                float score = 0;
                for (int d = 0; d < dimension; d++)
                    score += query[queryBase + d] * key[keyBase + d];
                score *= scale;
                if (attentionBias is not null)
                    score += attentionBias[
                        b * biasBatchStride + (qh * querySequence + qi) * keyValueSequence + ki];
                probabilities[probabilityBase + ki] = score;
                maximum = MathF.Max(maximum, score);
            }
            float sum = 0;
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                if (isCausal && ki > qi)
                {
                    probabilities[probabilityBase + ki] = 0;
                    continue;
                }
                float probability = MathF.Exp(probabilities[probabilityBase + ki] - maximum);
                probabilities[probabilityBase + ki] = probability;
                sum += probability;
            }
            for (int ki = 0; ki < keyValueSequence; ki++)
                probabilities[probabilityBase + ki] /= sum;
        }
        return probabilities;
    }

    private static (float[] GradQuery, float[] GradKey, float[] GradValue) AttentionBackwardOracle(
        float[] gradOutput,
        float[] query,
        float[] key,
        float[] value,
        float[] probabilities,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale)
    {
        const int dimension = 64;
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        var gradQuery = new float[query.Length];
        var gradKey = new float[key.Length];
        var gradValue = new float[value.Length];
        var gradProbability = new float[keyValueSequence];
        for (int b = 0; b < batch; b++)
        for (int qh = 0; qh < queryHeads; qh++)
        for (int qi = 0; qi < querySequence; qi++)
        {
            int kvh = qh / queriesPerKeyValue;
            int queryBase = ((b * queryHeads + qh) * querySequence + qi) * dimension;
            int probabilityBase = ((b * queryHeads + qh) * querySequence + qi) * keyValueSequence;
            float delta = 0;
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                int valueBase = ((b * keyValueHeads + kvh) * keyValueSequence + ki) * dimension;
                float dot = 0;
                for (int d = 0; d < dimension; d++)
                    dot += gradOutput[queryBase + d] * value[valueBase + d];
                gradProbability[ki] = dot;
                delta += probabilities[probabilityBase + ki] * dot;
            }
            for (int ki = 0; ki < keyValueSequence; ki++)
            {
                int keyValueBase = ((b * keyValueHeads + kvh) * keyValueSequence + ki) * dimension;
                float probability = probabilities[probabilityBase + ki];
                float gradScore = scale * probability * (gradProbability[ki] - delta);
                for (int d = 0; d < dimension; d++)
                {
                    gradQuery[queryBase + d] += gradScore * key[keyValueBase + d];
                    gradKey[keyValueBase + d] += gradScore * query[queryBase + d];
                    gradValue[keyValueBase + d] += probability * gradOutput[queryBase + d];
                }
            }
        }
        return (gradQuery, gradKey, gradValue);
    }

    private static float[] Values(Random random, int count) =>
        Enumerable.Range(0, count)
            .Select(_ => (random.NextSingle() - 0.5f) * 0.5f)
            .ToArray();

    private static float[] Values(Random random, int count, float magnitude) =>
        Enumerable.Range(0, count)
            .Select(_ => (random.NextSingle() * 2f - 1f) * magnitude)
            .ToArray();

    private static float[] FusedLinearGeluOracle(
        float[] input,
        float[] weights,
        float[] bias,
        int inputFeatures,
        int outputFeatures)
    {
        var output = new float[outputFeatures];
        for (int column = 0; column < outputFeatures; column++)
        {
            float value = bias[column];
            for (int inner = 0; inner < inputFeatures; inner++)
                value += input[inner] * weights[column * inputFeatures + inner];
            output[column] = 0.5f * value *
                (1f + MathF.Tanh(0.7978845608f *
                    (value + 0.044715f * value * value * value)));
        }
        return output;
    }

    private static void AssertAttentionGradientClose(float[] actual, float[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float error = 0;
        for (int i = 0; i < actual.Length; i++)
            error = MathF.Max(error, MathF.Abs(actual[i] - expected[i]));
        Assert.True(error < 5e-5f, $"Direct PTX attention backward {name} max error {error:G9}.");
    }

    private static ushort[] RandomHalfRange(Random random, int length, float magnitude)
    {
        var result = new ushort[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.HalfToUInt16Bits(
                (Half)((random.NextSingle() - 0.5f) * 2f * magnitude));
        return result;
    }

    private static float[] DecodeOracle(
        float[] query,
        float[] key,
        float[] value,
        int queryHeads,
        int keyValueHeads,
        int sequenceLength,
        int blockSize,
        int[]? blockTable,
        float scale)
    {
        const int dimension = 64;
        var output = new float[queryHeads * dimension];
        var logits = new float[sequenceLength];
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        for (int head = 0; head < queryHeads; head++)
        {
            int keyValueHead = head / queriesPerKeyValue;
            float maximum = float.NegativeInfinity;
            for (int token = 0; token < sequenceLength; token++)
            {
                int physicalToken = blockTable is null
                    ? token
                    : blockTable[token / blockSize] * blockSize + token % blockSize;
                int keyBase = (physicalToken * keyValueHeads + keyValueHead) * dimension;
                float dot = 0;
                for (int d = 0; d < dimension; d++)
                    dot += query[head * dimension + d] * key[keyBase + d];
                logits[token] = dot * scale;
                maximum = MathF.Max(maximum, logits[token]);
            }
            float denominator = 0;
            for (int token = 0; token < sequenceLength; token++)
            {
                logits[token] = MathF.Exp(logits[token] - maximum);
                denominator += logits[token];
            }
            for (int token = 0; token < sequenceLength; token++)
            {
                int physicalToken = blockTable is null
                    ? token
                    : blockTable[token / blockSize] * blockSize + token % blockSize;
                int valueBase = (physicalToken * keyValueHeads + keyValueHead) * dimension;
                float probability = logits[token] / denominator;
                for (int d = 0; d < dimension; d++)
                    output[head * dimension + d] += probability * value[valueBase + d];
            }
        }
        return output;
    }

    private static void AssertDecodeClose(float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        float error = 0;
        for (int i = 0; i < actual.Length; i++)
            error = MathF.Max(error, MathF.Abs(actual[i] - expected[i]));
        Assert.True(error < 2e-5f, $"Direct PTX decode max error {error:G9}.");
    }

    private static float[] PagedPrefillOracle(
        float[] query,
        float[] key,
        float[] value,
        int[] blockTable,
        int queryHeads,
        int keyValueHeads,
        int queryCount,
        int startPosition,
        int blockSize,
        float scale)
    {
        const int dimension = 64;
        var output = new float[query.Length];
        var logits = new float[startPosition + queryCount];
        int queriesPerKeyValue = queryHeads / keyValueHeads;
        for (int queryIndex = 0; queryIndex < queryCount; queryIndex++)
        for (int head = 0; head < queryHeads; head++)
        {
            int keyValueHead = head / queriesPerKeyValue;
            int keyLength = startPosition + queryIndex + 1;
            int queryBase = (queryIndex * queryHeads + head) * dimension;
            float maximum = float.NegativeInfinity;
            for (int token = 0; token < keyLength; token++)
            {
                int physicalToken = blockTable[token / blockSize] * blockSize + token % blockSize;
                int keyBase = (physicalToken * keyValueHeads + keyValueHead) * dimension;
                float dot = 0;
                for (int d = 0; d < dimension; d++)
                    dot += query[queryBase + d] * key[keyBase + d];
                logits[token] = dot * scale;
                maximum = MathF.Max(maximum, logits[token]);
            }
            float denominator = 0;
            for (int token = 0; token < keyLength; token++)
            {
                logits[token] = MathF.Exp(logits[token] - maximum);
                denominator += logits[token];
            }
            for (int token = 0; token < keyLength; token++)
            {
                int physicalToken = blockTable[token / blockSize] * blockSize + token % blockSize;
                int valueBase = (physicalToken * keyValueHeads + keyValueHead) * dimension;
                float probability = logits[token] / denominator;
                for (int d = 0; d < dimension; d++)
                    output[queryBase + d] += probability * value[valueBase + d];
            }
        }
        return output;
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
                float actualRowStats = actualStats[flatQueryHead * querySequence + row];
                if (lastKey < 0)
                    Assert.Equal(float.NegativeInfinity, actualRowStats);
                else
                {
                    float expectedStats = maximum + MathF.Log(sum);
                    maxStatsError = MathF.Max(maxStatsError,
                        MathF.Abs(actualRowStats - expectedStats));
                }
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

    private static (float[] Cosine, float[] Sine) RopeTables(int capacity)
    {
        const int pairs = PtxFusedQkvRopeCacheD64Kernel.HeadDimension / 2;
        var cosine = new float[capacity * pairs];
        var sine = new float[capacity * pairs];
        for (int position = 0; position < capacity; position++)
        for (int pair = 0; pair < pairs; pair++)
        {
            float angle = position * MathF.Pow(10_000f, -2f * pair /
                PtxFusedQkvRopeCacheD64Kernel.HeadDimension);
            cosine[position * pairs + pair] = MathF.Cos(angle);
            sine[position * pairs + pair] = MathF.Sin(angle);
        }
        return (cosine, sine);
    }

    private static (float[] Query, float[] KeyCache, float[] ValueCache) QkvRopeCacheOracle(
        float[] input,
        float[] weights,
        float[] bias,
        float[] cosine,
        float[] sine,
        float[] initialKeyCache,
        float[] initialValueCache,
        int heads,
        int cacheCapacity,
        int position)
    {
        const int dimension = PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
        int modelDimension = heads * dimension;
        var projection = new float[3 * modelDimension];
        for (int output = 0; output < projection.Length; output++)
        {
            float sum = bias[output];
            int weightBase = output * modelDimension;
            for (int inner = 0; inner < modelDimension; inner++)
                sum += input[inner] * weights[weightBase + inner];
            projection[output] = sum;
        }

        var query = new float[modelDimension];
        var keyCache = (float[])initialKeyCache.Clone();
        var valueCache = (float[])initialValueCache.Clone();
        int cacheBase = position * modelDimension;
        int ropeBase = position * (dimension / 2);
        for (int head = 0; head < heads; head++)
        for (int pair = 0; pair < dimension / 2; pair++)
        {
            int feature = pair * 2;
            int outputIndex = head * dimension + feature;
            float c = cosine[ropeBase + pair];
            float s = sine[ropeBase + pair];
            float qe = projection[outputIndex];
            float qo = projection[outputIndex + 1];
            float ke = projection[modelDimension + outputIndex];
            float ko = projection[modelDimension + outputIndex + 1];
            query[outputIndex] = qe * c - qo * s;
            query[outputIndex + 1] = qe * s + qo * c;
            keyCache[cacheBase + outputIndex] = ke * c - ko * s;
            keyCache[cacheBase + outputIndex + 1] = ke * s + ko * c;
            valueCache[cacheBase + outputIndex] = projection[2 * modelDimension + outputIndex];
            valueCache[cacheBase + outputIndex + 1] = projection[2 * modelDimension + outputIndex + 1];
        }
        return (query, keyCache, valueCache);
    }

    private static float[] FusedLinearGeluOracle(
        float[] input,
        float[] weights,
        float[] bias,
        int rows,
        int inputFeatures,
        int outputFeatures)
    {
        var output = new float[rows * outputFeatures];
        for (int row = 0; row < rows; row++)
        for (int column = 0; column < outputFeatures; column++)
        {
            float value = bias[column];
            for (int inner = 0; inner < inputFeatures; inner++)
                value += input[row * inputFeatures + inner] *
                    weights[column * inputFeatures + inner];
            output[row * outputFeatures + column] = 0.5f * value *
                (1f + MathF.Tanh(0.7978845608f *
                    (value + 0.044715f * value * value * value)));
        }
        return output;
    }

    private static void AssertVectorClose(
        float[] actual,
        float[] expected,
        float tolerance,
        string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maximum = 0;
        int maximumIndex = -1;
        for (int i = 0; i < actual.Length; i++)
        {
            float error = MathF.Abs(actual[i] - expected[i]);
            if (error > maximum) { maximum = error; maximumIndex = i; }
        }
        Assert.True(maximum <= tolerance,
            $"{name} max absolute error {maximum:G9} at {maximumIndex}; tolerance {tolerance:G9}.");
    }

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
