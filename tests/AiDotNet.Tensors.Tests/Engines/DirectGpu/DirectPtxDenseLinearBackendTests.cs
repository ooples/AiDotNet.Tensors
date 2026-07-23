#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Backend lifecycle contracts for the issue-#836 dense/linear PTX families.
/// The collection serializes the process-wide deterministic-mode mutation.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class DirectPtxDenseLinearBackendTests
{
    [SkippableFact]
    public void UnsupportedSemantics_FailClosedWithStableReasonsBeforeJit()
    {
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

            AssertRejected(
                () => backend.PrewarmDirectPtxFp16Gemm(0, 16, 32),
                backend, "16-bit-gemm-shape-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFp16Gemm(
                    16, 16, 32, inputType: (DirectPtx16BitInputType)99),
                backend, "16-bit-gemm-semantics-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearCrossEntropy(
                    (DirectPtxCrossEntropyTarget)99, 4, 16, 32),
                backend, "fused-linear-ce-target-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearTiled(
                    64, 256, 256, (DirectPtxLinearActivation)99),
                backend, "fused-linear-activation-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearTiled(
                    64, 256, 256, DirectPtxLinearActivation.Relu,
                    (DirectPtxLinearWeightLayout)99),
                backend, "linear-weight-layout-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearBackward(
                    64, 256, 256, DirectPtxLinearActivation.None),
                backend, "fused-linear-backward-activation-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLoRA(
                    8, 256, 8, 256, float.NaN),
                backend, "fused-lora-scaling-not-finite");
            AssertRejected(
                () => backend.PrewarmDirectPtxDenseVector(
                    (DirectPtxDenseVectorOperation)99, 4096),
                backend, "dense-vector-operation-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxBatchedVector(
                    (DirectPtxBatchedVectorOperation)99, 4, 512),
                backend, "batched-vector-operation-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxStridedDot(0, 512, 0, 1),
                backend, "strided-dot-shape-not-implemented");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearGeluFp16M16(256, 256),
                backend, "fp16-tensorcore-linear-shape-not-implemented");

            DirectPtxFeatureGate.FusedLinearExperimentOverride = false;
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearTiled(
                    64, 256, 256, DirectPtxLinearActivation.Relu),
                backend, "fused-linear-tiled-performance-gate-not-met");
            AssertRejected(
                () => backend.PrewarmDirectPtxGemmTiled(
                    64, 256, 256, DirectPtxLinearWeightLayout.InputMajor),
                backend, "gemm-tiled-performance-gate-not-met");
            AssertRejected(
                () => backend.PrewarmDirectPtxFusedLinearGeluFp16M16(512, 2_048),
                backend, "fp16-tensorcore-linear-performance-gate-not-met");
        }
        finally
        {
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
            DirectPtxFeatureGate.TestOverride = previousGate;
        }
    }

    [SkippableFact]
    public void PrewarmedExperimentalFamilies_AreZeroAllocationAndGraphCapturable()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.FusedLinearExperimentOverride;
        bool previousDeterministic = AiDotNetEngine.DeterministicMode;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = true;
        AiDotNetEngine.SetDeterministicMode(false);
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFusedLinearEnabled,
                "Requires the measured SM86 fused-linear architecture domain.");

            using var halfA = backend.AllocateByteBuffer(16 * 32 * sizeof(ushort));
            using var halfB = backend.AllocateByteBuffer(32 * 16 * sizeof(ushort));
            using var halfC = backend.AllocateBuffer(16 * 16);
            Assert.True(backend.PrewarmDirectPtxFp16Gemm(16, 16, 32), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () => backend.TryDirectPtxFp16Gemm(
                halfA, halfB, halfC, 16, 16, 32));

            const int tensorCoreK = 512, tensorCoreN = 2_048;
            using var tensorCoreInput = backend.AllocateByteBuffer(
                PtxFusedLinearGeluFp16M16Kernel.Rows * tensorCoreK * sizeof(ushort));
            using var tensorCoreWeights = backend.AllocateByteBuffer(
                tensorCoreN * tensorCoreK * sizeof(ushort));
            using var tensorCoreBias = backend.AllocateBuffer(tensorCoreN);
            using var tensorCoreOutput = backend.AllocateBuffer(
                PtxFusedLinearGeluFp16M16Kernel.Rows * tensorCoreN);
            using var tensorCoreWrongOutput = backend.AllocateBuffer(
                PtxFusedLinearGeluFp16M16Kernel.Rows * tensorCoreN + 1);
            Assert.False(backend.TryDirectPtxFusedLinearGeluFp16M16(
                tensorCoreInput, tensorCoreWeights, tensorCoreBias,
                tensorCoreWrongOutput, tensorCoreK, tensorCoreN));
            Assert.Equal("fp16-tensorcore-linear-physical-extent-mismatch",
                backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxFusedLinearGeluFp16M16(
                tensorCoreK, tensorCoreN), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () =>
                backend.TryDirectPtxFusedLinearGeluFp16M16(
                    tensorCoreInput, tensorCoreWeights, tensorCoreBias,
                    tensorCoreOutput, tensorCoreK, tensorCoreN));

            using var gradC16 = backend.AllocateByteBuffer(16 * 16 * sizeof(ushort));
            using var gradLeft = backend.AllocateBuffer(16 * 32);
            using var gradRight = backend.AllocateBuffer(32 * 16);
            Assert.True(backend.PrewarmDirectPtxFp16Gemm(
                16, 32, 16, transposeB: true), backend.DirectPtxLastError);
            Assert.True(backend.PrewarmDirectPtxFp16Gemm(
                32, 16, 16, transposeA: true), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () => backend.TryDirectPtxFp16Backward(
                gradC16, halfA, halfB, gradLeft, gradRight, 16, 16, 32, halfOutput: false));

            const int loraBatch = 8, inputFeatures = 256, rank = 8, outputFeatures = 256;
            using var loraInput = backend.AllocateBuffer(loraBatch * inputFeatures);
            using var baseOutput = backend.AllocateBuffer(loraBatch * outputFeatures);
            using var loraA = backend.AllocateBuffer(inputFeatures * rank);
            using var loraB = backend.AllocateBuffer(rank * outputFeatures);
            using var loraOutput = backend.AllocateBuffer(loraBatch * outputFeatures);
            Assert.True(backend.PrewarmDirectPtxFusedLoRA(
                loraBatch, inputFeatures, rank, outputFeatures, 0.125f), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () => backend.TryDirectPtxFusedLoRA(
                loraInput, baseOutput, loraA, loraB, loraOutput,
                loraBatch, inputFeatures, rank, outputFeatures, 0.125f));

            const int m = 64, k = 256, n = 256;
            using var gradOutput = backend.AllocateBuffer(m * n);
            using var input = backend.AllocateBuffer(m * k);
            using var weights = backend.AllocateBuffer(k * n);
            using var saved = backend.AllocateBuffer(m * n);
            using var dInput = backend.AllocateBuffer(m * k);
            using var dWeight = backend.AllocateBuffer(k * n);
            using var dBias = backend.AllocateBuffer(n);
            Assert.True(backend.PrewarmDirectPtxFusedLinearBackward(
                m, k, n, DirectPtxLinearActivation.Relu), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () => backend.TryDirectPtxFusedLinearBackward(
                gradOutput, input, weights, saved, dInput, dWeight, dBias,
                m, k, n, DirectPtxLinearActivation.Relu));

            const int rows = 4, hidden = 16, vocabulary = 32;
            using var ceInput = backend.AllocateBuffer(rows * hidden);
            using var ceWeight = backend.AllocateBuffer(hidden * vocabulary);
            using var ceBias = backend.AllocateBuffer(vocabulary);
            using var ceTargets = backend.AllocateBuffer(rows);
            using var ceLoss = backend.AllocateBuffer(1);
            Assert.True(backend.PrewarmDirectPtxFusedLinearCrossEntropy(
                DirectPtxCrossEntropyTarget.Index, rows, hidden, vocabulary),
                backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () =>
                backend.TryDirectPtxFusedLinearCrossEntropy(
                    DirectPtxCrossEntropyTarget.Index,
                    ceInput, ceWeight, ceBias, ceTargets, ceLoss,
                    rows, hidden, vocabulary));

            using var vectorA = backend.AllocateBuffer(4096);
            using var vectorB = backend.AllocateBuffer(4096);
            using var scalar = backend.AllocateBuffer(1);
            Assert.True(backend.PrewarmDirectPtxDenseVector(
                DirectPtxDenseVectorOperation.Dot, 4096), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () =>
                backend.TryDirectPtxDotProduct(vectorA, vectorB, scalar, 4096));

            using var outerA = backend.AllocateBuffer(64);
            using var outerB = backend.AllocateBuffer(128);
            using var outer = backend.AllocateBuffer(64 * 128);
            Assert.True(backend.PrewarmDirectPtxDenseVector(
                DirectPtxDenseVectorOperation.Outer, 64, 128), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () =>
                backend.TryDirectPtxOuterProduct(outerA, outerB, outer, 64, 128));

            using var batchA = backend.AllocateBuffer(4 * 512);
            using var batchB = backend.AllocateBuffer(4 * 512);
            using var batchOutput = backend.AllocateBuffer(4);
            Assert.True(backend.PrewarmDirectPtxBatchedVector(
                DirectPtxBatchedVectorOperation.Dot, 4, 512), backend.DirectPtxLastError);
            AssertZeroAllocationAndCapture(backend, () =>
                backend.TryDirectPtxBatchDotProduct(batchA, batchB, batchOutput, 4, 512));

            Assert.True(backend.PrewarmDirectPtxStridedDot(512, 512, 511, -1),
                backend.DirectPtxLastError);
            using var stridedA = backend.AllocateBuffer(512);
            using var stridedB = backend.AllocateBuffer(512);
            AssertZeroAllocationAndCapture(backend, () => backend.TryDirectPtxStridedDotProduct(
                stridedA, stridedB, scalar, 512, 512, 511, -1));
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previousDeterministic);
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
            DirectPtxFeatureGate.TestOverride = previousGate;
        }
    }

    [SkippableFact]
    public void FusedLinearCrossEntropy_AtomicReductionFailsClosedInDeterministicMode()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.FusedLinearExperimentOverride;
        bool previousDeterministic = AiDotNetEngine.DeterministicMode;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.FusedLinearExperimentOverride = true;
        AiDotNetEngine.SetDeterministicMode(true);
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxFusedLinearEnabled,
                "Requires the measured SM86 fused-linear architecture domain.");
            using var hidden = backend.AllocateBuffer(4 * 16);
            using var weights = backend.AllocateBuffer(16 * 32);
            using var bias = backend.AllocateBuffer(32);
            using var targets = backend.AllocateBuffer(4);
            using var loss = backend.AllocateBuffer(1);

            Assert.False(backend.TryDirectPtxFusedLinearCrossEntropy(
                DirectPtxCrossEntropyTarget.Index,
                hidden, weights, bias, targets, loss, 4, 16, 32));
            Assert.Contains("disabled in deterministic mode", backend.DirectPtxLastError,
                StringComparison.Ordinal);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(previousDeterministic);
            DirectPtxFeatureGate.FusedLinearExperimentOverride = previousExperiment;
            DirectPtxFeatureGate.TestOverride = previousGate;
        }
    }

    private static void AssertZeroAllocationAndCapture(
        CudaBackend backend,
        Func<bool> launch)
    {
        for (int i = 0; i < 8; i++) Assert.True(launch(), backend.DirectPtxLastError);
        backend.Synchronize();

        bool everyLaunchSucceeded = true;
        (long deviceCountBefore, long deviceBytesBefore) =
            backend.DirectPtxEvidenceDeviceAllocations;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 32; i++) everyLaunchSucceeded &= launch();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        backend.Synchronize();
        Assert.True(everyLaunchSucceeded, backend.DirectPtxLastError);
        Assert.Equal(0, allocated);
        Assert.Equal(
            (deviceCountBefore, deviceBytesBefore),
            backend.DirectPtxEvidenceDeviceAllocations);

        bool captureLaunchSucceeded = false;
        (long captureCountBefore, long captureBytesBefore) =
            backend.DirectPtxEvidenceDeviceAllocations;
        IntPtr graph = backend.CaptureGraph(() => captureLaunchSucceeded = launch());
        Assert.True(captureLaunchSucceeded, backend.DirectPtxLastError);
        Assert.NotEqual(IntPtr.Zero, graph);
        try
        {
            backend.LaunchCapturedGraph(graph);
        }
        finally
        {
            backend.DestroyCapturedGraph(graph);
        }
        Assert.Equal(
            (captureCountBefore, captureBytesBefore),
            backend.DirectPtxEvidenceDeviceAllocations);
    }

    private static void AssertRejected(
        Func<bool> action,
        CudaBackend backend,
        string expectedReason)
    {
        Assert.False(action());
        Assert.Equal(expectedReason, backend.DirectPtxLastError);
    }
}
#endif
