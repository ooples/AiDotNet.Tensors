using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Experimental exact-shape M=16 FP16 Tensor-Core fused-linear route. It shares
/// the dense-linear gate, remains fail-closed, and admits no implicit layout
/// conversion: callers provide canonical contiguous input and output-major
/// prepacked weights.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly record struct DirectPtxFp16TensorCoreLinearPlanKey(
        int InputFeatures, int OutputFeatures);

    private readonly record struct DirectPtxFp16TensorCoreLinearKey(
        int InputFeatures, int OutputFeatures, int OutputsPerBlock);

    private readonly DirectPtxPlanCache<
        DirectPtxFp16TensorCoreLinearPlanKey, int>
        _directPtxFp16TensorCoreLinearPlans =
            new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly DirectPtxKernelCache<
        DirectPtxFp16TensorCoreLinearKey, PtxFusedLinearGeluFp16M16Kernel>
        _directPtxFp16TensorCoreLinearKernels =
            new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private long _directPtxFp16TensorCoreLinearDispatchCount;

    internal long DirectPtxFp16TensorCoreLinearDispatchCount =>
        System.Threading.Interlocked.Read(
            ref _directPtxFp16TensorCoreLinearDispatchCount);

    internal bool TryDirectPtxFusedLinearGeluFp16M16(
        IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightsHalf,
        IGpuBuffer biasFloat,
        IGpuBuffer outputFloat,
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxFusedLinearEnabled)
            return false;
        if (!PtxFusedLinearGeluFp16M16Kernel.IsSupportedShape(
                inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "fp16-tensorcore-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(
                inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fp16-tensorcore-linear-performance-gate-not-met";
            return false;
        }
        if (DirectPtxBufferIsInvalid(inputHalf) ||
            DirectPtxBufferIsInvalid(outputMajorWeightsHalf) ||
            DirectPtxBufferIsInvalid(biasFloat) ||
            DirectPtxBufferIsInvalid(outputFloat))
        {
            DirectPtxLastError = "fp16-tensorcore-linear-null-or-invalid-buffer";
            return false;
        }
        long inputBytes = checked(
            (long)PtxFusedLinearGeluFp16M16Kernel.Rows *
            inputFeatures * sizeof(ushort));
        long weightBytes = checked(
            (long)inputFeatures * outputFeatures * sizeof(ushort));
        long biasBytes = checked((long)outputFeatures * sizeof(float));
        long outputBytes = checked(
            (long)PtxFusedLinearGeluFp16M16Kernel.Rows *
            outputFeatures * sizeof(float));
        if (inputHalf.SizeInBytes != inputBytes ||
            outputMajorWeightsHalf.SizeInBytes != weightBytes ||
            biasFloat.SizeInBytes != biasBytes ||
            outputFloat.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "fp16-tensorcore-linear-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(outputFloat, inputHalf) ||
            DirectPtxBuffersOverlap(outputFloat, outputMajorWeightsHalf) ||
            DirectPtxBuffersOverlap(outputFloat, biasFloat))
        {
            DirectPtxLastError = "fp16-tensorcore-linear-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var plan = new DirectPtxFp16TensorCoreLinearPlanKey(
                inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxFp16TensorCoreLinearPlans.TryGetValue(
                        plan, out int captureTile) ||
                     !_directPtxFp16TensorCoreLinearKernels.TryGetValue(
                        new DirectPtxFp16TensorCoreLinearKey(
                            inputFeatures, outputFeatures, captureTile), out _)))
                {
                    DirectPtxLastError =
                        "Direct PTX FP16 Tensor-Core linear must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                if (!_directPtxFp16TensorCoreLinearPlans.TryGetValue(
                        plan, out int outputsPerBlock))
                    outputsPerBlock = ResolveFp16TensorCoreLinearPlanSlow(
                        plan, inputHalf, outputMajorWeightsHalf,
                        biasFloat, outputFloat);
                var key = new DirectPtxFp16TensorCoreLinearKey(
                    inputFeatures, outputFeatures, outputsPerBlock);
                if (!_directPtxFp16TensorCoreLinearKernels.TryGetValue(
                        key, out PtxFusedLinearGeluFp16M16Kernel? kernel))
                    kernel = CreateAndCacheDirectPtxFp16TensorCoreLinearSlow(key);
                if (capturing && !_directPtxFp16TensorCoreLinearKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX FP16 Tensor-Core module for graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(inputHalf, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(
                            outputMajorWeightsHalf, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(biasFloat, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(
                ref _directPtxFp16TensorCoreLinearDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception exception)
        {
            DirectPtxLastError =
                $"{exception.GetType().Name}: {exception.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxFusedLinearGeluFp16M16(
        int inputFeatures,
        int outputFeatures)
    {
        if (!IsDirectPtxFusedLinearEnabled)
            return false;
        if (!PtxFusedLinearGeluFp16M16Kernel.IsSupportedShape(
                inputFeatures, outputFeatures))
        {
            DirectPtxLastError = "fp16-tensorcore-linear-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearGeluFp16M16Kernel.IsPromotedShape(
                inputFeatures, outputFeatures) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fp16-tensorcore-linear-performance-gate-not-met";
            return false;
        }
        if (IsStreamCapturing())
        {
            DirectPtxLastError =
                "Direct PTX FP16 Tensor-Core linear prewarm is not capture-safe.";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            var plan = new DirectPtxFp16TensorCoreLinearPlanKey(
                inputFeatures, outputFeatures);
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                int outputsPerBlock = PtxFusedLinearGeluFp16M16Kernel.DefaultOutputsPerBlock;
                if (DirectPtxFeatureGate.IsAutotuneEnabled &&
                    DirectPtxDenseLinearAutotuner.TryLoad(
                        _directPtxRuntime, inputFeatures, outputFeatures,
                        out int persistedOutputsPerBlock))
                    outputsPerBlock = persistedOutputsPerBlock;
                _directPtxFp16TensorCoreLinearPlans.Set(plan, outputsPerBlock);
                var key = new DirectPtxFp16TensorCoreLinearKey(
                    inputFeatures, outputFeatures, outputsPerBlock);
                if (!_directPtxFp16TensorCoreLinearKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxFp16TensorCoreLinearSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception exception)
        {
            DirectPtxLastError =
                $"{exception.GetType().Name}: {exception.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedLinearGeluFp16M16Kernel
        CreateAndCacheDirectPtxFp16TensorCoreLinearSlow(
            DirectPtxFp16TensorCoreLinearKey key) =>
        _directPtxFp16TensorCoreLinearKernels.GetOrAdd(
            key, () => new PtxFusedLinearGeluFp16M16Kernel(
                _directPtxRuntime!, key.InputFeatures, key.OutputFeatures,
                key.OutputsPerBlock));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private int ResolveFp16TensorCoreLinearPlanSlow(
        DirectPtxFp16TensorCoreLinearPlanKey plan,
        IGpuBuffer inputHalf,
        IGpuBuffer outputMajorWeightsHalf,
        IGpuBuffer biasFloat,
        IGpuBuffer outputFloat)
    {
        int selected = PtxFusedLinearGeluFp16M16Kernel.DefaultOutputsPerBlock;
        bool persisted = DirectPtxFeatureGate.IsAutotuneEnabled &&
            DirectPtxDenseLinearAutotuner.TryLoad(
                _directPtxRuntime!, plan.InputFeatures, plan.OutputFeatures,
                out selected);
        if (!persisted && DirectPtxFeatureGate.IsAutotuneEnabled)
        {
            double bestMilliseconds = double.PositiveInfinity;
            lock (GpuDispatchLock)
            {
                foreach (int candidate in DirectPtxDenseLinearAutotuner.Candidates(
                    plan.InputFeatures, plan.OutputFeatures))
                {
                    var key = new DirectPtxFp16TensorCoreLinearKey(
                        plan.InputFeatures, plan.OutputFeatures, candidate);
                    if (!_directPtxFp16TensorCoreLinearKernels.TryGetValue(
                            key, out PtxFusedLinearGeluFp16M16Kernel? kernel))
                        kernel = CreateAndCacheDirectPtxFp16TensorCoreLinearSlow(key);
                    float milliseconds = _directPtxRuntime!.MeasureKernelMilliseconds(
                        () => kernel.Launch(
                            DirectPtxTensorView.Create(
                                inputHalf, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(
                                outputMajorWeightsHalf, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(
                                biasFloat, kernel.Blueprint.Tensors[2]),
                            DirectPtxTensorView.Create(
                                outputFloat, kernel.Blueprint.Tensors[3])),
                        warmup: 3, iterations: 12);
                    if (milliseconds < bestMilliseconds)
                    {
                        bestMilliseconds = milliseconds;
                        selected = candidate;
                    }
                }
            }
            double work = 2d * PtxFusedLinearGeluFp16M16Kernel.Rows *
                plan.InputFeatures * plan.OutputFeatures;
            DirectPtxDenseLinearAutotuner.Store(
                _directPtxRuntime!, plan.InputFeatures, plan.OutputFeatures,
                selected, bestMilliseconds, work / (bestMilliseconds * 1_000_000d));
        }
        _directPtxFp16TensorCoreLinearPlans.Set(plan, selected);
        return selected;
    }

    internal bool TryGetDirectPtxFp16TensorCoreLinearAudit(
        int inputFeatures,
        int outputFeatures,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxFp16TensorCoreLinearPlanKey(
                inputFeatures, outputFeatures);
            int outputsPerBlock =
                PtxFusedLinearGeluFp16M16Kernel.DefaultOutputsPerBlock;
            if (_directPtxFp16TensorCoreLinearPlans.TryGetValue(
                    plan, out int selectedOutputsPerBlock))
                outputsPerBlock = selectedOutputsPerBlock;
            var key = new DirectPtxFp16TensorCoreLinearKey(
                inputFeatures, outputFeatures, outputsPerBlock);
            if (_directPtxFp16TensorCoreLinearKernels.TryGetValue(
                    key, out PtxFusedLinearGeluFp16M16Kernel? kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
