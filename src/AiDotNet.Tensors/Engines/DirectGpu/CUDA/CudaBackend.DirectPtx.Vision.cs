#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IDirectPtxVisionBackend
{
    private static readonly string[] DirectPtxVisionDisabledReasons =
        CreateDirectPtxVisionReasons("feature-disabled");
    private static readonly string[] DirectPtxVisionUnavailableReasons =
        CreateDirectPtxVisionReasons("backend-unavailable");
    private static readonly string[] DirectPtxVisionArchitectureReasons =
        CreateDirectPtxVisionReasons("architecture-not-implemented");
    private static readonly string[] DirectPtxVisionNotAdmittedReasons =
        CreateDirectPtxVisionReasons("specialization-not-admitted");
    private readonly DirectPtxKernelCache<DirectPtxVisionSpec, PtxVisionKernel>
        _directPtxVisionKernels = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly long[] _directPtxVisionDispatchCounts =
        new long[Enum.GetValues(typeof(DirectPtxVisionOperation)).Length];

    internal int DirectPtxVisionKernelCapacity => _directPtxVisionKernels.Capacity;
    internal int DirectPtxVisionPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxVisionKernels.PinnedCount; }
    }

    internal long DirectPtxVisionDispatchCount(DirectPtxVisionOperation operation) =>
        System.Threading.Interlocked.Read(ref _directPtxVisionDispatchCounts[(int)operation]);

    private bool RejectDirectPtxVisionSpecialization(
        DirectPtxVisionOperation operation)
    {
        DirectPtxLastError = DirectPtxVisionReason(
            DirectPtxVisionNotAdmittedReasons, operation);
        return false;
    }

    internal bool TryDirectPtxVisionKernel(
        DirectPtxVisionSpec spec,
        IGpuBuffer a,
        IGpuBuffer? b = null,
        IGpuBuffer? c = null,
        IGpuBuffer? d = null,
        IGpuBuffer? e = null,
        IGpuBuffer? f = null)
    {
        if (!DirectPtxVisionSpecializations.IsAdmitted(spec))
        {
            DirectPtxLastError = DirectPtxVisionReason(
                DirectPtxVisionNotAdmittedReasons, spec.Operation);
            return false;
        }
        if (!DirectPtxFeatureGate.IsVisionOperationEnabled(spec.Operation))
        {
            DirectPtxLastError = DirectPtxVisionDisabledReasons[(int)spec.Operation];
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxVisionUnavailableReasons[(int)spec.Operation];
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedVisionBoxIou(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxVisionArchitectureReasons[(int)spec.Operation];
            return false;
        }
        if (a is null)
        {
            DirectPtxLastError = $"vision-{spec.Operation}-null-buffer";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxVisionKernels.TryGetValue(spec, out _))
                {
                    DirectPtxLastError =
                        $"Direct PTX vision operation {spec.Operation} must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxVisionKernel kernel = GetOrCreateVisionKernel(spec);
                int count = kernel.Blueprint.Tensors.Count;
                if ((count > 1 && b is null) || (count > 2 && c is null) ||
                    (count > 3 && d is null) || (count > 4 && e is null) ||
                    (count > 5 && f is null))
                {
                    DirectPtxLastError = $"vision-{spec.Operation}-null-buffer";
                    return false;
                }
                if (capturing && !_directPtxVisionKernels.Pin(spec))
                    throw new InvalidOperationException(
                        $"Could not pin direct-PTX vision module {spec.Operation} for CUDA graph capture.");
                DirectPtxTensorView av = DirectPtxTensorView.Create(a, kernel.Blueprint.Tensors[0]);
                DirectPtxTensorView bv = count > 1 ? DirectPtxTensorView.Create(b!, kernel.Blueprint.Tensors[1]) : default;
                DirectPtxTensorView cv = count > 2 ? DirectPtxTensorView.Create(c!, kernel.Blueprint.Tensors[2]) : default;
                DirectPtxTensorView dv = count > 3 ? DirectPtxTensorView.Create(d!, kernel.Blueprint.Tensors[3]) : default;
                DirectPtxTensorView ev = count > 4 ? DirectPtxTensorView.Create(e!, kernel.Blueprint.Tensors[4]) : default;
                DirectPtxTensorView fv = count > 5 ? DirectPtxTensorView.Create(f!, kernel.Blueprint.Tensors[5]) : default;
                lock (GpuDispatchLock) kernel.Launch(av, bv, cv, dv, ev, fv);
            }
            System.Threading.Interlocked.Increment(
                ref _directPtxVisionDispatchCounts[(int)spec.Operation]);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryDirectPtxVisionVector(
        DirectPtxVisionOperation operation,
        IGpuBuffer a,
        IGpuBuffer b,
        IGpuBuffer? c,
        int length)
    {
        if (!DirectPtxVisionSpecializations.TryVector(
                operation, length, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(operation);
        return TryDirectPtxVisionKernel(spec, a, b, c);
    }

    internal bool TryDirectPtxVisionVectorBackward(
        DirectPtxVisionOperation operation,
        IGpuBuffer gradOutput,
        IGpuBuffer predicted,
        IGpuBuffer target,
        IGpuBuffer gradPredicted,
        int length)
    {
        if (!DirectPtxVisionSpecializations.TryVector(
                operation, length, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(operation);
        return TryDirectPtxVisionKernel(
            spec, gradOutput, predicted, target, gradPredicted);
    }

    internal bool TryDirectPtxVisionPairwise(
        DirectPtxVisionOperation operation,
        IGpuBuffer boxesA,
        IGpuBuffer boxesB,
        IGpuBuffer output,
        int n,
        int m)
    {
        if (!DirectPtxVisionSpecializations.TryPairwise(
                operation, n, m, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(operation);
        return TryDirectPtxVisionKernel(spec, boxesA, boxesB, output);
    }

    internal bool TryDirectPtxVisionBoxConvert(
        IGpuBuffer boxes,
        IGpuBuffer output,
        int n,
        int fromFormat,
        int toFormat)
    {
        if (!DirectPtxVisionSpecializations.TryBoxConvert(
                n, fromFormat, toFormat, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(
                DirectPtxVisionOperation.BoxConvert);
        return TryDirectPtxVisionKernel(spec, boxes, output);
    }

    internal bool TryDirectPtxVisionRoi(
        DirectPtxVisionOperation operation,
        IGpuBuffer input,
        IGpuBuffer boxes,
        IGpuBuffer output,
        int n,
        int channels,
        int height,
        int width,
        int boxCount,
        int outputHeight,
        int outputWidth,
        int outputChannels,
        float spatialScale,
        int samplingRatio,
        bool aligned)
    {
        if (!DirectPtxVisionSpecializations.TryRoi(
                operation, n, channels, height, width, boxCount,
                outputHeight, outputWidth, outputChannels, spatialScale,
                samplingRatio, aligned, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(operation);
        return TryDirectPtxVisionKernel(spec, input, boxes, output);
    }

    internal bool TryDirectPtxVisionNms(
        IGpuBuffer boxes,
        IGpuBuffer scores,
        IGpuBuffer classIds,
        IGpuBuffer suppressed,
        IGpuBuffer outputCapacity,
        IGpuBuffer outputCount,
        int length,
        float iouThreshold,
        bool batched)
    {
        if (!DirectPtxVisionSpecializations.TryNms(
                length, iouThreshold, batched, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(
                DirectPtxVisionOperation.Nms);
        return TryDirectPtxVisionKernel(
            spec, boxes, scores, classIds, suppressed, outputCapacity, outputCount);
    }

    internal bool TryDirectPtxVisionCross3(
        IGpuBuffer a,
        IGpuBuffer b,
        IGpuBuffer output,
        int outerSize,
        int innerSize)
    {
        if (!DirectPtxVisionSpecializations.TryCross3(
                outerSize, innerSize, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(
                DirectPtxVisionOperation.Cross3);
        return TryDirectPtxVisionKernel(spec, a, b, output);
    }

    internal bool TryDirectPtxVisionMasksToBoxes(
        IGpuBuffer masks,
        IGpuBuffer output,
        int n,
        int height,
        int width)
    {
        if (!DirectPtxVisionSpecializations.TryMasksToBoxes(
                n, height, width, out DirectPtxVisionSpec spec))
            return RejectDirectPtxVisionSpecialization(
                DirectPtxVisionOperation.MasksToBoxes);
        return TryDirectPtxVisionKernel(spec, masks, output);
    }

    internal bool TryDirectPtxVisionIouFamilyBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer boxesA,
        IGpuBuffer boxesB,
        IGpuBuffer gradA,
        IGpuBuffer gradB,
        int n,
        int m,
        int variant)
    {
        if (!DirectPtxVisionSpecializations.TryPairwiseBackward(
                true, n, m, variant, out DirectPtxVisionSpec specA) ||
            !DirectPtxVisionSpecializations.TryPairwiseBackward(
                false, n, m, variant, out DirectPtxVisionSpec specB))
            return RejectDirectPtxVisionSpecialization(
                DirectPtxVisionOperation.IouFamilyBackwardA);
        return TryDirectPtxVisionIouFamilyBackward(
            specA, specB, gradOutput, boxesA, boxesB, gradA, gradB);
    }

    internal bool TryDirectPtxVisionIouFamilyBackward(
        DirectPtxVisionSpec specA,
        DirectPtxVisionSpec specB,
        IGpuBuffer gradOutput,
        IGpuBuffer boxesA,
        IGpuBuffer boxesB,
        IGpuBuffer gradA,
        IGpuBuffer gradB)
    {
        if (!DirectPtxVisionSpecializations.IsAdmitted(specA) ||
            !DirectPtxVisionSpecializations.IsAdmitted(specB) ||
            specA.Operation != DirectPtxVisionOperation.IouFamilyBackwardA ||
            specB.Operation != DirectPtxVisionOperation.IouFamilyBackwardB ||
            specA.D0 != specB.D0 || specA.D1 != specB.D1 || specA.D2 != specB.D2)
        {
            DirectPtxLastError = DirectPtxVisionReason(
                DirectPtxVisionNotAdmittedReasons, specA.Operation);
            return false;
        }
        if (!DirectPtxFeatureGate.IsVisionOperationEnabled(specA.Operation) ||
            !DirectPtxFeatureGate.IsVisionOperationEnabled(specB.Operation))
        {
            DirectPtxLastError = DirectPtxVisionDisabledReasons[(int)specA.Operation];
            return false;
        }
        if (!IsAvailable ||
            !DirectPtxArchitecture.HasValidatedVisionBoxIou(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = !IsAvailable
                ? DirectPtxVisionUnavailableReasons[(int)specA.Operation]
                : DirectPtxVisionArchitectureReasons[(int)specA.Operation];
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxVisionKernels.TryGetValue(specA, out _) ||
                     !_directPtxVisionKernels.TryGetValue(specB, out _)))
                {
                    DirectPtxLastError =
                        "Direct PTX IoU-family backward must prewarm both owner modules before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxVisionKernel kernelA = GetOrCreateVisionKernel(specA);
                PtxVisionKernel kernelB = GetOrCreateVisionKernel(specB);
                DirectPtxTensorView goA = DirectPtxTensorView.Create(gradOutput, kernelA.Blueprint.Tensors[0]);
                DirectPtxTensorView boxesAA = DirectPtxTensorView.Create(boxesA, kernelA.Blueprint.Tensors[1]);
                DirectPtxTensorView boxesBA = DirectPtxTensorView.Create(boxesB, kernelA.Blueprint.Tensors[2]);
                DirectPtxTensorView gradAV = DirectPtxTensorView.Create(gradA, kernelA.Blueprint.Tensors[3]);
                DirectPtxTensorView goB = DirectPtxTensorView.Create(gradOutput, kernelB.Blueprint.Tensors[0]);
                DirectPtxTensorView boxesAB = DirectPtxTensorView.Create(boxesA, kernelB.Blueprint.Tensors[1]);
                DirectPtxTensorView boxesBB = DirectPtxTensorView.Create(boxesB, kernelB.Blueprint.Tensors[2]);
                DirectPtxTensorView gradBV = DirectPtxTensorView.Create(gradB, kernelB.Blueprint.Tensors[3]);

                // Both complete ABIs are validated before either owner writes.
                kernelA.Validate(goA, boxesAA, boxesBA, gradAV);
                kernelB.Validate(goB, boxesAB, boxesBB, gradBV);
                if (capturing &&
                    (!_directPtxVisionKernels.Pin(specA) ||
                     !_directPtxVisionKernels.Pin(specB)))
                    throw new InvalidOperationException(
                        "Could not pin both IoU-family backward modules for capture.");
                lock (GpuDispatchLock)
                {
                    kernelA.LaunchUnchecked(goA, boxesAA, boxesBA, gradAV);
                    kernelB.LaunchUnchecked(goB, boxesAB, boxesBB, gradBV);
                }
            }
            System.Threading.Interlocked.Increment(
                ref _directPtxVisionDispatchCounts[(int)specA.Operation]);
            System.Threading.Interlocked.Increment(
                ref _directPtxVisionDispatchCounts[(int)specB.Operation]);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    bool IDirectPtxVisionBackend.CanDirectPtxMeshgrid2D(int n0, int n1, bool xy)
    {
        if (!IsAvailable ||
            !DirectPtxArchitecture.HasValidatedVisionBoxIou(_ccMajor, _ccMinor) ||
            !DirectPtxFeatureGate.IsVisionOperationEnabled(
                DirectPtxVisionOperation.Meshgrid2D) ||
            !DirectPtxVisionSpecializations.TryMeshgrid2D(
                n0, n1, 0, xy, out DirectPtxVisionSpec spec0) ||
            !DirectPtxVisionSpecializations.TryMeshgrid2D(
                n0, n1, 1, xy, out DirectPtxVisionSpec spec1))
            return false;
        if (!IsStreamCapturing()) return true;
        lock (_directPtxLock)
            return _directPtxVisionKernels.TryGetValue(spec0, out _) &&
                _directPtxVisionKernels.TryGetValue(spec1, out _);
    }

    bool IDirectPtxVisionBackend.TryDirectPtxMeshgrid2DPair(
        IGpuBuffer source0,
        IGpuBuffer source1,
        IGpuBuffer output0,
        IGpuBuffer output1,
        int n0,
        int n1,
        bool xy)
    {
        if (!DirectPtxVisionSpecializations.TryMeshgrid2D(
                n0, n1, 0, xy, out DirectPtxVisionSpec spec0) ||
            !DirectPtxVisionSpecializations.TryMeshgrid2D(
                n0, n1, 1, xy, out DirectPtxVisionSpec spec1))
        {
            DirectPtxLastError =
                "vision-Meshgrid2D-specialization-not-admitted";
            return false;
        }
        if (!DirectPtxFeatureGate.IsVisionOperationEnabled(spec0.Operation))
        {
            DirectPtxLastError = DirectPtxVisionDisabledReasons[(int)spec0.Operation];
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = DirectPtxVisionUnavailableReasons[(int)spec0.Operation];
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedVisionBoxIou(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = DirectPtxVisionArchitectureReasons[(int)spec0.Operation];
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                if (capturing &&
                    (!_directPtxVisionKernels.TryGetValue(spec0, out _) ||
                     !_directPtxVisionKernels.TryGetValue(spec1, out _)))
                {
                    DirectPtxLastError =
                        "Direct PTX meshgrid must prewarm both output modules before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxVisionKernel kernel0 = GetOrCreateVisionKernel(spec0);
                PtxVisionKernel kernel1 = GetOrCreateVisionKernel(spec1);
                DirectPtxTensorView sourceView0 = DirectPtxTensorView.Create(
                    source0, kernel0.Blueprint.Tensors[0]);
                DirectPtxTensorView outputView0 = DirectPtxTensorView.Create(
                    output0, kernel0.Blueprint.Tensors[1]);
                DirectPtxTensorView sourceView1 = DirectPtxTensorView.Create(
                    source1, kernel1.Blueprint.Tensors[0]);
                DirectPtxTensorView outputView1 = DirectPtxTensorView.Create(
                    output1, kernel1.Blueprint.Tensors[1]);
                kernel0.Validate(sourceView0, outputView0);
                kernel1.Validate(sourceView1, outputView1);
                if (capturing &&
                    (!_directPtxVisionKernels.Pin(spec0) ||
                     !_directPtxVisionKernels.Pin(spec1)))
                    throw new InvalidOperationException(
                        "Could not pin both direct-PTX meshgrid modules for capture.");
                lock (GpuDispatchLock)
                {
                    kernel0.LaunchUnchecked(sourceView0, outputView0);
                    kernel1.LaunchUnchecked(sourceView1, outputView1);
                }
            }
            System.Threading.Interlocked.Add(
                ref _directPtxVisionDispatchCounts[(int)spec0.Operation], 2);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxVisionKernel(DirectPtxVisionSpec spec)
    {
        if (!DirectPtxVisionSpecializations.IsAdmitted(spec))
        {
            DirectPtxLastError = DirectPtxVisionReason(
                DirectPtxVisionNotAdmittedReasons, spec.Operation);
            return false;
        }
        if (!DirectPtxFeatureGate.IsVisionOperationEnabled(spec.Operation) || !IsAvailable ||
            !DirectPtxArchitecture.HasValidatedVisionBoxIou(_ccMajor, _ccMinor))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = $"Direct PTX vision prewarm for {spec.Operation} is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateVisionKernel(spec);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private PtxVisionKernel GetOrCreateVisionKernel(DirectPtxVisionSpec spec)
    {
        if (_directPtxVisionKernels.TryGetValue(spec, out PtxVisionKernel? existing))
            return existing;
        return CreateAndCacheVisionKernelSlow(spec);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxVisionKernel CreateAndCacheVisionKernelSlow(DirectPtxVisionSpec spec) =>
        _directPtxVisionKernels.GetOrAdd(spec, () => new PtxVisionKernel(_directPtxRuntime!, spec));

    internal bool TryGetDirectPtxVisionAudit(
        DirectPtxVisionSpec spec, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxVisionKernels.TryGetValue(spec, out PtxVisionKernel? kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private static string[] CreateDirectPtxVisionReasons(string suffix)
    {
        Array values = Enum.GetValues(typeof(DirectPtxVisionOperation));
        var reasons = new string[values.Length];
        foreach (DirectPtxVisionOperation operation in values)
            reasons[(int)operation] = $"vision-{operation}-{suffix}";
        return reasons;
    }

    private static string DirectPtxVisionReason(
        string[] reasons, DirectPtxVisionOperation operation)
    {
        int index = (int)operation;
        return (uint)index < (uint)reasons.Length
            ? reasons[index]
            : "vision-unknown-specialization-not-admitted";
    }
}
#endif
