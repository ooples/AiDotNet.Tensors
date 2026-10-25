using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the capsule-squash direct-PTX kernels (<see cref="PtxFusedSquashF32Kernel"/>, #833). Fails
/// closed to the NVRTC <c>squash</c> / <c>squash_backward</c> kernels for an unsupported shape, architecture or
/// epsilon, and for every shape until the family is promoted (<see cref="DirectPtxFeatureGate.CapsuleSquashPromoted"/>).
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxSquashKey, PtxFusedSquashF32Kernel>
        _directPtxSquashKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxSquashDispatchCount;

    internal long DirectPtxSquashDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSquashDispatchCount);

    private readonly record struct DirectPtxSquashKey(SquashDirection Direction, int NumCapsules, int CapsuleDim);

    internal bool TryDirectPtxSquash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon) =>
        TryDirectPtxSquashCore(SquashDirection.Forward, null, input, output, numCapsules, capsuleDim, epsilon);

    internal bool TryDirectPtxSquashBackward(
        IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon) =>
        TryDirectPtxSquashCore(SquashDirection.Backward, gradOutput, input, gradInput, numCapsules, capsuleDim, epsilon);

    private bool TryDirectPtxSquashCore(
        SquashDirection direction,
        IGpuBuffer? gradOutput,
        IGpuBuffer input,
        IGpuBuffer output,
        int numCapsules,
        int capsuleDim,
        float epsilon)
    {
        if (!DirectPtxFeatureGate.IsCapsuleEnabled)
        {
            DirectPtxLastError = "squash-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "squash-cuda-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.HasValidatedCapsuleSquash(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = "squash-architecture-not-validated";
            return false;
        }
        // The epsilon is baked into the PTX; a caller's other stability term keeps the NVRTC kernel, which takes it.
        if (epsilon != PtxFusedSquashF32Kernel.Epsilon)
        {
            DirectPtxLastError = "squash-epsilon-not-specialized";
            return false;
        }
        if (!PtxFusedSquashF32Kernel.IsSupportedShape(numCapsules, capsuleDim))
        {
            DirectPtxLastError = "squash-shape-not-implemented";
            return false;
        }
        if (!PtxFusedSquashF32Kernel.IsPromotedShape(numCapsules, capsuleDim) &&
            !DirectPtxFeatureGate.CapsuleExperimentOverride)
        {
            DirectPtxLastError = "squash-performance-gate-not-met";
            return false;
        }

        long bytes = checked((long)numCapsules * capsuleDim * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes ||
            (gradOutput is not null && gradOutput.SizeInBytes != bytes))
        {
            DirectPtxLastError = "squash-physical-extent-mismatch";
            return false;
        }

        try
        {
            EnsureContextCurrent();
            var key = new DirectPtxSquashKey(direction, numCapsules, capsuleDim);
            lock (_directPtxLock)
            {
                if (!_directPtxSquashKernels.TryGetValue(key, out PtxFusedSquashF32Kernel? kernel))
                {
                    if (IsStreamCapturing())
                    {
                        DirectPtxLastError = "Direct PTX squash must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
                    kernel = CreateAndCacheSquashKernelSlow(key);
                }
                lock (GpuDispatchLock)
                {
                    if (direction == SquashDirection.Forward)
                        kernel.LaunchForward(
                            DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
                    else
                        kernel.LaunchBackward(
                            DirectPtxTensorView.Create(gradOutput ?? throw new ArgumentNullException(nameof(gradOutput)),
                                kernel.Blueprint.Tensors[0]),
                            DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                            DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxSquashDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedSquashF32Kernel CreateAndCacheSquashKernelSlow(DirectPtxSquashKey key) =>
        _directPtxSquashKernels.GetOrAdd(key, () =>
            new PtxFusedSquashF32Kernel(
                _directPtxRuntime ?? throw new InvalidOperationException("The direct-PTX runtime is not initialized."),
                key.Direction, key.NumCapsules, key.CapsuleDim));

    /// <summary>The audit of a prewarmed squash specialization, or null when it has not been built.</summary>
    internal DirectPtxKernelAudit? GetDirectPtxSquashAudit(SquashDirection direction, int numCapsules, int capsuleDim)
    {
        lock (_directPtxLock)
        {
            return _directPtxSquashKernels.TryGetValue(
                new DirectPtxSquashKey(direction, numCapsules, capsuleDim), out var kernel)
                ? kernel.Audit
                : null;
        }
    }

    private void DisposeDirectPtxSquashKernels() => _directPtxSquashKernels.Dispose();
}
