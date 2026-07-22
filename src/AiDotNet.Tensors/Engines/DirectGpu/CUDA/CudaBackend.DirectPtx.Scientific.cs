using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the specialized-scientific direct-PTX kernels (issue #854): complex
/// arithmetic, octonion algebra, and Poincaré/Möbius hyperbolic geometry. Every entry point
/// fails closed to the established NVRTC kernel for unsupported shapes/architectures or until
/// a shape is GPU-validated and promoted, so production behavior is unchanged.
/// </summary>
public sealed partial class CudaBackend
{
    internal bool IsDirectPtxScientificEnabled =>
        DirectPtxFeatureGate.IsScientificEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedScientific(_ccMajor, _ccMinor);

    private bool ScientificGateOpen =>
        IsDirectPtxScientificEnabled && DirectPtxFeatureGate.ScientificExperimentOverride;

    private readonly record struct SciCountKey(int Count);
    private readonly record struct SciVectorKey(int Batch, int Dim, int ExtraBits);
    private readonly record struct SciRbfKey(int Batch, int NumCenters, int InputDim);
    private readonly record struct SciPairwiseKey(int M, int N, int Dim, bool Squared);

    private readonly DirectPtxKernelCache<SciCountKey, PtxComplexMultiplyKernel> _sciComplexMul =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciCountKey, PtxComplexConjugateKernel> _sciComplexConj =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciCountKey, PtxComplexMagnitudeKernel> _sciComplexMag =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciCountKey, PtxComplexPhaseKernel> _sciComplexPhase =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciCountKey, PtxOctonionAddKernel> _sciOctAdd =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciCountKey, PtxOctonionMultiplyKernel> _sciOctMul =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciVectorKey, PtxMobiusAddKernel> _sciMobius =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciVectorKey, PtxPoincareDistanceKernel> _sciPoincareDist =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciVectorKey, PtxPoincareProjectKernel> _sciPoincareProj =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciVectorKey, PtxPoincareExpMapKernel> _sciPoincareExp =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciRbfKey, PtxRbfForwardKernel> _sciRbf =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<SciPairwiseKey, PtxPairwiseDistanceKernel> _sciPairwise =
        new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private long _sciDispatchCount;
    internal long DirectPtxScientificDispatchCount => System.Threading.Interlocked.Read(ref _sciDispatchCount);

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    internal bool TryDirectPtxComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int pairs)
    {
        if (!ScientificGateOpen || !PtxComplexMultiplyKernel.IsSupportedPairs(pairs)) return Fail("complex-multiply");
        long bytes = checked((long)pairs * 2 * sizeof(float));
        if (a.SizeInBytes != bytes || b.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("complex-multiply-extent");
        return SciDispatch(() =>
        {
            var k = _sciComplexMul.GetOrAdd(new SciCountKey(pairs), () => new PtxComplexMultiplyKernel(_directPtxRuntime!, pairs));
            Launch3(k.Blueprint, a, b, output, (va, vb, vo) => k.Launch(va, vb, vo));
        });
    }

    internal bool TryDirectPtxComplexConjugate(IGpuBuffer input, IGpuBuffer output, int pairs)
    {
        if (!ScientificGateOpen || !PtxComplexConjugateKernel.IsSupportedPairs(pairs)) return Fail("complex-conjugate");
        long bytes = checked((long)pairs * 2 * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("complex-conjugate-extent");
        return SciDispatch(() =>
        {
            var k = _sciComplexConj.GetOrAdd(new SciCountKey(pairs), () => new PtxComplexConjugateKernel(_directPtxRuntime!, pairs));
            Launch2(k.Blueprint, input, output, (vi, vo) => k.Launch(vi, vo));
        });
    }

    internal bool TryDirectPtxComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer mag, int count)
    {
        if (!ScientificGateOpen || !PtxComplexMagnitudeKernel.IsSupportedCount(count)) return Fail("complex-magnitude");
        long bytes = checked((long)count * sizeof(float));
        if (real.SizeInBytes != bytes || imag.SizeInBytes != bytes || mag.SizeInBytes != bytes) return Fail("complex-magnitude-extent");
        return SciDispatch(() =>
        {
            var k = _sciComplexMag.GetOrAdd(new SciCountKey(count), () => new PtxComplexMagnitudeKernel(_directPtxRuntime!, count));
            Launch3(k.Blueprint, real, imag, mag, (vr, vi, vm) => k.Launch(vr, vi, vm));
        });
    }

    internal bool TryDirectPtxComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int count)
    {
        if (!ScientificGateOpen || !PtxComplexPhaseKernel.IsSupportedCount(count)) return Fail("complex-phase");
        long bytes = checked((long)count * sizeof(float));
        if (real.SizeInBytes != bytes || imag.SizeInBytes != bytes || phase.SizeInBytes != bytes) return Fail("complex-phase-extent");
        return SciDispatch(() =>
        {
            var k = _sciComplexPhase.GetOrAdd(new SciCountKey(count), () => new PtxComplexPhaseKernel(_directPtxRuntime!, count));
            Launch3(k.Blueprint, real, imag, phase, (vr, vi, vp) => k.Launch(vr, vi, vp));
        });
    }

    internal bool TryDirectPtxOctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!ScientificGateOpen || !PtxOctonionAddKernel.IsSupportedCount(count)) return Fail("octonion-add");
        long bytes = checked((long)count * 8 * sizeof(float));
        if (a.SizeInBytes != bytes || b.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("octonion-add-extent");
        return SciDispatch(() =>
        {
            var k = _sciOctAdd.GetOrAdd(new SciCountKey(count), () => new PtxOctonionAddKernel(_directPtxRuntime!, count));
            Launch3(k.Blueprint, a, b, output, (va, vb, vo) => k.Launch(va, vb, vo));
        });
    }

    internal bool TryDirectPtxOctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        if (!ScientificGateOpen || !PtxOctonionMultiplyKernel.IsSupportedCount(count)) return Fail("octonion-multiply");
        long bytes = checked((long)count * 8 * sizeof(float));
        if (a.SizeInBytes != bytes || b.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("octonion-multiply-extent");
        return SciDispatch(() =>
        {
            var k = _sciOctMul.GetOrAdd(new SciCountKey(count), () => new PtxOctonionMultiplyKernel(_directPtxRuntime!, count));
            Launch3(k.Blueprint, a, b, output, (va, vb, vo) => k.Launch(va, vb, vo));
        });
    }

    internal bool TryDirectPtxMobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batch, int dim, float curvature)
    {
        if (!ScientificGateOpen || !PtxMobiusAddKernel.IsSupportedShape(batch, dim)) return Fail("mobius-add");
        long bytes = checked((long)batch * dim * sizeof(float));
        if (x.SizeInBytes != bytes || y.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("mobius-add-extent");
        return SciDispatch(() =>
        {
            var k = _sciMobius.GetOrAdd(new SciVectorKey(batch, dim, Bits(curvature)), () => new PtxMobiusAddKernel(_directPtxRuntime!, batch, dim, curvature));
            Launch3(k.Blueprint, x, y, output, (vx, vy, vo) => k.Launch(vx, vy, vo));
        });
    }

    internal bool TryDirectPtxPoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batch, int dim, float curvature)
    {
        if (!ScientificGateOpen || !PtxPoincareDistanceKernel.IsSupportedShape(batch, dim)) return Fail("poincare-distance");
        if (x.SizeInBytes != checked((long)batch * dim * sizeof(float)) ||
            y.SizeInBytes != checked((long)batch * dim * sizeof(float)) ||
            output.SizeInBytes != checked((long)batch * sizeof(float))) return Fail("poincare-distance-extent");
        return SciDispatch(() =>
        {
            var k = _sciPoincareDist.GetOrAdd(new SciVectorKey(batch, dim, Bits(curvature)), () => new PtxPoincareDistanceKernel(_directPtxRuntime!, batch, dim, curvature));
            Launch3(k.Blueprint, x, y, output, (vx, vy, vo) => k.Launch(vx, vy, vo));
        });
    }

    internal bool TryDirectPtxPoincareProject(IGpuBuffer input, IGpuBuffer output, int batch, int dim, float curvature, float epsilon)
    {
        if (!ScientificGateOpen || !PtxPoincareProjectKernel.IsSupportedShape(batch, dim)) return Fail("poincare-project");
        long bytes = checked((long)batch * dim * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("poincare-project-extent");
        return SciDispatch(() =>
        {
            var k = _sciPoincareProj.GetOrAdd(new SciVectorKey(batch, dim, Bits(curvature) ^ Bits(epsilon)), () => new PtxPoincareProjectKernel(_directPtxRuntime!, batch, dim, curvature, epsilon));
            Launch2(k.Blueprint, input, output, (vi, vo) => k.Launch(vi, vo));
        });
    }

    internal bool TryDirectPtxPoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangent, IGpuBuffer output, int batch, int dim, float curvature)
    {
        if (!ScientificGateOpen || !PtxPoincareExpMapKernel.IsSupportedShape(batch, dim)) return Fail("poincare-exp-map");
        long bytes = checked((long)batch * dim * sizeof(float));
        if (basePoint.SizeInBytes != bytes || tangent.SizeInBytes != bytes || output.SizeInBytes != bytes) return Fail("poincare-exp-map-extent");
        return SciDispatch(() =>
        {
            var k = _sciPoincareExp.GetOrAdd(new SciVectorKey(batch, dim, Bits(curvature)), () => new PtxPoincareExpMapKernel(_directPtxRuntime!, batch, dim, curvature));
            Launch3(k.Blueprint, basePoint, tangent, output, (vb, vt, vo) => k.Launch(vb, vt, vo));
        });
    }

    internal bool TryDirectPtxRbfForward(
        IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output,
        int batchSize, int numCenters, int inputDim)
    {
        if (!ScientificGateOpen || !PtxRbfForwardKernel.IsSupportedShape(batchSize, numCenters, inputDim)) return Fail("rbf-forward");
        if (input.SizeInBytes != checked((long)batchSize * inputDim * sizeof(float)) ||
            centers.SizeInBytes != checked((long)numCenters * inputDim * sizeof(float)) ||
            epsilons.SizeInBytes != checked((long)numCenters * sizeof(float)) ||
            output.SizeInBytes != checked((long)batchSize * numCenters * sizeof(float))) return Fail("rbf-forward-extent");
        return SciDispatch(() =>
        {
            var k = _sciRbf.GetOrAdd(new SciRbfKey(batchSize, numCenters, inputDim),
                () => new PtxRbfForwardKernel(_directPtxRuntime!, batchSize, numCenters, inputDim));
            Launch4(k.Blueprint, input, centers, epsilons, output, (vi, vc, ve, vo) => k.Launch(vi, vc, ve, vo));
        });
    }

    internal bool TryDirectPtxPairwiseDistance(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim, bool squared)
    {
        string tag = squared ? "pairwise-distance-squared" : "pairwise-distance";
        if (!ScientificGateOpen || !PtxPairwiseDistanceKernel.IsSupportedShape(m, n, dim)) return Fail(tag);
        if (a.SizeInBytes != checked((long)m * dim * sizeof(float)) ||
            b.SizeInBytes != checked((long)n * dim * sizeof(float)) ||
            output.SizeInBytes != checked((long)m * n * sizeof(float))) return Fail($"{tag}-extent");
        return SciDispatch(() =>
        {
            var k = _sciPairwise.GetOrAdd(new SciPairwiseKey(m, n, dim, squared),
                () => new PtxPairwiseDistanceKernel(_directPtxRuntime!, m, n, dim, squared));
            Launch3(k.Blueprint, a, b, output, (va, vb, vo) => k.Launch(va, vb, vo));
        });
    }

    private bool Fail(string reason)
    {
        DirectPtxLastError = $"scientific-{reason}-not-eligible";
        return false;
    }

    private void Launch2(DirectPtxKernelBlueprint bp, IGpuBuffer a, IGpuBuffer b, Action<DirectPtxTensorView, DirectPtxTensorView> launch)
    {
        lock (GpuDispatchLock)
            launch(DirectPtxTensorView.Create(a, bp.Tensors[0]), DirectPtxTensorView.Create(b, bp.Tensors[1]));
    }

    private void Launch3(DirectPtxKernelBlueprint bp, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, Action<DirectPtxTensorView, DirectPtxTensorView, DirectPtxTensorView> launch)
    {
        lock (GpuDispatchLock)
            launch(DirectPtxTensorView.Create(a, bp.Tensors[0]), DirectPtxTensorView.Create(b, bp.Tensors[1]), DirectPtxTensorView.Create(c, bp.Tensors[2]));
    }

    private void Launch4(
        DirectPtxKernelBlueprint bp, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, IGpuBuffer d,
        Action<DirectPtxTensorView, DirectPtxTensorView, DirectPtxTensorView, DirectPtxTensorView> launch)
    {
        lock (GpuDispatchLock)
            launch(
                DirectPtxTensorView.Create(a, bp.Tensors[0]),
                DirectPtxTensorView.Create(b, bp.Tensors[1]),
                DirectPtxTensorView.Create(c, bp.Tensors[2]),
                DirectPtxTensorView.Create(d, bp.Tensors[3]));
    }

    private bool SciDispatch(Action launch)
    {
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX scientific kernels must be prewarmed before CUDA graph capture.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                launch();
            }
            System.Threading.Interlocked.Increment(ref _sciDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }
}
