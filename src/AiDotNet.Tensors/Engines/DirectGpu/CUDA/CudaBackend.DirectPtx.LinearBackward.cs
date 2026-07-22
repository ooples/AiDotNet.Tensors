using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Composite backend dispatch for the FusedLinear*Backward family (issue #836). The
/// gradient of <c>Y = act(X @ transpose(W) + bias)</c> decomposes into four direct-PTX
/// kernels sharing an in-flight dZ scratch:
/// <list type="number">
/// <item>dZ = gradOutput * act'(preActivation) — <see cref="PtxLinearActivationBackwardKernel"/>.</item>
/// <item>gradInput[M,K] = dZ[M,N] @ W[N,K] (contract N) — <see cref="PtxGemmKernel"/>.</item>
/// <item>gradWeight[K,N] = transpose(X[M,K]) @ dZ[M,N] (contract M) — <see cref="PtxGemmContractMKernel"/>
///       with operands swapped so the output is the caller's [inFeatures,outFeatures] layout.</item>
/// <item>gradBias[N] = colsum(dZ) — <see cref="PtxBiasGradientKernel"/>.</item>
/// </list>
/// Fails closed: the composite runs only under the explicit experiment override (never in
/// normal production) and only when every stage supports the shape, so the established
/// NVRTC backward path is unaffected until the whole composite is GPU-validated.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxActBackwardKey, PtxLinearActivationBackwardKernel>
        _directPtxActBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxDWeightKey, PtxGemmContractMKernel>
        _directPtxDWeightKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxDBiasKey, PtxBiasGradientKernel>
        _directPtxDBiasKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxLinearBackwardDispatchCount;

    internal long DirectPtxLinearBackwardDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxLinearBackwardDispatchCount);

    private readonly record struct DirectPtxActBackwardKey(int M, int N, int Activation);
    private readonly record struct DirectPtxDWeightKey(int M, int P, int Q);
    private readonly record struct DirectPtxDBiasKey(int M, int N);

    /// <summary>
    /// Attempts the full direct-PTX linear backward for
    /// <c>act(input[M,K] @ transpose(weight[N,K]) + bias[N])</c>, writing gradInput[M,K],
    /// gradWeight[K,N] (caller [inFeatures,outFeatures] layout) and gradBias[N]. Returns
    /// false (leaving the caller on the NVRTC path) for unsupported shapes, an extent
    /// mismatch, during graph capture, or unless the experiment override is set.
    /// </summary>
    internal bool TryDirectPtxFusedLinearBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer weight,
        IGpuBuffer preActivation,
        IGpuBuffer gradInput,
        IGpuBuffer gradWeight,
        IGpuBuffer gradBias,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        // Composite: every stage must support the shape.
        if (!PtxLinearActivationBackwardKernel.IsSupportedShape(m, n) ||
            !PtxGemmKernel.IsSupportedShape(m, n, k) ||          // dZ[M,N] @ W[N,K] -> [M,K]
            !PtxGemmContractMKernel.IsSupportedShape(m, k, n) || // transpose(X[M,K]) @ dZ[M,N] -> [K,N]
            !PtxBiasGradientKernel.IsSupportedShape(m, n))
        {
            DirectPtxLastError = "linear-backward-shape-not-implemented";
            return false;
        }
        // Fail closed: the composite runs only when explicitly enabled for validation.
        if (!DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "linear-backward-performance-gate-not-met";
            return false;
        }
        long mkBytes = checked((long)m * k * sizeof(float));
        long mnBytes = checked((long)m * n * sizeof(float));
        long nkBytes = checked((long)n * k * sizeof(float));
        long knBytes = checked((long)k * n * sizeof(float));
        long nBytes = checked((long)n * sizeof(float));
        if (gradOutput.SizeInBytes != mnBytes || input.SizeInBytes != mkBytes ||
            weight.SizeInBytes != nkBytes || preActivation.SizeInBytes != mnBytes ||
            gradInput.SizeInBytes != mkBytes || gradWeight.SizeInBytes != knBytes ||
            gradBias.SizeInBytes != nBytes)
        {
            DirectPtxLastError = "linear-backward-physical-extent-mismatch";
            return false;
        }

        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX linear backward composite is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            // dZ scratch [M,N]; disposed once the four launches complete.
            using IGpuBuffer dZ = AllocateBuffer(m * n);
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var actKey = new DirectPtxActBackwardKey(m, n, (int)activation);
                var giKey = new DirectPtxGemmKey(m, n, k);       // PtxGemmKernel(M, K=N, N=K)
                var gwKey = new DirectPtxDWeightKey(m, k, n);    // PtxGemmContractMKernel(M, P=K, Q=N)
                var gbKey = new DirectPtxDBiasKey(m, n);

                var actKernel = GetOrAddActBackward(actKey);
                var giKernel = GetOrAddGemm(giKey);
                var gwKernel = GetOrAddDWeight(gwKey);
                var gbKernel = GetOrAddDBias(gbKey);

                lock (GpuDispatchLock)
                {
                    // 1. dZ = gradOutput * act'(preActivation).
                    actKernel.Launch(
                        DirectPtxTensorView.Create(preActivation, actKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(gradOutput, actKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(dZ, actKernel.Blueprint.Tensors[2]));
                    // 2. gradInput[M,K] = dZ[M,N] @ weight[N,K] (contract N).
                    giKernel.Launch(
                        DirectPtxTensorView.Create(dZ, giKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weight, giKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradInput, giKernel.Blueprint.Tensors[2]));
                    // 3. gradWeight[K,N] = transpose(input[M,K]) @ dZ[M,N]: feed input as the
                    //    "dz" operand (rows over K) and dZ as the "x" operand (cols over N).
                    gwKernel.Launch(
                        DirectPtxTensorView.Create(input, gwKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(dZ, gwKernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gradWeight, gwKernel.Blueprint.Tensors[2]));
                    // 4. gradBias[N] = colsum(dZ).
                    gbKernel.Launch(
                        DirectPtxTensorView.Create(dZ, gbKernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(gradBias, gbKernel.Blueprint.Tensors[1]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxLinearBackwardDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxLinearActivationBackwardKernel GetOrAddActBackward(DirectPtxActBackwardKey key) =>
        _directPtxActBackwardKernels.GetOrAdd(key, () =>
            new PtxLinearActivationBackwardKernel(
                _directPtxRuntime!, key.M, key.N, (DirectPtxLinearActivation)key.Activation));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxGemmKernel GetOrAddGemm(DirectPtxGemmKey key) =>
        _directPtxGemmKernels.GetOrAdd(key, () =>
            new PtxGemmKernel(_directPtxRuntime!, key.M, key.K, key.N));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxGemmContractMKernel GetOrAddDWeight(DirectPtxDWeightKey key) =>
        _directPtxDWeightKernels.GetOrAdd(key, () =>
            new PtxGemmContractMKernel(_directPtxRuntime!, key.M, key.P, key.Q));

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxBiasGradientKernel GetOrAddDBias(DirectPtxDBiasKey key) =>
        _directPtxDBiasKernels.GetOrAdd(key, () =>
            new PtxBiasGradientKernel(_directPtxRuntime!, key.M, key.N));
}
