#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly object _directPtxLock = new();
    private readonly DirectPtxKernelCache<DirectPtxAttentionKey, PtxOnlineFusedAttention128x64Kernel>
        _directPtxAttentionKernels = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly DirectPtxPlanCache<DirectPtxAttentionPlanKey, int>
        _directPtxAttentionPlans = new(DirectPtxFeatureGate.CacheCapacity);
    private readonly DirectPtxKernelCache<DirectPtxResidualRmsNormKey, PtxFusedResidualRmsNormD64Kernel>
        _directPtxResidualRmsNormKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private DirectPtxRuntime? _directPtxRuntime;

    /// <summary>The last opt-in direct-PTX initialization/launch failure, if fallback was required.</summary>
    internal string? DirectPtxLastError { get; private set; }
    internal long DirectPtxAttentionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxAttentionDispatchCount);
    private long _directPtxAttentionDispatchCount;
    private long _directPtxResidualRmsNormDispatchCount;
    internal int DirectPtxCachedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxAttentionKernels.Count; }
    }

    internal bool IsDirectPtxAttentionEnabled =>
        DirectPtxFeatureGate.IsAttentionEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedOnlineAttention(
            DirectPtxArchitecture.Classify(_ccMajor, _ccMinor));

    internal bool IsDirectPtxResidualRmsNormEnabled =>
        DirectPtxFeatureGate.IsResidualRmsNormEnabled && IsAvailable &&
        DirectPtxArchitecture.Classify(_ccMajor, _ccMinor) == DirectPtxArchitectureFamily.Ampere;

    internal long DirectPtxResidualRmsNormDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxResidualRmsNormDispatchCount);

    /// <summary>
    /// Attempts the canonical FP16-BHSD S in {16,32,64,128}, D=64 online attention
    /// specialization. All layout and extent checks happen here; the emitted
    /// PTX has no dtype, shape, stride, or storage-offset branches.
    /// </summary>
    internal bool TryDirectPtxOnlineAttention(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer outputFloat,
        IGpuBuffer softmaxStatsFloat,
        int batchHeads,
        float scale,
        bool isCausal,
        int sequenceLength = PtxOnlineFusedAttention128x64Kernel.DefaultSequenceLength)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat: null, betaFloat: null,
            outputFloat, softmaxStatsFloat,
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            scale, isCausal, fuseLayerNormGelu: false, epsilon: 1e-5f,
            emitSoftmaxStats: true, causalQueryOffset: 0);
    }

    /// <summary>
    /// Attempts the baked dense-BHSD FP16 online-attention family for MHA, GQA,
    /// or MQA. Sq and Skv are independent specialization dimensions; the emitted
    /// PTX maps each query head to its KV head without runtime stride/layout checks.
    /// Causal alignment is a baked specialization value: offset zero implements
    /// FlashAttention's top-left convention; Skv-Sq implements SDPA bottom-right.
    /// </summary>
    internal bool TryDirectPtxOnlineAttentionFamily(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool emitSoftmaxStats = true,
        int causalQueryOffset = 0)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat: null, betaFloat: null,
            outputFloat, softmaxStatsFloat,
            batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
            scale, isCausal, fuseLayerNormGelu: false, epsilon: 1e-5f,
            emitSoftmaxStats, causalQueryOffset);
    }

    /// <summary>
    /// Same online attention dataflow with a head-local LayerNorm(D=64),
    /// affine transform, and tanh-GELU fused before the single output store.
    /// </summary>
    internal bool TryDirectPtxOnlineAttentionLayerNormGelu(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer gammaFloat,
        IGpuBuffer betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer softmaxStatsFloat,
        int batchHeads,
        float scale,
        bool isCausal,
        float epsilon = 1e-5f,
        int sequenceLength = PtxOnlineFusedAttention128x64Kernel.DefaultSequenceLength)
    {
        return TryDirectPtxOnlineAttentionCore(
            queryHalf, keyHalf, valueHalf,
            gammaFloat, betaFloat,
            outputFloat, softmaxStatsFloat,
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            scale, isCausal, fuseLayerNormGelu: true, epsilon,
            emitSoftmaxStats: true, causalQueryOffset: 0);
    }

    private bool TryDirectPtxOnlineAttentionCore(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        bool emitSoftmaxStats,
        int causalQueryOffset)
    {
        if (!IsDirectPtxAttentionEnabled)
            return false;
        if (causalQueryOffset < 0)
        {
            DirectPtxLastError = "causal-query-offset-negative-not-implemented";
            return false;
        }
        if (!isCausal && causalQueryOffset != 0)
        {
            DirectPtxLastError = "causal-query-offset-without-causal-mask";
            return false;
        }

        DirectPtxEligibilityResult eligibility = DirectPtxAttentionEligibility.Evaluate(
            new DirectPtxAttentionRequest(
                DirectPtxArchitecture.Classify(_ccMajor, _ccMinor),
                DirectPtxPhysicalType.Float16,
                DirectPtxPhysicalLayout.Bhsd,
                Batch: batch,
                QueryHeads: queryHeads,
                KeyValueHeads: keyValueHeads,
                QuerySequence: querySequence,
                KeyValueSequence: keyValueSequence,
                HeadDimension: PtxOnlineFusedAttention128x64Kernel.HeadDimension,
                Mask: isCausal
                    ? causalQueryOffset == 0
                        ? DirectPtxAttentionMaskKind.CausalTopLeft
                        : DirectPtxAttentionMaskKind.CausalBottomRight
                    : DirectPtxAttentionMaskKind.None,
                Phase: DirectPtxAttentionPhase.Inference,
                DropoutProbability: 0,
                IsRagged: false,
                UsesPagedKv: false));
        if (!eligibility.IsEligible)
        {
            DirectPtxLastError = eligibility.Reason;
            return false;
        }
        try
        {
            bool capturing = IsStreamCapturing();
            // Establish this backend's context once on the calling thread.
            // DirectPtxRuntime detects it and skips redundant push/pop pairs.
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var planKey = new DirectPtxAttentionPlanKey(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    isCausal, causalQueryOffset, fuseLayerNormGelu,
                    emitSoftmaxStats,
                    BitConverter.SingleToInt32Bits(scale),
                    BitConverter.SingleToInt32Bits(epsilon));

                // Capture may launch only an already-loaded in-memory plan. No
                // disk lookup, module JIT, event allocation, or autotune occurs
                // while the stream is recording.
                if (capturing && (!_directPtxAttentionPlans.TryGetValue(planKey, out int captureWarps) ||
                    !_directPtxAttentionKernels.TryGetValue(
                        DirectPtxAttentionKey.FromPlan(planKey, captureWarps), out _)))
                {
                    DirectPtxLastError = "Direct PTX attention must be prewarmed before CUDA graph capture.";
                    return false;
                }

                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);

                if (!_directPtxAttentionPlans.TryGetValue(planKey, out int selectedWarps))
                    selectedWarps = ResolveAttentionPlanSlow(
                        planKey, queryHalf, keyHalf, valueHalf,
                        gammaFloat, betaFloat, outputFloat, softmaxStatsFloat,
                        scale, epsilon);

                PtxOnlineFusedAttention128x64Kernel kernel = GetOrCreateAttentionKernel(
                    planKey, selectedWarps, scale, epsilon);
                lock (GpuDispatchLock)
                    LaunchAttentionKernel(
                        kernel, queryHalf, keyHalf, valueHalf,
                        gammaFloat, betaFloat, outputFloat, softmaxStatsFloat);
            }
            System.Threading.Interlocked.Increment(ref _directPtxAttentionDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            // This is an experimental opt-in path. A bad shape/capability/JIT
            // must preserve the existing CUDA implementation as the fallback.
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    // Keep all closure-bearing autotune code out of the resident dispatch
    // method. C# creates a display object at method entry even when the branch
    // containing a captured lambda is not taken.
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private int ResolveAttentionPlanSlow(
        DirectPtxAttentionPlanKey plan,
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat,
        float scale,
        float epsilon)
    {
        bool persisted = DirectPtxAttentionAutotuner.TryLoad(
            _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
            plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal, plan.CausalQueryOffset,
            plan.FuseLayerNormGelu, plan.EmitSoftmaxStats, scale, epsilon, out int selectedWarps);
        int[] candidates = DirectPtxAttentionAutotuner.Candidates(plan.QuerySequence);
        if (!persisted) selectedWarps = candidates[0];

        if (!persisted && DirectPtxFeatureGate.IsAutotuneEnabled && candidates.Length > 1)
        {
            double bestMilliseconds = double.PositiveInfinity;
            int bestWarps = selectedWarps;
            lock (GpuDispatchLock)
            {
                foreach (int candidate in candidates)
                {
                    PtxOnlineFusedAttention128x64Kernel candidateKernel = GetOrCreateAttentionKernel(
                        plan, candidate, scale, epsilon);
                    float milliseconds = _directPtxRuntime!.MeasureKernelMilliseconds(
                        () => LaunchAttentionKernel(
                            candidateKernel, queryHalf, keyHalf, valueHalf,
                            gammaFloat, betaFloat, outputFloat, softmaxStatsFloat),
                        warmup: 3, iterations: 12);
                    if (milliseconds < bestMilliseconds)
                    {
                        bestMilliseconds = milliseconds;
                        bestWarps = candidate;
                    }
                }
            }
            selectedWarps = bestWarps;
            PtxOnlineFusedAttention128x64Kernel winner = GetOrCreateAttentionKernel(
                plan, selectedWarps, scale, epsilon);
            DirectPtxAttentionAutotuner.Store(
                _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
                plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal, plan.CausalQueryOffset,
                plan.FuseLayerNormGelu, plan.EmitSoftmaxStats, scale, epsilon,
                selectedWarps, bestMilliseconds,
                winner.AttentionTflops((float)bestMilliseconds));
        }
        _directPtxAttentionPlans.Set(plan, selectedWarps);
        return selectedWarps;
    }

    private PtxOnlineFusedAttention128x64Kernel GetOrCreateAttentionKernel(
        DirectPtxAttentionPlanKey plan,
        int warpsPerBlock,
        float scale,
        float epsilon)
    {
        DirectPtxAttentionKey key = DirectPtxAttentionKey.FromPlan(plan, warpsPerBlock);
        if (_directPtxAttentionKernels.TryGetValue(key, out PtxOnlineFusedAttention128x64Kernel? existing))
            return existing;
        return CreateAndCacheAttentionKernelSlow(key, plan, warpsPerBlock, scale, epsilon);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxOnlineFusedAttention128x64Kernel CreateAndCacheAttentionKernelSlow(
        DirectPtxAttentionKey key,
        DirectPtxAttentionPlanKey plan,
        int warpsPerBlock,
        float scale,
        float epsilon)
    {
        return _directPtxAttentionKernels.GetOrAdd(key, () =>
            new PtxOnlineFusedAttention128x64Kernel(
                _directPtxRuntime!, plan.Batch, plan.QueryHeads, plan.KeyValueHeads,
                plan.QuerySequence, plan.KeyValueSequence, plan.IsCausal,
                plan.FuseLayerNormGelu, scale, epsilon,
                plan.EmitSoftmaxStats, warpsPerBlock, plan.CausalQueryOffset));
    }

    private static void LaunchAttentionKernel(
        PtxOnlineFusedAttention128x64Kernel kernel,
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer? gammaFloat,
        IGpuBuffer? betaFloat,
        IGpuBuffer outputFloat,
        IGpuBuffer? softmaxStatsFloat)
    {
        DirectPtxTensorView gamma = default;
        DirectPtxTensorView beta = default;
        if (kernel.FuseLayerNormGelu)
        {
            gamma = DirectPtxTensorView.Create(gammaFloat!, kernel.Blueprint.Tensors[3]);
            beta = DirectPtxTensorView.Create(betaFloat!, kernel.Blueprint.Tensors[4]);
        }
        DirectPtxTensorView stats = kernel.EmitSoftmaxStats
            ? DirectPtxTensorView.Create(softmaxStatsFloat!, kernel.Blueprint.Tensors[6])
            : default;
        kernel.Launch(
            DirectPtxTensorView.Create(queryHalf, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.Create(keyHalf, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.Create(valueHalf, kernel.Blueprint.Tensors[2]),
            gamma,
            beta,
            DirectPtxTensorView.Create(outputFloat, kernel.Blueprint.Tensors[5]),
            stats);
    }

    /// <summary>
    /// Loads a persisted/default specialization outside capture. Stable caller-
    /// owned buffers may subsequently launch it while a CUDA graph is recording.
    /// Autotuning itself requires representative buffers and occurs on first use.
    /// </summary>
    internal bool PrewarmDirectPtxAttention(
        int batchHeads,
        int sequenceLength,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon = 1e-5f,
        bool emitSoftmaxStats = true)
    {
        return PrewarmDirectPtxAttentionFamily(
            batch: 1, queryHeads: batchHeads, keyValueHeads: batchHeads,
            querySequence: sequenceLength, keyValueSequence: sequenceLength,
            isCausal, fuseLayerNormGelu, scale, epsilon, emitSoftmaxStats,
            causalQueryOffset: 0);
    }

    internal bool PrewarmDirectPtxAttentionFamily(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        bool isCausal,
        bool fuseLayerNormGelu,
        float scale,
        float epsilon = 1e-5f,
        bool emitSoftmaxStats = true,
        int causalQueryOffset = 0)
    {
        if (!IsDirectPtxAttentionEnabled || IsStreamCapturing()) return false;
        if (causalQueryOffset < 0)
        {
            DirectPtxLastError = "causal-query-offset-negative-not-implemented";
            return false;
        }
        DirectPtxEligibilityResult eligibility = DirectPtxAttentionEligibility.Evaluate(
            new DirectPtxAttentionRequest(
                DirectPtxArchitecture.Classify(_ccMajor, _ccMinor),
                DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Bhsd,
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                PtxOnlineFusedAttention128x64Kernel.HeadDimension,
                isCausal
                    ? causalQueryOffset == 0
                        ? DirectPtxAttentionMaskKind.CausalTopLeft
                        : DirectPtxAttentionMaskKind.CausalBottomRight
                    : DirectPtxAttentionMaskKind.None,
                DirectPtxAttentionPhase.Inference, 0, false, false));
        if (!eligibility.IsEligible)
        {
            DirectPtxLastError = eligibility.Reason;
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var plan = new DirectPtxAttentionPlanKey(
                    batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                    isCausal, causalQueryOffset, fuseLayerNormGelu, emitSoftmaxStats,
                    BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
                if (!_directPtxAttentionPlans.TryGetValue(plan, out int warps))
                {
                    if (!DirectPtxAttentionAutotuner.TryLoad(
                        _directPtxRuntime, batch, queryHeads, keyValueHeads,
                        querySequence, keyValueSequence, isCausal, causalQueryOffset,
                        fuseLayerNormGelu, emitSoftmaxStats, scale, epsilon, out warps))
                        warps = DirectPtxAttentionAutotuner.Candidates(querySequence)[0];
                    _directPtxAttentionPlans.Set(plan, warps);
                }
                _ = GetOrCreateAttentionKernel(plan, warps, scale, epsilon);
                DirectPtxLastError = null;
                return true;
            }
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Fuses a transformer residual addition and D=64 RMSNorm. Inputs, gamma,
    /// output, and saved RMS are FP32 canonical row-major allocations.
    /// </summary>
    internal bool TryDirectPtxFusedResidualRmsNorm(
        IGpuBuffer input,
        IGpuBuffer residual,
        IGpuBuffer gamma,
        IGpuBuffer output,
        IGpuBuffer savedRms,
        int rows,
        float epsilon = 1e-5f)
    {
        if (!IsDirectPtxResidualRmsNormEnabled || rows <= 0) return false;
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var key = new DirectPtxResidualRmsNormKey(
                    rows, BitConverter.SingleToInt32Bits(epsilon));
                if (capturing && !_directPtxResidualRmsNormKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = "Direct PTX residual RMSNorm must be prewarmed before CUDA graph capture.";
                    return false;
                }

                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedResidualRmsNormD64Kernel kernel =
                    _directPtxResidualRmsNormKernels.GetOrAdd(key, () =>
                        new PtxFusedResidualRmsNormD64Kernel(_directPtxRuntime, rows, epsilon));
                lock (GpuDispatchLock)
                {
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(residual, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(gamma, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(savedRms, kernel.Blueprint.Tensors[4]));
                }
            }
            System.Threading.Interlocked.Increment(ref _directPtxResidualRmsNormDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxFusedResidualRmsNorm(int rows, float epsilon = 1e-5f)
    {
        if (!IsDirectPtxResidualRmsNormEnabled || IsStreamCapturing()) return false;
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxResidualRmsNormKey(rows, BitConverter.SingleToInt32Bits(epsilon));
                _ = _directPtxResidualRmsNormKernels.GetOrAdd(key, () =>
                    new PtxFusedResidualRmsNormD64Kernel(_directPtxRuntime, rows, epsilon));
                return true;
            }
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxResidualRmsNormAudit(
        int rows,
        float epsilon,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxResidualRmsNormKey(rows, BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxResidualRmsNormKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxAttentionAudit(
        int batchHeads,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        int sequenceLength,
        out DirectPtxFunctionInfo info,
        out string jitInfoLog)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                1, batchHeads, batchHeads, sequenceLength, sequenceLength,
                isCausal, 0, fuseLayerNormGelu,
                true,
                BitConverter.SingleToInt32Bits(scale),
                BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                info = kernel.FunctionInfo;
                jitInfoLog = kernel.JitInfoLog;
                return true;
            }
        }
        info = default;
        jitInfoLog = string.Empty;
        return false;
    }

    internal bool TryGetDirectPtxAttentionKernelAudit(
        int batchHeads,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        int sequenceLength,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                1, batchHeads, batchHeads, sequenceLength, sequenceLength,
                isCausal, 0, fuseLayerNormGelu, true,
                BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool TryGetDirectPtxAttentionKernelAuditFamily(
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        float scale,
        bool isCausal,
        bool fuseLayerNormGelu,
        float epsilon,
        bool emitSoftmaxStats,
        int causalQueryOffset,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var plan = new DirectPtxAttentionPlanKey(
                batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                isCausal, causalQueryOffset, fuseLayerNormGelu, emitSoftmaxStats,
                BitConverter.SingleToInt32Bits(scale), BitConverter.SingleToInt32Bits(epsilon));
            if (_directPtxAttentionPlans.TryGetValue(plan, out int warps) &&
                _directPtxAttentionKernels.TryGetValue(
                    DirectPtxAttentionKey.FromPlan(plan, warps), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private void DisposeDirectPtxRuntime()
    {
        lock (_directPtxLock)
        {
            _directPtxAttentionKernels.Dispose();
            _directPtxAttentionPlans.Clear();
            _directPtxResidualRmsNormKernels.Dispose();
            _directPtxRuntime?.Dispose();
            _directPtxRuntime = null;
        }
    }

    private readonly record struct DirectPtxAttentionPlanKey(
        int Batch,
        int QueryHeads,
        int KeyValueHeads,
        int QuerySequence,
        int KeyValueSequence,
        bool IsCausal,
        int CausalQueryOffset,
        bool FuseLayerNormGelu,
        bool EmitSoftmaxStats,
        int ScaleBits,
        int EpsilonBits)
    {
        internal int BatchHeads => checked(Batch * QueryHeads);
    }

    private readonly record struct DirectPtxAttentionKey(
        DirectPtxAttentionPlanKey Plan,
        int WarpsPerBlock)
    {
        internal static DirectPtxAttentionKey FromPlan(
            DirectPtxAttentionPlanKey plan,
            int warpsPerBlock) => new(plan, warpsPerBlock);
    }

    private readonly record struct DirectPtxResidualRmsNormKey(int Rows, int EpsilonBits);

}
#endif
