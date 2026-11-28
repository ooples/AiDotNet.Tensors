// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;

/// <summary>
/// Captures and replays per-segment CUDA Graphs for gradient-checkpointed transformer segments and
/// local-credit block rules (<c>CreditRuleBase</c> / layerwise local objectives), sharing a single
/// segment scratch buffer across all blocks so memory stays <c>O(1)</c> in depth while eliminating
/// per-op kernel launch overhead inside each block.
/// </summary>
public sealed class SegmentedBlockCudaGraphRunner : IDisposable
{
    private readonly IDirectGpuBackend _backend;
    private readonly IGpuBuffer _sharedBlockScratch;
    private readonly StaticShapeCudaGraphScope[] _segmentScopes;
    private bool _disposed;

    public SegmentedBlockCudaGraphRunner(
        IDirectGpuBackend backend,
        int segmentCount,
        int sharedScratchElements,
        StaticShapeCudaGraphOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (segmentCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(segmentCount));
        if (sharedScratchElements <= 0)
            throw new ArgumentOutOfRangeException(nameof(sharedScratchElements));

        var effectiveOptions = options ?? new StaticShapeCudaGraphOptions
        {
            WarmupIterations = 2,
            SynchronizeOnReplay = false
        };
        // Validate before allocating so an invalid option cannot strand the scratch buffer.
        StaticShapeCudaGraphOptions.Validate(effectiveOptions, nameof(options));

        _sharedBlockScratch = _backend.AllocateBuffer(sharedScratchElements);
        _segmentScopes = new StaticShapeCudaGraphScope[segmentCount];
        int created = 0;
        try
        {
            for (; created < segmentCount; created++)
            {
                _segmentScopes[created] = new StaticShapeCudaGraphScope(
                    _backend,
                    CudaGraphExecutionScopeKind.SegmentedLocalCreditBlock,
                    effectiveOptions);
            }
        }
        catch
        {
            for (int i = 0; i < created; i++)
            {
                _segmentScopes[i].Dispose();
            }
            _sharedBlockScratch.Dispose();
            throw;
        }
    }

    public int SegmentCount => _segmentScopes.Length;

    public IGpuBuffer SharedBlockScratch => _sharedBlockScratch;

    public StaticShapeCudaGraphScope GetSegmentScope(int segmentIndex)
    {
        if ((uint)segmentIndex >= (uint)_segmentScopes.Length)
            throw new ArgumentOutOfRangeException(nameof(segmentIndex));
        return _segmentScopes[segmentIndex];
    }

    /// <summary>
    /// Executes segment <paramref name="segmentIndex"/> using the shared block scratch arena and automatic
    /// pointer/shape invalidation guards.
    /// </summary>
    public void ExecuteSegment(
        int segmentIndex,
        Action<IGpuBuffer> segmentAction,
        IReadOnlyList<IGpuBuffer> segmentBindings,
        long segmentSignature = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SegmentedBlockCudaGraphRunner));
        if ((uint)segmentIndex >= (uint)_segmentScopes.Length)
            throw new ArgumentOutOfRangeException(nameof(segmentIndex));
        if (segmentAction is null)
            throw new ArgumentNullException(nameof(segmentAction));
        if (segmentBindings is null)
            throw new ArgumentNullException(nameof(segmentBindings));

        var watched = new IGpuBuffer[1 + segmentBindings.Count];
        watched[0] = _sharedBlockScratch;
        for (int i = 0; i < segmentBindings.Count; i++)
        {
            watched[1 + i] = segmentBindings[i];
        }

        _segmentScopes[segmentIndex].Execute(
            () => segmentAction(_sharedBlockScratch),
            watched,
            segmentSignature);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _segmentScopes.Length; i++)
        {
            _segmentScopes[i].Dispose();
        }
        _sharedBlockScratch.Dispose();
    }
}

/// <summary>
/// 100% device-predicated mask, Top-K selection, adaptive halting, and GPU-side NaN/Inf finiteness flag kernels
/// + CPU deterministic equivalents. Designed for CUDA Graph capture where host <c>.AsSpan()</c> synchronization
/// is strictly forbidden.
/// </summary>
internal static class DevicePredicatedMaskKernels
{
    public const string ModuleName = "aidn_device_predicated_masks";

    public static readonly string[] KernelNames =
    {
        "accum_finite_flag_f32",
        "device_predicated_halting_mask_f32",
        "device_predicated_topk_mask_f32",
        "device_scalar_adamw_update_f32"
    };

    private static string? _lazyCudaSource;

    public static string CudaSource => _lazyCudaSource ??= GeneratedKernels.GeneratedKernelUtf8.Decode(@"
extern ""C"" __global__ void accum_finite_flag_f32(
    const float* __restrict__ data,
    int* __restrict__ is_finite_flag,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    bool localBad = false;
    for (int i = tid; i < n; i += stride) {
        float v = data[i];
        if (isnan(v) || isinf(v)) {
            localBad = true;
            break;
        }
    }
    if (localBad) {
        atomicExch(is_finite_flag, 0);
    }
}

extern ""C"" __global__ void device_predicated_halting_mask_f32(
    const float* __restrict__ halting_scores,
    float* __restrict__ active_mask_out,
    int n,
    float threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        float score = halting_scores[i];
        active_mask_out[i] = (score < threshold) ? 1.0f : 0.0f;
    }
}

extern ""C"" __global__ void device_predicated_topk_mask_f32(
    const float* __restrict__ scores,
    float* __restrict__ topk_mask_out,
    int rows,
    int cols,
    int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int baseIdx = row * cols;
    int effK = (k < cols) ? k : cols;
    // Total order: NaN ranks below every number (including -inf); equal keys (and NaN vs NaN) break ties by lower
    // index, so exactly effK entries are selected per row.
    for (int c = 0; c < cols; c++) {
        float sc = scores[baseIdx + c];
        bool scNan = isnan(sc);
        int greaterCount = 0;
        for (int j = 0; j < cols; j++) {
            float sj = scores[baseIdx + j];
            bool sjNan = isnan(sj);
            bool ranksAbove = sjNan
                ? (scNan && j < c)
                : (scNan || sj > sc || (sj == sc && j < c));
            if (ranksAbove) greaterCount++;
        }
        topk_mask_out[baseIdx + c] = (greaterCount < effK) ? 1.0f : 0.0f;
    }
}

extern ""C"" __global__ void device_scalar_adamw_update_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float* __restrict__ schedule_scalars,
    const int* __restrict__ is_finite_flag,
    int n,
    float eps)
{
    if (is_finite_flag != 0 && is_finite_flag[0] == 0) return;
    float lr = schedule_scalars[0];
    float wd = schedule_scalars[1];
    float beta1 = schedule_scalars[2];
    float beta2 = schedule_scalars[3];
    float bc1 = schedule_scalars[4];
    float bc2 = schedule_scalars[5];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        float g = grad[i];
        float p = param[i];
        float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
        float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
        exp_avg[i] = m;
        exp_avg_sq[i] = v;
        float mHat = m / bc1;
        float vHat = v / bc2;
        param[i] = p * (1.0f - lr * wd) - lr * (mHat / (sqrtf(vHat) + eps));
    }
}
"u8);

    public static void AccumulateFiniteFlagCpu(ReadOnlySpan<float> data, ref int isFiniteFlag)
    {
        if (isFiniteFlag == 0) return;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
            {
                isFiniteFlag = 0;
                return;
            }
        }
    }

    public static void ComputePredicatedHaltingMaskCpu(
        ReadOnlySpan<float> haltingScores,
        Span<float> activeMaskOut,
        float threshold)
    {
        if (activeMaskOut.Length < haltingScores.Length)
            throw new ArgumentException("activeMaskOut is shorter than haltingScores.");
        for (int i = 0; i < haltingScores.Length; i++)
        {
            activeMaskOut[i] = haltingScores[i] < threshold ? 1.0f : 0.0f;
        }
    }

    public static void ComputePredicatedTopKMaskCpu(
        ReadOnlySpan<float> scores,
        Span<float> maskOut,
        int rows,
        int cols,
        int k)
    {
        if (rows <= 0 || cols <= 0) return;
        int total = checked(rows * cols);
        if (scores.Length < total || maskOut.Length < total)
            throw new ArgumentException("Buffers are smaller than rows * cols.");
        int effK = Math.Max(0, Math.Min(k, cols));

        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;
            for (int c = 0; c < cols; c++)
            {
                float sc = scores[baseIdx + c];
                int greater = 0;
                for (int j = 0; j < cols; j++)
                {
                    if (RanksAbove(scores[baseIdx + j], j, sc, c))
                    {
                        greater++;
                    }
                }
                maskOut[baseIdx + c] = greater < effK ? 1.0f : 0.0f;
            }
        }
    }

    /// <summary>
    /// Total Top-K ordering shared with <c>device_predicated_topk_mask_f32</c>: NaN ranks below every number
    /// (including negative infinity), and equal keys (NaN vs NaN included) break ties by lower index, so every row
    /// selects exactly <c>min(k, cols)</c> entries.
    /// </summary>
    internal static bool RanksAbove(float sj, int j, float sc, int c)
    {
        bool sjNan = float.IsNaN(sj);
        bool scNan = float.IsNaN(sc);
        if (sjNan) return scNan && j < c;
        return scNan || sj > sc || (sj == sc && j < c);
    }

    public static void ExecuteDeviceScalarAdamWCpu(
        Span<float> param,
        ReadOnlySpan<float> grad,
        Span<float> expAvg,
        Span<float> expAvgSq,
        ReadOnlySpan<float> scheduleScalars,
        int isFiniteFlag,
        float epsilon = 1e-8f)
    {
        if (isFiniteFlag == 0) return;
        float lr = scheduleScalars[GpuResidentScheduleScalars.LearningRateSlot];
        float wd = scheduleScalars[GpuResidentScheduleScalars.WeightDecaySlot];
        float beta1 = scheduleScalars[GpuResidentScheduleScalars.Beta1Slot];
        float beta2 = scheduleScalars[GpuResidentScheduleScalars.Beta2Slot];
        float bc1 = scheduleScalars[GpuResidentScheduleScalars.BiasCorrection1Slot];
        float bc2 = scheduleScalars[GpuResidentScheduleScalars.BiasCorrection2Slot];

        for (int i = 0; i < param.Length; i++)
        {
            float g = grad[i];
            float p = param[i];
            float m = beta1 * expAvg[i] + (1.0f - beta1) * g;
            float v = beta2 * expAvgSq[i] + (1.0f - beta2) * g * g;
            expAvg[i] = m;
            expAvgSq[i] = v;
            float mHat = m / bc1;
            float vHat = v / bc2;
            param[i] = p * (1.0f - lr * wd) - lr * (mHat / ((float)Math.Sqrt(vHat) + epsilon));
        }
    }
}
