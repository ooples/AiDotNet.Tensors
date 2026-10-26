// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The FP16 flash-attention forward (#1549), behind the direct-PTX attention opt-in or wherever the caller already chose
/// FP16 (an autocast scope, FP16-resident activations): any sequence lengths, head dimension
/// 64 or 128, causal (with an offset) or not, grouped-query heads; FP32 output and the per-row log-sum-exp the
/// FlashAttention backward reads. Accumulates in FP16 (with the kernel's FP32 redo on overflow) where FP16 accumulation
/// issues faster than FP32 accumulation on this device, else in FP32; AIDOTNET_FP16_ACCUMULATE=0 keeps FP32.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly Dictionary<(int HeadDimension, bool IsCausal, bool Fp16Accumulate), PtxFlashAttentionKernel>
        _directPtxFlashAttentionKernels = new();
    private long _directPtxFlashAttentionDispatchCount;

    /// <summary>Flash-attention forward launches on this backend (test and diagnostics hook).</summary>
    internal long DirectPtxFlashAttentionDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxFlashAttentionDispatchCount);

    internal bool TryDirectPtxFlashAttention(
        IGpuBuffer queryHalf,
        IGpuBuffer keyHalf,
        IGpuBuffer valueHalf,
        IGpuBuffer outputFloat,
        IGpuBuffer softmaxStatsFloat,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int querySequence,
        int keyValueSequence,
        int headDimension,
        float scale,
        bool isCausal,
        int causalQueryOffset = 0,
        bool reducedPrecisionRequested = false)
    {
        // The direct-PTX attention opt-in, or a caller that already chose FP16 (autocast, FP16 activations).
        if (!(_directPtxAttentionOptedIn || reducedPrecisionRequested) || !IsAvailable || _ccMajor < 8)
            return false;
        if (!PtxFlashAttentionGenerator.SupportsHeadDimension(headDimension))
        {
            DirectPtxLastError = "flash-attention-head-dimension-not-64-or-128";
            return false;
        }
        if (batch <= 0 || queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0 ||
            querySequence <= 0 || keyValueSequence <= 0 || (long)batch * queryHeads > 65535)
        {
            DirectPtxLastError = "flash-attention-shape-not-admitted";
            return false;
        }
        if (!isCausal && causalQueryOffset != 0)
        {
            DirectPtxLastError = "causal-query-offset-without-causal-mask";
            return false;
        }
        long queryBytes = (long)batch * queryHeads * querySequence * headDimension * 2;
        long keyValueBytes = (long)batch * keyValueHeads * keyValueSequence * headDimension * 2;
        if (queryHalf.SizeInBytes < queryBytes || keyHalf.SizeInBytes < keyValueBytes || valueHalf.SizeInBytes < keyValueBytes ||
            outputFloat.SizeInBytes < queryBytes * 2 || softmaxStatsFloat.SizeInBytes < (long)batch * queryHeads * querySequence * 4)
        {
            DirectPtxLastError = "flash-attention-buffer-too-small";
            return false;
        }

        // FP16 accumulation only where it issues faster than FP32 accumulation (a GeForce card's half-rate form).
        bool fp16Accumulate = s_fp16AccumulateEnabled && (s_fp16AccumulateForced || MmaRate() is not { Fp32AccumulateIsFullRate: true });
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                var key = (headDimension, isCausal, fp16Accumulate);
                if (!_directPtxFlashAttentionKernels.TryGetValue(key, out var kernel))
                {
                    // Capture may launch only an already-loaded kernel: no module JIT while the stream records.
                    if (capturing)
                    {
                        DirectPtxLastError = "Direct PTX flash attention must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
                    kernel = new PtxFlashAttentionKernel(_directPtxRuntime, headDimension, isCausal, fp16Accumulate,
                        emitStats: true);
                    _directPtxFlashAttentionKernels[key] = kernel;
                }
                lock (GpuDispatchLock)
                    kernel.Launch(queryHalf.Handle, keyHalf.Handle, valueHalf.Handle, outputFloat.Handle,
                        softmaxStatsFloat.Handle, batch, queryHeads, keyValueHeads, querySequence, keyValueSequence,
                        scale, causalQueryOffset);
            }
            System.Threading.Interlocked.Increment(ref _directPtxFlashAttentionDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or NotSupportedException
                                       or PlatformNotSupportedException)
        {
            DirectPtxLastError = $"flash-attention-failed: {ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private void DisposeDirectPtxFlashAttentionKernels()
    {
        lock (_directPtxLock)
        {
            foreach (var kernel in _directPtxFlashAttentionKernels.Values) kernel.Dispose();
            _directPtxFlashAttentionKernels.Clear();
        }
    }
}