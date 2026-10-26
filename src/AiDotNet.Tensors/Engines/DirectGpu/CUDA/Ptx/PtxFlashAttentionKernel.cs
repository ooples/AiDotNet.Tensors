// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// A loaded <see cref="PtxFlashAttentionGenerator"/> specialization (head dimension, causal, accumulate form, statistics):
/// FP16 Q [B, Hq, Sq, D], K and V [B, Hkv, Sk, D] in, FP32 O [B, Hq, Sq, D] (and optionally the FP32 logsumexp
/// [B, Hq, Sq]) out, for any sequence lengths (#1549).
/// </summary>
internal sealed class PtxFlashAttentionKernel : IDisposable
{
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly uint _sharedBytes;

    internal int HeadDimension { get; }
    internal bool IsCausal { get; }
    internal bool Fp16Accumulate { get; }
    internal bool EmitStats { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFlashAttentionKernel(DirectPtxRuntime runtime, int headDimension, bool isCausal, bool fp16Accumulate,
        bool emitStats = false)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        HeadDimension = headDimension;
        IsCausal = isCausal;
        Fp16Accumulate = fp16Accumulate;
        EmitStats = emitStats;
        Ptx = PtxFlashAttentionGenerator.Emit(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            headDimension, isCausal, fp16Accumulate, emitStats);
        _sharedBytes = (uint)PtxFlashAttentionGenerator.SharedBytes(headDimension);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, headDimension, isCausal, fp16Accumulate, emitStats);
        DirectPtxLoadedKernel loaded = DirectPtxResourceInitialization.LoadKernel(
            runtime, Ptx, PtxFlashAttentionGenerator.EntryPoint, PtxFlashAttentionGenerator.Threads, Blueprint);
        _module = loaded.Module;
        _function = loaded.Function;
        Audit = loaded.Audit;
    }

    /// <summary>
    /// Attention over device pointers (16-byte aligned): <paramref name="queryHeads"/> must be a multiple of
    /// <paramref name="keyValueHeads"/> (grouped-query attention); <paramref name="scale"/> multiplies Q K^T;
    /// <paramref name="causalOffset"/> lets query row i see keys up to i + offset (causal kernels only).
    /// <paramref name="logSumExp"/> is written when the kernel emits statistics (zero otherwise).
    /// </summary>
    internal unsafe void Launch(IntPtr query, IntPtr key, IntPtr value, IntPtr output, IntPtr logSumExp,
        int batch, int queryHeads, int keyValueHeads, int querySequence, int keyValueSequence, float scale,
        int causalOffset = 0, IntPtr stream = default)
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (queryHeads <= 0) throw new ArgumentOutOfRangeException(nameof(queryHeads));
        if (keyValueHeads <= 0 || queryHeads % keyValueHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueHeads), "Query heads must be a multiple of key/value heads.");
        if (querySequence <= 0) throw new ArgumentOutOfRangeException(nameof(querySequence));
        if (keyValueSequence <= 0) throw new ArgumentOutOfRangeException(nameof(keyValueSequence));
        if ((long)batch * queryHeads > 65535) throw new ArgumentOutOfRangeException(nameof(queryHeads), "B * Hq exceeds grid y.");
        if (!IsCausal && causalOffset != 0) throw new ArgumentException("A causal offset needs a causal kernel.", nameof(causalOffset));
        if (EmitStats && logSumExp == IntPtr.Zero) throw new ArgumentException("This kernel writes logsumexp.", nameof(logSumExp));
        if ((((long)query | (long)key | (long)value | (long)output) & 15) != 0)
            throw new ArgumentException("Flash attention needs 16-byte-aligned operands.");

        ulong q = (ulong)query, k = (ulong)key, v = (ulong)value, o = (ulong)output, lse = (ulong)logSumExp;
        uint sq = (uint)querySequence, sk = (uint)keyValueSequence, hq = (uint)queryHeads;
        uint group = (uint)(queryHeads / keyValueHeads);
        float scaleLog2 = scale * 1.4426950408889634f;
        int offset = causalOffset;
        void** arguments = stackalloc void*[11];
        arguments[0] = &q;
        arguments[1] = &k;
        arguments[2] = &v;
        arguments[3] = &o;
        arguments[4] = &lse;
        arguments[5] = &sq;
        arguments[6] = &sk;
        arguments[7] = &hq;
        arguments[8] = &group;
        arguments[9] = &scaleLog2;
        arguments[10] = &offset;
        uint gridX = (uint)((querySequence + PtxFlashAttentionGenerator.BlockM - 1) / PtxFlashAttentionGenerator.BlockM);
        _module.Launch(_function, gridX, (uint)(batch * queryHeads), 1, PtxFlashAttentionGenerator.Threads, 1, 1,
            _sharedBytes, arguments, stream);
    }

    public void Dispose() => _module.Dispose();

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int headDimension,
        bool isCausal, bool fp16Accumulate, bool emitStats)
    {
        var any = new DirectPtxExtent(1, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "flash-attention",
            Version: 1,
            Architecture: architecture,
            Variant: "d" + headDimension + (isCausal ? "-causal" : "") + (fp16Accumulate ? "-f16acc" : "-f32acc") +
                     (emitStats ? "-lse" : ""),
            Tensors:
            [
                new("q", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("k", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("v", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("o", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    any, any, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.AtLeast),
            ],
            // Shared memory is dynamic (Q plus two K and two V stages), so the static budget is zero.
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 255,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "O = softmax(scale * Q K^T [+ causal mask]) V, per (batch, head); K/V heads shared by groups",
                ["math"] = fp16Accumulate
                    ? "FP16 mma.sync m16n8k16 with FP16 accumulation per 64-key tile, FP32 online softmax and output; a block with a non-finite score or output redoes its rows with FP32 accumulation"
                    : "FP16 mma.sync m16n8k16 with FP32 accumulation, FP32 online softmax and output",
                ["dynamic-shared-bytes"] = PtxFlashAttentionGenerator.SharedBytes(headDimension)
                    .ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["edges"] = "cp.async zero-fill past either sequence; keys past Sk masked; rows past Sq not stored; a row with no visible key stores zeros",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
            });
    }
}