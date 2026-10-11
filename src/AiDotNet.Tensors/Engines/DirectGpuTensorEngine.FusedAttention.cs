using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    /// <remarks>
    /// #1141: float32 on a backend with the fused attention kernels runs one tiled FlashAttention-2 forward on the
    /// device (mask, causal and soft-cap in the kernel, O(S) saved state) and a device backward; anything else takes
    /// the base implementation.
    /// </remarks>
    public override Tensor<T> ScaledDotProductAttention<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<bool>? mask = null, double? scale = null,
        bool isCausal = false, double softcap = 0.0)
    {
        if (query is not null && key is not null && value is not null && query.Rank == 4 && key.Rank == 4 && value.Rank == 4
            && !double.IsNaN(softcap) && softcap >= 0)
        {
            int batch = query._shape[0], heads = query._shape[1], seqQ = query._shape[2], headDim = query._shape[3];
            int seqK = key._shape[2];
            bool consistent = key._shape[0] == batch && key._shape[1] == heads && key._shape[3] == headDim
                && value._shape[0] == batch && value._shape[1] == heads && value._shape[2] == seqK && value._shape[3] == headDim;
            if (consistent && TryFusedAttentionOnDevice("ScaledDotProductAttentionFused", query, key, value, mask,
                    batch, heads, seqQ, seqK, headDim, interleaved: false, scale ?? 1.0 / Math.Sqrt(headDim), softcap,
                    isCausal, causalOffset: 0, new[] { batch, heads, seqQ, headDim }) is { } onDevice)
                return onDevice;
        }
        return base.ScaledDotProductAttention(query!, key!, value!, mask, scale, isCausal, softcap);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// #1141: float32 with equal query / value head widths on a backend with the fused attention kernels reads the
    /// head-interleaved [batch, seq, heads * dim] operands in place (no permutes) and writes the output the same way;
    /// otherwise the per-head primitive composition of the base class, each step of which stays on the device.
    /// Causal masking is bottom-right aligned, as on the CPU.
    /// </remarks>
    public override Tensor<T> MultiHeadAttentionCore<T>(
        Tensor<T> query, Tensor<T> key, Tensor<T> value, int numHeads, double? scale = null, bool causal = false)
    {
        if (query is not null && key is not null && value is not null && query.Rank == 3 && key.Rank == 3 && value.Rank == 3
            && numHeads > 0)
        {
            int batch = query._shape[0], seqQ = query._shape[1], width = query._shape[2], seqK = key._shape[1];
            bool consistent = width % numHeads == 0 && key._shape[0] == batch && value._shape[0] == batch
                && key._shape[2] == width && value._shape[1] == seqK && value._shape[2] == width;
            if (consistent)
            {
                int headDim = width / numHeads;
                if (TryFusedAttentionOnDevice("MultiHeadAttentionCore", query, key, value, null, batch, numHeads, seqQ, seqK,
                        headDim, interleaved: true, scale ?? 1.0 / Math.Sqrt(headDim), 0.0, causal,
                        causalOffset: seqK - seqQ, new[] { batch, seqQ, width }) is { } onDevice)
                    return onDevice;
            }
        }
        return base.MultiHeadAttentionCore(query!, key!, value!, numHeads, scale, causal);
    }

    // One fused attention on the device, or null when this call cannot take it (the caller then runs its base path).
    private Tensor<T>? TryFusedAttentionOnDevice<T>(string opName, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<bool>? mask, int batch, int heads, int seqQ, int seqK, int headDim, bool interleaved, double scale,
        double softcap, bool causal, int causalOffset, int[] outputShape)
    {
        if (typeof(T) != typeof(float) || GraphMode.IsActive || Gpu.AutocastScope.IsEnabled
            || !TryGetBackend(out var backend) || backend is not IFusedAttentionBackend fused
            || !fused.SupportsFusedAttention(headDim) || batch <= 0 || heads <= 0 || seqQ <= 0 || seqK < 0
            || (long)batch * heads * Math.Max(seqQ, seqK) * headDim > int.MaxValue)
            return null;
        if (mask is not null && !TryMaskStrides(mask, batch, heads, seqQ, seqK, out _)) return null;

        OwnedBuffer output = default, lse = default;
        bool ownsOutput = false, ownsLse = false;
        try
        {
            var p = FusedParams(batch, heads, seqQ, seqK, headDim, interleaved, scale, softcap, causal, causalOffset);
            Tensor<int>? maskDevice = mask is null ? null : UploadAttentionMask(backend, mask, batch, heads, seqQ, seqK, ref p);
            using var q = GetOrAllocateBuffer(backend, query.IsContiguous ? query : query.Contiguous());
            using var k = GetOrAllocateBuffer(backend, key.IsContiguous ? key : key.Contiguous());
            using var v = GetOrAllocateBuffer(backend, value.IsContiguous ? value : value.Contiguous());
            int outLength = batch * heads * seqQ * headDim, rows = batch * heads * seqQ;
            output = AllocateFullyWrittenOutputBuffer(backend, outLength);
            ownsOutput = true;
            lse = AllocateFullyWrittenOutputBuffer(backend, rows);
            ownsLse = true;
            fused.FusedAttentionForward(q.Buffer, k.Buffer, v.Buffer, output.Buffer, lse.Buffer, maskDevice?._gpuBuffer, p, headDim);
            var result = DeferTensorResult<T>(backend, output.Buffer, outLength, outputShape);
            output.RelinquishOwnership();
            ownsOutput = false;
            var lseTensor = DeferTensorResult<float>(backend, lse.Buffer, rows, new[] { rows });
            lse.RelinquishOwnership();
            ownsLse = false;
            Autodiff.DifferentiableOps.RecordIfActive(opName, result, new[] { query, key, value }, FusedAttentionDeviceBackward<T>,
                new object[] { p, headDim, lseTensor, maskDevice! });
            return result;
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            if (ThrowOnGpuKernelFallback) throw;
            return null;
        }
        finally
        {
            if (ownsOutput) output.Dispose();
            if (ownsLse) lse.Dispose();
        }
    }

    private static FusedAttentionParams FusedParams(int batch, int heads, int seqQ, int seqK, int headDim, bool interleaved,
        double scale, double softcap, bool causal, int causalOffset)
    {
        var p = new FusedAttentionParams
        {
            B = batch, H = heads, Sq = seqQ, Sk = seqK,
            Scale = (float)scale, Softcap = (float)softcap, Causal = causal ? 1 : 0, CausalOffset = causalOffset,
        };
        if (interleaved)
        {
            // [batch, seq, heads * dim]: head h of row r starts at r * heads * dim + h * dim.
            long width = (long)heads * headDim;
            p.Qb = seqQ * width; p.Qh = headDim; p.Qr = width;
            p.Kb = seqK * width; p.Kh = headDim; p.Kr = width;
            p.Vb = seqK * width; p.Vh = headDim; p.Vr = width;
            p.Ob = seqQ * width; p.Oh = headDim; p.Or = width;
        }
        else
        {
            p.Qb = (long)heads * seqQ * headDim; p.Qh = (long)seqQ * headDim; p.Qr = headDim;
            p.Kb = (long)heads * seqK * headDim; p.Kh = (long)seqK * headDim; p.Kr = headDim;
            p.Vb = p.Kb; p.Vh = p.Kh; p.Vr = headDim;
            p.Ob = p.Qb; p.Oh = p.Qh; p.Or = headDim;
        }
        return p;
    }

    // Strides of a mask that broadcasts to [batch, heads, seqQ, seqK] (right-aligned; 0 on an extent-1 axis).
    private static bool TryMaskStrides(Tensor<bool> mask, int batch, int heads, int seqQ, int seqK, out long[] strides)
    {
        strides = new long[4];
        if (mask.Rank < 2 || mask.Rank > 4) return false;
        var full = new[] { batch, heads, seqQ, seqK };
        long stride = 1;
        for (int d = mask.Rank - 1; d >= 0; d--)
        {
            int axis = 4 - mask.Rank + d, extent = mask._shape[d];
            if (extent != 1 && extent != full[axis]) return false;
            strides[axis] = extent == 1 ? 0 : stride;
            stride *= extent;
        }
        return true;
    }

    // The mask as int32 0 / 1 on the device (row-major over its own extents), with its strides set in p.
    private Tensor<int> UploadAttentionMask(IDirectGpuBackend backend, Tensor<bool> mask, int batch, int heads, int seqQ,
        int seqK, ref FusedAttentionParams p)
    {
        TryMaskStrides(mask, batch, heads, seqQ, seqK, out var s);
        var flat = (mask.IsContiguous ? mask : mask.Contiguous()).GetFlattenedData();
        var words = new int[flat.Length];
        for (int i = 0; i < flat.Length; i++) words[i] = flat[i] ? 1 : 0;
        var buffer = backend.AllocateIntBuffer(words);
        p.HasMask = 1; p.Mb = s[0]; p.Mh = s[1]; p.Mi = s[2]; p.Mj = s[3];
        return Tensor<int>.FromGpuBuffer(backend, buffer, new[] { words.Length }, Gpu.GpuTensorRole.Intermediate,
            ownsBuffer: true, bufferContainsRawInt32: true);
    }

    // Tape backward of the fused attention: dQ, dK and dV on the device from the saved log-sum-exp.
    private static void FusedAttentionDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (engine is not DirectGpuTensorEngine gpu)
            throw new InvalidOperationException("A fused device attention node must be differentiated by the GPU engine that recorded it.");
        var p = (FusedAttentionParams)savedState[0];
        int headDim = (int)savedState[1];
        var lse = (Tensor<float>)savedState[2];
        var mask = savedState[3] as Tensor<int>;
        var (dQ, dK, dV) = gpu.FusedAttentionBackwardOnDevice(gradOutput, inputs[0], inputs[1], inputs[2], output, lse, mask, p, headDim);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], dQ, engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[1], dK, engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[2], dV, engine);
    }

    private (Tensor<T> DQ, Tensor<T> DK, Tensor<T> DV) FusedAttentionBackwardOnDevice<T>(Tensor<T> gradOutput,
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<T> output, Tensor<float> lse, Tensor<int>? mask,
        FusedAttentionParams p, int headDim)
    {
        if (!TryGetBackend(out var backend) || backend is not IFusedAttentionBackend fused)
            throw new InvalidOperationException("The GPU backend that recorded a fused attention node is no longer available.");
        OwnedBuffer dq = default, dk = default, dv = default;
        bool ownsQ = false, ownsK = false, ownsV = false;
        try
        {
            using var q = GetOrAllocateBuffer(backend, query.IsContiguous ? query : query.Contiguous());
            using var k = GetOrAllocateBuffer(backend, key.IsContiguous ? key : key.Contiguous());
            using var v = GetOrAllocateBuffer(backend, value.IsContiguous ? value : value.Contiguous());
            using var o = GetOrAllocateBuffer(backend, output);
            using var dO = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            using var l = GetOrAllocateBuffer(backend, lse);
            dq = AllocateFullyWrittenOutputBuffer(backend, query.Length); ownsQ = true;
            dk = AllocateFullyWrittenOutputBuffer(backend, key.Length); ownsK = true;
            dv = AllocateFullyWrittenOutputBuffer(backend, value.Length); ownsV = true;
            fused.FusedAttentionBackward(q.Buffer, k.Buffer, v.Buffer, o.Buffer, dO.Buffer, l.Buffer, mask?._gpuBuffer,
                dq.Buffer, dk.Buffer, dv.Buffer, p, headDim);
            var gq = DeferTensorResult<T>(backend, dq.Buffer, query.Length, (int[])query._shape.Clone());
            dq.RelinquishOwnership(); ownsQ = false;
            var gk = DeferTensorResult<T>(backend, dk.Buffer, key.Length, (int[])key._shape.Clone());
            dk.RelinquishOwnership(); ownsK = false;
            var gv = DeferTensorResult<T>(backend, dv.Buffer, value.Length, (int[])value._shape.Clone());
            dv.RelinquishOwnership(); ownsV = false;
            return (gq, gk, gv);
        }
        finally
        {
            if (ownsQ) dq.Dispose();
            if (ownsK) dk.Dispose();
            if (ownsV) dv.Dispose();
        }
    }
}
