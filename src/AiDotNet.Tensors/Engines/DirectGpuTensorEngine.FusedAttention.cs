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
                if (TryShortAttentionOnDevice(query, key, value, batch, numHeads, seqQ, seqK, headDim,
                        scale ?? 1.0 / Math.Sqrt(headDim), causal, seqK - seqQ, new[] { batch, seqQ, width }) is { } shortForward)
                    return shortForward;
                if (TryFusedAttentionOnDevice("MultiHeadAttentionCore", query, key, value, null, batch, numHeads, seqQ, seqK,
                        headDim, interleaved: true, scale ?? 1.0 / Math.Sqrt(headDim), 0.0, causal,
                        causalOffset: seqK - seqQ, new[] { batch, seqQ, width }) is { } onDevice)
                    return onDevice;
            }
        }
        return base.MultiHeadAttentionCore(query!, key!, value!, numHeads, scale, causal);
    }

    // What a recorded fused-attention node's forward leaves for its backward: the log-sum-exp rows, in one device
    // buffer allocated at the node's first run and reused by every replay (a captured graph bakes its pointer). It
    // also carries the call's parameters, which MultiHeadAttentionPackedFusionPattern reads.
    internal sealed class FusedAttentionRecordedNode
    {
        public Tensor<float>? Lse { get; set; }
        public FusedAttentionParams Params { get; init; }
        public int HeadDim { get; init; }
        public bool Interleaved { get; init; }
    }

    // Under a compiled trace: ONE node whose forward is the fused FlashAttention-2 kernel and whose backward is its
    // device backward. Without it the trace kept the per-head primitive chain (slices, two GEMMs and a softmax per
    // head, forward and backward), which a captured CUDA graph replayed as ~200 kernels a step for a 2-layer encoder.
    // Null (the caller records its decomposed ops) wherever the eager fused path would also decline. Masks are not
    // recorded: the mask upload is host data, and the head-interleaved core has none.
    private Tensor<T>? TryRecordFusedAttention<T>(string opName, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<bool>? mask, int batch, int heads, int seqQ, int seqK, int headDim, bool interleaved, double scale,
        double softcap, bool causal, int causalOffset, int[] outputShape)
    {
        // PlacementBackend, not GetBackend: the latter refuses every backend while a graph is traced (#350).
        if (typeof(T) != typeof(float) || mask is not null || Gpu.AutocastScope.IsEnabled
            || GraphMode.Current is not { } scope
            || PlacementBackend is not IFusedAttentionBackend fused || !fused.SupportsFusedAttention(headDim)
            || (fused is IFusedAttentionPreference preference && !preference.PrefersFusedAttention(batch, heads, seqQ, seqK, headDim))
            || !query.IsContiguous || !key.IsContiguous || !value.IsContiguous
            || batch <= 0 || heads <= 0 || seqQ <= 0 || seqK < 0
            || (long)batch * heads * Math.Max(seqQ, seqK) * headDim > int.MaxValue)
            return null;
        if (!scope.BindEngine(this)) return null;

        var p = FusedParams(batch, heads, seqQ, seqK, headDim, interleaved, scale, softcap, causal, causalOffset);
        var node = new FusedAttentionRecordedNode { Params = p, HeadDim = headDim, Interleaved = interleaved };
        int outLength = batch * heads * seqQ * headDim, rows = batch * heads * seqQ;
        return scope.RecordVariadic(LazyNodeType.Custom, opName, new[] { query, key, value }, outputShape,
            (eng, output) =>
            {
                var gpu = (DirectGpuTensorEngine)eng;
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                var attention = (IFusedAttentionBackend)backend;
                node.Lse ??= Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(rows), new[] { rows },
                    Gpu.GpuTensorRole.Intermediate, ownsBuffer: true);
                using var q = gpu.GetOrAllocateBuffer(backend, query);
                using var k = gpu.GetOrAllocateBuffer(backend, key);
                using var v = gpu.GetOrAllocateBuffer(backend, value);
                var result = AllocateFullyWrittenOutputBuffer(backend, outLength);
                bool ownsResult = true;
                try
                {
                    attention.FusedAttentionForward(q.Buffer, k.Buffer, v.Buffer, result.Buffer,
                        node.Lse._gpuBuffer ?? throw new InvalidOperationException("The attention node lost its log-sum-exp buffer."),
                        null, p, headDim);
                    var tensor = gpu.DeferTensorResult<T>(backend, result.Buffer, outLength, outputShape);
                    result.RelinquishOwnership();
                    ownsResult = false;
                    CopyResultInto(eng, tensor, output);
                }
                finally
                {
                    if (ownsResult) result.Dispose();
                }
            },
            (gradOutput, inputs, output, savedState, engine, grads) =>
            {
                var lse = node.Lse ?? throw new InvalidOperationException("The fused attention node's backward ran before its forward.");
                var gpu = (DirectGpuTensorEngine)engine;
                // In a resident step an accumulator receiving its first contribution here is the kernel's output itself.
                Tensor<T>? FirstWriteTarget(Tensor<T> input) =>
                    Autodiff.DifferentiableOps.PeekDirectDeviceGradTarget(grads, input, engine, out bool add) is { } t && !add
                        ? t : null;
                var intoQ = FirstWriteTarget(inputs[0]);
                var intoK = FirstWriteTarget(inputs[1]);
                var intoV = FirstWriteTarget(inputs[2]);
                var (dQ, dK, dV) = gpu.FusedAttentionBackwardOnDevice(gradOutput, inputs[0], inputs[1], inputs[2], output,
                    lse, null, p, headDim, intoQ, intoK, intoV);
                void Land(Tensor<T> input, Tensor<T>? into, Tensor<T> grad)
                {
                    if (into is not null && ReferenceEquals(grad, into))
                        Autodiff.DifferentiableOps.CommitDirectGradWrite(grads, input, into);
                    else
                        Autodiff.DifferentiableOps.AccumulateGrad(grads, input, grad, engine);
                }
                Land(inputs[0], intoQ, dQ);
                Land(inputs[1], intoK, dK);
                Land(inputs[2], intoV, dV);
            },
            new object[] { node });
    }

    /// <summary>
    /// The packed form of a recorded head-interleaved attention (<see cref="MultiHeadAttentionCore{T}"/>) whose query,
    /// key and value are column blocks of one row-major [rows, 3 * width] projection: one node that reads q, k and v in
    /// place (row stride 3 * width, at <paramref name="qStart"/> / <paramref name="kStart"/> / <paramref name="vStart"/>)
    /// and writes their gradients straight into the projection's gradient, which they exactly cover. The composed form
    /// copied each block out in the forward (three kernels) and, in the backward, built each block's gradient as a
    /// zeroed full-width tensor plus a slice copy, then added the three (three memsets, three copies, two adds).
    /// Null when the recorded node is not one this can replace.
    /// </summary>
    /// <summary>
    /// True when this engine's backend runs the packed node <see cref="BuildPackedAttentionNode"/> builds. The blocks
    /// are viewed per launch at offsets of W floats: only where views are free device-pointer offsets (CUDA, HIP).
    /// Elsewhere a view can be refused for its offset's alignment or allocate, and those keep the split.
    /// </summary>
    internal bool SupportsPackedAttentionFusion
        => PlacementBackend is IGpuBufferViews { ViewsAreFreeAtAnyOffset: true } and IFusedAttentionBackend;

    internal static LazyNode<float>? BuildPackedAttentionNode(LazyNode<float> attention, Tensor<float> packed,
        int qStart, int kStart, int vStart)
    {
        if (attention.SavedState is not { Length: >= 1 } state || state[0] is not FusedAttentionRecordedNode recorded
            || !recorded.Interleaved || attention.RecordingEngine is not DirectGpuTensorEngine engine || packed.Rank != 2
            // The blocks are viewed per launch at offsets of W floats: only where views are free device-pointer offsets
            // (CUDA, HIP). Elsewhere a view can be refused for its offset's alignment or allocate; those keep the split.
            || !engine.SupportsPackedAttentionFusion)
            return null;
        var p0 = recorded.Params;
        int headDim = recorded.HeadDim, width = p0.H * headDim, rowWidth = packed._shape[1];
        if (p0.Sq != p0.Sk || packed._shape[0] != p0.B * p0.Sq || rowWidth != 3 * width) return null;
        var p = p0;
        p.Qb = (long)p.Sq * rowWidth; p.Qr = rowWidth;
        p.Kb = (long)p.Sk * rowWidth; p.Kr = rowWidth;
        p.Vb = p.Kb; p.Vr = rowWidth;
        int outLength = p.B * p.H * p.Sq * headDim, rows = p.B * p.H * p.Sq, packedLength = packed.Length;
        var outputShape = (int[])attention.Output._shape.Clone();
        var node = new FusedAttentionRecordedNode { Params = p, HeadDim = headDim, Interleaved = true };

        // Views of the three blocks; on a refused view the ones already made are disposed. The caller disposes them after
        // its launches.
        (IGpuBuffer Q, IGpuBuffer K, IGpuBuffer V) Blocks(IDirectGpuBackend backend, IGpuBuffer whole)
        {
            var views = backend as IGpuBufferViews ?? throw new NotSupportedException("The backend has no buffer views.");
            var made = new List<IGpuBuffer>(3);
            try
            {
                foreach (int start in new[] { qStart, kStart, vStart })
                    made.Add(views.TryCreateView(whole, start, packedLength - start)
                        ?? throw new NotSupportedException("A packed attention block's view was refused."));
                return (made[0], made[1], made[2]);
            }
            catch
            {
                foreach (var view in made) view.Dispose();
                throw;
            }
        }

        return new LazyNode<float>(LazyNodeType.Custom, "MultiHeadAttentionPacked", new[] { packed }, attention.Output,
            (eng, output) =>
            {
                var gpu = (DirectGpuTensorEngine)eng;
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                var fused = (IFusedAttentionBackend)backend;
                node.Lse ??= Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(rows), new[] { rows },
                    Gpu.GpuTensorRole.Intermediate, ownsBuffer: true);
                using var x = gpu.GetOrAllocateBuffer(backend, packed);
                var (q, k, v) = Blocks(backend, x.Buffer);
                using var qView = q;
                using var kView = k;
                using var vView = v;
                var result = AllocateFullyWrittenOutputBuffer(backend, outLength);
                bool ownsResult = true;
                try
                {
                    fused.FusedAttentionForward(q, k, v, result.Buffer,
                        node.Lse._gpuBuffer ?? throw new InvalidOperationException("The attention node lost its log-sum-exp buffer."),
                        null, p, headDim);
                    var tensor = gpu.DeferTensorResult<float>(backend, result.Buffer, outLength, outputShape);
                    result.RelinquishOwnership();
                    ownsResult = false;
                    CopyResultInto(eng, tensor, output);
                }
                finally
                {
                    if (ownsResult) result.Dispose();
                }
            },
            (gradOutput, inputs, output, savedState, engine, grads) =>
            {
                var lse = node.Lse ?? throw new InvalidOperationException("The packed attention node's backward ran before its forward.");
                var gpu = (DirectGpuTensorEngine)engine;
                var into = Autodiff.DifferentiableOps.PeekDirectDeviceGradTarget(grads, inputs[0], engine, out bool add) is { } t
                    && !add ? t : null;
                var grad = gpu.PackedAttentionBackwardOnDevice(gradOutput, inputs[0], output, lse, p, headDim, Blocks, into);
                if (into is not null && ReferenceEquals(grad, into))
                    Autodiff.DifferentiableOps.CommitDirectGradWrite(grads, inputs[0], into);
                else
                    Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
            },
            new object[] { node })
        {
            RecordingEngine = attention.RecordingEngine,
        };
    }

    // The packed projection's gradient from a packed attention: zeroed whole (dQ accumulates; dK and dV are written),
    // then the attention backward writes each block in place. Into the accumulator when it is receiving its first
    // contribution, else a fresh buffer.
    private Tensor<float> PackedAttentionBackwardOnDevice(Tensor<float> gradOutput, Tensor<float> packed, Tensor<float> output,
        Tensor<float> lse, FusedAttentionParams p, int headDim,
        Func<IDirectGpuBackend, IGpuBuffer, (IGpuBuffer Q, IGpuBuffer K, IGpuBuffer V)> blocks, Tensor<float>? into)
    {
        if (!TryGetBackend(out var backend) || backend is not IFusedAttentionBackend fused)
            throw new InvalidOperationException("The GPU backend that recorded a packed attention node is no longer available.");
        using var x = GetOrAllocateBuffer(backend, packed);
        using var o = GetOrAllocateBuffer(backend, output);
        using var dO = GetOrAllocateBuffer(backend, gradOutput);
        using var l = GetOrAllocateBuffer(backend, lse);
        var target = ResolveDirectGradTarget(backend, into, packed.Length, x.Buffer, o.Buffer, dO.Buffer);
        if (target is not null && target.Handle == l.Buffer.Handle) target = null;
        OwnedBuffer fresh = default;
        bool ownsFresh = false;
        try
        {
            if (target is null) { fresh = AllocateFullyWrittenOutputBuffer(backend, packed.Length); ownsFresh = true; }
            var gradBuffer = target ?? fresh.Buffer;
            var (q, k, v) = blocks(backend, x.Buffer);
            using var qView = q;
            using var kView = k;
            using var vView = v;
            var (dq, dk, dv) = blocks(backend, gradBuffer);
            using var dqView = dq;
            using var dkView = dk;
            using var dvView = dv;
            fused.FusedAttentionBackward(q, k, v, o.Buffer, dO.Buffer, l.Buffer, null, dq, dk, dv, p, headDim, gradBuffer);
            if (into is not null && target is not null)
            {
                into.IncrementVersion();
                BindResidentBuffer(into, target, backend);
                return into;
            }
            var result = DeferTensorResult<float>(backend, fresh.Buffer, packed.Length, (int[])packed._shape.Clone());
            fresh.RelinquishOwnership();
            ownsFresh = false;
            return result;
        }
        finally
        {
            if (ownsFresh) fresh.Dispose();
        }
    }

    // One fused attention on the device, or null when this call cannot take it (the caller then runs its base path).
    private Tensor<T>? TryFusedAttentionOnDevice<T>(string opName, Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<bool>? mask, int batch, int heads, int seqQ, int seqK, int headDim, bool interleaved, double scale,
        double softcap, bool causal, int causalOffset, int[] outputShape)
    {
        if (GraphMode.IsActive)
            return TryRecordFusedAttention(opName, query, key, value, mask, batch, heads, seqQ, seqK, headDim, interleaved,
                scale, softcap, causal, causalOffset, outputShape);
        if (typeof(T) != typeof(float) || Gpu.AutocastScope.IsEnabled
            || !TryGetBackend(out var backend) || backend is not IFusedAttentionBackend fused
            || !fused.SupportsFusedAttention(headDim) || batch <= 0 || heads <= 0 || seqQ <= 0 || seqK < 0
            || (fused is IFusedAttentionPreference preference && !preference.PrefersFusedAttention(batch, heads, seqQ, seqK, headDim))
            || (long)batch * heads * Math.Max(seqQ, seqK) * headDim > int.MaxValue)
            return null;
        if (mask is not null && !TryMaskStrides(mask, batch, heads, seqQ, seqK, out _)) return null;

        OwnedBuffer output = default, lse = default;
        bool ownsOutput = false, ownsLse = false;
        try
        {
            var p = FusedParams(batch, heads, seqQ, seqK, headDim, interleaved, scale, softcap, causal, causalOffset);
            Tensor<int>? maskDevice = mask is null ? null : UploadAttentionMask(backend, mask, batch, heads, seqQ, seqK, ref p);
            using var q = GetOrAllocateBuffer(backend, query);
            using var k = GetOrAllocateBuffer(backend, key);
            using var v = GetOrAllocateBuffer(backend, value);
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

    // Settable for tests, which drive the batched-GEMM and fused branches behind the short-sequence kernel at shapes
    // it would take.
    internal static bool ShortAttentionEnabled { get; set; } = true;

    // Inference only (nothing records a backward): the short-sequence forward where the backend would otherwise compose
    // the attention (permutes, two batched GEMMs, scale, softmax); a ViT-Tiny block's 65-token attention went from six
    // passes over memory to one kernel. Null wherever it does not apply, and the caller continues as before.
    private Tensor<T>? TryShortAttentionOnDevice<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value, int batch, int heads,
        int seqQ, int seqK, int headDim, double scale, bool causal, int causalOffset, int[] outputShape)
    {
        // An inference trace (a compiled Predict) records the kernel as one node; a training or compatibility trace needs
        // the differentiable primitives, so it keeps them.
        if (!ShortAttentionEnabled) return null;
        if (GraphMode.IsActive)
            return GraphMode.IsInferenceTrace
                ? TryRecordShortAttention(query, key, value, batch, heads, seqQ, seqK, headDim, scale, causal, causalOffset, outputShape)
                : null;
        if (typeof(T) != typeof(float) || Autodiff.DifferentiableOps.IsRecording<T>()
            || Gpu.AutocastScope.IsEnabled || !TryGetBackend(out var backend)
            || backend is not IShortAttentionForwardBackend shortAttention || !shortAttention.SupportsShortAttention(headDim)
            || (backend is IFusedAttentionPreference preference && preference.PrefersFusedAttention(batch, heads, seqQ, seqK, headDim))
            || seqK <= 0 || (long)batch * seqQ * heads * headDim > int.MaxValue
            || (long)batch * seqK * heads * headDim > int.MaxValue)
            return null;
        int outLength = batch * seqQ * heads * headDim;
        var output = AllocateFullyWrittenOutputBuffer(backend, outLength);
        bool ownsOutput = true;
        try
        {
            using var q = GetOrAllocateBuffer(backend, query);
            using var k = GetOrAllocateBuffer(backend, key);
            using var v = GetOrAllocateBuffer(backend, value);
            if (!shortAttention.TryShortAttentionForward(q.Buffer, k.Buffer, v.Buffer, output.Buffer, batch, heads, seqQ, seqK,
                    headDim, (float)scale, causal, causalOffset))
                return null;
            var result = DeferTensorResult<T>(backend, output.Buffer, outLength, outputShape);
            output.RelinquishOwnership();
            ownsOutput = false;
            return result;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            GpuLaunchProbe.OnFallback("ShortAttentionForward", ex);
            if (ThrowOnGpuKernelFallback) throw;
            return null;
        }
        finally
        {
            if (ownsOutput) output.Dispose();
        }
    }

    private Tensor<T>? TryRecordShortAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value, int batch, int heads,
        int seqQ, int seqK, int headDim, double scale, bool causal, int causalOffset, int[] outputShape)
    {
        // PlacementBackend, not GetBackend: the latter refuses every backend while a graph is traced (#350).
        if (typeof(T) != typeof(float) || Gpu.AutocastScope.IsEnabled || GraphMode.Current is not { } scope
            || PlacementBackend is not IShortAttentionForwardBackend shortAttention || !shortAttention.SupportsShortAttention(headDim)
            || (PlacementBackend is IFusedAttentionPreference preference
                && preference.PrefersFusedAttention(batch, heads, seqQ, seqK, headDim))
            || seqK <= 0 || (long)batch * Math.Max(seqQ, seqK) * heads * headDim > int.MaxValue)
            return null;
        if (!scope.BindEngine(this)) return null;
        int outLength = batch * seqQ * heads * headDim;
        return scope.RecordVariadic(LazyNodeType.Custom, "ShortAttentionForward", new[] { query, key, value }, outputShape,
            (eng, output) =>
            {
                var gpu = (DirectGpuTensorEngine)eng;
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                var kernel = backend as IShortAttentionForwardBackend
                    ?? throw new InvalidOperationException("The recorded short attention's backend lost the kernel.");
                using var q = gpu.GetOrAllocateBuffer(backend, query);
                using var k = gpu.GetOrAllocateBuffer(backend, key);
                using var v = gpu.GetOrAllocateBuffer(backend, value);
                var result = AllocateFullyWrittenOutputBuffer(backend, outLength);
                bool ownsResult = true;
                try
                {
                    if (!kernel.TryShortAttentionForward(q.Buffer, k.Buffer, v.Buffer, result.Buffer, batch, heads, seqQ, seqK,
                            headDim, (float)scale, causal, causalOffset))
                        throw new InvalidOperationException("The recorded short attention declined at replay.");
                    var tensor = gpu.DeferTensorResult<T>(backend, result.Buffer, outLength, outputShape);
                    result.RelinquishOwnership();
                    ownsResult = false;
                    CopyResultInto(eng, tensor, output);
                }
                finally
                {
                    if (ownsResult) result.Dispose();
                }
            },
            null, null);
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

    /// <summary>dQ, dK and dV of a fused attention on the device. In a resident step, <paramref name="intoQ"/> /
    /// <paramref name="intoK"/> / <paramref name="intoV"/> name gradient accumulators receiving their first contribution
    /// of the step (<see cref="Autodiff.DifferentiableOps.PeekDirectDeviceGradTarget{T}"/>): the kernel writes those
    /// gradients into them (its outputs are overwritten, never accumulated) and returns that accumulator as the
    /// gradient, which the caller then commits instead of copying in.</summary>
    private (Tensor<T> DQ, Tensor<T> DK, Tensor<T> DV) FusedAttentionBackwardOnDevice<T>(Tensor<T> gradOutput,
        Tensor<T> query, Tensor<T> key, Tensor<T> value, Tensor<T> output, Tensor<float> lse, Tensor<int>? mask,
        FusedAttentionParams p, int headDim, Tensor<T>? intoQ = null, Tensor<T>? intoK = null, Tensor<T>? intoV = null)
    {
        if (!TryGetBackend(out var backend) || backend is not IFusedAttentionBackend fused)
            throw new InvalidOperationException("The GPU backend that recorded a fused attention node is no longer available.");
        OwnedBuffer dq = default, dk = default, dv = default;
        bool ownsQ = false, ownsK = false, ownsV = false;
        try
        {
            using var q = GetOrAllocateBuffer(backend, query);
            using var k = GetOrAllocateBuffer(backend, key);
            using var v = GetOrAllocateBuffer(backend, value);
            using var o = GetOrAllocateBuffer(backend, output);
            using var dO = GetOrAllocateBuffer(backend, gradOutput);
            using var l = GetOrAllocateBuffer(backend, lse);
            // A target must differ from every buffer the kernel reads and from the other targets.
            var targetQ = ResolveDirectGradTarget(backend, intoQ, query.Length, q.Buffer, k.Buffer, v.Buffer);
            var targetK = ResolveDirectGradTarget(backend, intoK, key.Length, q.Buffer, k.Buffer, v.Buffer);
            var targetV = ResolveDirectGradTarget(backend, intoV, value.Length, q.Buffer, k.Buffer, v.Buffer);
            IGpuBuffer[] otherReads = { o.Buffer, dO.Buffer, l.Buffer };
            if (targetQ is not null && Array.Exists(otherReads, r => r.Handle == targetQ.Handle)) targetQ = null;
            if (targetK is not null && (Array.Exists(otherReads, r => r.Handle == targetK.Handle)
                || (targetQ is not null && targetK.Handle == targetQ.Handle))) targetK = null;
            if (targetV is not null && (Array.Exists(otherReads, r => r.Handle == targetV.Handle)
                || (targetQ is not null && targetV.Handle == targetQ.Handle)
                || (targetK is not null && targetV.Handle == targetK.Handle))) targetV = null;
            if (targetQ is null) { dq = AllocateFullyWrittenOutputBuffer(backend, query.Length); ownsQ = true; }
            if (targetK is null) { dk = AllocateFullyWrittenOutputBuffer(backend, key.Length); ownsK = true; }
            if (targetV is null) { dv = AllocateFullyWrittenOutputBuffer(backend, value.Length); ownsV = true; }
            fused.FusedAttentionBackward(q.Buffer, k.Buffer, v.Buffer, o.Buffer, dO.Buffer, l.Buffer, mask?._gpuBuffer,
                targetQ ?? dq.Buffer, targetK ?? dk.Buffer, targetV ?? dv.Buffer, p, headDim);
            Tensor<T> Written(Tensor<T>? into, IGpuBuffer? target, ref OwnedBuffer owned, ref bool owns, Tensor<T> input)
            {
                if (into is not null && target is not null)
                {
                    into.IncrementVersion();
                    BindResidentBuffer(into, target, backend);
                    return into;
                }
                var result = DeferTensorResult<T>(backend, owned.Buffer, input.Length, (int[])input._shape.Clone());
                owned.RelinquishOwnership();
                owns = false;
                return result;
            }
            var gq = Written(intoQ, targetQ, ref dq, ref ownsQ, query);
            var gk = Written(intoK, targetK, ref dk, ref ownsK, key);
            var gv = Written(intoV, targetV, ref dv, ref ownsV, value);
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
