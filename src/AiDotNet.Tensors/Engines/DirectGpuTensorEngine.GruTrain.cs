// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Compiled training of the GRU sequence (#1573): one recorded node whose forward is the backend's GRU sequence
// (IFusedGruSequence) into per-layer device caches and whose backward is its BPTT over them, as the LSTM's
// TryRecordLstmSequenceTrain. Without it a training trace recorded the host op's lazy node, whose backward replayed the
// forward under a tape: the captured RTX 3080 step ran the forward twice, sliced the last step out with two copy
// kernels, and sat idle 42 us between them (0.123 ms against PyTorch's 0.10).
public partial class DirectGpuTensorEngine
{
    // Per-layer device caches, keyed by the layer's input-weight tensor: stable across steps (a captured graph bakes the
    // pointers) and never shared between two GRU layers.
    private sealed class GruTrainCache
    {
        public GruTrainCache(IDirectGpuBackend backend, int b, int t, int inSize, int h)
        {
            B = b; T = t; In = inSize; H = h;
            AllH = backend.AllocateBuffer(checked((t + 1) * b * h));
            Gates = backend.AllocateBuffer(checked(t * b * 3 * h));
            Ghn = backend.AllocateBuffer(checked(t * b * h));
            H0 = backend.AllocateBuffer(b * h);
            ZeroBias = backend.AllocateBuffer(3 * h);
            GradH0 = backend.AllocateBuffer(b * h);
            GradBiasIh = backend.AllocateBuffer(3 * h);
            GradBiasHh = backend.AllocateBuffer(3 * h);
            backend.Fill(H0, 0f, b * h);
            backend.Fill(ZeroBias, 0f, 3 * h);
        }

        public int B { get; }
        public int T { get; }
        public int In { get; }
        public int H { get; }
        public IGpuBuffer AllH { get; }
        public IGpuBuffer Gates { get; }
        public IGpuBuffer Ghn { get; }
        public IGpuBuffer H0 { get; }
        public IGpuBuffer ZeroBias { get; }
        // Where the backward writes the gradients of inputs the node does not have (h0, a missing bias).
        public IGpuBuffer GradH0 { get; }
        public IGpuBuffer GradBiasIh { get; }
        public IGpuBuffer GradBiasHh { get; }

        /// <summary>The [B, T, H] sequence a last-step node's forward writes but does not return; made on first use.</summary>
        public IGpuBuffer? SequenceScratch { get; set; }

        /// <summary>Counts forwards into these buffers; each forward's backward checks it still owns them.</summary>
        public int Generation { get; set; }
    }

    private readonly ConditionalWeakTable<object, GruTrainCache> _gruTrainCaches = new();

    private GruTrainCache GetGruTrainCache(IDirectGpuBackend backend, object key, int b, int t, int inSize, int h)
    {
        if (_gruTrainCaches.TryGetValue(key, out var c) && c.B == b && c.T == t && c.In == inSize && c.H == h)
            return c;
        var created = new GruTrainCache(backend, b, t, inSize, h);
        lock (_gruTrainCaches)
        {
            _gruTrainCaches.Remove(key);
            _gruTrainCaches.Add(key, created);
        }
        return created;
    }

    // What a recorded node's forward leaves for its backward.
    private sealed class GruRecordedNode
    {
        public GruTrainCache? Cache { get; set; }
        public int Generation { get; set; }
        public bool LastStepOnly { get; init; }
        public GruOptionalInputs Present { get; init; }
    }

    /// <summary>
    /// Under a training trace, records ONE GRU node (reset gate after the recurrent sum) whose forward and backward run
    /// on the device through <see cref="IFusedGruSequence"/>. With <paramref name="lastStepOnly"/> the node's output is
    /// the final hidden state [B, H] and its backward takes that state's gradient alone. Null (the caller records the
    /// host op's node) where the backend lacks the capability or the shape is empty.
    /// </summary>
    private Tensor<T>? TryRecordGruSequenceTrain<T>(LazyTensorScope scope, Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh,
        Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh, bool lastStepOnly)
    {
        // PlacementBackend, not GetBackend: the latter refuses every backend while a graph is traced (#350).
        if (typeof(T) != typeof(float) || PlacementBackend is not IFusedGruSequence) return null;
        int b = input._shape[0], t = input._shape[1], inSize = input._shape[2], h = wHh._shape[1];
        if (b <= 0 || t <= 0 || inSize <= 0 || h <= 0) return null;
        if (!scope.BindEngine(this)) return null;

        var inputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var present);
        var node = new GruRecordedNode { LastStepOnly = lastStepOnly, Present = present };
        return scope.RecordVariadic(LazyNodeType.Custom, lastStepOnly ? "GruSequenceTrainLast" : "GruSequenceTrain", inputs,
            lastStepOnly ? new[] { b, h } : new[] { b, t, h },
            (eng, output) =>
            {
                var gpu = eng.Cast<DirectGpuTensorEngine>();
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                // Resolved when the node runs, with the backend live; keyed by this recorded node (stable across
                // steps), so two calls of one GRU in a step (a siamese encoder, a weight-tied decoder) keep separate
                // activations rather than the second overwriting the first's before its backward.
                var cache = gpu.GetGruTrainCache(backend, node, b, t, inSize, h);
                node.Cache = cache;
                var result = gpu.RunGruTrainForward(backend, inputs, present, cache, lastStepOnly);
                node.Generation = ++cache.Generation;
                CopyResultInto(eng, result, output);
            },
            GruSequenceTrainBackward<T>, new object[] { node });
    }

    private Tensor<T> RunGruTrainForward<T>(IDirectGpuBackend backend, Tensor<T>[] inputs, GruOptionalInputs present,
        GruTrainCache c, bool lastStepOnly)
    {
        var (x, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);
        var xc = x.IsContiguous ? x : x.Contiguous();
        using var bufX = GetOrAllocateBuffer(backend, xc);
        using var bufWih = GetOrAllocateBuffer(backend, wIh);
        using var bufWhh = GetOrAllocateBuffer(backend, wHh);
        using var bufBih = bIh is null ? default : GetOrAllocateBuffer(backend, bIh);
        using var bufBhh = bHh is null ? default : GetOrAllocateBuffer(backend, bHh);
        using var bufH0 = h0 is null ? default : GetOrAllocateBuffer(backend, h0);
        int n = lastStepOnly ? c.B * c.H : c.B * c.T * c.H;
        if (lastStepOnly) c.SequenceScratch ??= backend.AllocateBuffer(c.B * c.T * c.H);
        var output = AllocateFullyWrittenOutputBuffer(backend, n);
        try
        {
            ((IFusedGruSequence)backend).GruForwardSequence(bufX.Buffer, h0 is null ? c.H0 : bufH0.Buffer, bufWih.Buffer,
                bufWhh.Buffer, bIh is null ? c.ZeroBias : bufBih.Buffer, bHh is null ? c.ZeroBias : bufBhh.Buffer,
                lastStepOnly ? c.SequenceScratch! : output.Buffer, c.AllH, c.Gates, c.Ghn, c.T, c.B, c.In, c.H,
                lastStepOnly ? output.Buffer : null);
            var result = DeferTensorResult<T>(backend, output.Buffer, n, lastStepOnly ? new[] { c.B, c.H } : new[] { c.B, c.T, c.H });
            output.RelinquishOwnership();
            return result;
        }
        catch { output.Dispose(); throw; }
    }

    private static void GruSequenceTrainBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var node = (GruRecordedNode)savedState[0];
        var c = node.Cache ?? throw new InvalidOperationException("The fused GPU GRU node's backward ran before its forward.");
        if (node.Generation != c.Generation)
            throw new InvalidOperationException(
                "The fused GPU GRU node ran its forward again before its backward, so its activation buffers hold the "
                + "later call's state and these gradients would come from it.");
        var gpu = engine.Cast<DirectGpuTensorEngine>();
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var (x, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, node.Present);

        var gy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        using var bufGy = gpu.GetOrAllocateBuffer(backend, gy);
        var xc = x.IsContiguous ? x : x.Contiguous();
        using var bufX = gpu.GetOrAllocateBuffer(backend, xc);
        using var bufWih = gpu.GetOrAllocateBuffer(backend, wIh);
        using var bufWhh = gpu.GetOrAllocateBuffer(backend, wHh);

        int g3 = 3 * c.H, nIn = c.B * c.T * c.In, nWih = g3 * c.In, nWhh = g3 * c.H, nState = c.B * c.H;
        // The backward writes every gradient output.
        var gIn = AllocateFullyWrittenOutputBuffer(backend, nIn);
        var gWih = AllocateFullyWrittenOutputBuffer(backend, nWih);
        var gWhh = AllocateFullyWrittenOutputBuffer(backend, nWhh);
        var gBih = bIh is null ? default : AllocateFullyWrittenOutputBuffer(backend, g3);
        var gBhh = bHh is null ? default : AllocateFullyWrittenOutputBuffer(backend, g3);
        var gH0 = h0 is null ? default : AllocateFullyWrittenOutputBuffer(backend, nState);
        ((IFusedGruSequence)backend).GruBackwardSequence(bufGy.Buffer, c.AllH, c.Gates, c.Ghn, bufWih.Buffer, bufWhh.Buffer,
            bufX.Buffer, gIn.Buffer, h0 is null ? c.GradH0 : gH0.Buffer, gWih.Buffer, gWhh.Buffer,
            bIh is null ? c.GradBiasIh : gBih.Buffer, bHh is null ? c.GradBiasHh : gBhh.Buffer, c.T, c.B, c.In, c.H,
            node.LastStepOnly);

        Tensor<T> Result(OwnedBuffer buffer, int length, int[] shape)
        {
            var tensor = gpu.DeferTensorResult<T>(backend, buffer.Buffer, length, shape);
            buffer.RelinquishOwnership();
            return tensor;
        }
        Autodiff.DifferentiableOps.AccumulateGrad(grads, x, Result(gIn, nIn, new[] { c.B, c.T, c.In }), engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, wIh, Result(gWih, nWih, new[] { g3, c.In }), engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, wHh, Result(gWhh, nWhh, new[] { g3, c.H }), engine);
        if (bIh is not null) Autodiff.DifferentiableOps.AccumulateGrad(grads, bIh, Result(gBih, g3, new[] { g3 }), engine);
        if (bHh is not null) Autodiff.DifferentiableOps.AccumulateGrad(grads, bHh, Result(gBhh, g3, new[] { g3 }), engine);
        if (h0 is not null) Autodiff.DifferentiableOps.AccumulateGrad(grads, h0, Result(gH0, nState, new[] { c.B, c.H }), engine);
    }
}
