using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Differentiable whole-sequence LSTM on the persistent-RNN CUDA kernels (lstm_forward_sequence /
// lstm_backward_sequence): one launch per sequence for the forward and one for the backward, as cuDNN does,
// instead of a per-timestep op chain (the PyTorch-parity LSTM issued ~2,800 kernels per CUDA training step).
public partial class DirectGpuTensorEngine
{

    // Per-layer device caches, keyed by the layer's input-weight tensor: stable across steps (a captured graph
    // bakes the pointers) and never shared between two LSTM layers, so a stacked LSTM's second forward cannot
    // overwrite the first layer's caches before its backward reads them.
    private sealed class LstmTrainCache
    {
        public LstmTrainCache(IDirectGpuBackend backend, int b, int t, int inSize, int h)
        {
            B = b; T = t; In = inSize; H = h;
            AllH = backend.AllocateBuffer((t + 1) * b * h);
            AllC = backend.AllocateBuffer((t + 1) * b * h);
            Gates = backend.AllocateBuffer(t * b * h * 4);
            H0 = backend.AllocateBuffer(b * h);
            C0 = backend.AllocateBuffer(b * h);
            ZeroBias = backend.AllocateBuffer(4 * h);
            FinalH = backend.AllocateBuffer(b * h);
            FinalC = backend.AllocateBuffer(b * h);
            GradH0 = backend.AllocateBuffer(b * h);
            GradC0 = backend.AllocateBuffer(b * h);
            GradBiasHh = backend.AllocateBuffer(4 * h);
            backend.Fill(H0, 0f, b * h);
            backend.Fill(C0, 0f, b * h);
            backend.Fill(ZeroBias, 0f, 4 * h);
        }

        public int B { get; }
        public int T { get; }
        public int In { get; }
        public int H { get; }
        public IGpuBuffer AllH { get; }
        public IGpuBuffer AllC { get; }
        public IGpuBuffer Gates { get; }
        public IGpuBuffer H0 { get; }
        public IGpuBuffer C0 { get; }
        public IGpuBuffer ZeroBias { get; }
        public IGpuBuffer FinalH { get; }
        public IGpuBuffer FinalC { get; }
        public IGpuBuffer GradH0 { get; }
        public IGpuBuffer GradC0 { get; }
        public IGpuBuffer GradBiasHh { get; }

        /// <summary>Counts forwards into these buffers; each forward's backward checks it still owns them.</summary>
        public int Generation { get; set; }
    }

    // The forward a recorded node ran, compared with LstmTrainCache.Generation by its backward. One layer run twice
    // in a step (weight sharing, an unrolled loop) writes the same per-layer buffers twice, so the first call's
    // backward would read the second call's activations: silently wrong gradients. The check turns that into an error.
    private sealed class LstmForwardStamp
    {
        public int Generation { get; set; }
    }

    private static void StampLstmForward(LstmTrainCache cache, LstmForwardStamp stamp)
        => stamp.Generation = ++cache.Generation;

    // The backward kernel accumulates, so its outputs start at zero: a stream memset on CUDA (capturable), a fill on
    // any other backend.
    private static void ZeroLstmGradient(IDirectGpuBackend backend, IGpuBuffer buffer, int length)
    {
        if (backend is Engines.DirectGpu.CUDA.CudaBackend cuda) cuda.MemsetBuffer(buffer, 0, (long)length * sizeof(float));
        else backend.Fill(buffer, 0f, length);
    }

    private readonly ConditionalWeakTable<object, LstmTrainCache> _lstmTrainCaches = new();

    private LstmTrainCache GetLstmTrainCache(IDirectGpuBackend backend, object key, int b, int t, int inSize, int h)
    {
        if (_lstmTrainCaches.TryGetValue(key, out var c) && c.B == b && c.T == t && c.In == inSize && c.H == h)
            return c;
        var created = new LstmTrainCache(backend, b, t, inSize, h);
        // Remove + Add rather than AddOrUpdate, which net471's ConditionalWeakTable lacks.
        lock (_lstmTrainCaches)
        {
            _lstmTrainCaches.Remove(key);
            _lstmTrainCaches.Add(key, created);
        }
        return created;
    }

    /// <summary>
    /// Differentiable LSTM over a whole sequence: input [B, T, in], weights wIh [4H, in] and wHh [4H, H] with gate
    /// rows in PyTorch order (input, forget, cell, output), one bias [4H]; h0 = c0 = 0. Returns the hidden sequence
    /// [B, T, H], or null when this engine cannot run it (non-float, a backend without
    /// <see cref="IFusedLstmSequenceTraining"/>, H above that backend's limit, other shapes, or a lazy
    /// graph being traced) so the caller keeps its decomposed path. Records one tape node whose backward is the BPTT
    /// kernel.
    /// </summary>
    /// <remarks>
    /// An explicit capability check rather than an assumption: every backend exposes
    /// <c>LstmForwardSequence</c>/<c>LstmBackwardSequence</c>, but only those declaring
    /// <see cref="IFusedLstmSequenceTraining"/> are verified against the contract this op needs (batch-major
    /// [B, T, *], PyTorch gate order, full BPTT). Elsewhere the caller's per-timestep ops run instead, on the device.
    /// </remarks>
    public Tensor<T>? TryLstmSequenceTrain<T>(Tensor<T> input, Tensor<T> wIh, Tensor<T> wHh, Tensor<T> bias)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || backend is not IFusedLstmSequenceTraining fused
            || input.Rank != 3 || wIh.Rank != 2 || wHh.Rank != 2 || bias.Length != wIh._shape[0])
            return null;
        int b = input._shape[0], t = input._shape[1], inSize = input._shape[2];
        int gateRows = wIh._shape[0], h = gateRows / 4;
        if (gateRows % 4 != 0 || wIh._shape[1] != inSize || wHh._shape[0] != gateRows || wHh._shape[1] != h
            || h <= 0 || h > fused.MaxFusedLstmHidden || b <= 0 || t <= 0)
            return null;

        // No lazy-graph branch: TryGetBackend refuses the backend while a graph is traced (#350), so a trace never
        // reaches here and records the caller's decomposed ops instead.
        var cache = GetLstmTrainCache(backend, wIh, b, t, inSize, h);
        var stamp = new LstmForwardStamp();
        var state = new object[] { cache, stamp };

        var outputTensor = RunLstmForward(input, wIh, wHh, bias, cache);
        StampLstmForward(cache, stamp);
        Autodiff.DifferentiableOps.RecordIfActive("LstmSequenceTrain", outputTensor,
            new[] { input, wIh, wHh, bias }, LstmSequenceTrainBackward<T>, state);
        return outputTensor;
    }

    private Tensor<T> RunLstmForward<T>(Tensor<T> input, Tensor<T> wIh, Tensor<T> wHh, Tensor<T> bias, LstmTrainCache c)
    {
        var backend = GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var inputC = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
        using var bufInput = GetOrAllocateBuffer(backend, inputC);
        using var bufWih = GetOrAllocateBuffer(backend, wIh);
        using var bufWhh = GetOrAllocateBuffer(backend, wHh);
        using var bufBias = GetOrAllocateBuffer(backend, bias);
        int n = c.B * c.T * c.H;
        var output = AllocateOutputBuffer(backend, n);
        try
        {
            backend.LstmForwardSequence(bufInput.Buffer, c.H0, c.C0, bufWih.Buffer, bufWhh.Buffer, bufBias.Buffer, c.ZeroBias,
                output.Buffer, c.FinalH, c.FinalC, c.AllH, c.AllC, c.Gates, c.T, c.B, c.In, c.H);
            var result = DeferTensorResult<T>(backend, output.Buffer, n, new[] { c.B, c.T, c.H });
            output.RelinquishOwnership();
            return result;
        }
        catch { output.Dispose(); throw; }
    }

    private static void LstmSequenceTrainBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var gpu = (DirectGpuTensorEngine)engine;
        var c = (LstmTrainCache)savedState[0];
        var stamp = (LstmForwardStamp)savedState[1];
        if (stamp.Generation != c.Generation)
        {
            throw new InvalidOperationException(
                "The fused GPU LSTM ran this layer's forward again before this call's backward, and both calls share "
                + "one set of activation buffers, so these gradients would come from the later call. Run one forward "
                + "per layer per step on this path, or train the layer through the per-timestep ops.");
        }
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var input = inputs[0]; var wIh = inputs[1]; var wHh = inputs[2];

        var gradOutC = gradOutput.IsContiguous ? gradOutput : (Tensor<T>)gradOutput.Contiguous();
        using var bufGradOut = gpu.GetOrAllocateBuffer(backend, gradOutC);
        var inputC = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
        using var bufInput = gpu.GetOrAllocateBuffer(backend, inputC);
        using var bufWih = gpu.GetOrAllocateBuffer(backend, wIh);
        using var bufWhh = gpu.GetOrAllocateBuffer(backend, wHh);

        int nIn = c.B * c.T * c.In, nWih = 4 * c.H * c.In, nWhh = 4 * c.H * c.H, nBias = 4 * c.H;
        var gIn = AllocateOutputBuffer(backend, nIn);
        var gWih = AllocateOutputBuffer(backend, nWih);
        var gWhh = AllocateOutputBuffer(backend, nWhh);
        var gBias = AllocateOutputBuffer(backend, nBias);
        // The kernel accumulates every gradient with atomicAdd: zero them first (capturable memset).
        ZeroLstmGradient(backend, gIn.Buffer, nIn);
        ZeroLstmGradient(backend, gWih.Buffer, nWih);
        ZeroLstmGradient(backend, gWhh.Buffer, nWhh);
        ZeroLstmGradient(backend, gBias.Buffer, nBias);
        ZeroLstmGradient(backend, c.GradBiasHh, nBias);

        backend.LstmBackwardSequence(bufGradOut.Buffer, c.AllH, c.AllC, c.Gates, c.H0, c.C0,
            bufWih.Buffer, bufWhh.Buffer, bufInput.Buffer,
            gIn.Buffer, c.GradH0, c.GradC0, gWih.Buffer, gWhh.Buffer, gBias.Buffer, c.GradBiasHh,
            c.T, c.B, c.In, c.H);

        var gradInput = gpu.DeferTensorResult<T>(backend, gIn.Buffer, nIn, new[] { c.B, c.T, c.In }); gIn.RelinquishOwnership();
        var gradWih = gpu.DeferTensorResult<T>(backend, gWih.Buffer, nWih, new[] { 4 * c.H, c.In }); gWih.RelinquishOwnership();
        var gradWhh = gpu.DeferTensorResult<T>(backend, gWhh.Buffer, nWhh, new[] { 4 * c.H, c.H }); gWhh.RelinquishOwnership();
        var gradBias = gpu.DeferTensorResult<T>(backend, gBias.Buffer, nBias, new[] { 4 * c.H }); gBias.RelinquishOwnership();

        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], gradInput, engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[1], gradWih, engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[2], gradWhh, engine);
        Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[3], gradBias, engine);
    }
}
