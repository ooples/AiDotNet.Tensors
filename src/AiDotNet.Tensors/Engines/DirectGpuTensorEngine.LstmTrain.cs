using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Differentiable whole-sequence LSTM on the CUDA training-sequence kernels (CudaBackend.LstmSequenceForwardTrain /
// LstmSequenceBackwardTrain): a persistent recurrence per sequence for the forward and the BPTT, with the weight and
// input gradients as GEMMs over all timesteps, as cuDNN does, instead of a per-timestep op chain (the PyTorch-parity
// LSTM issued ~2,800 kernels per CUDA training step).
public partial class DirectGpuTensorEngine
{
    // Per-layer device caches, keyed by the layer's input-weight tensor: stable across steps (a captured graph
    // bakes the pointers) and never shared between two LSTM layers, so a stacked LSTM's second forward cannot
    // overwrite the first layer's caches before its backward reads them.
    private sealed class LstmTrainCache
    {
        // Buffers for CudaBackend.LstmSequenceForwardTrain / LstmSequenceBackwardTrain (layouts documented there).
        public LstmTrainCache(IDirectGpuBackend backend, int b, int t, int inSize, int h)
        {
            B = b; T = t; In = inSize; H = h;
            AllH = backend.AllocateBuffer((t + 1) * b * h);
            AllC = backend.AllocateBuffer((t + 1) * b * h);
            Gates = backend.AllocateBuffer(t * b * h * 4);
            GradGates = backend.AllocateBuffer(t * b * h * 4);
            PackedWeightsT = backend.AllocateBuffer((inSize + h) * h * 4);
            H0 = backend.AllocateBuffer(b * h);
            C0 = backend.AllocateBuffer(b * h);
            GradH0 = backend.AllocateBuffer(b * h);
            GradC0 = backend.AllocateBuffer(b * h);
            backend.Fill(H0, 0f, b * h);
            backend.Fill(C0, 0f, b * h);
        }

        public int B { get; }
        public int T { get; }
        public int In { get; }
        public int H { get; }
        public IGpuBuffer AllH { get; }
        public IGpuBuffer AllC { get; }
        public IGpuBuffer Gates { get; }
        public IGpuBuffer GradGates { get; }
        public IGpuBuffer PackedWeightsT { get; }
        public IGpuBuffer H0 { get; }
        public IGpuBuffer C0 { get; }
        public IGpuBuffer GradH0 { get; }
        public IGpuBuffer GradC0 { get; }
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
    /// [B, T, H], or null when this engine cannot run it (non-float, non-CUDA, a shape
    /// CudaBackend.CanRunLstmSequenceTrain rejects) so the caller keeps its decomposed path. Records one tape /
    /// lazy-graph node whose backward is the BPTT kernels.
    /// </summary>
    public Tensor<T>? TryLstmSequenceTrain<T>(Tensor<T> input, Tensor<T> wIh, Tensor<T> wHh, Tensor<T> bias)
    {
        if (typeof(T) != typeof(float) || ResolveFusedLstmBackend() is not Engines.DirectGpu.CUDA.CudaBackend backend
            || input.Rank != 3 || wIh.Rank != 2 || wHh.Rank != 2 || bias.Length != wIh._shape[0])
            return null;
        int b = input._shape[0], t = input._shape[1], inSize = input._shape[2];
        int gateRows = wIh._shape[0], h = gateRows / 4;
        if (gateRows % 4 != 0 || wIh._shape[1] != inSize || wHh._shape[0] != gateRows || wHh._shape[1] != h
            || h <= 0 || b <= 0 || t <= 0 || !backend.CanRunLstmSequenceTrain(b, inSize, h))
            return null;

        var cache = GetLstmTrainCache(backend, wIh, b, t, inSize, h);
        var state = new object[] { cache };
        var outShape = new[] { b, t, h };

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var capturedInput = input; var capturedWih = wIh; var capturedWhh = wHh; var capturedBias = bias;
            return scope.RecordVariadic(LazyNodeType.Custom, "LstmSequenceTrain",
                new[] { input, wIh, wHh, bias }, outShape,
                (eng, output) =>
                {
                    var gpu = (DirectGpuTensorEngine)eng;
                    var result = gpu.RunLstmForward(capturedInput, capturedWih, capturedWhh, capturedBias, cache);
                    CopyResultInto(eng, result, output);
                },
                LstmSequenceTrainBackward<T>, state);
        }

        var outputTensor = RunLstmForward(input, wIh, wHh, bias, cache);
        Autodiff.DifferentiableOps.RecordIfActive("LstmSequenceTrain", outputTensor,
            new[] { input, wIh, wHh, bias }, LstmSequenceTrainBackward<T>, state);
        return outputTensor;
    }

    /// <summary>
    /// The CUDA backend for the fused LSTM, INCLUDING while a compiled plan traces. <see cref="TryGetBackend"/>
    /// refuses the backend under GraphMode so that ops which launch kernels directly fall back to CpuEngine's
    /// recording overloads; this op records its own lazy node instead, whose forward and BPTT closures run on the
    /// device when the plan executes. Refusing it here made every compiled CUDA LSTM step trace (and capture) the
    /// per-timestep op chain, about 2,000 kernels per step, instead of two sequence kernels.
    /// </summary>
    private IDirectGpuBackend? ResolveFusedLstmBackend()
    {
        if (!GraphMode.IsActive)
            return GetBackend();
        if (!IsGpuAvailable || _directGpu?.Backend is not { } backend)
            return null;
        (backend as Engines.DirectGpu.CUDA.CudaBackend)?.EnsureContextCurrent();
        return backend;
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
            ((Engines.DirectGpu.CUDA.CudaBackend)backend).LstmSequenceForwardTrain(bufInput.Buffer, c.H0, c.C0,
                bufWih.Buffer, bufWhh.Buffer, bufBias.Buffer, c.PackedWeightsT, output.Buffer, c.AllH, c.AllC, c.Gates,
                c.T, c.B, c.In, c.H);
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
        var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
        var cuda = (Engines.DirectGpu.CUDA.CudaBackend)backend;
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
        // Every output is overwritten by the kernels (no accumulation), so nothing is zeroed first.
        cuda.LstmSequenceBackwardTrain(bufGradOut.Buffer, bufInput.Buffer, bufWih.Buffer, bufWhh.Buffer,
            c.AllH, c.AllC, c.Gates, c.GradGates, gIn.Buffer, gWih.Buffer, gWhh.Buffer, gBias.Buffer,
            c.GradH0, c.GradC0, c.T, c.B, c.In, c.H);

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
