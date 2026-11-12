using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths of the model-level ops (<c>CpuEngine.TorchModelOps</c>) that the CPU computes on the host: the eight
/// pointwise activations, fake quantization, top-k softmax, the sinusoidal time embedding and the CRF log-partition.
/// Each runs its <see cref="TorchModelOpsKernels"/> kernels and, under a tape, records the CPU's node with a backward
/// that also runs on the device (a composition of device ops, or the op's own backward kernels), so a training step
/// stays resident. Graph recording goes through the CPU op, whose replay re-enters these paths.
/// </summary>
/// <remarks>
/// A method returns null (and the override takes the CPU path, which reports the error) for arguments the CPU rejects,
/// for an empty tensor, for precisions the GPU does not compute in, and when the backend lacks the kernels.
/// </remarks>
public partial class DirectGpuTensorEngine
{
    // ---- pointwise activations ----

    private Tensor<T>? TryModelActivationOnDevice<T>(Tensor<T> input, TorchParityUnaryOp op, float p0, float p1,
        Func<double, double> hostDerivative, string opName)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null || input.Length == 0 || !TryGetBackend(out var backend)
            || backend is not ITorchParityElementwiseBackend kernels || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        Tensor<T> result;
        try
        {
            var source = input.IsContiguous ? input : input.Contiguous();
            int n = source.Length;
            using var x = GetOrAllocateBuffer(backend, source);
            result = DispatchDeferredGpuOp<T>(backend, n, (int[])input._shape.Clone(),
                output => kernels.TorchParityUnary(op, x.Buffer, output, n, p0, p1), fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
        DifferentiableOps.RecordUnary(opName, result, input, ModelActivationDeviceBackward<T>,
            new object[] { TorchModelOpsCodes.ActivationCode(op), p0, p1, hostDerivative, opName });
        return result;
    }

    // dx = dy * f'(x) on the device; the host formula only when the device cannot run it.
    private static void ModelActivationDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int code = (int)savedState[0];
        float p0 = (float)savedState[1], p1 = (float)savedState[2];
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            && gpu.TryModelActivationBackwardOnDevice(gradOutput, inputs[0], code, p0, p1) is { } onDevice)
        {
            DifferentiableOps.AccumulateGrad(grads, inputs[0], onDevice, engine);
            return;
        }
        if (engine.Is<DirectGpuTensorEngine>()) GpuLaunchProbe.OnFallback((string)savedState[4] + " backward: host derivative", null);
        var derivative = (Func<double, double>)savedState[3];
        var xs = RoIAlignPlan.ToDoubles(inputs[0]);
        var gs = RoIAlignPlan.ToDoubles(gradOutput);
        var gx = new double[xs.Length];
        for (int i = 0; i < gx.Length; i++) gx[i] = gs[i] * derivative(xs[i]);
        DifferentiableOps.AccumulateGrad(grads, inputs[0], RoIAlignPlan.FromDoubles<T>(gx, (int[])inputs[0]._shape.Clone()), engine);
    }

    private Tensor<T>? TryModelActivationBackwardOnDevice<T>(Tensor<T> gradOutput, Tensor<T> input, int code, float p0, float p1)
    {
        using var __gpuOp = EnterGpuOp();
        if (gradOutput.Length != input.Length || input.Length == 0 || !TryGetBackend(out var backend)
            || backend is not ITorchModelOpsBackend models || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = input.Length;
            using var x = GetOrAllocateBuffer(backend, input.IsContiguous ? input : input.Contiguous());
            using var dy = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            return DispatchDeferredGpuOp<T>(backend, n, (int[])input._shape.Clone(),
                output => models.ModelActivationBackward(code, dy.Buffer, x.Buffer, output, n, p0, p1), fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("model activation backward", ex);
            return null;
        }
    }

    // ---- fake quantization ----

    private Tensor<T>? TryFakeQuantizeOnDevice<T>(Tensor<T> tensor, int numBits, double? scale, int? zeroPoint, bool symmetric)
    {
        using var __gpuOp = EnterGpuOp();
        if (tensor is null || tensor.Length == 0 || numBits < 1 || numBits > 30
            || (scale is { } given && (double.IsNaN(given) || double.IsInfinity(given) || given < 0))
            // A zero point so far outside the range that the clamp bounds leave 32-bit integers: the host computes it.
            || !PortableTorchModelOps.FakeQuantizeFitsDevice(numBits, symmetric, zeroPoint)
            || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        // The kernel reproduces the CPU's double arithmetic exactly (the step is passed as a double), so the integer
        // bounds stay exact up to 30 bits, where a float would round qMax and the quantized values.
        double step = scale ?? 0;
        int n = tensor.Length;
        var shape = (int[])tensor._shape.Clone();
        OwnedBuffer y = default, mask = default;
        bool ownsY = false, ownsMask = false;
        try
        {
            using var x = GetOrAllocateBuffer(backend, tensor.IsContiguous ? tensor : tensor.Contiguous());
            int slots = Math.Min(n, TorchModelOpsCodes.MinMaxSlots);
            using var partial = AllocateTransientScratch(backend, 2 * slots, fullyWritten: true);
            using var minMax = AllocateTransientScratch(backend, 2, fullyWritten: true);
            y = AllocateFullyWrittenOutputBuffer(backend, n);
            ownsY = true;
            mask = AllocateFullyWrittenOutputBuffer(backend, n);
            ownsMask = true;
            models.ModelFakeQuantize(x.Buffer, partial.Buffer, minMax.Buffer, y.Buffer, mask.Buffer, n, numBits, symmetric,
                step, zeroPoint);
            var result = DeferTensorResult<T>(backend, y.Buffer, n, shape);
            y.RelinquishOwnership();
            ownsY = false;
            var passMask = DeferTensorResult<T>(backend, mask.Buffer, n, (int[])shape.Clone());
            mask.RelinquishOwnership();
            ownsMask = false;
            // The CPU's node and saved state: the backward multiplies by the (device-resident) pass-through mask.
            DifferentiableOps.RecordUnary("TensorFakeQuantize", result, tensor, BackwardFunctions<T>.MaskedPassThroughBackward,
                new object[] { passMask });
            return result;
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorFakeQuantize", ex);
            return null;
        }
        finally
        {
            if (ownsY) y.Dispose();
            if (ownsMask) mask.Dispose();
        }
    }

    // ---- top-k softmax ----

    private Tensor<T>? TryTopKSoftmaxOnDevice<T>(Tensor<T> scores, int k)
    {
        using var __gpuOp = EnterGpuOp();
        if (scores is null || scores.Rank == 0 || scores.Length == 0) return null;
        int width = scores._shape[scores.Rank - 1];
        if (k < 1 || k > width || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        Tensor<T> result;
        try
        {
            int n = scores.Length, rows = n / width;
            using var x = GetOrAllocateBuffer(backend, scores.IsContiguous ? scores : scores.Contiguous());
            var plan = TorchModelOpsCodes.PlanTopK(rows, width, k);
            // The direct plan reads only the values scratch; the row-group plan only the (int) counts and the weights.
            using var values = AllocateTransientScratch(backend, plan.RowGroups ? 1 : plan.ScratchLength, fullyWritten: true);
            using var indices = AllocateTransientScratch(backend, plan.RowGroups ? plan.ScratchLength : 1, fullyWritten: true);
            using var weights = AllocateTransientScratch(backend, plan.RowGroups ? plan.ScratchLength : 1, fullyWritten: true);
            result = DispatchDeferredGpuOp<T>(backend, n, (int[])scores._shape.Clone(),
                output => models.ModelTopKSoftmax(x.Buffer, values.Buffer, indices.Buffer, weights.Buffer, output, rows, width, k),
                fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorTopKSoftmax", ex);
            return null;
        }
        // The CPU's backward: y (dy - sum(dy y)), engine ops that run on the device.
        DifferentiableOps.RecordUnary("TensorTopKSoftmax", result, scores, BackwardFunctions<T>.TopKSoftmaxBackward);
        return result;
    }

    // ---- sinusoidal time embedding ----

    // The CPU's frequencies (double) and their three-float device split.
    private static (double[] Frequencies, float[] Split) TimeEmbeddingFrequencies(int half)
    {
        double logTimescale = Math.Log(10000.0) / Math.Max(half - 1, 1);
        var freq = new double[half];
        var split = new float[Math.Max(3, 3 * half)];
        for (int i = 0; i < half; i++)
        {
            freq[i] = Math.Exp(-logTimescale * i);
            TorchModelOpsCodes.SplitFrequency(freq[i], split, 3 * i);
        }
        return (freq, split);
    }

    private Tensor<T>? TrySinusoidalTimeEmbeddingOnDevice<T>(Tensor<T> timesteps, int embeddingDim)
    {
        using var __gpuOp = EnterGpuOp();
        if (timesteps is null || embeddingDim < 1 || (timesteps.Rank != 1 && !(timesteps.Rank == 2 && timesteps._shape[1] == 1))
            || timesteps.Length == 0 || (long)timesteps._shape[0] * embeddingDim > int.MaxValue
            || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        int batch = timesteps._shape[0], half = embeddingDim / 2;
        var (freq, split) = TimeEmbeddingFrequencies(half);
        Tensor<T> result;
        try
        {
            using var t = GetOrAllocateBuffer(backend, timesteps.IsContiguous ? timesteps : timesteps.Contiguous());
            using var f = new OwnedBuffer(backend.AllocateBuffer(split), ownsBuffer: true);
            result = DispatchDeferredGpuOp<T>(backend, batch * embeddingDim, new[] { batch, embeddingDim },
                output => models.ModelTimeEmbedding(t.Buffer, f.Buffer, output, batch, half, embeddingDim), fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorSinusoidalTimeEmbedding", ex);
            return null;
        }
        // The CPU's saved state (frequencies, dim) plus the device split, read by the device backward.
        DifferentiableOps.RecordUnary("TensorSinusoidalTimeEmbedding", result, timesteps, SinusoidalTimeEmbeddingDeviceBackward<T>,
            new object[] { freq, embeddingDim, split });
        return result;
    }

    private static void SinusoidalTimeEmbeddingDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            && gpu.TrySinusoidalTimeEmbeddingBackwardOnDevice(gradOutput, inputs[0], ((double[])savedState[0]).Length,
                (int)savedState[1], (float[])savedState[2]) is { } onDevice)
        {
            DifferentiableOps.AccumulateGrad(grads, inputs[0], onDevice, engine);
            return;
        }
        if (engine.Is<DirectGpuTensorEngine>()) GpuLaunchProbe.OnFallback("TensorSinusoidalTimeEmbedding backward: host sum", null);
        BackwardFunctions<T>.SinusoidalTimeEmbeddingBackward(gradOutput, inputs, output, savedState, engine, grads);
    }

    private Tensor<T>? TrySinusoidalTimeEmbeddingBackwardOnDevice<T>(Tensor<T> gradOutput, Tensor<T> timesteps, int half, int dim,
        float[] split)
    {
        using var __gpuOp = EnterGpuOp();
        int batch = timesteps._shape[0];
        if (gradOutput.Length != (long)batch * dim || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            using var t = GetOrAllocateBuffer(backend, timesteps.IsContiguous ? timesteps : timesteps.Contiguous());
            using var dy = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            using var f = new OwnedBuffer(backend.AllocateBuffer(split), ownsBuffer: true);
            return DispatchDeferredGpuOp<T>(backend, batch, (int[])timesteps._shape.Clone(),
                output => models.ModelTimeEmbeddingBackward(t.Buffer, f.Buffer, dy.Buffer, output, batch, half, dim), fullyWritten: true);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorSinusoidalTimeEmbedding backward", ex);
            return null;
        }
    }

    // ---- linear-chain CRF ----

    private Tensor<T>? TryCrfForwardOnDevice<T>(Tensor<T> emissions, Tensor<T> transitions, Tensor<T>? startScores, Tensor<T>? endScores)
    {
        using var __gpuOp = EnterGpuOp();
        if (emissions is null || transitions is null || (emissions.Rank != 2 && emissions.Rank != 3)) return null;
        bool batched = emissions.Rank == 3;
        int batch = batched ? emissions._shape[0] : 1;
        int seqLen = emissions._shape[emissions.Rank - 2];
        int tags = emissions._shape[emissions.Rank - 1];
        if (batch < 1 || seqLen < 1 || tags < 1 || transitions.Rank != 2 || transitions._shape[0] != tags || transitions._shape[1] != tags
            || (startScores is not null && startScores.Length != tags) || (endScores is not null && endScores.Length != tags)
            || PortableTorchModelOps.CrfAlphaLengthLong(batch, seqLen, tags) > MaxCrfStateLength
            // One work group per sequence: a batch whose launch grid outgrows an int runs on the host.
            || (long)batch * PortableTorchModelOps.CrfGroupSize > int.MaxValue
            || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        int volume = batch * seqLen * tags;
        int alphaLength = PortableTorchModelOps.CrfAlphaLength(batch, seqLen, tags);
        bool record = DifferentiableOps.IsTapeActiveForThread<T>();
        OwnedBuffer alpha = default;
        bool ownsAlpha = false;
        try
        {
            using var e = GetOrAllocateBuffer(backend, emissions.IsContiguous ? emissions : emissions.Contiguous());
            using var tr = GetOrAllocateBuffer(backend, transitions.IsContiguous ? transitions : transitions.Contiguous());
            using var start = CrfScoresBuffer(backend, startScores, tags);
            using var end = CrfScoresBuffer(backend, endScores, tags);
            alpha = record ? AllocateFullyWrittenOutputBuffer(backend, alphaLength) : AllocateTransientScratch(backend, alphaLength, fullyWritten: true);
            ownsAlpha = true;
            var result = DispatchDeferredGpuOp<T>(backend, batch, new[] { batched ? batch : 1 },
                logZ => models.ModelCrfForward(e.Buffer, tr.Buffer, start.Buffer, end.Buffer, alpha.Buffer, logZ, batch, seqLen, tags),
                fullyWritten: true);
            if (!record) return result;
            var alphaTensor = DeferTensorResult<T>(backend, alpha.Buffer, alphaLength, new[] { alphaLength });
            alpha.RelinquishOwnership();
            ownsAlpha = false;
            var inputs = new List<Tensor<T>>(4) { emissions, transitions };
            if (startScores is not null) inputs.Add(startScores);
            if (endScores is not null) inputs.Add(endScores);
            DifferentiableOps.RecordIfActive("TensorCrfForward", result, inputs.ToArray(), CrfForwardDeviceBackward<T>,
                new object[] { alphaTensor, startScores is not null, endScores is not null, batch, seqLen, tags });
            return result;
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorCrfForward", ex);
            return null;
        }
        finally
        {
            if (ownsAlpha) alpha.Dispose();
        }
    }

    // A [tags] score buffer: the tensor's, or zeros for a missing one (the CPU's default).
    // A fresh allocation is not reliably zeroed on every backend, so the zeros are written explicitly.
    private OwnedBuffer CrfScoresBuffer<T>(IDirectGpuBackend backend, Tensor<T>? scores, int tags)
    {
        if (scores is not null) return GetOrAllocateBuffer(backend, scores.IsContiguous ? scores : scores.Contiguous());
        var zeros = AllocateTransientScratch(backend, tags, fullyWritten: true);
        backend.Fill(zeros.Buffer, 0f, tags);
        return zeros;
    }

    private static void CrfForwardDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var alpha = (Tensor<T>)savedState[0];
        bool hasStart = (bool)savedState[1], hasEnd = (bool)savedState[2];
        int batch = (int)savedState[3], seqLen = (int)savedState[4], tags = (int)savedState[5];
        Tensor<T>? endScores = hasEnd ? inputs[hasStart ? 3 : 2] : null;
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            && gpu.TryCrfBackwardOnDevice(gradOutput, inputs[0], inputs[1], endScores, alpha, output, batch, seqLen, tags) is { } g)
        {
            DifferentiableOps.AccumulateGrad(grads, inputs[0], g.Emissions, engine);
            DifferentiableOps.AccumulateGrad(grads, inputs[1], g.Transitions, engine);
            int next = 2;
            if (hasStart) DifferentiableOps.AccumulateGrad(grads, inputs[next++], engine.Reshape(g.Start, (int[])inputs[2]._shape.Clone()), engine);
            if (hasEnd) DifferentiableOps.AccumulateGrad(grads, inputs[next], engine.Reshape(g.End, (int[])inputs[next]._shape.Clone()), engine);
            return;
        }
        if (engine.Is<DirectGpuTensorEngine>()) GpuLaunchProbe.OnFallback("TensorCrfForward backward: host forward-backward", null);
        // The CPU's state rebuilt from the inputs and the saved forward variables.
        var state = new BackwardFunctions<T>.CrfForwardState
        {
            Batch = batch,
            SeqLen = seqLen,
            NumTags = tags,
            HasStart = hasStart,
            HasEnd = hasEnd,
            Emissions = RoIAlignPlan.ToDoubles(inputs[0]),
            Transitions = RoIAlignPlan.ToDoubles(inputs[1]),
            Start = hasStart ? RoIAlignPlan.ToDoubles(inputs[2]) : new double[tags],
            End = hasEnd ? RoIAlignPlan.ToDoubles(inputs[hasStart ? 3 : 2]) : new double[tags],
            Alpha = CrfTrueAlpha(RoIAlignPlan.ToDoubles(alpha), batch, seqLen, tags),
            LogZ = RoIAlignPlan.ToDoubles(output),
        };
        BackwardFunctions<T>.CrfForwardBackward(gradOutput, inputs, output, new object[] { state }, engine, grads);
    }

    // log alpha from the device layout: each stored row plus its step's offset (a hi + lo pair after the rows).
    private static double[] CrfTrueAlpha(double[] stored, int batch, int seqLen, int tags)
    {
        int volume = batch * seqLen * tags;
        var alpha = new double[volume];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seqLen; t++)
            {
                int pair = volume + (b * (seqLen + 1) + t) * 2;
                double offset = stored[pair] + stored[pair + 1];
                int row = (b * seqLen + t) * tags;
                for (int j = 0; j < tags; j++) alpha[row + j] = stored[row + j] + offset;
            }
        return alpha;
    }

    private (Tensor<T> Emissions, Tensor<T> Transitions, Tensor<T> Start, Tensor<T> End)? TryCrfBackwardOnDevice<T>(
        Tensor<T> gradOutput, Tensor<T> emissions, Tensor<T> transitions, Tensor<T>? endScores, Tensor<T> alpha, Tensor<T> logZ,
        int batch, int seqLen, int tags)
    {
        using var __gpuOp = EnterGpuOp();
        if (gradOutput.Length != batch || !TryGetBackend(out var backend) || backend is not ITorchModelOpsBackend models
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        int volume = batch * seqLen * tags;
        OwnedBuffer dEmit = default, dTrans = default, dStart = default, dEnd = default;
        bool ownsEmit = false, ownsTrans = false, ownsStart = false, ownsEnd = false;
        try
        {
            using var e = GetOrAllocateBuffer(backend, emissions.IsContiguous ? emissions : emissions.Contiguous());
            using var tr = GetOrAllocateBuffer(backend, transitions.IsContiguous ? transitions : transitions.Contiguous());
            using var end = CrfScoresBuffer(backend, endScores, tags);
            using var a = GetOrAllocateBuffer(backend, alpha);
            using var dz = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            using var beta = AllocateTransientScratch(backend, PortableTorchModelOps.CrfBetaLength(batch, seqLen, tags), fullyWritten: true);
            dEmit = AllocateFullyWrittenOutputBuffer(backend, volume);
            ownsEmit = true;
            dTrans = AllocateFullyWrittenOutputBuffer(backend, tags * tags);
            ownsTrans = true;
            dStart = AllocateFullyWrittenOutputBuffer(backend, tags);
            ownsStart = true;
            dEnd = AllocateFullyWrittenOutputBuffer(backend, tags);
            ownsEnd = true;
            models.ModelCrfBackward(e.Buffer, tr.Buffer, end.Buffer, a.Buffer, dz.Buffer, beta.Buffer, dEmit.Buffer,
                dTrans.Buffer, dStart.Buffer, dEnd.Buffer, batch, seqLen, tags);
            var gEmit = DeferTensorResult<T>(backend, dEmit.Buffer, volume, (int[])emissions._shape.Clone());
            dEmit.RelinquishOwnership();
            ownsEmit = false;
            var gTrans = DeferTensorResult<T>(backend, dTrans.Buffer, tags * tags, (int[])transitions._shape.Clone());
            dTrans.RelinquishOwnership();
            ownsTrans = false;
            var gStart = DeferTensorResult<T>(backend, dStart.Buffer, tags, new[] { tags });
            dStart.RelinquishOwnership();
            ownsStart = false;
            var gEnd = DeferTensorResult<T>(backend, dEnd.Buffer, tags, new[] { tags });
            dEnd.RelinquishOwnership();
            ownsEnd = false;
            return (gEmit, gTrans, gStart, gEnd);
        }
        catch (NotSupportedException ex)
        {
            GpuLaunchProbe.OnFallback("TensorCrfForward backward", ex);
            return null;
        }
        finally
        {
            if (ownsEmit) dEmit.Dispose();
            if (ownsTrans) dTrans.Dispose();
            if (ownsStart) dStart.Dispose();
            if (ownsEnd) dEnd.Dispose();
        }
    }
}
