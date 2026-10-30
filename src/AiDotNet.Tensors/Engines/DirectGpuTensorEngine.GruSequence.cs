// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// The fused GRU sequence on the device (#1568): the shared gru_p_ / gru_rb_ kernels on every backend, with a recorded
    /// node whose backward runs on the device too. Before this a GRU ran one timestep and one op at a time (11-13x slower
    /// than PyTorch to train on CUDA). Float only; other types, and a backend that cannot launch the kernels, take the host
    /// path.
    /// </summary>
    public override Tensor<T> GruSequenceForward<T>(Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, out Tensor<T> finalHidden, bool returnSequences = false, bool resetAfter = true)
    {
        var (batch, seqLen, inputSize, hidden) = ValidateGruSequence(input, h0, wIh, wHh, bIh, bHh);
        // Graph capture records the host op's lazy node, whose replay runs this engine eagerly. An empty extent has
        // nothing to launch and the kernels require positive ones: the host path returns the empty result.
        if (seqLen == 0 || batch == 0 || inputSize == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedLstmSequenceTraining fused)
            return base.GruSequenceForward(input, h0, wIh, wHh, bIh, bHh, out finalHidden, returnSequences, resetAfter);
        var sequence = GruSequenceOnDevice(backend, fused, input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden, resetAfter)
            ?? GruSequenceRecorded(input, h0, wIh, wHh, bIh, bHh, batch, seqLen, hidden, resetAfter);
        return GruSequenceOutputs(sequence, batch, seqLen, hidden, returnSequences, out finalHidden);
    }

    // Null when the backend cannot launch the kernels (the caller takes the host path).
    private Tensor<T>? GruSequenceOnDevice<T>(IDirectGpuBackend backend, IFusedLstmSequenceTraining fused,
        Tensor<T> input, Tensor<T>? h0, Tensor<T> wIh, Tensor<T> wHh, Tensor<T>? bIh, Tensor<T>? bHh,
        int batch, int seqLen, int hidden, bool resetAfter)
    {
        int inFeatures = input.Shape[2];
        var (allHSize, gatesSize, ghnSize, rhSize) = PortableGruSequence.CacheSizes(seqLen, batch, hidden);
        OwnedBuffer y = default, allH = default, gates = default, ghn = default, rh = default;
        bool handedOff = false;
        try
        {
            using var x = GetOrAllocateBuffer(backend, input);
            using var hInit = h0 is null ? ZeroBuffer(backend, batch * hidden) : GetOrAllocateBuffer(backend, h0);
            using var wi = GetOrAllocateBuffer(backend, wIh);
            using var wh = GetOrAllocateBuffer(backend, wHh);
            using var bi = bIh is null ? ZeroBuffer(backend, 3 * hidden) : GetOrAllocateBuffer(backend, bIh);
            using var bh = bHh is null ? ZeroBuffer(backend, 3 * hidden) : GetOrAllocateBuffer(backend, bHh);
            y = AllocateFullyWrittenOutputBuffer(backend, batch * seqLen * hidden);
            allH = AllocateFullyWrittenOutputBuffer(backend, allHSize);
            gates = AllocateFullyWrittenOutputBuffer(backend, gatesSize);
            ghn = AllocateFullyWrittenOutputBuffer(backend, ghnSize);
            if (!resetAfter) rh = AllocateFullyWrittenOutputBuffer(backend, rhSize);
            PortableGruSequence.Forward(backend, fused.StatefulLstmLauncher, x.Buffer, hInit.Buffer, wi.Buffer, wh.Buffer,
                bi.Buffer, bh.Buffer, y.Buffer, allH.Buffer, gates.Buffer, ghn.Buffer, resetAfter ? null : rh.Buffer,
                seqLen, batch, inFeatures, hidden, resetAfter);

            var sequence = DeferTensorResult<T>(backend, y.Buffer, batch * seqLen * hidden, new[] { batch, seqLen, hidden });
            y.RelinquishOwnership();
            if (!DifferentiableOps.IsRecording<T>())
            {
                // No backward will read the caches (about 5x the output): back to the pool once the forward has run,
                // not when a collection finalizes them.
                allH.DisposeAfterQueuedUse(backend); gates.DisposeAfterQueuedUse(backend); ghn.DisposeAfterQueuedUse(backend);
                if (!resetAfter) rh.DisposeAfterQueuedUse(backend);
                handedOff = true;
                return sequence;
            }
            // The caches live as long as the recorded node: owned tensors over their buffers.
            var allHTensor = DeferTensorResult<float>(backend, allH.Buffer, allHSize, new[] { allHSize });
            var gatesTensor = DeferTensorResult<float>(backend, gates.Buffer, gatesSize, new[] { gatesSize });
            var ghnTensor = DeferTensorResult<float>(backend, ghn.Buffer, ghnSize, new[] { ghnSize });
            var rhTensor = resetAfter ? null : DeferTensorResult<float>(backend, rh.Buffer, rhSize, new[] { rhSize });
            allH.RelinquishOwnership(); gates.RelinquishOwnership(); ghn.RelinquishOwnership();
            if (!resetAfter) rh.RelinquishOwnership();
            handedOff = true;

            var inputs = GruSequenceInputs(input, h0, wIh, wHh, bIh, bHh, out var present);
            DifferentiableOps.RecordIfActive("GruSequenceForward", sequence, inputs, GruSequenceBackwardOnDevice<T>,
                new object[]
                {
                    new[] { batch, seqLen, inFeatures, hidden }, present, allHTensor, gatesTensor, ghnTensor,
                    resetAfter, rhTensor ?? (object)resetAfter,
                });
            return sequence;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("GruSequenceForward", capabilityGap);
            return null;
        }
        finally
        {
            if (!handedOff) { y.Dispose(); allH.Dispose(); gates.Dispose(); ghn.Dispose(); rh.Dispose(); }
        }
    }

    private static OwnedBuffer ZeroBuffer(IDirectGpuBackend backend, int length)
    {
        var buffer = backend.AllocateBuffer(length);
        backend.Fill(buffer, 0f, length);
        return new OwnedBuffer(buffer, ownsBuffer: true);
    }

    private static void GruSequenceBackwardOnDevice<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (engine is not DirectGpuTensorEngine gpu || !gpu.TryGetBackend(out var backend)
            || backend is not IFusedLstmSequenceTraining fused)
            throw new InvalidOperationException("A device GRU node must be differentiated by the GPU engine that recorded it.");
        var dims = (int[])savedState[0];
        var present = (GruOptionalInputs)savedState[1];
        var allH = (Tensor<float>)savedState[2];
        var gates = (Tensor<float>)savedState[3];
        var ghn = (Tensor<float>)savedState[4];
        bool resetAfter = (bool)savedState[5];
        var rhTensor = savedState[6] as Tensor<float>;
        int batch = dims[0], seqLen = dims[1], inFeatures = dims[2], hidden = dims[3], g3 = 3 * hidden;
        var (input, h0, wIh, wHh, bIh, bHh) = UnpackGruSequenceInputs(inputs, present);

        using var gy = gpu.GetOrAllocateBuffer(backend, gradOutput);
        using var cH = gpu.GetOrAllocateBuffer(backend, allH);
        using var cG = gpu.GetOrAllocateBuffer(backend, gates);
        using var cN = gpu.GetOrAllocateBuffer(backend, ghn);
        using var cR = rhTensor is null ? default : gpu.GetOrAllocateBuffer(backend, rhTensor);
        using var x = gpu.GetOrAllocateBuffer(backend, input);
        using var wi = gpu.GetOrAllocateBuffer(backend, wIh);
        using var wh = gpu.GetOrAllocateBuffer(backend, wHh);
        var dx = AllocateFullyWrittenOutputBuffer(backend, batch * seqLen * inFeatures);
        var dh0 = AllocateFullyWrittenOutputBuffer(backend, batch * hidden);
        var dwi = AllocateFullyWrittenOutputBuffer(backend, g3 * inFeatures);
        var dwh = AllocateFullyWrittenOutputBuffer(backend, g3 * hidden);
        var dbi = AllocateFullyWrittenOutputBuffer(backend, g3);
        var dbh = AllocateFullyWrittenOutputBuffer(backend, g3);
        var owned = new[] { dx, dh0, dwi, dwh, dbi, dbh };
        try
        {
            PortableGruSequence.Backward(backend, fused.StatefulLstmLauncher, gy.Buffer, cH.Buffer, cG.Buffer, cN.Buffer,
                rhTensor is null ? null : cR.Buffer, wi.Buffer, wh.Buffer, x.Buffer, dx.Buffer, dh0.Buffer, dwi.Buffer,
                dwh.Buffer, dbi.Buffer, dbh.Buffer, seqLen, batch, inFeatures, hidden, resetAfter: resetAfter);
            Tensor<T> Result(int index, int length, int[] shape)
            {
                var tensor = gpu.DeferTensorResult<T>(backend, owned[index].Buffer, length, shape);
                owned[index].RelinquishOwnership();
                return tensor;
            }
            DifferentiableOps.AccumulateGrad(grads, input, Result(0, batch * seqLen * inFeatures, new[] { batch, seqLen, inFeatures }), engine);
            var gradH0 = Result(1, batch * hidden, new[] { batch, hidden });
            if (h0 is not null) DifferentiableOps.AccumulateGrad(grads, h0, gradH0, engine);
            DifferentiableOps.AccumulateGrad(grads, wIh, Result(2, g3 * inFeatures, new[] { g3, inFeatures }), engine);
            DifferentiableOps.AccumulateGrad(grads, wHh, Result(3, g3 * hidden, new[] { g3, hidden }), engine);
            var gradBi = Result(4, g3, new[] { g3 });
            var gradBh = Result(5, g3, new[] { g3 });
            if (bIh is not null) DifferentiableOps.AccumulateGrad(grads, bIh, gradBi, engine);
            if (bHh is not null) DifferentiableOps.AccumulateGrad(grads, bHh, gradBh, engine);
        }
        finally
        {
            foreach (var o in owned) o.Dispose();   // a no-op for the ones handed to tensors
        }
    }
}
