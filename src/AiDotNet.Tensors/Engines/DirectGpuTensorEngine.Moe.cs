// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The fused MoE on the device through IMoeKernelBackend (every backend): routing, sort, both projections and the
    // combine with no host read, every intermediate resident for the backward.
    internal override Tensor<T> MoeSwiGluForward<T>(Tensor<T> x, Tensor<T> scores, Tensor<T>? bias, Tensor<T> w1, Tensor<T> w3,
        Tensor<T> w2, int topK, int sequences, bool keepSaved, out Tensor<T> loads, out MoeSaved? saved)
    {
        using var __gpuOp = EnterGpuOp();
        ValidateMoe(x, scores, bias, w1, w3, w2, topK, sequences, out int n, out int d, out int g, out int f);
        var s = new MoeShape(n, d, g, topK, f, sequences);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !s.Admitted || !TryGetBackend(out var backend)
            || backend is not IMoeKernelBackend moe || !moe.MoeKernelsReady)
            return base.MoeSwiGluForward(x, scores, bias, w1, w3, w2, topK, sequences, keepSaved, out loads, out saved);

        using var xB = GetOrAllocateBuffer(backend, x);
        using var sB = GetOrAllocateBuffer(backend, scores);
        using var bB = bias is null ? (OwnedBuffer?)null : GetOrAllocateBuffer(backend, bias);
        using var w1B = GetOrAllocateBuffer(backend, w1);
        using var w3B = GetOrAllocateBuffer(backend, w3);
        using var w2B = GetOrAllocateBuffer(backend, w2);
        int nk = s.NK;
        OwnedBuffer ids = default, gates = default, ssum = default, perm = default, pos = default,
            ends = default, load = default, h13 = default, o = default, y = default;
        try
        {
            ids = AllocateFullyWrittenOutputBuffer(backend, nk);
            gates = AllocateFullyWrittenOutputBuffer(backend, nk);
            ssum = AllocateFullyWrittenOutputBuffer(backend, n);
            perm = AllocateFullyWrittenOutputBuffer(backend, nk);
            pos = AllocateFullyWrittenOutputBuffer(backend, nk);
            ends = AllocateFullyWrittenOutputBuffer(backend, g);
            load = AllocateFullyWrittenOutputBuffer(backend, sequences * g);
            h13 = AllocateFullyWrittenOutputBuffer(backend, nk * 2 * f);
            o = AllocateFullyWrittenOutputBuffer(backend, nk * d);
            y = AllocateFullyWrittenOutputBuffer(backend, n * d);
            PortableMoe.Forward(moe, s, xB.Buffer, sB.Buffer, bB?.Buffer, w1B.Buffer, w3B.Buffer, w2B.Buffer, ids.Buffer,
                gates.Buffer, ssum.Buffer, perm.Buffer, pos.Buffer, ends.Buffer, load.Buffer, h13.Buffer, o.Buffer, y.Buffer);
        }
        catch
        {
            ids.DisposeAfterQueuedUse(backend); gates.DisposeAfterQueuedUse(backend); ssum.DisposeAfterQueuedUse(backend);
            perm.DisposeAfterQueuedUse(backend); pos.DisposeAfterQueuedUse(backend); ends.DisposeAfterQueuedUse(backend);
            load.DisposeAfterQueuedUse(backend); h13.DisposeAfterQueuedUse(backend); o.DisposeAfterQueuedUse(backend);
            y.DisposeAfterQueuedUse(backend);
            throw;
        }

        Tensor<float> Keep(OwnedBuffer b, int count, int[] shape)
        {
            var t = DeferTensorResult<float>(backend, b.Buffer, count, shape);
            b.RelinquishOwnership();
            return t;
        }

        saved = null;
        if (keepSaved)
        {
            saved = new MoeSaved(n, d, g, topK, f, sequences)
            {
                DeviceBackend = PlacementBackend ?? backend,
                DIds = Keep(ids, nk, new[] { nk }), DGates = Keep(gates, nk, new[] { nk }), DSsum = Keep(ssum, n, new[] { n }),
                DPerm = Keep(perm, nk, new[] { nk }), DPos = Keep(pos, nk, new[] { nk }), DEnds = Keep(ends, g, new[] { g }),
                DH13 = Keep(h13, nk * 2 * f, new[] { nk, 2 * f }), DO = Keep(o, nk * d, new[] { nk, d }),
            };
        }
        else
        {
            ids.DisposeAfterQueuedUse(backend);
            gates.DisposeAfterQueuedUse(backend);
            ssum.DisposeAfterQueuedUse(backend);
            perm.DisposeAfterQueuedUse(backend);
            pos.DisposeAfterQueuedUse(backend);
            ends.DisposeAfterQueuedUse(backend);
            h13.DisposeAfterQueuedUse(backend);
            o.DisposeAfterQueuedUse(backend);
        }

        loads = DeferTensorResult<T>(backend, load.Buffer, sequences * g, new[] { sequences, g });
        load.RelinquishOwnership();
        var result = DeferTensorResult<T>(backend, y.Buffer, n * d, new[] { n, d });
        y.RelinquishOwnership();
        return result;
    }

    internal override (Tensor<T> Dx, Tensor<T> Ds, Tensor<T> Dw1, Tensor<T> Dw3, Tensor<T> Dw2) MoeSwiGluGradients<T>(
        Tensor<T> dy, Tensor<T> x, Tensor<T> w1, Tensor<T> w3, Tensor<T> w2, MoeSaved saved)
    {
        using var __gpuOp = EnterGpuOp();
        if (saved.DH13 is null) return base.MoeSwiGluGradients(dy, x, w1, w3, w2, saved);
        if (!TryGetBackend(out var backend) || backend is not IMoeKernelBackend moe || !moe.MoeKernelsReady)
            throw new System.InvalidOperationException("The fused MoE forward ran on a device that is no longer available.");
        var s = saved.Shape;
        int n = s.N, d = s.D, g = s.G, f = s.F, nk = s.NK;
        using var dyB = GetOrAllocateBuffer(backend, dy);
        using var xB = GetOrAllocateBuffer(backend, x);
        using var w1B = GetOrAllocateBuffer(backend, w1);
        using var w3B = GetOrAllocateBuffer(backend, w3);
        using var w2B = GetOrAllocateBuffer(backend, w2);
        using var ids = GetOrAllocateBuffer(backend, saved.DIds!);
        using var gates = GetOrAllocateBuffer(backend, saved.DGates!);
        using var ssum = GetOrAllocateBuffer(backend, saved.DSsum!);
        using var perm = GetOrAllocateBuffer(backend, saved.DPerm!);
        using var pos = GetOrAllocateBuffer(backend, saved.DPos!);
        using var ends = GetOrAllocateBuffer(backend, saved.DEnds!);
        using var h13 = GetOrAllocateBuffer(backend, saved.DH13);
        using var o = GetOrAllocateBuffer(backend, saved.DO!);
        OwnedBuffer dh13 = default, dxs = default, ds = default, dx = default, dw1 = default, dw3 = default, dw2 = default;
        try
        {
            dh13 = AllocateFullyWrittenOutputBuffer(backend, nk * 2 * f);
            dxs = AllocateFullyWrittenOutputBuffer(backend, nk * d);
            ds = AllocateFullyWrittenOutputBuffer(backend, n * g);
            dx = AllocateFullyWrittenOutputBuffer(backend, n * d);
            dw1 = AllocateFullyWrittenOutputBuffer(backend, g * d * f);
            dw3 = AllocateFullyWrittenOutputBuffer(backend, g * d * f);
            dw2 = AllocateFullyWrittenOutputBuffer(backend, g * f * d);
            PortableMoe.Backward(moe, s, dyB.Buffer, xB.Buffer, w1B.Buffer, w3B.Buffer, w2B.Buffer, ids.Buffer, gates.Buffer,
                ssum.Buffer, perm.Buffer, pos.Buffer, ends.Buffer, h13.Buffer, o.Buffer, ds.Buffer, dh13.Buffer, dxs.Buffer,
                dx.Buffer, dw1.Buffer, dw3.Buffer, dw2.Buffer);
        }
        catch
        {
            ds.DisposeAfterQueuedUse(backend); dx.DisposeAfterQueuedUse(backend); dw1.DisposeAfterQueuedUse(backend);
            dw3.DisposeAfterQueuedUse(backend); dw2.DisposeAfterQueuedUse(backend);
            throw;
        }
        finally
        {
            dh13.DisposeAfterQueuedUse(backend);
            dxs.DisposeAfterQueuedUse(backend);
        }

        Tensor<T> Out(OwnedBuffer b, int count, int[] shape)
        {
            var t = DeferTensorResult<T>(backend, b.Buffer, count, shape);
            b.RelinquishOwnership();
            return t;
        }
        return (Out(dx, n * d, new[] { n, d }), Out(ds, n * g, new[] { n, g }), Out(dw1, g * d * f, new[] { g, d, f }),
            Out(dw3, g * d * f, new[] { g, d, f }), Out(dw2, g * f * d, new[] { g, f, d }));
    }
}
