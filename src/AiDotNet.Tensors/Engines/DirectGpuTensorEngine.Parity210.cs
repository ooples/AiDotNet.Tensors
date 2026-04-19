// Copyright (c) AiDotNet. All rights reserved.
// DirectGpuTensorEngine overrides that route the Parity-210 hot-path ops
// through IParity210Backend when the active backend supports it. Fallback
// is automatic: if the backend isn't a parity-210 backend, or the runtime
// kernel launch throws, we drop to base.TensorX which uses the CpuEngine
// reference (since DirectGpuTensorEngine : CpuEngine).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // -----------------------------------------------------------------------
    // Helper: run a Parity-210 unary op on the GPU, returning a new Tensor<T>
    // whose data is lazily materialised from the GPU buffer.
    // -----------------------------------------------------------------------

    private Tensor<T>? TryRunParity210Unary<T>(
        Tensor<T> input, int[] outputShape,
        Action<IParity210Backend, IGpuBuffer, IGpuBuffer, int> launch)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (backend is not IParity210Backend p210)
            return null;

        using var inBuf = GetOrAllocateBuffer(backend, input);
        int outSize = 1;
        foreach (var d in outputShape) outSize *= d;
        var outBuf = AllocateOutputBuffer(backend, outSize);
        try
        {
            launch(p210, inBuf.Buffer, outBuf.Buffer, input.Length);
            var arr = FinishGpuOp<T>(backend, outBuf, outSize);
            return new Tensor<T>(arr, outputShape);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    private Tensor<T>? TryRunParity210Binary<T>(
        Tensor<T> a, Tensor<T> b, int[] outputShape,
        Action<IParity210Backend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int> launch)
    {
        if (!TryGetBackend(out var backend))
            return null;
        if (backend is not IParity210Backend p210)
            return null;
        if (a.Length != b.Length)
            return null;

        using var aBuf = GetOrAllocateBuffer(backend, a);
        using var bBuf = GetOrAllocateBuffer(backend, b);
        int outSize = 1;
        foreach (var d in outputShape) outSize *= d;
        var outBuf = AllocateOutputBuffer(backend, outSize);
        try
        {
            launch(p210, aBuf.Buffer, bBuf.Buffer, outBuf.Buffer, a.Length);
            var arr = FinishGpuOp<T>(backend, outBuf, outSize);
            return new Tensor<T>(arr, outputShape);
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    // =======================================================================
    // Element-wise unary special — flat length in == length out
    // =======================================================================

    public override Tensor<T> TensorErfc<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210Erfc(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorErfc", r, tensor,
                    Autodiff.BackwardFunctions<T>.ErfcBackward);
                return r;
            }
        }
        catch { }
        return base.TensorErfc(tensor);
    }

    public override Tensor<T> TensorLgamma<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210Lgamma(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLgamma", r, tensor,
                    Autodiff.BackwardFunctions<T>.LgammaBackward);
                return r;
            }
        }
        catch { }
        return base.TensorLgamma(tensor);
    }

    public override Tensor<T> TensorDigamma<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210Digamma(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorDigamma", r, tensor,
                    Autodiff.BackwardFunctions<T>.DigammaBackward);
                return r;
            }
        }
        catch { }
        return base.TensorDigamma(tensor);
    }

    public override Tensor<T> TensorErfinv<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210Erfinv(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorErfinv", r, tensor,
                    Autodiff.BackwardFunctions<T>.ErfinvBackward);
                return r;
            }
        }
        catch { }
        return base.TensorErfinv(tensor);
    }

    public override Tensor<T> TensorI0<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210I0(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorI0", r, tensor,
                    Autodiff.BackwardFunctions<T>.I0Backward);
                return r;
            }
        }
        catch { }
        return base.TensorI0(tensor);
    }

    public override Tensor<T> TensorI1<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, inp, outp, n) => p.Parity210I1(inp, outp, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorI1", r, tensor,
                    Autodiff.BackwardFunctions<T>.I1Backward);
                return r;
            }
        }
        catch { }
        return base.TensorI1(tensor);
    }

    // =======================================================================
    // Element-wise binary special
    // =======================================================================

    public override Tensor<T> TensorHypot<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Hypot(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorHypot", r, a, b,
                    Autodiff.BackwardFunctions<T>.HypotBackward);
                return r;
            }
        }
        catch { }
        return base.TensorHypot(a, b);
    }

    public override Tensor<T> TensorCopysign<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Copysign(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorCopysign", r, a, b,
                    Autodiff.BackwardFunctions<T>.CopysignBackward);
                return r;
            }
        }
        catch { }
        return base.TensorCopysign(a, b);
    }

    public override Tensor<T> TensorLogAddExp<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210LogAddExp(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorLogAddExp", r, a, b,
                    Autodiff.BackwardFunctions<T>.LogAddExpBackward);
                return r;
            }
        }
        catch { }
        return base.TensorLogAddExp(a, b);
    }

    public override Tensor<T> TensorXlogy<T>(Tensor<T> x, Tensor<T> y)
    {
        try
        {
            var r = TryRunParity210Binary(x, y, (int[])x._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Xlogy(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorXlogy", r, x, y,
                    Autodiff.BackwardFunctions<T>.XlogyBackward);
                return r;
            }
        }
        catch { }
        return base.TensorXlogy(x, y);
    }

    // =======================================================================
    // Movement (Triu / Tril)
    // =======================================================================

    public override Tensor<T> TensorTriu<T>(Tensor<T> tensor, int diagonal = 0)
    {
        try
        {
            if (TryGetBackend(out var backend) && backend is IParity210Backend p210
                && tensor.Rank >= 2)
            {
                int rank = tensor.Rank;
                int rows = tensor._shape[rank - 2];
                int cols = tensor._shape[rank - 1];
                int batch = 1; for (int i = 0; i < rank - 2; i++) batch *= tensor._shape[i];
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    p210.Parity210Triu(inBuf.Buffer, outBuf.Buffer, batch, rows, cols, diagonal);
                    var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                    var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                    Autodiff.DifferentiableOps.RecordUnary("TensorTriu", r, tensor,
                        Autodiff.BackwardFunctions<T>.TriuBackward,
                        savedState: new object[] { diagonal });
                    return r;
                }
                catch
                {
                    outBuf.Dispose();
                    throw;
                }
            }
        }
        catch { }
        return base.TensorTriu(tensor, diagonal);
    }

    public override Tensor<T> TensorTril<T>(Tensor<T> tensor, int diagonal = 0)
    {
        try
        {
            if (TryGetBackend(out var backend) && backend is IParity210Backend p210
                && tensor.Rank >= 2)
            {
                int rank = tensor.Rank;
                int rows = tensor._shape[rank - 2];
                int cols = tensor._shape[rank - 1];
                int batch = 1; for (int i = 0; i < rank - 2; i++) batch *= tensor._shape[i];
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    p210.Parity210Tril(inBuf.Buffer, outBuf.Buffer, batch, rows, cols, diagonal);
                    var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                    var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                    Autodiff.DifferentiableOps.RecordUnary("TensorTril", r, tensor,
                        Autodiff.BackwardFunctions<T>.TrilBackward,
                        savedState: new object[] { diagonal });
                    return r;
                }
                catch
                {
                    outBuf.Dispose();
                    throw;
                }
            }
        }
        catch { }
        return base.TensorTril(tensor, diagonal);
    }

    // =======================================================================
    // Cumulative  (CumSum: one thread per (outer, inner) line)
    // =======================================================================

    public new Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            if (TryGetBackend(out var backend) && backend is IParity210Backend p210)
            {
                int rank = tensor.Rank;
                if (axis < 0) axis += rank;
                int outer = 1; for (int i = 0; i < axis; i++) outer *= tensor._shape[i];
                int axisSize = tensor._shape[axis];
                int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= tensor._shape[i];
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    p210.Parity210CumSum(inBuf.Buffer, outBuf.Buffer, outer, axisSize, inner);
                    var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                    return new Tensor<T>(arr, (int[])tensor._shape.Clone());
                }
                catch
                {
                    outBuf.Dispose();
                    throw;
                }
            }
        }
        catch { }
        return base.TensorCumSum(tensor, axis);
    }
}
