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

    public override Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            if (TryGetBackend(out var backend) && backend is IParity210Backend p210)
            {
                int rank = tensor.Rank;
                if (axis < 0) axis += rank;
                // Validate the normalised axis BEFORE indexing _shape so an
                // out-of-range axis (e.g. -rank-1 which normalises to -1)
                // falls through to the CPU reference path — which throws the
                // canonical ArgumentOutOfRangeException — instead of picking
                // up a wrong axis silently.
                if (axis < 0 || axis >= rank)
                    return base.TensorCumSum(tensor, axis);
                int outer = 1; for (int i = 0; i < axis; i++) outer *= tensor._shape[i];
                int axisSize = tensor._shape[axis];
                int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= tensor._shape[i];
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    p210.Parity210CumSum(inBuf.Buffer, outBuf.Buffer, outer, axisSize, inner);
                    var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                    var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                    Autodiff.DifferentiableOps.RecordUnary("TensorCumSum", r, tensor,
                        Autodiff.BackwardFunctions<T>.CumSumBackward, new object[] { axis });
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
        return base.TensorCumSum(tensor, axis);
    }

    // -----------------------------------------------------------------------
    // Cumulative ops (CumProd / CumMax / CumMin / LogCumSumExp)
    // -----------------------------------------------------------------------

    private Tensor<T>? TryRunCumulative<T>(Tensor<T> tensor, int axis,
        System.Action<IParity210Backend, IGpuBuffer, IGpuBuffer, int, int, int> launch)
    {
        if (!TryGetBackend(out var backend)) return null;
        if (backend is not IParity210Backend p210) return null;
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) return null;
        int outer = 1; for (int i = 0; i < axis; i++) outer *= tensor._shape[i];
        int axisSize = tensor._shape[axis];
        int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= tensor._shape[i];
        using var inBuf = GetOrAllocateBuffer(backend, tensor);
        var outBuf = AllocateOutputBuffer(backend, tensor.Length);
        try
        {
            launch(p210, inBuf.Buffer, outBuf.Buffer, outer, axisSize, inner);
            var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch
        {
            outBuf.Dispose();
            throw;
        }
    }

    public override Tensor<T> TensorCumProd<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            var r = TryRunCumulative(tensor, axis,
                static (p, i, o, outer, ax, inner) => p.Parity210CumProd(i, o, outer, ax, inner));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorCumProd", r, tensor,
                    Autodiff.BackwardFunctions<T>.CumProdBackward, new object[] { axis });
                return r;
            }
        }
        catch { }
        return base.TensorCumProd(tensor, axis);
    }

    public override Tensor<T> TensorCumMax<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            var r = TryRunCumulative(tensor, axis,
                static (p, i, o, outer, ax, inner) => p.Parity210CumMax(i, o, outer, ax, inner));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorCumMax", r, tensor,
                    Autodiff.BackwardFunctions<T>.CumMaxBackward, new object[] { axis });
                return r;
            }
        }
        catch { }
        return base.TensorCumMax(tensor, axis);
    }

    public override Tensor<T> TensorCumMin<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            var r = TryRunCumulative(tensor, axis,
                static (p, i, o, outer, ax, inner) => p.Parity210CumMin(i, o, outer, ax, inner));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorCumMin", r, tensor,
                    Autodiff.BackwardFunctions<T>.CumMinBackward, new object[] { axis });
                return r;
            }
        }
        catch { }
        return base.TensorCumMin(tensor, axis);
    }

    public override Tensor<T> TensorLogCumSumExp<T>(Tensor<T> tensor, int axis)
    {
        try
        {
            var r = TryRunCumulative(tensor, axis,
                static (p, i, o, outer, ax, inner) => p.Parity210LogCumSumExp(i, o, outer, ax, inner));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorLogCumSumExp", r, tensor,
                    Autodiff.BackwardFunctions<T>.LogCumSumExpBackward, new object[] { axis });
                return r;
            }
        }
        catch { }
        return base.TensorLogCumSumExp(tensor, axis);
    }

    // -----------------------------------------------------------------------
    // Movement — Roll / Flip / DiagEmbed
    // -----------------------------------------------------------------------

    public override Tensor<T> TensorRoll<T>(Tensor<T> tensor, int[] shifts, int[] axes)
    {
        // GPU path only supports single-axis roll; composite rolls compose
        // in CPU via the base class.
        if (shifts.Length == 1 && axes.Length == 1
            && TryGetBackend(out var backend) && backend is IParity210Backend p210)
        {
            int axis = axes[0];
            int rank = tensor.Rank;
            if (axis < 0) axis += rank;
            if (axis >= 0 && axis < rank)
            {
                try
                {
                    int outer = 1; for (int i = 0; i < axis; i++) outer *= tensor._shape[i];
                    int axisSize = tensor._shape[axis];
                    int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= tensor._shape[i];
                    using var inBuf = GetOrAllocateBuffer(backend, tensor);
                    var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                    try
                    {
                        p210.Parity210Roll1D(inBuf.Buffer, outBuf.Buffer, outer, axisSize, inner, shifts[0]);
                        var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                        var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                        Autodiff.DifferentiableOps.RecordUnary("TensorRoll", r, tensor,
                            Autodiff.BackwardFunctions<T>.RollBackward,
                            savedState: new object[] { (int[])shifts.Clone(), (int[])axes.Clone() });
                        return r;
                    }
                    catch
                    {
                        outBuf.Dispose();
                        throw;
                    }
                }
                catch { }
            }
        }
        return base.TensorRoll(tensor, shifts, axes);
    }

    public override Tensor<T> TensorFlip<T>(Tensor<T> tensor, int[] axes)
    {
        // Single-axis flip → GPU; multi-axis composes in CPU.
        if (axes.Length == 1
            && TryGetBackend(out var backend) && backend is IParity210Backend p210)
        {
            int axis = axes[0];
            int rank = tensor.Rank;
            if (axis < 0) axis += rank;
            if (axis >= 0 && axis < rank)
            {
                try
                {
                    int outer = 1; for (int i = 0; i < axis; i++) outer *= tensor._shape[i];
                    int axisSize = tensor._shape[axis];
                    int inner = 1; for (int i = axis + 1; i < rank; i++) inner *= tensor._shape[i];
                    using var inBuf = GetOrAllocateBuffer(backend, tensor);
                    var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                    try
                    {
                        p210.Parity210FlipAxis(inBuf.Buffer, outBuf.Buffer, outer, axisSize, inner);
                        var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                        var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                        Autodiff.DifferentiableOps.RecordUnary("TensorFlip", r, tensor,
                            Autodiff.BackwardFunctions<T>.FlipBackward,
                            savedState: new object[] { (int[])axes.Clone() });
                        return r;
                    }
                    catch
                    {
                        outBuf.Dispose();
                        throw;
                    }
                }
                catch { }
            }
        }
        return base.TensorFlip(tensor, axes);
    }

    public override Tensor<T> TensorDiagEmbed<T>(Tensor<T> tensor, int offset = 0)
    {
        if (TryGetBackend(out var backend) && backend is IParity210Backend p210
            && tensor.Rank >= 1)
        {
            try
            {
                int rank = tensor.Rank;
                int diagLen = tensor._shape[rank - 1];
                int matSize = diagLen + System.Math.Abs(offset);
                int batch = 1; for (int k = 0; k < rank - 1; k++) batch *= tensor._shape[k];
                var outShape = new int[rank + 1];
                for (int i = 0; i < rank - 1; i++) outShape[i] = tensor._shape[i];
                outShape[rank - 1] = matSize;
                outShape[rank] = matSize;
                int outLen = batch * matSize * matSize;
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, outLen);
                try
                {
                    p210.Parity210DiagEmbed(inBuf.Buffer, outBuf.Buffer, batch, diagLen, matSize, offset);
                    var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                    var r = new Tensor<T>(arr, outShape);
                    Autodiff.DifferentiableOps.RecordUnary("TensorDiagEmbed", r, tensor,
                        Autodiff.BackwardFunctions<T>.DiagEmbedBackward,
                        savedState: new object[] { offset });
                    return r;
                }
                catch
                {
                    outBuf.Dispose();
                    throw;
                }
            }
            catch { }
        }
        return base.TensorDiagEmbed(tensor, offset);
    }

    // -----------------------------------------------------------------------
    // Remaining element-wise binary special
    // -----------------------------------------------------------------------

    public override Tensor<T> TensorFmod<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Fmod(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorFmod", r, a, b,
                    Autodiff.BackwardFunctions<T>.FmodBackward);
                return r;
            }
        }
        catch { }
        return base.TensorFmod(a, b);
    }

    public override Tensor<T> TensorRemainder<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Remainder(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorRemainder", r, a, b,
                    Autodiff.BackwardFunctions<T>.RemainderBackward);
                return r;
            }
        }
        catch { }
        return base.TensorRemainder(a, b);
    }

    public override Tensor<T> TensorFloatPower<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210FloatPower(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorFloatPower", r, a, b,
                    Autodiff.BackwardFunctions<T>.FloatPowerBackward);
                return r;
            }
        }
        catch { }
        return base.TensorFloatPower(a, b);
    }

    public override Tensor<T> TensorLogAddExp2<T>(Tensor<T> a, Tensor<T> b)
    {
        try
        {
            var r = TryRunParity210Binary(a, b, (int[])a._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210LogAddExp2(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorLogAddExp2", r, a, b,
                    Autodiff.BackwardFunctions<T>.LogAddExp2Backward);
                return r;
            }
        }
        catch { }
        return base.TensorLogAddExp2(a, b);
    }

    public override Tensor<T> TensorXlog1py<T>(Tensor<T> x, Tensor<T> y)
    {
        try
        {
            var r = TryRunParity210Binary(x, y, (int[])x._shape.Clone(),
                static (p, aa, bb, oo, n) => p.Parity210Xlog1py(aa, bb, oo, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordBinary("TensorXlog1py", r, x, y,
                    Autodiff.BackwardFunctions<T>.Xlog1pyBackward);
                return r;
            }
        }
        catch { }
        return base.TensorXlog1py(x, y);
    }

    // -----------------------------------------------------------------------
    // Remaining unary special
    // -----------------------------------------------------------------------

    public override Tensor<T> TensorI0e<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, i, o, n) => p.Parity210I0e(i, o, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorI0e", r, tensor,
                    Autodiff.BackwardFunctions<T>.I0eBackward);
                return r;
            }
        }
        catch { }
        return base.TensorI0e(tensor);
    }

    public override Tensor<T> TensorI1e<T>(Tensor<T> tensor)
    {
        try
        {
            var r = TryRunParity210Unary(tensor, (int[])tensor._shape.Clone(),
                static (p, i, o, n) => p.Parity210I1e(i, o, n));
            if (r != null)
            {
                Autodiff.DifferentiableOps.RecordUnary("TensorI1e", r, tensor,
                    Autodiff.BackwardFunctions<T>.I1eBackward);
                return r;
            }
        }
        catch { }
        return base.TensorI1e(tensor);
    }

    public override Tensor<T> TensorNanToNum<T>(
        Tensor<T> tensor, double? nan = null, double? posinf = null, double? neginf = null)
    {
        if (TryGetBackend(out var backend) && backend is IParity210Backend p210)
        {
            try
            {
                float nanVal = (float)(nan ?? 0.0);
                float posV = (float)(posinf ?? (double)float.MaxValue);
                float negV = (float)(neginf ?? -(double)float.MaxValue);
                using var inBuf = GetOrAllocateBuffer(backend, tensor);
                var outBuf = AllocateOutputBuffer(backend, tensor.Length);
                try
                {
                    p210.Parity210NanToNum(inBuf.Buffer, outBuf.Buffer, tensor.Length, nanVal, posV, negV);
                    var arr = FinishGpuOp<T>(backend, outBuf, tensor.Length);
                    var r = new Tensor<T>(arr, (int[])tensor._shape.Clone());
                    Autodiff.DifferentiableOps.RecordUnary("TensorNanToNum", r, tensor,
                        Autodiff.BackwardFunctions<T>.NanToNumBackward);
                    return r;
                }
                catch
                {
                    outBuf.Dispose();
                    throw;
                }
            }
            catch { }
        }
        return base.TensorNanToNum(tensor, nan, posinf, neginf);
    }
}
