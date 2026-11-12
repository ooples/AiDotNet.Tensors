using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The batched numeric recurrences on the device through the backend's INumericRecurrenceBackend, which every backend
    // implements with NumericRecurrenceKernelTemplate's kernels (WGSL on WebGPU): each recurrence runs inside one kernel,
    // one thread or one work-group per independent problem, so no intermediate returns to the host. The tridiagonal
    // solve records the CPU op's tape node, whose backward calls back into this override. Float only, as every device
    // kernel here; other element types, a graph trace, or a shape past the kernels' 32-bit
    // indices or float-exact integers run the CPU op. A graph trace and a non-float T are intended CPU routes; every
    // other refusal of a float call (no recurrence backend, a shape past the kernels' limits, a failed launch) is
    // recorded through GpuLaunchProbe.OnFallback, so a host fallback never runs silently.

    // Records a float call's guard refusal (no exception, so the probe keys it as a declined guard). Returns true so
    // CtcDeviceAdmits can return !RefuseRecurrence(op).
    private static bool RefuseRecurrence(string op)
    {
        GpuLaunchProbe.OnFallback(op, null);
        return true;
    }

    /// <inheritdoc />
    public override Tensor<T> TensorTridiagonalSolve<T>(Tensor<T> lower, Tensor<T> diagonal, Tensor<T> upper, Tensor<T> rhs)
    {
        using var __gpuOp = EnterGpuOp();
        var (batch, n) = TridiagonalShape(lower, diagonal, upper, rhs);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || batch == 0)
            return base.TensorTridiagonalSolve(lower, diagonal, upper, rhs);
        if (!TryGetBackend(out var backend) || backend is not INumericRecurrenceBackend recurrences)
        {
            RefuseRecurrence("TridiagonalSolve");
            return base.TensorTridiagonalSolve(lower, diagonal, upper, rhs);
        }

        using var dB = GetOrAllocateBuffer(backend, diagonal);
        using var rB = GetOrAllocateBuffer(backend, rhs);
        // A single unknown has empty bands: the kernel never reads them, so the diagonal stands in.
        using var lB = n > 1 ? GetOrAllocateBuffer(backend, lower) : (OwnedBuffer?)null;
        using var uB = n > 1 ? GetOrAllocateBuffer(backend, upper) : (OwnedBuffer?)null;
        using var work = AllocateTransientScratch(backend, Math.Max(1, batch * (n - 1)), fullyWritten: true);
        using var xB = AllocateFullyWrittenOutputBuffer(backend, batch * n);
        if (!recurrences.TryTridiagonalSolve(lB?.Buffer ?? dB.Buffer, dB.Buffer, uB?.Buffer ?? dB.Buffer, rB.Buffer, xB.Buffer,
                work.Buffer, batch, n))
        {
            GpuLaunchProbe.OnFallback("TridiagonalSolve", null);
            return base.TensorTridiagonalSolve(lower, diagonal, upper, rhs);
        }
        var result = DeferTensorResult<T>(backend, xB.Buffer, batch * n, (int[])diagonal._shape.Clone());
        xB.RelinquishOwnership();
        RecordTridiagonalSolve(result, lower, diagonal, upper, rhs);
        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> TensorFritschCarlsonLimit<T>(Tensor<T> slopes, Tensor<T> secants)
    {
        using var __gpuOp = EnterGpuOp();
        var (batch, n) = FritschCarlsonShape(slopes, secants);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || batch == 0)
            return base.TensorFritschCarlsonLimit(slopes, secants);
        if (!TryGetBackend(out var backend) || backend is not INumericRecurrenceBackend recurrences)
        {
            RefuseRecurrence("FritschCarlsonLimit");
            return base.TensorFritschCarlsonLimit(slopes, secants);
        }

        using var mB = GetOrAllocateBuffer(backend, slopes);
        using var dB = n > 1 ? GetOrAllocateBuffer(backend, secants) : (OwnedBuffer?)null;
        using var oB = AllocateFullyWrittenOutputBuffer(backend, batch * n);
        if (!recurrences.TryFritschCarlsonLimit(mB.Buffer, dB?.Buffer ?? mB.Buffer, oB.Buffer, batch, n))
        {
            GpuLaunchProbe.OnFallback("FritschCarlsonLimit", null);
            return base.TensorFritschCarlsonLimit(slopes, secants);
        }
        var result = DeferTensorResult<T>(backend, oB.Buffer, batch * n, (int[])slopes._shape.Clone());
        oB.RelinquishOwnership();
        return result;
    }

    // The targets and the per-item input lengths, target lengths and target offsets, as the exactly-valued floats the
    // kernels read (CtcLattice.Prepare has validated them).
    private static (float[] Targets, float[] Lengths) CtcDeviceInputs(CtcLattice.Problem p)
    {
        var targets = new float[Math.Max(1, p.TotalTargets)];
        for (int i = 0; i < p.TotalTargets; i++) targets[i] = p.Targets[i];
        var lengths = new float[3 * p.Batch];
        for (int n = 0; n < p.Batch; n++)
        {
            lengths[n] = p.InputLengths[n];
            lengths[p.Batch + n] = p.TargetLengths[n];
            lengths[2 * p.Batch + n] = p.Offsets[n];
        }
        return (targets, lengths);
    }

    // The kernels index the lattice [N, T, S] and gamma [T, N, C] in 32-bit ints, and read labels and lengths as floats
    // that are exact below 2^24. An empty problem has no device work.
    private static bool CtcDeviceAdmits(CtcLattice.Problem p, string op)
    {
        if (p.Batch <= 0 || p.MaxTime <= 0)
            return false;
        if (p.MaxStates >= (1 << 24) || p.TotalTargets >= (1 << 24) || p.Classes >= (1 << 24))
            return !RefuseRecurrence(op);
        if ((long)p.Batch * p.MaxTime * p.MaxStates > int.MaxValue || (long)p.MaxTime * p.Batch * p.Classes > int.MaxValue)
            return !RefuseRecurrence(op);
        return true;
    }

    /// <inheritdoc />
    public override (Tensor<T> LogAlpha, Tensor<T> LogBeta) TensorCTCForwardBackward<T>(Tensor<T> logProbs, Tensor<int> targets,
        int[] inputLengths, int[] targetLengths, int blank = 0)
    {
        using var __gpuOp = EnterGpuOp();
        Compilation.GraphMode.ThrowIfInferenceUnsupported(Compilation.GraphCaptureLimitation.HeterogeneousInput);
        var p = CtcLattice.Prepare(logProbs, targets, inputLengths, targetLengths, blank);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !CtcDeviceAdmits(p, "CTCForwardBackward"))
            return base.TensorCTCForwardBackward(logProbs, targets, inputLengths, targetLengths, blank);
        if (!TryGetBackend(out var backend) || backend is not INumericRecurrenceBackend recurrences)
        {
            RefuseRecurrence("CTCForwardBackward");
            return base.TensorCTCForwardBackward(logProbs, targets, inputLengths, targetLengths, blank);
        }

        var (targetValues, lengthValues) = CtcDeviceInputs(p);
        int count = p.Batch * p.MaxTime * p.MaxStates;
        using var lpB = GetOrAllocateBuffer(backend, logProbs);
        using var tgB = GetOrAllocateBuffer(backend, targetValues);
        using var lenB = GetOrAllocateBuffer(backend, lengthValues);
        using var lz = AllocateTransientScratch(backend, p.Batch, fullyWritten: true);
        using var laB = AllocateFullyWrittenOutputBuffer(backend, count);
        using var lbB = AllocateFullyWrittenOutputBuffer(backend, count);
        if (!recurrences.TryCtcForwardBackward(lpB.Buffer, tgB.Buffer, lenB.Buffer, laB.Buffer, lbB.Buffer, lz.Buffer,
                p.MaxTime, p.Batch, p.Classes, p.MaxStates, blank))
        {
            GpuLaunchProbe.OnFallback("CTCForwardBackward", null);
            return base.TensorCTCForwardBackward(logProbs, targets, inputLengths, targetLengths, blank);
        }
        int[] shape = { p.Batch, p.MaxTime, p.MaxStates };
        var alpha = DeferTensorResult<T>(backend, laB.Buffer, count, shape);
        laB.RelinquishOwnership();
        var beta = DeferTensorResult<T>(backend, lbB.Buffer, count, (int[])shape.Clone());
        lbB.RelinquishOwnership();
        return (alpha, beta);
    }

    /// <inheritdoc />
    public override Tensor<T> TensorCTCLabelPosteriors<T>(Tensor<T> logProbs, Tensor<int> targets, int[] inputLengths,
        int[] targetLengths, int blank = 0)
    {
        using var __gpuOp = EnterGpuOp();
        Compilation.GraphMode.ThrowIfInferenceUnsupported(Compilation.GraphCaptureLimitation.HeterogeneousInput);
        var p = CtcLattice.Prepare(logProbs, targets, inputLengths, targetLengths, blank);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !CtcDeviceAdmits(p, "CTCLabelPosteriors"))
            return base.TensorCTCLabelPosteriors(logProbs, targets, inputLengths, targetLengths, blank);
        if (!TryGetBackend(out var backend) || backend is not INumericRecurrenceBackend recurrences)
        {
            RefuseRecurrence("CTCLabelPosteriors");
            return base.TensorCTCLabelPosteriors(logProbs, targets, inputLengths, targetLengths, blank);
        }

        var (targetValues, lengthValues) = CtcDeviceInputs(p);
        int count = p.Batch * p.MaxTime * p.MaxStates, gammaCount = p.MaxTime * p.Batch * p.Classes;
        using var lpB = GetOrAllocateBuffer(backend, logProbs);
        using var tgB = GetOrAllocateBuffer(backend, targetValues);
        using var lenB = GetOrAllocateBuffer(backend, lengthValues);
        // Transient scratch: the lattice never leaves the device (AiDotNet#1804).
        using var lz = AllocateTransientScratch(backend, p.Batch, fullyWritten: true);
        using var la = AllocateTransientScratch(backend, count, fullyWritten: true);
        using var lb = AllocateTransientScratch(backend, count, fullyWritten: true);
        using var gB = AllocateFullyWrittenOutputBuffer(backend, gammaCount);
        if (!recurrences.TryCtcForwardBackward(lpB.Buffer, tgB.Buffer, lenB.Buffer, la.Buffer, lb.Buffer, lz.Buffer,
                p.MaxTime, p.Batch, p.Classes, p.MaxStates, blank)
            || !recurrences.TryCtcLabelPosteriors(la.Buffer, lb.Buffer, tgB.Buffer, lenB.Buffer, lz.Buffer, gB.Buffer,
                p.MaxTime, p.Batch, p.Classes, p.MaxStates, blank))
        {
            GpuLaunchProbe.OnFallback("CTCLabelPosteriors", null);
            return base.TensorCTCLabelPosteriors(logProbs, targets, inputLengths, targetLengths, blank);
        }
        var gamma = DeferTensorResult<T>(backend, gB.Buffer, gammaCount, new[] { p.MaxTime, p.Batch, p.Classes });
        gB.RelinquishOwnership();
        return gamma;
    }

    /// <inheritdoc />
    public override Tensor<int> TensorLinearSumAssignment<T>(Tensor<T> cost)
    {
        using var __gpuOp = EnterGpuOp();
        Compilation.GraphMode.ThrowIfInferenceUnsupported(Compilation.GraphCaptureLimitation.MixedElementTypes);
        var (batch, size) = LinearSumAssignmentShape(cost);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || batch == 0 || size == 0)
            return base.TensorLinearSumAssignment(cost);
        // The solver state is 8 * size floats per item, in one buffer with 32-bit indices.
        if (size > NumericRecurrenceKernelTemplate.AssignmentMaxSize || (long)batch * 8 * size > int.MaxValue
            || !TryGetBackend(out var backend) || backend is not INumericRecurrenceBackend recurrences)
        {
            RefuseRecurrence("LinearSumAssignment");
            return base.TensorLinearSumAssignment(cost);
        }

        using var cB = GetOrAllocateBuffer(backend, cost);
        using var aB = AllocateFullyWrittenOutputBuffer(backend, batch * size);
        // The solver state (duals, path costs, matchings) of every item, in global memory: no size cap.
        using var work = AllocateTransientScratch(backend, batch * 8 * size, fullyWritten: true);
        if (!recurrences.TryLinearSumAssignment(cB.Buffer, aB.Buffer, work.Buffer, batch, size))
        {
            GpuLaunchProbe.OnFallback("LinearSumAssignment", null);
            return base.TensorLinearSumAssignment(cost);
        }
        var result = DeferTensorResult<int>(backend, aB.Buffer, batch * size, cost.Rank == 2 ? new[] { size } : new[] { batch, size });
        aB.RelinquishOwnership();
        return result;
    }
}
