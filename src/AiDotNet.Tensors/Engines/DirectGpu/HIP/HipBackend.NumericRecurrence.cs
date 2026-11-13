// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The batched numeric recurrences (INumericRecurrenceBackend) on HIP, through PortableNumericRecurrence and
// NumericRecurrenceKernelTemplate's kernels, compiled on first need.
public sealed partial class HipBackend : INumericRecurrenceBackend
{
    private IntPtr _numericRecurrenceModule;
    private bool? _numericRecurrenceReady;

    // The module, built once with IEEE math; a compile the device rejects leaves the host ops, and says why.
    private bool NumericRecurrenceReady
    {
        get
        {
            if (_numericRecurrenceReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_numericRecurrenceReady is null)
                {
                    bool ok;
                    try
                    {
                        var failure = CompileKernelModule(NumericRecurrenceKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                            "numeric_recurrence", ref _numericRecurrenceModule, NumericRecurrenceKernelTemplate.GetKernelNames(),
                            useFastMath: false);
                        if (failure is not null)
                            System.Diagnostics.Trace.TraceWarning($"HIP numeric recurrence kernels unavailable: {failure.Message}");
                        ok = failure is null;
                        foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    }
                    catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"HIP numeric recurrence kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _numericRecurrenceReady = ok;
                }
                return _numericRecurrenceReady.Value;
            }
        }
    }

    private unsafe void LaunchNumericRecurrence(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)NumericRecurrenceKernelTemplate.GroupSize, args);
    }

    bool INumericRecurrenceBackend.TryTridiagonalSolve(IGpuBuffer lower, IGpuBuffer diagonal, IGpuBuffer upper, IGpuBuffer rhs,
        IGpuBuffer x, IGpuBuffer workspace, int batch, int n)
        => NumericRecurrenceReady
           && PortableNumericRecurrence.TryTridiagonalSolve(LaunchNumericRecurrence, lower, diagonal, upper, rhs, x, workspace, batch, n);

    bool INumericRecurrenceBackend.TryFritschCarlsonLimit(IGpuBuffer slopes, IGpuBuffer secants, IGpuBuffer limited, int batch, int n)
        => NumericRecurrenceReady
           && PortableNumericRecurrence.TryFritschCarlsonLimit(LaunchNumericRecurrence, slopes, secants, limited, batch, n);

    bool INumericRecurrenceBackend.TryCtcForwardBackward(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer lengths,
        IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer logLikelihood, int maxTime, int batch, int classes, int maxStates, int blank)
        => NumericRecurrenceReady
           && PortableNumericRecurrence.TryCtcForwardBackward(LaunchNumericRecurrence, logProbs, targets, lengths, logAlpha, logBeta,
               logLikelihood, maxTime, batch, classes, maxStates, blank);

    bool INumericRecurrenceBackend.TryCtcLabelPosteriors(IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer targets,
        IGpuBuffer lengths, IGpuBuffer logLikelihood, IGpuBuffer gamma, int maxTime, int batch, int classes, int maxStates, int blank)
        => NumericRecurrenceReady
           && PortableNumericRecurrence.TryCtcLabelPosteriors(LaunchNumericRecurrence, logAlpha, logBeta, targets, lengths,
               logLikelihood, gamma, maxTime, batch, classes, maxStates, blank);

    bool INumericRecurrenceBackend.TryLinearSumAssignment(IGpuBuffer cost, IGpuBuffer assignment, IGpuBuffer workspace, int batch,
        int size)
        => NumericRecurrenceReady
           && PortableNumericRecurrence.TryLinearSumAssignment(LaunchNumericRecurrence, cost, assignment, workspace, batch, size);
}
