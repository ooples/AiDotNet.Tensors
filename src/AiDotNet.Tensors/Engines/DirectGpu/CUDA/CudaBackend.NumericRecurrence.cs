// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The batched numeric recurrences (INumericRecurrenceBackend) on CUDA: NumericRecurrenceKernelTemplate's HIP dialect is
// CUDA C++ as well, compiled by NVRTC on first need and launched through PortableNumericRecurrence.
public sealed partial class CudaBackend : INumericRecurrenceBackend
{
    private readonly object _numericRecurrenceLock = new();
    private bool _numericRecurrenceTried;
    private bool _numericRecurrenceReady;

    // The module, built once with IEEE math (the log-domain lattice and the assignment's comparisons need it);
    // otherwise the host ops run, and the trace says why.
    private bool NumericRecurrenceReady
    {
        get
        {
            lock (_numericRecurrenceLock)
            {
                if (_numericRecurrenceTried) return _numericRecurrenceReady;
                _numericRecurrenceTried = true;
                try
                {
                    using var _ = PushContext();
                    if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                    CompileKernelModule(device, NumericRecurrenceKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip),
                        "numeric_recurrence_kernels", NumericRecurrenceKernelTemplate.GetKernelNames(), useFastMath: false);
                    bool ok = true;
                    foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames()) ok &= _kernelCache.ContainsKey(name);
                    _numericRecurrenceReady = ok;
                }
                catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
                {
                    System.Diagnostics.Trace.TraceWarning($"CUDA numeric recurrence kernels unavailable: {ex.GetType().Name}: {ex.Message}");
                }
                return _numericRecurrenceReady;
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
        using var _ = PushContext();
        LaunchKernel(_kernelCache[kernelName], (uint)groups, (uint)NumericRecurrenceKernelTemplate.GroupSize, args);
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
