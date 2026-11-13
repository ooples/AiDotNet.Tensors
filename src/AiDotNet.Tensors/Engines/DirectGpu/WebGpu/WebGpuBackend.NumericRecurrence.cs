#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The batched numeric recurrences (INumericRecurrenceBackend) on WebGPU, through PortableNumericRecurrence and the WGSL
// restatement of NumericRecurrenceKernelTemplate (WebGpuNumericRecurrenceKernels), built on first need.
public sealed partial class WebGpuBackend : INumericRecurrenceBackend
{
    private const string NumericRecurrenceModulePrefix = "NumericRecurrence_";
    private bool? _numericRecurrenceReady;

    // Every pipeline, built once; a module the implementation rejects leaves the host ops, and says why.
    private bool NumericRecurrenceReady
    {
        get
        {
            if (_numericRecurrenceReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_numericRecurrenceReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(NumericRecurrenceModulePrefix + name,
                                    WebGpuNumericRecurrenceKernels.SourceFor(name), name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU numeric recurrence kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _numericRecurrenceReady = ok;
                }
                return _numericRecurrenceReady.Value;
            }
        }
    }

    private void LaunchNumericRecurrence(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds eight ints (bit-cast); the unused words are padding.
        var uniforms = new float[WebGpuNumericRecurrenceKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(NumericRecurrenceModulePrefix + kernelName, WebGpuNumericRecurrenceKernels.SourceFor(kernelName),
            kernelName, buffers, uniforms, groups * NumericRecurrenceKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif
