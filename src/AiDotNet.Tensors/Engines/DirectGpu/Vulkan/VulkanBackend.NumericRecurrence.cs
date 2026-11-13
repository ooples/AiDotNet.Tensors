// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// The batched numeric recurrences (INumericRecurrenceBackend) on Vulkan, through PortableNumericRecurrence and
// NumericRecurrenceKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : INumericRecurrenceBackend
{
    private static readonly Dictionary<string, string> s_numericRecurrenceShaders = BuildNumericRecurrenceShaders();

    private static Dictionary<string, string> BuildNumericRecurrenceShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames())
            shaders[name] = NumericRecurrenceKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _numericRecurrenceReady;

    // Every pipeline, built once; a shader the driver rejects leaves the host ops, and says why.
    private bool NumericRecurrenceReady
    {
        get
        {
            if (_numericRecurrenceReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_numericRecurrenceReady is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in NumericRecurrenceKernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_numericRecurrenceShaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan numeric recurrence kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan numeric recurrence kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _numericRecurrenceReady = ok;
                }
                return _numericRecurrenceReady.Value;
            }
        }
    }

    private void LaunchNumericRecurrence(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // Explicit push words, one per int, in the shader's push-constant order (no 64-bit pairs to align).
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_numericRecurrenceShaders[kernelName], push, groups * NumericRecurrenceKernelTemplate.GroupSize, buffers);
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
