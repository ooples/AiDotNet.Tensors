// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL, HIP and Vulkan instances, whose NumericRecurrencePortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// The batched numeric recurrences (INumericRecurrenceBackend) on Metal, through PortableNumericRecurrence and
// NumericRecurrenceKernelTemplate's kernels, compiled as their own strict-math library on first need.
public sealed partial class MetalBackend : INumericRecurrenceBackend
{
    private const string NumericRecurrenceLibraryName = "NumericRecurrence";
    private IntPtr _numericRecurrenceLibrary;
    private bool? _numericRecurrenceReady;

    // The library, compiled once with fast math off (the log-domain lattice and the assignment's comparisons need IEEE
    // semantics), and every kernel able to run 256 threads in one threadgroup; otherwise the host ops run.
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
                        _numericRecurrenceLibrary = _shaderLibrary.CompileLibrary(NumericRecurrenceLibraryName,
                            NumericRecurrenceKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal), requireStrictFloatingPoint: true);
                        ok = _numericRecurrenceLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(NumericRecurrenceLibraryName, _numericRecurrenceLibrary, name).MaxTotalThreadsPerThreadgroup
                                      >= NumericRecurrenceKernelTemplate.GroupSize;
                    }
                    catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal numeric recurrence kernels unavailable: {ex.Message}");
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
        var pipeline = GetPipeline(NumericRecurrenceLibraryName, _numericRecurrenceLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)NumericRecurrenceKernelTemplate.GroupSize, 1, 1));
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
