// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// GPU compiled plan that captures fused execution as a CUDA graph via <see cref="StaticShapeCudaGraphScope"/>
/// for zero per-kernel launch overhead on subsequent static-shape replays, with automatic shape and device-pointer
/// invalidation guards and eager fallback on non-graph backends.
/// </summary>
internal sealed class GpuCompiledPlan<T> : IDisposable
{
    private readonly CompiledStep<T>[] _steps;
    private readonly Tensor<T> _finalOutput;
    private readonly IEngine _engine;
    private readonly Tensor<T>[] _watchedTensors;
    private StaticShapeCudaGraphScope? _graphScope;
    private IDirectGpuBackend? _boundBackend;
    private bool _disposed;

    internal GpuCompiledPlan(CompiledStep<T>[] steps, Tensor<T> finalOutput, IEngine engine, IDirectGpuBackend? explicitBackend = null)
    {
        _steps = steps ?? throw new ArgumentNullException(nameof(steps));
        _finalOutput = finalOutput ?? throw new ArgumentNullException(nameof(finalOutput));
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _watchedTensors = CollectWatchedTensors(steps, finalOutput);

        IDirectGpuBackend? backend = explicitBackend;
        if (backend is null && engine.TryGet<DirectGpuTensorEngine>(out var gpuEngine))
        {
            // Backend discovery under an op lease, so engine disposal cannot free the backend mid-read.
            using var lease = gpuEngine.EnterGpuOp();
            backend = gpuEngine.GetBackend();
        }

        if (backend is not null)
        {
            _boundBackend = backend;
            _graphScope = CreateScope(backend);
        }
    }

    // The plan's steps recompute their outputs from device inputs only, so re-running them after a failed capture
    // cannot double-apply host-side state.
    private static StaticShapeCudaGraphScope CreateScope(IDirectGpuBackend backend) => new(
        backend,
        CudaGraphExecutionScopeKind.InferenceForward,
        new StaticShapeCudaGraphOptions { WarmupIterations = 1, SynchronizeOnReplay = true, StepBodyIsReplayable = true });

    /// <summary>Whether a CUDA graph has been captured and is active for zero-overhead replay.</summary>
    internal bool IsCaptured => _graphScope?.HasCapturedGraph == true;

    /// <summary>Underlying guarded CUDA Graph scope when bound to a DirectGpu backend.</summary>
    internal StaticShapeCudaGraphScope? GraphScope => _graphScope;

    /// <summary>
    /// Executes the plan. Warms up and captures a CUDA graph when supported and all watched tensors are
    /// device-resident; replays via <c>cuGraphLaunch</c> while shapes and device pointers remain invariant,
    /// and automatically invalidates + recaptures if any shape or pointer changes.
    /// </summary>
    internal Tensor<T> Execute()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GpuCompiledPlan<T>));
        using var recordingSuspension = GraphMode.SuspendRecording();

        // Hold an engine op lease across backend discovery and graph replay (nested leases are allowed, so the
        // steps' own ops do not conflict). A closed engine yields an empty lease and the sequential fallback.
        DirectGpuTensorEngine? gpuEngine = _engine.TryGet<DirectGpuTensorEngine>(out var engineGpu) ? engineGpu : null;
        using var lease = gpuEngine?.EnterGpuOp();

        EnsureGraphScopeInitialized(gpuEngine);
        if (_graphScope is not null && _watchedTensors.Length > 0)
        {
            long replaysBefore = _graphScope.GraphReplayCount;
            _graphScope.Execute(RunSequentialSteps, _watchedTensors);
            if (gpuEngine is not null && _graphScope.GraphReplayCount != replaysBefore)
            {
                RearmOutputsAfterReplay(gpuEngine);
            }
            return _finalOutput;
        }

        RunSequentialSteps();
        return _finalOutput;
    }

    private void RunSequentialSteps()
    {
        var engine = _engine;
        var steps = _steps;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }
    }

    // A graph replay writes the outputs on the device without going through the engine's bind path, so a host copy
    // that was already downloaded (its pending download consumed) would keep the previous values. Re-arm each written
    // tensor's download exactly as the eager path does after a kernel writes a resident buffer.
    private void RearmOutputsAfterReplay(DirectGpuTensorEngine gpuEngine)
    {
        var rearmed = new HashSet<Tensor<T>>(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);
        for (int i = 0; i < _steps.Length; i++)
        {
            RearmOne(gpuEngine, _steps[i].OutputBuffer, rearmed);
        }
        RearmOne(gpuEngine, _finalOutput, rearmed);
    }

    private static void RearmOne(DirectGpuTensorEngine gpuEngine, Tensor<T>? tensor, HashSet<Tensor<T>> rearmed)
    {
        if (tensor is null || !rearmed.Add(tensor)) return;
        if (tensor._gpuBuffer is { } buffer && buffer.Handle != IntPtr.Zero)
        {
            gpuEngine.RearmStateAfterReplay(tensor, buffer);
        }
    }

    private void EnsureGraphScopeInitialized(DirectGpuTensorEngine? gpuEngine)
    {
        if (_graphScope is not null || gpuEngine is null) return;
        var backend = gpuEngine.GetBackend();
        if (backend is not null && !ReferenceEquals(backend, _boundBackend))
        {
            _boundBackend = backend;
            _graphScope = CreateScope(backend);
        }
    }

    private static Tensor<T>[] CollectWatchedTensors(CompiledStep<T>[] steps, Tensor<T> finalOutput)
    {
        var unique = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>(Autodiff.ReferenceEqualityComparer<Tensor<T>>.Instance);

        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            if (step.Inputs is not null)
            {
                for (int k = 0; k < step.Inputs.Length; k++)
                {
                    var inp = step.Inputs[k];
                    if (inp is not null && seen.Add(inp))
                        unique.Add(inp);
                }
            }
            if (step.OutputBuffer is not null && seen.Add(step.OutputBuffer))
            {
                unique.Add(step.OutputBuffer);
            }
        }

        if (seen.Add(finalOutput))
        {
            unique.Add(finalOutput);
        }

        return unique.ToArray();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _graphScope?.Dispose();
        _graphScope = null;
    }
}
