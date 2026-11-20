using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    private sealed class HostIndexBinding
    {
        internal readonly HostIndexDeviceCopy Copy;
        internal readonly Tensor<int> Source;
        internal readonly IntPtr Handle;
        internal int UpperBound;
        internal bool RequireUnique;

        internal HostIndexBinding(HostIndexDeviceCopy copy, Tensor<int> source, int upperBound, bool requireUnique)
        {
            Copy = copy;
            Source = source;
            Handle = copy.Buffer!.Handle;
            UpperBound = upperBound;
            RequireUnique = requireUnique;
        }
    }

    private sealed class IndexReplayBindings
    {
        internal readonly DirectGpuTensorEngine Engine;
        internal readonly List<HostIndexBinding> Host = new();
        internal readonly List<(Tensor<int> Source, IntPtr Handle)> Resident = new();
        internal IndexReplayBindings(DirectGpuTensorEngine engine) => Engine = engine;
    }

    [ThreadStatic] private static IndexReplayBindings? t_indexReplayCapture;
    private readonly Dictionary<IntPtr, IndexReplayBindings> _indexReplayGraphs = new();
    private readonly object _indexReplayGraphLock = new();

    // Graph executables own strong references to exactly the index tensors they read. Vocabulary bounds belong to
    // each consumer graph, while device copies remain reusable by subsequent graphs with different vocabularies.
    internal IntPtr CaptureGraphWithIndexBindings(IGpuGraphCapture capture, Action forward)
    {
        var bindings = new IndexReplayBindings(this);
        IntPtr exec = CaptureIndexBindings(capture, forward, bindings);
        if (exec != IntPtr.Zero)
            lock (_indexReplayGraphLock) _indexReplayGraphs[exec] = bindings;
        return exec;
    }

    private static IntPtr CaptureIndexBindings(IGpuGraphCapture capture, Action forward, IndexReplayBindings bindings)
    {
        var previous = t_indexReplayCapture;
        t_indexReplayCapture = bindings;
        try
        {
            return capture.CaptureGraph(forward);
        }
        finally
        {
            t_indexReplayCapture = previous;
        }
    }

    private void RecordHostIndexBinding(HostIndexDeviceCopy copy, Tensor<int> source, int upperBound, bool requireUnique)
    {
        var bindings = t_indexReplayCapture;
        if (bindings is null || !ReferenceEquals(bindings.Engine, this)) return;
        foreach (var binding in bindings.Host)
            if (ReferenceEquals(binding.Copy, copy))
            {
                binding.RequireUnique |= requireUnique;
                if (upperBound > 0 && (binding.UpperBound == 0 || upperBound < binding.UpperBound))
                    binding.UpperBound = upperBound;
                return;
            }
        bindings.Host.Add(new HostIndexBinding(copy, source, upperBound, requireUnique));
    }

    private void RecordResidentIndexBinding(Tensor<int> indices, IntPtr handle)
    {
        var bindings = t_indexReplayCapture;
        if (bindings is null || !ReferenceEquals(bindings.Engine, this)) return;
        foreach (var binding in bindings.Resident)
            if (ReferenceEquals(binding.Source, indices)) return;
        bindings.Resident.Add((indices, handle));
    }

    private void RefreshIndexGraphBindings(IDirectGpuBackend backend, IntPtr graphExec)
    {
        IndexReplayBindings bindings;
        lock (_indexReplayGraphLock)
            if (!_indexReplayGraphs.TryGetValue(graphExec, out bindings!))
                throw new InvalidOperationException("The captured graph's index bindings have been retired.");
        foreach (var binding in bindings.Host)
        {
            if (binding.Copy.Buffer?.Handle != binding.Handle || binding.Handle == IntPtr.Zero)
                throw new InvalidOperationException("A captured host-index device buffer was released or replaced.");
            RefreshHostIndexDeviceCopy(backend, binding.Copy, binding.Source, binding.UpperBound, binding.RequireUnique);
        }
        foreach (var binding in bindings.Resident)
            RefreshResidentIndexInPlace(backend, binding.Source, binding.Handle);
    }

    internal void RetireIndexGraphBindings(IntPtr graphExec)
    {
        lock (_indexReplayGraphLock) _indexReplayGraphs.Remove(graphExec);
    }

    private void ReleaseIndexReplayBindings()
    {
        lock (_indexReplayGraphLock) _indexReplayGraphs.Clear();
    }
}
