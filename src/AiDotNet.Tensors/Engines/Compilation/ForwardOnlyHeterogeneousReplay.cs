using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// One complete forward schedule for data branches crossing element types. Cross-type edges are
/// gradient stops; parameter backward remains in the training plan's element type.
/// </summary>
internal sealed class ForwardOnlyHeterogeneousReplay
{
    internal const string UnsupportedScheduleReason =
        "Heterogeneous forward-data replay requires its complete per-node schedule; typed-only checkpoint, frozen-weight, serialization and static-pointer schedules are not supported.";

    private readonly ILazyNode[] _nodes;
    private readonly Action<IEngine>[] _actions;

    private ForwardOnlyHeterogeneousReplay(ILazyNode[] nodes)
    {
        _nodes = nodes;
        _actions = nodes.Select(node => node is ICompiledForwardNode dispatch
            ? dispatch.CompileForwardAction()
            : throw new NotSupportedException($"No compiled forward dispatch for {node.OpName}.")).ToArray();
    }

    internal int Count => _actions.Length;

    internal static ForwardOnlyHeterogeneousReplay? Create<T>(IReadOnlyList<ILazyNode> nodes)
    {
        if (nodes.All(node => node is LazyNode<T>)) return null;
        var bridges = nodes.OfType<ICrossTypeNodeMetadata>().ToArray();
        if (bridges.Any(bridge => bridge.HasBackward))
        {
            // Preserve the existing differentiable FP32/FP16 implementation. It is not a general
            // heterogeneous backward and must not silently accept a double or integer bridge.
            bool existingFp16 = typeof(T) == typeof(float)
                && nodes.All(node => node is LazyNode<float> || node is LazyNode<Half>
                    || node is ICrossTypeNodeMetadata cross
                    && ((cross.InputElementType == typeof(float) && cross.OutputElementType == typeof(Half))
                        || (cross.InputElementType == typeof(Half) && cross.OutputElementType == typeof(float))));
            if (existingFp16) return null;
            throw new NotSupportedException("Differentiable heterogeneous training is supported only by the existing FP32/FP16 path; forward-data cross-type bridges must have no backward.");
        }
        return new ForwardOnlyHeterogeneousReplay(nodes.ToArray());
    }

    internal void AcquireStorage(TensorStorageLeaseSet leases)
    {
        foreach (var node in _nodes) node.AddStorageLeases(leases);
    }

    internal void DetachOutputs()
    {
        foreach (var node in _nodes) node.ClearOutputLazySource();
    }

    internal void Execute(IEngine engine)
    {
        // This replaces the typed forward schedule. Each action writes its original output slot;
        // none calls Realize, so an input read cannot recursively repeat RNG or an in-place update.
        foreach (var action in _actions) action(engine);
    }

    internal void ProtectDeviceState(HashSet<object> keys)
    {
        foreach (var node in _nodes)
            if (node is IDeviceStateProtectionNode stateful)
                stateful.ProtectDeviceState(keys);
    }

    internal void DetachDeviceBindings(HashSet<DirectGpu.IGpuBuffer> released, HashSet<object> retained)
    {
        foreach (var tensor in retained.OfType<ITensorStorageLeaseSource>().ToArray()) retained.Add(tensor.StorageIdentity);
        // A view can share the state tensor's pending download. Detaching that view's backing
        // would discard the state's current value, so protect every alias by storage identity.
        ProtectDeviceState(retained);
        var handles = new HashSet<IntPtr>(released.Select(buffer => buffer.Handle).Where(handle => handle != IntPtr.Zero));
        foreach (ICompiledForwardNode node in _nodes) node.DetachDeviceBindings(released, handles, retained);
    }

    internal static void ProtectTensor<T>(HashSet<object> keys, Tensor<T> tensor)
    {
        keys.Add(((ITensorStorageLeaseSource)tensor).StorageIdentity);
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } backing) keys.Add(backing);
        keys.Add(tensor.DataVector);
        if (tensor._gpuBuffer is { } buffer) keys.Add(buffer);
    }

    internal static void DetachTensor<T>(Tensor<T>? tensor, HashSet<DirectGpu.IGpuBuffer> released,
        HashSet<IntPtr> handles, bool recomputedOutput = false)
    {
        if (tensor?._gpuBuffer is not { } buffer
            || !recomputedOutput && !released.Contains(buffer) && !handles.Contains(buffer.Handle)) return;
        if (tensor.GetBackingArrayForCacheLookupUnsafe() is { } backing) Helpers.HostSync.Remove(backing);
        tensor._gpuBuffer = null;
        tensor._gpuBackend = null;
        tensor._gpuBufferVersion = -1;
    }

    internal static void CommitOutput<T>(Tensor<T> output, string operation)
    {
        if (output._gpuBuffer is null && !output.HasPendingGpuData) output.IncrementVersion();
        DirectGpuTensorEngine.TagProducer(output, operation);
    }
}
