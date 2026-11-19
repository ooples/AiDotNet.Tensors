using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    internal static bool CpuWeightNormWorkspaceEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_CPU_WEIGHTNORM_WORKSPACE") == "1";

    private WeightNormWorkspaceLayout? _weightNormWorkspace;

    private sealed class WeightNormWorkspaceGroup
    {
        internal readonly CompiledStep<T> Normalization;
        internal readonly int FirstStep;
        internal readonly HashSet<Tensor<T>> Outputs = new(ReferenceEqualityComparer<Tensor<T>>.Instance);
        internal readonly List<int> Consumers = new();
        internal readonly HashSet<int> Conflicts = new();
        internal int Color;
        internal T[] Bank = Array.Empty<T>();
        internal bool Safe = true;
        internal WeightNormWorkspaceGroup(CompiledStep<T> step, int index)
        { Normalization = step; FirstStep = index; Outputs.Add(step.OutputBuffer); }
    }

    private sealed class WeightNormWorkspaceLayout
    {
        private readonly Dictionary<int, WeightNormWorkspaceGroup> _consumers;
        private readonly IReadOnlyList<CompiledStep<T>> _steps;
        private readonly List<Tensor<T>> _temporaryViews = new();
        private DeviceWeightNormWorkspace? _deviceWorkspace;
        internal readonly long OriginalBytes;
        internal readonly long WorkspaceBytes;

        internal WeightNormWorkspaceLayout(Dictionary<int, WeightNormWorkspaceGroup> consumers,
            IReadOnlyList<CompiledStep<T>> steps, long originalBytes, long workspaceBytes)
        { _consumers = consumers; _steps = steps; OriginalBytes = originalBytes; WorkspaceBytes = workspaceBytes; }

        internal void ProtectDeviceOutputs(Dictionary<Tensor<T>, int> lastReads)
        {
            if (_deviceWorkspace is not null)
                foreach (var output in _deviceWorkspace.Outputs) lastReads.Remove(output);
        }

        internal void Dispose() => _deviceWorkspace?.Dispose();
        internal void UseDeviceWorkspace(DeviceWeightNormWorkspace workspace) => _deviceWorkspace = workspace;

        internal void KeepUntilSnapshot(Tensor<T> view) => _temporaryViews.Add(view);
        internal void CompleteSnapshot()
        {
            foreach (var view in _temporaryViews) view.Dispose();
            _temporaryViews.Clear();
        }

        private void Refresh(IEngine engine, int consumer)
        {
            var group = _consumers[consumer];
            var normalization = group.Normalization;
            _deviceWorkspace?.PrepareDestination(group, _steps[consumer].Inputs[1]);
            engine.TensorWeightNormInto(_steps[consumer].Inputs[1], normalization.Inputs[0],
                normalization.Inputs[1], (double)normalization.SavedState![0]);
            // Independent tensor views intentionally share this private array. A
            // per-tensor version cannot describe which layer last wrote it.
            if (_deviceWorkspace is null) Engines.Simd.SimdGemm.MarkWeightDirty(group.Bank);
            else _deviceWorkspace.RefreshAliases(group, _steps[consumer].Inputs[1]);
        }

        internal void WrapForward(Action<IEngine>[] actions, int[] actionSteps)
        {
            if (actions.Length != actionSteps.Length) throw new InvalidOperationException("Forward action metadata differs.");
            for (int i = 0; i < actions.Length; i++)
            {
                int step = actionSteps[i];
                if (_deviceWorkspace?.Normalizations.Contains(step) == true)
                { actions[i] = _ => { }; continue; }
                if (!_consumers.ContainsKey(step)) continue;
                var action = actions[i];
                actions[i] = engine => { Refresh(engine, step); action(engine); };
            }
        }

        internal void WrapBackward(List<Action<IEngine>> actions, List<int[]?> coverage)
        {
            for (int i = 0; i < actions.Count; i++)
            {
                var readers = coverage[i]!.Where(_consumers.ContainsKey).Distinct().ToArray();
                if (readers.Length == 0) continue;
                var action = actions[i];
                actions[i] = engine =>
                {
                    foreach (int reader in readers) Refresh(engine, reader);
                    action(engine);
                };
            }
        }
    }

    private static WeightNormWorkspaceLayout? PrepareWeightNormWorkspace(
        IReadOnlyList<CompiledStep<T>> steps, Action<IEngine>[] forward, List<Action<IEngine>> backward,
        ForwardEmit[] forwardKinds, List<int[]?> coverage, Tensor<T>[] parameters, Tensor<T> loss,
        HashSet<Tensor<T>> retained, LazyTensorScope scope, IEngine engine)
    {
        bool cpu = CpuWeightNormWorkspaceEnabled && LazyTensorScope.DeferCpuTrainingOutputs
            && engine.Unwrap().GetType() == typeof(CpuEngine);
        bool gpu = GpuWeightNormWorkspaceEnabled && engine.Unwrap().GetType() == typeof(DirectGpuTensorEngine)
            && engine.Cast<DirectGpuTensorEngine>().SupportsWeightNormWorkspace
            && engine.Cast<DirectGpuTensorEngine>().QueryBackend is DirectGpu.IGpuBatchExecution
            && engine.Cast<DirectGpuTensorEngine>().QueryBackend is DirectGpu.IGpuBufferViews;
        if ((!cpu && !gpu) || typeof(T) != typeof(float)
            || !ReferenceEquals(engine, engine.Unwrap())
            || scope.HostReadOp is not null || ForwardStepObserver is not null || SanitizeForwardBuffers
            || coverage.Count != backward.Count || coverage.Any(row => row is null)) return null;
        var actionSteps = Enumerable.Range(0, forwardKinds.Length).Where(i => forwardKinds[i] != ForwardEmit.Skip).ToArray();
        if (actionSteps.Length != forward.Length) return null;
        var emitted = new HashSet<int>(actionSteps);
        var groups = new Dictionary<object, WeightNormWorkspaceGroup>(ReferenceEqualityComparer<object>.Instance);
        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];
            if (!IsPureWeightNorm(step)) continue;
            object storage = step.OutputBuffer.StorageIdentity;
            if (!groups.TryGetValue(storage, out var group)) groups[storage] = new WeightNormWorkspaceGroup(step, i);
            else
            {
                group.Outputs.Add(step.OutputBuffer);
                var prior = group.Normalization;
                group.Safe &= new WeightNormForwardKey(prior.Inputs[0], prior.Inputs[1], (double)prior.SavedState![0])
                    .Equals(new WeightNormForwardKey(step.Inputs[0], step.Inputs[1], (double)step.SavedState![0]));
            }
        }
        if (groups.Count < 2) return null;
        var tensors = new HashSet<Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);
        var protectedStorage = new HashSet<object>(ReferenceEqualityComparer<object>.Instance);
        foreach (var tensor in parameters.Concat(retained).Concat(new[] { loss }))
        { tensors.Add(tensor); protectedStorage.Add(tensor.StorageIdentity); }
        var visitor = new WeightNormSavedStorageVisitor(protectedStorage, tensors);
        var lastWrite = new Dictionary<object, int>(ReferenceEqualityComparer<object>.Instance);
        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i]; tensors.Add(step.OutputBuffer);
            if (!IsWeightNormMetadataView(step)) lastWrite[step.OutputBuffer.StorageIdentity] = i;
            SavedStateTensorTraversal.Visit(step.SavedState, ref visitor);
            for (int input = 0; input < step.Inputs.Length; input++)
            {
                var tensor = step.Inputs[input]; tensors.Add(tensor);
                if (!groups.TryGetValue(tensor.StorageIdentity, out var group)) continue;
                bool convolution = IsWeightNormWorkspaceConvolution(step);
                if (input != 1 || !convolution || step.DifferentiableInputIndices is not null || !emitted.Contains(i))
                    group.Safe = false;
                else group.Consumers.Add(i);
            }
        }
        foreach (var tensor in tensors)
            if (groups.TryGetValue(tensor.StorageIdentity, out var group) && !group.Outputs.Contains(tensor)) group.Safe = false;
        var eligible = groups.Where(pair => pair.Value.Safe && !protectedStorage.Contains(pair.Key))
            .Select(pair => pair.Value).Where(group => group.Consumers.Count > 0
                && group.Outputs.All(output => output.HasDeferredHostStorage && !output.IsView && output._storageOffset == 0
                    && (!gpu || output._gpuBuffer is null))
                && group.Normalization.Inputs.All(input => !lastWrite.TryGetValue(input.StorageIdentity, out int writer)
                    || writer < group.FirstStep)).ToArray();
        if (gpu && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
        {
            AiDotNetEngine.LogDiagnostic($"[WN-WORKSPACE-ELIGIBILITY] groups={groups.Count} safe={groups.Values.Count(group => group.Safe)} protected={groups.Count(pair => protectedStorage.Contains(pair.Key))} deferred={groups.Values.Count(group => group.Outputs.All(output => output.HasDeferredHostStorage))} bound={groups.Values.Count(group => group.Outputs.Any(output => output._gpuBuffer is not null))} eligible={eligible.Length}");
            var rejected = new Dictionary<string, int>();
            for (int i = 0; i < steps.Count; i++)
                for (int slot = 0; slot < steps[i].Inputs.Length; slot++)
                    if (groups.ContainsKey(steps[i].Inputs[slot].StorageIdentity)
                        && (slot != 1 || !IsWeightNormWorkspaceConvolution(steps[i])
                            || steps[i].DifferentiableInputIndices is not null || !emitted.Contains(i)))
                    {
                        string reason = $"{steps[i].OpName}:{steps[i].OpType}:slot{slot}:custom{steps[i].DifferentiableInputIndices is not null}:emitted{emitted.Contains(i)}";
                        rejected.TryGetValue(reason, out int count);
                        rejected[reason] = count + 1;
                    }
            AiDotNetEngine.LogDiagnostic("[WN-WORKSPACE-CONSUMERS] " + string.Join(";", rejected.Select(pair => $"{pair.Key}={pair.Value}")));
        }

        if (eligible.Length < 2) return null;
        var byConsumer = new Dictionary<int, WeightNormWorkspaceGroup>();
        var groupIndex = new Dictionary<WeightNormWorkspaceGroup, int>();
        for (int i = 0; i < eligible.Length; i++)
        {
            groupIndex[eligible[i]] = i;
            foreach (int consumer in eligible[i].Consumers) byConsumer[consumer] = eligible[i];
        }
        // Kernels read by one atomic backward must occupy different banks.
        foreach (var row in coverage)
        {
            var readers = row!.Where(byConsumer.ContainsKey).Select(i => groupIndex[byConsumer[i]]).Distinct().ToArray();
            foreach (int a in readers) foreach (int b in readers) if (a != b) eligible[a].Conflicts.Add(b);
        }
        var capacities = new List<int>(); var colored = new HashSet<int>();
        foreach (int index in Enumerable.Range(0, eligible.Length).OrderByDescending(i => eligible[i].Normalization.OutputBuffer.Length))
        {
            var group = eligible[index]; int color = 0;
            while (group.Conflicts.Any(other => colored.Contains(other) && eligible[other].Color == color)) color++;
            if (color == capacities.Count) capacities.Add(0);
            capacities[color] = Math.Max(capacities[color], group.Normalization.OutputBuffer.Length);
            group.Color = color; colored.Add(index);
        }
        int elementBytes = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        long originalBytes = eligible.Sum(group => (long)group.Normalization.OutputBuffer.Length * elementBytes);
        long workspaceBytes = capacities.Sum(length => (long)length * elementBytes);
        if (workspaceBytes >= originalBytes) return null;
        var banks = gpu ? Array.Empty<T[]>() : capacities.Select(length => new T[length]).ToArray();
        var layout = new WeightNormWorkspaceLayout(byConsumer, steps, originalBytes, workspaceBytes);
        if (gpu)
        {
            var device = DeviceWeightNormWorkspace.TryCreate(engine.Cast<DirectGpuTensorEngine>(), eligible, capacities, steps);
            if (device is null) return null;
            layout.UseDeviceWorkspace(device);
        }
        else foreach (var group in eligible)
        {
            group.Bank = banks[group.Color];
            foreach (var output in group.Outputs)
            {
                var view = Tensor<T>.FromMemory(new Memory<T>(group.Bank, 0, output.Length), (int[])output._shape.Clone());
                layout.KeepUntilSnapshot(view);
                output.RebindStorageFromGraph(view);
            }
        }
        layout.WrapForward(forward, actionSteps);
        layout.WrapBackward(backward, coverage);
        if (Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
            AiDotNetEngine.LogDiagnostic($"[WN-WORKSPACE] device={(gpu ? "gpu" : "cpu")} groups={eligible.Length} banks={capacities.Count} original_bytes={originalBytes} workspace_bytes={workspaceBytes}");
        return layout;
    }
}
