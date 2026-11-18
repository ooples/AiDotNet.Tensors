using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    internal static bool CpuWeightNormWorkspaceEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_CPU_WEIGHTNORM_WORKSPACE") == "1";

    private CpuWeightNormWorkspace? _cpuWeightNormWorkspace;

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

    private sealed class CpuWeightNormWorkspace
    {
        private readonly Dictionary<int, WeightNormWorkspaceGroup> _consumers;
        private readonly IReadOnlyList<CompiledStep<T>> _steps;
        private readonly List<Tensor<T>> _temporaryViews = new();
        internal readonly long OriginalBytes;
        internal readonly long WorkspaceBytes;

        internal CpuWeightNormWorkspace(Dictionary<int, WeightNormWorkspaceGroup> consumers,
            IReadOnlyList<CompiledStep<T>> steps, long originalBytes, long workspaceBytes)
        { _consumers = consumers; _steps = steps; OriginalBytes = originalBytes; WorkspaceBytes = workspaceBytes; }

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
            engine.TensorWeightNormInto(_steps[consumer].Inputs[1], normalization.Inputs[0],
                normalization.Inputs[1], (double)normalization.SavedState![0]);
            // Independent tensor views intentionally share this private array. A
            // per-tensor version cannot describe which layer last wrote it.
            Engines.Simd.SimdGemm.MarkWeightDirty(group.Bank);
        }

        internal void WrapForward(Action<IEngine>[] actions, int[] actionSteps)
        {
            if (actions.Length != actionSteps.Length) throw new InvalidOperationException("Forward action metadata differs.");
            for (int i = 0; i < actions.Length; i++)
            {
                int step = actionSteps[i];
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

    private static CpuWeightNormWorkspace? PrepareCpuWeightNormWorkspace(
        IReadOnlyList<CompiledStep<T>> steps, Action<IEngine>[] forward, List<Action<IEngine>> backward,
        ForwardEmit[] forwardKinds, List<int[]?> coverage, Tensor<T>[] parameters, Tensor<T> loss,
        HashSet<Tensor<T>> retained, LazyTensorScope scope, IEngine engine)
    {
        if (!CpuWeightNormWorkspaceEnabled || !LazyTensorScope.DeferCpuTrainingOutputs || typeof(T) != typeof(float)
            || !engine.Is<CpuEngine>() || !ReferenceEquals(engine, engine.Unwrap())
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
        foreach (var tensor in parameters.Concat(retained).Append(loss))
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
                bool convolution = step.OpType is OpType.Conv2D or OpType.GroupedConv2D;
                if (input != 1 || !convolution || step.DifferentiableInputIndices is not null || !emitted.Contains(i))
                    group.Safe = false;
                else group.Consumers.Add(i);
            }
        }
        foreach (var tensor in tensors)
            if (groups.TryGetValue(tensor.StorageIdentity, out var group) && !group.Outputs.Contains(tensor)) group.Safe = false;
        var eligible = groups.Where(pair => pair.Value.Safe && !protectedStorage.Contains(pair.Key))
            .Select(pair => pair.Value).Where(group => group.Consumers.Count > 0
                && group.Outputs.All(output => output.HasDeferredHostStorage && !output.IsView && output._storageOffset == 0)
                && group.Normalization.Inputs.All(input => !lastWrite.TryGetValue(input.StorageIdentity, out int writer)
                    || writer < group.FirstStep)).ToArray();
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
        var banks = capacities.Select(length => new T[length]).ToArray();
        var layout = new CpuWeightNormWorkspace(byConsumer, steps, originalBytes, workspaceBytes);
        foreach (var group in eligible)
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
            AiDotNetEngine.LogDiagnostic($"[WN-WORKSPACE] groups={eligible.Length} banks={banks.Length} original_bytes={originalBytes} workspace_bytes={workspaceBytes}");
        return layout;
    }
}
