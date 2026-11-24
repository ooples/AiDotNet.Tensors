using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    internal static bool GpuWeightNormWorkspaceEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_GPU_WEIGHTNORM_WORKSPACE") == "1";

    private static bool IsWeightNormWorkspaceConvolution(CompiledStep<T> step)
    {
        if (step.OpType is OpType.Conv2D or OpType.GroupedConv2D) return true;
        if (step.OpType == OpType.ConvTranspose2D)
            return step.BackwardFn == BackwardFunctions<T>.ConvTranspose2DBackward;
        // A transposed convolution with its bias fused (ConvTransposeBiasFusionPattern): kernel at slot 1 like the
        // unfused op, saved state (stride, padding).
        if (step.OpType == OpType.Unknown && step.OpName == "FusedConvTranspose2DBias")
            return step.Inputs.Length == 3
                && step.BackwardFn == BackwardFunctions<T>.FusedConvTranspose2DBiasBackward
                && step.SavedState is { Length: 2 } transposed && transposed[0] is int[] { Length: 2 }
                && transposed[1] is int[] { Length: 2 };
        // The graph optimizer represents this known convolution epilogue as a
        // custom node. Match its VJP and state contract as well as its name.
        return step.OpType == OpType.Unknown && step.OpName == "FusedConv2DBias"
            && step.Inputs.Length == 3
            && step.BackwardFn == BackwardFunctions<T>.FusedConv2DBiasActivationBackward
            && step.SavedState is { Length: 4 } state && state[0] is int[] { Length: 2 }
            && state[1] is int[] { Length: 2 } && state[2] is int[] { Length: 2 } && state[3] is bool;
    }

    // Only pure normalization outputs proven private by PrepareWeightNormWorkspace
    // use these banks. Spectral normalization is stateful and is never replayed.
    private sealed class DeviceWeightNormWorkspace : IDisposable
    {
        private readonly DirectGpuTensorEngine _engine;
        private readonly IDirectGpuBackend _backend;
        private readonly List<IGpuBuffer> _banks = new();
        private readonly Dictionary<WeightNormWorkspaceGroup, IGpuBuffer> _views = new();
        internal readonly HashSet<int> Normalizations = new();
        private bool _disposed;

        private DeviceWeightNormWorkspace(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
        { _engine = engine; _backend = backend; }

        internal static DeviceWeightNormWorkspace? TryCreate(DirectGpuTensorEngine engine,
            WeightNormWorkspaceGroup[] groups, List<int> capacities, IReadOnlyList<CompiledStep<T>> steps)
        {
            using var operation = engine.EnterGpuOp();
            if (engine.QueryBackend is not IGpuBufferViews views || engine.QueryBackend is not { } backend)
                return null;
            var workspace = new DeviceWeightNormWorkspace(engine, backend);
            bool complete = false;
            try
            {
                foreach (int capacity in capacities)
                    workspace._banks.Add(backend.AllocateBuffer(capacity));
                foreach (var group in groups)
                {
                    var view = views.TryCreateView(workspace._banks[group.Color], 0, group.Normalization.OutputBuffer.Length);
                    if (view is null) return null;
                    workspace._views.Add(group, view);
                }
                var outputs = new HashSet<Tensor<T>>(groups.SelectMany(group => group.Outputs),
                    ReferenceEqualityComparer<Tensor<T>>.Instance);
                for (int i = 0; i < steps.Count; i++)
                    if (IsPureWeightNorm(steps[i]) && outputs.Contains(steps[i].OutputBuffer))
                        workspace.Normalizations.Add(i);
                // No tensor changes until every backend view has been granted.
                foreach (var group in groups)
                    workspace.RefreshAliases(group, null);
                complete = true;
                return workspace;
            }
            finally
            {
                if (!complete) workspace.Dispose();
            }
        }

        internal void PrepareDestination(WeightNormWorkspaceGroup group, Tensor<T> destination)
            // Pending device ownership is what GetOrCreateResidentBuffer uses
            // to distinguish this plan-owned bank from an evicted cache entry.
            => _engine.BindResidentBuffer(destination, _views[group], _backend, isDeviceModified: true);

        internal void RefreshAliases(WeightNormWorkspaceGroup group, Tensor<T>? written)
        {
            if (written is not null && !ReferenceEquals(written._gpuBuffer, _views[group]))
                throw new InvalidOperationException("Normalization did not write its workspace bank.");
            // Aliases share a storage epoch: publish bindings only after every
            // alias advanced it, so all views receive the final current epoch.
            if (written is not null)
                foreach (var output in group.Outputs)
                    if (!ReferenceEquals(output, written)) output.IncrementVersion();
            foreach (var output in group.Outputs)
                _engine.BindResidentBuffer(output, _views[group], _backend, isDeviceModified: written is not null);
        }

        internal IEnumerable<Tensor<T>> Outputs => _views.Keys.SelectMany(group => group.Outputs);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // Releasing a plan is outside capture. Finish queued readers before
            // releasing native views (OpenCL/Vulkan views have their own handles).
            _backend.Synchronize();
            foreach (var pair in _views)
            {
                ReleaseViewOutputs(pair.Key.Outputs, pair.Value);
            }
            foreach (var view in _views.Values) view.Dispose();
            foreach (var bank in _banks) bank.Dispose();
            _views.Clear();
            _banks.Clear();
        }

        private static void ReleaseViewOutputs(System.Collections.Generic.IEnumerable<Tensor<T>> outputs, Engines.DirectGpu.IGpuBuffer view)
        {
            foreach (var output in outputs)
            {
                // View creation can be refused before any tensor was changed.
                // A rejected layout must leave the ordinary replay path intact.
                if (!ReferenceEquals(output._gpuBuffer, view)) continue;
                HostSync.Release(output.DataVector, DirectGpuTensorEngine.ReleasedIntermediateMessage);
                if (output.GetBackingArrayForCacheLookupUnsafe() is { } array)
                    HostSync.Release(array, DirectGpuTensorEngine.ReleasedIntermediateMessage);
                output._gpuBuffer = null;
                output._gpuBackend = null;
                output._gpuBufferVersion = -1;
            }
        }
    }
}
