using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // What a recorded softmax cross-entropy node's forward leaves for its backward, in device buffers allocated at the
    // node's first run and reused by every replay (a captured graph bakes their pointers).
    private sealed class SoftmaxCrossEntropyRecordedNode
    {
        public Tensor<float>? RowStats { get; set; }
        public Tensor<float>? Count { get; set; }
    }

    /// <inheritdoc/>
    /// <remarks>Under a compiled trace on a backend with fused kernels (<see cref="ISoftmaxCrossEntropyBackend"/>) the
    /// loss is ONE node -- two forward kernels and one backward kernel that writes the logits' gradient straight into
    /// its accumulator. Everywhere else it is the composed primitives, as before.</remarks>
    public override Tensor<T> TensorSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        ValidateSoftmaxCrossEntropy(logits, target);
        if (TryRecordSoftmaxCrossEntropy(logits, target) is { } recorded) return recorded;
        return base.TensorSoftmaxCrossEntropy(logits, target);
    }

    private Tensor<T>? TryRecordSoftmaxCrossEntropy<T>(Tensor<T> logits, Tensor<T> target)
    {
        // PlacementBackend, not GetBackend: the latter refuses every backend while a graph is traced (#350).
        if (typeof(T) != typeof(float) || Gpu.AutocastScope.IsEnabled
            || GraphMode.Current is not { } scope
            || PlacementBackend is not ISoftmaxCrossEntropyBackend
            || !logits.IsContiguous || !target.IsContiguous || logits.Length == 0)
            return null;
        int cols = logits._shape[logits.Rank - 1];
        if (cols <= 0) return null;
        int rows = logits.Length / cols;
        if (!scope.BindEngine(this)) return null;

        var node = new SoftmaxCrossEntropyRecordedNode();
        return scope.RecordVariadic(LazyNodeType.Custom, "SoftmaxCrossEntropy", new[] { logits, target }, Array.Empty<int>(),
            (eng, output) =>
            {
                var gpu = (DirectGpuTensorEngine)eng;
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                var fused = (ISoftmaxCrossEntropyBackend)backend;
                node.RowStats ??= Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(4 * rows), new[] { 4 * rows },
                    Gpu.GpuTensorRole.Intermediate, ownsBuffer: true);
                node.Count ??= Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(1), new[] { 1 },
                    Gpu.GpuTensorRole.Intermediate, ownsBuffer: true);
                var stats = node.RowStats._gpuBuffer ?? throw new InvalidOperationException("The loss node lost its row statistics.");
                var count = node.Count._gpuBuffer ?? throw new InvalidOperationException("The loss node lost its row count.");
                using var z = gpu.GetOrAllocateBuffer(backend, logits);
                using var t = gpu.GetOrAllocateBuffer(backend, target);
                var loss = AllocateFullyWrittenOutputBuffer(backend, 1);
                bool ownsLoss = true;
                try
                {
                    if (!fused.TrySoftmaxCrossEntropyForward(z.Buffer, t.Buffer, stats, loss.Buffer, count, rows, cols))
                        throw new NotSupportedException("The backend declined the fused softmax cross-entropy.");
                    var tensor = gpu.DeferTensorResult<T>(backend, loss.Buffer, 1, Array.Empty<int>());
                    loss.RelinquishOwnership();
                    ownsLoss = false;
                    CopyResultInto(eng, tensor, output);
                }
                finally
                {
                    if (ownsLoss) loss.Dispose();
                }
            },
            (gradOutput, inputs, output, savedState, engine, grads) =>
            {
                var gpu = (DirectGpuTensorEngine)engine;
                var backend = gpu.GetBackend() ?? throw new InvalidOperationException("No GPU backend.");
                var fused = (ISoftmaxCrossEntropyBackend)backend;
                var stats = node.RowStats?._gpuBuffer ?? throw new InvalidOperationException("The loss node's backward ran before its forward.");
                var count = node.Count?._gpuBuffer ?? throw new InvalidOperationException("The loss node's backward ran before its forward.");
                using var z = gpu.GetOrAllocateBuffer(backend, inputs[0]);
                using var t = gpu.GetOrAllocateBuffer(backend, inputs[1]);
                using var g = gpu.GetOrAllocateBuffer(backend, gradOutput);
                // In a resident step the logits' gradient lands in its accumulator directly.
                var into = Autodiff.DifferentiableOps.PeekDirectDeviceGradTarget(grads, inputs[0], engine, out bool add);
                var target = gpu.ResolveDirectGradTarget(backend, into, inputs[0].Length, z.Buffer, t.Buffer, g.Buffer);
                if (into is not null && target is not null && target.Handle != stats.Handle && target.Handle != count.Handle)
                {
                    if (!fused.TrySoftmaxCrossEntropyBackward(z.Buffer, t.Buffer, stats, count, g.Buffer, target, rows, cols, add))
                        throw new NotSupportedException("The backend declined the fused softmax cross-entropy backward.");
                    into.IncrementVersion();
                    gpu.BindResidentBuffer(into, target, backend);
                    Autodiff.DifferentiableOps.CommitDirectGradWrite(grads, inputs[0], into);
                    return;
                }
                var dz = AllocateFullyWrittenOutputBuffer(backend, inputs[0].Length);
                bool owns = true;
                try
                {
                    if (!fused.TrySoftmaxCrossEntropyBackward(z.Buffer, t.Buffer, stats, count, g.Buffer, dz.Buffer, rows, cols, add: false))
                        throw new NotSupportedException("The backend declined the fused softmax cross-entropy backward.");
                    var grad = gpu.DeferTensorResult<T>(backend, dz.Buffer, inputs[0].Length, (int[])inputs[0]._shape.Clone());
                    dz.RelinquishOwnership();
                    owns = false;
                    Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
                }
                finally
                {
                    if (owns) dz.Dispose();
                }
            },
            new object[] { node });
    }
}
