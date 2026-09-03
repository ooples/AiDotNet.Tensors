// Copyright (c) AiDotNet. All rights reserved.

using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Captures a kernel as one opaque inference node while preserving the public tensor operands
    /// as graph leaves. The eager trace invocation establishes the exact output shape; replay calls
    /// the engine bound to the graph, so a DirectGpuTensorEngine trace continues to dispatch to GPU
    /// kernels instead of being pinned to the CPU implementation.
    /// </summary>
    /// <remarks>
    /// This helper is inference-only by construction. Training and compatibility traces continue
    /// through each method's primitive implementation so an opaque node cannot silently erase a
    /// backward graph.
    /// </remarks>
    protected Tensor<T> CaptureInferenceKernel<T>(
        Tensor<T>[] inputs,
        Func<IEngine, Tensor<T>> execute,
        [CallerMemberName] string operationName = "")
    {
        var scope = GraphMode.Current
            ?? throw new InvalidOperationException("Inference kernel capture requires an active graph scope.");
        if (!GraphMode.IsInferenceTrace)
            throw new InvalidOperationException("Opaque kernel capture is valid only for an inference trace.");
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("An inference kernel must have at least one tensor input.", nameof(inputs));
        if (execute is null) throw new ArgumentNullException(nameof(execute));

        scope.BindEngineIfUnset(this);

        Tensor<T> tracedOutput;
        using (GraphMode.SuspendRecording())
            tracedOutput = execute(this);

        return scope.RecordMaterializedVariadic(
            LazyNodeType.Custom,
            operationName,
            inputs,
            tracedOutput,
            (engine, output) =>
            {
                Tensor<T> result = execute(engine);
                DirectGpuTensorEngine.CopyResultInto(engine, result, output);
            });
    }

    /// <summary>
    /// Captures every tensor returned by a homogeneous multi-output inference kernel. Each output
    /// remains independently selectable as a compiled-plan result. Coordinating sibling execution
    /// into a single launch is a performance optimization; it is deliberately separate from this
    /// correctness boundary.
    /// </summary>
    protected Tensor<T>[] CaptureInferenceKernelOutputs<T>(
        Tensor<T>[] inputs,
        Func<IEngine, Tensor<T>[]> execute,
        [CallerMemberName] string operationName = "")
    {
        var scope = GraphMode.Current
            ?? throw new InvalidOperationException("Inference kernel capture requires an active graph scope.");
        if (!GraphMode.IsInferenceTrace)
            throw new InvalidOperationException("Opaque kernel capture is valid only for an inference trace.");
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("An inference kernel must have at least one tensor input.", nameof(inputs));
        if (execute is null) throw new ArgumentNullException(nameof(execute));

        scope.BindEngineIfUnset(this);

        Tensor<T>[] tracedOutputs;
        using (GraphMode.SuspendRecording())
            tracedOutputs = execute(this);
        if (tracedOutputs is null || tracedOutputs.Length == 0)
            throw new InvalidOperationException("A multi-output inference kernel returned no tensors.");

        var captured = new Tensor<T>[tracedOutputs.Length];
        for (int outputIndex = 0; outputIndex < captured.Length; outputIndex++)
        {
            int selectedOutput = outputIndex;
            captured[outputIndex] = scope.RecordMaterializedVariadic(
                LazyNodeType.Custom,
                $"{operationName}[{selectedOutput}]",
                inputs,
                tracedOutputs[outputIndex],
                (engine, output) =>
                {
                    Tensor<T>[] results = execute(engine);
                    if (results.Length != tracedOutputs.Length)
                        throw new InvalidOperationException(
                            $"{operationName} changed its output count between trace and replay.");
                    DirectGpuTensorEngine.CopyResultInto(engine, results[selectedOutput], output);
                });
        }

        return captured;
    }

    /// <summary>
    /// Captures an inference kernel whose public contract writes into a caller-provided output.
    /// The destination is the node output, not an input dependency, so memory planning may safely
    /// bind its storage while the read operands remain explicit graph leaves.
    /// </summary>
    protected void CaptureInferenceIntoKernel<T>(
        Tensor<T> destination,
        Tensor<T>[] inputs,
        Action<IEngine, Tensor<T>> execute,
        [CallerMemberName] string operationName = "")
    {
        var scope = GraphMode.Current
            ?? throw new InvalidOperationException("Inference kernel capture requires an active graph scope.");
        if (!GraphMode.IsInferenceTrace)
            throw new InvalidOperationException("Opaque kernel capture is valid only for an inference trace.");
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length == 0)
            throw new ArgumentException("An inference kernel must have at least one tensor input.", nameof(inputs));
        if (execute is null) throw new ArgumentNullException(nameof(execute));

        scope.BindEngineIfUnset(this);
        using (GraphMode.SuspendRecording())
            execute(this, destination);

        scope.RecordMaterializedVariadic(
            LazyNodeType.Custom,
            operationName,
            inputs,
            destination,
            execute);
    }
}
