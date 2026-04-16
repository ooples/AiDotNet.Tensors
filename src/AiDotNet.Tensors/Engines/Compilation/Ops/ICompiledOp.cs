using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Ops;

/// <summary>
/// Base interface for typed IR operations. Each implementation encapsulates an
/// operation's metadata (attributes like numGroups, epsilon, stride, etc.) and
/// can build the forward/backward closures needed by the compilation pipeline.
///
/// <para><b>Why class-per-op?</b> The existing infrastructure uses
/// <see cref="OpType"/> enum + <see cref="CompiledStep{T}"/> with opaque
/// <c>object[]</c> SavedState. That works for execution but fusion passes
/// can't introspect attributes without type-unsafe casts. Typed Op classes let
/// fusion passes pattern-match: <c>if (op is GroupNormOp gn) { ... fuse with
/// next SiLU ... }</c>. Both representations coexist — Op classes produce
/// CompiledSteps, and existing CompiledSteps without an Op class continue to
/// work unchanged.</para>
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
internal interface ICompiledOp<T>
{
    /// <summary>The operation type enum value.</summary>
    OpType OpType { get; }

    /// <summary>The operation name string (must match <see cref="OpTypeParser"/>).</summary>
    string OpName { get; }

    /// <summary>Input tensor references (captured at trace time).</summary>
    Tensor<T>[] Inputs { get; }

    /// <summary>The expected output shape.</summary>
    int[] OutputShape { get; }

    /// <summary>
    /// Builds the forward-pass closure for a <see cref="CompiledStep{T}"/>.
    /// The closure calls the engine method with the op's stored attributes
    /// and writes the result into the pre-allocated output buffer.
    /// </summary>
    Action<IEngine, Tensor<T>> BuildForwardClosure();

    /// <summary>
    /// Returns the backward function delegate for gradient computation,
    /// or null if this op doesn't support training (inference-only fused ops).
    /// </summary>
    BackwardFunction<T>? GetBackwardFunction();

    /// <summary>
    /// Builds the SavedState array for serialization + backward pass.
    /// Must match the order that <see cref="GetBackwardFunction"/>'s
    /// delegate reads from.
    /// </summary>
    object[]? BuildSavedState();

    /// <summary>
    /// Converts this Op into a <see cref="CompiledStep{T}"/> ready for
    /// insertion into a compiled plan's step array.
    /// </summary>
    CompiledStep<T> ToCompiledStep(Tensor<T> outputBuffer);
}
