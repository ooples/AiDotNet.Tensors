using System;

namespace AiDotNet.Tensors.Engines.Optimization.Optimizers;

/// <summary>
/// Marker attribute declaring that an optimizer's <see cref="IOptimizer.Step"/> is safe
/// to capture inside a CUDA graph. The contract is:
///  * No host-side branches whose condition depends on a runtime tensor value
///    (host-side branches on a fixed integer <c>step</c> counter are fine).
///  * No reallocation of state buffers during <see cref="IOptimizer.Step"/>.
///    All buffers are pre-allocated by <see cref="OptimizerBase.GetOrCreateState"/>
///    on the first step before capture.
///  * No data-dependent dispatch (<c>if (grad.AnyZero) ...</c>). The 2:4
///    <see cref="SparseAdam24Optimizer"/> avoids this by encoding the sparsity
///    pattern in metadata; SparseAdam (dense-zero detection) is therefore
///    NOT capture-safe and is not annotated.
///
/// Every optimizer wrapper in this assembly that satisfies the contract carries
/// this attribute. Tests assert the attribute appears on the expected types.
/// </summary>
[AttributeUsage(AttributeTargets.Class)]
public sealed class CudaGraphSafeAttribute : Attribute
{
    /// <summary>Optional human-readable note about preconditions or caveats.</summary>
    public string? Note { get; set; }
}
