using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Records a training-trace GroupNorm as one lazy node whose forward refreshes a shared statistics holder and whose
    /// backward reads that holder's replay-time mean and variance. The previous record site ran the normalization
    /// eagerly while tracing: a probe trace aborted on it, every upstream lazy value was realized mid-trace, and the
    /// trace-time statistics entered the saved state as external tensors the objective signature had to compare.
    /// </summary>
    /// <remarks>
    /// The statistics are lazy outputs of the same forward (<c>GroupNormMean</c> / <c>GroupNormVariance</c>), so a
    /// consumer is scheduled after the replay that refreshed them. Like the eager tape, no gradient flows through them.
    /// The saved state keeps the <c>[numGroups, mean, variance, epsilon]</c> layout every GroupNorm reader expects,
    /// with the holder in both statistic slots.
    /// </remarks>
    private static Tensor<T> RecordGroupNormTraining<T>(LazyTensorScope scope, Tensor<T> input, int numGroups,
        Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        var ci = input; var cg = gamma; var cb = beta; int cn = numGroups; double ce = epsilon;
        var statisticShape = new[] { input._shape[0], numGroups };
        var stateRef = new GroupNormStateRef<T>(input._shape[0] * numGroups);
        var lazyResult = scope.RecordVariadic(LazyNodeType.Custom, "GroupNorm",
            new[] { input, gamma, beta }, (int[])input._shape.Clone(),
            (eng, output) =>
            {
                var r = eng.GroupNorm(ci, cn, cg, cb, ce, out var freshMean, out var freshVar);
                DirectGpuTensorEngine.CopyResultInto(eng, r, output);
                stateRef.Refresh(freshMean, freshVar);
            },
            BackwardFunctions<T>.GroupNormBackward, new object[] { numGroups, stateRef, stateRef, epsilon });
        mean = scope.RecordUnary(LazyNodeType.Custom, "GroupNormMean", lazyResult, statisticShape,
            (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, stateRef.RequireMean(), output));
        variance = scope.RecordUnary(LazyNodeType.Custom, "GroupNormVariance", lazyResult, statisticShape,
            (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, stateRef.RequireVariance(), output));
        return lazyResult;
    }
}
