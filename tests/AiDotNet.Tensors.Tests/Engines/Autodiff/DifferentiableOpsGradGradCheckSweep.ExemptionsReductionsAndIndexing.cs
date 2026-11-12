namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public partial class DifferentiableOpsGradGradCheckSweep
{
    /// <summary>Documented exclusions for the reduction and indexing area (each with the reason).</summary>
    static partial void AddExemptionsReductionsAndIndexing()
    {
        // Class indices are integer targets stored as values: no gradient, as torch gives a target none.
        NonDifferentiableInputs["TensorGatherClassValues"] = (new[] { 1 }, "class indices: an integer target");
        NonDifferentiableInputs["TensorEmbeddingLookupFromFloatIndices"] = (new[] { 1 }, "float indices: integer row ids");

        // The table's index (built for dim 0) read along dim 1 leaves output groups empty, and ScatterMax's documented
        // empty-group value is -inf (argmax -1): the squared output is infinite, so its first-order finite differences
        // are NaN. The second order is still checked here, and ReductionsAndIndexingSecondOrderTests checks dim 1 with
        // every group hit, first and second order.
        const string emptyGroup = "an empty group's -inf output makes the squared loss infinite (first-order finite differences NaN)";
        FirstOrderExemptVariants["ScatterMax|dim=1"] = emptyGroup;
        FirstOrderExemptVariants["ScatterMax|dim=-1"] = emptyGroup;
    }
}
