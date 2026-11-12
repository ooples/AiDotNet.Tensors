namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Documented exclusions for the ops that take a fused activation epilogue.</summary>
public partial class DifferentiableOpsGradGradCheckSweep
{
    static partial void AddExemptionsFusedActivations()
    {
        // Gumbel-Softmax draws fresh Gumbel noise on every forward, so no finite difference of the op (first or
        // second order) has a deterministic reference. Its designed gradient (the temperature-1 softmax relaxation)
        // is recorded under create-graph all the same; FusedActivationRecordedParityTests pins it.
        const string gumbel = "Gumbel-Softmax epilogue: stochastic forward (fresh noise per call)";
        foreach (var op in new[] { "FusedLinear", "FusedConv2D", "FusedConv3D", "FusedConvTranspose2D", "FusedConv2DBatchNorm", "FusedBatchNorm" })
            NotSecondOrderCheckableVariants[$"{op}|activation=GumbelSoftmax"] = gumbel;
    }
}
