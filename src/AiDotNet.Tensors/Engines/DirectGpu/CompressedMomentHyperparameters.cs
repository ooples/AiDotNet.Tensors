using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static class CompressedMomentHyperparameters
{
    internal static void Validate(float lr, float beta1, float beta2, float epsilon, float weightDecay,
        string learningRateParameter = "lr")
    {
        if (float.IsNaN(lr) || float.IsInfinity(lr) || lr < 0f)
            throw new ArgumentOutOfRangeException(learningRateParameter, lr, "Learning rate must be nonnegative and finite.");
        if (float.IsNaN(beta1) || float.IsInfinity(beta1) || beta1 < 0f || beta1 >= 1f)
            throw new ArgumentOutOfRangeException(nameof(beta1), beta1, "Beta1 must be finite and in [0, 1).");
        if (float.IsNaN(beta2) || float.IsInfinity(beta2) || beta2 < 0f || beta2 >= 1f)
            throw new ArgumentOutOfRangeException(nameof(beta2), beta2, "Beta2 must be finite and in [0, 1).");
        if (float.IsNaN(epsilon) || float.IsInfinity(epsilon) || epsilon < 0f)
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be nonnegative and finite.");
        if (float.IsNaN(weightDecay) || float.IsInfinity(weightDecay) || weightDecay < 0f)
            throw new ArgumentOutOfRangeException(nameof(weightDecay), weightDecay, "Weight decay must be nonnegative and finite.");
    }

    internal static void ValidateInt8Factors(float oneMinusBeta1, float oneMinusBeta2, float biasCorrection1, float biasCorrection2)
    {
        RequireUnitFactor(oneMinusBeta1, nameof(oneMinusBeta1), allowZero: true);
        RequireUnitFactor(oneMinusBeta2, nameof(oneMinusBeta2), allowZero: true);
        RequireUnitFactor(biasCorrection1, nameof(biasCorrection1), allowZero: false);
        RequireUnitFactor(biasCorrection2, nameof(biasCorrection2), allowZero: false);
    }

    private static void RequireUnitFactor(float value, string name, bool allowZero)
    {
        if (float.IsNaN(value) || float.IsInfinity(value) || value > 1f || (allowZero ? value < 0f : value <= 0f))
            throw new ArgumentOutOfRangeException(name, value, "Adam factors must be finite and in their unit interval.");
    }
}
