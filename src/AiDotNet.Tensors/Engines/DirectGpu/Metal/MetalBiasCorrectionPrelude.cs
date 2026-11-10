// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// The MSL header every Metal optimizer library starts with: the standard includes and <c>mtl_bias_correction</c>, Adam's
/// 1 - beta^t without cancellation. One copy, so the resident and compressed-moment libraries cannot drift apart.
/// </summary>
/// <remarks>
/// The same series as <see cref="Vulkan.VulkanBiasCorrection"/>, which documents the error bounds: log(beta) for beta in
/// [0.5, 1] as 2 atanh(-d / (2 - d)) with the exact d = 1 - beta, and -expm1(x) as a Taylor sum on [-0.35, 0]. Neither
/// half calls a library log near 1, so the result does not depend on how accurate the device's log is there (Kahan's
/// expm1 = (e - 1) x / log(e) does). Checked against a double reference for beta up to 0.9999 by
/// CompressedMomentGpuOptimizerTests.BiasCorrectionNearBetaOne_MatchesDoubleReference_OnMetalVulkanAndCuda.
/// </remarks>
internal static class MetalBiasCorrectionPrelude
{
    internal const string Source = @"
#include <metal_stdlib>
using namespace metal;
static inline float mtl_bias_correction(float beta, float t) {
    float l;
    if (beta >= 0.5f) {
        float d = 1.0f - beta;
        float s = -d / (2.0f - d);
        float s2 = s * s;
        l = 2.0f * s * (1.0f + s2 * (1.0f / 3.0f + s2 * (1.0f / 5.0f + s2 * (1.0f / 7.0f + s2 * (1.0f / 9.0f
            + s2 * (1.0f / 11.0f + s2 * (1.0f / 13.0f + s2 * (1.0f / 15.0f + s2 * (1.0f / 17.0f)))))))));
    } else {
        l = precise::log(beta);
    }
    float x = t * l;
    if (x < -0.35f) return 1.0f - precise::exp(x);
    return -(x * (1.0f + x * (1.0f / 2.0f + x * (1.0f / 6.0f + x * (1.0f / 24.0f + x * (1.0f / 120.0f + x * (1.0f / 720.0f
        + x * (1.0f / 5040.0f + x * (1.0f / 40320.0f + x * (1.0f / 362880.0f))))))))));
}
";
}
