// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// <c>wgsl_bias_correction</c>, Adam's 1 - beta^t without cancellation, appended to every WGSL optimizer module that
/// uses it (WGSL resolves module-scope functions regardless of order).
/// </summary>
/// <remarks>
/// The same series as <see cref="Vulkan.VulkanBiasCorrection"/>, which documents the error bounds. WGSL's <c>log</c> has the
/// same 2^-21 absolute bound on [0.5, 2] as GLSL's, and the naive <c>1.0 - pow(beta, t)</c> loses ~3e-4 relative at
/// beta = 0.9999, t = 2.
/// </remarks>
internal static class WebGpuBiasCorrection
{
    internal const string Wgsl = @"
fn wgsl_bias_correction(beta: f32, t: f32) -> f32 {
    var l: f32;
    if (beta >= 0.5) {
        let d = 1.0 - beta;
        let s = -d / (2.0 - d);
        let s2 = s * s;
        l = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0 + s2 * (1.0 / 11.0
            + s2 * (1.0 / 13.0 + s2 * (1.0 / 15.0 + s2 * (1.0 / 17.0)))))))));
    } else {
        l = log(beta);
    }
    let x = t * l;
    if (x < -0.35) {
        return 1.0 - exp(x);
    }
    return -(x * (1.0 + x * (1.0 / 2.0 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x * (1.0 / 720.0
        + x * (1.0 / 5040.0 + x * (1.0 / 40320.0 + x * (1.0 / 362880.0))))))))));
}
";
}
