// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// <c>vk_bias_correction</c>, Adam's 1 - beta^t without cancellation, shared by every Vulkan optimizer shader.
/// </summary>
/// <remarks>
/// <para>pow's rounding, subtracted from 1, grows by 1/(1 - beta^t), so the naive form loses about 3e-4 relative at
/// beta = 0.9999, t = 2 (see CudaOptimizerKernels.bias_correction, which uses log1pf and expm1f).</para>
/// <para>GLSL has neither, and its <c>log</c> is only bounded by an absolute error of 2^-21 on [0.5, 2], so log(beta)
/// near 1 (and Kahan's expm1 = (e - 1) x / log(e), which divides by log(e) near 1) loses relative accuracy: on an AMD
/// driver that form missed by 2.5e-5 at beta = 0.9999. Both halves are therefore series that use no library log:</para>
/// <list type="bullet">
/// <item>log(beta) for beta in [0.5, 1]: 1 - beta = d is exact there, and log(1 - d) = 2 atanh(s) with s = -d / (2 - d),
/// |s| &lt;= 1/3, summed to s^17 (the next term is below 2e-10 relative). Below 0.5, log(beta) is at least 0.69 in
/// magnitude, so the library log's absolute bound is a small relative one.</item>
/// <item>-expm1(x) for x = t log(beta) in [-0.35, 0]: Taylor to x^9 (the next term is below 3e-11 relative). Below
/// -0.35, exp(x) &lt; 0.71, so 1 - exp(x) cancels by at most 3.4x and is used directly.</item>
/// </list>
/// <para><c>precise</c> keeps the compiler from contracting or reassociating the sums.</para>
/// <para>beta = 0 (a valid Adam setting) returns 1, which is 1 - 0^t for every step t &gt;= 1: GLSL leaves log(0)
/// undefined, where the earlier <c>1.0 - pow(0.0, t)</c> was defined.</para>
/// </remarks>
internal static class VulkanBiasCorrection
{
    internal const string Glsl = @"
float vk_bias_correction(float beta, float t){
    if (beta <= 0.0) return 1.0;
    precise float l;
    if (beta >= 0.5) {
        precise float d = 1.0 - beta;
        precise float s = -d / (2.0 - d);
        precise float s2 = s * s;
        l = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0 + s2 * (1.0 / 9.0 + s2 * (1.0 / 11.0
            + s2 * (1.0 / 13.0 + s2 * (1.0 / 15.0 + s2 * (1.0 / 17.0)))))))));
    } else {
        l = log(beta);
    }
    precise float x = t * l;
    if (x < -0.35) return 1.0 - exp(x);
    return -(x * (1.0 + x * (1.0 / 2.0 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x * (1.0 / 720.0
        + x * (1.0 / 5040.0 + x * (1.0 / 40320.0 + x * (1.0 / 362880.0))))))))));
}
";
}
