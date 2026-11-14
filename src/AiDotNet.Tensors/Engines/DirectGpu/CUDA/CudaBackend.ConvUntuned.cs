using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The CUDA convolution slots' defaults for shapes the process has no tuning budget left to measure (#2561). Without
/// them every such shape ran the slot's reference: on a cold cache DenseNet-121's 128->32 3x3 forward took 3.2 ms per
/// call at batch 64 on an RTX 1000 Ada against 0.55 ms tuned, and only the first few shapes fit the 2 s budget. The
/// lists follow what online tuning chose on that device across 417 ResNet / DenseNet / MobileNet / VGG shapes:
/// Winograd F(2x2,3x3) for undilated stride-1 3x3 (forward, input and weight gradient), the 8x8-micro-tile weight
/// gradient elsewhere, and the sub-pixel input-gradient kernels for stride 2. A default is used only after one run
/// of it matches the reference within the harness tolerance; shapes the lists do not name keep the reference.
/// Shape layout (TunedConv2DArgs.Shape): N, C, H, W, K, kH, kW, sH, sW, pH, pW, dH, dW, oH, oW.
/// </summary>
internal static class ConvUntuned
{
    private static readonly string[] s_fwd3x3 = { "nvrtc.conv2d.winograd_f23_fused", "nvrtc.conv2d.generic" };
    private static readonly string[] s_fwd3x3s2 = { "nvrtc.conv2d.r8_128x64" };
    private static readonly string[] s_dx3x3 = { "nvrtc.conv2d_dx.winograd_f23_fused" };
    private static readonly string[] s_dxS2k3 = { "nvrtc.conv2d_dx.s2k3" };
    private static readonly string[] s_dxS2k1 = { "nvrtc.conv2d_dx.s2k1" };
    private static readonly string[] s_dw3x3 = { "nvrtc.conv2d_dw.winograd_fused", "nvrtc.conv2d_dw.r8_128x64" };
    private static readonly string[] s_dw = { "nvrtc.conv2d_dw.r8_128x64", "nvrtc.conv2d_dw.r8_64x128" };

    private static bool Is3x3(in TunedShape s, int stride) =>
        s.Count >= 15 && s[5] == 3 && s[6] == 3 && s[7] == stride && s[8] == stride && s[11] == 1 && s[12] == 1;

    internal static IReadOnlyList<string>? Forward(TunedShape s) =>
        Is3x3(s, 1) ? s_fwd3x3 : Is3x3(s, 2) ? s_fwd3x3s2 : null;

    internal static IReadOnlyList<string>? Data(TunedShape s)
    {
        if (Is3x3(s, 1)) return s_dx3x3;
        if (Is3x3(s, 2)) return s_dxS2k3;
        if (s.Count >= 15 && s[5] == 1 && s[6] == 1 && s[7] == 2 && s[8] == 2) return s_dxS2k1;
        return null;
    }

    internal static IReadOnlyList<string>? Filter(TunedShape s) => Is3x3(s, 1) ? s_dw3x3 : s_dw;
}
