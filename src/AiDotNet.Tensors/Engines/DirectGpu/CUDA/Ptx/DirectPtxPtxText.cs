using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Canonical text encodings shared by direct PTX emitters.</summary>
internal static class DirectPtxPtxText
{
    internal static void AppendModuleHeader(
        StringBuilder text,
        int computeCapabilityMajor,
        int computeCapabilityMinor,
        bool disableLoopUnrolling = false)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (computeCapabilityMajor <= 0) throw new ArgumentOutOfRangeException(nameof(computeCapabilityMajor));
        if (computeCapabilityMinor < 0) throw new ArgumentOutOfRangeException(nameof(computeCapabilityMinor));

        text.AppendLine(".version 7.1");
        text.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        text.AppendLine(".address_size 64");
        if (disableLoopUnrolling)
            text.AppendLine(".pragma \"nounroll\";");
    }

    internal static string Hex(float value)
        => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");
}
