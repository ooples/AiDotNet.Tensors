namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static partial class VulkanBatchExecutionKernels
{
    internal static class GroupedConv2DShaders
    {
        internal static readonly string Forward = Shader(new[] { "r:first", "r:second", "w:resultValues" },
            GroupedConv2DShaderBodies.ScalarDeclarations(false), GroupedConv2DShaderBodies.Forward.Replace("0.0f", "0.0"));
        internal static readonly string InputGrad = Shader(new[] { "r:first", "r:second", "rw:resultValues" },
            GroupedConv2DShaderBodies.ScalarDeclarations(true), GroupedConv2DShaderBodies.InputGrad.Replace("0.0f", "0.0"));
        internal static readonly string KernelGrad = Shader(new[] { "r:first", "r:second", "rw:resultValues" },
            GroupedConv2DShaderBodies.ScalarDeclarations(true), GroupedConv2DShaderBodies.KernelGrad.Replace("0.0f", "0.0"));
    }
}

public sealed partial class VulkanBackend : IGroupedConv2DBackend
{
    private static uint[] GroupedConvPushConstants(in GroupedConv2DGeometry g, int? accumulate)
    {
        var values = new uint[accumulate is null ? 16 : 17];
        int[] integers = { g.Batch, g.InChannels, g.InHeight, g.InWidth, g.OutChannels, g.OutHeight, g.OutWidth,
            g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW, g.DilationH, g.DilationW, g.Groups };
        for (int i = 0; i < integers.Length; ++i) values[i] = Bits(integers[i]);
        if (accumulate is int add) values[16] = Bits(add);
        return values;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        in GroupedConv2DGeometry g)
    {
        if (!g.Accepts(input, g.InputCount, kernel, g.KernelCount, output, g.OutputCount)) return false;
        GlslNaryOp(VulkanBatchExecutionKernels.GroupedConv2DShaders.Forward, new[] { input, kernel, output },
            g.OutputCount, GroupedConvPushConstants(g, null));
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        if (!g.Accepts(gradOutput, g.OutputCount, kernel, g.KernelCount, gradInput, g.InputCount)) return false;
        GlslNaryOp(VulkanBatchExecutionKernels.GroupedConv2DShaders.InputGrad, new[] { gradOutput, kernel, gradInput },
            g.InputCount, GroupedConvPushConstants(g, accumulate ? 1 : 0));
        return true;
    }

    bool IGroupedConv2DBackend.TryGroupedConv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        if (!g.Accepts(input, g.InputCount, gradOutput, g.OutputCount, gradKernel, g.KernelCount)) return false;
        GlslNaryOp(VulkanBatchExecutionKernels.GroupedConv2DShaders.KernelGrad, new[] { input, gradOutput, gradKernel },
            g.KernelCount, GroupedConvPushConstants(g, accumulate ? 1 : 0));
        return true;
    }
}
