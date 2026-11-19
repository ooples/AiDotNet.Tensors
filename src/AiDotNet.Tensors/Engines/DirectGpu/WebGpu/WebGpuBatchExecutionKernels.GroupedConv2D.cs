#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    internal static class GroupedConv2DShaders
    {
        internal static readonly string Forward = GroupedConv2DShaderBodies.Wgsl(GroupedConv2DShaderBodies.Forward, false);
        internal static readonly string InputGrad = GroupedConv2DShaderBodies.Wgsl(GroupedConv2DShaderBodies.InputGrad, true);
        internal static readonly string KernelGrad = GroupedConv2DShaderBodies.Wgsl(GroupedConv2DShaderBodies.KernelGrad, true);
    }
}
#endif
