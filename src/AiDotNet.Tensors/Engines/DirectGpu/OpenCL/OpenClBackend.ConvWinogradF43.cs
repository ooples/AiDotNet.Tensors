// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1432: IWinogradF43Transforms on OpenCL's own F(4x4, 3x3) kernels (WinogradF43Kernels, #1431), which have the
    // shared pipeline's layouts and epilogue. OpenCL's forward route (TryConv2DForwardWinogradF43) keeps its own driver,
    // which caches the transformed filters.
    public sealed partial class OpenClBackend : IWinogradF43Transforms
    {
        void IWinogradF43Transforms.FilterF43(IGpuBuffer w, IGpuBuffer u, int k, int c)
        {
            var filter = WinogradKernel(Kernels.WinogradF43Kernels.Filter, w, u);
            filter.SetArg(2, k); filter.SetArg(3, c);
            RunWinograd(filter, (long)k * c);
        }

        void IWinogradF43Transforms.InputF43(IGpuBuffer x, IGpuBuffer v, int n, int c, int h, int w, int pad, int tilesH,
            int tilesW)
        {
            var input = WinogradKernel(Kernels.WinogradF43Kernels.Input, x, v);
            uint arg = 2;
            input.SetArg(arg++, n); input.SetArg(arg++, c); input.SetArg(arg++, h); input.SetArg(arg++, w);
            input.SetArg(arg++, pad); input.SetArg(arg++, tilesH); input.SetArg(arg++, tilesW);
            RunWinograd(input, (long)c * n * tilesH * tilesW);
        }

        void IWinogradF43Transforms.OutputF43(IGpuBuffer m, IGpuBuffer y, int n, int k, int outH, int outW, int tilesH,
            int tilesW, IGpuBuffer? bias, int epilogue)
        {
            var output = WinogradKernel(Kernels.WinogradF43Kernels.Output, m, y);
            uint arg = 2;
            output.SetArg(arg++, n); output.SetArg(arg++, k); output.SetArg(arg++, outH); output.SetArg(arg++, outW);
            output.SetArg(arg++, tilesH); output.SetArg(arg++, tilesW);
            output.SetArg(arg++, BufferHandle(bias ?? y)); output.SetArg(arg++, epilogue);
            RunWinograd(output, (long)k * n * tilesH * tilesW);
        }
    }
}
