// Copyright (c) AiDotNet. All rights reserved.
// CUDA launcher shims for the RoI family (Issue #217 tail).
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IRoiBackend
{
    private IntPtr ResolveRoiKernel(string name)
    {
        if (_roiModule == IntPtr.Zero)
            throw new InvalidOperationException("RoI CUDA module was not compiled.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    public unsafe void RoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW,
        float spatialScale, int samplingRatio, bool aligned)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        var kernel = ResolveRoiKernel("roi_align");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, bP = boxes.Handle, oP = output.Handle;
        int nn = N, cc = C, hh = H, ww = W, kk = K, oh = outH, ow = outW;
        int sr = samplingRatio, al = aligned ? 1 : 0;
        float ss = spatialScale;
        void** args = stackalloc void*[13];
        args[0] = &inP; args[1] = &bP; args[2] = &oP;
        args[3] = &nn; args[4] = &cc; args[5] = &hh; args[6] = &ww;
        args[7] = &kk; args[8] = &oh; args[9] = &ow;
        args[10] = &ss; args[11] = &sr; args[12] = &al;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void RoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, float spatialScale)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        var kernel = ResolveRoiKernel("roi_pool");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, bP = boxes.Handle, oP = output.Handle;
        int nn = N, cc = C, hh = H, ww = W, kk = K, oh = outH, ow = outW;
        float ss = spatialScale;
        void** args = stackalloc void*[11];
        args[0] = &inP; args[1] = &bP; args[2] = &oP;
        args[3] = &nn; args[4] = &cc; args[5] = &hh; args[6] = &ww;
        args[7] = &kk; args[8] = &oh; args[9] = &ow; args[10] = &ss;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PsRoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        var kernel = ResolveRoiKernel("ps_roi_align");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, bP = boxes.Handle, oP = output.Handle;
        int nn = N, cc = C, hh = H, ww = W, kk = K, oh = outH, ow = outW, oc = outputChannels;
        int sr = samplingRatio;
        float ss = spatialScale;
        void** args = stackalloc void*[13];
        args[0] = &inP; args[1] = &bP; args[2] = &oP;
        args[3] = &nn; args[4] = &cc; args[5] = &hh; args[6] = &ww;
        args[7] = &kk; args[8] = &oh; args[9] = &ow; args[10] = &oc;
        args[11] = &ss; args[12] = &sr;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void PsRoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        var kernel = ResolveRoiKernel("ps_roi_pool");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, bP = boxes.Handle, oP = output.Handle;
        int nn = N, cc = C, hh = H, ww = W, kk = K, oh = outH, ow = outW, oc = outputChannels;
        float ss = spatialScale;
        void** args = stackalloc void*[12];
        args[0] = &inP; args[1] = &bP; args[2] = &oP;
        args[3] = &nn; args[4] = &cc; args[5] = &hh; args[6] = &ww;
        args[7] = &kk; args[8] = &oh; args[9] = &ow; args[10] = &oc; args[11] = &ss;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
}
