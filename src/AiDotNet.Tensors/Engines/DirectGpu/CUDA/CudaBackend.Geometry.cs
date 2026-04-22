// Copyright (c) AiDotNet. All rights reserved.
// CUDA launcher shims for the geometry / sampling kernels (Issue #217).

using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGeometryBackend
{
    private IntPtr ResolveGeometryKernel(string name)
    {
        if (_geometryModule == IntPtr.Zero)
            throw new InvalidOperationException(
                "Geometry CUDA module was not compiled. Falling back to CPU reference.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    public unsafe void Interpolate2D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int Hout, int Wout,
        int mode, bool alignCorners)
    {
        int total = N * C * Hout * Wout;
        if (total <= 0) return;
        var kernel = ResolveGeometryKernel("geometry_interpolate_2d");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, outP = output.Handle;
        int ac = alignCorners ? 1 : 0;
        int nn = N, cc = C, hi = Hin, wi = Win, ho = Hout, wo = Wout, mm = mode;
        void** args = stackalloc void*[10];
        args[0] = &inP; args[1] = &outP;
        args[2] = &nn; args[3] = &cc;
        args[4] = &hi; args[5] = &wi;
        args[6] = &ho; args[7] = &wo;
        args[8] = &mm; args[9] = &ac;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Pad4D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win,
        int padN0, int padN1, int padC0, int padC1,
        int padH0, int padH1, int padW0, int padW1,
        int mode, float padValue)
    {
        int Nout = N + padN0 + padN1;
        int Cout = C + padC0 + padC1;
        int Hout = Hin + padH0 + padH1;
        int Wout = Win + padW0 + padW1;
        int total = Nout * Cout * Hout * Wout;
        if (total <= 0) return;
        var kernel = ResolveGeometryKernel("geometry_pad_4d");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, outP = output.Handle;
        int nn = N, cc = C, hi = Hin, wi = Win;
        int pn0 = padN0, pn1 = padN1, pc0 = padC0, pc1 = padC1;
        int ph0 = padH0, ph1 = padH1, pw0 = padW0, pw1 = padW1;
        int mm = mode; float pv = padValue;
        void** args = stackalloc void*[16];
        args[0] = &inP; args[1] = &outP;
        args[2] = &nn; args[3] = &cc; args[4] = &hi; args[5] = &wi;
        args[6] = &pn0; args[7] = &pn1; args[8] = &pc0; args[9] = &pc1;
        args[10] = &ph0; args[11] = &ph1; args[12] = &pw0; args[13] = &pw1;
        args[14] = &mm; args[15] = &pv;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int N, int H, int W, int C, int outH, int outW,
        int mode, int padding, bool alignCorners)
    {
        int total = N * outH * outW;
        if (total <= 0) return;
        var kernel = ResolveGeometryKernel("geometry_grid_sample_2d");
        using var _ = PushContext();
        uint gridLaunch = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, grP = grid.Handle, outP = output.Handle;
        int nn = N, hh = H, ww = W, cc = C, oh = outH, ow = outW, mm = mode, pp = padding;
        int ac = alignCorners ? 1 : 0;
        void** args = stackalloc void*[12];
        args[0] = &inP; args[1] = &grP; args[2] = &outP;
        args[3] = &nn; args[4] = &hh; args[5] = &ww; args[6] = &cc;
        args[7] = &oh; args[8] = &ow; args[9] = &mm; args[10] = &pp; args[11] = &ac;
        LaunchKernel(kernel, gridLaunch, DefaultBlockSize, args);
    }

    public unsafe void AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
        int N, int D, int H, int W, bool alignCorners)
    {
        int total = N * D * H * W;
        if (total <= 0) return;
        var kernel = ResolveGeometryKernel("geometry_affine_grid_3d");
        using var _ = PushContext();
        uint gridLaunch = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr tP = theta.Handle, gP = grid.Handle;
        int nn = N, dd = D, hh = H, ww = W;
        int ac = alignCorners ? 1 : 0;
        void** args = stackalloc void*[7];
        args[0] = &tP; args[1] = &gP;
        args[2] = &nn; args[3] = &dd; args[4] = &hh; args[5] = &ww; args[6] = &ac;
        LaunchKernel(kernel, gridLaunch, DefaultBlockSize, args);
    }
}
