using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #775: HIP implementations of the per-family extended-conv capability interfaces. Added incrementally
// (interface + kernels) so a partially-ported backend opts into exactly the families it can run; the
// engine routes the rest to the CPU. Kernels live in the existing compiled modules (trilinear in the
// convolution module); this partial only wires the launch.
public sealed partial class HipBackend : ITrilinearInterpolationKernels
{
    public unsafe void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: trilinear_interpolate");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr g = grid.Handle, pos = positions.Handle, o = output.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &g; args[1] = &pos; args[2] = &o; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        if (!_kernelCache.TryGetValue("trilinear_interpolate_backward", out var kernel))
            throw new InvalidOperationException("HIP kernel not found: trilinear_interpolate_backward");
        uint gridDim = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr go = gradOutput.Handle, pos = positions.Handle, gg = gradGrid.Handle;
        int dd = d, hh = h, ww = w, cc = c, pp = p;
        float eps = upperEps;
        void** args = stackalloc void*[9];
        args[0] = &go; args[1] = &pos; args[2] = &gg; args[3] = &dd; args[4] = &hh;
        args[5] = &ww; args[6] = &cc; args[7] = &pp; args[8] = &eps;
        LaunchKernel(kernel, gridDim, DefaultBlockSize, args);
        Synchronize();
    }
}
