// Copyright (c) AiDotNet. All rights reserved.
// OpenCL launcher shims for the geometry / sampling kernels (Issue #217).
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IGeometryBackend
    {
        private const int GeometryLocalSize = 256;

        private DirectOpenClKernel GetGeometryKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var kernel))
                throw new InvalidOperationException(
                    $"OpenCL geometry kernel not found: {name}. Module may have failed to compile.");
            return kernel;
        }

        private static int RoundUpToGeometryGroup(int v) =>
            ((v + GeometryLocalSize - 1) / GeometryLocalSize) * GeometryLocalSize;

        private static IntPtr GeometryBufHandle(IGpuBuffer b) =>
            ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        public void Interpolate2D(IGpuBuffer input, IGpuBuffer output,
            int N, int C, int Hin, int Win, int Hout, int Wout,
            int mode, bool alignCorners)
        {
            int total = N * C * Hout * Wout;
            if (total <= 0) return;
            var k = GetGeometryKernel("geometry_interpolate_2d");
            k.SetArg(0, GeometryBufHandle(input));
            k.SetArg(1, GeometryBufHandle(output));
            k.SetArg(2, N); k.SetArg(3, C);
            k.SetArg(4, Hin); k.SetArg(5, Win);
            k.SetArg(6, Hout); k.SetArg(7, Wout);
            k.SetArg(8, mode); k.SetArg(9, alignCorners ? 1 : 0);
            k.Execute1D(RoundUpToGeometryGroup(total), GeometryLocalSize);
        }

        public void Pad4D(IGpuBuffer input, IGpuBuffer output,
            int N, int C, int Hin, int Win,
            int padN0, int padN1, int padC0, int padC1,
            int padH0, int padH1, int padW0, int padW1,
            int mode, float padValue)
        {
            int total = (N + padN0 + padN1) * (C + padC0 + padC1) * (Hin + padH0 + padH1) * (Win + padW0 + padW1);
            if (total <= 0) return;
            var k = GetGeometryKernel("geometry_pad_4d");
            k.SetArg(0, GeometryBufHandle(input));
            k.SetArg(1, GeometryBufHandle(output));
            k.SetArg(2, N); k.SetArg(3, C);
            k.SetArg(4, Hin); k.SetArg(5, Win);
            k.SetArg(6, padN0); k.SetArg(7, padN1);
            k.SetArg(8, padC0); k.SetArg(9, padC1);
            k.SetArg(10, padH0); k.SetArg(11, padH1);
            k.SetArg(12, padW0); k.SetArg(13, padW1);
            k.SetArg(14, mode); k.SetArg(15, padValue);
            k.Execute1D(RoundUpToGeometryGroup(total), GeometryLocalSize);
        }

        public void GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
            int N, int H, int W, int C, int outH, int outW,
            int mode, int padding, bool alignCorners)
        {
            int total = N * outH * outW;
            if (total <= 0) return;
            var k = GetGeometryKernel("geometry_grid_sample_2d");
            k.SetArg(0, GeometryBufHandle(input));
            k.SetArg(1, GeometryBufHandle(grid));
            k.SetArg(2, GeometryBufHandle(output));
            k.SetArg(3, N); k.SetArg(4, H); k.SetArg(5, W); k.SetArg(6, C);
            k.SetArg(7, outH); k.SetArg(8, outW);
            k.SetArg(9, mode); k.SetArg(10, padding); k.SetArg(11, alignCorners ? 1 : 0);
            k.Execute1D(RoundUpToGeometryGroup(total), GeometryLocalSize);
        }

        public void AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
            int N, int D, int H, int W, bool alignCorners)
        {
            int total = N * D * H * W;
            if (total <= 0) return;
            var k = GetGeometryKernel("geometry_affine_grid_3d");
            k.SetArg(0, GeometryBufHandle(theta));
            k.SetArg(1, GeometryBufHandle(grid));
            k.SetArg(2, N); k.SetArg(3, D); k.SetArg(4, H); k.SetArg(5, W);
            k.SetArg(6, alignCorners ? 1 : 0);
            k.Execute1D(RoundUpToGeometryGroup(total), GeometryLocalSize);
        }
    }
}
#endif
