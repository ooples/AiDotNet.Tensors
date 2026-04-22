// Copyright (c) AiDotNet. All rights reserved.
// Vulkan launcher shims for the geometry / sampling kernels (Issue #217).

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IGeometryBackend
{
    // Push-constant sizes (bytes):
    //  Interpolate2D = 8 ints = 32
    //  Pad4D         = 13 ints + 1 float = 56
    //  GridSample2D  = 9 ints = 36
    //  AffineGrid3D  = 5 ints = 20
    private const uint GeometryPushInterp = 32u;
    private const uint GeometryPushPad    = 56u;
    private const uint GeometryPushGrid   = 36u;
    private const uint GeometryPushAff    = 20u;

    public void Interpolate2D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int Hout, int Wout,
        int mode, bool alignCorners)
    {
        int total = N * C * Hout * Wout;
        if (total <= 0) return;
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)Hin, (uint)Win,
            (uint)Hout, (uint)Wout, (uint)mode, (uint)(alignCorners ? 1 : 0)
        };
        GlslUnaryOp(VulkanGeometryKernels.Interpolate2D, input, output, total, pc, GeometryPushInterp);
    }

    public void Pad4D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win,
        int padN0, int padN1, int padC0, int padC1,
        int padH0, int padH1, int padW0, int padW1,
        int mode, float padValue)
    {
        int total = (N + padN0 + padN1) * (C + padC0 + padC1) * (Hin + padH0 + padH1) * (Win + padW0 + padW1);
        if (total <= 0) return;
        // Pack 13 ints + 1 float = 56 bytes. Layout matches the shader's
        // P block member order; re-interpret the float as raw bits.
        uint packedVal = BitConverter.ToUInt32(BitConverter.GetBytes(padValue), 0);
        var pc = new uint[] {
            (uint)N, (uint)C, (uint)Hin, (uint)Win,
            (uint)padN0, (uint)padN1, (uint)padC0, (uint)padC1,
            (uint)padH0, (uint)padH1, (uint)padW0, (uint)padW1,
            (uint)mode, packedVal
        };
        GlslUnaryOp(VulkanGeometryKernels.Pad4D, input, output, total, pc, GeometryPushPad);
    }

    public void GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int N, int H, int W, int C, int outH, int outW,
        int mode, int padding, bool alignCorners)
    {
        int total = N * outH * outW;
        if (total <= 0) return;
        var pc = new uint[] {
            (uint)N, (uint)H, (uint)W, (uint)C,
            (uint)outH, (uint)outW, (uint)mode, (uint)padding, (uint)(alignCorners ? 1 : 0)
        };
        GlslBinaryOp(VulkanGeometryKernels.GridSample2D, input, grid, output, total, pc, GeometryPushGrid);
    }

    public void AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
        int N, int D, int H, int W, bool alignCorners)
    {
        int total = N * D * H * W;
        if (total <= 0) return;
        var pc = new uint[] {
            (uint)N, (uint)D, (uint)H, (uint)W, (uint)(alignCorners ? 1 : 0)
        };
        GlslUnaryOp(VulkanGeometryKernels.AffineGrid3D, theta, grid, total, pc, GeometryPushAff);
    }
}
