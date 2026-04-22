// Copyright (c) AiDotNet. All rights reserved.
// Metal launcher shims for the geometry / sampling kernels (Issue #217).

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IGeometryBackend
{
    private const string GeometryLibName = "Geometry";

    private MetalPipelineState GetGeometryPipeline(string kernelName)
    {
        if (_geometryLibrary == IntPtr.Zero)
            throw new InvalidOperationException(
                "Metal Geometry library was not compiled. Falling back to CPU reference.");
        return GetPipeline(GeometryLibName, _geometryLibrary, kernelName);
    }

    public void Interpolate2D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win, int Hout, int Wout,
        int mode, bool alignCorners)
    {
        int total = N * C * Hout * Wout;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetGeometryPipeline("geometry_interpolate_2d");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipe.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(N, 2);
        encoder.SetBytes(C, 3);
        encoder.SetBytes(Hin, 4);
        encoder.SetBytes(Win, 5);
        encoder.SetBytes(Hout, 6);
        encoder.SetBytes(Wout, 7);
        encoder.SetBytes(mode, 8);
        encoder.SetBytes(alignCorners ? 1 : 0, 9);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void Pad4D(IGpuBuffer input, IGpuBuffer output,
        int N, int C, int Hin, int Win,
        int padN0, int padN1, int padC0, int padC1,
        int padH0, int padH1, int padW0, int padW1,
        int mode, float padValue)
    {
        int total = (N + padN0 + padN1) * (C + padC0 + padC1) * (Hin + padH0 + padH1) * (Win + padW0 + padW1);
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetGeometryPipeline("geometry_pad_4d");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipe.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(outBuf, 1);
        encoder.SetBytes(N, 2); encoder.SetBytes(C, 3);
        encoder.SetBytes(Hin, 4); encoder.SetBytes(Win, 5);
        encoder.SetBytes(padN0, 6); encoder.SetBytes(padN1, 7);
        encoder.SetBytes(padC0, 8); encoder.SetBytes(padC1, 9);
        encoder.SetBytes(padH0, 10); encoder.SetBytes(padH1, 11);
        encoder.SetBytes(padW0, 12); encoder.SetBytes(padW1, 13);
        encoder.SetBytes(mode, 14);
        encoder.SetBytes(padValue, 15);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int N, int H, int W, int C, int outH, int outW,
        int mode, int padding, bool alignCorners)
    {
        int total = N * outH * outW;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || grid is not MetalGpuBuffer grBuf
            || output is not MetalGpuBuffer outBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetGeometryPipeline("geometry_grid_sample_2d");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipe.Handle);
        encoder.SetBuffer(inBuf, 0);
        encoder.SetBuffer(grBuf, 1);
        encoder.SetBuffer(outBuf, 2);
        encoder.SetBytes(N, 3); encoder.SetBytes(H, 4); encoder.SetBytes(W, 5); encoder.SetBytes(C, 6);
        encoder.SetBytes(outH, 7); encoder.SetBytes(outW, 8);
        encoder.SetBytes(mode, 9); encoder.SetBytes(padding, 10);
        encoder.SetBytes(alignCorners ? 1 : 0, 11);
        encoder.DispatchThreadgroups(tgr, tpg);
    }

    public void AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
        int N, int D, int H, int W, bool alignCorners)
    {
        int total = N * D * H * W;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (theta is not MetalGpuBuffer tBuf || grid is not MetalGpuBuffer gBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetGeometryPipeline("geometry_affine_grid_3d");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipe.Handle);
        encoder.SetBuffer(tBuf, 0);
        encoder.SetBuffer(gBuf, 1);
        encoder.SetBytes(N, 2); encoder.SetBytes(D, 3); encoder.SetBytes(H, 4); encoder.SetBytes(W, 5);
        encoder.SetBytes(alignCorners ? 1 : 0, 6);
        encoder.DispatchThreadgroups(tgr, tpg);
    }
}
