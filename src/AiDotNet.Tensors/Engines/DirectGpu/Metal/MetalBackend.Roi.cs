// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IRoiBackend
{
    private const string RoiLibName = "Roi";
    private MetalPipelineState GetRoiPipeline(string n)
    {
        if (_roiLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal RoI library was not compiled.");
        return GetPipeline(RoiLibName, _roiLibrary, n);
    }

    public void RoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW,
        float spatialScale, int samplingRatio, bool aligned)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || boxes is not MetalGpuBuffer bBuf
            || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetRoiPipeline("roi_align");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0);
        enc.SetBuffer(bBuf, 1);
        enc.SetBuffer(oBuf, 2);
        enc.SetBytes(N, 3); enc.SetBytes(C, 4); enc.SetBytes(H, 5); enc.SetBytes(W, 6);
        enc.SetBytes(K, 7); enc.SetBytes(outH, 8); enc.SetBytes(outW, 9);
        enc.SetBytes(spatialScale, 10); enc.SetBytes(samplingRatio, 11);
        enc.SetBytes(aligned ? 1 : 0, 12);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void RoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, float spatialScale)
    {
        int total = K * C * outH * outW;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || boxes is not MetalGpuBuffer bBuf
            || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetRoiPipeline("roi_pool");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0);
        enc.SetBuffer(bBuf, 1);
        enc.SetBuffer(oBuf, 2);
        enc.SetBytes(N, 3); enc.SetBytes(C, 4); enc.SetBytes(H, 5); enc.SetBytes(W, 6);
        enc.SetBytes(K, 7); enc.SetBytes(outH, 8); enc.SetBytes(outW, 9);
        enc.SetBytes(spatialScale, 10);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void PsRoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || boxes is not MetalGpuBuffer bBuf
            || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetRoiPipeline("ps_roi_align");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(bBuf, 1); enc.SetBuffer(oBuf, 2);
        enc.SetBytes(N, 3); enc.SetBytes(C, 4); enc.SetBytes(H, 5); enc.SetBytes(W, 6);
        enc.SetBytes(K, 7); enc.SetBytes(outH, 8); enc.SetBytes(outW, 9);
        enc.SetBytes(outputChannels, 10);
        enc.SetBytes(spatialScale, 11); enc.SetBytes(samplingRatio, 12);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void PsRoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
        int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
        float spatialScale)
    {
        int total = K * outputChannels * outH * outW;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || boxes is not MetalGpuBuffer bBuf
            || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetRoiPipeline("ps_roi_pool");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(bBuf, 1); enc.SetBuffer(oBuf, 2);
        enc.SetBytes(N, 3); enc.SetBytes(C, 4); enc.SetBytes(H, 5); enc.SetBytes(W, 6);
        enc.SetBytes(K, 7); enc.SetBytes(outH, 8); enc.SetBytes(outW, 9);
        enc.SetBytes(outputChannels, 10); enc.SetBytes(spatialScale, 11);
        enc.DispatchThreadgroups(tgr, tpg);
    }
}
