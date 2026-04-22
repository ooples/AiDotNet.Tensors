// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IAudioBackend
{
    private const string AudioLibName = "Audio";
    private MetalPipelineState GetAudioPipeline(string n)
    {
        if (_audioLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal Audio library was not compiled.");
        return GetPipeline(AudioLibName, _audioLibrary, n);
    }

    public void AmplitudeToDB(IGpuBuffer input, IGpuBuffer output, int length,
        float minAmplitude, float topDb, bool clipTopDb)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetAudioPipeline("audio_amplitude_to_db");
        var (tgr, tpg) = pipe.Calculate1DDispatch(length);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(oBuf, 1);
        enc.SetBytes(length, 2); enc.SetBytes(minAmplitude, 3);
        enc.SetBytes(topDb, 4); enc.SetBytes(clipTopDb ? 1 : 0, 5);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void MuLawEncoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetAudioPipeline("audio_mulaw_encoding");
        var (tgr, tpg) = pipe.Calculate1DDispatch(length);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(oBuf, 1);
        enc.SetBytes(length, 2); enc.SetBytes(qc, 3);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void MuLawDecoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetAudioPipeline("audio_mulaw_decoding");
        var (tgr, tpg) = pipe.Calculate1DDispatch(length);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(oBuf, 1);
        enc.SetBytes(length, 2); enc.SetBytes(qc, 3);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void ComputeDeltas(IGpuBuffer input, IGpuBuffer output,
        int leading, int timeAxis, int winLength)
    {
        int total = leading * timeAxis;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetAudioPipeline("audio_compute_deltas");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(oBuf, 1);
        enc.SetBytes(leading, 2); enc.SetBytes(timeAxis, 3); enc.SetBytes(winLength, 4);
        enc.DispatchThreadgroups(tgr, tpg);
    }

    public void Resample(IGpuBuffer input, IGpuBuffer output,
        int leading, int inLen, int outLen, int up, int down, int halfWidth)
    {
        int total = leading * outLen;
        if (total <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inBuf || output is not MetalGpuBuffer oBuf)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipe = GetAudioPipeline("audio_resample");
        var (tgr, tpg) = pipe.Calculate1DDispatch(total);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        enc.SetBuffer(inBuf, 0); enc.SetBuffer(oBuf, 1);
        enc.SetBytes(leading, 2); enc.SetBytes(inLen, 3); enc.SetBytes(outLen, 4);
        enc.SetBytes(up, 5); enc.SetBytes(down, 6); enc.SetBytes(halfWidth, 7);
        enc.DispatchThreadgroups(tgr, tpg);
    }
}
