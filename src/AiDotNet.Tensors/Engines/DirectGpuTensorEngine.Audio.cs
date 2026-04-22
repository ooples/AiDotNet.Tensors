// Copyright (c) AiDotNet. All rights reserved.
// GPU dispatch for the audio primitives that have native kernels (the
// composable ops — Spectrogram, PitchShift, TimeStretch — stay on the
// inherited CpuEngine path since they internally call the already
// GPU-accelerated STFT).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> AmplitudeToDB<T>(Tensor<T> input, float minAmplitude = 1e-10f, float? topDb = null)
    {
        if (typeof(T) == typeof(float))
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
                {
                    int len = input.Length;
                    if (len == 0) return new Tensor<T>((int[])input._shape.Clone());
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    var outBuf = AllocateOutputBuffer(backend, len);
                    try
                    {
                        // When topDb is requested, we still need the peak to set
                        // the floor. Rather than reduce on GPU here we route
                        // through CPU for that case (rare fast path).
                        if (!topDb.HasValue)
                        {
                            audio.AmplitudeToDB(inBuf.Buffer, outBuf.Buffer, len,
                                minAmplitude, 0.0f, clipTopDb: false);
                            var arr = FinishGpuOp<T>(backend, outBuf, len);
                            return new Tensor<T>(arr, (int[])input._shape.Clone());
                        }
                    }
                    catch { outBuf.Dispose(); throw; }
                    outBuf.Dispose();
                }
            }
            catch { }
        }
        return base.AmplitudeToDB(input, minAmplitude, topDb);
    }

    /// <inheritdoc/>
    public override Tensor<T> MuLawEncoding<T>(Tensor<T> input, int quantizationChannels = 256)
        => DispatchMuLaw(input, quantizationChannels, encoding: true)
           ?? base.MuLawEncoding(input, quantizationChannels);

    /// <inheritdoc/>
    public override Tensor<T> MuLawDecoding<T>(Tensor<T> input, int quantizationChannels = 256)
        => DispatchMuLaw(input, quantizationChannels, encoding: false)
           ?? base.MuLawDecoding(input, quantizationChannels);

    private Tensor<T>? DispatchMuLaw<T>(Tensor<T> input, int qc, bool encoding)
    {
        if (typeof(T) != typeof(float)) return null;
        try
        {
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int len = input.Length;
                if (len == 0) return new Tensor<T>((int[])input._shape.Clone());
                using var inBuf = GetOrAllocateBuffer(backend, input);
                var outBuf = AllocateOutputBuffer(backend, len);
                try
                {
                    if (encoding) audio.MuLawEncoding(inBuf.Buffer, outBuf.Buffer, len, qc);
                    else audio.MuLawDecoding(inBuf.Buffer, outBuf.Buffer, len, qc);
                    var arr = FinishGpuOp<T>(backend, outBuf, len);
                    return new Tensor<T>(arr, (int[])input._shape.Clone());
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        catch { }
        return null;
    }

    /// <inheritdoc/>
    public override Tensor<T> ComputeDeltas<T>(Tensor<T> input, int winLength = 5)
    {
        if (typeof(T) == typeof(float) && input.Rank >= 1)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
                {
                    int timeAxis = input._shape[input.Rank - 1];
                    int leading = input.Length / timeAxis;
                    using var inBuf = GetOrAllocateBuffer(backend, input);
                    var outBuf = AllocateOutputBuffer(backend, input.Length);
                    try
                    {
                        audio.ComputeDeltas(inBuf.Buffer, outBuf.Buffer, leading, timeAxis, winLength);
                        var arr = FinishGpuOp<T>(backend, outBuf, input.Length);
                        return new Tensor<T>(arr, (int[])input._shape.Clone());
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.ComputeDeltas(input, winLength);
    }

    /// <inheritdoc/>
    public override Tensor<T> Resample<T>(Tensor<T> waveform, int origRate, int newRate)
    {
        if (typeof(T) == typeof(float) && waveform.Rank >= 1 && origRate > 0 && newRate > 0 && origRate != newRate)
        {
            try
            {
                if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
                {
                    int gcd = Gcd(origRate, newRate);
                    int up = newRate / gcd;
                    int down = origRate / gcd;
                    int halfWidth = Math.Max(8, Math.Min(256, up * 8));

                    int inLen = waveform._shape[waveform.Rank - 1];
                    int outLen = (int)((long)inLen * up / down);
                    int leading = waveform.Length / inLen;
                    int outTotal = leading * outLen;
                    if (outTotal == 0)
                    {
                        var emptyShape = (int[])waveform._shape.Clone();
                        emptyShape[waveform.Rank - 1] = outLen;
                        return new Tensor<T>(emptyShape);
                    }

                    using var inBuf = GetOrAllocateBuffer(backend, waveform);
                    var outBuf = AllocateOutputBuffer(backend, outTotal);
                    try
                    {
                        audio.Resample(inBuf.Buffer, outBuf.Buffer, leading, inLen, outLen, up, down, halfWidth);
                        var arr = FinishGpuOp<T>(backend, outBuf, outTotal);
                        var outShape = (int[])waveform._shape.Clone();
                        outShape[waveform.Rank - 1] = outLen;
                        return new Tensor<T>(arr, outShape);
                    }
                    catch { outBuf.Dispose(); throw; }
                }
            }
            catch { }
        }
        return base.Resample(waveform, origRate, newRate);
    }

    private static int Gcd(int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; }
}
