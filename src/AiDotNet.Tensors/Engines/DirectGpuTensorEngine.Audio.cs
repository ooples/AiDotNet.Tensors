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
        if (typeof(T) == typeof(float) && !topDb.HasValue)
        {
            // topDb=null has no reduction dependency so the GPU path is
            // safe. topDb path needs a global peak; we leave it to CPU.
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int len = input.Length;
                if (len == 0) return new Tensor<T>((int[])input._shape.Clone());
                using var inBuf = GetOrAllocateBuffer(backend, input);
                var outBuf = AllocateOutputBuffer(backend, len);
                try
                {
                    audio.AmplitudeToDB(inBuf.Buffer, outBuf.Buffer, len,
                        minAmplitude, 0.0f, clipTopDb: false);
                    var result = DeferTensorResult<T>(
                        backend, outBuf.Buffer, len, (int[])input._shape.Clone());
                    outBuf.RelinquishOwnership();
                    return result;
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.AmplitudeToDB(input, minAmplitude, topDb);
    }

    /// <inheritdoc/>
    public override Tensor<int> MuLawEncoding<T>(Tensor<T> input, int quantizationChannels = 256)
    {
        if (quantizationChannels <= 1)
            throw new ArgumentException("quantizationChannels must be > 1 (μ = qc − 1 must be positive).",
                nameof(quantizationChannels));
        if (typeof(T) == typeof(float))
        {
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int len = input.Length;
                if (len == 0) return new Tensor<int>((int[])input._shape.Clone());
                using var inBuf = GetOrAllocateBuffer(backend, input);
                var outBuf = AllocateOutputBuffer(backend, len);
                try
                {
                    audio.MuLawEncoding(inBuf.Buffer, outBuf.Buffer, len, quantizationChannels);
                    var result = DeferTensorResult<int>(
                        backend, outBuf.Buffer, len, (int[])input._shape.Clone());
                    outBuf.RelinquishOwnership();
                    return result;
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.MuLawEncoding(input, quantizationChannels);
    }

    /// <inheritdoc/>
    public override Tensor<T> MuLawDecoding<T>(Tensor<int> input, int quantizationChannels = 256)
    {
        if (quantizationChannels <= 1)
            throw new ArgumentException("quantizationChannels must be > 1.", nameof(quantizationChannels));
        if (typeof(T) == typeof(float))
        {
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int len = input.Length;
                if (len == 0) return new Tensor<T>((int[])input._shape.Clone());
                var contiguousInput = input.IsContiguous
                    ? input
                    : (Tensor<int>)input.Contiguous();
                using var inBuf = GetOrAllocateInt32IndexBuffer(backend, contiguousInput);
                var outBuf = AllocateOutputBuffer(backend, len);
                bool outputHandedOff = false;
                try
                {
                    audio.MuLawDecoding(inBuf.Buffer, outBuf.Buffer, len, quantizationChannels);
                    var result = DeferTensorResult<T>(
                        backend, outBuf.Buffer, len, (int[])input._shape.Clone());
                    outBuf.RelinquishOwnership();
                    outputHandedOff = true;
                    return result;
                }
                finally
                {
                    if (!outputHandedOff)
                        outBuf.Dispose();
                }
            }
        }
        return base.MuLawDecoding<T>(input, quantizationChannels);
    }

    /// <inheritdoc/>
    public override Tensor<T> ComputeDeltas<T>(Tensor<T> input, int winLength = 5)
    {
        if (winLength < 3 || (winLength & 1) == 0)
            throw new ArgumentException(
                "winLength must be an odd integer >= 3 (denominator 2·Σk² collapses to 0 otherwise).",
                nameof(winLength));
        if (typeof(T) == typeof(float) && input.Rank >= 1)
        {
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int timeAxis = input._shape[input.Rank - 1];
                if (timeAxis == 0)
                    return new Tensor<T>((int[])input._shape.Clone());
                int leading = input.Length / timeAxis;
                using var inBuf = GetOrAllocateBuffer(backend, input);
                var outBuf = AllocateOutputBuffer(backend, input.Length);
                try
                {
                    audio.ComputeDeltas(inBuf.Buffer, outBuf.Buffer, leading, timeAxis, winLength);
                    var result = DeferTensorResult<T>(
                        backend, outBuf.Buffer, input.Length, (int[])input._shape.Clone());
                    outBuf.RelinquishOwnership();
                    return result;
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.ComputeDeltas(input, winLength);
    }

    /// <inheritdoc/>
    public override Tensor<T> Resample<T>(Tensor<T> waveform, int origRate, int newRate)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        if (typeof(T) == typeof(float) && origRate > 0 && newRate == origRate
            && TryGetBackend(out var copyBackend))
        {
            using var inputBuffer = GetOrAllocateBuffer(copyBackend, waveform);
            return DispatchDeferredGpuOp<T>(copyBackend, waveform.Length, waveform.Shape.ToArray(),
                output => copyBackend.Copy(inputBuffer.Buffer, output, waveform.Length));
        }
        if (typeof(T) == typeof(float) && waveform.Rank >= 1
            && origRate > 0 && newRate > 0 && origRate != newRate)
        {
            if (TryGetBackend(out var backend) && backend is IAudioBackend audio)
            {
                int inLen = waveform._shape[waveform.Rank - 1];
                if (inLen == 0)
                {
                    var emptyShape = (int[])waveform._shape.Clone();
                    emptyShape[waveform.Rank - 1] = 0;
                    return new Tensor<T>(emptyShape);
                }
                int gcd = Gcd(origRate, newRate);
                int up = newRate / gcd;
                int down = origRate / gcd;
                int halfWidth = Math.Max(8, Math.Min(256, up * 8));

                int outLen = (int)((long)inLen * up / down);
                int leading = waveform.Length / inLen;
                // Widened arithmetic guards int overflow on huge waveforms.
                long outTotal64 = checked((long)leading * outLen);
                if (outTotal64 > int.MaxValue)
                    throw new OverflowException(
                        $"Resample output element count {outTotal64} exceeds Int32.MaxValue.");
                int outTotal = (int)outTotal64;
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
                    var outShape = (int[])waveform._shape.Clone();
                    outShape[waveform.Rank - 1] = outLen;
                    var result = DeferTensorResult<T>(backend, outBuf.Buffer, outTotal, outShape);
                    outBuf.RelinquishOwnership();
                    return result;
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.Resample(waveform, origRate, newRate);
    }

    private static int Gcd(int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; }
}
