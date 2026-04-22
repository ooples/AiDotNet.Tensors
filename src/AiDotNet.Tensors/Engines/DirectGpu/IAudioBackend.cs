// Copyright (c) AiDotNet. All rights reserved.
// Native GPU kernels for the audio primitives in Issue #217 tail. Ops
// that compose existing GPU-accelerated STFT (Spectrogram, PitchShift,
// TimeStretch) stay in managed code and don't live on this interface.
namespace AiDotNet.Tensors.Engines.DirectGpu
{
    public interface IAudioBackend
    {
        /// <summary>Element-wise 20·log10(max(x, minAmp)); optional topDb floor.</summary>
        void AmplitudeToDB(IGpuBuffer input, IGpuBuffer output, int length, float minAmplitude, float topDb, bool clipTopDb);

        /// <summary>μ-law companding encoder (torchaudio's MuLawEncoding).</summary>
        void MuLawEncoding(IGpuBuffer input, IGpuBuffer output, int length, int quantizationChannels);

        /// <summary>μ-law companding decoder.</summary>
        void MuLawDecoding(IGpuBuffer input, IGpuBuffer output, int length, int quantizationChannels);

        /// <summary>Time-axis derivative for audio features. <c>input</c>
        /// is laid out as <c>[leading, T]</c> row-major with <c>T</c> =
        /// audio time axis.</summary>
        void ComputeDeltas(IGpuBuffer input, IGpuBuffer output, int leading, int timeAxis, int winLength);

        /// <summary>Polyphase resampler (Hann-windowed sinc). Same layout
        /// convention as <see cref="ComputeDeltas"/>.</summary>
        void Resample(IGpuBuffer input, IGpuBuffer output,
            int leading, int inLen, int outLen, int up, int down, int halfWidth);
    }
}
