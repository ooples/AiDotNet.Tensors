// Copyright (c) AiDotNet. All rights reserved.
#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend : IAudioBackend
    {
        private const int AudioLocalSize = 256;
        private DirectOpenClKernel GetAudioKernel(string name)
        {
            if (!_kernelCache.TryGetValue(name, out var k))
                throw new InvalidOperationException($"OpenCL audio kernel not found: {name}.");
            return k;
        }
        private static int RoundUpAudio(int v) => ((v + AudioLocalSize - 1) / AudioLocalSize) * AudioLocalSize;
        private static IntPtr AudioHandle(IGpuBuffer b) => ((DirectOpenClGpuBuffer)b).Buffer.Handle;

        public void AmplitudeToDB(IGpuBuffer input, IGpuBuffer output, int length,
            float minAmplitude, float topDb, bool clipTopDb)
        {
            if (length <= 0) return;
            var k = GetAudioKernel("audio_amplitude_to_db");
            k.SetArg(0, AudioHandle(input));
            k.SetArg(1, AudioHandle(output));
            k.SetArg(2, length); k.SetArg(3, minAmplitude);
            k.SetArg(4, topDb); k.SetArg(5, clipTopDb ? 1 : 0);
            k.Execute1D(RoundUpAudio(length), AudioLocalSize);
        }

        public void MuLawEncoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
        {
            if (length <= 0) return;
            var k = GetAudioKernel("audio_mulaw_encoding");
            k.SetArg(0, AudioHandle(input));
            k.SetArg(1, AudioHandle(output));
            k.SetArg(2, length); k.SetArg(3, qc);
            k.Execute1D(RoundUpAudio(length), AudioLocalSize);
        }

        public void MuLawDecoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
        {
            if (length <= 0) return;
            var k = GetAudioKernel("audio_mulaw_decoding");
            k.SetArg(0, AudioHandle(input));
            k.SetArg(1, AudioHandle(output));
            k.SetArg(2, length); k.SetArg(3, qc);
            k.Execute1D(RoundUpAudio(length), AudioLocalSize);
        }

        public void ComputeDeltas(IGpuBuffer input, IGpuBuffer output,
            int leading, int timeAxis, int winLength)
        {
            int total = leading * timeAxis;
            if (total <= 0) return;
            var k = GetAudioKernel("audio_compute_deltas");
            k.SetArg(0, AudioHandle(input));
            k.SetArg(1, AudioHandle(output));
            k.SetArg(2, leading); k.SetArg(3, timeAxis); k.SetArg(4, winLength);
            k.Execute1D(RoundUpAudio(total), AudioLocalSize);
        }

        public void Resample(IGpuBuffer input, IGpuBuffer output,
            int leading, int inLen, int outLen, int up, int down, int halfWidth)
        {
            int total = leading * outLen;
            if (total <= 0) return;
            var k = GetAudioKernel("audio_resample");
            k.SetArg(0, AudioHandle(input));
            k.SetArg(1, AudioHandle(output));
            k.SetArg(2, leading); k.SetArg(3, inLen); k.SetArg(4, outLen);
            k.SetArg(5, up); k.SetArg(6, down); k.SetArg(7, halfWidth);
            k.Execute1D(RoundUpAudio(total), AudioLocalSize);
        }
    }
}
#endif
