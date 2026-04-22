// Copyright (c) AiDotNet. All rights reserved.
// CUDA audio launcher shims (Issue #217 tail).
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IAudioBackend
{
    private IntPtr ResolveAudioKernel(string name)
    {
        if (_audioModule == IntPtr.Zero)
            throw new InvalidOperationException("Audio CUDA module was not compiled.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        return kernel;
    }

    public unsafe void AmplitudeToDB(IGpuBuffer input, IGpuBuffer output, int length,
        float minAmplitude, float topDb, bool clipTopDb)
    {
        if (length <= 0) return;
        var kernel = ResolveAudioKernel("audio_amplitude_to_db");
        using var _ = PushContext();
        uint grid = (uint)((length + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, oP = output.Handle;
        int len = length;
        float minA = minAmplitude;
        float topDbFloor = topDb;
        int clip = clipTopDb ? 1 : 0;
        void** args = stackalloc void*[6];
        args[0] = &inP; args[1] = &oP;
        args[2] = &len; args[3] = &minA; args[4] = &topDbFloor; args[5] = &clip;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MuLawEncoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        var kernel = ResolveAudioKernel("audio_mulaw_encoding");
        using var _ = PushContext();
        uint grid = (uint)((length + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, oP = output.Handle;
        int len = length, q = qc;
        void** args = stackalloc void*[4];
        args[0] = &inP; args[1] = &oP; args[2] = &len; args[3] = &q;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void MuLawDecoding(IGpuBuffer input, IGpuBuffer output, int length, int qc)
    {
        if (length <= 0) return;
        var kernel = ResolveAudioKernel("audio_mulaw_decoding");
        using var _ = PushContext();
        uint grid = (uint)((length + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, oP = output.Handle;
        int len = length, q = qc;
        void** args = stackalloc void*[4];
        args[0] = &inP; args[1] = &oP; args[2] = &len; args[3] = &q;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void ComputeDeltas(IGpuBuffer input, IGpuBuffer output,
        int leading, int timeAxis, int winLength)
    {
        long totalLong = (long)leading * timeAxis;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"ComputeDeltas total {totalLong} exceeds Int32.MaxValue.");
        int total = (int)totalLong;
        if (total <= 0) return;
        var kernel = ResolveAudioKernel("audio_compute_deltas");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, oP = output.Handle;
        int ld = leading, t = timeAxis, w = winLength;
        void** args = stackalloc void*[5];
        args[0] = &inP; args[1] = &oP;
        args[2] = &ld; args[3] = &t; args[4] = &w;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }

    public unsafe void Resample(IGpuBuffer input, IGpuBuffer output,
        int leading, int inLen, int outLen, int up, int down, int halfWidth)
    {
        long totalLong = (long)leading * outLen;
        if (totalLong > int.MaxValue)
            throw new OverflowException($"Resample total {totalLong} exceeds Int32.MaxValue.");
        int total = (int)totalLong;
        if (total <= 0) return;
        var kernel = ResolveAudioKernel("audio_resample");
        using var _ = PushContext();
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr inP = input.Handle, oP = output.Handle;
        int ld = leading, il = inLen, ol = outLen, u = up, d = down, hw = halfWidth;
        void** args = stackalloc void*[8];
        args[0] = &inP; args[1] = &oP;
        args[2] = &ld; args[3] = &il; args[4] = &ol;
        args[5] = &u; args[6] = &d; args[7] = &hw;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
}
