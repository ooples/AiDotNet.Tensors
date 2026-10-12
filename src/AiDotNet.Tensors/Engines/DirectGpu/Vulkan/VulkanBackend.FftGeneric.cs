// Copyright (c) AiDotNet. All rights reserved.
// Generic-storage, arbitrary-length Vulkan FFT dispatch (#964).
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <remarks>
/// Batched radix-2 for power-of-two lengths and Bluestein for every other length, on split real / imaginary buffers of
/// float32, float16 or bfloat16 storage (see <see cref="VulkanFftGenericKernels"/>). Every step runs on the device:
/// the twiddle and chirp tables are built by kernels, and no value is read back.
/// </remarks>
public sealed unsafe partial class VulkanBackend : IGenericFftBackend
{
    private const int MaximumGenericFftLength = 1 << 29;

    private readonly object _fftGenericLock = new object();
    // Per element type: null until first asked, then whether every kernel of the type compiled.
    private readonly bool?[] _fftGenericReady = new bool?[3];
    private readonly string?[] _fftGenericError = new string?[3];

    bool IGenericFftBackend.SupportsFftElementType(FftElementType type)
    {
        int index = FftTypeIndex(type);
        if (!IsGlslCompilerAvailable) return false;
        lock (_fftGenericLock)
        {
            if (_fftGenericReady[index] is bool known) return known;
            // The capability is reported only once every pipeline the transform can launch exists.
            bool ready = true;
            foreach (var kernel in VulkanFftGenericKernels.All(type))
            {
                if (GetOrCreateGlslPipeline(kernel.Source, kernel.Bindings, (uint)(kernel.PushWords * sizeof(uint))) is not null) continue;
                ready = false;
                _fftGenericError[index] = $"{kernel.Name}: {_glslCompiler?.LastError ?? "pipeline creation failed"}";
                break;
            }
            _fftGenericReady[index] = ready;
            return ready;
        }
    }

    private static int FftTypeIndex(FftElementType type) => type switch
    {
        FftElementType.Float32 => 0,
        FftElementType.Float16 => 1,
        FftElementType.BFloat16 => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FFT element type."),
    };

    private void EnsureGenericFftSupported(FftElementType type)
    {
        if (((IGenericFftBackend)this).SupportsFftElementType(type)) return;
        string? error = _fftGenericError[FftTypeIndex(type)];
        throw new NotSupportedException(
            $"Vulkan generic FFT storage {type} is unavailable on {DeviceName}." +
            (string.IsNullOrWhiteSpace(error) ? " The runtime GLSL compiler (libshaderc) is required." : " " + error));
    }

    private void FftDispatch(VulkanFftGenericKernel kernel, int threads, uint[] push, params IGpuBuffer[] buffers)
    {
        if (push.Length != kernel.PushWords || buffers.Length != kernel.Bindings)
            throw new InvalidOperationException($"Vulkan generic FFT kernel {kernel.Name} was launched with the wrong argument list.");
        GlslNaryOp(kernel.Source, buffers, threads, push, VulkanFftGenericKernels.LocalSize);
    }

    private static uint FftFloatBits(float value) => *(uint*)&value;

    void IGenericFftBackend.ConvertFloatToFftStorage(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        ValidateFftConversion(source, destination, count, type, floatToStorage: true);
        if (type == FftElementType.Float32) { CopyBuffer(source, destination, count); return; }
        FftDispatch(VulkanFftGenericKernels.FromFloat(type), VulkanFftGenericKernels.WordsFor(count, type), new[] { (uint)count }, source, destination);
    }

    void IGenericFftBackend.ConvertFftStorageToFloat(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type)
    {
        ValidateFftConversion(source, destination, count, type, floatToStorage: false);
        if (type == FftElementType.Float32) { CopyBuffer(source, destination, count); return; }
        FftDispatch(VulkanFftGenericKernels.ToFloat(type), count, new[] { (uint)count }, source, destination);
    }

    private void ValidateFftConversion(IGpuBuffer source, IGpuBuffer destination, int count, FftElementType type, bool floatToStorage)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        EnsureGenericFftSupported(type);
        long floatBytes = (long)count * sizeof(float);
        long storageBytes = (long)count * type.ByteSize();
        long requiredSource = floatToStorage ? floatBytes : storageBytes;
        long requiredDestination = floatToStorage ? storageBytes : floatBytes;
        if (source.SizeInBytes < requiredSource)
            throw new ArgumentException($"Source buffer provides {source.SizeInBytes} bytes; {requiredSource} are required.", nameof(source));
        if (destination.SizeInBytes < requiredDestination)
            throw new ArgumentException($"Destination buffer provides {destination.SizeInBytes} bytes; {requiredDestination} are required.", nameof(destination));
    }

    void IGenericFftBackend.LaunchFftGeneric(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
    {
        if (real is null) throw new ArgumentNullException(nameof(real));
        if (imaginary is null) throw new ArgumentNullException(nameof(imaginary));
        if (ReferenceEquals(real, imaginary)) throw new ArgumentException("The real and imaginary buffers must be distinct.", nameof(imaginary));
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch), batch, "Batch must be positive.");
        if (n <= 0 || n > MaximumGenericFftLength)
            throw new ArgumentOutOfRangeException(nameof(n), n, $"Transform length must be in [1, {MaximumGenericFftLength}].");
        long count = (long)batch * n;
        if (count > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "batch * n must fit a 32-bit element count.");
        EnsureGenericFftSupported(type);
        long requiredBytes = count * type.ByteSize();
        if (real.SizeInBytes < requiredBytes)
            throw new ArgumentException($"Real buffer provides {real.SizeInBytes} bytes; {requiredBytes} are required.", nameof(real));
        if (imaginary.SizeInBytes < requiredBytes)
            throw new ArgumentException($"Imaginary buffer provides {imaginary.SizeInBytes} bytes; {requiredBytes} are required.", nameof(imaginary));

        // A length-1 transform is the identity in both directions (the inverse scale is 1).
        if (n == 1) return;
        if ((n & (n - 1)) == 0) FftRadix2(real, imaginary, batch, n, inverse, type);
        else FftBluestein(real, imaginary, batch, n, inverse, type);
    }

    private static int FftLog2(int n)
    {
        int result = 0;
        while ((1 << result) < n) result++;
        return result;
    }

    // In-place radix-2 over `batch` rows of power-of-two length n >= 2.
    private void FftRadix2(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
    {
        int count = checked(batch * n);
        int words = VulkanFftGenericKernels.WordsFor(count, type);
        int perWord = VulkanFftGenericKernels.ElementsPerWord(type);
        int halfLength = n / 2;
        using IGpuBuffer twiddleReal = AllocateBuffer(halfLength);
        using IGpuBuffer twiddleImaginary = AllocateBuffer(halfLength);
        FftDispatch(VulkanFftGenericKernels.BuildTwiddles, halfLength,
            new[] { (uint)halfLength, (uint)n, inverse ? 1u : 0u }, twiddleReal, twiddleImaginary);

        using (IGpuBuffer scratchReal = AllocateBuffer(words))
        using (IGpuBuffer scratchImaginary = AllocateBuffer(words))
        {
            FftDispatch(VulkanFftGenericKernels.BitReverse(type), words,
                new[] { (uint)count, (uint)n, (uint)FftLog2(n) }, real, imaginary, scratchReal, scratchImaginary);
            CopyBuffer(scratchReal, real, words);
            CopyBuffer(scratchImaginary, imaginary, words);
        }

        var butterfly = VulkanFftGenericKernels.Butterfly(type);
        for (int stride = 2; ; stride *= 2)
        {
            // A thread owns `perWord` wings; the first stage of a two-per-word type owns one word instead.
            int threads = stride / 2 < perWord ? count / 2 : count / (2 * perWord);
            FftDispatch(butterfly, threads, new[] { (uint)count, (uint)n, (uint)stride }, real, imaginary, twiddleReal, twiddleImaginary);
            if (stride == n) break;
        }

        if (inverse)
            FftDispatch(VulkanFftGenericKernels.Scale(type), words, new[] { (uint)count, FftFloatBits(1.0f / n) }, real, imaginary);
    }

    /// <summary>Smallest power of two at least 2n - 1: the convolution length Bluestein needs.</summary>
    internal static int BluesteinLength(int n)
    {
        if (n <= 0 || n > MaximumGenericFftLength)
            throw new ArgumentOutOfRangeException(nameof(n), n, $"Transform length must be in the range [1, {MaximumGenericFftLength}].");
        if (n == 1) return 1;
        long target = 2L * n - 1;
        long m = 1;
        while (m < target) m <<= 1;
        return checked((int)m);
    }

    // Bluestein: x[k] conj(c[k]) convolved with the chirp c through three float32 radix-2 transforms of length m.
    private void FftBluestein(IGpuBuffer real, IGpuBuffer imaginary, int batch, int n, bool inverse, FftElementType type)
    {
        int count = checked(batch * n);
        int m = BluesteinLength(n);
        long workspaceLong = (long)batch * m;
        if (workspaceLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "The Bluestein workspace (batch * m) must fit a 32-bit element count.");
        int workspaceCount = (int)workspaceLong;
        uint inverseFlag = inverse ? 1u : 0u;
        using IGpuBuffer chirpReal = AllocateBuffer(n);
        using IGpuBuffer chirpImaginary = AllocateBuffer(n);
        using IGpuBuffer kernelReal = AllocateBuffer(m);
        using IGpuBuffer kernelImaginary = AllocateBuffer(m);
        using IGpuBuffer workspaceReal = AllocateBuffer(workspaceCount);
        using IGpuBuffer workspaceImaginary = AllocateBuffer(workspaceCount);

        FftDispatch(VulkanFftGenericKernels.BuildChirp, n, new[] { (uint)n, inverseFlag }, chirpReal, chirpImaginary);
        FftDispatch(VulkanFftGenericKernels.BuildChirpPadded, m, new[] { (uint)n, (uint)m, inverseFlag }, kernelReal, kernelImaginary);
        FftRadix2(kernelReal, kernelImaginary, 1, m, inverse: false, FftElementType.Float32);
        FftDispatch(VulkanFftGenericKernels.BluesteinPremultiply(type), workspaceCount,
            new[] { (uint)workspaceCount, (uint)n, (uint)m }, real, imaginary, chirpReal, chirpImaginary, workspaceReal, workspaceImaginary);
        FftRadix2(workspaceReal, workspaceImaginary, batch, m, inverse: false, FftElementType.Float32);
        FftDispatch(VulkanFftGenericKernels.BluesteinPointwise, workspaceCount,
            new[] { (uint)workspaceCount, (uint)m }, workspaceReal, workspaceImaginary, kernelReal, kernelImaginary);
        FftRadix2(workspaceReal, workspaceImaginary, batch, m, inverse: true, FftElementType.Float32);
        FftDispatch(VulkanFftGenericKernels.BluesteinPostmultiply(type), VulkanFftGenericKernels.WordsFor(count, type),
            new[] { (uint)count, (uint)n, (uint)m, FftFloatBits(inverse ? 1.0f / n : 1.0f) },
            workspaceReal, workspaceImaginary, chirpReal, chirpImaginary, real, imaginary);
    }
}
