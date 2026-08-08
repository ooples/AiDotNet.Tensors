using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Hardware-independent checks that HIP FFT entry points fail before touching buffers when their
/// compiled kernel module is incomplete.
/// </summary>
public sealed class HipFftKernelAvailabilityTests
{
    private static readonly string[] AllRequiredKernels =
    [
        "bit_reverse_permutation",
        "fft_butterfly",
        "batched_bit_reverse",
        "batched_fft_butterfly",
        "fft_rows_bit_reverse",
        "fft_rows_butterfly",
        "fft_cols_bit_reverse",
        "fft_cols_butterfly",
        "scale_inverse"
    ];

    public static TheoryData<string, string, bool> MissingKernelCases => new()
    {
        { "FFT", "bit_reverse_permutation", false },
        { "FFT", "fft_butterfly", false },
        { "FFT", "scale_inverse", true },
        { "BatchedFFT", "batched_bit_reverse", false },
        { "BatchedFFT", "batched_fft_butterfly", false },
        { "BatchedFFT", "scale_inverse", true },
        { "FFT2D", "fft_rows_bit_reverse", false },
        { "FFT2D", "fft_rows_butterfly", false },
        { "FFT2D", "fft_cols_bit_reverse", false },
        { "FFT2D", "fft_cols_butterfly", false },
        { "FFT2D", "scale_inverse", true }
    };

    [Theory]
    [MemberData(nameof(MissingKernelCases))]
    public void EntryPoint_MissingRequiredKernel_FailsClosed(
        string entryPoint,
        string missingKernel,
        bool inverse)
    {
        HipBackend backend = CreateWithoutHardware(missingKernel);
        var inputReal = new MockGpuBuffer(new float[16]);
        var inputImag = new MockGpuBuffer(new float[16]);
        var outputReal = new MockGpuBuffer(new float[16]);
        var outputImag = new MockGpuBuffer(new float[16]);

        Action transform = entryPoint switch
        {
            "FFT" => () => backend.FFT(inputReal, inputImag, outputReal, outputImag, 16, inverse),
            "BatchedFFT" => () => backend.BatchedFFT(
                inputReal, inputImag, outputReal, outputImag, batch: 2, n: 8, inverse),
            "FFT2D" => () => backend.FFT2D(
                inputReal, inputImag, outputReal, outputImag, height: 4, width: 4, inverse),
            _ => throw new ArgumentOutOfRangeException(nameof(entryPoint), entryPoint, null)
        };

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(transform);

        Assert.Contains(missingKernel, exception.Message, StringComparison.Ordinal);
        Assert.Contains("Mock AMD GPU", exception.Message, StringComparison.Ordinal);
        Assert.Contains("cannot run", exception.Message, StringComparison.Ordinal);
    }

    private static HipBackend CreateWithoutHardware(string missingKernel)
    {
#pragma warning disable SYSLIB0050 // Kernel availability is deliberately tested without constructing a native HIP context.
        var backend = (HipBackend)FormatterServices.GetUninitializedObject(typeof(HipBackend));
#pragma warning restore SYSLIB0050
        GC.SuppressFinalize(backend);

        var kernels = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        foreach (string kernelName in AllRequiredKernels)
        {
            if (!string.Equals(kernelName, missingKernel, StringComparison.Ordinal))
                kernels.Add(kernelName, new IntPtr(kernels.Count + 1));
        }

        SetField(backend, "_kernelCache", kernels);
        SetField(backend, "<IsAvailable>k__BackingField", true);
        SetField(backend, "<DeviceName>k__BackingField", "Mock AMD GPU");
        return backend;
    }

    private static void SetField(object target, string name, object value)
    {
        FieldInfo field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field not found: {name}");
        field.SetValue(target, value);
    }
}
