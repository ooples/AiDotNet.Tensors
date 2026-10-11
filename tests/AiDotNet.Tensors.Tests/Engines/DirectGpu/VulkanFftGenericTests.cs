using System;
using System.Linq;
using System.Numerics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #964: the Vulkan generic-precision, arbitrary-length FFT. Radix-2 and Bluestein transforms, forward and inverse, in
/// every storage type, are compared with a direct DFT of the values the device actually stored; the IEngine contract
/// must run them on the device (no readback, no recorded fallback) and keep an active gradient tape connected.
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanFftGenericTests
{
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
    }

    // The backend is a process-wide singleton, so the engine around it is non-owning and never disposed.
    private static DirectGpuTensorEngine Engine(VulkanBackend backend) => new DirectGpuTensorEngine(new DirectGpuEngine(backend));

    private static Complex[] Random(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new Complex[count];
        for (int i = 0; i < count; i++) values[i] = new Complex(rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1);
        return values;
    }

    private static Complex[] Dft(Complex[] x, bool inverse)
    {
        int n = x.Length;
        var y = new Complex[n];
        double sign = inverse ? 2 * Math.PI : -2 * Math.PI;
        for (int k = 0; k < n; k++)
        {
            Complex sum = Complex.Zero;
            // (j * k) mod n keeps the phase exact for long transforms.
            for (int j = 0; j < n; j++) sum += x[j] * Complex.FromPolarCoordinates(1.0, sign * (((long)j * k) % n) / n);
            y[k] = inverse ? sum / n : sum;
        }
        return y;
    }

    private static Complex[] DftBatched(Complex[] values, int batch, int n, bool inverse)
    {
        var result = new Complex[values.Length];
        for (int b = 0; b < batch; b++)
        {
            var row = new Complex[n];
            Array.Copy(values, b * n, row, 0, n);
            Array.Copy(Dft(row, inverse), 0, result, b * n, n);
        }
        return result;
    }

    private static double RelErr(Complex[] actual, Complex[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            num += (actual[i] - expected[i]).Magnitude * (actual[i] - expected[i]).Magnitude;
            den += expected[i].Magnitude * expected[i].Magnitude;
        }
        return Math.Sqrt(num / Math.Max(den, 1e-300));
    }

    private static Complex[] DownloadComplex(VulkanBackend backend, IGpuBuffer real, IGpuBuffer imaginary, int count, FftElementType type)
    {
        var fft = (IGenericFftBackend)backend;
        var realValues = new float[count];
        var imaginaryValues = new float[count];
        if (type == FftElementType.Float32)
        {
            Array.Copy(backend.DownloadBuffer(real), realValues, count);
            Array.Copy(backend.DownloadBuffer(imaginary), imaginaryValues, count);
        }
        else
        {
            using IGpuBuffer realFloat = backend.AllocateBuffer(count);
            using IGpuBuffer imaginaryFloat = backend.AllocateBuffer(count);
            fft.ConvertFftStorageToFloat(real, realFloat, count, type);
            fft.ConvertFftStorageToFloat(imaginary, imaginaryFloat, count, type);
            Array.Copy(backend.DownloadBuffer(realFloat), realValues, count);
            Array.Copy(backend.DownloadBuffer(imaginaryFloat), imaginaryValues, count);
        }
        var values = new Complex[count];
        for (int i = 0; i < count; i++) values[i] = new Complex(realValues[i], imaginaryValues[i]);
        return values;
    }

    private static double Tolerance(FftElementType type) => type switch
    {
        FftElementType.Float32 => 2e-5,
        FftElementType.Float16 => 3e-2,
        FftElementType.BFloat16 => 8e-2,
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    public static TheoryData<FftElementType, int, int> ExecutionCases()
    {
        var data = new TheoryData<FftElementType, int, int>();
        foreach (FftElementType type in new[] { FftElementType.Float32, FftElementType.Float16, FftElementType.BFloat16 })
        {
            // Radix-2: the one-stage, two-stage and longer transforms; an odd batch leaves a half-filled last word.
            data.Add(type, 2, 3);
            data.Add(type, 4, 3);
            data.Add(type, 8, 2);
            data.Add(type, 64, 5);
            data.Add(type, 1024, 1);
            // Bluestein: odd, even non-power-of-two, and prime lengths; odd rows do not align to 16-bit word pairs.
            data.Add(type, 3, 1);
            data.Add(type, 5, 2);
            data.Add(type, 6, 3);
            data.Add(type, 15, 3);
            data.Add(type, 100, 2);
            data.Add(type, 127, 1);
            data.Add(type, 1, 3);
        }
        return data;
    }

    [SkippableTheory]
    [MemberData(nameof(ExecutionCases))]
    public void Execution_MatchesDirectDft_AndRoundTrips(FftElementType type, int n, int batch)
    {
        var backend = BackendOrSkip();
        var fft = (IGenericFftBackend)backend;
        Skip.IfNot(fft.SupportsFftElementType(type), $"Vulkan {type} FFT storage is unavailable.");

        int count = batch * n;
        Complex[] source = Random(count, 4200 + n * 7 + (int)type);
        var sourceReal = new float[count];
        var sourceImaginary = new float[count];
        for (int i = 0; i < count; i++) { sourceReal[i] = (float)source[i].Real; sourceImaginary[i] = (float)source[i].Imaginary; }

        using IGpuBuffer real = type == FftElementType.Float32 ? backend.AllocateBuffer(sourceReal) : backend.AllocateByteBuffer(count * type.ByteSize());
        using IGpuBuffer imaginary = type == FftElementType.Float32 ? backend.AllocateBuffer(sourceImaginary) : backend.AllocateByteBuffer(count * type.ByteSize());
        if (type != FftElementType.Float32)
        {
            using IGpuBuffer sourceRealBuffer = backend.AllocateBuffer(sourceReal);
            using IGpuBuffer sourceImaginaryBuffer = backend.AllocateBuffer(sourceImaginary);
            fft.ConvertFloatToFftStorage(sourceRealBuffer, real, count, type);
            fft.ConvertFloatToFftStorage(sourceImaginaryBuffer, imaginary, count, type);
        }

        // The oracle transforms what the device stored, so the storage type's quantization is not counted as error.
        Complex[] quantizedInput = DownloadComplex(backend, real, imaginary, count, type);
        fft.LaunchFftGeneric(real, imaginary, batch, n, inverse: false, type);
        Complex[] actualForward = DownloadComplex(backend, real, imaginary, count, type);
        Complex[] expectedForward = DftBatched(quantizedInput, batch, n, inverse: false);
        double forwardError = RelErr(actualForward, expectedForward);
        Assert.True(forwardError < Tolerance(type), $"{type} forward n={n} batch={batch}: rel-err {forwardError:E3}");

        fft.LaunchFftGeneric(real, imaginary, batch, n, inverse: true, type);
        Complex[] roundTrip = DownloadComplex(backend, real, imaginary, count, type);
        double roundTripError = RelErr(roundTrip, quantizedInput);
        Assert.True(roundTripError < Tolerance(type), $"{type} round-trip n={n} batch={batch}: rel-err {roundTripError:E3}");
    }

    [SkippableTheory]
    [InlineData(8)]
    [InlineData(12)]
    [InlineData(35)]
    public void Inverse_IsScaledByOneOverN_AndForwardIsUnscaled(int n)
    {
        var backend = BackendOrSkip();
        var fft = (IGenericFftBackend)backend;
        Skip.IfNot(fft.SupportsFftElementType(FftElementType.Float32), "Vulkan Float32 FFT storage is unavailable.");
        // A unit impulse: the forward transform is all ones (unscaled), the inverse transform all 1 / n.
        foreach (bool inverse in new[] { false, true })
        {
            var impulse = new float[n];
            impulse[0] = 1f;
            using IGpuBuffer real = backend.AllocateBuffer(impulse);
            using IGpuBuffer imaginary = backend.AllocateBuffer(new float[n]);
            fft.LaunchFftGeneric(real, imaginary, 1, n, inverse, FftElementType.Float32);
            float[] re = backend.DownloadBuffer(real), im = backend.DownloadBuffer(imaginary);
            float expected = inverse ? 1f / n : 1f;
            for (int k = 0; k < n; k++)
            {
                Assert.True(Math.Abs(re[k] - expected) < 1e-5, $"n={n} inverse={inverse} re[{k}] = {re[k]:R}");
                Assert.True(Math.Abs(im[k]) < 1e-5, $"n={n} inverse={inverse} im[{k}] = {im[k]:R}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(FftElementType.Float16)]
    [InlineData(FftElementType.BFloat16)]
    public void StorageConversion_RoundsToNearestEven_AndKeepsSpecialValues(FftElementType type)
    {
        var backend = BackendOrSkip();
        var fft = (IGenericFftBackend)backend;
        Skip.IfNot(fft.SupportsFftElementType(type), $"Vulkan {type} FFT storage is unavailable.");
        float[] values =
        {
            0f, -0f, 1f, -1f, 0.1f, -0.3f, 1.5f, 1024.25f, 3.1415927f, 65504f, 65520f, 1e-5f, 6e-8f, 2.98e-8f, 1e-10f,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN, 1.0002441f, 1.0003662f, 1e30f,
        };
        using IGpuBuffer source = backend.AllocateBuffer(values);
        using IGpuBuffer storage = backend.AllocateByteBuffer(values.Length * type.ByteSize());
        using IGpuBuffer back = backend.AllocateBuffer(values.Length);
        fft.ConvertFloatToFftStorage(source, storage, values.Length, type);
        fft.ConvertFftStorageToFloat(storage, back, values.Length, type);
        float[] actual = backend.DownloadBuffer(back);
        for (int i = 0; i < values.Length; i++)
        {
            float expected = type == FftElementType.Float16 ? RoundToHalf(values[i]) : RoundToBFloat16(values[i]);
            if (float.IsNaN(expected)) { Assert.True(float.IsNaN(actual[i]), $"{type} [{i}] {values[i]:R}: expected NaN, got {actual[i]:R}"); continue; }
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(actual[i]),
                $"{type} [{i}] {values[i]:R}: expected {expected:R}, got {actual[i]:R}");
        }
    }

    // The nearest binary16 value (ties to even) as a float, by exact arithmetic on the value.
    private static float RoundToHalf(float v)
    {
        if (float.IsNaN(v) || float.IsInfinity(v) || v == 0f) return v;
        double a = Math.Abs((double)v);
        double result;
        if (a < Math.Pow(2, -14))
        {
            // Subnormal half: a multiple of 2^-24.
            result = Math.Round(a * Math.Pow(2, 24), MidpointRounding.ToEven) * Math.Pow(2, -24);
        }
        else
        {
            int e = (int)Math.Floor(Math.Log(a, 2));
            if (Math.Pow(2, e) > a) e--;
            if (Math.Pow(2, e + 1) <= a) e++;
            double unit = Math.Pow(2, e - 10);
            result = Math.Round(a / unit, MidpointRounding.ToEven) * unit;
        }
        if (result >= 65536.0) result = double.PositiveInfinity;
        return (float)(v < 0 ? -result : result);
    }

    // The nearest bfloat16 value (ties to even): float32 with an 8-bit significand.
    private static float RoundToBFloat16(float v)
    {
        if (float.IsNaN(v) || float.IsInfinity(v) || v == 0f) return v;
        uint bits = unchecked((uint)TestHelpers.MathCompat.SingleToInt32Bits(v));
        uint rounded = ((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) << 16;
        byte[] raw = BitConverter.GetBytes(rounded);
        return BitConverter.ToSingle(raw, 0);
    }

    [SkippableFact]
    public void Capability_IsReportedPerElementType_AndRejectsUnknownTypes()
    {
        var backend = BackendOrSkip();
        var fft = (IGenericFftBackend)backend;
        var engine = Engine(backend);
        foreach (FftElementType type in new[] { FftElementType.Float32, FftElementType.Float16, FftElementType.BFloat16 })
            Assert.Equal(fft.SupportsFftElementType(type), engine.SupportsFftElementType(type));
        // With the GLSL compiler present every storage type compiles: none needs a device feature.
        Assert.True(fft.SupportsFftElementType(FftElementType.Float32));
        Assert.Throws<ArgumentOutOfRangeException>(() => fft.SupportsFftElementType((FftElementType)99));
    }

    [SkippableFact]
    public void Launch_RejectsBadArguments()
    {
        var backend = BackendOrSkip();
        var fft = (IGenericFftBackend)backend;
        using IGpuBuffer small = backend.AllocateBuffer(8);
        using IGpuBuffer large = backend.AllocateBuffer(64);
        using IGpuBuffer large2 = backend.AllocateBuffer(64);
        Assert.Throws<ArgumentException>(() => fft.LaunchFftGeneric(large, small, 4, 16, false, FftElementType.Float32));
        Assert.Throws<ArgumentException>(() => fft.LaunchFftGeneric(large, large, 4, 16, false, FftElementType.Float32));
        Assert.Throws<ArgumentOutOfRangeException>(() => fft.LaunchFftGeneric(large, large2, 0, 16, false, FftElementType.Float32));
        Assert.Throws<ArgumentOutOfRangeException>(() => fft.LaunchFftGeneric(large, large2, 4, 0, false, FftElementType.Float32));
        Assert.Throws<ArgumentNullException>(() => fft.LaunchFftGeneric(null!, large2, 4, 16, false, FftElementType.Float32));
        Assert.Throws<ArgumentException>(() => fft.ConvertFloatToFftStorage(large, small, 64, FftElementType.Float16));
        Assert.Equal(1, VulkanBackend.BluesteinLength(1));
        Assert.Equal(16, VulkanBackend.BluesteinLength(5));
        Assert.Equal(2048, VulkanBackend.BluesteinLength(896));
    }

    public static TheoryData<FftElementType, int[]> EngineCases() => new()
    {
        { FftElementType.Float32, new[] { 10 } },          // one row of 5 (Bluestein)
        { FftElementType.Float32, new[] { 3, 16 } },       // three rows of 8 (radix-2)
        { FftElementType.Float32, new[] { 2, 3, 14 } },    // six rows of 7
        { FftElementType.Float16, new[] { 4, 12 } },
        { FftElementType.BFloat16, new[] { 4, 12 } },
    };

    [SkippableTheory]
    [MemberData(nameof(EngineCases))]
    public void IEngineContract_RunsOnTheDevice_WithoutHostFallback(FftElementType type, int[] shape)
    {
        var backend = BackendOrSkip();
        var engine = Engine(backend);
        Skip.IfNot(engine.SupportsFftElementType(type), $"Vulkan {type} FFT storage is unavailable.");

        var input = new Tensor<float>(shape);
        var rng = new Random(77 + shape.Length);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        int n = shape[shape.Length - 1] / 2, batch = input.Length / (2 * n);
        var complexInput = new Complex[batch * n];
        for (int i = 0; i < complexInput.Length; i++) complexInput[i] = new Complex(input[2 * i], input[2 * i + 1]);

        foreach (bool inverse in new[] { false, true })
        {
            long readbacks = GpuLaunchProbe.Readbacks;
            int fallbacks = GpuLaunchProbe.Fallbacks.Count();
            Tensor<float> output = ((IEngine)engine).FftGeneric(input, inverse, type);
            // The transform itself reads nothing back and records no fallback; the assertions below do the first read.
            Assert.Equal(readbacks, GpuLaunchProbe.Readbacks);
            Assert.Equal(fallbacks, GpuLaunchProbe.Fallbacks.Count());
            Assert.Equal(shape, output._shape);

            var actual = new Complex[batch * n];
            for (int i = 0; i < actual.Length; i++) actual[i] = new Complex(output[2 * i], output[2 * i + 1]);
            // Against the unquantized input, so the tolerance also covers one narrowing of the input.
            double error = RelErr(actual, DftBatched(complexInput, batch, n, inverse));
            Assert.True(error < Tolerance(type), $"{type} {string.Join("x", shape)} inverse={inverse}: rel-err {error:E3}");
        }
    }

    [SkippableFact]
    public void PublicFft_TakesTheDevicePath_ForNonPowerOfTwoLengths()
    {
        var backend = BackendOrSkip();
        var engine = Engine(backend);
        Skip.IfNot(engine.SupportsFftElementType(FftElementType.Float32), "Vulkan Float32 FFT storage is unavailable.");
        var input = new Tensor<float>(new[] { 2, 12 });   // two rows of 6 complex values
        var rng = new Random(5);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        Tensor<float>? output = engine.TryBackendFft(input, inverse: false);
        Assert.NotNull(output);
        var x = new Complex[12];
        var y = new Complex[12];
        for (int i = 0; i < 12; i++) { x[i] = new Complex(input[2 * i], input[2 * i + 1]); y[i] = new Complex(output![2 * i], output[2 * i + 1]); }
        Assert.True(RelErr(y, DftBatched(x, 2, 6, inverse: false)) < 2e-5);
    }

    [SkippableFact]
    public void FftGeneric_RecordsAutograd()
    {
        var backend = BackendOrSkip();
        var engine = Engine(backend);
        Skip.IfNot(engine.SupportsFftElementType(FftElementType.Float32), "Vulkan Float32 FFT storage is unavailable.");

        var input = new Tensor<float>(new float[] { 1, 0, 2, -1, 3, 0.5f }, new[] { 6 });
        using var tape = new GradientTape<float>();
        Tensor<float> result = ((IEngine)engine).FftGeneric(input);
        Assert.NotNull(result.GradFn);
        Tensor<float> loss = engine.ReduceSum(result, new[] { 0 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss);
        Assert.True(gradients.ContainsKey(input), "Vulkan FftGeneric must keep the active tape connected to its input.");
    }
}
