#if NET6_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IFusedDecodeLinearBackend"/> (#974) on every backend against a CPU oracle that does what each route
/// documents: output-major weights, tanh-GELU, FP16 operands with FP32 accumulation, and exact int32 accumulation
/// for the quantized route. Feature counts include ones that are not a multiple of the 32-bit words the shared
/// kernels read FP16 and int8 operands through.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class FusedDecodeLinearBackendParityTests
{
    // CUDA's direct-PTX specializations use tanh.approx; the routes document 5e-4 for them. The shared kernels
    // differ from the oracle only by FP32 accumulation and the device's tanh.
    private const float CudaTolerance = 5e-4f;
    private const float PortableTolerance = 2e-5f;

    private static double Gelu(double x) => 0.5 * x * (1.0 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

    private static float[] Values(int count, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return a;
    }

    private static IGpuBuffer Bytes(IDirectGpuBackend backend, byte[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length);
        backend.UploadByteBuffer(buffer, data);
        return buffer;
    }

    private static IGpuBuffer Halves(IDirectGpuBackend backend, float[] values, out float[] rounded)
    {
        var bytes = new byte[values.Length * 2];
        rounded = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            var half = (Half)values[i];
            rounded[i] = (float)half;
            ushort bits = BitConverter.HalfToUInt16Bits(half);
            bytes[2 * i] = (byte)bits;
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return Bytes(backend, bytes);
    }

    private static void AssertClose(double[] expected, float[] actual, float tolerance, string what)
    {
        Assert.True(actual.Length >= expected.Length, $"{what}: {actual.Length} outputs for {expected.Length} expected");
        for (int i = 0; i < expected.Length; i++)
        {
            double scale = Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * scale, $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    private static double[] Oracle(float[] x, float[] w, float[] bias, int rows, int k, int n)
    {
        var y = new double[rows * n];
        for (int m = 0; m < rows; m++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int i = 0; i < k; i++) s += (double)x[m * k + i] * w[j * k + i];
                y[m * n + j] = Gelu(s + bias[j]);
            }
        return y;
    }

    private static void Check(IDirectGpuBackend backend)
        => Check(backend, Assert.IsAssignableFrom<IFusedDecodeLinearBackend>(backend));

    private static void Check(IDirectGpuBackend backend, IFusedDecodeLinearBackend route)
    {
        float tolerance = route is CudaBackend ? CudaTolerance : PortableTolerance;

        // (inFeatures, outFeatures): odd and even counts, a single feature each way, and a decode-sized projection.
        foreach (var (k, n) in new[] { (1, 1), (7, 5), (33, 3), (64, 16), (129, 31), (768, 96) })
        {
            // ---- FP32
            float[] x = Values(k, 11 + k), w = Values(n * k, 13 + n, 0.25f), bias = Values(n, 17 + k + n, 0.5f);
            using (var dx = backend.AllocateBuffer(x))
            using (var dw = backend.AllocateBuffer(w))
            using (var db = backend.AllocateBuffer(bias))
            using (var dy = backend.AllocateBuffer(n))
            {
                backend.Fill(dy, float.NaN, n);
                route.FusedLinearGELUTransposedM1(dx, dw, db, dy, k, n);
                AssertClose(Oracle(x, w, bias, 1, k, n), backend.DownloadBuffer(dy), tolerance, $"FP32 M1 K={k} N={n}");
            }

            // ---- FP16, one row and sixteen. The oracle uses the FP16-rounded operands, as the routes do.
            foreach (int rows in new[] { 1, 16 })
            {
                float[] xr = Values(rows * k, 19 + k + rows);
                using var dxh = Halves(backend, xr, out float[] xHalf);
                using var dwh = Halves(backend, w, out float[] wHalf);
                using var db = backend.AllocateBuffer(bias);
                using var dy = backend.AllocateBuffer(rows * n);
                backend.Fill(dy, float.NaN, rows * n);
                try
                {
                    if (rows == 1) route.FusedLinearGELUFp16TransposedM1(dxh, dwh, db, dy, k, n);
                    else route.FusedLinearGELUFp16TransposedM16(dxh, dwh, db, dy, k, n);
                }
                catch (NotSupportedException) when (route is CudaBackend)
                {
                    continue; // a CUDA device below the routes' compute capability
                }
                AssertClose(Oracle(xHalf, wHalf, bias, rows, k, n), backend.DownloadBuffer(dy), tolerance, $"FP16 M{rows} K={k} N={n}");
            }

            // ---- W8A8: every byte value appears, the extremes included.
            var rng = new Random(23 + k * 3 + n);
            var x8 = new byte[k];
            var w8 = new byte[n * k];
            rng.NextBytes(x8);
            rng.NextBytes(w8);
            x8[0] = 0x80;               // -128
            w8[0] = 0x80;
            if (k > 1) { x8[k - 1] = 0x7F; w8[n * k - 1] = 0x7F; }
            float[] actScale = { 0.0123f };
            float[] wScales = Values(n, 29 + n, 0.01f).Select(v => Math.Abs(v) + 0.001f).ToArray();
            var accumulator = new int[n];
            var expected = new double[n];
            for (int j = 0; j < n; j++)
            {
                int acc = 0;
                for (int i = 0; i < k; i++) acc += (sbyte)x8[i] * (sbyte)w8[j * k + i];
                accumulator[j] = acc;
                expected[j] = Gelu((double)(float)((float)acc * actScale[0] * wScales[j] + bias[j]));
            }
            using (var dx8 = Bytes(backend, x8))
            using (var dw8 = Bytes(backend, w8))
            using (var dScale = backend.AllocateBuffer(actScale))
            using (var dwScales = backend.AllocateBuffer(wScales))
            using (var db = backend.AllocateBuffer(bias))
            using (var dy = backend.AllocateBuffer(n))
            {
                backend.Fill(dy, float.NaN, n);
                route.FusedLinearGELUW8A8TransposedM1(dx8, dw8, dScale, dwScales, db, dy, k, n);
                AssertClose(expected, backend.DownloadBuffer(dy), tolerance, $"W8A8 M1 K={k} N={n}");

                // ---- the epilogue alone, over the oracle's accumulator
                var accBytes = new byte[n * sizeof(int)];
                Buffer.BlockCopy(accumulator, 0, accBytes, 0, accBytes.Length);
                using var dAcc = Bytes(backend, accBytes);
                backend.Fill(dy, float.NaN, n);
                route.W8A8DequantBiasGelu(dAcc, dScale, dwScales, db, dy, n);
                AssertClose(expected, backend.DownloadBuffer(dy), tolerance, $"W8A8 epilogue N={n}");
            }
        }

        CheckHalfDecoding(backend, route, tolerance);
        CheckInt32Accumulation(backend, route, tolerance);
    }

    // One weight per output, input 1.0: each output is GELU of one FP16 value, so the kernel's half decoding is
    // visible element by element across the format (zero, subnormals, the extremes, both signs).
    private static void CheckHalfDecoding(IDirectGpuBackend backend, IFusedDecodeLinearBackend route, float tolerance)
    {
        float[] specials =
        {
            0f, 1f, -1f, 0.5f, -2.5f, 6.1035156e-5f /* smallest normal */, 5.9604645e-8f /* smallest subnormal */,
            -3.0e-5f /* a subnormal */, 1023.5f, -65504f /* most negative finite half */, 65504f, 3.1415927f, -0.33325195f,
        };
        int n = specials.Length;
        using var dx = Halves(backend, new[] { 1f }, out _);
        using var dw = Halves(backend, specials, out float[] rounded);
        using var db = backend.AllocateBuffer(new float[n]);
        using var dy = backend.AllocateBuffer(n);
        backend.Fill(dy, float.NaN, n);
        try
        {
            route.FusedLinearGELUFp16TransposedM1(dx, dw, db, dy, 1, n);
        }
        catch (NotSupportedException) when (route is CudaBackend)
        {
            return;
        }
        AssertClose(rounded.Select(v => Gelu(v)).ToArray(), backend.DownloadBuffer(dy), tolerance, "FP16 values");
    }

    // 4,096 products of -128 * -128: the accumulator reaches 67,108,864, far past the 2^24 where an FP32 running sum
    // stops being exact. The scales are chosen so the dequantized value is 1 + bias, which only an exact integer sum gives.
    private static void CheckInt32Accumulation(IDirectGpuBackend backend, IFusedDecodeLinearBackend route, float tolerance)
    {
        const int k = 4096, n = 3;
        var x8 = Enumerable.Repeat((byte)0x80, k).ToArray();
        var w8 = Enumerable.Repeat((byte)0x80, n * k).ToArray();
        w8[2 * k + 5] = 0x7F; // one +127 in the last row: -128 * 127 instead of -128 * -128
        int full = k * 128 * 128;
        int last = full - 128 * 128 - 128 * 127;
        float[] actScale = { 1f / 8192f };
        float[] wScales = { 1f / 8192f, 2f / 8192f, 1f / 8192f };
        float[] bias = { 0.25f, -0.5f, 0f };
        double[] expected =
        {
            Gelu((double)(float)((float)full * actScale[0] * wScales[0] + bias[0])),
            Gelu((double)(float)((float)full * actScale[0] * wScales[1] + bias[1])),
            Gelu((double)(float)((float)last * actScale[0] * wScales[2] + bias[2])),
        };
        using var dx8 = Bytes(backend, x8);
        using var dw8 = Bytes(backend, w8);
        using var dScale = backend.AllocateBuffer(actScale);
        using var dwScales = backend.AllocateBuffer(wScales);
        using var db = backend.AllocateBuffer(bias);
        using var dy = backend.AllocateBuffer(n);
        backend.Fill(dy, float.NaN, n);
        route.FusedLinearGELUW8A8TransposedM1(dx8, dw8, dScale, dwScales, db, dy, k, n);
        AssertClose(expected, backend.DownloadBuffer(dy), tolerance, "W8A8 int32 accumulation");
    }

    [SkippableFact]
    public void Vulkan_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void OpenCl_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Cuda_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_FusedDecodeLinear_MatchesTheCpuOracle()
    {
        // The real backend, executing on Dawn (see DawnHostedWebGpu). A freshly constructed WebGpuBackend is never
        // available outside a browser, so the old form of this test could only skip.
        Check(DawnHostedWebGpu.BackendOrSkip());
    }

    // The shared dl_p_ kernels through a launcher (the WebGPU WGSL ports on Dawn), with buffers on the host backend.
    private sealed class LaunchedFusedDecodeLinear : IFusedDecodeLinearBackend
    {
        private readonly IDirectGpuBackend _host;
        private readonly PortableGenericFft.Launcher _launch;
        public LaunchedFusedDecodeLinear(IDirectGpuBackend host, PortableGenericFft.Launcher launch) { _host = host; _launch = launch; }
        public void FusedLinearGELUTransposedM1(IGpuBuffer input, IGpuBuffer outputMajorWeight, IGpuBuffer bias, IGpuBuffer output,
            int inFeatures, int outFeatures)
            => PortableFusedDecodeLinear.Linear(_launch, input, outputMajorWeight, bias, output, 1, inFeatures, outFeatures);
        public void FusedLinearGELUFp16TransposedM1(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat,
            IGpuBuffer outputFloat, int inFeatures, int outFeatures)
            => PortableFusedDecodeLinear.LinearFp16(_host, _launch, inputHalf, outputMajorWeightHalf, biasFloat, outputFloat, 1, inFeatures, outFeatures);
        public void FusedLinearGELUFp16TransposedM16(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat,
            IGpuBuffer outputFloat, int inFeatures, int outFeatures)
            => PortableFusedDecodeLinear.LinearFp16Rows16(_host, _launch, inputHalf, outputMajorWeightHalf, biasFloat, outputFloat, inFeatures, outFeatures);
        public void FusedLinearGELUW8A8TransposedM1(IGpuBuffer inputInt8, IGpuBuffer outputMajorWeightsInt8, IGpuBuffer activationScaleFloat,
            IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int inputFeatures, int outputFeatures)
            => PortableFusedDecodeLinear.LinearW8A8(_host, _launch, inputInt8, outputMajorWeightsInt8, activationScaleFloat, weightScalesFloat,
                biasFloat, outputFloat, inputFeatures, outputFeatures);
        public void W8A8DequantBiasGelu(IGpuBuffer accumulatorInt32, IGpuBuffer activationScaleFloat, IGpuBuffer weightScalesFloat,
            IGpuBuffer biasFloat, IGpuBuffer outputFloat, int outputFeatures)
            => PortableFusedDecodeLinear.W8A8Epilogue(_launch, accumulatorInt32, activationScaleFloat, weightScalesFloat, biasFloat, outputFloat, outputFeatures);
    }

    [SkippableFact]
    public void WebGpuWgsl_FusedDecodeLinear_MatchesTheCpuOracle_OnDawn()
    {
        using var dawn = DawnWgslLauncher.TryCreate();
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        using var host = new OpenClBackend();
        Skip.IfNot(host.IsAvailable, "OpenCL hosts the buffers and is not available.");
        Check(host, new LaunchedFusedDecodeLinear(host, dawn!.For(host)));
    }
#endif

    // Whole-word operands (every Vulkan byte buffer is one): each route is device work only. The scope counts this
    // thread's transfers, so other test classes running in parallel do not land in it.
    [SkippableFact]
    public void Vulkan_EveryRoute_KeepsItsOperandsOnTheDevice()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        var route = (IFusedDecodeLinearBackend)backend;
        const int k = 33, n = 5; // an odd count of halves and an int8 count that is not a multiple of four
        float[] bias = Values(n, 3);
        using var dx = backend.AllocateBuffer(Values(k, 1));
        using var dw = backend.AllocateBuffer(Values(n * k, 2));
        using var dxh = Halves(backend, Values(16 * k, 4), out _);
        using var dwh = Halves(backend, Values(n * k, 5), out _);
        using var dx8 = Bytes(backend, new byte[k]);
        using var dw8 = Bytes(backend, new byte[n * k]);
        using var dAcc = Bytes(backend, new byte[n * sizeof(int)]);
        using var dScale = backend.AllocateBuffer(new[] { 0.5f });
        using var dwScales = backend.AllocateBuffer(Values(n, 6));
        using var db = backend.AllocateBuffer(bias);
        using var dy = backend.AllocateBuffer(16 * n);

        using var scope = AiDotNet.Tensors.Engines.Diagnostics.GpuResidencyScope.Begin();
        route.FusedLinearGELUTransposedM1(dx, dw, db, dy, k, n);
        route.FusedLinearGELUFp16TransposedM1(dxh, dwh, db, dy, k, n);
        route.FusedLinearGELUFp16TransposedM16(dxh, dwh, db, dy, k, n);
        route.FusedLinearGELUW8A8TransposedM1(dx8, dw8, dScale, dwScales, db, dy, k, n);
        route.W8A8DequantBiasGelu(dAcc, dScale, dwScales, db, dy, n);
        Assert.Equal(0, scope.Downloads);
        Assert.Equal(0, scope.Uploads);
    }

    [SkippableFact]
    public void ShortBuffers_AndNonPositiveExtents_AreRejectedBeforeAnyDispatch()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        var route = (IFusedDecodeLinearBackend)backend;
        using var four = backend.AllocateBuffer(4);
        using var one = backend.AllocateBuffer(1);
        // weights [N = 2, K = 4] need 8 floats; `four` holds 4.
        Assert.Throws<ArgumentException>(() => route.FusedLinearGELUTransposedM1(four, four, four, four, 4, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => route.FusedLinearGELUTransposedM1(four, four, four, four, 0, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => route.FusedLinearGELUFp16TransposedM1(four, four, four, four, 4, -1));
        // output [16, N = 1] needs 16 floats.
        Assert.Throws<ArgumentException>(() => route.FusedLinearGELUFp16TransposedM16(four, four, four, four, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => route.FusedLinearGELUW8A8TransposedM1(
            four, four, one, four, four, four, PortableFusedDecodeLinear.MaxW8A8InputFeatures + 1, 1));
        Assert.Throws<ArgumentException>(() => route.W8A8DequantBiasGelu(one, one, four, four, four, 4));
    }

    [Fact]
    public void TheQuantizedRoute_RejectsADotProductLongEnoughToOverflowInt32()
    {
        Assert.Equal(131_071, PortableFusedDecodeLinear.MaxW8A8InputFeatures);
        // 131,071 products of magnitude 16,384 stay below 2^31; one more does not.
        Assert.True(131_071L * 128 * 128 < int.MaxValue);
        Assert.True(131_072L * 128 * 128 > int.MaxValue);
    }
}
#endif
