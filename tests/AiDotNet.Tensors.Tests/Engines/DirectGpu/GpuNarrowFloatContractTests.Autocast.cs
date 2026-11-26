// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NET6_0_OR_GREATER

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

// Matmul autocast on every backend: the device rounds 32-bit floats to a 16-bit float type exactly as the host
// conversion does, and under an open scope every GEMM entry point multiplies operands rounded that way.
public abstract partial class GpuNarrowFloatContractTests
{
    // Values that exercise the rounding: ties (both ways to even), the overflow threshold, subnormals of the narrow
    // type, values below its smallest subnormal, signed zeros and infinities, and ordinary values.
    private static float[] RoundingInputs(GpuElementType type, int length, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        float[] special =
        {
            0f, -0f, 1f, -1f, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue, -float.MaxValue, float.Epsilon,
            1.0004883f, 1.0014648f, 1.00390625f, 1.01171875f,   // ties of FP16 (2^-11 steps) and BF16 (2^-8 steps) at 1
            65504f, 65519.996f, 65520f, 65536f,                 // FP16's largest finite value and its overflow tie
            3.3895314e38f, 3.3961775e38f, 3.4e38f,              // BF16's largest finite value and its overflow tie
            5.9604645e-8f, 2.9802322e-8f, 2.9802326e-8f, 8.940697e-8f, 6.0975552e-5f,   // FP16 subnormals and ties
            1e-39f, 9.18355e-41f, 4.591775e-41f, 4.5917755e-41f, 1.4e-45f,              // BF16 subnormals and ties
        };
        var values = new float[length];
        for (int i = 0; i < length; i++)
        {
            if (i % 5 == 2) values[i] = special[(i / 5) % special.Length];
            else if (i % 5 == 4) values[i] = BitConverter.Int32BitsToSingle(random.Next());   // any positive pattern
            else values[i] = (float)((random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-6, 6)));
            if (float.IsNaN(values[i])) values[i] = 1f;
        }

        return values;
    }

    [SkippableFact]
    public void Round_kernel_matches_the_host_conversion_bit_for_bit()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        var rounding = (IGpuNarrowFloatRounding)backend;
        int ran = 0;
        foreach (var type in Types)
        {
            if (!rounding.SupportsRoundToNarrowFloat(type)) continue;
            foreach (int size in Sizes)
            {
                var input = RoundingInputs(type, size, 11 + size);
                using var source = backend.AllocateBuffer(input);
                using var destination = backend.AllocateBuffer(size);
                rounding.RoundToNarrowFloat(type, source, destination, size);
                var output = new float[size];
                backend.DownloadBuffer(destination, output);
                for (int i = 0; i < size; i++)
                {
                    float expected = Widen(type, Narrow(type, input[i]));
                    Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(output[i]),
                        $"{type} round of {input[i]:R} (element {i} of {size}): host {expected:R}, device {output[i]:R}.");
                }

                // In place: the destination may be the source.
                rounding.RoundToNarrowFloat(type, source, source, size);
                backend.DownloadBuffer(source, output);
                for (int i = 0; i < size; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(Widen(type, Narrow(type, input[i]))) == BitConverter.SingleToInt32Bits(output[i]),
                        $"{type} in-place round of {input[i]:R} (element {i} of {size}): device {output[i]:R}.");
            }

            ran++;
        }

        RequireRan(ran, Types.Length, "rounding kernel");
    }

    // 1 + r/64 with |r| < 1: FP16 keeps 2^-10 steps and BF16 2^-7 near 1, so rounding moves most elements, and a
    // product of rounded operands is far (by many FP32 ulps) from the product of the operands as given.
    private static float[] Operand(int length, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)((random.Next(2) == 0 ? 1.0 : -1.0) * (1.0 + (random.NextDouble() - 0.5) / 32.0));
        return values;
    }

    private static float[] RoundAll(GpuElementType? type, float[] values)
        => type is { } t ? Array.ConvertAll(values, v => Widen(t, Narrow(t, v))) : values;

    // c[batch, m, n] = a[batch, m, k] b[batch, k, n] (b as [n, k] when transposed), in double.
    private static double[] Product(float[] a, float[] b, int batch, int m, int n, int k, bool transposedB)
    {
        var c = new double[batch * m * n];
        for (int s = 0; s < batch; s++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                        sum += (double)a[(s * m * k) + (i * k) + p]
                            * (transposedB ? b[(s * n * k) + (j * k) + p] : b[(s * k * n) + (p * n) + j]);
                    c[(s * m * n) + (i * n) + j] = sum;
                }

        return c;
    }

    private static double Distance(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        return worst;
    }

    private enum GemmEntry
    {
        Gemm,
        MatMul,
        MatMulTransposed,
        BatchedGemm,
        GemmBias,
        GemmBiasRelu,
    }

    private static float[] Run(IDirectGpuBackend backend, GemmEntry entry, float[] a, float[] b, float[] bias, int batch, int m, int n, int k)
    {
        using var bufferA = backend.AllocateBuffer(a);
        using var bufferB = backend.AllocateBuffer(b);
        var result = new float[batch * m * n];
        switch (entry)
        {
            case GemmEntry.MatMul:
            {
                using var c = backend.MatMul(bufferA, bufferB, m, n, k);
                backend.DownloadBuffer(c, result);
                return result;
            }

            case GemmEntry.GemmBias:
            case GemmEntry.GemmBiasRelu:
            {
                using var bufferBias = backend.AllocateBuffer(bias);
                using var c = entry == GemmEntry.GemmBias
                    ? backend.GemmBias(bufferA, bufferB, bufferBias, m, n, k)
                    : backend.GemmBiasRelu(bufferA, bufferB, bufferBias, m, n, k);
                backend.DownloadBuffer(c, result);
                return result;
            }

            default:
            {
                using var c = backend.AllocateBuffer(result.Length);
                if (entry == GemmEntry.Gemm) backend.Gemm(bufferA, bufferB, c, m, n, k);
                else if (entry == GemmEntry.MatMulTransposed) backend.MatMulTransposed(bufferA, bufferB, c, m, n, k);
                else backend.BatchedGemm(bufferA, bufferB, c, m, n, k, batch);
                backend.DownloadBuffer(c, result);
                return result;
            }
        }
    }

    private static double[] Expected(GemmEntry entry, GpuElementType? type, float[] a, float[] b, float[] bias, int batch, int m, int n, int k)
    {
        var product = Product(RoundAll(type, a), RoundAll(type, b), batch, m, n, k, entry == GemmEntry.MatMulTransposed);
        if (entry is GemmEntry.GemmBias or GemmEntry.GemmBiasRelu)
        {
            // The bias is not a GEMM operand: it is added as given.
            for (int i = 0; i < product.Length; i++)
            {
                product[i] += bias[i % n];
                if (entry == GemmEntry.GemmBiasRelu) product[i] = Math.Max(product[i], 0.0);
            }
        }

        return product;
    }

    [SkippableTheory]
    [InlineData(PrecisionMode.Float16, 5, 7, 9)]
    [InlineData(PrecisionMode.BFloat16, 5, 7, 9)]
    [InlineData(PrecisionMode.Float16, 33, 65, 130)]
    [InlineData(PrecisionMode.BFloat16, 64, 48, 256)]
    public void Matmul_autocast_rounds_the_operands_of_every_gemm_entry_point(PrecisionMode precision, int m, int n, int k)
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        Skip.If(backend is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "Not applicable: CUDA multiplies on cuBLAS's 16-bit kernels (CublasComputeTypeTests cover its autocast).");
        var type = precision == PrecisionMode.Float16 ? GpuElementType.Float16 : GpuElementType.BFloat16;
        var autocast = (IGemmAutocastBackend)backend;
        Skip.If(!((IGpuNarrowFloatRounding)backend).SupportsRoundToNarrowFloat(type), $"This device has no rounding kernel for {type}.");
        const int Batch = 3;
        // FP32 accumulation error of a k-term product of values near 1, with room for any summation order.
        double tolerance = 4e-6 * k;
        foreach (GemmEntry entry in Enum.GetValues(typeof(GemmEntry)))
        {
            int batch = entry == GemmEntry.BatchedGemm ? Batch : 1;
            var a = Operand(batch * m * k, 3);
            var b = Operand(batch * k * n, 5);
            var bias = Operand(n, 7);
            var plain = Expected(entry, null, a, b, bias, batch, m, n, k);
            var rounded = Expected(entry, type, a, b, bias, batch, m, n, k);

            // The control: the two expectations are told apart by far more than the tolerance.
            double apart = 0;
            for (int i = 0; i < plain.Length; i++) apart = Math.Max(apart, Math.Abs(plain[i] - rounded[i]));
            Assert.True(apart > 20 * tolerance, $"{entry}: the rounded and plain products differ by only {apart:E2}.");

            Assert.True(Distance(plain, Run(backend, entry, a, b, bias, batch, m, n, k)) <= tolerance, $"{entry}: FP32 before any scope.");
            using (autocast.EnterGemmAutocast(precision, MatMulAutocastMode.CastOperands))
            {
                double off = Distance(rounded, Run(backend, entry, a, b, bias, batch, m, n, k));
                Assert.True(off <= tolerance, $"{entry} under {precision} autocast is {off:E2} from the product of rounded operands.");

                // An FP32 scope inside turns the cast off; closing it restores the outer one.
                using (autocast.EnterGemmAutocast(PrecisionMode.Float32, MatMulAutocastMode.CastOperands))
                    Assert.True(Distance(plain, Run(backend, entry, a, b, bias, batch, m, n, k)) <= tolerance, $"{entry}: FP32 scope inside.");
                Assert.True(Distance(rounded, Run(backend, entry, a, b, bias, batch, m, n, k)) <= tolerance, $"{entry}: outer scope restored.");
            }

            // The operands are untouched by the cast, and the GEMM is FP32 again once the scope closes.
            Assert.True(Distance(plain, Run(backend, entry, a, b, bias, batch, m, n, k)) <= tolerance, $"{entry}: FP32 after the scope.");

            // This device has no 16-bit compute type to ask for: the mode named after it rounds the operands too.
            using (autocast.EnterGemmAutocast(precision, MatMulAutocastMode.FastComputeType))
                Assert.True(Distance(rounded, Run(backend, entry, a, b, bias, batch, m, n, k)) <= tolerance, $"{entry}: fast compute type.");
        }
    }

    [SkippableFact]
    public void Matmul_autocast_scopes_close_in_any_order()
    {
        var backend = Device();
        Skip.If(backend is null, SkipReason);
        Skip.If(backend is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Not applicable: CUDA multiplies on cuBLAS's 16-bit kernels (CublasComputeTypeTests cover its autocast).");
        var rounding = (IGpuNarrowFloatRounding)backend;
        Skip.If(!rounding.SupportsRoundToNarrowFloat(GpuElementType.Float16) || !rounding.SupportsRoundToNarrowFloat(GpuElementType.BFloat16),
            "Needs both rounding kernels.");
        var autocast = (IGemmAutocastBackend)backend;
        const int M = 6, N = 5, K = 40;
        var a = Operand(M * K, 21);
        var b = Operand(K * N, 22);
        var bias = Array.Empty<float>();
        double tolerance = 4e-6 * K;
        double Off(GpuElementType? type)
            => Distance(Expected(GemmEntry.Gemm, type, a, b, bias, 1, M, N, K), Run(backend, GemmEntry.Gemm, a, b, bias, 1, M, N, K));

        var half = autocast.EnterGemmAutocast(PrecisionMode.Float16, MatMulAutocastMode.CastOperands);
        var brain = autocast.EnterGemmAutocast(PrecisionMode.BFloat16, MatMulAutocastMode.CastOperands);
        try
        {
            Assert.True(Off(GpuElementType.BFloat16) <= tolerance, "The newest scope decides.");
            half.Dispose();
            Assert.True(Off(GpuElementType.BFloat16) <= tolerance, "Closing an older scope leaves the newest in force.");
            brain.Dispose();
            Assert.True(Off(null) <= tolerance, "FP32 once every scope is closed.");
            brain.Dispose();
            Assert.True(Off(null) <= tolerance, "A second dispose changes nothing.");
        }
        finally
        {
            half.Dispose();
            brain.Dispose();
        }
    }
}

/// <summary>OpenCL constructed directly, so the contract runs on it whichever engine the process picked.</summary>
[Xunit.Collection("DirectGpuSerial")]
public sealed class OpenClNarrowFloatTests : GpuNarrowFloatContractTests, System.IDisposable
{
    private readonly AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend? _backend;

    public OpenClNarrowFloatTests()
    {
        try
        {
            var backend = new AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend();
            if (backend.IsAvailable) _backend = backend;
            else backend.Dispose();
        }
        catch (System.Exception unavailable) when (unavailable is System.DllNotFoundException
            or System.InvalidOperationException or System.NotSupportedException or System.TypeInitializationException)
        {
            System.Diagnostics.Trace.TraceInformation("OpenCL is not available on this host: {0}", unavailable);
            _backend = null;
        }
    }

    protected override AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend? Device() => _backend;

    protected override string SkipReason => "No OpenCL device on this host.";

    public void Dispose() => _backend?.Dispose();
}

#endif
