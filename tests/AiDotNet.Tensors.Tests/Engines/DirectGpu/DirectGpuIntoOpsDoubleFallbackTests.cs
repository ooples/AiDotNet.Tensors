using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU engine's destination-writing ops take a float-only GPU path. For any other element type they must fall
/// back to the CPU implementation. Eleven of them cast to Tensor&lt;float&gt; without checking the type, and their
/// fallback caught only NotSupportedException, so a double tensor threw InvalidCastException out of the engine. Found
/// by AiDotNet's UNet noise predictor in double, through GroupNormSwishInto.
/// </summary>
[Collection("DirectGpuSerial")]
public class DirectGpuIntoOpsDoubleFallbackTests
{
    private static DirectGpuTensorEngine RequireEngine()
    {
        DirectGpuTensorEngine engine;
        try
        {
            engine = new DirectGpuTensorEngine();
        }
        catch (GpuCapabilityUnavailableException)
        {
            Skip.If(true, "No GPU engine.");
            throw;
        }
        if (!engine.SupportsGpu)
        {
            engine.Dispose();
            Skip.If(true, "No GPU backend.");
        }
        return engine;
    }

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    // relativeTolerance 0 demands the CPU result bit for bit.
    private static void AssertSame(Tensor<double> expected, Tensor<double> actual, string op, double relativeTolerance)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= relativeTolerance * Math.Max(1.0, Math.Abs(expected[i])),
                $"{op}[{i}]: CPU {expected[i]}, GPU engine {actual[i]}.");
    }

    // Runs the op on the GPU engine and on the CPU engine and requires the same result: for double, the GPU
    // engine's answer is the CPU implementation's.
    private static void Check(string op, int[] outShape, Action<IEngine, Tensor<double>> run, double relativeTolerance = 0)
    {
        using var gpu = RequireEngine();
        var cpu = new CpuEngine();
        var fromGpu = new Tensor<double>(outShape);
        var fromCpu = new Tensor<double>(outShape);
        run(gpu, fromGpu);
        run(cpu, fromCpu);
        AssertSame(fromCpu, fromGpu, op, relativeTolerance);
    }

    [SkippableFact]
    public void Activations_and_softmax_fall_back_for_double()
    {
        var x = Random(new[] { 2, 3, 8 }, 1);
        int[] s = { 2, 3, 8 };
        Check("SoftmaxInto", s, (e, d) => e.SoftmaxInto(d, x, -1));
        Check("LogSoftmaxInto", s, (e, d) => e.LogSoftmaxInto(d, x, -1));
        Check("SwishInto", s, (e, d) => e.SwishInto(d, x));
        Check("GELUInto", s, (e, d) => e.GELUInto(d, x));
        Check("TanhInto", s, (e, d) => e.TanhInto(d, x));
        Check("MishInto", s, (e, d) => e.MishInto(d, x));
        Check("LeakyReLUInto", s, (e, d) => e.LeakyReLUInto(d, x, 0.1));
    }

    [SkippableFact]
    public void Concat_and_transpose_fall_back_for_double()
    {
        var a = Random(new[] { 2, 3, 4 }, 2);
        var b = Random(new[] { 2, 5, 4 }, 3);
        Check("ConcatInto", new[] { 2, 8, 4 }, (e, d) => e.ConcatInto(d, new[] { a, b }, 1));
        Check("TransposeInto", new[] { 4, 2, 3 }, (e, d) => e.TransposeInto(d, a, new[] { 2, 0, 1 }));
    }

    [SkippableFact]
    public void Group_norm_fusions_fall_back_for_double()
    {
        var x = Random(new[] { 2, 8, 4, 4 }, 4);
        var y = Random(new[] { 2, 8, 4, 4 }, 5);
        var gamma = Random(new[] { 8 }, 6);
        var beta = Random(new[] { 8 }, 7);
        int[] s = { 2, 8, 4, 4 };
        // The fallback composes the CPU group norm with the engine's own in-place activation and add, which the GPU
        // engine runs for every float-convertible type as an FP32 kernel by design ("every DirectGpu backend stores
        // public numeric tensors as FP32"). So these agree with the CPU engine to FP32 precision, not bit for bit;
        // before the fix they threw InvalidCastException instead.
        const double fp32 = 1e-6;
        Check("GroupNormSwishInto", s, (e, d) => e.GroupNormSwishInto(d, x, 4, gamma, beta, 1e-5), fp32);
        Check("AddGroupNormInto", s, (e, d) => e.AddGroupNormInto(d, x, y, 4, gamma, beta, 1e-5), fp32);
    }
}
