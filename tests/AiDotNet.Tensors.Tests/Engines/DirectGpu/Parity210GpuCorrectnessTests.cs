// Copyright (c) AiDotNet. All rights reserved.
// GPU↔CPU parity tests for the Issue #210 hot-path op surface. Each test
// runs an op through DirectGpuTensorEngine (which routes to the active
// GPU backend via IParity210Backend) and compares the result to the
// CpuEngine reference element-by-element. Tests are skipped with a
// documented reason when no GPU backend is available — they act as
// compile-time-only guards on CI runners without GPUs.

using System;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class Parity210GpuCorrectnessTests
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;
    private readonly CpuEngine _cpu = new();
    private const float Tolerance = 1e-4f;

    public Parity210GpuCorrectnessTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _available = _gpu.IsGpuAvailable;
        }
        catch
        {
            _available = false;
        }
    }

    private static Tensor<float> Randn(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var d in shape) total *= d;
        var arr = new float[total];
        for (int i = 0; i < total; i++)
            arr[i] = (float)(rng.NextDouble() * 4 - 2);
        return new Tensor<float>(arr, shape);
    }

    private static bool Close(float a, float b, float tol)
        => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    private static void AssertClose(Tensor<float> gpu, Tensor<float> cpu, float tol = Tolerance)
    {
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        var g = gpu.GetDataArray();
        var c = cpu.GetDataArray();
        for (int i = 0; i < g.Length; i++)
        {
            if (!Close(g[i], c[i], tol))
                throw new Xunit.Sdk.XunitException(
                    $"GPU vs CPU mismatch at index {i}: gpu={g[i]}, cpu={c[i]}, tol={tol}");
        }
    }

    // ---------------------------------------------------------------------
    // Unary special (erfc / lgamma / digamma / erfinv / I0 / I1 / I0e / I1e)
    // ---------------------------------------------------------------------

    [SkippableFact]
    public void Erfc_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(42, 64);
        var g = _gpu!.TensorErfc(x);
        var c = _cpu.TensorErfc(x);
        AssertClose(g, c, 5e-4f);
    }

    [SkippableFact]
    public void Lgamma_GpuMatchesCpu_PositiveInputs()
    {
        Skip.If(!_available, "GPU backend not available");
        // lgamma requires positive arguments; shift Randn into (0.1, 10).
        var src = Randn(1, 64).GetDataArray();
        for (int i = 0; i < src.Length; i++) src[i] = MathF.Abs(src[i]) + 0.1f;
        var x = new Tensor<float>(src, new[] { 64 });
        var g = _gpu!.TensorLgamma(x);
        var c = _cpu.TensorLgamma(x);
        AssertClose(g, c, 5e-3f);  // lgamma has higher precision drift
    }

    [SkippableFact]
    public void I0_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(7, 32);
        var g = _gpu!.TensorI0(x);
        var c = _cpu.TensorI0(x);
        AssertClose(g, c, 1e-3f);
    }

    [SkippableFact]
    public void Erfinv_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        // Input must be in (-1, 1).
        var src = new float[32];
        var rng = new Random(3);
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 1.8 - 0.9);
        var x = new Tensor<float>(src, new[] { 32 });
        var g = _gpu!.TensorErfinv(x);
        var c = _cpu.TensorErfinv(x);
        AssertClose(g, c, 5e-3f);
    }

    // ---------------------------------------------------------------------
    // Binary special
    // ---------------------------------------------------------------------

    [SkippableFact]
    public void Hypot_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var a = Randn(10, 128);
        var b = Randn(11, 128);
        var g = _gpu!.TensorHypot(a, b);
        var c = _cpu.TensorHypot(a, b);
        AssertClose(g, c);
    }

    [SkippableFact]
    public void LogAddExp_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var a = Randn(20, 128);
        var b = Randn(21, 128);
        var g = _gpu!.TensorLogAddExp(a, b);
        var c = _cpu.TensorLogAddExp(a, b);
        AssertClose(g, c);
    }

    // ---------------------------------------------------------------------
    // Movement
    // ---------------------------------------------------------------------

    [SkippableFact]
    public void Triu_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(30, 8, 8);
        var g = _gpu!.TensorTriu(x, diagonal: 0);
        var c = _cpu.TensorTriu(x, diagonal: 0);
        AssertClose(g, c, 0f);  // exact — just a mask
    }

    [SkippableFact]
    public void Tril_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(31, 6, 8);
        var g = _gpu!.TensorTril(x, diagonal: 1);
        var c = _cpu.TensorTril(x, diagonal: 1);
        AssertClose(g, c, 0f);
    }

    [SkippableFact]
    public void Flip_GpuMatchesCpu_SingleAxis()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(40, 6, 4);
        var g = _gpu!.TensorFlip(x, new[] { 1 });
        var c = _cpu.TensorFlip(x, new[] { 1 });
        AssertClose(g, c, 0f);
    }

    [SkippableFact]
    public void Roll_GpuMatchesCpu_SingleAxis()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(50, 4, 6);
        var g = _gpu!.TensorRoll(x, new[] { 2 }, new[] { 1 });
        var c = _cpu.TensorRoll(x, new[] { 2 }, new[] { 1 });
        AssertClose(g, c, 0f);
    }

    [SkippableFact]
    public void DiagEmbed_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(60, 4, 5);
        var g = _gpu!.TensorDiagEmbed(x, offset: 0);
        var c = _cpu.TensorDiagEmbed(x, offset: 0);
        AssertClose(g, c, 0f);
    }

    // ---------------------------------------------------------------------
    // Cumulative
    // ---------------------------------------------------------------------

    [SkippableFact]
    public void CumSum_GpuMatchesCpu_LastAxis()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(70, 8, 64);
        var g = _gpu!.TensorCumSum(x, axis: -1);
        var c = _cpu.TensorCumSum(x, axis: -1);
        AssertClose(g, c, 1e-3f);
    }

    [SkippableFact]
    public void CumMax_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(71, 4, 32);
        var g = _gpu!.TensorCumMax(x, axis: -1);
        var c = _cpu.TensorCumMax(x, axis: -1);
        AssertClose(g, c, 0f);
    }

    [SkippableFact]
    public void LogCumSumExp_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Randn(72, 4, 32);
        var g = _gpu!.TensorLogCumSumExp(x, axis: -1);
        var c = _cpu.TensorLogCumSumExp(x, axis: -1);
        AssertClose(g, c, 5e-3f);
    }

    // ---------------------------------------------------------------------
    // NaN-hygiene
    // ---------------------------------------------------------------------

    [SkippableFact]
    public void NanToNum_GpuMatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var arr = new float[] { 1f, float.NaN, 3f, float.PositiveInfinity, -2f, float.NegativeInfinity };
        var x = new Tensor<float>(arr, new[] { 6 });
        var g = _gpu!.TensorNanToNum(x);
        var c = _cpu.TensorNanToNum(x);
        AssertClose(g, c, 1e-3f);
    }
}
