// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA's fused GRU sequence (CudaBackend.FusedGru: one GEMM plus one register-resident recurrence launch per direction)
/// against the CPU engine: output, and every gradient by cosine and element-wise, over the hidden sizes of both kernel
/// widths (padded and exact) and one above them, which must take the portable kernels instead. Each case checks which
/// path ran, so a fallback cannot pass for the fused kernels.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CudaFusedGruTests
{
    private static DirectGpuTensorEngine? TryCuda()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        return null;
    }

    private static Tensor<float> Rand(Random rng, double scale, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(a, shape);
    }

    private static (double[] Y, Dictionary<string, double[]> Grads) Train(IEngine e, Tensor<float>[] p, Tensor<float> projection)
    {
        using var tape = new GradientTape<float>();
        var y = e.GruSequenceForward(p[0], p[1], p[2], p[3], p[4], p[5], returnSequences: true, resetAfter: true);
        var loss = e.ReduceMean(e.TensorMultiply(y, projection), new[] { 0, 1, 2 }, keepDims: false);
        var g = tape.ComputeGradients(loss, p);
        var names = new[] { "x", "h0", "wIh", "wHh", "bIh", "bHh" };
        var grads = new Dictionary<string, double[]>();
        for (int i = 0; i < p.Length; i++) grads[names[i]] = Array.ConvertAll(g[p[i]].ToArray(), v => (double)v);
        return (Array.ConvertAll(y.ToArray(), v => (double)v), grads);
    }

    private static double Cosine(double[] a, double[] b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        return ab / Math.Sqrt(aa * bb);
    }

    private static void AssertClose(string name, double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double cosine = Cosine(expected, actual);
        Assert.True(cosine >= 0.99999, $"{name}: cosine {cosine} against the CPU.");
        // A cosine cannot see a uniform scale error (a doubled bias gradient, a missing r factor).
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 + 1e-3 * Math.Abs(expected[i]),
                $"{name}[{i}]: CPU {expected[i]}, CUDA {actual[i]}.");
    }

    private static void RunCase(DirectGpuTensorEngine gpu, int batch, int steps, int inputs, int hidden, bool expectFused)
    {
        var rng = new Random(batch * 1000 + hidden);
        double scale = 1.0 / Math.Sqrt(hidden);
        var p = new[]
        {
            Rand(rng, 1.0, batch, steps, inputs), Rand(rng, 0.5, batch, hidden),
            Rand(rng, scale, 3 * hidden, inputs), Rand(rng, scale, 3 * hidden, hidden),
            Rand(rng, scale, 3 * hidden), Rand(rng, scale, 3 * hidden),
        };
        var projection = Rand(rng, 1.0, batch, steps, hidden);
        var cpu = Train(new CpuEngine(), p, projection);
        long forward = CudaBackend.FusedGruForwardRuns, backward = CudaBackend.FusedGruBackwardRuns;
        var cuda = Train(gpu, p, projection);
        Assert.Equal(expectFused, CudaBackend.FusedGruForwardRuns > forward);
        Assert.Equal(expectFused, CudaBackend.FusedGruBackwardRuns > backward);
        AssertClose("y", cpu.Y, cuda.Y);
        foreach (var name in cpu.Grads.Keys) AssertClose(name, cpu.Grads[name], cuda.Grads[name]);
    }

    [SkippableTheory]
    [InlineData(3, 4, 5, 6)]       // the 32-wide kernel, padded
    [InlineData(32, 16, 32, 64)]   // the head-to-head GRU's shape: the 64-wide kernel, exact
    [InlineData(5, 7, 9, 40)]      // the 64-wide kernel, padded
    [InlineData(2, 3, 4, 80)]      // above the fused kernels: the portable ones
    public void TheFusedSequenceMatchesTheCpu(int batch, int steps, int inputs, int hidden)
    {
        using var gpu = TryCuda();
        Skip.If(gpu is null, "No CUDA backend.");
        RunCase(gpu!, batch, steps, inputs, hidden, expectFused: hidden <= 64);
    }

    [SkippableFact]
    public void TheSwitchSendsTheSequenceToThePortableKernels()
    {
        using var gpu = TryCuda();
        Skip.If(gpu is null, "No CUDA backend.");
        bool previous = CudaBackend.UseFusedGru;
        try
        {
            CudaBackend.UseFusedGru = false;
            RunCase(gpu!, 4, 5, 6, 32, expectFused: false);
        }
        finally
        {
            CudaBackend.UseFusedGru = previous;
        }
    }
}
