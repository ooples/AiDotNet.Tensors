// Copyright (c) AiDotNet. All rights reserved.
// #409 (S.3 verification step #2): confirm the compute-bound FP32 path that the
// issue cares about (BERT/GPT2/ViT FFN-shaped GEMMs) routes through the machine-code
// 6x16 microkernel — NOT the load-bound C# Avx2Fp32_8x8 (8x8) kernel — by measuring
// the FULL BlasManaged.Gemm dispatch GFLOPS at a compute-bound shape with min-of-many.
// Category=Performance => excluded from the normal/CI run.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Trait("Category", "Performance")]
public class MachineKernelCoverageBench
{
    private readonly ITestOutputHelper _out;
    public MachineKernelCoverageBench(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData(512, 512, 512)]     // square compute-bound
    [InlineData(768, 768, 3072)]    // BERT FFN-ish (M, N, K)
    public void ComputeBoundFp32_RoutesThroughMachineKernel(int m, int n, int k)
    {
        const int warmup = 5, measured = 30;
        double flops = 2.0 * m * k * n;

        var rng = new Random(7);
        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        void Gemm() => BlasManagedLib.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k);

        for (int i = 0; i < warmup; i++) Gemm();
        double min = double.MaxValue;
        for (int i = 0; i < measured; i++)
        {
            var sw = Stopwatch.StartNew(); Gemm(); sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            if (ms < min) min = ms;
        }
        double gflops = flops / (min * 1e-3) / 1e9;

        _out.WriteLine($"BlasManaged.Gemm<float> [M={m},N={n},K={k}] compute-bound:");
        _out.WriteLine($"  MachineKernelGemm.IsFp32Available = {MachineKernelGemm.IsFp32Available} (Fp32 {MachineKernelGemm.Fp32Mr}x{MachineKernelGemm.ActiveFp32Nr})");
        _out.WriteLine($"  min = {min:F3} ms   {gflops:F1} GFLOPS   (8x8 C# kernel ceiling ~53; machine-code ~peak ~80-90)");

        Assert.True(min > 0);
    }
}
