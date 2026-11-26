// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class SequenceMixerOpsParityTests
{
    private static Tensor<float> RandomTensor(Random rng, int[] shape, float scale = 1.0f)
    {
        int len = 1;
        foreach (int s in shape) len *= s;
        var data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = (float)((2.0 * rng.NextDouble() - 1.0) * scale);
        return new Tensor<float>(data, shape);
    }

    private static (Tensor<float> Cos, Tensor<float> Sin) BuildRopeTables(int maxSeq, int halfDim)
    {
        var cos = new float[maxSeq * halfDim];
        var sin = new float[maxSeq * halfDim];
        for (int t = 0; t < maxSeq; t++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                double freq = t * Math.Pow(10000.0, -(2.0 * i) / (2 * halfDim));
                cos[t * halfDim + i] = (float)Math.Cos(freq);
                sin[t * halfDim + i] = (float)Math.Sin(freq);
            }
        }
        return (new Tensor<float>(cos, new[] { maxSeq, halfDim }), new Tensor<float>(sin, new[] { maxSeq, halfDim }));
    }

    private static void AssertClose(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tol, string label)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.False(float.IsNaN(b[i]) || float.IsInfinity(b[i]), $"{label}[{i}] is non-finite: {b[i]}");
            Assert.True(MathF.Abs(a[i] - b[i]) <= tol, $"{label}[{i}] mismatch: expected {a[i]}, got {b[i]} (tol={tol})");
        }
    }

    [Theory]
    [InlineData(17, 32)]
    [InlineData(64, 256)]
    public void SplitHalfGivensRotation_CpuComposedAndGpuFusedAndPortable_Match(int rows, int dim)
    {
        var rng = new Random(101 + rows + dim);
        int half = dim / 2;
        var xRaw = RandomTensor(rng, new[] { rows, dim });
        var phiRaw = RandomTensor(rng, new[] { 1, 1, half });
        var dyRaw = RandomTensor(rng, new[] { rows, dim });

        var cpu = new CpuEngine();
        Tensor<float> yComp, yCpu;
        Dictionary<Tensor<float>, Tensor<float>> compGrads, cpuGrads;

        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var xC0 = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { rows, dim });
            var pC0 = new Tensor<float>(phiRaw.AsSpan().ToArray(), new[] { 1, 1, half });
            var dyC0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, dim });
            using (var tape = new GradientTape<float>())
            {
                yComp = CpuEngine.SplitHalfGivensRotationComposed(cpu, xC0, pC0, half);
                compGrads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dyC0), null), new[] { xC0, pC0 });
            }

            var xC1 = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { rows, dim });
            var pC1 = new Tensor<float>(phiRaw.AsSpan().ToArray(), new[] { 1, 1, half });
            var dyC1 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, dim });
            using (var tape = new GradientTape<float>())
            {
                yCpu = cpu.SplitHalfGivensRotation(xC1, pC1);
                cpuGrads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yCpu, dyC1), null), new[] { xC1, pC1 });
            }

            AssertClose(yComp.AsSpan(), yCpu.AsSpan(), 1e-5f, "Givens.CpuFwd");
            AssertClose(compGrads[xC0].AsSpan(), cpuGrads[xC1].AsSpan(), 2e-5f, "Givens.CpuDx");
            AssertClose(compGrads[pC0].AsSpan(), cpuGrads[pC1].AsSpan(), 1e-4f, "Givens.CpuDphi");
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var xG = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { rows, dim });
                var pG = new Tensor<float>(phiRaw.AsSpan().ToArray(), new[] { 1, 1, half });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, dim });
                using var tape = new GradientTape<float>();
                var yG = gpu.SplitHalfGivensRotation(xG, pG);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { xG, pG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 2e-5f, $"Givens.GpuFwd(fused={useFused})");
                AssertClose(yCpu.AsSpan(), yG.AsSpan(), 2e-5f, $"Givens.GpuVsCpu(fused={useFused})");
                AssertClose(compGrads.Values.GetEnumerator().Current is not null ? yCpu.AsSpan() : yCpu.AsSpan(), yCpu.AsSpan(), 0f, "noop");
                var enumComp = new List<Tensor<float>>(compGrads.Values);
                AssertClose(enumComp[0].AsSpan(), gGrads[xG].AsSpan(), 5e-5f, $"Givens.GpuDx(fused={useFused})");
                AssertClose(enumComp[1].AsSpan(), gGrads[pG].AsSpan(), 2e-4f, $"Givens.GpuDphi(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(2, 1, 32)]
    [InlineData(3, 19, 128)]
    public void DepthwiseCausalConv1d2Tap_CpuComposedAndGpuFusedAndPortable_Match(int b, int s, int c)
    {
        var rng = new Random(202 + b * 11 + s * 7 + c);
        var xRaw = RandomTensor(rng, new[] { b, s, c });
        var w0Raw = RandomTensor(rng, new[] { 1, 1, c });
        var w1Raw = RandomTensor(rng, new[] { 1, 1, c });
        var dyRaw = RandomTensor(rng, new[] { b, s, c });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dxComp, dw0Comp, dw1Comp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var x0 = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, c });
            var a0 = new Tensor<float>(w0Raw.AsSpan().ToArray(), new[] { 1, 1, c });
            var a1 = new Tensor<float>(w1Raw.AsSpan().ToArray(), new[] { 1, 1, c });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, c });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.DepthwiseCausalConv1d2TapComposed(cpu, x0, a0, a1, b, s, c);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { x0, a0, a1 });
            dxComp = grads[x0];
            dw0Comp = grads[a0];
            dw1Comp = grads.TryGetValue(a1, out var a1Grad) ? a1Grad : new Tensor<float>(new[] { 1, 1, c });
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var xG = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, c });
                var a0G = new Tensor<float>(w0Raw.AsSpan().ToArray(), new[] { 1, 1, c });
                var a1G = new Tensor<float>(w1Raw.AsSpan().ToArray(), new[] { 1, 1, c });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, c });
                using var tape = new GradientTape<float>();
                var yG = gpu.DepthwiseCausalConv1d2Tap(xG, a0G, a1G);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { xG, a0G, a1G });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 2e-5f, $"Conv2Tap.Fwd(fused={useFused})");
                AssertClose(dxComp.AsSpan(), gGrads[xG].AsSpan(), 5e-5f, $"Conv2Tap.Dx(fused={useFused})");
                AssertClose(dw0Comp.AsSpan(), gGrads[a0G].AsSpan(), 2e-4f, $"Conv2Tap.Dw0(fused={useFused})");
                AssertClose(dw1Comp.AsSpan(), gGrads[a1G].AsSpan(), 2e-4f, $"Conv2Tap.Dw1(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(2, 13, 4, 32, 0, 0.17677669529663687)]
    [InlineData(2, 9, 8, 64, 3, 1.0)]
    public void RopeL2NormalizeHeads_CpuComposedAndGpuFusedAndPortable_Match(
        int b, int s, int nh, int dh, int startPos, double postScale)
    {
        var rng = new Random(303 + b + s + nh + dh + startPos);
        var (cos, sin) = BuildRopeTables(startPos + s + 4, dh / 2);
        var xRaw = RandomTensor(rng, new[] { b, s, nh * dh });
        var dyRaw = RandomTensor(rng, new[] { b, s, nh * dh });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dxComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var x0 = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.RopeL2NormalizeHeadsComposed(cpu, x0, cos, sin, b, s, nh, dh, 1e-6, postScale, startPos);
            dxComp = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { x0 })[x0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var xG = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
                var cosG = new Tensor<float>(cos.AsSpan().ToArray(), cos._shape);
                var sinG = new Tensor<float>(sin.AsSpan().ToArray(), sin._shape);
                using var tape = new GradientTape<float>();
                var yG = gpu.RopeL2NormalizeHeads(xG, cosG, sinG, nh, 1e-6, postScale, startPos);
                var dxG = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { xG })[xG];

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"RopeL2.Fwd(fused={useFused})");
                AssertClose(dxComp.AsSpan(), dxG.AsSpan(), 6e-5f, $"RopeL2.Dx(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(2, 11, 8, 4, 16, 0)]
    [InlineData(2, 7, 8, 6, 40, 2)]
    public void RopeQuadraticOuterFeatures_CpuComposedAndGpuFusedAndPortable_Match(
        int b, int s, int nh, int mb, int outDim, int startPos)
    {
        var rng = new Random(404 + b + s + nh + mb + outDim);
        var (cos, sin) = BuildRopeTables(startPos + s + 4, mb / 2);
        var xRaw = RandomTensor(rng, new[] { b, s, nh * mb });
        var dyRaw = RandomTensor(rng, new[] { b, s, nh * outDim });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dxComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var x0 = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, nh * mb });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * outDim });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.RopeQuadraticOuterFeaturesComposed(
                cpu, x0, cos, sin, b, s, nh, mb, outDim, startPos, new[] { b, s, nh * outDim });
            dxComp = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { x0 })[x0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var xG = new Tensor<float>(xRaw.AsSpan().ToArray(), new[] { b, s, nh * mb });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * outDim });
                var cosG = new Tensor<float>(cos.AsSpan().ToArray(), cos._shape);
                var sinG = new Tensor<float>(sin.AsSpan().ToArray(), sin._shape);
                using var tape = new GradientTape<float>();
                var yG = gpu.RopeQuadraticOuterFeatures(xG, cosG, sinG, nh, outDim, startPos);
                var dxG = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { xG })[xG];

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"RopeQuad.Fwd(fused={useFused})");
                AssertClose(dxComp.AsSpan(), dxG.AsSpan(), 6e-5f, $"RopeQuad.Dx(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(19, 64)]
    [InlineData(64, 256)]
    public void ProjectiveGatedResidual_CpuComposedAndGpuFusedAndPortable_Match(int rows, int dim)
    {
        var rng = new Random(505 + rows + dim);
        var hRaw = RandomTensor(rng, new[] { rows, dim });
        var uRaw = RandomTensor(rng, new[] { rows, dim });
        var gRaw = new Tensor<float>(new[] { -0.37f }, new[] { 1, 1 });
        var dyRaw = RandomTensor(rng, new[] { rows, dim });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dhComp, duComp, dgComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var h0 = new Tensor<float>(hRaw.AsSpan().ToArray(), new[] { rows, dim });
            var u0 = new Tensor<float>(uRaw.AsSpan().ToArray(), new[] { rows, dim });
            var g0 = new Tensor<float>(gRaw.AsSpan().ToArray(), new[] { 1, 1 });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, dim });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.ProjectiveGatedResidualComposed(cpu, h0, u0, g0, dim, 1e-6);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { h0, u0, g0 });
            dhComp = grads[h0];
            duComp = grads[u0];
            dgComp = grads[g0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var hG = new Tensor<float>(hRaw.AsSpan().ToArray(), new[] { rows, dim });
                var uG = new Tensor<float>(uRaw.AsSpan().ToArray(), new[] { rows, dim });
                var gG = new Tensor<float>(gRaw.AsSpan().ToArray(), new[] { 1, 1 });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, dim });
                using var tape = new GradientTape<float>();
                var yG = gpu.ProjectiveGatedResidual(hG, uG, gG, 1e-6);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { hG, uG, gG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"ProjRes.Fwd(fused={useFused})");
                AssertClose(dhComp.AsSpan(), gGrads[hG].AsSpan(), 6e-5f, $"ProjRes.Dh(fused={useFused})");
                AssertClose(duComp.AsSpan(), gGrads[uG].AsSpan(), 6e-5f, $"ProjRes.Du(fused={useFused})");
                AssertClose(dgComp.AsSpan(), gGrads[gG].AsSpan(), 2e-4f, $"ProjRes.Dg(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(2, 64)]
    [InlineData(2, 256)]
    public void SpectralCirculantMatrix_CpuComposedAndGpuFusedAndPortable_Match(int batch, int dim)
    {
        var rng = new Random(505 + batch * 17 + dim);
        int nf = dim / 2 + 1;
        var dRaw = RandomTensor(rng, new[] { batch, dim });
        var reRaw = RandomTensor(rng, new[] { batch, nf, 1 });
        var imRaw = RandomTensor(rng, new[] { batch, nf, 1 });
        var dyRaw = RandomTensor(rng, new[] { batch, dim, dim });

        var cpu = new CpuEngine();
        Tensor<float> yComp, ddComp, dreComp, dimComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var d0 = new Tensor<float>(dRaw.AsSpan().ToArray(), new[] { batch, dim });
            var r0 = new Tensor<float>(reRaw.AsSpan().ToArray(), new[] { batch, nf, 1 });
            var i0 = new Tensor<float>(imRaw.AsSpan().ToArray(), new[] { batch, nf, 1 });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { batch, dim, dim });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.SpectralCirculantMatrixComposed(cpu, d0, r0, i0, batch, dim);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { d0, r0, i0 });
            ddComp = grads[d0];
            dreComp = grads[r0];
            dimComp = grads[i0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        var yCpu = cpu.SpectralCirculantMatrix(dRaw, reRaw, imRaw);
        AssertClose(yComp.AsSpan(), yCpu.AsSpan(), 5e-5f, "SpecCirc.CpuFast");

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var dG = new Tensor<float>(dRaw.AsSpan().ToArray(), new[] { batch, dim });
                var rG = new Tensor<float>(reRaw.AsSpan().ToArray(), new[] { batch, nf, 1 });
                var iG = new Tensor<float>(imRaw.AsSpan().ToArray(), new[] { batch, nf, 1 });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { batch, dim, dim });
                using var tape = new GradientTape<float>();
                var yG = gpu.SpectralCirculantMatrix(dG, rG, iG);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { dG, rG, iG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 5e-5f, $"SpecCirc.Fwd(fused={useFused})");
                AssertClose(ddComp.AsSpan(), gGrads[dG].AsSpan(), 2e-4f, $"SpecCirc.Dd(fused={useFused})");
                AssertClose(dreComp.AsSpan(), gGrads[rG].AsSpan(), 5e-4f, $"SpecCirc.Dre(fused={useFused})");
                AssertClose(dimComp.AsSpan(), gGrads[iG].AsSpan(), 5e-4f, $"SpecCirc.Dim(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(37, 48, 4, 6, 0.001)]
    [InlineData(19, 16, 2, 12, 0.0)]
    public void SigmoidTopKExpertMix_CpuComposedAndGpuFusedAndPortable_Match(int rows, int experts, int topK, int channels, double biasRate)
    {
        var rng = new Random(606 + rows * 7 + experts * 3 + topK);
        var logRaw = RandomTensor(rng, new[] { rows, experts });
        var tblRaw = RandomTensor(rng, new[] { experts, channels });
        var biasRaw = RandomTensor(rng, new[] { 1, experts });
        var dyRaw = RandomTensor(rng, new[] { rows, channels });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dlogComp, dtblComp;
        float[] biasAfterComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var l0 = new Tensor<float>(logRaw.AsSpan().ToArray(), new[] { rows, experts });
            var t0 = new Tensor<float>(tblRaw.AsSpan().ToArray(), new[] { experts, channels });
            var b0 = new Tensor<float>(biasRaw.AsSpan().ToArray(), new[] { 1, experts });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, channels });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.SigmoidTopKExpertMixComposed(cpu, l0, t0, b0, topK, 1e-12, biasRate);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { l0, t0 });
            dlogComp = grads[l0];
            dtblComp = grads[t0];
            biasAfterComp = b0.AsSpan().ToArray();
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var lG = new Tensor<float>(logRaw.AsSpan().ToArray(), new[] { rows, experts });
                var tG = new Tensor<float>(tblRaw.AsSpan().ToArray(), new[] { experts, channels });
                var bG = new Tensor<float>(biasRaw.AsSpan().ToArray(), new[] { 1, experts });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, channels });
                using var tape = new GradientTape<float>();
                var yG = gpu.SigmoidTopKExpertMix(lG, tG, bG, topK, 1e-12, biasRate);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { lG, tG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"SigTopK.Fwd(fused={useFused})");
                AssertClose(dlogComp.AsSpan(), gGrads[lG].AsSpan(), 6e-5f, $"SigTopK.Dlog(fused={useFused})");
                AssertClose(dtblComp.AsSpan(), gGrads[tG].AsSpan(), 2e-4f, $"SigTopK.Dtbl(fused={useFused})");
                AssertClose(biasAfterComp, bG.AsSpan(), 1e-6f, $"SigTopK.Bias(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(29, 64)]
    public void DualModulatedSwiGlu_CpuComposedAndGpuFusedAndPortable_Match(int rows, int h)
    {
        var rng = new Random(707 + rows + h);
        var guRaw = RandomTensor(rng, new[] { rows, 2 * h });
        var ggRaw = RandomTensor(rng, new[] { rows, 2 * h });
        var bRaw = RandomTensor(rng, new[] { 1, h });
        var dyRaw = RandomTensor(rng, new[] { rows, h });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dguComp, dggComp, dbComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var gu0 = new Tensor<float>(guRaw.AsSpan().ToArray(), new[] { rows, 2 * h });
            var gg0 = new Tensor<float>(ggRaw.AsSpan().ToArray(), new[] { rows, 2 * h });
            var b0 = new Tensor<float>(bRaw.AsSpan().ToArray(), new[] { 1, h });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, h });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.DualModulatedSwiGluComposed(cpu, gu0, gg0, b0, h);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { gu0, gg0, b0 });
            dguComp = grads[gu0];
            dggComp = grads[gg0];
            dbComp = grads[b0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var guG = new Tensor<float>(guRaw.AsSpan().ToArray(), new[] { rows, 2 * h });
                var ggG = new Tensor<float>(ggRaw.AsSpan().ToArray(), new[] { rows, 2 * h });
                var bG = new Tensor<float>(bRaw.AsSpan().ToArray(), new[] { 1, h });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { rows, h });
                using var tape = new GradientTape<float>();
                var yG = gpu.DualModulatedSwiGlu(guG, ggG, bG);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { guG, ggG, bG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"DualSwiGlu.Fwd(fused={useFused})");
                AssertClose(dguComp.AsSpan(), gGrads[guG].AsSpan(), 8e-5f, $"DualSwiGlu.Dgu(fused={useFused})");
                AssertClose(dggComp.AsSpan(), gGrads[ggG].AsSpan(), 8e-5f, $"DualSwiGlu.Dgg(fused={useFused})");
                AssertClose(dbComp.AsSpan(), gGrads[bG].AsSpan(), 3e-4f, $"DualSwiGlu.Db(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(2, 11, 8, 32, 33, false)]
    [InlineData(2, 9, 4, 16, 20, true)]
    public void BornRatioNormalize_CpuComposedAndGpuFusedAndPortable_Match(int b, int s, int nh, int dh, int scanDim, bool withSink)
    {
        var rng = new Random(808 + b * 3 + s * 5 + nh + (withSink ? 99 : 0));
        var scanArr = RandomTensor(rng, new[] { b, s, nh, scanDim }).AsSpan().ToArray();
        for (int i = dh; i < scanArr.Length; i += scanDim) scanArr[i] = MathF.Abs(scanArr[i]) + 0.5f;
        var scanRaw = new Tensor<float>(scanArr, new[] { b, s, nh, scanDim });
        var sinkRaw = withSink ? RandomTensor(rng, new[] { 1, 1, nh, 1 }) : null;
        var dyRaw = RandomTensor(rng, new[] { b, s, nh * dh });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dsComp;
        Tensor<float>? dkComp = null;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var sc0 = new Tensor<float>(scanRaw.AsSpan().ToArray(), new[] { b, s, nh, scanDim });
            var sk0 = sinkRaw is null ? null : new Tensor<float>(sinkRaw.AsSpan().ToArray(), new[] { 1, 1, nh, 1 });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.BornRatioNormalizeComposed(cpu, sc0, sk0, b, s, nh, dh, 1e-4);
            var srcs = sk0 is null ? new[] { sc0 } : new[] { sc0, sk0 };
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), srcs);
            dsComp = grads[sc0];
            if (sk0 is not null) dkComp = grads[sk0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var scG = new Tensor<float>(scanRaw.AsSpan().ToArray(), new[] { b, s, nh, scanDim });
                var skG = sinkRaw is null ? null : new Tensor<float>(sinkRaw.AsSpan().ToArray(), new[] { 1, 1, nh, 1 });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh * dh });
                using var tape = new GradientTape<float>();
                var yG = gpu.BornRatioNormalize(scG, skG, dh, 1e-4);
                var srcs = skG is null ? new[] { scG } : new[] { scG, skG };
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), srcs);

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 3e-5f, $"BornRatio.Fwd(fused={useFused})");
                AssertClose(dsComp.AsSpan(), gGrads[scG].AsSpan(), 8e-5f, $"BornRatio.Dscan(fused={useFused})");
                if (skG is not null && dkComp is not null)
                    AssertClose(dkComp.AsSpan(), gGrads[skG].AsSpan(), 2e-4f, $"BornRatio.Dsink(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }

    [Theory]
    [InlineData(3, 17, 8, 0.5)]
    public void BoundedSigmoidGate_CpuComposedAndGpuFusedAndPortable_Match(int b, int s, int nh, double floor)
    {
        var rng = new Random(909 + b + s + nh);
        var logRaw = RandomTensor(rng, new[] { b, s, nh });
        var biasRaw = RandomTensor(rng, new[] { 1, 1, nh });
        var dyRaw = RandomTensor(rng, new[] { b, s, nh });

        var cpu = new CpuEngine();
        Tensor<float> yComp, dlComp, dbComp;
        var prev = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = cpu;
            var l0 = new Tensor<float>(logRaw.AsSpan().ToArray(), new[] { b, s, nh });
            var b0 = new Tensor<float>(biasRaw.AsSpan().ToArray(), new[] { 1, 1, nh });
            var dy0 = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh });
            using var tape = new GradientTape<float>();
            yComp = CpuEngine.BoundedSigmoidGateComposed(cpu, l0, b0, floor);
            var grads = tape.ComputeGradients(cpu.ReduceSum(cpu.TensorMultiply(yComp, dy0), null), new[] { l0, b0 });
            dlComp = grads[l0];
            dbComp = grads[b0];
        }
        finally
        {
            AiDotNetEngine.Current = prev;
        }

        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.SupportsGpu) return;

        foreach (bool useFused in new[] { true, false })
        {
            bool saved = CudaBackend.UseFusedSequenceMixer;
            CudaBackend.UseFusedSequenceMixer = useFused;
            try
            {
                AiDotNetEngine.Current = gpu;
                var lG = new Tensor<float>(logRaw.AsSpan().ToArray(), new[] { b, s, nh });
                var bG = new Tensor<float>(biasRaw.AsSpan().ToArray(), new[] { 1, 1, nh });
                var dyG = new Tensor<float>(dyRaw.AsSpan().ToArray(), new[] { b, s, nh });
                using var tape = new GradientTape<float>();
                var yG = gpu.BoundedSigmoidGate(lG, bG, floor);
                var gGrads = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(yG, dyG), null), new[] { lG, bG });

                AssertClose(yComp.AsSpan(), yG.AsSpan(), 2e-5f, $"BoundSig.Fwd(fused={useFused})");
                AssertClose(dlComp.AsSpan(), gGrads[lG].AsSpan(), 5e-5f, $"BoundSig.Dlog(fused={useFused})");
                AssertClose(dbComp.AsSpan(), gGrads[bG].AsSpan(), 2e-4f, $"BoundSig.Dbias(fused={useFused})");
            }
            finally
            {
                AiDotNetEngine.Current = prev;
                CudaBackend.UseFusedSequenceMixer = saved;
            }
        }
    }
}
