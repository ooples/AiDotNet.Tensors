// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
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

    // A consumer after the GRU, so the node's gradient is not just the seed.
    private static Tensor<float> Loss(IEngine e, Tensor<float>[] p, bool returnSequences)
    {
        var y = e.GruSequenceForward(p[0], null, p[2], p[3], p[4], p[5], returnSequences: returnSequences, resetAfter: true);
        var z = e.Tanh(y);
        return e.ReduceSum(e.TensorMultiply(z, z), null);
    }

    /// <summary>
    /// Compiled training on the GPU engine records one GRU node with a direct device backward (TryRecordGruSequenceTrain):
    /// every step runs exactly one fused forward and one fused backward (the host op's node replayed the forward in its
    /// backward), and its loss and gradients match the CPU engine's eager tape, step after step.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]   // the final state only: the last-step node and backward
    [InlineData(true)]
    public void CompiledTrainingRunsOneForwardAndOneBackwardPerStep(bool returnSequences)
    {
        using var gpu = TryCuda();
        Skip.If(gpu is null, "No CUDA backend.");
        const int batch = 8, steps = 6, inputs = 5, hidden = 24;
        var rng = new Random(77);
        double scale = 1.0 / Math.Sqrt(hidden);
        Tensor<float>[] Make() => new[]
        {
            Rand(new Random(1), 1.0, batch, steps, inputs), new Tensor<float>(new[] { batch, hidden }),
            Rand(new Random(3), scale, 3 * hidden, inputs), Rand(new Random(4), scale, 3 * hidden, hidden),
            Rand(new Random(5), scale, 3 * hidden), Rand(new Random(6), scale, 3 * hidden),
        };
        var cpu = Make();
        var cpuSources = new[] { cpu[0], cpu[2], cpu[3], cpu[4], cpu[5] };
        float cpuLoss;
        var cpuGrads = new double[cpuSources.Length][];
        using (var tape = new GradientTape<float>())
        {
            var loss = Loss(new CpuEngine(), cpu, returnSequences);
            cpuLoss = loss[0];
            var g = tape.ComputeGradients(loss, cpuSources);
            for (int i = 0; i < cpuSources.Length; i++) cpuGrads[i] = Array.ConvertAll(g[cpuSources[i]].ToArray(), v => (double)v);
        }

        var p = Make();
        var sources = new[] { p[0], p[2], p[3], p[4], p[5] };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Loss(gpu!, p, returnSequences);
            plan = scope.CompileTraining(sources);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 3; step++)
            {
                long forward = CudaBackend.FusedGruForwardRuns, backward = CudaBackend.FusedGruBackwardRuns;
                var loss = plan.Step();
                Assert.Equal(1, CudaBackend.FusedGruForwardRuns - forward);
                Assert.Equal(1, CudaBackend.FusedGruBackwardRuns - backward);
                Assert.True(Math.Abs(loss[0] - cpuLoss) <= 1e-4 + 1e-4 * Math.Abs(cpuLoss), $"step {step}: loss {loss[0]}, CPU {cpuLoss}.");
                for (int i = 0; i < sources.Length; i++)
                {
                    var g = sources[i].Grad;
                    Assert.True(g is not null, $"source {i} has no gradient at step {step}.");
                    AssertClose($"step {step} source {i}", cpuGrads[i], Array.ConvertAll(g!.ToArray(), v => (double)v));
                }
            }
        }
    }

    /// <summary>
    /// One GRU applied to two sequences in a step (a siamese encoder): the two recorded nodes share the weights but not
    /// their activation buffers, so both backwards see their own forward's state. Keyed by the weights, the second
    /// forward overwrote the first's activations and the first backward threw.
    /// </summary>
    [SkippableFact]
    public void CompiledTraining_TwoCallsOfOneGru_KeepSeparateActivations()
    {
        using var gpu = TryCuda();
        Skip.If(gpu is null, "No CUDA backend.");
        const int batch = 4, steps = 5, inputs = 6, hidden = 16;
        double scale = 1.0 / Math.Sqrt(hidden);
        Tensor<float>[] Make() => new[]
        {
            Rand(new Random(11), 1.0, batch, steps, inputs), Rand(new Random(12), 1.0, batch, steps, inputs),
            Rand(new Random(13), scale, 3 * hidden, inputs), Rand(new Random(14), scale, 3 * hidden, hidden),
            Rand(new Random(15), scale, 3 * hidden), Rand(new Random(16), scale, 3 * hidden),
        };
        static Tensor<float> Siamese(IEngine e, Tensor<float>[] p)
        {
            Tensor<float> Branch(Tensor<float> x)
            {
                var y = e.GruSequenceForward(x, null, p[2], p[3], p[4], p[5], returnSequences: true, resetAfter: true);
                var z = e.Tanh(y);
                return e.ReduceSum(e.TensorMultiply(z, z), null);
            }
            // Unequal weights on the branches, so each branch's gradient matters to the total.
            return e.TensorAdd(Branch(p[0]), e.TensorMultiplyScalar(Branch(p[1]), 0.5f));
        }
        var cpu = Make();
        var cpuSources = new[] { cpu[0], cpu[1], cpu[2], cpu[3], cpu[4], cpu[5] };
        float cpuLoss;
        var cpuGrads = new double[cpuSources.Length][];
        using (var tape = new GradientTape<float>())
        {
            var loss = Siamese(new CpuEngine(), cpu);
            cpuLoss = loss[0];
            var g = tape.ComputeGradients(loss, cpuSources);
            for (int i = 0; i < cpuSources.Length; i++) cpuGrads[i] = Array.ConvertAll(g[cpuSources[i]].ToArray(), v => (double)v);
        }

        var p = Make();
        var sources = new[] { p[0], p[1], p[2], p[3], p[4], p[5] };
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Siamese(gpu!, p);
            plan = scope.CompileTraining(sources);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 2; step++)
            {
                long forward = CudaBackend.FusedGruForwardRuns, backward = CudaBackend.FusedGruBackwardRuns;
                var loss = plan.Step();
                Assert.Equal(2, CudaBackend.FusedGruForwardRuns - forward);
                Assert.Equal(2, CudaBackend.FusedGruBackwardRuns - backward);
                Assert.True(Math.Abs(loss[0] - cpuLoss) <= 1e-4 + 1e-4 * Math.Abs(cpuLoss), $"step {step}: loss {loss[0]}, CPU {cpuLoss}.");
                for (int i = 0; i < sources.Length; i++)
                {
                    var g = sources[i].Grad;
                    Assert.True(g is not null, $"source {i} has no gradient at step {step}.");
                    AssertClose($"step {step} source {i}", cpuGrads[i], Array.ConvertAll(g!.ToArray(), v => (double)v));
                }
            }
        }
    }

    /// <summary>
    /// The backend's last-step-only backward (the gradient of h_T alone, [B, H]) is the full backward given a [B, T, H]
    /// gradient that is zero except at the last step, bit for bit; and the forward's h_T output is the last step of its
    /// sequence output.
    /// </summary>
    [SkippableTheory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(80)]   // the portable kernels: h_T copied from the state cache, the gradient as the final state's
    public void TheLastStepFormsMatchTheFullSequenceForms(int hidden)
    {
        using var gpu = TryCuda();
        Skip.If(gpu is null, "No CUDA backend.");
        var backend = gpu!.GetBackend()!;
        var gru = (IFusedGruSequence)backend;
        const int b = 4, t = 5, inputs = 7;
        int g3 = 3 * hidden;
        var rng = new Random(hidden);
        float[] R(int n, double scale) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale); return a; }
        double scale = 1.0 / Math.Sqrt(hidden);
        using var x = backend.AllocateBuffer(R(b * t * inputs, 1.0));
        using var h0 = backend.AllocateBuffer(R(b * hidden, 0.5));
        using var wih = backend.AllocateBuffer(R(g3 * inputs, scale));
        using var whh = backend.AllocateBuffer(R(g3 * hidden, scale));
        using var bih = backend.AllocateBuffer(R(g3, scale));
        using var bhh = backend.AllocateBuffer(R(g3, scale));
        using var y = backend.AllocateBuffer(b * t * hidden);
        using var allH = backend.AllocateBuffer((t + 1) * b * hidden);
        using var gates = backend.AllocateBuffer(t * b * g3);
        using var ghn = backend.AllocateBuffer(t * b * hidden);
        using var hFinal = backend.AllocateBuffer(b * hidden);
        gru.GruForwardSequence(x, h0, wih, whh, bih, bhh, y, allH, gates, ghn, t, b, inputs, hidden, hFinal);
        var yHost = backend.DownloadBuffer(y);
        var last = backend.DownloadBuffer(hFinal);
        for (int bb = 0; bb < b; bb++)
            for (int j = 0; j < hidden; j++)
                Assert.Equal(yHost[(bb * t + t - 1) * hidden + j], last[bb * hidden + j]);

        var gLast = R(b * hidden, 1.0);
        var gSeq = new float[b * t * hidden];
        for (int bb = 0; bb < b; bb++) Array.Copy(gLast, bb * hidden, gSeq, (bb * t + t - 1) * hidden, hidden);
        float[][] Backward(float[] gradient, bool lastStepOnly)
        {
            using var gy = backend.AllocateBuffer(gradient);
            var outputs = new[]
            {
                backend.AllocateBuffer(b * t * inputs), backend.AllocateBuffer(b * hidden), backend.AllocateBuffer(g3 * inputs),
                backend.AllocateBuffer(g3 * hidden), backend.AllocateBuffer(g3), backend.AllocateBuffer(g3),
            };
            try
            {
                gru.GruBackwardSequence(gy, allH, gates, ghn, wih, whh, x, outputs[0], outputs[1], outputs[2], outputs[3],
                    outputs[4], outputs[5], t, b, inputs, hidden, lastStepOnly);
                return Array.ConvertAll(outputs, o => backend.DownloadBuffer(o));
            }
            finally
            {
                foreach (var o in outputs) o.Dispose();
            }
        }
        var full = Backward(gSeq, lastStepOnly: false);
        var lastOnly = Backward(gLast, lastStepOnly: true);
        for (int i = 0; i < full.Length; i++) Assert.Equal(full[i], lastOnly[i]);
    }
}
