using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// ParameterBuffer views on the GPU engine. Every view shares the flat array as its device-cache key; the weight
/// path served the buffer cached for that array -- the FIRST parameter's data -- to every other parameter, so a model
/// whose parameters live in one ParameterBuffer (eager TrainWithTape) trained on wrong gradients.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ParameterBufferViewGpuCacheTests
{
    private static DirectGpuTensorEngine? TryGpu()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            return gpu.IsGpuAvailable ? gpu : null;
        }
        catch
        {
            return null;
        }
    }

    private static float[] Rand(int n, int seed, double scale = 1)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(scale * (2 * r.NextDouble() - 1))).ToArray();
    }

    /// <summary>
    /// Layout of a tied LM: [table (offset 0), gamma (offset table.Length)]. The table is read first (registering and
    /// caching the flat array); RMSNorm's backward then reads gamma through the weight path.
    /// </summary>
    private static (float[] Dx, float[] Dgamma, float[] Dtable) Grads(IEngine eng, bool registerPersistent,
        float[] x, float[] tableData, float[] gammaData, int rows, int dim, int vocab)
    {
        var pb = new ParameterBuffer<float>(new[] { new[] { vocab, dim }, new[] { dim } }.ToList());
        pb.CopyFrom(new[]
        {
            new Tensor<float>(new[] { vocab, dim }, new Vector<float>((float[])tableData.Clone())),
            new Tensor<float>(new[] { dim }, new Vector<float>((float[])gammaData.Clone())),
        }.ToList());
        var views = pb.CreateAllViews();
        var table = views[0];
        var gamma = views[1];
        if (registerPersistent)
        {
            // What LayerBase.RegisterTrainableParameter does for every layer weight.
            eng.RegisterPersistentTensor(table, PersistentTensorRole.Weights);
            eng.RegisterPersistentTensor(gamma, PersistentTensorRole.Weights);
        }

        var input = new Tensor<float>(new[] { rows, dim }, new Vector<float>((float[])x.Clone()));
        var probe = new Tensor<float>(new[] { rows, vocab }, new Vector<float>(Rand(rows * vocab, 99)));
        using var tape = new GradientTape<float>();
        // The table is consumed before gamma, as in the model (embedding lookup, then the tied head).
        eng.TensorMatMulTransposed(input, table);
        var hn = eng.RMSNorm(input, gamma, 1e-6, out _);
        var logits = eng.TensorMatMulTransposed(hn, table);
        var loss = eng.ReduceSum(eng.TensorMultiply(logits, probe), new[] { 0, 1 }, keepDims: false);
        var g = tape.ComputeGradients(loss, new[] { input, gamma, table });
        return (g[input].ToArray(), g[gamma].ToArray(), g[table].ToArray());
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void RmsNormBackward_GammaViewAtNonzeroOffset_MatchesCpu(bool registerPersistent)
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        using var gpuScope = gpu;
        const int rows = 64, dim = 128, vocab = 512;
        var x = Rand(rows * dim, 1);
        var table = Rand(vocab * dim, 2, 0.02);       // small, like an embedding init
        var gamma = Enumerable.Repeat(1f, dim).ToArray();   // RMSNorm gain init: ones

        var expected = Grads(new CpuEngine(), registerPersistent, x, table, gamma, rows, dim, vocab);
        var actual = Grads(gpu!, registerPersistent, x, table, gamma, rows, dim, vocab);

        void Close(string name, float[] want, float[] got)
        {
            double scale = want.Max(v => Math.Abs((double)v));
            double maxDiff = want.Zip(got, (a, b) => Math.Abs((double)a - b)).Max();
            Assert.True(maxDiff <= 1e-4 * Math.Max(scale, 1e-30) + 1e-6,
                $"{name}: max |gpu - cpu| {maxDiff:E3} against max |cpu| {scale:E3}");
        }

        Close("d input", expected.Dx, actual.Dx);
        Close("d gamma", expected.Dgamma, actual.Dgamma);
        Close("d table", expected.Dtable, actual.Dtable);
    }

    [SkippableFact]
    public void ViewLookups_UploadTheFlatArrayOncePerVersion()
    {
        var gpu = TryGpu();
        Skip.If(gpu is null, "GPU engine unavailable.");
        using var gpuScope = gpu;
        const int dim = 256, count = 8;
        var pb = new ParameterBuffer<float>(Enumerable.Range(0, count).Select(_ => new[] { dim }).ToList());
        pb.CopyFrom(Enumerable.Range(0, count).Select(i => new Tensor<float>(new[] { dim }, new Vector<float>(Rand(dim, 10 + i)))).ToList());
        var views = pb.CreateAllViews();
        foreach (var v in views) gpu!.RegisterPersistentTensor(v, PersistentTensorRole.Weights);
        var x0 = new Tensor<float>(new[] { 4, dim }, new Vector<float>(Rand(4 * dim, 3)));
        var x = gpu!.TensorAdd(x0, x0);   // device-resident input: any upload below is a parameter upload

        // Warm: the first read of the flat array uploads it once.
        foreach (var v in views) gpu.RMSNorm(x, v, 1e-6, out _);

        GpuLaunchProbe.Reset();
        foreach (var v in views) gpu.RMSNorm(x, v, 1e-6, out _);
        Assert.True(GpuLaunchProbe.Uploads == 0,
            $"{GpuLaunchProbe.Uploads} uploads ({GpuLaunchProbe.UploadBytes:N0} bytes) for {count} reads of host-current parameter views: " +
            string.Join("; ", GpuLaunchProbe.UploadSites));

        // Each view's forward is its own gain, not the first parameter's.
        var cpu = new CpuEngine();
        var xHost = cpu.TensorAdd(x0, x0);
        for (int i = 0; i < count; i++)
        {
            var want = cpu.RMSNorm(xHost, new Tensor<float>(new[] { dim }, new Vector<float>(views[i].ToArray())), 1e-6, out _).ToArray();
            var got = gpu.RMSNorm(x, views[i], 1e-6, out _).ToArray();
            for (int j = 0; j < want.Length; j++)
                Assert.True(Math.Abs(want[j] - got[j]) < 1e-4, $"view {i} element {j}: {got[j]} vs {want[j]}");
        }
    }
}
