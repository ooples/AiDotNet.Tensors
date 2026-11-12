using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Wall-clock measurements of the model-op kernels at sizes that stress them (a vocabulary-sized top-k row, a long CRF
/// sequence), printed to the test output. Opt-in (AIDOTNET_RUN_GPU_TIMINGS=1): timings are for comparing builds on one
/// machine, not assertions. Correctness at these sizes is asserted by TorchModelOpsGpuTests and the op-parity registry.
/// The one exception is <see cref="TopKSoftmax_LargeK_VocabularyRows_StayFast"/>, which always runs and bounds a
/// large-k top-k over vocabulary rows (a per-thread insertion selection took seconds there).
/// </summary>
[Collection("DirectGpuSerial")]
public class TorchModelOpsGpuTimingTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IClassFixture<CudaTensorEngineTestFixture>,
    IClassFixture<OpenClTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly CudaTensorEngineTestFixture _cuda;
    private readonly OpenClTensorEngineTestFixture _openCl;
    private readonly ITestOutputHelper _out;

    public TorchModelOpsGpuTimingTests(DirectGpuTensorEngineTestFixture fixture, CudaTensorEngineTestFixture cuda,
        OpenClTensorEngineTestFixture openCl, ITestOutputHelper output)
    {
        _fixture = fixture;
        _cuda = cuda;
        _openCl = openCl;
        _out = output;
    }

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        return new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
            .Select(_ => (float)(rng.NextDouble() * 6 - 3)).ToArray(), shape);
    }

    private double Time(string what, Func<Tensor<float>> run, int reps)
    {
        _ = run().ToArray();   // warm-up: kernel compilation, buffer pools
        // The median of single runs: the machine is shared, and a mean takes in every stall.
        var samples = new double[Math.Max(reps, 15)];
        for (int i = 0; i < samples.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = run().ToArray();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        double ms = samples[samples.Length / 2];
        _out.WriteLine($"{what}: {ms:F3} ms (median of {samples.Length})");
        return ms;
    }

    [SkippableFact]
    public void Measure()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_GPU_TIMINGS") == "1", "Set AIDOTNET_RUN_GPU_TIMINGS=1 to measure.");
        Skip.IfNot(_fixture.IsAvailable || _cuda.IsAvailable, "No GPU device.");
        if (_fixture.IsAvailable) MeasureOn(_fixture.Engine!);
        if (_cuda.IsAvailable) MeasureOn(_cuda.Engine!);
    }

    /// <summary>
    /// k = 1024 and k = 8192 over 8 rows of 50,000 (a vocabulary) on each device backend: the median wall time of a
    /// forward, including the download, must stay under <see cref="LargeKBoundMs"/> (one work group per row, about 50
    /// passes over the row whatever k is). The selection itself is checked by TorchModelOpsGpuTests.
    /// </summary>
    [SkippableFact]
    public void TopKSoftmax_LargeK_VocabularyRows_StayFast()
    {
        var engines = new[] { _fixture.IsAvailable ? _fixture.Engine : null, _cuda.IsAvailable ? _cuda.Engine : null,
            _openCl.IsAvailable ? _openCl.Engine : null };
        var seen = new System.Collections.Generic.HashSet<Type>();
        int measured = 0;
        foreach (var gpu in engines)
        {
            var backend = gpu?.GetBackend();
            if (gpu is null || backend is not AiDotNet.Tensors.Engines.DirectGpu.ITorchModelOpsBackend || !seen.Add(backend.GetType()))
                continue;
            _out.WriteLine($"backend: {backend.GetType().Name}");
            var x = gpu.TensorMultiplyScalar(Rand(4, 8, 50000), 1f);
            foreach (int k in new[] { 1024, 8192 })
            {
                double ms = Time($"TopKSoftmax rows=8 width=50000 k={k}", () => gpu.TensorTopKSoftmax(x, k), 5);
                Assert.True(ms < LargeKBoundMs, $"{backend.GetType().Name}: TopKSoftmax k={k} over 8 x 50000 took {ms:F1} ms.");
            }
            measured++;
        }
        Skip.If(measured == 0, "No GPU backend with model-op kernels.");
    }

    /// <summary>
    /// The top-k dispatch table (opt-in, AIDOTNET_RUN_GPU_TIMINGS=1): for k x width x rows on the CUDA and OpenCL
    /// engines, the median wall time of each mode forced (direct for rows up to 1024 wide; chunked while its per-chunk
    /// insertion stays short; row groups) and of the planner's own choice. The chosen mode must be within noise of the
    /// fastest mode measured for the cell. This table set <see cref="TorchModelOpsCodes.TopKChunkedMaxK"/>.
    /// </summary>
    [SkippableFact]
    public void TopKSoftmax_DispatchTable()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_GPU_TIMINGS") == "1", "Set AIDOTNET_RUN_GPU_TIMINGS=1 to measure.");
        var engines = new[] { _cuda.IsAvailable ? _cuda.Engine : null, _openCl.IsAvailable ? _openCl.Engine : null };
        var slower = new System.Collections.Generic.List<string>();
        int measured = 0;
        foreach (var gpu in engines)
        {
            var backend = gpu?.GetBackend();
            if (gpu is null || backend is not AiDotNet.Tensors.Engines.DirectGpu.ITorchModelOpsBackend) continue;
            string name = backend.GetType().Name;
            _out.WriteLine($"backend: {name}");
            _out.WriteLine("rows | width | k | direct ms | chunked ms | rowgroups ms | auto ms | auto mode");
            foreach (int width in new[] { 64, 256, 512, 1024, 2048, 50000 })
                foreach (int rows in new[] { 1, 8, 64 })
                {
                    var x = gpu.TensorMultiplyScalar(Rand(5, rows, width), 1f);
                    foreach (int k in new[] { 1, 2, 4, 8, 32, 64, 256, 1024, 8192 })
                    {
                        if (k > width) continue;
                        double Forced(TorchModelOpsCodes.TopKMode mode)
                        {
                            TorchModelOpsCodes.ForcedMode = mode;
                            try { return Quiet(() => gpu.TensorTopKSoftmax(x, k)); }
                            finally { TorchModelOpsCodes.ForcedMode = null; }
                        }
                        double direct = width <= TorchModelOpsCodes.TopKDirectWidth ? Forced(TorchModelOpsCodes.TopKMode.Direct) : double.NaN;
                        // The chunked insertion is O(chunk * k) per chunk: past k = 1024 (or a big batch at 1024) it takes seconds.
                        bool chunkedSafe = k <= 256 || (k <= 1024 && rows <= 8);
                        double chunked = chunkedSafe ? Forced(TorchModelOpsCodes.TopKMode.Chunked) : double.NaN;
                        double rowGroups = Forced(TorchModelOpsCodes.TopKMode.RowGroups);
                        double auto = Quiet(() => gpu.TensorTopKSoftmax(x, k));
                        var mode = TorchModelOpsCodes.ChooseTopKMode(rows, width, k);
                        // The chosen mode's own forced time against the fastest: the auto run is the same kernels.
                        double Chosen() => mode switch
                        {
                            TorchModelOpsCodes.TopKMode.Direct => direct,
                            TorchModelOpsCodes.TopKMode.Chunked => chunked,
                            _ => rowGroups,
                        };
                        double Best() => new[] { direct, chunked, rowGroups }.Where(v => !double.IsNaN(v)).Min();
                        // A shared machine stalls now and then (a 64 x 50,000 cell is mostly its 12.8 MB download): while the
                        // choice looks slower, measure every mode again, up to best of three, keeping each one's fastest run.
                        for (int retry = 0; retry < 2 && Chosen() > Best() * 1.15 + 0.05; retry++)
                        {
                            if (!double.IsNaN(direct)) direct = Math.Min(direct, Forced(TorchModelOpsCodes.TopKMode.Direct));
                            if (!double.IsNaN(chunked)) chunked = Math.Min(chunked, Forced(TorchModelOpsCodes.TopKMode.Chunked));
                            rowGroups = Math.Min(rowGroups, Forced(TorchModelOpsCodes.TopKMode.RowGroups));
                        }
                        _out.WriteLine($"{rows} | {width} | {k} | {Fmt(direct)} | {Fmt(chunked)} | {Fmt(rowGroups)} | {auto:F3} | {mode}");
                        if (Chosen() > Best() * 1.15 + 0.05)
                            slower.Add($"{name} rows={rows} width={width} k={k}: {mode} {Chosen():F3} ms vs best {Best():F3} ms");
                    }
                }
            measured++;
        }
        Skip.If(measured == 0, "No CUDA or OpenCL engine with model-op kernels.");
        foreach (var s in slower) _out.WriteLine("SLOWER: " + s);
        Assert.True(slower.Count == 0, "The planner picked a slower mode:\n" + string.Join("\n", slower));
    }

    private static string Fmt(double ms) => double.IsNaN(ms) ? "-" : ms.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);

    // The median wall time of a run (with its download), not printed.
    private static double Quiet(Func<Tensor<float>> run)
    {
        _ = run().ToArray();
        var samples = new double[21];
        for (int i = 0; i < samples.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = run().ToArray();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    /// <summary>The bound of <see cref="TopKSoftmax_LargeK_VocabularyRows_StayFast"/>: generous for a shared machine.</summary>
    private const double LargeKBoundMs = 100;

    private void MeasureOn(DirectGpuTensorEngine gpu)
    {
        _out.WriteLine($"backend: {gpu.GetBackend()?.GetType().Name}");

        foreach (var (rows, width, k) in new[] { (1, 50000, 8), (8, 50000, 8), (64, 1024, 4), (1, 50000, 1024), (8, 50000, 8192) })
        {
            var x = gpu.TensorMultiplyScalar(Rand(1, rows, width), 1f);
            Time($"TopKSoftmax rows={rows} width={width} k={k}", () => gpu.TensorTopKSoftmax(x, k), 5);
        }

        foreach (var (b, s, tags) in new[] { (32, 128, 16), (8, 512, 32), (1, 1024, 8) })
        {
            var e = gpu.TensorMultiplyScalar(Rand(2, b, s, tags), 1f);
            var t = gpu.TensorMultiplyScalar(Rand(3, tags, tags), 1f);
            Time($"CrfForward b={b} s={s} k={tags}", () => gpu.TensorCrfForward(e, t), 5);
            Time($"CrfForward+backward b={b} s={s} k={tags}", () =>
            {
                using var tape = new GradientTape<float>();
                var z = gpu.TensorCrfForward(e, t);
                var loss = gpu.ReduceSum(z, new[] { 0 }, keepDims: false);
                var g = tape.ComputeGradients(loss, new[] { e, t });
                _ = g[t].ToArray();
                return g[e];
            }, 5);
        }
    }
}
