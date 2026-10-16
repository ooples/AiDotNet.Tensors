using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1055: input gradient of ResNet-18's stride-2 convolutions (3x3 and the 1x1 downsample, batch 8, 32x32 input) per
/// route, interleaved round by round (min / median us per call). cuDNN, measured the same way (each op in its own CUDA
/// graph): 3x3 64-128@32 67, 128-256@16 38.5, 256-512@8 59; 1x1 22.5, 21.3, 26. Gated on AIDOTNET_PERF_RUNNER=1.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class StridedInputGradientBenchTests
{
    private readonly ITestOutputHelper _out;
    public StridedInputGradientBenchTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void Bench_StridedInputGradient_PerRoute()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_PERF_RUNNER") == "1", "Set AIDOTNET_PERF_RUNNER=1 on a quiet GPU machine.");
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = CudaBackend.CreateOrThrow();
        const int batch = 8, rounds = 31, burst = 10;
        var rng = new Random(2);
        float[] Fill(int n) { var v = new float[n]; for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1); return v; }
        string[] routes = { "slot", "engine", "implicit", "col2im" };
        foreach (var (inC, outC, hw, k) in new[] { (64, 128, 32, 3), (128, 256, 16, 3), (256, 512, 8, 3), (64, 128, 32, 1), (128, 256, 16, 1), (256, 512, 8, 1) })
        {
            int pad = k / 2, o = (hw + 2 * pad - k) / 2 + 1;
            using var dy = backend.AllocateBuffer(Fill(batch * outC * o * o));
            using var w = backend.AllocateBuffer(Fill(outC * inC * k * k));
            using var dx = backend.AllocateBuffer(batch * inC * hw * hw);
            var a = new TunedConv2DBackwardArgs(dy, w, dx, false, batch, inC, hw, hw, outC, o, o, k, k, 2, 2, pad, pad, 1, 1);
            var chosen = backend.Conv2DBackwardDataSlot.Resolve(a.Shape(), a);
            Func<bool>[] run =
            {
                () => { chosen!.Execute(a); return true; },
                () => backend.Conv2DBackwardInputViaEngine(a),
                () => backend.Conv2DBackwardInputViaImplicit(a),
                () => backend.Conv2DBackwardInputViaCol2im(a),
            };
            var samples = routes.Select(_ => new List<double>()).ToArray();
            var declined = new bool[routes.Length];
            for (int r = 0; r < rounds; r++)
                for (int i = 0; i < run.Length; i++)
                {
                    if (declined[i]) continue;
                    if (!run[i]()) { declined[i] = true; continue; }
                    backend.Synchronize();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int b = 0; b < burst; b++) run[i]();
                    backend.Synchronize();
                    if (r > 0) samples[i].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / burst);
                }
            double Med(List<double> v) { var s = v.OrderBy(t => t).ToList(); return s[s.Count / 2]; }
            _out.WriteLine($"{k}x{k} s2 {inC}-{outC}@{hw} (slot: {chosen?.Id}): " + string.Join(" | ", routes.Select((rt, i) =>
                declined[i] || samples[i].Count == 0 ? $"{rt} declined" : $"{rt} {samples[i].Min():F1}/{Med(samples[i]):F1}")));
        }
    }
}
