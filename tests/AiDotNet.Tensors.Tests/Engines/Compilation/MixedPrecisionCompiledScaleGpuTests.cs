using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase D (Tensors #558): the compiled mixed-dtype plan trains at the design target scale (d512/L6/B256)
/// on the real GPU. Builds a 6-layer matmul stack with FP16 activations, compiles it into a
/// <see cref="MixedPrecisionCompiledPlan"/>, and runs several training steps — asserting the loss stays
/// finite, descends, and never NaNs, and reporting device VRAM. Skipped when no CUDA backend is active
/// (the activation-storage 0.500x ratio itself is proven deterministically in Fp16ActivationMemoryTests).
/// </summary>
[Collection(MixedPrecisionTestCollection.Name)] // serializes MixedPrecisionEmit.TestOverrideEnabled mutators
public class MixedPrecisionCompiledScaleGpuTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _out;
    public MixedPrecisionCompiledScaleGpuTests(ITestOutputHelper o) => _out = o;

    private static Tensor<float> Rand(int r, int c, int seed, double s)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * s);
        return new Tensor<float>(d, new[] { r, c });
    }

    private long? GpuUsedMiB()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
            using var p = Process.Start(psi);
            if (p is null) return null;
            string o = p.StandardOutput.ReadLine() ?? "";
            p.WaitForExit(5000);
            return long.TryParse(o.Trim(), out var v) ? v : (long?)null;
        }
        catch { return null; }
    }

    [Fact]
    public void CompiledMixedPrecision_TrainsAtTargetScale_OnGpu()
    {
        if (_engine is not DirectGpuTensorEngine)
        {
            _out.WriteLine($"skipped: engine={_engine.GetType().Name} (not a CUDA backend)");
            return;
        }

        const int B = 256, d = 512, L = 6;
        var x = Rand(B, d, 1, 1.0);
        var W = new Tensor<float>[L];
        for (int l = 0; l < L; l++) W[l] = Rand(d, d, 10 + l, 1.0 / Math.Sqrt(d));
        var target = Rand(B, d, 999, 0.5);

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            Tensor<float> lossT;
            var scope = new LazyTensorScope(null);
            using (new AutocastScope(PrecisionMode.Float16))
            {
                GraphMode.SetCurrent(scope);
                try
                {
                    var h = x;
                    for (int l = 0; l < L; l++) h = _engine.TensorMatMul(h, W[l]); // FP16 activations
                    var hh = h;
                    lossT = scope.RecordUnary<float>(LazyNodeType.Sum, "sqerr", hh, new[] { 1 },
                        (e, o) => { var ya = hh.ToArray(); var ta = target.ToArray(); float s = 0; for (int i = 0; i < ya.Length; i++) { float dd = ya[i] - ta[i]; s += dd * dd; } o.AsWritableSpan()[0] = s; },
                        (gradOut, inputs, output, state, e, grads) =>
                        {
                            var yt = inputs[0]; var ya = yt.ToArray(); var ta = target.ToArray(); float go = gradOut.ToArray()[0];
                            var g = new float[yt.Length];
                            for (int i = 0; i < g.Length; i++) g[i] = 2f * (ya[i] - ta[i]) * go;
                            grads[yt] = grads.TryGetValue(yt, out var ex) ? e.TensorAdd(ex, new Tensor<float>(g, yt._shape)) : new Tensor<float>(g, yt._shape);
                        });
                }
                finally { GraphMode.SetCurrent(null); }
            }

            var plan = MixedPrecisionCompiledPlan.Compile(lossT, _engine);
            var pars = new List<Tensor<float>>(W);

            float first = 0, last = 0; int nan = 0;
            const int steps = 10;
            for (int s = 0; s < steps; s++)
            {
                var r = plan.Step(pars, learningRate: 1e-3f / B, scaler: null);
                if (s == 0) first = r.Loss;
                if (s == steps - 1) last = r.Loss;
                if (float.IsNaN(r.Loss) || float.IsInfinity(r.Loss)) nan++;
            }

            long? vram = GpuUsedMiB();
            _out.WriteLine($"engine={_engine.GetType().Name} d={d} L={L} B={B}");
            _out.WriteLine($"loss first={first:F3} last={last:F3}  NaN/Inf steps={nan}");
            if (vram is not null) _out.WriteLine($"device VRAM used = {vram} MiB");

            static bool Finite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
            Assert.Equal(0, nan);
            Assert.True(first > 0 && Finite(first), "first loss finite positive");
            Assert.True(Finite(last), "final loss finite");
            Assert.True(last <= first, $"loss should not increase: first {first}, last {last}");
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }
    }
}
