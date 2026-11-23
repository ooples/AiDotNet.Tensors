#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The prelude's wgsl_tanh, wgsl_log and wgsl_log2 on Dawn against double-precision references: within a few ULP of
/// the answer everywhere, including the inputs Dawn's builtins get wrong (tanh near zero, log and log2 near one), the
/// branch points (tanh's 0.55, log's sqrt(1/2) and sqrt(2)) and the values the helpers hand to the builtin (zero,
/// negatives, infinities, NaN).
/// </summary>
public sealed class WebGpuShaderAccuracyTests
{
    private const string Kernel = @"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
struct Params { n: u32, op: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    let v = x[i];
    var r = 0.0;
    if (p.op == 0u) { r = tanh(v); } else if (p.op == 1u) { r = log(v); } else { r = log2(v); }
    y[i] = r;
}
";

    private const int MaxUlp = 4;

    private static IEnumerable<float> Around(double centre, int steps)
    {
        float c = (float)centre;
        for (int k = -steps; k <= steps; k++) yield return BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(c) + k);
    }

    private static float[] TanhInputs()
    {
        var values = new List<float>();
        for (int e = -30; e <= 1; e++) values.Add((float)Math.Pow(10, e));
        values.AddRange(Around(0.55, 64));
        for (double v = 0.01; v < 12; v *= 1.07) values.Add((float)v);
        values.AddRange(new[] { 1e-40f, 0f, 9.5f, 20f, 88f, 1e30f });
        return values.Concat(values.Select(v => -v)).ToArray();
    }

    private static float[] LogInputs()
    {
        var values = new List<float>();
        values.AddRange(Around(1.0, 256));
        values.AddRange(Around(Math.Sqrt(0.5), 64));
        values.AddRange(Around(Math.Sqrt(2.0), 64));
        for (double v = 1.2e-38; v < 3e38; v *= 3.7) values.Add((float)v);
        for (int e = -126; e <= 127; e += 7) values.Add((float)Math.Pow(2, e));
        values.AddRange(new[] { 1.17549435e-38f, float.MaxValue, 0.5f, 2f, 10f });
        return values.ToArray();
    }

    private static readonly float[] Specials = { 0f, -0f, -1f, -1e-20f, float.PositiveInfinity, float.NegativeInfinity, float.NaN };

    private static float[] Run(OpenClBackend host, DawnWgslLauncher dawn, int op, float[] input)
    {
        using var dx = host.AllocateBuffer(input);
        using var dy = host.AllocateBuffer(input.Length);
        dawn.For(host)("accuracy", input.Length, new object[] { dx, dy, input.Length, op });
        return host.DownloadBuffer(dy);
    }

    private static long UlpDistance(float a, float b)
    {
        static long Ordered(float f) { int bits = BitConverter.SingleToInt32Bits(f); return bits < 0 ? int.MinValue - (long)bits : bits; }
        return Math.Abs(Ordered(a) - Ordered(b));
    }

    private static void AssertAccurate(string what, float[] input, float[] output, Func<double, double> reference)
    {
        var failures = new List<string>();
        for (int i = 0; i < input.Length; i++)
        {
            float expected = (float)reference(input[i]);
            float actual = output[i];
            bool ok = float.IsNaN(expected) ? float.IsNaN(actual)
                : float.IsInfinity(expected) ? actual == expected
                : UlpDistance(expected, actual) <= MaxUlp
                  // a GPU may flush a subnormal input (and so its answer) to zero
                  || (float.IsSubnormal(input[i]) && actual == (float)reference(0.0));
            if (!ok) failures.Add($"{what}({input[i]:R}) = {actual:R}, expected {expected:R} ({UlpDistance(expected, actual)} ulp)");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(20)));
    }

    [SkippableFact]
    public void Helpers_are_within_a_few_ulp_of_the_double_reference_on_Dawn()
    {
        using var dawn = DawnWgslLauncher.TryCreate(new Dictionary<string, string> { ["accuracy"] = Kernel });
        Skip.If(dawn is null, "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        using var host = new OpenClBackend();
        Skip.IfNot(host.IsAvailable, "OpenCL hosts the buffers and is not available.");

        var tanh = TanhInputs().Concat(new[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN }).ToArray();
        AssertAccurate("tanh", tanh, Run(host, dawn!, 0, tanh), Math.Tanh);
        var log = LogInputs().Concat(Specials).ToArray();
        AssertAccurate("log", log, Run(host, dawn!, 1, log), Math.Log);
        AssertAccurate("log2", log, Run(host, dawn!, 2, log), Math.Log2);
    }
}
#endif
