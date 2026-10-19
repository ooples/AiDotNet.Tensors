using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// <c>--parity-step mlp|cnn|resnet18 [steps]</c>: the head-to-head parity MLP (784-512-256-10, batch 128), CNN
/// (LeNet-style, batch 64) or ResNet-18 (32x32, batch 8, training-mode batch norm) training step on the CPU engine through the compiled training plan, outside any test host. A profiler
/// attached to the test host loses its method names (vstest ends the host before the runtime writes its rundown), so
/// this is the process to trace when a runner's parity ratio needs explaining (#1055). Thread count follows
/// AIDOTNET_CPU_THREADS.
/// </summary>
internal static class ParityStepProbe
{
    public static int Run(string[] args)
    {
        string network = args.Length > 1 ? args[1] : "mlp";
        int steps = args.Length > 2 ? int.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 300;
        if (network != "mlp" && network != "cnn" && network != "resnet18")
        {
            Console.Error.WriteLine($"--parity-step: unknown network '{network}' (mlp, cnn or resnet18).");
            return 2;
        }
        if (steps < 3)
        {
            Console.Error.WriteLine("--parity-step: steps must be at least 3.");
            return 2;
        }

        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var rng = new Random(1055);
        Tensor<float> Rand(int[] shape, double scale)
        {
            var data = new float[shape.Aggregate(1, (a, d) => a * d)];
            for (int i = 0; i < data.Length; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
            return Tensor<float>.FromMemory(data, shape);
        }

        var parameters = new List<Tensor<float>>();
        var layers = new List<Func<Tensor<float>, Tensor<float>>>();
        void Linear(int inF, int outF, bool relu)
        {
            var w = Rand(new[] { inF, outF }, 1.0 / Math.Sqrt(inF));
            var b = Rand(new[] { outF }, 1.0 / Math.Sqrt(inF));
            parameters.Add(w); parameters.Add(b);
            layers.Add(h => engine.FusedLinear(h, w, b, relu ? FusedActivationType.ReLU : FusedActivationType.None));
        }
        void Conv(int inC, int outC)
        {
            var w = Rand(new[] { outC, inC, 3, 3 }, 1.0 / Math.Sqrt(inC * 9));
            var b = Rand(new[] { outC }, 1.0 / Math.Sqrt(inC * 9));
            parameters.Add(w); parameters.Add(b);
            layers.Add(h => engine.FusedConv2D(h, w, b, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU));
        }

        Tensor<float> x, y;
        if (network == "resnet18")
        {
            // parity/networks/resnet18.json, built as the harness builds it: 3x3 stem, four stages of two basic blocks
            // (conv-BN-ReLU-conv-BN, a 1x1 projection when the shape changes), global average pool, linear head.
            Tensor<float> Ones(int n) => Tensor<float>.FromMemory(Enumerable.Repeat(1f, n).ToArray(), new[] { n });
            Tensor<float> Zeros(int n) => Tensor<float>.FromMemory(new float[n], new[] { n });
            Tensor<float> Norm(Tensor<float> h, Tensor<float> g, Tensor<float> b) => engine.BatchNorm(h, g, b, 1e-5, out _, out _);
            Tensor<float> ConvWeight(int o, int i, int k) { var w = Rand(new[] { o, i, k, k }, Math.Sqrt(2.0 / (i * k * k))); parameters.Add(w); return w; }
            (Tensor<float>, Tensor<float>) Bn(int c) { var g = Ones(c); var b = Zeros(c); parameters.Add(g); parameters.Add(b); return (g, b); }

            var stem = ConvWeight(64, 3, 3); var (sg, sb) = Bn(64);
            layers.Add(h => engine.ReLU(Norm(engine.FusedConv2D(h, stem, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None), sg, sb)));
            int channels = 64;
            foreach (var (outC, stride) in new[] { (64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1) })
            {
                int inC = channels, s0 = stride;
                var c1 = ConvWeight(outC, inC, 3); var (g1, b1) = Bn(outC);
                var c2 = ConvWeight(outC, outC, 3); var (g2, b2) = Bn(outC);
                bool project = s0 != 1 || inC != outC;
                Tensor<float>? pw = null, pg = null, pb = null;
                if (project) { pw = ConvWeight(outC, inC, 1); (pg, pb) = Bn(outC); }
                layers.Add(h =>
                {
                    var o = engine.ReLU(Norm(engine.FusedConv2D(h, c1, null, s0, s0, 1, 1, 1, 1, FusedActivationType.None), g1, b1));
                    o = Norm(engine.FusedConv2D(o, c2, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None), g2, b2);
                    var identity = pw is not null && pg is not null && pb is not null
                        ? Norm(engine.FusedConv2D(h, pw, null, s0, s0, 0, 0, 1, 1, FusedActivationType.None), pg, pb)
                        : h;
                    return engine.ReLU(engine.TensorAdd(o, identity));
                });
                channels = outC;
            }
            layers.Add(h => engine.Reshape(engine.GlobalAvgPool2D(h), new[] { 8, 512 }));
            Linear(512, 10, false);
            x = Rand(new[] { 8, 3, 32, 32 }, 1); y = Rand(new[] { 8, 10 }, 1);
        }
        else if (network == "cnn")
        {
            Conv(1, 32); layers.Add(h => engine.MaxPool2D(h, 2));
            Conv(32, 64); layers.Add(h => engine.MaxPool2D(h, 2));
            layers.Add(h => engine.Reshape(h, new[] { 64, 64 * 7 * 7 }));
            Linear(64 * 7 * 7, 128, true); Linear(128, 10, false);
            x = Rand(new[] { 64, 1, 28, 28 }, 1); y = Rand(new[] { 64, 10 }, 1);
        }
        else
        {
            Linear(784, 512, true); Linear(512, 256, true); Linear(256, 10, false);
            x = Rand(new[] { 128, 784 }, 1); y = Rand(new[] { 128, 10 }, 1);
        }

        var axes = new[] { 0, 1 };
        using var plan = TensorCodec.CompileTraining<float>(() =>
        {
            var h = x;
            foreach (var layer in layers) h = layer(h);
            engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(h, y)), axes, keepDims: false);
        }, parameters.ToArray());
        plan.ConfigureOptimizer(OptimizerType.SGD, 0.01);

        var times = new double[steps];
        var sw = new Stopwatch();
        for (int i = 0; i < steps; i++)
        {
            sw.Restart();
            plan.Step();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        var tail = times.Skip(steps / 3).OrderBy(t => t).ToArray();
        Console.WriteLine(FormattableString.Invariant(
            $"PARITY-STEP {network} steps={steps} threads={CpuParallelSettings.MaxDegreeOfParallelism} min_ms={tail[0]:F3} median_ms={tail[tail.Length / 2]:F3}"));
        return 0;
    }
}
