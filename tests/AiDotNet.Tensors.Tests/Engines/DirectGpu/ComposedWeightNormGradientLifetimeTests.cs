using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Exercises the production primitive composition on the current baseline, independent of the future fused operation.</summary>
[Collection("DirectGpuSerial")]
public sealed class ComposedWeightNormGradientLifetimeTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public ComposedWeightNormGradientLifetimeTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    public static IEnumerable<object[]> Cases()
    {
        foreach (var (rows, columns, layers) in new[] { (2, 3, 1), (2, 257, 6), (8, 17, 6), (64, 1024, 6) })
            for (int storage = 0; storage < 3; storage++) yield return new object[] { rows, columns, layers, storage };
    }

    public static IEnumerable<object[]> FinalizerCases()
    {
        foreach (var (rows, columns) in new[] { (2, 257), (8, 17) })
            for (int storage = 0; storage < 3; storage++) yield return new object[] { rows, columns, storage };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CpuPrimitiveCompositionRetainsEveryParameterGradient(int rows, int columns, int layers, int storage)
        => Verify(new CpuEngine(), rows, columns, layers, storage);

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void CudaPrimitiveCompositionRetainsEveryParameterGradient(int rows, int columns, int layers, int storage)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend available.");
        var engine = _fixture.Engine ?? throw new InvalidOperationException();
        Skip.IfNot(engine.Name.IndexOf("CUDA", StringComparison.OrdinalIgnoreCase) >= 0, "This regression requires CUDA.");
        Verify(engine, rows, columns, layers, storage);
    }

    [Theory]
    [MemberData(nameof(FinalizerCases))]
    public void CpuUnmaterializedComposedGradientsSurviveProducerFinalizers(int rows, int columns, int storage)
        => Verify(new CpuEngine(), rows, columns, layers: 6, storage, forceFinalizers: true);

    [SkippableTheory]
    [MemberData(nameof(FinalizerCases))]
    public void CudaUnmaterializedComposedGradientsSurviveProducerFinalizers(int rows, int columns, int storage)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend available.");
        var engine = _fixture.Engine ?? throw new InvalidOperationException();
        Skip.IfNot(engine.Name.IndexOf("CUDA", StringComparison.OrdinalIgnoreCase) >= 0, "This regression requires CUDA.");
        Verify(engine, rows, columns, layers: 6, storage, forceFinalizers: true);
    }

    private static void Verify(IEngine engine, int rows, int columns, int layers, int storage, bool forceFinalizers = false)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var data = new Inputs(rows, columns, layers);
            // Production constructs and evaluates the model before its isolated training-gradient check.
            // Warm resident primitive paths without a tape, using distinct owning tensors.
            using (var noGrad = new NoGradScope<float>())
            {
                var warm = new Parameters(data, storage: 0, createViews: true);
                var value = new Tensor<float>((float[])data.Initial.Clone(), data.Shape);
                for (int layer = 0; layer < layers; layer++)
                    value = engine.TensorAdd(value, engine.TensorMultiply(value,
                        ComposedKernel(engine, warm.Direction[layer], warm.Gain[layer])));
                Assert.True(float.IsFinite(value[0]));
            }
            // First read at ComputeGradients return; second round deliberately leaves the result unread until after
            // tape disposal and unrelated GPU work, so the first read cannot mask a deferred-buffer lifetime error.
            for (int round = 0; round < 2; round++)
            {
                var parameters = new Parameters(data, storage, createViews: storage != 2);
                List<(string Name, Tensor<float> Actual, double[] Expected)> retained;
                Dictionary<Tensor<float>, Tensor<float>> gradients;
                using (var tape = new GradientTape<float>())
                {
                    if (storage == 2) parameters.CreateViews();
                    var activation = new Tensor<float>((float[])data.Initial.Clone(), data.Shape);
                    for (int layer = 0; layer < layers; layer++)
                    {
                        var weights = ComposedKernel(engine, parameters.Direction[layer], parameters.Gain[layer]);
                        // A sequence of actual differentiable operators gives earlier-produced gradients time to
                        // survive the rest of backward; each layer has an independent scalar reference below.
                        activation = engine.TensorAdd(activation, engine.TensorMultiply(activation, weights));
                    }
                    var seed = new Tensor<float>((float[])data.Seed.Clone(), data.Shape);
                    var loss = engine.ReduceSum(engine.TensorMultiply(activation, seed), null, false);
                    var sources = parameters.Sources();
                    gradients = tape.ComputeGradients(loss, sources);
                    retained = new List<(string, Tensor<float>, double[])>();
                    var expected = data.ExpectedGradients();
                    if (storage == 2)
                    {
                        Assert.True(gradients.TryGetValue(parameters.Parent!, out var parentGradient), "Missing flat-parent gradient.");
                        var flat = new double[parameters.Parent!.Length];
                        for (int layer = 0; layer < layers; layer++)
                        {
                            Array.Copy(expected[layer].Direction, 0, flat, parameters.Start(layer), data.Length);
                            Array.Copy(expected[layer].Gain, 0, flat, parameters.Start(layer) + data.Length, rows);
                        }
                        retained.Add(("flat-parent", parentGradient!, flat));
                    }
                    else
                        for (int layer = 0; layer < layers; layer++)
                        {
                            Assert.True(gradients.TryGetValue(parameters.Direction[layer], out var dv), $"Missing direction gradient {layer}.");
                            Assert.True(gradients.TryGetValue(parameters.Gain[layer], out var dg), $"Missing gain gradient {layer}.");
                            retained.Add(($"layer{layer}/direction", dv!, expected[layer].Direction));
                            retained.Add(($"layer{layer}/gain", dg!, expected[layer].Gain));
                        }
                    if (round == 0) Compare(retained, "at-return");
                }
                ApplyUnrelatedPressure(engine, data.Shape);
                if (forceFinalizers)
                {
                    // The dictionary and every returned gradient remain strongly reachable while producer
                    // temporaries/finalizers are collected. In round one no gradient host read has happened.
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                }
                Compare(retained, forceFinalizers ? "after-producer-finalizers" :
                    round == 0 ? "after-more-ops" : "first-read-after-more-ops");
                GC.KeepAlive(gradients);
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    // Exact old NormedConv1DLayer composition: no semantic fused forward or VJP is called here.
    private static Tensor<float> ComposedKernel(IEngine engine, Tensor<float> direction, Tensor<float> gain)
    {
        int rows = direction.Shape[0], columns = direction.Length / rows;
        var matrix = engine.Reshape(direction, new[] { rows, columns });
        var inverse = engine.TensorPow(engine.TensorAddScalar(
            engine.ReduceSum(engine.TensorMultiply(matrix, matrix), new[] { 1 }, keepDims: true), 1e-24f), -.5f);
        var scale = engine.TensorMultiply(engine.Reshape(gain, new[] { rows, 1 }), inverse);
        var result = engine.Reshape(engine.TensorMultiply(matrix, scale), direction.Shape.ToArray());
        engine.InvalidatePersistentTensor(result);
        return result;
    }

    private static void ApplyUnrelatedPressure(IEngine engine, int[] shape)
    {
        using var noGrad = new NoGradScope<float>();
        var input = new Tensor<float>(shape); input.Fill(.25f);
        for (int i = 0; i < 12; i++)
        {
            using var scope = TensorScope.Begin();
            var value = engine.TensorAddScalar(input, (i + 1) * .03125f);
            value = engine.TensorMultiplyScalar(value, .75f);
            value = engine.ReLU(value);
            Assert.True(float.IsFinite(scope.Keep(value)[0]));
        }
        // This supported API promises materialize-before-free for held outputs. Do not use DropActivationCache,
        // whose contract explicitly discards dead values, or manufacture a GPU OOM to provoke collection.
        if (engine is DirectGpuTensorEngine gpu) gpu.ClearActivationCache();
    }

    private static void Compare(IEnumerable<(string Name, Tensor<float> Actual, double[] Expected)> values, string stage)
    {
        foreach (var (name, tensor, expected) in values)
        {
            var actual = tensor.ToArray();
            Assert.Equal(expected.Length, actual.Length);
            double difference = 0, squared = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]), $"{stage}/{name}[{i}] is nonfinite.");
                double error = actual[i] - expected[i];
                difference += error * error;
                squared += expected[i] * expected[i];
                Assert.True(Math.Abs(error) <= 3e-5 + 3e-4 * Math.Abs(expected[i]),
                    $"{stage}/{name}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}.");
            }
            double relative = Math.Sqrt(difference / Math.Max(squared, 1e-30));
            Assert.True(relative <= 5e-4, $"{stage}/{name}: relative L2 {relative:R}; a retained gradient may have vanished or been overwritten.");
        }
    }

    private sealed class Parameters
    {
        private readonly Inputs _data;
        private readonly int _storage;
        internal readonly Tensor<float>[] Direction, Gain;
        internal readonly Tensor<float>? Parent;
        internal int Start(int layer) => 7 + layer * (_data.Length + _data.Rows);

        internal Parameters(Inputs data, int storage, bool createViews)
        {
            _data = data; _storage = storage;
            Direction = new Tensor<float>[data.Layers]; Gain = new Tensor<float>[data.Layers];
            if (storage != 0)
            {
                var flat = new float[7 + data.Layers * (data.Length + data.Rows) + 5];
                for (int i = 0; i < flat.Length; i++) flat[i] = 19;
                for (int layer = 0; layer < data.Layers; layer++)
                {
                    Array.Copy(data.Direction[layer], 0, flat, Start(layer), data.Length);
                    Array.Copy(data.Gain[layer], 0, flat, Start(layer) + data.Length, data.Rows);
                }
                Parent = new Tensor<float>(flat, new[] { flat.Length });
            }
            if (createViews) CreateViews();
        }

        internal void CreateViews()
        {
            for (int layer = 0; layer < _data.Layers; layer++)
                if (_storage == 0)
                {
                    Direction[layer] = new Tensor<float>((float[])_data.Direction[layer].Clone(), _data.Shape);
                    Gain[layer] = new Tensor<float>((float[])_data.Gain[layer].Clone(), new[] { _data.Rows });
                }
                else
                {
                    Direction[layer] = Parent!.Slice(0, Start(layer), Start(layer) + _data.Length).Reshape(_data.Shape);
                    Gain[layer] = Parent.Slice(0, Start(layer) + _data.Length, Start(layer) + _data.Length + _data.Rows);
                    Assert.Equal(Start(layer), Direction[layer].StorageOffset);
                    Assert.Same(Direction[layer].StorageIdentity, Gain[layer].StorageIdentity);
                }
        }

        internal Tensor<float>[] Sources() => _storage == 2 ? new[] { Parent! }
            : Direction.Zip(Gain, (v, g) => new[] { v, g }).SelectMany(x => x).ToArray();
    }

    private sealed class Inputs
    {
        internal readonly int Rows, Columns, Layers, Length;
        internal readonly int[] Shape;
        internal readonly float[][] Direction, Gain;
        internal readonly float[] Initial, Seed;

        internal Inputs(int rows, int columns, int layers)
        {
            Rows = rows; Columns = columns; Layers = layers; Length = rows * columns;
            Shape = new[] { rows, 1, 1, columns };
            Direction = new float[layers][]; Gain = new float[layers][];
            for (int layer = 0; layer < layers; layer++)
            {
                Direction[layer] = Enumerable.Range(0, Length).Select(i => (float)(.2 + .07 * Math.Sin(i * .13 + layer * .2))).ToArray();
                Gain[layer] = Enumerable.Range(0, rows).Select(i => i == 1 && layer % 2 == 0 ? 0f
                    : (float)((i % 2 == 0 ? 1 : -1) * (.8 + .02 * i - .05 * layer))).ToArray();
            }
            Initial = Enumerable.Range(0, Length).Select(i => (float)(.1 + .05 * Math.Cos(i * .17))).ToArray();
            Seed = Enumerable.Range(0, Length).Select(i => (float)(.2 * Math.Sin(i * .31 + .2) + .03 * (i % 5))).ToArray();
        }

        internal (double[] Direction, double[] Gain)[] ExpectedGradients()
        {
            var weights = new double[Layers][];
            var inverse = new double[Layers][];
            for (int layer = 0; layer < Layers; layer++)
            {
                weights[layer] = new double[Length]; inverse[layer] = new double[Rows];
                for (int row = 0; row < Rows; row++)
                {
                    double squared = 0;
                    for (int col = 0; col < Columns; col++) { double v = Direction[layer][row * Columns + col]; squared += v * v; }
                    inverse[layer][row] = 1 / Math.Sqrt(squared + 1e-24);
                    for (int col = 0; col < Columns; col++)
                        weights[layer][row * Columns + col] = Direction[layer][row * Columns + col] * Gain[layer][row] * inverse[layer][row];
                }
            }
            var result = new (double[], double[])[Layers];
            for (int layer = 0; layer < Layers; layer++)
            {
                var upstream = new double[Length];
                for (int i = 0; i < Length; i++)
                {
                    double value = (double)Seed[i] * Initial[i];
                    for (int other = 0; other < Layers; other++) if (other != layer) value *= 1 + weights[other][i];
                    upstream[i] = value;
                }
                var dv = new double[Length]; var dg = new double[Rows];
                for (int row = 0; row < Rows; row++)
                {
                    double dot = 0, r = inverse[layer][row];
                    for (int col = 0; col < Columns; col++) { int i = row * Columns + col; dot += upstream[i] * Direction[layer][i]; }
                    dg[row] = r * dot;
                    for (int col = 0; col < Columns; col++)
                    {
                        int i = row * Columns + col;
                        dv[i] = Gain[layer][row] * r * (upstream[i] - Direction[layer][i] * dot * r * r);
                    }
                }
                result[layer] = (dv, dg);
            }
            return result;
        }
    }
}
