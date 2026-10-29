using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// The inference plan's conv-BN fusion must not convolve in place. Memory planning runs before the fusion pass and is
/// liveness-based, so the BN's (or trailing activation's) output can be given the conv input's storage once the conv
/// input is dead. These ResNet-shaped graphs hit that layout; before the fix the first returned values that disagreed
/// with ONNX Runtime and changed from run to run, and the second disagreed at the ReLU after the residual add.
/// </summary>
public class ConvBnFusionAliasingTests
{
    [SkippableFact]
    public void ReluConvBn_AfterAStem_MatchesOnnxRuntime_AndIsDeterministic()
    {
        // torchvision resnet18's stem and the first half of its first basic block: conv-bn-relu-maxpool, conv-bn-relu, conv-bn.
        var g = new Graph("relu_conv_bn");
        var x = g.Input("X", 1, 3, 32, 32);
        var h = g.Op("Relu", g.Bn(g.Conv(x, 3, 64, 7, 2), 64));
        h = g.Op("MaxPool", new[] { h }, Ints("kernel_shape", 3, 3), Ints("strides", 2, 2), Ints("pads", 1, 1, 1, 1));
        h = g.Op("Relu", g.Bn(g.Conv(h, 64, 64, 3, 1), 64));
        var y = g.Bn(g.Conv(h, 64, 64, 3, 1), 64);
        AssertMatchesOrtAndRepeats(g.Build(y, 1, 64, 8, 8), x);
    }

    [SkippableFact]
    public void BasicBlock_ReluAfterTheResidualAdd_MatchesOnnxRuntime_AndIsDeterministic()
    {
        var g = new Graph("basic_block");
        var x = g.Input("X", 1, 8, 6, 6);
        var a = g.Op("Relu", g.Bn(g.Conv(x, 8, 8, 3, 1), 8));
        var sum = g.Op("Add", g.Bn(g.Conv(a, 8, 8, 3, 1), 8), x);
        var y = g.Op("Relu", sum);
        AssertMatchesOrtAndRepeats(g.Build(y, 1, 8, 6, 6), x);
    }

    private static void AssertMatchesOrtAndRepeats(byte[] model, (string Name, int[] Shape, float[] Data) x)
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var ort = OnnxRuntimeReference.RunSingleOutput(model, new[] { x });
        var first = ImportAndExecute(model, (x.Name, x.Data));
        AssertClose(ort, first, 1e-4f);
        for (int i = 0; i < 3; i++)
            Assert.Equal(first, ImportAndExecute(model, (x.Name, x.Data)));
    }

    private static AttributeProto Ints(string name, params long[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        a.Ints.AddRange(values);
        return a;
    }

    /// <summary>A seeded graph under construction; weights have variance 1/fan_in so activations stay O(1).</summary>
    private sealed class Graph
    {
        private readonly GraphProto _g;
        private int _next;
        private int _seed = 1;

        internal Graph(string name) => _g = new GraphProto { Name = name };

        internal (string Name, int[] Shape, float[] Data) Input(string name, params int[] shape)
        {
            _g.Input.Add(OnnxTestGraphBuilder.MakeValueInfo(name, shape, 1));
            return (name, shape, RandomArray(_seed++, shape.Aggregate(1, (p, d) => p * d)));
        }

        internal string Op(string type, params string[] inputs) => Op(type, inputs, Array.Empty<AttributeProto>());

        internal string Op(string type, string[] inputs, params AttributeProto[] attributes)
        {
            var node = new NodeProto { OpType = type, Name = type + "_" + _next };
            node.Input.AddRange(inputs);
            var output = "v" + _next++;
            node.Output.Add(output);
            node.Attribute.AddRange(attributes);
            _g.Node.Add(node);
            return output;
        }

        internal string Conv((string Name, int[] Shape, float[] Data) input, int cin, int cout, int k, int stride)
            => Conv(input.Name, cin, cout, k, stride);

        internal string Conv(string input, int cin, int cout, int k, int stride)
        {
            float b = (float)Math.Sqrt(3.0 / (cin * k * k));
            int p = k / 2;
            return Op("Conv", new[] { input, Weight(new[] { cout, cin, k, k }, -b, b) },
                Ints("kernel_shape", k, k), Ints("strides", stride, stride), Ints("pads", p, p, p, p));
        }

        internal string Bn(string input, int c) => Op("BatchNormalization", new[]
        {
            input, Weight(new[] { c }, 0.8f, 1.2f), Weight(new[] { c }, -0.1f, 0.1f),
            Weight(new[] { c }, -0.1f, 0.1f), Weight(new[] { c }, 0.8f, 1.2f),
        }, new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f });

        internal string Op(string type, string a, (string Name, int[] Shape, float[] Data) b) => Op(type, a, b.Name);

        internal byte[] Build(string output, params int[] shape)
        {
            var last = _g.Node[_g.Node.Count - 1];
            last.Output[0] = "Y";
            _g.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", shape, 1));
            return OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(_g, 17));
        }

        private string Weight(int[] shape, float lo, float hi)
        {
            var name = "w" + _next++;
            _g.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer(name, shape, RandomArray(_seed++, shape.Aggregate(1, (p, d) => p * d), lo, hi)));
            return name;
        }
    }
}
