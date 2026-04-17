using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Numerical-parity tests for the ONNX arithmetic translators (MatMul, Gemm,
/// Add, Mul, Sub, Div). Each test builds a single-node ONNX model, runs it
/// through both <c>Microsoft.ML.OnnxRuntime</c> (ground truth) and our
/// <see cref="OnnxImporter"/>, and asserts the outputs agree within the
/// Issue #169 tolerance of 1e-4.
/// </summary>
public class ArithOperatorTests
{
    private const float Tolerance = 1e-4f;
    private const int FLOAT = 1;

    [SkippableFact]
    public void Add_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "Add",
            inputs: new[] { ("A", new[] { 2, 3 }, FLOAT), ("B", new[] { 2, 3 }, FLOAT) },
            output: ("C", new[] { 2, 3 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { 1, 2, 3, 4, 5, 6 };
        var b = new float[] { 6, 5, 4, 3, 2, 1 };

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes,
            ("A", new[] { 2, 3 }, a), ("B", new[] { 2, 3 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));

        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Mul_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "Mul",
            inputs: new[] { ("A", new[] { 4 }, FLOAT), ("B", new[] { 4 }, FLOAT) },
            output: ("C", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { 0.5f, 1.5f, 2.5f, 3.5f };
        var b = new float[] { 2f, 2f, 2f, 2f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 4 }, a), ("B", new[] { 4 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void MatMul_2D_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] { ("A", new[] { 3, 4 }, FLOAT), ("B", new[] { 4, 5 }, FLOAT) },
            output: ("C", new[] { 3, 5 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var rng = new Random(42);
        var a = RandomArray(rng, 3 * 4);
        var b = RandomArray(rng, 4 * 5);
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 3, 4 }, a), ("B", new[] { 4, 5 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Gemm_WithBias_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Gemm: Y = 1*(A*B) + 1*C with transA=transB=0, all defaults.
        var graph = new GraphProto { Name = "gemm_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { 2, 3 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("B", new[] { 3, 4 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("C", new[] { 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4 }, FLOAT));
        var node = new NodeProto { OpType = "Gemm" };
        node.Input.Add("A"); node.Input.Add("B"); node.Input.Add("C");
        node.Output.Add("Y");
        graph.Node.Add(node);
        var model = OnnxTestGraphBuilder.WrapModel(graph);
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var rng = new Random(7);
        var a = RandomArray(rng, 6);
        var b = RandomArray(rng, 12);
        var c = RandomArray(rng, 4);
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes,
            ("A", new[] { 2, 3 }, a), ("B", new[] { 3, 4 }, b), ("C", new[] { 4 }, c));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b), ("C", c));
        AssertClose(ort, ours);
    }

    // ─── helpers ────────────────────────────────────────────────────────

    internal static float[] ImportAndExecute(byte[] modelBytes, params (string name, float[] data)[] inputs)
    {
        using var stream = new MemoryStream(modelBytes);
        var engine = new CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        foreach (var (name, data) in inputs)
        {
            var t = result.Inputs[name];
            data.AsSpan().CopyTo(t.AsWritableSpan());
        }
        var output = result.Plan!.Execute();
        var arr = new float[output.AsSpan().Length];
        output.AsSpan().CopyTo(arr);
        return arr;
    }

    internal static float[] RandomArray(Random rng, int n)
    {
        var arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);
        return arr;
    }

    internal static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            float scale = Math.Max(Math.Abs(expected[i]), 1f);
            Assert.True(diff <= Tolerance * scale,
                $"Element {i}: expected {expected[i]}, got {actual[i]}, diff={diff}, tol={Tolerance * scale}");
        }
    }
}
