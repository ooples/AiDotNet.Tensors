using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Minimum reproducer: a single lazy-graph plan with a placeholder input
/// and a trivial op, replayed twice with different input values. If the
/// second Execute produces the first execute's output rather than the
/// new one, the state bleed is a core-engine bug, not ONNX-specific.
/// </summary>
public class MinimalPlanRepro
{
    private readonly ITestOutputHelper _output;
    public MinimalPlanRepro(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void PlaceholderPlusConstant_ReflectsNewInput()
    {
        var engine = new CpuEngine();

        // Create a placeholder-like tensor (freshly allocated, user fills it).
        var placeholder = new Tensor<float>(new[] { 4 });
        var constant = new Tensor<float>(new[] { 4 }, new Vector<float>(new[] { 0f, 0f, 0f, 0f }));

        using var scope = GraphMode.Enable();
        var sum = engine.TensorAdd(placeholder, constant);
        var plan = scope.CompileInference<float>();

        // First execute: fill with 1s.
        placeholder.AsWritableSpan().Fill(1f);
        plan.Execute();
        var run1 = sum.AsSpan().ToArray();

        // Second execute: fill with 2s.
        placeholder.AsWritableSpan().Fill(2f);
        plan.Execute();
        var run2 = sum.AsSpan().ToArray();

        _output.WriteLine($"run1: [{string.Join(", ", run1)}]");
        _output.WriteLine($"run2: [{string.Join(", ", run2)}]");
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, run1);
        Assert.Equal(new[] { 2f, 2f, 2f, 2f }, run2);
    }

    [Fact]
    public void PlaceholderPlusZero_LikeOnnxWrap()
    {
        var engine = new CpuEngine();

        var placeholder = new Tensor<float>(new[] { 4 });
        using var scope = GraphMode.Enable();
        // Mimic OnnxImporter's TensorAdd(x, fresh_zero) wrap.
        var zero = new Tensor<float>(new[] { 4 });
        var wrapped = engine.TensorAdd(placeholder, zero);
        var plan = scope.CompileInference<float>();

        placeholder.AsWritableSpan().Fill(1f);
        plan.Execute();
        var run1 = wrapped.AsSpan().ToArray();

        placeholder.AsWritableSpan().Fill(2f);
        plan.Execute();
        var run2 = wrapped.AsSpan().ToArray();

        _output.WriteLine($"run1: [{string.Join(", ", run1)}]");
        _output.WriteLine($"run2: [{string.Join(", ", run2)}]");
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, run1);
        Assert.Equal(new[] { 2f, 2f, 2f, 2f }, run2);
    }

    [Fact]
    public void GatherEmbedding_ReflectsNewInput()
    {
        var engine = new CpuEngine();

        // 10 words, each 4-dim; indices into a [10,4] table.
        var embeddings = new Tensor<float>(new[] { 10, 4 });
        var embSpan = embeddings.AsWritableSpan();
        for (int i = 0; i < embSpan.Length; i++) embSpan[i] = i; // [0..39]

        // Float "placeholder" holding INDICES as floats (like ONNX Gather with float-encoded int64 placeholder).
        var indices = new Tensor<float>(new[] { 3 });

        using var scope = GraphMode.Enable();
        // Mirror ONNX Gather translator's deferred-cast closure.
        var capturedData = embeddings;
        var capturedIndices = indices;
        var outShape = new[] { 3, 4 };
        var gathered = scope!.RecordBinary(LazyNodeType.Custom, "Gather",
            capturedData, capturedIndices, outShape,
            (eng, output) =>
            {
                var idxSrc = capturedIndices.AsSpan();
                var idxArr = new int[idxSrc.Length];
                for (int i = 0; i < idxSrc.Length; i++) idxArr[i] = (int)Math.Round(idxSrc[i]);
                var intIndices = new Tensor<int>(capturedIndices._shape, new Vector<int>(idxArr));
                var r = eng.Gather(capturedData, intIndices, 0);
                r.AsSpan().CopyTo(output.AsWritableSpan());
            });
        var zero = new Tensor<float>(outShape);
        var wrapped = engine.TensorAdd(gathered, zero);
        var plan = scope.CompileInference<float>();

        // Run 1: indices [0, 1, 2] — expect rows 0,1,2 = [[0..3],[4..7],[8..11]]
        indices.AsWritableSpan()[0] = 0;
        indices.AsWritableSpan()[1] = 1;
        indices.AsWritableSpan()[2] = 2;
        plan.Execute();
        var run1 = wrapped.AsSpan().ToArray();

        // Run 2: indices [5, 6, 7]
        indices.AsWritableSpan()[0] = 5;
        indices.AsWritableSpan()[1] = 6;
        indices.AsWritableSpan()[2] = 7;
        plan.Execute();
        var run2 = wrapped.AsSpan().ToArray();

        _output.WriteLine($"run1: [{string.Join(", ", run1)}]");
        _output.WriteLine($"run2: [{string.Join(", ", run2)}]");
        Assert.Equal(new[] { 0f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, run1);
        Assert.Equal(new[] { 20f, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 }, run2);
    }

    [Fact]
    public void PlaceholderThroughReshape_ReflectsNewInput()
    {
        var engine = new CpuEngine();

        var placeholder = new Tensor<float>(new[] { 2, 2 });
        using var scope = GraphMode.Enable();
        var reshaped = engine.Reshape(placeholder, new[] { 4 });
        var zero = new Tensor<float>(new[] { 4 });
        var wrapped = engine.TensorAdd(reshaped, zero);
        var plan = scope.CompileInference<float>();

        placeholder.AsWritableSpan().Fill(1f);
        plan.Execute();
        var run1 = wrapped.AsSpan().ToArray();

        placeholder.AsWritableSpan().Fill(2f);
        plan.Execute();
        var run2 = wrapped.AsSpan().ToArray();

        _output.WriteLine($"run1: [{string.Join(", ", run1)}]");
        _output.WriteLine($"run2: [{string.Join(", ", run2)}]");
        Assert.Equal(new[] { 1f, 1f, 1f, 1f }, run1);
        Assert.Equal(new[] { 2f, 2f, 2f, 2f }, run2);
    }
}
