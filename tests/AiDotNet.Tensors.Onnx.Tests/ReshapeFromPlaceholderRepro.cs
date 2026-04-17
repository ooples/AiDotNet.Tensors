using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Minimal repro of the bug that BERT's bert/embeddings/Reshape_2 hits:
/// a graph-input-like placeholder (allocated as an eager Tensor, filled
/// post-import, then read by a Reshape under GraphMode) shows only the
/// FIRST HALF of its data in the Reshape output. Suggests either a
/// stale-reference bug in the Reshape's captured closure or a length-
/// cache bug in the placeholder.
/// </summary>
public class ReshapeFromPlaceholderRepro
{
    private readonly ITestOutputHelper _output;
    public ReshapeFromPlaceholderRepro(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Reshape_OfFilledPlaceholder_PreservesAllElements()
    {
        var engine = new CpuEngine();
        CompiledInferencePlan<float> plan;
        var placeholder = new LinearAlgebra.Tensor<float>(new[] { 1, 256 });
        using (var scope = GraphMode.Enable())
        {
            var reshaped = engine.Reshape(placeholder, new[] { 256 });
            plan = scope.CompileInference<float>();
        }

        // Fill placeholder post-compile — this is the pattern the ONNX
        // importer uses for graph inputs.
        var dst = placeholder.AsWritableSpan();
        for (int i = 0; i < 128; i++) dst[i] = 0f;
        for (int i = 128; i < 256; i++) dst[i] = 1f;

        var output = plan.Execute();
        var ours = new float[output.AsSpan().Length];
        output.AsSpan().CopyTo(ours);

        _output.WriteLine($"Output length: {ours.Length}");
        _output.WriteLine($"  [0..5]:     {ours[0]} {ours[1]} {ours[2]} {ours[3]} {ours[4]}");
        _output.WriteLine($"  [125..131]: {ours[125]} {ours[126]} {ours[127]} {ours[128]} {ours[129]} {ours[130]} {ours[131]}");
        _output.WriteLine($"  [250..255]: {ours[250]} {ours[251]} {ours[252]} {ours[253]} {ours[254]} {ours[255]}");

        Assert.Equal(256, ours.Length);
        for (int i = 0; i < 128; i++) Assert.Equal(0f, ours[i]);
        for (int i = 128; i < 256; i++) Assert.Equal(1f, ours[i]);
    }
}
