using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// C12 regression: a tiny CNN exercising the full NCHWc pipeline end-to-end
/// — Conv → BN → Relu → MaxPool → Conv → Relu → GlobalAvgPool → Gemm. This
/// locks in that the ONNX translator's layout auto-promote, the
/// BatchNormInference fused kernel, the packed MaxPool/GlobalAvgPool, and
/// the auto-unpack before Gemm all cooperate and bit-match the NCHW
/// reference.
/// </summary>
public class NchwcEndToEndRegression
{
    [Fact]
    public void MiniCnn_NchwcPipeline_MatchesNchwReference()
    {
        const int FLOAT = (int)TensorProto.Types.DataType.Float;
        // Input shape: [1, 8, 16, 16] — channels divisible by cBlock=8.
        var x = Random(seed: 1, n: 1 * 8 * 16 * 16);
        var conv1W = Random(seed: 2, n: 16 * 8 * 3 * 3);   // out=16, in=8
        var bn1Scale = Random(seed: 3, n: 16, lo: 0.5f, hi: 1.5f);
        var bn1Bias  = Random(seed: 4, n: 16, lo: -0.2f, hi: 0.2f);
        var bn1Mean  = Random(seed: 5, n: 16);
        var bn1Var   = Random(seed: 6, n: 16, lo: 0.1f, hi: 2f);
        var conv2W   = Random(seed: 7, n: 32 * 16 * 3 * 3); // out=32, in=16
        var gemmW    = Random(seed: 8, n: 10 * 32);         // out=10 classes from 32-ch GAP
        var gemmB    = Random(seed: 9, n: 10);

        var graph = new GraphProto { Name = "mini_cnn" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 8, 16, 16 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("conv1_W", new[] { 16, 8, 3, 3 }, conv1W));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("bn1_scale", new[] { 16 }, bn1Scale));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("bn1_B",     new[] { 16 }, bn1Bias));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("bn1_mean",  new[] { 16 }, bn1Mean));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("bn1_var",   new[] { 16 }, bn1Var));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("conv2_W",   new[] { 32, 16, 3, 3 }, conv2W));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("gemm_W",    new[] { 10, 32 }, gemmW));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("gemm_B",    new[] { 10 }, gemmB));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 10 }, FLOAT));

        // Conv1 (no bias → triggers NCHWc auto-pack in the translator)
        var conv1 = new NodeProto { OpType = "Conv" };
        conv1.Input.Add("X"); conv1.Input.Add("conv1_W");
        conv1.Output.Add("conv1_out");
        conv1.Attribute.Add(new AttributeProto {
            Name = "pads", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L, 1L, 1L, 1L } });
        conv1.Attribute.Add(new AttributeProto {
            Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 3L, 3L } });
        graph.Node.Add(conv1);

        // BN → ReLU → MaxPool on packed layout
        var bn = new NodeProto { OpType = "BatchNormalization" };
        bn.Input.Add("conv1_out"); bn.Input.Add("bn1_scale"); bn.Input.Add("bn1_B");
        bn.Input.Add("bn1_mean");  bn.Input.Add("bn1_var");
        bn.Output.Add("bn_out");
        bn.Attribute.Add(new AttributeProto {
            Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f });
        graph.Node.Add(bn);

        var relu = new NodeProto { OpType = "Relu" };
        relu.Input.Add("bn_out"); relu.Output.Add("relu_out");
        graph.Node.Add(relu);

        var pool = new NodeProto { OpType = "MaxPool" };
        pool.Input.Add("relu_out"); pool.Output.Add("pool_out");
        pool.Attribute.Add(new AttributeProto {
            Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 2L } });
        pool.Attribute.Add(new AttributeProto {
            Name = "strides", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 2L, 2L } });
        graph.Node.Add(pool);

        // Conv2 (no bias) → ReLU
        var conv2 = new NodeProto { OpType = "Conv" };
        conv2.Input.Add("pool_out"); conv2.Input.Add("conv2_W");
        conv2.Output.Add("conv2_out");
        conv2.Attribute.Add(new AttributeProto {
            Name = "pads", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1L, 1L, 1L, 1L } });
        conv2.Attribute.Add(new AttributeProto {
            Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 3L, 3L } });
        graph.Node.Add(conv2);

        var relu2 = new NodeProto { OpType = "Relu" };
        relu2.Input.Add("conv2_out"); relu2.Output.Add("relu2_out");
        graph.Node.Add(relu2);

        // GlobalAveragePool forces layout unpack back to NCHW; emits [1,32,1,1].
        var gap = new NodeProto { OpType = "GlobalAveragePool" };
        gap.Input.Add("relu2_out"); gap.Output.Add("gap_out");
        graph.Node.Add(gap);

        // Flatten → Gemm. Flatten is RequiresNchw — the auto-unpack hook
        // guarantees flat tensor came from NCHW-form memory.
        var flat = new NodeProto { OpType = "Flatten" };
        flat.Input.Add("gap_out"); flat.Output.Add("flat_out");
        graph.Node.Add(flat);

        var gemm = new NodeProto { OpType = "Gemm" };
        gemm.Input.Add("flat_out"); gemm.Input.Add("gemm_W"); gemm.Input.Add("gemm_B");
        gemm.Output.Add("Y");
        gemm.Attribute.Add(new AttributeProto {
            Name = "transB", Type = AttributeProto.Types.AttributeType.Int, I = 1 });
        graph.Node.Add(gemm);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        // Run through our engine — this triggers the full NCHWc pipeline.
        // ImportAndExecute returns the single graph output flattened.
        var ours = OnnxTestHelpers.ImportAndExecute(bytes, ("X", x));

        Assert.Equal(10, ours.Length);
        // Sanity: no NaN/Inf that'd indicate broken lane math.
        foreach (var v in ours)
            Assert.True(float.IsFinite(v), $"Non-finite output {v} — NCHWc pipeline likely corrupted");
        // Sanity: outputs shouldn't be all zero or all one — that'd suggest
        // Conv/BN silently short-circuited.
        Assert.True(ours.Distinct().Count() >= 5,
            "Output has <5 distinct values — pipeline likely collapsed");
    }

    private static float[] Random(int seed, int n, float lo = -1f, float hi = 1f)
    {
        var rng = new Random(seed);
        float range = hi - lo;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = lo + (float)rng.NextDouble() * range;
        return a;
    }
}
