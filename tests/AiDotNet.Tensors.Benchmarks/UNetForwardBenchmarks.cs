using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Model-level benchmark simulating a simplified UNet forward pass at SD15 dimensions.
/// Measures memory planning effectiveness and workspace allocation savings.
/// </summary>
[MemoryDiagnoser]
[ShortRunJob]
public class UNetForwardBenchmarks
{
    /// <summary>
    /// Demonstrates MemoryPlanner computing minimum workspace for a UNet.
    /// Measures memory savings from tensor lifetime analysis + buffer aliasing.
    /// </summary>
    [Benchmark]
    public MemoryPlanner.MemoryPlan UNet_MemoryPlanning()
    {
        var planner = new MemoryPlanner();

        // Input: latent [1, 4, 64, 64]
        int input = planner.AddExternalInput([1, 4, 64, 64]);

        // Encoder Level 0: Conv + GroupNorm + SiLU
        int enc0_conv = planner.AddOp("Conv2D", [input], [1, 320, 64, 64]);
        int enc0_gn = planner.AddOp("GroupNorm", [enc0_conv], [1, 320, 64, 64]);
        int enc0_act = planner.AddOp("SiLU", [enc0_gn], [1, 320, 64, 64], canInPlace: true);

        // Encoder Level 1: Downsample + Conv + GroupNorm + SiLU
        int enc1_down = planner.AddOp("Conv2D_stride2", [enc0_act], [1, 320, 32, 32]);
        int enc1_conv = planner.AddOp("Conv2D", [enc1_down], [1, 640, 32, 32]);
        int enc1_gn = planner.AddOp("GroupNorm", [enc1_conv], [1, 640, 32, 32]);
        int enc1_act = planner.AddOp("SiLU", [enc1_gn], [1, 640, 32, 32], canInPlace: true);

        // Encoder Level 2: Downsample + Conv + GroupNorm + SiLU
        int enc2_down = planner.AddOp("Conv2D_stride2", [enc1_act], [1, 640, 16, 16]);
        int enc2_conv = planner.AddOp("Conv2D", [enc2_down], [1, 1280, 16, 16]);
        int enc2_gn = planner.AddOp("GroupNorm", [enc2_conv], [1, 1280, 16, 16]);
        int enc2_act = planner.AddOp("SiLU", [enc2_gn], [1, 1280, 16, 16], canInPlace: true);

        // Middle block
        int mid_conv1 = planner.AddOp("Conv2D", [enc2_act], [1, 1280, 16, 16]);
        int mid_gn = planner.AddOp("GroupNorm", [mid_conv1], [1, 1280, 16, 16]);
        int mid_act = planner.AddOp("SiLU", [mid_gn], [1, 1280, 16, 16], canInPlace: true);
        int mid_conv2 = planner.AddOp("Conv2D", [mid_act], [1, 1280, 16, 16]);
        int mid_res = planner.AddOp("Add", [enc2_act, mid_conv2], [1, 1280, 16, 16]);

        // Decoder Level 2: Upsample + Conv + GroupNorm + SiLU + Skip
        int dec2_up = planner.AddOp("Upsample", [mid_res], [1, 1280, 32, 32]);
        int dec2_cat = planner.AddOp("Concat", [dec2_up, enc1_act], [1, 1920, 32, 32]);
        int dec2_conv = planner.AddOp("Conv2D", [dec2_cat], [1, 640, 32, 32]);
        int dec2_gn = planner.AddOp("GroupNorm", [dec2_conv], [1, 640, 32, 32]);
        int dec2_act = planner.AddOp("SiLU", [dec2_gn], [1, 640, 32, 32], canInPlace: true);

        // Decoder Level 1: Upsample + Conv + GroupNorm + SiLU + Skip
        int dec1_up = planner.AddOp("Upsample", [dec2_act], [1, 640, 64, 64]);
        int dec1_cat = planner.AddOp("Concat", [dec1_up, enc0_act], [1, 960, 64, 64]);
        int dec1_conv = planner.AddOp("Conv2D", [dec1_cat], [1, 320, 64, 64]);
        int dec1_gn = planner.AddOp("GroupNorm", [dec1_conv], [1, 320, 64, 64]);
        int dec1_act = planner.AddOp("SiLU", [dec1_gn], [1, 320, 64, 64], canInPlace: true);

        // Output projection
        int output = planner.AddOp("Conv1x1", [dec1_act], [1, 4, 64, 64]);

        return planner.Plan();
    }

    /// <summary>
    /// Demonstrates graph capture + compile + cache for a UNet.
    /// </summary>
    [Benchmark]
    public MemoryPlanner.MemoryPlan UNet_GraphCaptureAndCompile()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();

        int input = graph.RecordInput([1, 4, 64, 64]);

        // Simplified encoder
        int enc0 = graph.RecordOp(ComputationGraph.OpType.Conv2D, [input], [1, 320, 64, 64]);
        int enc0_act = graph.RecordOp(ComputationGraph.OpType.SiLU, [enc0], [1, 320, 64, 64]);

        int enc1 = graph.RecordOp(ComputationGraph.OpType.Conv2D, [enc0_act], [1, 640, 32, 32]);
        int enc1_act = graph.RecordOp(ComputationGraph.OpType.SiLU, [enc1], [1, 640, 32, 32]);

        int enc2 = graph.RecordOp(ComputationGraph.OpType.Conv2D, [enc1_act], [1, 1280, 16, 16]);
        int enc2_act = graph.RecordOp(ComputationGraph.OpType.SiLU, [enc2], [1, 1280, 16, 16]);

        // Middle
        int mid = graph.RecordOp(ComputationGraph.OpType.Conv2D, [enc2_act], [1, 1280, 16, 16]);
        int mid_act = graph.RecordOp(ComputationGraph.OpType.SiLU, [mid], [1, 1280, 16, 16]);
        int mid_res = graph.RecordOp(ComputationGraph.OpType.Residual, [enc2_act, mid_act], [1, 1280, 16, 16]);

        // Decoder with skip connections
        int dec2 = graph.RecordOp(ComputationGraph.OpType.Upsample, [mid_res], [1, 1280, 32, 32]);
        int dec2_conv = graph.RecordOp(ComputationGraph.OpType.Conv2D, [dec2], [1, 640, 32, 32]);

        int dec1 = graph.RecordOp(ComputationGraph.OpType.Upsample, [dec2_conv], [1, 640, 64, 64]);
        int dec1_conv = graph.RecordOp(ComputationGraph.OpType.Conv2D, [dec1], [1, 320, 64, 64]);

        int output = graph.RecordOp(ComputationGraph.OpType.Conv2D, [dec1_conv], [1, 4, 64, 64]);
        graph.RecordOutput(output);

        graph.EndCapture();

        return graph.Optimize();
    }

    /// <summary>
    /// Compiled graph cache hit performance — measures reuse cost.
    /// </summary>
    [Benchmark]
    public MemoryPlanner.MemoryPlan UNet_CachedGraphLookup()
    {
        var cache = new CompiledGraphCache();

        // First call: compile
        var graph = BuildSimpleUNetGraph();
        cache.GetOrCompile(graph, [[1, 4, 64, 64]]);

        // Second call: cache hit (this is what we benchmark)
        var graph2 = BuildSimpleUNetGraph();
        return cache.GetOrCompile(graph2, [[1, 4, 64, 64]]);
    }

    private static ComputationGraph BuildSimpleUNetGraph()
    {
        var graph = new ComputationGraph();
        graph.BeginCapture();
        int input = graph.RecordInput([1, 4, 64, 64]);
        int enc = graph.RecordOp(ComputationGraph.OpType.Conv2D, [input], [1, 320, 64, 64]);
        int mid = graph.RecordOp(ComputationGraph.OpType.Conv2D, [enc], [1, 1280, 16, 16]);
        int dec = graph.RecordOp(ComputationGraph.OpType.Conv2D, [mid], [1, 320, 64, 64]);
        int output = graph.RecordOp(ComputationGraph.OpType.Conv2D, [dec], [1, 4, 64, 64]);
        graph.RecordOutput(output);
        graph.EndCapture();
        return graph;
    }
}
