// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of DataTransformKernelTemplate, restated operation for operation. Each kernel is its own module: the
// storage buffers in the template's binding order, the ints in a uniform struct after them (padded to eight words);
// float parameters arrive as their bits. NaN is recognised by its bits, never by a self-comparison (WGSL lets an
// implementation assume NaN is absent).
internal static class WebGpuDataTransformKernels
{
    /// <summary>Words of each uniform struct (every kernel's ints, padded to 8).</summary>
    internal const int UniformWords = 8;

    private static readonly string BoxMullerSource = KernelSourceResources.Read("DataTransform.dt_box_muller.wgsl");

    private static readonly string ThresholdSource = KernelSourceResources.Read("DataTransform.dt_uniform_threshold.wgsl");

    private static readonly string LinearFilterSource = KernelSourceResources.Read("DataTransform.dt_linear_filter.wgsl");

    private static readonly string GatherRowsSource = KernelSourceResources.Read("DataTransform.dt_gather_rows.wgsl");

    private static readonly string ScatterRowsSource = KernelSourceResources.Read("DataTransform.dt_scatter_rows.wgsl");

    private static readonly string RoundIntegralSource = KernelSourceResources.Read("DataTransform.dt_round_integral.wgsl");

    private static readonly System.Collections.Generic.Dictionary<string, string> Sources = new()
    {
        [DataTransformKernelTemplate.BoxMuller] = BoxMullerSource,
        [DataTransformKernelTemplate.Threshold] = ThresholdSource,
        [DataTransformKernelTemplate.LinearFilter] = LinearFilterSource,
        [DataTransformKernelTemplate.GatherRows] = GatherRowsSource,
        [DataTransformKernelTemplate.ScatterRows] = ScatterRowsSource,
        [DataTransformKernelTemplate.RoundIntegral] = RoundIntegralSource,
    };

    /// <summary>The WGSL module for one kernel name, its entry point named after the kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];
}
