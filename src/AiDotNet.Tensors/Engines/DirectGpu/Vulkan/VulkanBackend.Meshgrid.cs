// Copyright (c) AiDotNet. All rights reserved.
// Vulkan specialized two-vector Meshgrid2D (#923): one dispatch writes both grids.
using System;
using System.Collections.Concurrent;
using System.Globalization;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <remarks>
/// The established Vulkan route builds each grid with a reshape and a broadcast, one dispatch per grid. This route is
/// its fused peer for the two admitted shapes: a single dispatch reads both vectors and writes both grids, with the
/// extents compiled into the shader. It is an optimization only; any shape or state it declines keeps the composed
/// route.
/// </remarks>
public sealed unsafe partial class VulkanBackend : IDirectPtxVisionBackend
{
    private static readonly ConcurrentDictionary<(int Rows, int Columns, bool Xy), string> Meshgrid2DSources =
        new ConcurrentDictionary<(int Rows, int Columns, bool Xy), string>();

    /// <summary>
    /// False keeps every shape on the composed route: the baseline side of the route's benchmark
    /// (<c>--vulkan-meshgrid2d</c>), and a switch to rule the route out when diagnosing a result.
    /// </summary>
    internal static volatile bool Meshgrid2DFusedEnabled = true;

    /// <summary>The shapes the fused Meshgrid2D route is admitted for (those of the CUDA specialization).</summary>
    internal static bool IsMeshgrid2DAdmitted(int n0, int n1) => (n0, n1) is (256, 256) or (1024, 256);

    /// <summary>
    /// The fused shader for vectors of <paramref name="n0"/> and <paramref name="n1"/> elements. 'ij' grids are
    /// [n0, n1]: grid 0 repeats vector 0 down the rows, grid 1 repeats vector 1 across the columns. 'xy' grids are
    /// [n1, n0]: grid 0 repeats vector 0 across the columns, grid 1 repeats vector 1 down the rows.
    /// </summary>
    internal static string Meshgrid2DSource(int n0, int n1, bool xy)
    {
        int rows = xy ? n1 : n0, columns = xy ? n0 : n1;
        return Meshgrid2DSources.GetOrAdd((rows, columns, xy), key =>
        {
            string c = key.Columns.ToString(CultureInfo.InvariantCulture);
            string total = ((long)key.Rows * key.Columns).ToString(CultureInfo.InvariantCulture);
            return @"#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer V0 { float v0[]; };
layout(set = 0, binding = 1) readonly buffer V1 { float v1[]; };
layout(set = 0, binding = 2) writeonly buffer G0 { float g0[]; };
layout(set = 0, binding = 3) writeonly buffer G1 { float g1[]; };
layout(push_constant) uniform P { uint unused; };
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= " + total + @"u) return;
    uint r = i / " + c + @"u, c = i % " + c + @"u;
" + (key.Xy ? "    g0[i] = v0[c];\n    g1[i] = v1[r];\n" : "    g0[i] = v0[r];\n    g1[i] = v1[c];\n") + @"}
";
        });
    }

    bool IDirectPtxVisionBackend.CanDirectPtxMeshgrid2D(int n0, int n1, bool xy)
        => Meshgrid2DFusedEnabled && IsGlslCompilerAvailable && IsMeshgrid2DAdmitted(n0, n1);

    bool IDirectPtxVisionBackend.TryDirectPtxMeshgrid2DPair(IGpuBuffer source0, IGpuBuffer source1, IGpuBuffer output0,
        IGpuBuffer output1, int n0, int n1, bool xy)
    {
        if (!Meshgrid2DFusedEnabled || !IsGlslCompilerAvailable || !IsMeshgrid2DAdmitted(n0, n1)) return false;
        if (source0 is null) throw new ArgumentNullException(nameof(source0));
        if (source1 is null) throw new ArgumentNullException(nameof(source1));
        if (output0 is null) throw new ArgumentNullException(nameof(output0));
        if (output1 is null) throw new ArgumentNullException(nameof(output1));
        int total = checked(n0 * n1);
        if (source0.Size < n0) throw new ArgumentException($"source0 holds {source0.Size} elements, fewer than n0 = {n0}.", nameof(source0));
        if (source1.Size < n1) throw new ArgumentException($"source1 holds {source1.Size} elements, fewer than n1 = {n1}.", nameof(source1));
        if (output0.Size < total) throw new ArgumentException($"output0 holds {output0.Size} elements, fewer than n0 * n1 = {total}.", nameof(output0));
        if (output1.Size < total) throw new ArgumentException($"output1 holds {output1.Size} elements, fewer than n0 * n1 = {total}.", nameof(output1));
        // One buffer bound as two grids, or as a vector and a grid, would be read and written through different
        // bindings in the same dispatch.
        if (ReferenceEquals(output0, output1) || ReferenceEquals(source0, output0) || ReferenceEquals(source0, output1)
            || ReferenceEquals(source1, output0) || ReferenceEquals(source1, output1))
            return false;
        string glsl = Meshgrid2DSource(n0, n1, xy);
        // A device whose shader cannot be built keeps the composed route.
        if (GetOrCreateGlslPipeline(glsl, 4, sizeof(uint)) is null) return false;
        GlslQuadOp(glsl, source0, source1, output0, output1, total, new uint[] { 0u }, sizeof(uint));
        return true;
    }
}
