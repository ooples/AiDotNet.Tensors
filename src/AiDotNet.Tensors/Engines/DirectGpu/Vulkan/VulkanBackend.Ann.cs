// Copyright (c) AiDotNet. All rights reserved.
// Vulkan launcher shims for the fused ANN kernels (IAnnBackend). Mirrors
// VulkanBackend.Detection.cs — pipelines are cached by GLSL source hash via
// GetOrCreateGlslPipeline, so the first call compiles SPIR-V and subsequent
// dispatches are O(1) lookups. All four ops use the 3-SSBO GlslBinaryOp
// plumbing (in0, in1, out) with an explicit push-constant block. The kernels
// are numerically faithful to AiDotNet.Tensors.Ann.AnnPrimitives.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IAnnBackend
{
    private const uint AnnPush4 = 4u * sizeof(uint); // numX, numY, dim/m, metric/...
    private const uint AnnPush5 = 5u * sizeof(uint); // PQ tables: +dsub

    private static int CheckedTotal(long total, string op)
    {
        if (total > int.MaxValue)
            throw new OverflowException($"ANN {op} total {total} exceeds Int32.MaxValue.");
        return (int)total;
    }

    /// <inheritdoc />
    public void ComputeDistances(IGpuBuffer queries, IGpuBuffer database, IGpuBuffer distances,
        int numQueries, int numDatabase, int dim, AnnMetric metric)
    {
        if (numQueries <= 0 || numDatabase <= 0) return;
        int total = CheckedTotal((long)numQueries * numDatabase, "ComputeDistances");
        var pc = new uint[] { (uint)numQueries, (uint)numDatabase, (uint)dim, (uint)metric };
        GlslBinaryOp(VulkanAnnKernels.ComputeDistances, queries, database, distances, total, pc, AnnPush4);
    }

    /// <inheritdoc />
    public void IvfAssign(IGpuBuffer vectors, IGpuBuffer centroids, IGpuBuffer assignments,
        int numVectors, int numCentroids, int dim, AnnMetric metric)
    {
        if (numVectors <= 0) return;
        if (numCentroids <= 0) throw new ArgumentException("numCentroids must be positive", nameof(numCentroids));
        var pc = new uint[] { (uint)numVectors, (uint)numCentroids, (uint)dim, (uint)metric };
        GlslBinaryOp(VulkanAnnKernels.IvfAssign, vectors, centroids, assignments, numVectors, pc, AnnPush4);
    }

    /// <inheritdoc />
    public void PqComputeDistanceTables(IGpuBuffer queries, IGpuBuffer codebooks, IGpuBuffer tables,
        int numQueries, int m, int ksub, int dsub, AnnMetric metric)
    {
        if (numQueries <= 0 || m <= 0 || ksub <= 0) return;
        int total = CheckedTotal((long)numQueries * m * ksub, "PqComputeDistanceTables");
        var pc = new uint[] { (uint)numQueries, (uint)m, (uint)ksub, (uint)dsub, (uint)metric };
        GlslBinaryOp(VulkanAnnKernels.PqDistanceTables, queries, codebooks, tables, total, pc, AnnPush5);
    }

    /// <inheritdoc />
    public void PqAdcScan(IGpuBuffer codes, IGpuBuffer tables, IGpuBuffer distances,
        int numQueries, int numCodes, int m, int ksub)
    {
        if (numQueries <= 0 || numCodes <= 0) return;
        int total = CheckedTotal((long)numQueries * numCodes, "PqAdcScan");
        var pc = new uint[] { (uint)numQueries, (uint)numCodes, (uint)m, (uint)ksub };
        GlslBinaryOp(VulkanAnnKernels.PqAdcScan, codes, tables, distances, total, pc, AnnPush4);
    }
}
