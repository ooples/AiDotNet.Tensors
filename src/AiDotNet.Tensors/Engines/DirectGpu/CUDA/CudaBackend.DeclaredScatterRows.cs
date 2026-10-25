// Copyright (c) AiDotNet. All rights reserved.
// The row-wise scatter mean and softmax through their declared programs (#890 phase 4e, #1441).

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    // AIDOTNET_DECLARED_SCATTER_ROWS=1 opts the row-wise scatter mean and softmax into the declared route; they are
    // off by default until verified on an NVIDIA GPU (see ScatterRowsDeclaredPromoted).
    private static readonly bool s_declaredScatterRowsOptIn =
        Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_SCATTER_ROWS") == "1";

    /// <summary>
    /// Whether the declared row-wise scatter mean and softmax (#1441) have been verified on an NVIDIA GPU. While false
    /// they run only with AIDOTNET_DECLARED_SCATTER_ROWS=1 (or <see cref="DeclaredScatterRowsSwitch"/>); the
    /// hand-written resident_scatter_mean_rows_counts / resident_scatter_softmax_rows serve every call otherwise. Set it
    /// to true once DeclaredScatterRowsRouteTests passes on the device: the routes are then on by default, like every
    /// other declared family, and AIDOTNET_DECLARED_KERNELS=0 still turns them off.
    /// </summary>
    internal const bool ScatterRowsDeclaredPromoted = false;

    /// <summary>Overrides AIDOTNET_DECLARED_SCATTER_ROWS for this backend (verification tooling and tests).</summary>
    internal bool? DeclaredScatterRowsSwitch { get; set; }

    private bool DeclaredScatterRowsEnabled =>
        DeclaredScatterRowsSwitch ?? (ScatterRowsDeclaredPromoted || s_declaredScatterRowsOptIn);

    /// <summary>
    /// The most row-scatter programs built per backend. Each bakes its three extents, so a workload that varies them
    /// would otherwise keep one set of modules per shape; past the cap a new shape keeps the hand-written kernel.
    /// </summary>
    internal const int DeclaredScatterRowsProgramCap = 32;

    private int _declaredScatterRowsPrograms;

    private static readonly string[] s_scatterMeanRowsNames = { "source", "indices", "counts", "out" };
    private static readonly string[] s_scatterSoftmaxRowsNames = { "indices", "source", "out" };

    private Func<DeclaredKey, CodegenProgram>? _buildScatterMeanRows;

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildScatterSoftmaxRows =
        key => CodegenBackendPrograms.ScatterSoftmaxRows(key.D0, key.D1, key.D2);

    // The mean's clamp calls the derived not_equal_scalar, so its builder needs this backend.
    private CodegenProgram BuildScatterMeanRows(DeclaredKey key)
    {
        var notEqual = (ResolveElementwise("not_equal_scalar")
            ?? throw new NotSupportedException("not_equal_scalar has no derived function: " +
                ElementwiseRefusal("not_equal_scalar"))).Handle;
        return CodegenBackendPrograms.ScatterMeanRows(key.D0, key.D1, key.D2, notEqual);
    }

    /// <summary>The key of a row-wise scatter: source rows, inner size, groups.</summary>
    internal static DeclaredKey ScatterRowsKey(DeclaredFamily family, int sourceRows, int innerSize, int groups) =>
        new(family, sourceRows, innerSize, groups, 0, 0, 0);

    // The scan is quadratic in the group count, as the hand-written kernels are; admitted only where that stays small,
    // like the declared scatter-add. One already built, or a new one within the cap, which it then claims.
    private bool AdmitScatterRows(DeclaredKey key, int sourceRows, int innerSize, int groups)
    {
        if (!DeclaredScatterRowsEnabled || sourceRows <= 0 || innerSize <= 0 || groups <= 0 ||
            (long)groups * sourceRows > 1L << 22)
            return false;
        lock (_directPtxLock)
        {
            if (_declaredPrograms.ContainsKey(key)) return true;
            if (_declaredScatterRowsPrograms >= DeclaredScatterRowsProgramCap) return false;
            _declaredScatterRowsPrograms++;
            return true;
        }
    }

    /// <summary>
    /// ScatterMeanRowsWithCounts through its declared program (#1441): the counts and the means, with no atomic. Off
    /// by default (<see cref="ScatterRowsDeclaredPromoted"/>). False -- nothing launched -- when off or declined.
    /// </summary>
    internal bool TryDeclaredScatterMeanRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        IGpuBuffer counts, int sourceRows, int innerSize, int outputRows)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterMeanRows, sourceRows, innerSize, outputRows);
        if (!AdmitScatterRows(key, sourceRows, innerSize, outputRows) || indices.Size < sourceRows ||
            (long)source.Size < (long)sourceRows * innerSize || (long)output.Size < (long)outputRows * innerSize ||
            counts.Size < outputRows)
            return false;
        return TryRunDeclaredProgram(key, _buildScatterMeanRows ??= BuildScatterMeanRows, s_scatterMeanRowsNames,
            new DeclaredBindings(source, indices, counts, output));
    }

    /// <summary>
    /// ScatterSoftmaxRows through its declared program (#1441), within one FP32 ulp of the hand-written kernel (it
    /// multiplies by the reciprocal sum where that divides). Off by default. False -- nothing launched -- when off or
    /// declined.
    /// </summary>
    internal bool TryDeclaredScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterSoftmaxRows, sourceRows, innerSize, numGroups);
        if (!AdmitScatterRows(key, sourceRows, innerSize, numGroups) || indices.Size < sourceRows ||
            (long)source.Size < (long)sourceRows * innerSize || (long)output.Size < (long)sourceRows * innerSize)
            return false;
        return TryRunDeclaredProgram(key, s_buildScatterSoftmaxRows, s_scatterSoftmaxRowsNames,
            new DeclaredBindings(indices, source, output));
    }

    /// <summary>The row-wise scatter mean as a resident-program stage.</summary>
    internal DeclaredResidentStage ScatterMeanRowsStage(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        IGpuBuffer counts, int sourceRows, int innerSize, int outputRows) =>
        new(ScatterRowsKey(DeclaredFamily.ScatterMeanRows, sourceRows, innerSize, outputRows),
            _buildScatterMeanRows ??= BuildScatterMeanRows, s_scatterMeanRowsNames,
            new DeclaredBindings(source, indices, counts, output));

    /// <summary>The row-wise scatter softmax as a resident-program stage.</summary>
    internal DeclaredResidentStage ScatterSoftmaxRowsStage(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups) =>
        new(ScatterRowsKey(DeclaredFamily.ScatterSoftmaxRows, sourceRows, innerSize, numGroups),
            s_buildScatterSoftmaxRows, s_scatterSoftmaxRowsNames, new DeclaredBindings(indices, source, output));
}
