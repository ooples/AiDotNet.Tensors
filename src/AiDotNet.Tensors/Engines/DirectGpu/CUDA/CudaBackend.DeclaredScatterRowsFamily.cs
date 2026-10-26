// Copyright (c) AiDotNet. All rights reserved.
// The rest of the row-wise scatter family through declared programs (#890 phase 4e, #1441): the scatter maximum with
// its argmax, the backward passes, the row-wise add (the deterministic scatter-add's program), its backward (the
// embedding's gather), and the activation-layout ScatterMean. They share the scatter rows' switch, cap and size limit.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private static readonly string[] s_scatterMaxArgmaxRowsNames = { "indices", "source", "out", "argmax" };
    private static readonly string[] s_scatterMaxBackwardRowsNames = { "argmax", "gradOutput", "gradSource" };
    private static readonly string[] s_scatterMeanBackwardRowsNames = { "counts", "indices", "gradOutput", "gradSource" };
    private static readonly string[] s_scatterSoftmaxBackwardRowsNames = { "indices", "output", "gradOutput", "gradSource" };
    private static readonly string[] s_scatterMeanActivationNames = { "source", "indices", "counts", "output" };

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildScatterMaxArgmaxRows =
        key => CodegenBackendPrograms.ScatterMaxArgmaxRows(key.D0, key.D1, key.D2);

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildScatterMaxBackwardRows =
        key => CodegenBackendPrograms.ScatterMaxBackwardRows(key.D0, key.D1, key.D2);

    private Func<DeclaredKey, CodegenProgram>? _buildScatterMeanBackwardRows;
    private Func<DeclaredKey, CodegenProgram>? _buildScatterSoftmaxBackwardRows;
    private Func<DeclaredKey, CodegenProgram>? _buildScatterMeanActivation;

    // The division the hand-written kernels' '/' compiles to under the same flags: the derived divide_vectors.
    private CodegenLibraryFunction DivideVectors() =>
        (ResolveElementwise("divide_vectors")
            ?? throw new NotSupportedException("divide_vectors has no derived function: " +
                ElementwiseRefusal("divide_vectors"))).Handle;

    private CodegenProgram BuildScatterMeanBackwardRows(DeclaredKey key) =>
        CodegenBackendPrograms.ScatterMeanBackwardRows(key.D0, key.D1, key.D2, NotEqualScalar(), DivideVectors());

    private CodegenProgram BuildScatterSoftmaxBackwardRows(DeclaredKey key) =>
        CodegenBackendPrograms.ScatterSoftmaxBackwardRows(key.D0, key.D1, key.D2, NotEqualScalar());

    private CodegenProgram BuildScatterMeanActivation(DeclaredKey key) =>
        CodegenBackendPrograms.ScatterMeanActivation(key.D0, key.D1, key.D2, NotEqualScalar(), DivideVectors());

    /// <summary>ScatterMaxWithArgmaxRows through its declared program (#1441). False -- nothing launched -- when declined.</summary>
    internal bool TryDeclaredScatterMaxArgmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        IGpuBuffer argmax, int sourceRows, int innerSize, int outputRows)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterMaxArgmaxRows, sourceRows, innerSize, outputRows);
        long cells = (long)outputRows * innerSize;
        if (indices.Size < sourceRows || (long)source.Size < (long)sourceRows * innerSize || output.Size < cells ||
            argmax.Size < cells || !AdmitScatterRows(key, sourceRows, innerSize, outputRows))
            return false;
        return TryRunDeclaredProgram(key, s_buildScatterMaxArgmaxRows, s_scatterMaxArgmaxRowsNames,
            new DeclaredBindings(indices, source, output, argmax), residentWhenSmall: true);
    }

    /// <summary>ScatterMaxBackwardRows through its declared program (#1441). False when declined.</summary>
    internal bool TryDeclaredScatterMaxBackwardRows(IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterMaxBackwardRows, sourceRows, innerSize, outputRows);
        long cells = (long)outputRows * innerSize;
        if (argmax.Size < cells || gradOutput.Size < cells || (long)gradSource.Size < (long)sourceRows * innerSize ||
            !AdmitScatterRows(key, sourceRows, innerSize, outputRows))
            return false;
        return TryRunDeclaredProgram(key, s_buildScatterMaxBackwardRows, s_scatterMaxBackwardRowsNames,
            new DeclaredBindings(argmax, gradOutput, gradSource), residentWhenSmall: true);
    }

    /// <summary>ScatterMeanBackwardRows through its declared program (#1441). False when declined.</summary>
    internal bool TryDeclaredScatterMeanBackwardRows(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts,
        IGpuBuffer gradSource, int sourceRows, int innerSize, int outputRows)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterMeanBackwardRows, sourceRows, innerSize, outputRows);
        if (counts.Size < outputRows || indices.Size < sourceRows || (long)gradOutput.Size < (long)outputRows * innerSize ||
            (long)gradSource.Size < (long)sourceRows * innerSize ||
            !AdmitScatterRows(key, sourceRows, innerSize, outputRows))
            return false;
        return TryRunDeclaredProgram(key, _buildScatterMeanBackwardRows ??= BuildScatterMeanBackwardRows,
            s_scatterMeanBackwardRowsNames, new DeclaredBindings(counts, indices, gradOutput, gradSource), residentWhenSmall: true);
    }

    /// <summary>ScatterSoftmaxBackwardRows through its declared program (#1441). False when declined.</summary>
    internal bool TryDeclaredScatterSoftmaxBackwardRows(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices,
        IGpuBuffer gradSource, int sourceRows, int innerSize, int numGroups)
    {
        var key = ScatterRowsKey(DeclaredFamily.ScatterSoftmaxBackwardRows, sourceRows, innerSize, numGroups);
        long cells = (long)sourceRows * innerSize;
        if (indices.Size < sourceRows || gradOutput.Size < cells || output.Size < cells || gradSource.Size < cells ||
            !AdmitScatterRows(key, sourceRows, innerSize, numGroups))
            return false;
        return TryRunDeclaredProgram(key, _buildScatterSoftmaxBackwardRows ??= BuildScatterSoftmaxBackwardRows,
            s_scatterSoftmaxBackwardRowsNames, new DeclaredBindings(indices, output, gradOutput, gradSource), residentWhenSmall: true);
    }

    /// <summary>ScatterAddRows through the deterministic scatter-add's program (#1441): the same sums in the same order.</summary>
    internal bool TryDeclaredScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows)
    {
        if (!DeclaredScatterRowsEnabled) return false;
        return TryDeclaredScatterAddDeterministic(source, indices, output, sourceRows, outputRows, innerSize);
    }

    /// <summary>
    /// ScatterAddBackwardRows through the embedding's declared gather (#1441): <c>gradSource[i, j] =
    /// gradOutput[indices[i], j]</c>, zero for a row outside the groups -- the embedding's contract exactly.
    /// </summary>
    internal bool TryDeclaredScatterAddBackwardRows(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        if (!DeclaredScatterRowsEnabled || sourceRows <= 0 || innerSize <= 0 || outputRows <= 0 ||
            indices.Size < sourceRows || (long)gradOutput.Size < (long)outputRows * innerSize ||
            (long)gradSource.Size < (long)sourceRows * innerSize)
            return false;
        return TryRunDeclaredProgram(EmbeddingKey(sourceRows, outputRows, innerSize), s_buildEmbedding,
            s_embeddingNames, new DeclaredBindings(indices, gradOutput, gradSource), residentWhenSmall: true);
    }

    /// <summary>
    /// ScatterMean (the activation layout, int32 counts) through its declared program (#1441), deterministic in every
    /// mode. False when declined; the caller then zeroes and runs the hand-written kernels.
    /// </summary>
    internal bool TryDeclaredScatterMeanActivation(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        IGpuBuffer counts, int sourceSize, int outputSize, int featureSize)
    {
        if (featureSize <= 0 || outputSize <= 0 || sourceSize <= 0 || sourceSize % featureSize != 0) return false;
        int sourceRows = sourceSize / featureSize;
        var key = ScatterRowsKey(DeclaredFamily.ScatterMeanActivation, sourceRows, outputSize, featureSize);
        if (indices.Size < sourceRows || source.Size < sourceSize || (long)output.Size < (long)outputSize * featureSize ||
            counts.Size < outputSize || !AdmitScatterRows(key, sourceRows, featureSize, outputSize))
            return false;
        return TryRunDeclaredProgram(key, _buildScatterMeanActivation ??= BuildScatterMeanActivation,
            s_scatterMeanActivationNames, new DeclaredBindings(source, indices, counts, output), residentWhenSmall: true);
    }
}