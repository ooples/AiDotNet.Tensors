// Copyright (c) AiDotNet. All rights reserved.
// The index family through declared programs (#890 phase 4e, #1441): the deterministic index write / copy / fill (the
// last index wins) and index add, the scatter reduction's sum, maximum and minimum, and the remaining gathers (int32
// take-along-dim, the linear take, float-id index_select and ScatterAddBackward). The writes and reductions share the
// scatter rows' switch, cap and size limit; the gathers, like the existing ones, keep the hand-written kernel below a
// few thousand elements, where the declared call's bookkeeping costs more than it saves.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private static readonly string[] s_indexWriteCopyNames = { "seed", "indices", "source", "out" };
    private static readonly string[] s_indexWriteFillNames = { "seed", "indices", "out" };
    private static readonly string[] s_indexAddNames = { "indices", "source", "seed", "out" };
    private static readonly string[] s_scatterReduceNames = { "index", "source", "seed", "out" };
    private static readonly string[] s_takeAlongDimIntNames = { "indices", "input", "out" };

    private Func<DeclaredKey, CodegenProgram>? _buildIndexWrite;
    private Func<DeclaredKey, CodegenProgram>? _buildScatterReduce;

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildIndexAdd =
        key => CodegenBackendPrograms.IndexAdd(key.D0, key.D1, key.D2, key.D3);

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildTakeAlongDimInt =
        key => CodegenBackendPrograms.TakeAlongDimInt(key.D0, key.D1, key.D2, key.D3);

    private static readonly Func<DeclaredKey, CodegenProgram> s_buildEmbeddingFloatIds =
        key => CodegenBackendPrograms.EmbeddingFloatIds(key.D0, key.D1, key.D2);

    // D0..D3 = outer, index length, inner, destination axis; D4 = 1 for a copy; the fill value is the key's epsilon.
    private CodegenProgram BuildIndexWrite(DeclaredKey key) =>
        CodegenBackendPrograms.IndexWrite(key.D0, key.D1, key.D2, key.D3, copy: key.D4 != 0, key.Epsilon,
            NotEqualScalar(), DerivedFunction("where_select"));

    // D0..D3 = outer, source dim, destination dim, inner; D4 = the mode.
    private CodegenProgram BuildScatterReduce(DeclaredKey key) =>
        CodegenBackendPrograms.ScatterReduce(key.D0, key.D1, key.D2, key.D3, key.D4,
            key.D4 < 2 ? null : DerivedFunction(key.D4 == 2 ? "max_vectors" : "min_vectors"));

    private const long DeclaredGatherFloor = 16384;

    /// <summary>
    /// IndexWrite (mode 0 copies, otherwise fills), Parity210IndexCopy and Parity210IndexFill through the declared index
    /// write (#1441). In place: the output is its own seed. A non-finite fill value keeps the hand-written kernel (a
    /// body constant is finite). False -- nothing launched -- when declined.
    /// </summary>
    internal bool TryDeclaredIndexWrite(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer? source, bool copy,
        float fillValue, int outer, int indexLength, int inner, int destinationAxis)
    {
        if (!DeclaredScatterRowsEnabled || outer <= 0 || indexLength <= 0 || inner <= 0 || destinationAxis <= 0 ||
            (copy && source is null) || (!copy && (float.IsNaN(fillValue) || float.IsInfinity(fillValue))) ||
            (long)indexLength * indexLength > DeclaredScatterRowsWorkLimit ||
            (long)destinationAxis * indexLength > DeclaredScatterRowsWorkLimit || indices.Size < indexLength ||
            (long)output.Size < (long)outer * destinationAxis * inner ||
            (copy && (long)source!.Size < (long)outer * indexLength * inner))
            return false;
        var key = new DeclaredKey(DeclaredFamily.IndexWrite, outer, indexLength, inner, destinationAxis,
            copy ? 0f : fillValue, 0f, d4: copy ? 1 : 0);
        if (!ClaimDeclaredSlot(key, ref _declaredScatterRowsPrograms, DeclaredScatterRowsProgramCap)) return false;
        return copy
            ? TryRunDeclaredProgram(key, _buildIndexWrite ??= BuildIndexWrite, s_indexWriteCopyNames,
                new DeclaredBindings(output, indices, source!, output), residentWhenSmall: true)
            : TryRunDeclaredProgram(key, _buildIndexWrite ??= BuildIndexWrite, s_indexWriteFillNames,
                new DeclaredBindings(output, indices, output), residentWhenSmall: true);
    }

    /// <summary>Parity210IndexAdd through the declared index add (#1441), with no atomic. False when declined.</summary>
    internal bool TryDeclaredIndexAdd(IGpuBuffer output, IGpuBuffer indices, IGpuBuffer source, int outer,
        int destinationAxis, int inner, int indexLength)
    {
        // 0.94-0.99x small, 0.02-0.10x against the atomic kernel at scale: forced only.
        if (!ScatterRowsForced) return false;
        if (!DeclaredScatterRowsEnabled || outer <= 0 || destinationAxis <= 0 || inner <= 0 || indexLength <= 0 ||
            (long)destinationAxis * indexLength > DeclaredScatterRowsWorkLimit || indices.Size < indexLength ||
            (long)output.Size < (long)outer * destinationAxis * inner ||
            (long)source.Size < (long)outer * indexLength * inner)
            return false;
        var key = new DeclaredKey(DeclaredFamily.IndexAdd, outer, destinationAxis, inner, indexLength, 0f, 0f);
        if (!ClaimDeclaredSlot(key, ref _declaredScatterRowsPrograms, DeclaredScatterRowsProgramCap)) return false;
        return TryRunDeclaredProgram(key, s_buildIndexAdd, s_indexAddNames,
            new DeclaredBindings(indices, source, output, output), residentWhenSmall: true);
    }

    /// <summary>
    /// ScatterReduce (every mode) through the declared reduction (#1441), deterministic where the hand-written
    /// kernel's compare-and-swap is not.
    /// </summary>
    internal bool TryDeclaredScatterReduce(IGpuBuffer output, IGpuBuffer source, IGpuBuffer index, int outer,
        int sourceDim, int destinationDim, int inner, int mode)
    {
        // The sum where the compare-and-swap contends -- many sources a destination -- measured 3.7x at 16384 -> 32;
        // every other mode and shape lost (0.13-0.95x).
        if (!ScatterRowsForced && !(mode == 0 && sourceDim >= 128L * destinationDim)) return false;
        if (!DeclaredScatterRowsEnabled || mode is < 0 or > 3 || outer <= 0 || sourceDim <= 0 ||
            destinationDim <= 0 || inner <= 0 || (long)sourceDim * destinationDim > DeclaredScatterRowsWorkLimit ||
            (long)index.Size < (long)outer * sourceDim * inner || (long)source.Size < (long)outer * sourceDim * inner ||
            (long)output.Size < (long)outer * destinationDim * inner)
            return false;
        var key = new DeclaredKey(DeclaredFamily.ScatterReduce, outer, sourceDim, destinationDim, inner, 0f, 0f, d4: mode);
        if (!ClaimDeclaredSlot(key, ref _declaredScatterRowsPrograms, DeclaredScatterRowsProgramCap)) return false;
        return TryRunDeclaredProgram(key, _buildScatterReduce ??= BuildScatterReduce, s_scatterReduceNames,
            new DeclaredBindings(index, source, output, output), residentWhenSmall: true);
    }

    private static readonly string[] s_maskedScatterNames = { "seed", "mask", "prefix", "source", "out" };

    private Func<DeclaredKey, CodegenProgram>? _buildMaskedScatter;

    private CodegenProgram BuildMaskedScatter(DeclaredKey key) =>
        CodegenBackendPrograms.MaskedScatter(key.D0, key.D1, NotEqualScalar(), DerivedFunction("where_select"));

    /// <summary>
    /// Parity210MaskedScatter through its declared program (#1441), in place. The mask is a byte buffer (its size is in
    /// bytes, one per element). False -- nothing launched -- when declined.
    /// </summary>
    internal bool TryDeclaredMaskedScatter(IGpuBuffer output, IGpuBuffer mask, IGpuBuffer prefix, IGpuBuffer source,
        int total)
    {
        // 0.42x: forced only.
        if (!ScatterRowsForced) return false;
        int sourceLength = source.Size;
        if (!DeclaredScatterRowsEnabled || total <= 0 || sourceLength <= 0 || (long)total < DeclaredGatherFloor ||
            mask.Size < total || prefix.Size < total || output.Size < total)
            return false;
        var key = new DeclaredKey(DeclaredFamily.MaskedScatter, total, sourceLength, 0, 0, 0f, 0f);
        if (!ClaimDeclaredSlot(key, ref _declaredScatterRowsPrograms, DeclaredScatterRowsProgramCap)) return false;
        return TryRunDeclaredProgram(key, _buildMaskedScatter ??= BuildMaskedScatter, s_maskedScatterNames,
            new DeclaredBindings(output, mask, prefix, source, output), residentWhenSmall: true);
    }

    /// <summary>Parity210TakeAlongDim (int32 indices) through its declared gather (#1441). False when declined.</summary>
    internal bool TryDeclaredTakeAlongDimInt(IGpuBuffer input, IGpuBuffer indices, IGpuBuffer output, int outer,
        int axisOut, int inner, int axisIn)
    {
        long count = (long)outer * axisOut * inner;
        if (outer <= 0 || axisOut <= 0 || inner <= 0 || axisIn <= 0 || count < DeclaredGatherFloor ||
            indices.Size < count || (long)input.Size < (long)outer * axisIn * inner || output.Size < count)
            return false;
        return TryRunDeclaredProgram(new DeclaredKey(DeclaredFamily.TakeAlongDimInt, outer, axisOut, inner, axisIn, 0f, 0f),
            s_buildTakeAlongDimInt, s_takeAlongDimIntNames, new DeclaredBindings(indices, input, output), residentWhenSmall: true);
    }

    /// <summary>
    /// A row gather by int32 ids through the embedding's declared program (Parity210TakeLinear with width 1,
    /// ScatterAddBackward): an id outside [0, vocabulary) reads a zero row.
    /// </summary>
    internal bool TryDeclaredRowGather(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int tokens, int width,
        int vocabulary)
    {
        if (tokens <= 0 || width <= 0 || vocabulary <= 0 || (long)tokens * width < DeclaredGatherFloor ||
            ids.Size < tokens || (long)output.Size < (long)tokens * width || (long)table.Size < (long)vocabulary * width)
            return false;
        return TryRunDeclaredProgram(EmbeddingKey(tokens, vocabulary, width), s_buildEmbedding, s_embeddingNames,
            new DeclaredBindings(ids, table, output), residentWhenSmall: true);
    }

    /// <summary>
    /// index_select's float-id row gather through its declared program (#1441), the row count from the input's size:
    /// an id outside the input reads a zero row where the hand-written kernel read outside it.
    /// </summary>
    internal bool TryDeclaredRowGatherFloatIds(IGpuBuffer ids, IGpuBuffer table, IGpuBuffer output, int tokens, int width)
    {
        if (tokens <= 0 || width <= 0 || (long)tokens * width < DeclaredGatherFloor || ids.Size < tokens ||
            (long)output.Size < (long)tokens * width || table.Size < width)
            return false;
        return TryRunDeclaredProgram(
            new DeclaredKey(DeclaredFamily.EmbeddingFloatIds, tokens, table.Size / width, width, 0, 0f, 0f),
            s_buildEmbeddingFloatIds, s_embeddingNames, new DeclaredBindings(ids, table, output), residentWhenSmall: true);
    }
}