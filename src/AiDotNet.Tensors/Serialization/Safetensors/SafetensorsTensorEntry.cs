// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// One entry from a safetensors file's JSON header — the dtype, the
/// shape, and the byte range inside the data block where this tensor
/// lives. Returned by <see cref="SafetensorsReader"/> alongside the
/// open data stream so callers can either materialise the tensor
/// eagerly or hold the entry for lazy zero-copy slicing.
/// </summary>
public sealed class SafetensorsTensorEntry
{
    /// <summary>Tensor name from the header (the JSON key).</summary>
    public string Name { get; }

    /// <summary>Element type tag.</summary>
    public SafetensorsDtype Dtype { get; }

    /// <summary>Shape — a copy of the parsed array, safe to mutate.</summary>
    public long[] Shape { get; }

    /// <summary>
    /// Start offset of this tensor's payload in the data block (0-based;
    /// add the data-block start offset to convert to absolute file
    /// offset).
    /// </summary>
    public long DataOffsetStart { get; }

    /// <summary>
    /// End offset (exclusive) of this tensor's payload in the data
    /// block. <c>DataOffsetEnd - DataOffsetStart</c> is the byte size
    /// of the tensor's data on disk.
    /// </summary>
    public long DataOffsetEnd { get; }

    /// <summary>Byte length of this tensor's payload.</summary>
    public long ByteLength => DataOffsetEnd - DataOffsetStart;

    /// <summary>
    /// Total element count — the product of <see cref="Shape"/>. Note
    /// for sub-byte dtypes this is the *logical* element count, not
    /// the packed-byte count; multiply by <c>1 / packingFactor</c> to
    /// get the byte count instead. Computed once at construction in a
    /// <c>checked</c> context — a malicious header that would have
    /// wrapped the product to a small positive value is rejected
    /// before it can bypass the reader's byte-length guard.
    /// </summary>
    public long ElementCount { get; }

    /// <summary>Constructs an entry. Cloning the shape so callers can mutate freely.</summary>
    /// <exception cref="ArgumentException">
    /// Thrown when any shape dimension is negative,
    /// <paramref name="dataOffsetStart"/> is negative,
    /// <paramref name="dataOffsetEnd"/> is below
    /// <paramref name="dataOffsetStart"/>, or when the element-count
    /// product would overflow <see cref="long"/>.
    /// </exception>
    public SafetensorsTensorEntry(string name, SafetensorsDtype dtype, long[] shape, long dataOffsetStart, long dataOffsetEnd)
    {
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (dataOffsetStart < 0)
            throw new ArgumentException(
                $"DataOffsetStart ({dataOffsetStart}) is negative.", nameof(dataOffsetStart));
        if (dataOffsetEnd < dataOffsetStart)
            throw new ArgumentException(
                $"DataOffsetEnd ({dataOffsetEnd}) must be >= DataOffsetStart ({dataOffsetStart}).",
                nameof(dataOffsetEnd));

        long elemCount = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException(
                    $"Shape dimension {i} = {shape[i]} is negative for tensor '{name}'.",
                    nameof(shape));
            try { elemCount = checked(elemCount * shape[i]); }
            catch (OverflowException ex)
            {
                throw new ArgumentException(
                    $"Shape product overflows long for tensor '{name}'.", nameof(shape), ex);
            }
        }

        Name = name;
        Dtype = dtype;
        Shape = (long[])shape.Clone();
        DataOffsetStart = dataOffsetStart;
        DataOffsetEnd = dataOffsetEnd;
        ElementCount = elemCount;
    }
}
