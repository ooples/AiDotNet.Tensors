// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Serialization.Pickle;

/// <summary>
/// One tensor recovered from a PyTorch <c>.pt</c> / <c>.pth</c>
/// file's pickle stream. Holds the metadata + payload bytes; callers
/// turn it into a <see cref="LinearAlgebra.Tensor{T}"/> via
/// <see cref="PtReader.ToTensor{T}"/>.
/// </summary>
public sealed class PtTensorRef
{
    /// <summary>Element type tag (PyTorch dtype string: <c>FloatStorage</c>, <c>HalfStorage</c>, …).</summary>
    public string DtypeStorage { get; }

    /// <summary>Tensor shape (count of elements per dim).</summary>
    public long[] Shape { get; }

    /// <summary>Storage offset in elements (almost always 0 for state-dict tensors).</summary>
    public long StorageOffset { get; }

    /// <summary>Per-dim strides in elements.</summary>
    public long[] Strides { get; }

    /// <summary>The raw byte payload from the storage stream.</summary>
    public byte[] Bytes { get; }

    /// <summary>True if the layout is contiguous in row-major order — i.e. zero strides logic, just a flat copy.</summary>
    public bool IsContiguous
    {
        get
        {
            long expected = 1;
            for (int i = Shape.Length - 1; i >= 0; i--)
            {
                if (Shape[i] == 0) return true;
                if (Strides[i] != expected) return false;
                expected *= Shape[i];
            }
            return true;
        }
    }

    /// <summary>Constructs a reference.</summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="shape"/> and
    /// <paramref name="strides"/> have different ranks (would break
    /// <see cref="IsContiguous"/>'s rank-walk and the strided
    /// materialisation in <c>PtReader.MaterialiseStrided</c>), or when
    /// <paramref name="storageOffset"/> is negative.
    /// </exception>
    public PtTensorRef(string dtypeStorage, long[] shape, long storageOffset, long[] strides, byte[] bytes)
    {
        if (dtypeStorage is null) throw new ArgumentNullException(nameof(dtypeStorage));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (strides is null) throw new ArgumentNullException(nameof(strides));
        if (bytes is null) throw new ArgumentNullException(nameof(bytes));
        if (shape.Length != strides.Length)
            throw new ArgumentException(
                $"Shape rank ({shape.Length}) must equal strides rank ({strides.Length}).",
                nameof(strides));
        if (storageOffset < 0)
            throw new ArgumentException(
                $"StorageOffset ({storageOffset}) is negative.", nameof(storageOffset));

        DtypeStorage = dtypeStorage;
        // Defensive copies of the mutable arrays — without them, a
        // caller who retains the original arrays could mutate the
        // ref's observed shape/strides/payload after construction
        // and break the IsContiguous invariant the rank walk depends on.
        Shape = (long[])shape.Clone();
        Strides = (long[])strides.Clone();
        Bytes = (byte[])bytes.Clone();
        StorageOffset = storageOffset;
    }
}
