// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// Round-trips <see cref="SparseTensor{T}"/> through the safetensors
/// container by writing the format's underlying integer index arrays
/// and value array as three named sub-tensors plus a metadata pointer
/// linking them. Mirrors how PyTorch's <c>safetensors</c> binding
/// handles <c>torch.sparse_csr_tensor</c> — there's no native sparse
/// dtype in safetensors, so the convention is "name + suffix" with a
/// metadata key keeping them associated.
///
/// <para><b>Naming convention</b> (so other safetensors readers can
/// reconstruct without our metadata): <c>{name}.values</c>,
/// <c>{name}.indices0</c>, <c>{name}.indices1</c>. The metadata entry
/// <c>{name}.sparse_format</c> carries the format tag plus row/col
/// dimensions and (for block formats) the block shape.</para>
/// </summary>
public static class SafetensorsSparseExtensions
{
    /// <summary>Writes a sparse tensor under <paramref name="name"/>
    /// using the convention above. The values dtype must match what the
    /// safetensors container supports for <typeparamref name="T"/>.</summary>
    public static void AddSparse<T>(this SafetensorsWriter writer, string name, SparseTensor<T> sparse)
        where T : struct
    {
        if (writer is null) throw new ArgumentNullException(nameof(writer));
        if (sparse is null) throw new ArgumentNullException(nameof(sparse));
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be empty.", nameof(name));

        // Values are written as a 1-D dense tensor — safetensors reader
        // gets a normal Tensor<T> back even without sparse-aware code.
        var valuesTensor = new Tensor<T>(new[] { sparse.NonZeroCount });
        sparse.DataVector.AsSpan().CopyTo(valuesTensor.AsWritableSpan());
        writer.Add($"{name}.values", valuesTensor);

        switch (sparse.Format)
        {
            case SparseStorageFormat.Coo:
                AddIntTensor(writer, $"{name}.indices0", sparse.RowIndices);
                AddIntTensor(writer, $"{name}.indices1", sparse.ColumnIndices);
                writer.Metadata[$"{name}.sparse_format"] = $"coo:{sparse.Rows}x{sparse.Columns}";
                break;
            case SparseStorageFormat.Csr:
                AddIntTensor(writer, $"{name}.indices0", sparse.RowPointers);
                AddIntTensor(writer, $"{name}.indices1", sparse.ColumnIndices);
                writer.Metadata[$"{name}.sparse_format"] = $"csr:{sparse.Rows}x{sparse.Columns}";
                break;
            case SparseStorageFormat.Csc:
                AddIntTensor(writer, $"{name}.indices0", sparse.ColumnPointers);
                AddIntTensor(writer, $"{name}.indices1", sparse.RowIndices);
                writer.Metadata[$"{name}.sparse_format"] = $"csc:{sparse.Rows}x{sparse.Columns}";
                break;
            case SparseStorageFormat.Bsr:
                AddIntTensor(writer, $"{name}.indices0", sparse.RowPointers);
                AddIntTensor(writer, $"{name}.indices1", sparse.ColumnIndices);
                writer.Metadata[$"{name}.sparse_format"] =
                    $"bsr:{sparse.Rows}x{sparse.Columns}:{sparse.BlockRowSize}x{sparse.BlockColSize}";
                break;
            case SparseStorageFormat.Bsc:
                AddIntTensor(writer, $"{name}.indices0", sparse.ColumnPointers);
                AddIntTensor(writer, $"{name}.indices1", sparse.RowIndices);
                writer.Metadata[$"{name}.sparse_format"] =
                    $"bsc:{sparse.Rows}x{sparse.Columns}:{sparse.BlockRowSize}x{sparse.BlockColSize}";
                break;
            default:
                throw new NotSupportedException($"Unknown sparse format {sparse.Format}.");
        }
    }

    /// <summary>Reads a sparse tensor previously written via
    /// <see cref="AddSparse{T}"/>. The reader must already have the
    /// safetensors file open and the metadata loaded.</summary>
    public static SparseTensor<T> ReadSparse<T>(this SafetensorsReader reader, string name)
        where T : struct
    {
        if (reader is null) throw new ArgumentNullException(nameof(reader));
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Name cannot be empty.", nameof(name));

        if (!reader.Metadata.TryGetValue($"{name}.sparse_format", out var formatTag) || formatTag is null)
            throw new InvalidOperationException(
                $"Metadata key '{name}.sparse_format' missing — was '{name}' written via AddSparse?");

        var values = reader.ReadTensor<T>($"{name}.values");
        var idx0 = reader.ReadTensor<int>($"{name}.indices0");
        var idx1 = reader.ReadTensor<int>($"{name}.indices1");

        var (kind, rows, cols, blockRow, blockCol) = ParseFormatTag(formatTag);
        var valuesArr = values.AsSpan().ToArray();
        var idx0Arr = idx0.AsSpan().ToArray();
        var idx1Arr = idx1.AsSpan().ToArray();

        return kind switch
        {
            "coo" => new SparseTensor<T>(rows, cols, idx0Arr, idx1Arr, valuesArr),
            "csr" => SparseTensor<T>.FromCsr(rows, cols, idx0Arr, idx1Arr, valuesArr),
            "csc" => SparseTensor<T>.FromCsc(rows, cols, idx0Arr, idx1Arr, valuesArr),
            "bsr" => SparseTensor<T>.FromBsr(rows, cols, blockRow, blockCol, idx0Arr, idx1Arr, valuesArr),
            "bsc" => SparseTensor<T>.FromBsc(rows, cols, blockRow, blockCol, idx0Arr, idx1Arr, valuesArr),
            _ => throw new InvalidOperationException($"Unknown sparse-format tag '{formatTag}'."),
        };
    }

    private static void AddIntTensor(SafetensorsWriter writer, string name, int[] data)
    {
        var t = new Tensor<int>(new[] { data.Length });
        data.AsSpan().CopyTo(t.AsWritableSpan());
        writer.Add(name, t);
    }

    private static (string Kind, int Rows, int Cols, int BlockRow, int BlockCol) ParseFormatTag(string tag)
    {
        // tag := "<kind>:<rows>x<cols>" or "<kind>:<rows>x<cols>:<br>x<bc>"
        var parts = tag.Split(':');
        if (parts.Length < 2 || parts.Length > 3)
            throw new FormatException($"Malformed sparse-format tag '{tag}'.");
        string kind = parts[0];
        var dims = parts[1].Split('x');
        if (dims.Length != 2)
            throw new FormatException($"Malformed dim spec in sparse-format tag '{tag}'.");
        int rows = int.Parse(dims[0]);
        int cols = int.Parse(dims[1]);
        int br = 0, bc = 0;
        if (parts.Length >= 3)
        {
            var blockDims = parts[2].Split('x');
            if (blockDims.Length != 2)
                throw new FormatException($"Malformed block-dim spec in sparse-format tag '{tag}'.");
            br = int.Parse(blockDims[0]);
            bc = int.Parse(blockDims[1]);
        }
        // Cross-validate kind vs structure: dense-pointer formats
        // (coo/csr/csc) must NOT carry a block-dim tail; block formats
        // (bsr/bsc) MUST carry positive block dims. The earlier code
        // accepted "bsr:4x4" with br=bc=0 and forwarded that into
        // FromBsr's `blockRowSize <= 0` guard — leaking the corruption
        // path through several downstream layers before throwing.
        return kind switch
        {
            "coo" or "csr" or "csc" when parts.Length == 2 => (kind, rows, cols, 0, 0),
            "bsr" or "bsc" when parts.Length == 3 && br > 0 && bc > 0 => (kind, rows, cols, br, bc),
            "coo" or "csr" or "csc" => throw new FormatException(
                $"Unexpected block dims in sparse-format tag '{tag}' for non-block kind '{kind}'."),
            "bsr" or "bsc" => throw new FormatException(
                $"Missing or non-positive block dims in sparse-format tag '{tag}' for block kind '{kind}'."),
            _ => throw new FormatException($"Unknown sparse-format tag '{tag}'."),
        };
    }
}
