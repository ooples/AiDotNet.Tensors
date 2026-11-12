using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Reads the tensor table section produced by <see cref="TensorTableWriter"/>.
/// Returns an array of <see cref="Tensor{T}"/> indexed by tensor ID, with
/// shape metadata restored and weight data populated. Intermediate buffers
/// are allocated with the correct shape but left uninitialized (they'll be
/// overwritten during Execute()).
/// </summary>
/// <remarks>
/// Every count, rank, dimension and byte length is checked against the bytes that remain and against an element
/// budget before anything is allocated, so a corrupt or hostile table fails with <see cref="InvalidDataException"/>
/// instead of an overflow or an allocation of whatever size the file names.
/// </remarks>
internal static class TensorTableReader
{
    // id, rank, flags, element count: the smallest an entry can be.
    private const int MinEntryBytes = 4 + 4 + 1 + 4;

    /// <summary>
    /// Reads all tensors from the stream. Returns an array indexed by
    /// tensor ID. For entries flagged <see cref="PlanFormatConstants.TensorFlagHasData"/>
    /// (weights/parameters), the raw element data is read and populated into
    /// the tensor. Entries flagged <see cref="PlanFormatConstants.TensorFlagExternal"/> are bound to
    /// <paramref name="externals"/>. All other tensors are allocated with the correct shape
    /// but left at their default (zero) data.
    /// </summary>
    internal static Tensor<T>[] Read<T>(BinaryReader reader, IReadOnlyList<Tensor<T>?>? externals = null,
        long maxElements = PlanFormatConstants.DefaultMaxTableElements)
    {
        var stream = reader.BaseStream;
        int count = reader.ReadInt32();
        if (count < 0 || (long)count * MinEntryBytes > Remaining(stream))
            throw new InvalidDataException($"Tensor table count {count} is corrupt.");
        var tensors = new Tensor<T>[count];
        long budget = maxElements;
        int elementSize = Marshal.SizeOf<T>();

        for (int i = 0; i < count; i++)
        {
            int id = reader.ReadInt32();
            if (id < 0 || id >= count)
                throw new InvalidDataException(
                    $"Tensor ID {id} is out of range [0, {count}). File may be corrupt.");
            if (tensors[id] is not null)
                throw new InvalidDataException($"Tensor ID {id} appears twice in the tensor table. File may be corrupt.");

            // Shape
            int rank = reader.ReadInt32();
            if (rank < 0 || (long)rank * sizeof(int) > Remaining(stream))
                throw new InvalidDataException($"Tensor {id} has a corrupt rank {rank}.");
            var shape = new int[rank];
            long elements = 1;
            for (int d = 0; d < rank; d++)
            {
                shape[d] = reader.ReadInt32();
                if (shape[d] < 0) throw new InvalidDataException($"Tensor {id} has a negative dimension.");
                elements *= shape[d];
                if (elements > int.MaxValue)
                    throw new InvalidDataException($"Tensor {id} shape [{string.Join(",", shape)}] exceeds the largest tensor.");
            }

            // Flags
            byte flags = reader.ReadByte();
            bool hasData = (flags & PlanFormatConstants.TensorFlagHasData) != 0;
            bool external = (flags & PlanFormatConstants.TensorFlagExternal) != 0;

            // Element count
            int elementCount = reader.ReadInt32();
            if (elementCount != elements)
                throw new InvalidDataException($"Tensor {id} declares {elementCount} elements for shape [{string.Join(",", shape)}].");

            if (external)
            {
                if (hasData) throw new InvalidDataException($"Tensor {id} is both external and stored.");
                int index = reader.ReadInt32();
                if (externals is null)
                    throw new InvalidDataException($"Tensor {id} references an external tensor, and none were supplied.");
                if ((uint)index >= (uint)externals.Count || externals[index] is not { } bound)
                    throw new InvalidDataException($"Tensor {id} references external tensor {index}, which was not supplied.");
                if (!bound._shape.SequenceEqual(shape))
                    throw new InvalidDataException(
                        $"Tensor {id} references external tensor {index} as [{string.Join(",", shape)}]; it is [{string.Join(",", bound._shape)}].");
                tensors[id] = bound;
                continue;
            }

            if (hasData && elements * elementSize > Remaining(stream))
                throw new InvalidDataException($"Tensor {id} data is truncated. File may be corrupt.");
            budget -= elements;
            if (budget < 0)
                throw new InvalidDataException($"The tensor table asks for more than {maxElements} elements. File may be corrupt.");

            // Allocate tensor with the saved shape.
            var tensor = new Tensor<T>(shape);

            // Read data if present (weights/parameters).
            if (hasData)
            {
                ReadRawElements(reader, tensor, elementCount, elementSize);
            }

            tensors[id] = tensor;
        }

        for (int i = 0; i < tensors.Length; i++)
            if (tensors[i] is null) throw new InvalidDataException($"Tensor ID {i} is missing from the tensor table.");
        return tensors;
    }

    private static long Remaining(Stream stream) => stream.Length - stream.Position;

    /// <summary>
    /// Reads raw element bytes from the stream into an existing tensor.
    /// </summary>
    private static void ReadRawElements<T>(BinaryReader reader, Tensor<T> tensor, int elementCount, int elementSize)
    {
        int byteCount = checked(elementCount * elementSize);
        var bytes = reader.ReadBytes(byteCount);
        if (bytes.Length < byteCount)
            throw new InvalidDataException(
                $"Truncated tensor data: expected {byteCount} bytes, got {bytes.Length}. File may be corrupt.");

        using var dataLease = tensor.LeaseArray();
        var data = dataLease.Array;
        Buffer.BlockCopy(bytes, 0, data, 0, byteCount);
    }
}
