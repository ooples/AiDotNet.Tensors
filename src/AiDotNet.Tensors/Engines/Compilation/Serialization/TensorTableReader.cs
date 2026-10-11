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
internal static class TensorTableReader
{
    /// <summary>
    /// Reads all tensors from the stream. Returns an array indexed by
    /// tensor ID. For entries flagged <see cref="PlanFormatConstants.TensorFlagHasData"/>
    /// (weights/parameters), the raw element data is read and populated into
    /// the tensor. All other tensors are allocated with the correct shape
    /// but left at their default (zero) data.
    /// </summary>
    internal static Tensor<T>[] Read<T>(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var tensors = new Tensor<T>[count];

        for (int i = 0; i < count; i++)
        {
            int id = reader.ReadInt32();
            if (id < 0 || id >= count)
                throw new InvalidDataException(
                    $"Tensor ID {id} is out of range [0, {count}). File may be corrupt.");

            // Shape
            int rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int d = 0; d < rank; d++)
                shape[d] = reader.ReadInt32();

            // Flags
            byte flags = reader.ReadByte();
            bool hasData = (flags & PlanFormatConstants.TensorFlagHasData) != 0;

            // Element count
            int elementCount = reader.ReadInt32();

            // Allocate tensor with the saved shape.
            var tensor = new Tensor<T>(shape);

            // Read data if present (weights/parameters).
            if (hasData)
            {
                ReadRawElements(reader, tensor, elementCount);
            }

            tensors[id] = tensor;
        }

        return tensors;
    }

    /// <summary>
    /// Reads raw element bytes from the stream into an existing tensor.
    /// </summary>
    private static void ReadRawElements<T>(BinaryReader reader, Tensor<T> tensor, int elementCount)
    {
        int byteCount = elementCount * Marshal.SizeOf<T>();
        var bytes = reader.ReadBytes(byteCount);
        if (bytes.Length < byteCount)
            throw new InvalidDataException(
                $"Truncated tensor data: expected {byteCount} bytes, got {bytes.Length}. File may be corrupt.");

        var data = tensor.GetDataArray();
        Buffer.BlockCopy(bytes, 0, data, 0, byteCount);
    }
}
