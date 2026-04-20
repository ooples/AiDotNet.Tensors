using System.IO;
using System.Text;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Serializes and deserializes the <c>object[]? SavedState</c> attached to each
/// <see cref="CompiledStep{T}"/>. Each entry is type-tagged so the reader can
/// reconstruct the original CLR type without reflection.
///
/// <para>Supported types: <c>null</c>, <c>int</c>, <c>int[]</c>, <c>double</c>,
/// <c>float</c>, <c>bool</c>, <c>string</c>, <c>byte[]</c>, and
/// <c>Tensor&lt;T&gt;</c> (serialized as a tensor-table ID reference).</para>
/// </summary>
internal static class SavedStateSerializer
{
    /// <summary>
    /// Writes a saved-state array. Writes <c>-1</c> for null, or the entry
    /// count followed by each type-tagged entry.
    /// </summary>
    internal static void Write<T>(BinaryWriter writer, object[]? state, TensorIdMap<T> tensorMap)
    {
        if (state is null)
        {
            writer.Write(-1);
            return;
        }

        writer.Write(state.Length);
        for (int i = 0; i < state.Length; i++)
        {
            WriteEntry(writer, state[i], tensorMap);
        }
    }

    /// <summary>
    /// Reads a saved-state array previously written by <see cref="Write{T}"/>.
    /// Returns null if the stored count was <c>-1</c>.
    /// </summary>
    internal static object[]? Read<T>(BinaryReader reader, Tensor<T>[] tensorTable)
    {
        int count = reader.ReadInt32();
        if (count < 0) return null;

        var result = new object[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadEntry<T>(reader, tensorTable)!;
        }
        return result;
    }

    private static void WriteEntry<T>(BinaryWriter writer, object? value, TensorIdMap<T> tensorMap)
    {
        switch (value)
        {
            case null:
                writer.Write(PlanFormatConstants.TagNull);
                break;

            case int intVal:
                writer.Write(PlanFormatConstants.TagInt32);
                writer.Write(intVal);
                break;

            case int[] intArr:
                writer.Write(PlanFormatConstants.TagInt32Array);
                writer.Write(intArr.Length);
                for (int j = 0; j < intArr.Length; j++)
                    writer.Write(intArr[j]);
                break;

            case double doubleVal:
                writer.Write(PlanFormatConstants.TagDouble);
                writer.Write(doubleVal);
                break;

            case float floatVal:
                writer.Write(PlanFormatConstants.TagFloat);
                writer.Write(floatVal);
                break;

            case bool boolVal:
                writer.Write(PlanFormatConstants.TagBool);
                writer.Write(boolVal);
                break;

            case string strVal:
                writer.Write(PlanFormatConstants.TagString);
                var bytes = Encoding.UTF8.GetBytes(strVal);
                writer.Write(bytes.Length);
                writer.Write(bytes);
                break;

            case byte[] byteArr:
                writer.Write(PlanFormatConstants.TagByteArray);
                writer.Write(byteArr.Length);
                writer.Write(byteArr);
                break;

            case Tensor<T> tensor:
                writer.Write(PlanFormatConstants.TagTensorRef);
                writer.Write(tensorMap.GetId(tensor));
                break;

            case Enum enumVal:
                // Enum-valued savedState entries (e.g. FusedActivationType
                // captured by CpuFusionPass). Serialize as
                // (assembly-qualified type name, int value) so the reader
                // can reconstruct the typed enum without reflection into a
                // wrong-sized underlying type. We write the int value
                // (widened from the enum's actual byte/short/long backing
                // on both ends) so the payload stays type-code-independent.
                writer.Write(PlanFormatConstants.TagEnum);
                var enumTypeName = enumVal.GetType().AssemblyQualifiedName
                    ?? throw new NotSupportedException(
                        $"Enum type {enumVal.GetType().FullName} has no AssemblyQualifiedName and cannot be serialized.");
                var enumTypeBytes = Encoding.UTF8.GetBytes(enumTypeName);
                writer.Write(enumTypeBytes.Length);
                writer.Write(enumTypeBytes);
                writer.Write(Convert.ToInt64(enumVal)); // int64 to cover all underlying types
                break;

            default:
                // Unsupported type in SavedState — fail at save time so
                // load never encounters a mystery tag byte.
                throw new NotSupportedException(
                    $"SavedState entry of type {value.GetType().FullName} cannot be serialized. " +
                    "Supported types: null, int, int[], double, float, bool, string, byte[], Tensor<T>, Enum.");
        }
    }

    private static object? ReadEntry<T>(BinaryReader reader, Tensor<T>[] tensorTable)
    {
        byte tag = reader.ReadByte();
        return tag switch
        {
            PlanFormatConstants.TagNull       => null,
            PlanFormatConstants.TagInt32      => reader.ReadInt32(),
            PlanFormatConstants.TagInt32Array => ReadInt32Array(reader),
            PlanFormatConstants.TagDouble     => reader.ReadDouble(),
            PlanFormatConstants.TagFloat      => (object)reader.ReadSingle(),
            PlanFormatConstants.TagBool       => reader.ReadBoolean(),
            PlanFormatConstants.TagString     => ReadString(reader),
            PlanFormatConstants.TagByteArray  => ReadByteArray(reader),
            PlanFormatConstants.TagTensorRef  => ReadTensorRef(reader, tensorTable),
            PlanFormatConstants.TagEnum       => ReadEnum(reader),
            _ => throw new InvalidDataException(
                $"Unknown SavedState type tag 0x{tag:X2} — the file may be corrupt or " +
                "from a newer format version that this binary cannot read."),
        };
    }

    private static object ReadEnum(BinaryReader reader)
    {
        int typeNameLen = reader.ReadInt32();
        if (typeNameLen < 0)
            throw new InvalidDataException(
                $"SavedState enum type-name length {typeNameLen} cannot be negative. The plan file is corrupt.");
        var nameBytes = reader.ReadBytes(typeNameLen);
        if (nameBytes.Length != typeNameLen)
            throw new InvalidDataException(
                $"SavedState enum type-name payload was truncated: expected {typeNameLen} bytes, got {nameBytes.Length}.");
        string typeName = Encoding.UTF8.GetString(nameBytes);
        long rawValue = reader.ReadInt64();

        var enumType = Type.GetType(typeName, throwOnError: false);
        if (enumType is null || !enumType.IsEnum)
            throw new InvalidDataException(
                $"SavedState enum type '{typeName}' cannot be resolved or is not an enum. " +
                "The plan may have been saved by a different assembly version.");

        // Narrow through the underlying type so Enum.ToObject constructs the
        // right boxed value even for byte/short-backed enums.
        var underlying = Enum.GetUnderlyingType(enumType);
        object narrowed = Convert.ChangeType(rawValue, underlying);
        return Enum.ToObject(enumType, narrowed);
    }

    private static Tensor<T> ReadTensorRef<T>(BinaryReader reader, Tensor<T>[] tensorTable)
    {
        int id = reader.ReadInt32();
        // Even though the header carries a checksum, an out-of-range tensor ID
        // here would produce a confusing IndexOutOfRangeException deep in the
        // replay path; surface it as InvalidDataException (same contract the
        // tag switch uses) so corruption is caught at load time by the
        // CompiledPlanLoader try/catch.
        if ((uint)id >= (uint)tensorTable.Length)
        {
            throw new InvalidDataException(
                $"SavedState tensor reference ID {id} is out of range " +
                $"[0, {tensorTable.Length}). The plan file is corrupt.");
        }
        return tensorTable[id];
    }

    private static int[] ReadInt32Array(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0)
            throw new InvalidDataException($"SavedState int[] length {len} cannot be negative. The plan file is corrupt.");
        var arr = new int[len];
        for (int i = 0; i < len; i++)
            arr[i] = reader.ReadInt32();
        return arr;
    }

    private static string ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0)
            throw new InvalidDataException($"SavedState string length {len} cannot be negative. The plan file is corrupt.");
        var bytes = reader.ReadBytes(len);
        // BinaryReader.ReadBytes returns fewer bytes on truncated streams
        // (silent short read) instead of throwing EndOfStreamException —
        // treat the mismatch as corruption so the loader triggers its
        // cache-miss fallback.
        if (bytes.Length != len)
            throw new InvalidDataException(
                $"SavedState string payload was truncated: expected {len} bytes, got {bytes.Length}.");
        return Encoding.UTF8.GetString(bytes);
    }

    private static byte[] ReadByteArray(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0)
            throw new InvalidDataException($"SavedState byte[] length {len} cannot be negative. The plan file is corrupt.");
        var bytes = reader.ReadBytes(len);
        if (bytes.Length != len)
            throw new InvalidDataException(
                $"SavedState byte[] payload was truncated: expected {len} bytes, got {bytes.Length}.");
        return bytes;
    }
}
