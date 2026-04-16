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

            default:
                // Unsupported type in SavedState — fail at save time so
                // load never encounters a mystery tag byte.
                throw new NotSupportedException(
                    $"SavedState entry of type {value.GetType().FullName} cannot be serialized. " +
                    "Supported types: null, int, int[], double, float, bool, string, byte[], Tensor<T>.");
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
            PlanFormatConstants.TagTensorRef  => tensorTable[reader.ReadInt32()],
            _ => throw new InvalidDataException(
                $"Unknown SavedState type tag 0x{tag:X2} — the file may be corrupt or " +
                "from a newer format version that this binary cannot read."),
        };
    }

    private static int[] ReadInt32Array(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var arr = new int[len];
        for (int i = 0; i < len; i++)
            arr[i] = reader.ReadInt32();
        return arr;
    }

    private static string ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        var bytes = reader.ReadBytes(len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static byte[] ReadByteArray(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        return reader.ReadBytes(len);
    }
}
