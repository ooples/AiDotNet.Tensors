using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Writes the tensor table section of the serialized plan. Each tensor gets
/// its shape metadata and (for weights/parameters) its raw element data.
/// Intermediate buffers are recorded as shape-only placeholders — their data
/// is recreated on load by running the plan.
/// </summary>
internal static class TensorTableWriter
{
    /// <summary>
    /// Scans all steps to build a complete tensor ID map. Every tensor
    /// referenced by any step's Inputs or OutputBuffer (and any tensor
    /// in SavedState) is registered. Call once before serialization.
    /// </summary>
    internal static TensorIdMap<T> BuildMap<T>(
        CompiledStep<T>[] steps,
        Tensor<T>? compiledInputTensor,
        Tensor<T>[]? parameterTensors = null)
    {
        var map = new TensorIdMap<T>();

        // Register compiled input first (ID 0 by convention).
        if (compiledInputTensor is not null)
            map.GetOrAdd(compiledInputTensor);

        // Register parameter tensors (training plans).
        if (parameterTensors is not null)
        {
            for (int i = 0; i < parameterTensors.Length; i++)
                map.GetOrAdd(parameterTensors[i]);
        }

        // Walk steps: register each input and output.
        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            for (int j = 0; j < step.Inputs.Length; j++)
                map.GetOrAdd(step.Inputs[j]);
            map.GetOrAdd(step.OutputBuffer);

            // SavedState may contain tensor references.
            if (step.SavedState is not null)
            {
                for (int j = 0; j < step.SavedState.Length; j++)
                {
                    if (step.SavedState[j] is Tensor<T> tensor)
                        map.GetOrAdd(tensor);
                }
            }
        }

        return map;
    }

    /// <summary>
    /// Writes all tensors from the map to the stream. Format per tensor:
    /// <c>[id:int32] [rank:int32] [shape:int32*rank] [flags:byte] [dataLen:int32] [data:bytes]</c>.
    /// Weight/parameter tensors include their element data; intermediates
    /// are shape-only placeholders.
    /// </summary>
    internal static void Write<T>(
        BinaryWriter writer,
        TensorIdMap<T> map,
        CompiledStep<T>[] steps,
        Tensor<T>? compiledInputTensor,
        Tensor<T>[]? parameterTensors)
    {
        var tensors = map.Ordered;
        writer.Write(tensors.Count);

        // Build sets for flag assignment.
        var paramSet = new HashSet<Tensor<T>>(
            parameterTensors ?? Array.Empty<Tensor<T>>(),
            System.Collections.Generic.EqualityComparer<Tensor<T>>.Default);

        // Build the set of tensors that are step OUTPUT buffers. Any tensor
        // that appears in step Inputs but is NOT a step output AND is not the
        // compiled-input leaf is a "frozen parameter" (weight captured at
        // compile time) — its data must be serialized so the loaded plan
        // can produce correct results without the original weight objects.
        var outputBufferSet = new HashSet<Tensor<T>>(
            System.Collections.Generic.EqualityComparer<Tensor<T>>.Default);
        for (int s = 0; s < steps.Length; s++)
            outputBufferSet.Add(steps[s].OutputBuffer);

        for (int i = 0; i < tensors.Count; i++)
        {
            var tensor = tensors[i];
            int id = i;

            writer.Write(id);

            // Shape
            writer.Write(tensor._shape.Length);
            for (int d = 0; d < tensor._shape.Length; d++)
                writer.Write(tensor._shape[d]);

            // Classify this tensor.
            bool isExplicitParam = paramSet.Contains(tensor);
            bool isLeaf = ReferenceEquals(tensor, compiledInputTensor);
            bool isOutputBuffer = outputBufferSet.Contains(tensor);

            // A "frozen weight" is a tensor that:
            //   1. Is not the compiled-input leaf (the user fills that)
            //   2. Is not any step's output buffer (those are intermediate)
            //   3. Appears as a step input (it was captured at compile time)
            // OR is explicitly in the parameter set.
            bool isFrozenWeight = isExplicitParam || (!isLeaf && !isOutputBuffer);

            byte flags;
            if (isLeaf)        flags = PlanFormatConstants.TensorFlagLeafInput;
            else if (isFrozenWeight) flags = PlanFormatConstants.TensorFlagWeight;
            else               flags = PlanFormatConstants.TensorFlagIntermediate;

            bool writeData = isFrozenWeight;
            if (writeData) flags |= PlanFormatConstants.TensorFlagHasData;

            writer.Write(flags);

            int elementCount = tensor.Length;
            writer.Write(elementCount);

            if (writeData)
            {
                WriteRawElements(writer, tensor);
            }
        }
    }

    /// <summary>
    /// Writes the raw element data of a tensor as a contiguous byte block.
    /// </summary>
    private static void WriteRawElements<T>(BinaryWriter writer, Tensor<T> tensor)
    {
        // Ensure contiguous layout.
        var contiguous = tensor.IsContiguous && tensor._storageOffset == 0
            ? tensor
            : tensor.Contiguous();

        var data = contiguous.GetDataArray();
        int byteCount = data.Length * Marshal.SizeOf<T>();
        var bytes = new byte[byteCount];
        Buffer.BlockCopy(data, 0, bytes, 0, byteCount);
        writer.Write(bytes);
    }
}
