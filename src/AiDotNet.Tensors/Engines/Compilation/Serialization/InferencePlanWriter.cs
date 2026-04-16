using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Writes a <see cref="CompiledInferencePlan{T}"/> to a binary stream.
/// Format: header → tensor table → op sequence → footer (XXHash64).
/// See <see cref="PlanFormatConstants"/> for the wire-format specification.
/// </summary>
internal static class InferencePlanWriter
{
    /// <summary>
    /// Serializes the plan to the given stream. The stream is NOT closed
    /// — the caller owns its lifetime.
    /// </summary>
    internal static void Write<T>(
        Stream stream,
        CompiledStep<T>[] steps,
        Tensor<T> finalOutput,
        int[] compiledInputShape,
        Tensor<T>? compiledInputTensor)
    {
        // We write to a MemoryStream first so we can compute the checksum
        // over the complete body, then flush to the real stream.
        using var body = new MemoryStream();
        using var writer = new BinaryWriter(body, Encoding.UTF8, leaveOpen: true);

        // ── Header ──────────────────────────────────────────────────────
        writer.Write(PlanFormatConstants.Magic);
        writer.Write(PlanFormatConstants.CurrentFormatVersion);
        writer.Write(PlanFormatConstants.PlanTypeInference);
        writer.Write(PlanFormatConstants.GetElementTypeCode<T>());
        writer.Write(steps.Length);

        // Input shape
        writer.Write(compiledInputShape.Length);
        for (int i = 0; i < compiledInputShape.Length; i++)
            writer.Write(compiledInputShape[i]);

        // Tensor-codec version
        writer.Write(PlanFormatConstants.TensorCodecVersion);

        // Hardware fingerprint (length-prefixed UTF-8)
        var fpBytes = Encoding.UTF8.GetBytes(HardwareFingerprint.Current);
        writer.Write(fpBytes.Length);
        writer.Write(fpBytes);

        // ── Tensor table ────────────────────────────────────────────────
        var tensorMap = TensorTableWriter.BuildMap(steps, compiledInputTensor);
        TensorTableWriter.Write(writer, tensorMap, compiledInputTensor, parameterTensors: null);

        // ── Op sequence ─────────────────────────────────────────────────
        writer.Write(steps.Length);
        for (int i = 0; i < steps.Length; i++)
        {
            WriteStep(writer, steps[i], tensorMap);
        }

        // ── Footer (checksum) ───────────────────────────────────────────
        writer.Flush();
        var bodyBytes = body.ToArray();
        ulong checksum = XXHash64.Compute(bodyBytes, 0, bodyBytes.Length);

        // Now write body + footer to the real stream.
        stream.Write(bodyBytes, 0, bodyBytes.Length);
        using var footerWriter = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        footerWriter.Write((long)bodyBytes.Length);
        footerWriter.Write((long)checksum);
    }

    private static void WriteStep<T>(BinaryWriter writer, CompiledStep<T> step, TensorIdMap<T> tensorMap)
    {
        // OpType byte
        writer.Write((byte)step.OpType);

        // OpName (length-prefixed UTF-8 for forward compat)
        var nameBytes = Encoding.UTF8.GetBytes(step.OpName);
        writer.Write(nameBytes.Length);
        writer.Write(nameBytes);

        // Input tensor IDs
        writer.Write(step.Inputs.Length);
        for (int j = 0; j < step.Inputs.Length; j++)
            writer.Write(tensorMap.GetId(step.Inputs[j]));

        // Output tensor ID
        writer.Write(tensorMap.GetId(step.OutputBuffer));

        // SavedState
        SavedStateSerializer.Write(writer, step.SavedState, tensorMap);
    }
}
