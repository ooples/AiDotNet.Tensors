using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Writes a <see cref="CompiledTrainingPlan{T}"/> to a binary stream.
/// Extends the inference format with training-specific sections: parameter
/// tensor IDs, gradient buffer shapes, backward function OpType mapping,
/// and optimizer state.
/// </summary>
internal static class TrainingPlanWriter
{
    internal static void Write<T>(Stream stream, CompiledTrainingPlan<T> plan)
    {
        var forwardSteps = plan.ForwardStepsForSerialization;
        if (forwardSteps is null || forwardSteps.Length == 0)
            throw new InvalidOperationException(
                "Cannot serialize a training plan that has no retained forward steps. " +
                "This plan may have been constructed before plan serialization support was added.");

        using var body = new MemoryStream();
        using var writer = new BinaryWriter(body, Encoding.UTF8, leaveOpen: true);

        var inputShape = plan.SerializedInputShape ?? Array.Empty<int>();
        var inputTensor = plan.SerializedInputTensor;

        // ── Header ──────────────────────────────────────────────────────
        writer.Write(PlanFormatConstants.Magic);
        writer.Write(PlanFormatConstants.CurrentFormatVersion);
        writer.Write(PlanFormatConstants.PlanTypeTraining);
        writer.Write(PlanFormatConstants.GetElementTypeCode<T>());
        writer.Write(forwardSteps.Length);

        // Input shape
        writer.Write(inputShape.Length);
        for (int i = 0; i < inputShape.Length; i++)
            writer.Write(inputShape[i]);

        // Tensor-codec version + hardware fingerprint
        writer.Write(PlanFormatConstants.TensorCodecVersion);
        var fpBytes = Encoding.UTF8.GetBytes(HardwareFingerprint.Current);
        writer.Write(fpBytes.Length);
        writer.Write(fpBytes);

        // ── Tensor table ────────────────────────────────────────────────
        var tensorMap = TensorTableWriter.BuildMap(
            forwardSteps, inputTensor, plan.Parameters);
        TensorTableWriter.Write(writer, tensorMap, forwardSteps, inputTensor, plan.Parameters);

        // ── Forward op sequence ─────────────────────────────────────────
        writer.Write(forwardSteps.Length);
        for (int i = 0; i < forwardSteps.Length; i++)
        {
            WriteStep(writer, forwardSteps[i], tensorMap);
        }

        // ── Training extension ──────────────────────────────────────────
        // Parameter tensor IDs
        writer.Write(plan.Parameters.Length);
        for (int i = 0; i < plan.Parameters.Length; i++)
            writer.Write(tensorMap.GetId(plan.Parameters[i]));

        // Gradient buffer shapes (for pre-allocation on load)
        var grads = plan.Gradients;
        writer.Write(grads.Length);
        for (int i = 0; i < grads.Length; i++)
        {
            writer.Write(grads[i]._shape.Length);
            for (int d = 0; d < grads[i]._shape.Length; d++)
                writer.Write(grads[i]._shape[d]);
        }

        // ── Footer ──────────────────────────────────────────────────────
        writer.Flush();
        var bodyBytes = body.ToArray();
        ulong checksum = XXHash64.Compute(bodyBytes, 0, bodyBytes.Length);
        stream.Write(bodyBytes, 0, bodyBytes.Length);
        using var footerWriter = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        footerWriter.Write((long)bodyBytes.Length);
        footerWriter.Write((long)checksum);
    }

    private static void WriteStep<T>(BinaryWriter writer, CompiledStep<T> step, TensorIdMap<T> tensorMap)
    {
        writer.Write((byte)step.OpType);
        var nameBytes = Encoding.UTF8.GetBytes(step.OpName);
        writer.Write(nameBytes.Length);
        writer.Write(nameBytes);

        writer.Write(step.Inputs.Length);
        for (int j = 0; j < step.Inputs.Length; j++)
            writer.Write(tensorMap.GetId(step.Inputs[j]));

        writer.Write(tensorMap.GetId(step.OutputBuffer));

        SavedStateSerializer.Write(writer, step.SavedState, tensorMap);
    }
}
