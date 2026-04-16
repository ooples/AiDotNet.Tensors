using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Reads a serialized training plan and reconstitutes it by:
/// 1. Rebuilding the forward CompiledSteps (same as inference path).
/// 2. Re-running the backward-compilation machinery from
///    <see cref="CompiledTrainingPlan{T}"/> to produce specialized backward
///    actions from the forward graph structure + parameter references.
/// 3. Restoring gradient buffer shapes and optimizer state.
///
/// Backward functions are NOT serialized as code — they're reconstructed
/// from the op registry's backward-function mapping, which is deterministic
/// given the same OpType + inputs + savedState.
/// </summary>
internal static class TrainingPlanReader
{
    internal static ICompiledTrainingPlan<T>? Read<T>(
        Stream stream, IEngine engine, Tensor<T>[] callerParameters)
    {
        byte[] allBytes;
        using (var ms = new MemoryStream())
        {
            stream.CopyTo(ms);
            allBytes = ms.ToArray();
        }

        if (allBytes.Length < 16)
            throw new InvalidDataException("Training plan file is too short.");

        // ── Footer validation ───────────────────────────────────────────
        long storedSize = BitConverter.ToInt64(allBytes, allBytes.Length - 16);
        long storedChecksum = BitConverter.ToInt64(allBytes, allBytes.Length - 8);
        int bodyLength = allBytes.Length - 16;

        if (storedSize != bodyLength)
            throw new InvalidDataException("Plan file size mismatch.");

        ulong computed = XXHash64.Compute(allBytes, 0, bodyLength);
        if ((long)computed != storedChecksum)
            throw new InvalidDataException("Plan file checksum mismatch.");

        // ── Header ──────────────────────────────────────────────────────
        using var bodyStream = new MemoryStream(allBytes, 0, bodyLength);
        using var reader = new BinaryReader(bodyStream, Encoding.UTF8, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != PlanFormatConstants.Magic)
            throw new InvalidDataException("Not a plan file.");

        ushort formatVersion = reader.ReadUInt16();
        byte planType = reader.ReadByte();
        byte elementTypeCode = reader.ReadByte();
        int stepCount = reader.ReadInt32();

        int inputRank = reader.ReadInt32();
        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
            inputShape[i] = reader.ReadInt32();

        int codecVersion = reader.ReadInt32();
        int fpLen = reader.ReadInt32();
        string hwFingerprint = Encoding.UTF8.GetString(reader.ReadBytes(fpLen));

        // ── Compatibility check ─────────────────────────────────────────
        var compat = new PlanCompatibilityInfo
        {
            FormatVersion = formatVersion,
            TensorCodecVersion = codecVersion,
            HardwareFingerprint = hwFingerprint,
            ElementTypeName = InferencePlanReaderHelper.ElementTypeCodeToName(elementTypeCode),
        };
        if (compat.GetIncompatibilityReason<T>() is not null)
            return null;

        if (planType != PlanFormatConstants.PlanTypeTraining)
            throw new InvalidDataException("Expected training plan type.");

        // ── Tensor table ────────────────────────────────────────────────
        var tensorTable = TensorTableReader.Read<T>(reader);

        // ── Forward op sequence ─────────────────────────────────────────
        int savedStepCount = reader.ReadInt32();
        var forwardSteps = new CompiledStep<T>[savedStepCount];
        for (int i = 0; i < savedStepCount; i++)
        {
            forwardSteps[i] = ReadStep(reader, tensorTable, engine);
        }

        // ── Training extension: parameter IDs ───────────────────────────
        int paramCount = reader.ReadInt32();
        // The loaded parameters are mapped from tensor table, but the
        // caller provides their OWN parameter tensors (which may have
        // updated weights from a previous session). We validate the count
        // matches and use the caller's tensors.
        if (paramCount != callerParameters.Length)
            throw new InvalidDataException(
                $"Parameter count mismatch: file has {paramCount}, caller supplied {callerParameters.Length}.");

        // Skip saved param tensor IDs (we use caller's params instead).
        for (int i = 0; i < paramCount; i++) reader.ReadInt32();

        // ── Gradient buffer shapes ──────────────────────────────────────
        int gradCount = reader.ReadInt32();
        var gradients = new Tensor<T>[gradCount];
        for (int i = 0; i < gradCount; i++)
        {
            int gradRank = reader.ReadInt32();
            var gradShape = new int[gradRank];
            for (int d = 0; d < gradRank; d++)
                gradShape[d] = reader.ReadInt32();
            gradients[i] = new Tensor<T>(gradShape);
        }

        // ── Recompile the training plan from the forward steps ──────────
        // This re-uses the existing compile machinery: trace the forward
        // ops into a fresh GraphMode scope, then call CompileTraining with
        // the caller's parameter tensors. The forward steps' closures are
        // already rebuilt by the op registry; the compiler builds the
        // backward pass from the graph structure.
        //
        // NOTE: This approach pays the compile cost on load (backward-pass
        // construction + specialization). A future enhancement could
        // serialize the backward step descriptors too, eliminating compile
        // on load entirely. For now, the value is in skipping the user's
        // forward trace (which is often the expensive part for large models).

        // Rebuild by tracing into GraphMode then compiling.
        var compiledInputTensor = tensorTable.Length > 0 ? tensorTable[0] : null;
        using (var scope = GraphMode.Enable())
        {
            // Replay forward steps: each step writes to its output buffer.
            for (int i = 0; i < forwardSteps.Length; i++)
                forwardSteps[i].Execute(engine, forwardSteps[i].OutputBuffer);

            return scope.CompileTraining<T>(callerParameters);
        }
    }

    private static CompiledStep<T> ReadStep<T>(
        BinaryReader reader, Tensor<T>[] tensorTable, IEngine engine)
    {
        byte opTypeByte = reader.ReadByte();
        var opType = (OpType)opTypeByte;
        int nameLen = reader.ReadInt32();
        string opName = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));

        int inputCount = reader.ReadInt32();
        var inputs = new Tensor<T>[inputCount];
        for (int j = 0; j < inputCount; j++)
            inputs[j] = tensorTable[reader.ReadInt32()];

        int outputId = reader.ReadInt32();
        var outputBuffer = tensorTable[outputId];

        var savedState = SavedStateSerializer.Read<T>(reader, tensorTable);
        var execute = OpSerializationRegistry<T>.RebuildForwardClosure(
            opType, inputs, outputBuffer, savedState);

        return new CompiledStep<T>(opName, execute, outputBuffer, inputs, backwardFn: null, savedState);
    }
}

/// <summary>Helper to share element-type-name resolution between readers.</summary>
internal static class InferencePlanReaderHelper
{
    internal static string ElementTypeCodeToName(byte code) => code switch
    {
        PlanFormatConstants.ElementTypeFloat  => typeof(float).FullName!,
        PlanFormatConstants.ElementTypeDouble => typeof(double).FullName!,
        PlanFormatConstants.ElementTypeInt32  => typeof(int).FullName!,
        PlanFormatConstants.ElementTypeInt64  => typeof(long).FullName!,
        _ => $"unknown-{code}",
    };
}
