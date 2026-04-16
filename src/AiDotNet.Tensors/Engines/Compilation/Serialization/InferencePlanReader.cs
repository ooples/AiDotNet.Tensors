using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Reads a serialized inference plan from a binary stream and reconstitutes a
/// <see cref="CompiledInferencePlan{T}"/>. Returns null (rather than throwing)
/// if the version stamp doesn't match the current runtime — this forces
/// recompile rather than silently mis-replaying an incompatible plan.
/// </summary>
internal static class InferencePlanReader
{
    /// <summary>
    /// Reads and validates the plan from <paramref name="stream"/>. Returns
    /// the rehydrated plan, or null if incompatible (version, codec, hardware,
    /// element type mismatch). Throws <see cref="InvalidDataException"/> on
    /// corruption (bad magic, truncated, checksum mismatch).
    /// </summary>
    internal static ICompiledPlan<T>? Read<T>(Stream stream, IEngine engine)
    {
        // Read the entire stream for checksum validation.
        byte[] allBytes;
        using (var ms = new MemoryStream())
        {
            stream.CopyTo(ms);
            allBytes = ms.ToArray();
        }

        if (allBytes.Length < 16) // minimum: some header + footer
            throw new InvalidDataException("Plan file is too short to be valid.");

        // ── Footer validation ───────────────────────────────────────────
        long storedSize = BitConverter.ToInt64(allBytes, allBytes.Length - 16);
        long storedChecksum = BitConverter.ToInt64(allBytes, allBytes.Length - 8);
        int bodyLength = allBytes.Length - 16;

        if (storedSize != bodyLength)
            throw new InvalidDataException(
                $"Plan file size mismatch: footer says {storedSize} bytes, " +
                $"actual body is {bodyLength}. File may be corrupt or truncated.");

        ulong computed = XXHash64.Compute(allBytes, 0, bodyLength);
        if ((long)computed != storedChecksum)
            throw new InvalidDataException(
                "Plan file checksum mismatch — the file is corrupt. " +
                "Delete and recompile.");

        // ── Header ──────────────────────────────────────────────────────
        using var bodyStream = new MemoryStream(allBytes, 0, bodyLength);
        using var reader = new BinaryReader(bodyStream, Encoding.UTF8, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != PlanFormatConstants.Magic)
            throw new InvalidDataException(
                $"Not a plan file: expected magic 0x{PlanFormatConstants.Magic:X8}, " +
                $"got 0x{magic:X8}.");

        ushort formatVersion = reader.ReadUInt16();
        byte planType = reader.ReadByte();
        byte elementTypeCode = reader.ReadByte();
        int stepCount = reader.ReadInt32();

        // Input shape
        int inputRank = reader.ReadInt32();
        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
            inputShape[i] = reader.ReadInt32();

        // Tensor-codec version
        int codecVersion = reader.ReadInt32();

        // Hardware fingerprint
        int fpLen = reader.ReadInt32();
        var fpBytes = reader.ReadBytes(fpLen);
        string hwFingerprint = Encoding.UTF8.GetString(fpBytes);

        // ── Compatibility check ─────────────────────────────────────────
        var compat = new PlanCompatibilityInfo
        {
            FormatVersion = formatVersion,
            TensorCodecVersion = codecVersion,
            HardwareFingerprint = hwFingerprint,
            ElementTypeName = ElementTypeCodeToName(elementTypeCode),
        };
        string? reason = compat.GetIncompatibilityReason<T>();
        if (reason is not null)
            return null; // Incompatible — caller should recompile.

        if (planType != PlanFormatConstants.PlanTypeInference)
            throw new InvalidDataException(
                $"Expected inference plan (type {PlanFormatConstants.PlanTypeInference}), " +
                $"got type {planType}. Use LoadTrainingAsync for training plans.");

        // ── Tensor table ────────────────────────────────────────────────
        var tensorTable = TensorTableReader.Read<T>(reader);

        // Compiled input tensor is ID 0 by convention.
        var compiledInputTensor = tensorTable.Length > 0 ? tensorTable[0] : null;

        // ── Op sequence ─────────────────────────────────────────────────
        int savedStepCount = reader.ReadInt32();
        if (savedStepCount != stepCount)
            throw new InvalidDataException(
                $"Step count mismatch: header says {stepCount}, op section says {savedStepCount}.");

        var steps = new CompiledStep<T>[stepCount];
        for (int i = 0; i < stepCount; i++)
        {
            steps[i] = ReadStep(reader, tensorTable, engine);
        }

        // ── Construct plan ──────────────────────────────────────────────
        var finalOutput = stepCount > 0 ? steps[stepCount - 1].OutputBuffer : new Tensor<T>(new[] { 0 });
        return CompiledInferencePlan<T>.CreateFromDeserialized(
            steps, finalOutput, engine, inputShape, compiledInputTensor);
    }

    private static CompiledStep<T> ReadStep<T>(BinaryReader reader, Tensor<T>[] tensorTable, IEngine engine)
    {
        // OpType
        byte opTypeByte = reader.ReadByte();
        var opType = (OpType)opTypeByte;

        // OpName
        int nameLen = reader.ReadInt32();
        var nameBytes = reader.ReadBytes(nameLen);
        string opName = Encoding.UTF8.GetString(nameBytes);

        // Input tensor IDs
        int inputCount = reader.ReadInt32();
        var inputs = new Tensor<T>[inputCount];
        for (int j = 0; j < inputCount; j++)
        {
            int tensorId = reader.ReadInt32();
            inputs[j] = tensorTable[tensorId];
        }

        // Output tensor ID
        int outputId = reader.ReadInt32();
        var outputBuffer = tensorTable[outputId];

        // SavedState
        var savedState = SavedStateSerializer.Read<T>(reader, tensorTable);

        // Rebuild the forward closure via the op registry.
        var execute = OpSerializationRegistry<T>.RebuildForwardClosure(
            opType, inputs, outputBuffer, savedState);

        return new CompiledStep<T>(opName, execute, outputBuffer, inputs, backwardFn: null, savedState);
    }

    private static string ElementTypeCodeToName(byte code) => code switch
    {
        PlanFormatConstants.ElementTypeFloat  => typeof(float).FullName!,
        PlanFormatConstants.ElementTypeDouble => typeof(double).FullName!,
        PlanFormatConstants.ElementTypeInt32  => typeof(int).FullName!,
        PlanFormatConstants.ElementTypeInt64  => typeof(long).FullName!,
        _ => $"unknown-{code}",
    };
}
