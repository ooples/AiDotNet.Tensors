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
        // Checksum is written as a ulong (XXHash64 returns unsigned). Reading
        // it as ToUInt64 keeps the types symmetrical between reader and writer
        // and avoids the need for a signed reinterpret cast on the comparison.
        ulong storedChecksum = BitConverter.ToUInt64(allBytes, allBytes.Length - 8);
        int bodyLength = allBytes.Length - 16;

        if (storedSize != bodyLength)
            throw new InvalidDataException(
                $"Plan file size mismatch: footer says {storedSize} bytes, " +
                $"actual body is {bodyLength}. File may be corrupt or truncated.");

        ulong computed = XXHash64.Compute(allBytes, 0, bodyLength);
        if (computed != storedChecksum)
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
        RequireNonNegative(stepCount, nameof(stepCount));

        // Input shape. Rank bounded by the remaining body size — every rank
        // entry consumes 4 bytes, so a corrupt file claiming 2^30 dims can't
        // slip past RequireWithinStream.
        int inputRank = reader.ReadInt32();
        RequireNonNegative(inputRank, nameof(inputRank));
        RequireWithinStream(bodyStream, (long)inputRank * sizeof(int), nameof(inputRank));
        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
            inputShape[i] = reader.ReadInt32();

        // Tensor-codec version
        int codecVersion = reader.ReadInt32();

        // Hardware fingerprint — UTF-8 bytes, length-prefixed.
        int fpLen = reader.ReadInt32();
        RequireNonNegative(fpLen, nameof(fpLen));
        RequireWithinStream(bodyStream, fpLen, nameof(fpLen));
        var fpBytes = reader.ReadBytes(fpLen);
        if (fpBytes.Length != fpLen)
            throw new InvalidDataException(
                $"Inference plan hardware fingerprint was truncated: expected {fpLen} bytes, got {fpBytes.Length}.");
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
            steps[i] = ReadStep(reader, tensorTable, engine, bodyStream);
        }

        // ── Final output identity (format v2+) ──────────────────────────
        Tensor<T> finalOutput;
        if (formatVersion >= 2)
        {
            int finalOutputId = reader.ReadInt32();
            if (finalOutputId >= 0 && finalOutputId < tensorTable.Length)
                finalOutput = tensorTable[finalOutputId];
            else
                finalOutput = stepCount > 0 ? steps[stepCount - 1].OutputBuffer : new Tensor<T>(new[] { 0 });
        }
        else
        {
            // v1 used the last-step heuristic implicitly.
            finalOutput = stepCount > 0 ? steps[stepCount - 1].OutputBuffer : new Tensor<T>(new[] { 0 });
        }

        // ── Construct plan ──────────────────────────────────────────────
        return CompiledInferencePlan<T>.CreateFromDeserialized(
            steps, finalOutput, engine, inputShape, compiledInputTensor);
    }

    private static CompiledStep<T> ReadStep<T>(BinaryReader reader, Tensor<T>[] tensorTable, IEngine engine, Stream bodyStream)
    {
        // OpType
        byte opTypeByte = reader.ReadByte();
        var opType = (OpType)opTypeByte;

        // OpName
        int nameLen = reader.ReadInt32();
        RequireNonNegative(nameLen, nameof(nameLen));
        RequireWithinStream(bodyStream, nameLen, nameof(nameLen));
        var nameBytes = reader.ReadBytes(nameLen);
        if (nameBytes.Length != nameLen)
            throw new InvalidDataException(
                $"Inference plan op name was truncated: expected {nameLen} bytes, got {nameBytes.Length}.");
        string opName = Encoding.UTF8.GetString(nameBytes);

        // Input tensor IDs
        int inputCount = reader.ReadInt32();
        RequireNonNegative(inputCount, nameof(inputCount));
        RequireWithinStream(bodyStream, (long)inputCount * sizeof(int), nameof(inputCount));
        var inputs = new Tensor<T>[inputCount];
        for (int j = 0; j < inputCount; j++)
        {
            int tensorId = reader.ReadInt32();
            if ((uint)tensorId >= (uint)tensorTable.Length)
                throw new InvalidDataException(
                    $"Step '{opName}' input {j} references tensor ID {tensorId} " +
                    $"out of range [0, {tensorTable.Length}). The plan file is corrupt.");
            inputs[j] = tensorTable[tensorId];
        }

        // Output tensor ID
        int outputId = reader.ReadInt32();
        if ((uint)outputId >= (uint)tensorTable.Length)
            throw new InvalidDataException(
                $"Step '{opName}' writes output tensor ID {outputId} " +
                $"out of range [0, {tensorTable.Length}). The plan file is corrupt.");
        var outputBuffer = tensorTable[outputId];

        // SavedState
        var savedState = SavedStateSerializer.Read<T>(reader, tensorTable);

        // Rebuild the forward closure via the op registry.
        var execute = OpSerializationRegistry<T>.RebuildForwardClosure(
            opType, inputs, outputBuffer, savedState);

        return new CompiledStep<T>(opName, execute, outputBuffer, inputs, backwardFn: null, savedState);
    }

    // ── Validation helpers ──────────────────────────────────────────────
    // Used throughout the reader to keep file-provided counts/lengths from
    // causing IndexOutOfRange or OutOfMemoryException. All failures raise
    // InvalidDataException so the CompiledPlanLoader try/catch treats them
    // as corrupt-file errors rather than unexpected exceptions.

    private static void RequireNonNegative(int value, string name)
    {
        if (value < 0)
            throw new InvalidDataException(
                $"Plan file field '{name}' is negative ({value}). The file is corrupt.");
    }

    private static void RequireWithinStream(Stream bodyStream, long requiredBytes, string name)
    {
        long remaining = bodyStream.Length - bodyStream.Position;
        if (requiredBytes > remaining)
            throw new InvalidDataException(
                $"Plan file field '{name}' requires {requiredBytes} bytes but only " +
                $"{remaining} remain. The file is truncated or corrupt.");
    }

    private static string ElementTypeCodeToName(byte code) => code switch
    {
        PlanFormatConstants.ElementTypeFloat   => typeof(float).FullName!,
        PlanFormatConstants.ElementTypeDouble  => typeof(double).FullName!,
        PlanFormatConstants.ElementTypeInt32   => typeof(int).FullName!,
        PlanFormatConstants.ElementTypeInt64   => typeof(long).FullName!,
        // System.Half exists on net5+ but not on net471. Use the well-known
        // full name directly so the compatibility comparison matches whatever
        // runtime the plan was originally written on, regardless of whether
        // the current build can reference System.Half.
        PlanFormatConstants.ElementTypeFloat16 => "System.Half",
        _ => $"unknown-{code}",
    };
}
