using System.IO;
using System.Text;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Reads a serialized training plan. Two-phase approach:
/// 1. Read header + tensor table + raw op descriptors + param IDs
/// 2. Substitute caller's parameter tensors into the tensor table
/// 3. Replay forward ops through the engine under GraphMode
/// 4. CompileTraining from the resulting LazyNode graph
///
/// Backward functions are NOT deserialized — they're reconstructed from
/// the forward graph by the training compiler, just like the original
/// compilation did. This guarantees backward correctness matches forward.
/// </summary>
internal static class TrainingPlanReader
{
    /// <summary>Raw op descriptor — buffered during phase 1 before step construction.</summary>
    private readonly struct RawOp
    {
        public readonly OpType OpType;
        public readonly string OpName;
        public readonly int[] InputIds;
        public readonly int OutputId;
        public readonly object[]? SavedState;

        public RawOp(OpType opType, string opName, int[] inputIds, int outputId, object[]? savedState)
        {
            OpType = opType;
            OpName = opName;
            InputIds = inputIds;
            OutputId = outputId;
            SavedState = savedState;
        }
    }

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

        // ── Phase 1: Read tensor table + raw op descriptors ─────────────
        var tensorTable = TensorTableReader.Read<T>(reader);

        int savedStepCount = reader.ReadInt32();
        var rawOps = new RawOp[savedStepCount];
        for (int i = 0; i < savedStepCount; i++)
            rawOps[i] = ReadRawOp<T>(reader, tensorTable);

        // ── Read training extension: param IDs ──────────────────────────
        int paramCount = reader.ReadInt32();
        if (paramCount != callerParameters.Length)
            throw new InvalidDataException(
                $"Parameter count mismatch: file has {paramCount}, caller supplied {callerParameters.Length}.");

        // Substitute caller's parameter tensors into the tensor table.
        // The forward graph MUST reference the caller's parameter objects
        // so the training compiler can find them during gradient analysis
        // and backward closures accumulate into the right buffers.
        for (int i = 0; i < paramCount; i++)
        {
            int paramTensorId = reader.ReadInt32();
            tensorTable[paramTensorId] = callerParameters[i];
        }

        // ── Gradient buffer shapes (skip — the compiler allocates its own) ─
        int gradCount = reader.ReadInt32();
        for (int i = 0; i < gradCount; i++)
        {
            int gradRank = reader.ReadInt32();
            for (int d = 0; d < gradRank; d++) reader.ReadInt32();
        }

        // ── Phase 2: Replay forward ops through engine under GraphMode ──
        // Each engine call under GraphMode returns a NEW lazy tensor. We must
        // wire these new outputs as inputs to subsequent ops — the tensor-table
        // references from the saved plan are stale (they point at pre-allocated
        // buffers, not at the fresh lazy outputs the compiler needs to trace).
        //
        // Strategy: maintain a live-tensor map parallel to the tensor table.
        // After each op, capture the return value and store it at the op's
        // output tensor-table ID. When resolving inputs for the next op, look
        // up the live map first; fall back to the (patched) tensor table for
        // leaf inputs and parameters (which aren't produced by any op).
        var liveTensors = new Tensor<T>?[tensorTable.Length];
        // Pre-populate with leaf inputs + parameters (tensor table entries
        // that are NOT any op's output).
        for (int i = 0; i < tensorTable.Length; i++)
            liveTensors[i] = tensorTable[i];

        using (var scope = GraphMode.Enable())
        {
            for (int i = 0; i < rawOps.Length; i++)
            {
                var op = rawOps[i];
                var inputs = new Tensor<T>[op.InputIds.Length];
                for (int j = 0; j < op.InputIds.Length; j++)
                    inputs[j] = liveTensors[op.InputIds[j]]!;

                var output = OpReplay.ReplayThroughEngine(engine, op.OpType, op.OpName, inputs, op.SavedState);
                if (output is not null)
                    liveTensors[op.OutputId] = output;
            }

            return scope.CompileTraining<T>(callerParameters);
        }
    }

    private static RawOp ReadRawOp<T>(BinaryReader reader, Tensor<T>[] tensorTable)
    {
        byte opTypeByte = reader.ReadByte();
        var opType = (OpType)opTypeByte;
        int nameLen = reader.ReadInt32();
        string opName = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));

        int inputCount = reader.ReadInt32();
        var inputIds = new int[inputCount];
        for (int j = 0; j < inputCount; j++)
            inputIds[j] = reader.ReadInt32();

        int outputId = reader.ReadInt32();

        // SavedState must be read now (it's in-stream), but tensor refs
        // inside it may point to pre-substitution tensor table entries.
        // This is acceptable because savedState tensors (like BatchNorm
        // mean/variance) are NOT parameters — they're intermediate state
        // that the compiler produces fresh during re-compilation.
        var savedState = SavedStateSerializer.Read<T>(reader, tensorTable);

        return new RawOp(opType, opName, inputIds, outputId, savedState);
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
