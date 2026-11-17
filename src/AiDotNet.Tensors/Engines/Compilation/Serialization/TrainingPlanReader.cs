using System.IO;
using System.Text;
using AiDotNet.Tensors.Engines.Autodiff;
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
        public readonly bool RecordedUnderNoGrad;
        public readonly bool HasBackward;
        public readonly int[]? DifferentiableInputIndices;

        public RawOp(OpType opType, string opName, int[] inputIds, int outputId, object[]? savedState,
            bool recordedUnderNoGrad, bool hasBackward, int[]? differentiableInputIndices)
        {
            OpType = opType;
            OpName = opName;
            InputIds = inputIds;
            OutputId = outputId;
            SavedState = savedState;
            RecordedUnderNoGrad = recordedUnderNoGrad;
            HasBackward = hasBackward;
            DifferentiableInputIndices = differentiableInputIndices;
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
        // Read checksum as ulong to stay symmetrical with the writer (XXHash64
        // returns ulong). See InferencePlanReader for rationale.
        ulong storedChecksum = BitConverter.ToUInt64(allBytes, allBytes.Length - 8);
        int bodyLength = allBytes.Length - 16;

        if (storedSize != bodyLength)
            throw new InvalidDataException("Plan file size mismatch.");

        ulong computed = XXHash64.Compute(allBytes, 0, bodyLength);
        if (computed != storedChecksum)
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
        RequireNonNegative(stepCount, nameof(stepCount));

        int inputRank = reader.ReadInt32();
        RequireNonNegative(inputRank, nameof(inputRank));
        RequireWithinStream(bodyStream, (long)inputRank * sizeof(int), nameof(inputRank));
        var inputShape = new int[inputRank];
        for (int i = 0; i < inputRank; i++)
            inputShape[i] = reader.ReadInt32();

        int codecVersion = reader.ReadInt32();
        int fpLen = reader.ReadInt32();
        RequireNonNegative(fpLen, nameof(fpLen));
        RequireWithinStream(bodyStream, fpLen, nameof(fpLen));
        // BinaryReader.ReadBytes returns a short array on truncated streams
        // (silent short read) — validate length matches the requested count,
        // matching the pattern in SavedStateSerializer.ReadString. Without
        // this, a truncated fingerprint would decode to a short string and
        // let the compatibility check compare against a wrong value.
        var fpBytes = reader.ReadBytes(fpLen);
        if (fpBytes.Length != fpLen)
            throw new InvalidDataException(
                $"Training plan hardware fingerprint was truncated: expected {fpLen} bytes, got {fpBytes.Length}.");
        string hwFingerprint = Encoding.UTF8.GetString(fpBytes);

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
        RequireNonNegative(savedStepCount, nameof(savedStepCount));
        // Cross-check the header's stepCount against the op-section count.
        // Divergence means the writer mis-matched its own two fields, or the
        // file was truncated mid-op-stream. Either way it's corrupt.
        if (savedStepCount != stepCount)
            throw new InvalidDataException(
                $"Training plan step count mismatch: header says {stepCount}, op section says {savedStepCount}.");
        RequireWithinStream(bodyStream, (long)savedStepCount * PlanFormatConstants.MinTrainingStepDescriptorBytes, nameof(savedStepCount));
        var rawOps = new RawOp[savedStepCount];
        for (int i = 0; i < savedStepCount; i++)
            rawOps[i] = ReadRawOp<T>(reader, tensorTable, bodyStream);

        // ── Read training extension: param IDs ──────────────────────────
        int paramCount = reader.ReadInt32();
        RequireNonNegative(paramCount, nameof(paramCount));
        if (paramCount != callerParameters.Length)
            throw new InvalidDataException(
                $"Parameter count mismatch: file has {paramCount}, caller supplied {callerParameters.Length}.");

        // Substitute caller's parameter tensors into the tensor table.
        // The forward graph MUST reference the caller's parameter objects
        // so the training compiler can find them during gradient analysis
        // and backward closures accumulate into the right buffers.
        var parameterIds = new HashSet<int>();
        for (int i = 0; i < paramCount; i++)
        {
            int paramTensorId = reader.ReadInt32();
            if ((uint)paramTensorId >= (uint)tensorTable.Length)
                throw new InvalidDataException(
                    $"Training plan parameter {i} references tensor ID {paramTensorId} " +
                    $"out of range [0, {tensorTable.Length}). The file is corrupt.");
            if (!parameterIds.Add(paramTensorId))
                throw new InvalidDataException($"Training plan parameter {i} repeats tensor ID {paramTensorId}.");
            tensorTable[paramTensorId] = callerParameters[i];
        }

        // Gradient buffers are reconstructed, but their metadata must describe these canonical parameters.
        int gradCount = reader.ReadInt32();
        RequireNonNegative(gradCount, nameof(gradCount));
        if (gradCount != paramCount)
            throw new InvalidDataException($"Gradient count mismatch: file has {gradCount}, expected {paramCount}.");
        for (int i = 0; i < gradCount; i++)
        {
            int gradRank = reader.ReadInt32();
            RequireNonNegative(gradRank, nameof(gradRank));
            RequireWithinStream(bodyStream, (long)gradRank * sizeof(int), nameof(gradRank));
            if (gradRank != callerParameters[i].Rank)
                throw new InvalidDataException($"Gradient {i} rank does not match its parameter.");
            for (int d = 0; d < gradRank; d++)
                if (reader.ReadInt32() != callerParameters[i]._shape[d])
                    throw new InvalidDataException($"Gradient {i} shape does not match its parameter.");
        }

        var expectedParamLengths = new int[paramCount];
        for (int i = 0; i < paramCount; i++) expectedParamLengths[i] = callerParameters[i].Length;
        var optimizerCheckpoint = FusedOptimizerCheckpointSerializer.Read(reader, expectedParameterLengths: expectedParamLengths);
        if (optimizerCheckpoint is not null && optimizerCheckpoint.Parameters.Length != paramCount)
            throw new InvalidDataException("Optimizer parameter-state count does not match the canonical parameter count.");

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

        using (var scope = GraphMode.EnableTraining(callerParameters))
        {
            for (int i = 0; i < rawOps.Length; i++)
            {
                var op = rawOps[i];
                var inputs = new Tensor<T>[op.InputIds.Length];
                for (int j = 0; j < op.InputIds.Length; j++)
                {
                    int inputId = op.InputIds[j];
                    // Bounds-check before indexing so a truncated or fabricated
                    // plan surfaces an InvalidDataException via the CompiledPlanLoader
                    // try/catch instead of an IndexOutOfRangeException deep in
                    // replay.
                    if ((uint)inputId >= (uint)liveTensors.Length)
                    {
                        throw new InvalidDataException(
                            $"Training plan op {i} ('{op.OpName}') references input tensor ID {inputId} " +
                            $"which is out of range [0, {liveTensors.Length}). The plan file is corrupt.");
                    }
                    var inputTensor = liveTensors[inputId];
                    if (inputTensor is null)
                    {
                        // Null means the tensor was never seeded from the tensor
                        // table and no preceding op produced it — the graph is
                        // topologically invalid.
                        throw new InvalidDataException(
                            $"Training plan op {i} ('{op.OpName}') consumes tensor ID {inputId} " +
                            "before any producer ran. The plan file is corrupt.");
                    }
                    inputs[j] = inputTensor;
                }

                // Each omitted operand occurrence gets its own forward identity with a permanent gradient stop.
                // Tensor-key filtering would be wrong for Multiply(p, p) when only one occurrence carries gradient,
                // or when the omitted operand's producer also feeds a live branch elsewhere in the graph.
                if (op.HasBackward && op.DifferentiableInputIndices is { } derivativeInputs)
                {
                    int derivative = 0;
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        if (derivative < derivativeInputs.Length && derivativeInputs[derivative] == j)
                        {
                            derivative++;
                            continue;
                        }
                        if (inputs[j].LazySource is not LazyNode<T> { OpName: "StopGradient", BackwardFn: null, WritesInPlace: false })
                            inputs[j] = engine.StopGradient(inputs[j]);
                    }
                }

                if ((uint)op.OutputId >= (uint)liveTensors.Length)
                {
                    throw new InvalidDataException(
                        $"Training plan op {i} ('{op.OpName}') writes to output tensor ID {op.OutputId} " +
                        $"which is out of range [0, {liveTensors.Length}). The plan file is corrupt.");
                }

                Tensor<T>? output;
                // Rebuild the operation's own derivative only when the saved graph carried it. A permanent
                // gradient stop and a no-grad in-place write remain distinct through the metadata below.
                using (op.HasBackward && !op.RecordedUnderNoGrad ? null : new NoGradScope<T>())
                    output = OpReplay.ReplayThroughEngine(
                        engine, op.OpType, op.OpName, inputs, op.SavedState, tensorTable[op.OutputId]._shape);
                if (output is null)
                {
                    // ReplayThroughEngine returned null because this runtime
                    // has no replay handler for the op — silently leaving the
                    // producing step absent would let CompileTraining succeed
                    // with a partial graph whose downstream steps read stale
                    // or default tensors. Force the cache-miss path instead
                    // so the caller recompiles from source.
                    throw new InvalidDataException(
                        $"Training plan op {i} ('{op.OpName}', {op.OpType}) is not supported by this runtime. " +
                        "Re-compile from source instead of loading this plan.");
                }

                if (output.LazySource is not LazyNode<T> restoredNode)
                    throw new InvalidDataException($"Training plan op {i} ('{op.OpName}') did not record a typed operation.");
                bool noGradIdentity = op.RecordedUnderNoGrad && restoredNode.WritesInPlace && restoredNode.PriorWriter is not null;
                if (op.HasBackward && restoredNode.BackwardFn is null && !noGradIdentity)
                    throw new InvalidDataException($"Training plan op {i} ('{op.OpName}') cannot restore its backward function.");
                restoredNode.RecordedUnderNoGrad = op.RecordedUnderNoGrad;
                var restoredInputs = restoredNode.GetInputsArray();
                // A registered composite (for example FusedLinear) can expand to several native nodes. Its final
                // node has different input positions; the operand stops above already preserve the original edges.
                if (restoredInputs.Length == inputs.Length
                    && restoredInputs.Zip(inputs, ReferenceEquals).All(equal => equal))
                    restoredNode.SetDifferentiableInputIndices(op.DifferentiableInputIndices);

                liveTensors[op.OutputId] = output;
            }

            var plan = scope.CompileTraining<T>(callerParameters);
            if (optimizerCheckpoint is not null)
            {
                if (plan is not CompiledTrainingPlan<T> concrete)
                {
                    plan.Dispose();
                    throw new InvalidDataException(
                        "Training plan loader produced an unknown ICompiledTrainingPlan implementation; " +
                        "cannot restore fused optimizer checkpoint state.");
                }
                try
                {
                    concrete.RestoreFusedOptimizerCheckpoint(optimizerCheckpoint);
                }
                catch
                {
                    concrete.Dispose();
                    throw;
                }
            }
            return plan;
        }
    }

    private static RawOp ReadRawOp<T>(BinaryReader reader, Tensor<T>[] tensorTable, Stream bodyStream)
    {
        byte opTypeByte = reader.ReadByte();
        var opType = (OpType)opTypeByte;
        int nameLen = reader.ReadInt32();
        RequireNonNegative(nameLen, nameof(nameLen));
        RequireWithinStream(bodyStream, nameLen, nameof(nameLen));
        var nameBytes = reader.ReadBytes(nameLen);
        if (nameBytes.Length != nameLen)
            throw new InvalidDataException(
                $"Training plan op name was truncated: expected {nameLen} bytes, got {nameBytes.Length}.");
        string opName = Encoding.UTF8.GetString(nameBytes);
        byte gradientFlags = reader.ReadByte();
        if ((gradientFlags & ~PlanFormatConstants.OpGradientFlagsMask) != 0)
            throw new InvalidDataException($"Training plan op '{opName}' has invalid gradient flags {gradientFlags}.");
        int derivativeCount = reader.ReadInt32();
        if (derivativeCount < -1)
            throw new InvalidDataException($"Training plan op '{opName}' has invalid derivative input count {derivativeCount}.");
        int[]? differentiableInputs = null;
        if (derivativeCount >= 0)
        {
            RequireWithinStream(bodyStream, (long)derivativeCount * sizeof(int), nameof(derivativeCount));
            differentiableInputs = new int[derivativeCount];
            for (int i = 0; i < derivativeCount; i++) differentiableInputs[i] = reader.ReadInt32();
        }
        // A plan written before this runtime knew the op stored OpType.Unknown; its name still identifies it.
        if (opType == OpType.Unknown)
            opType = OpTypeParser.Parse(opName);

        int inputCount = reader.ReadInt32();
        RequireNonNegative(inputCount, nameof(inputCount));
        RequireWithinStream(bodyStream, (long)inputCount * sizeof(int), nameof(inputCount));
        var inputIds = new int[inputCount];
        for (int j = 0; j < inputCount; j++)
            inputIds[j] = reader.ReadInt32();
        if (differentiableInputs is not null)
        {
            int previous = -1;
            foreach (int index in differentiableInputs)
            {
                if (index <= previous || index >= inputCount)
                    throw new InvalidDataException($"Training plan op '{opName}' has unsorted, repeated or out-of-range derivative inputs.");
                previous = index;
            }
        }

        int outputId = reader.ReadInt32();
        if (opType == OpType.TensorAddInPlace)
        {
            if (inputCount != 2 || outputId != inputIds[0])
                throw new InvalidDataException("TensorAddInPlace must write to its first of exactly two inputs.");
            // The target is an alias, not a replaceable operand. Wrapping it in StopGradient would mutate a copy
            // and change the forward. RecordInPlace does not declare a target-detaching partial-edge contract.
            if (differentiableInputs is not null && (differentiableInputs.Length == 0 || differentiableInputs[0] != 0))
                throw new InvalidDataException("TensorAddInPlace derivative metadata cannot exclude its mutation target.");
        }

        // SavedState must be read now (it's in-stream), but tensor refs
        // inside it may point to pre-substitution tensor table entries.
        // This is acceptable because savedState tensors (like BatchNorm
        // mean/variance) are NOT parameters — they're intermediate state
        // that the compiler produces fresh during re-compilation.
        var savedState = SavedStateSerializer.Read<T>(reader, tensorTable);

        return new RawOp(opType, opName, inputIds, outputId, savedState,
            recordedUnderNoGrad: (gradientFlags & PlanFormatConstants.OpGradientFlagRecordedUnderNoGrad) != 0,
            hasBackward: (gradientFlags & PlanFormatConstants.OpGradientFlagHasBackward) != 0,
            differentiableInputIndices: differentiableInputs);
    }

    // ── Validation helpers ──────────────────────────────────────────────
    // Mirror of InferencePlanReader's RequireNonNegative / RequireWithinStream.
    // Keep both readers symmetric so corruption detection is consistent
    // across inference-only and training plan files.

    private static void RequireNonNegative(int value, string name)
    {
        if (value < 0)
            throw new InvalidDataException(
                $"Training plan field '{name}' is negative ({value}). The file is corrupt.");
    }

    private static void RequireWithinStream(Stream bodyStream, long requiredBytes, string name)
    {
        long remaining = bodyStream.Length - bodyStream.Position;
        if (requiredBytes > remaining)
            throw new InvalidDataException(
                $"Training plan field '{name}' requires {requiredBytes} bytes but only " +
                $"{remaining} remain. The file is truncated or corrupt.");
    }
}

/// <summary>Helper to share element-type-name resolution between readers.</summary>
internal static class InferencePlanReaderHelper
{
    internal static string ElementTypeCodeToName(byte code) => code switch
    {
        PlanFormatConstants.ElementTypeFloat   => typeof(float).FullName!,
        PlanFormatConstants.ElementTypeDouble  => typeof(double).FullName!,
        PlanFormatConstants.ElementTypeInt32   => typeof(int).FullName!,
        PlanFormatConstants.ElementTypeInt64   => typeof(long).FullName!,
        PlanFormatConstants.ElementTypeFloat16 => "System.Half",
        _ => $"unknown-{code}",
    };
}
