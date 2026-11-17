using System.IO;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

internal static class FusedOptimizerCheckpointSerializer
{
    private const int MaxUntrustedScheduleCount = 1_048_576;
    private const int MaxUntrustedScheduleElements = 65_536;
    private const int MaxUntrustedParameterCount = 4_194_304;
    private const int MaxLbfgsMemorySize = 4096;
    private const long MaxUntrustedMetadataBytes = 16L * 1024 * 1024;
    private const long MaxUntrustedParameterBufferBytes = 256L * 1024 * 1024;

    internal static void Write(BinaryWriter writer, FusedOptimizerCheckpoint? checkpoint)
    {
        if (checkpoint is not null) ValidateStateMetadata(checkpoint);
        writer.Write(checkpoint is not null);
        if (checkpoint is null) return;

        writer.Write((int)checkpoint.OptimizerType);
        writer.Write(checkpoint.IsGrouped);
        writer.Write(checkpoint.OptimizerStep);
        writer.Write(checkpoint.Beta1);
        writer.Write(checkpoint.Beta2);
        writer.Write(checkpoint.Epsilon);
        writer.Write(checkpoint.WeightDecay);
        writer.Write((int)checkpoint.MomentStorageMode);
        writer.Write(checkpoint.Int8MomentBlockSize);
        // Which parameters hold int8 rather than fp32 moments (format version 7).
        writer.Write(checkpoint.Int8MinQuantizedLength);
        writer.Write(checkpoint.MaxGradNorm);
        WriteExtras(writer, checkpoint.Extras);
        WriteLrSchedules(writer, checkpoint.Schedules);
        WriteIntArray(writer, checkpoint.ParamToGroup);
        // Per-group optimizer types and weight decays. Null (uniform) is encoded as a -1 length so it stays
        // distinguishable from an empty array, and so a restore cannot silently rebuild a heterogeneous plan
        // as a uniform one — every group quietly switched to the fallback optimizer, mid-run, with no error.
        WriteOptimizerTypeArray(writer, checkpoint.GroupOptimizerTypes);
        WriteDoubleArray(writer, checkpoint.GroupWeightDecays);
        WriteScalars(writer, checkpoint.Scalars);
        WriteFixedActiveDomain(writer, checkpoint.FixedActiveDomain);

        writer.Write(checkpoint.Parameters.Length);
        for (int i = 0; i < checkpoint.Parameters.Length; i++)
            WriteParameter(writer, checkpoint.Parameters[i]);
        WriteLbfgsHistory(writer, checkpoint.LbfgsHistory);
    }

    /// <param name="reader">The reader positioned at the checkpoint.</param>
    /// <param name="doubleHyperparameters">Whether beta1, beta2, epsilon and weight decay were written as double.</param>
    /// <param name="hasVelocityIncludesLearningRate">Whether the extras end with the VelocityIncludesLearningRate flag.</param>
    /// <param name="hasEnrollmentMetadata">Whether fixed active domain, per-parameter accepted-update ages, and L-BFGS history are serialized.</param>
    /// <param name="expectedParameterLengths">Optional canonical parameter element lengths known by the caller before reading.</param>
    internal static FusedOptimizerCheckpoint? Read(
        BinaryReader reader,
        bool doubleHyperparameters = true,
        bool hasVelocityIncludesLearningRate = true,
        bool hasEnrollmentMetadata = true,
        int[]? expectedParameterLengths = null)
    {
        if (!reader.ReadBoolean()) return null;
        double ReadHyperparameter() => doubleHyperparameters ? reader.ReadDouble() : reader.ReadSingle();

        int? expectedParamCount = expectedParameterLengths?.Length;
        long? expectedTotalElements = null;
        if (expectedParameterLengths is not null)
        {
            long sum = 0;
            for (int i = 0; i < expectedParameterLengths.Length; i++)
                sum = checked(sum + expectedParameterLengths[i]);
            expectedTotalElements = sum;
        }

        var checkpoint = new FusedOptimizerCheckpoint
        {
            OptimizerType = (OptimizerType)reader.ReadInt32(),
            IsGrouped = reader.ReadBoolean(),
            OptimizerStep = reader.ReadInt32(),
            Beta1 = ReadHyperparameter(),
            Beta2 = ReadHyperparameter(),
            Epsilon = ReadHyperparameter(),
            WeightDecay = ReadHyperparameter(),
            MomentStorageMode = (FusedMomentStorageMode)reader.ReadInt32(),
            Int8MomentBlockSize = reader.ReadInt32(),
            Int8MinQuantizedLength = reader.ReadInt32(),
            MaxGradNorm = reader.ReadDouble(),
            Extras = ReadExtras(reader, hasVelocityIncludesLearningRate),
        };
        checkpoint.Schedules = ReadLrSchedules(reader, expectedParamCount);
        int maxParamGroupEntries = expectedParamCount ?? MaxUntrustedParameterCount;
        checkpoint.ParamToGroup = ReadIntArray(reader, maxParamGroupEntries);
        if (expectedParamCount is int exactParamCount
            && checkpoint.ParamToGroup is { } p2g
            && p2g.Length != exactParamCount)
        {
            throw new InvalidDataException("Optimizer parameter-to-group mapping count does not match the canonical parameter count.");
        }
        int maxGroupEntries = Math.Max(checkpoint.Schedules.Length, expectedParamCount ?? MaxUntrustedScheduleCount);
        checkpoint.GroupOptimizerTypes = ReadOptimizerTypeArray(reader, maxGroupEntries);
        checkpoint.GroupWeightDecays = doubleHyperparameters
            ? ReadDoubleArray(reader, maxGroupEntries)
            : ReadFloatArray(reader, maxGroupEntries) is { } singles ? System.Array.ConvertAll(singles, wd => (double)wd) : null;
        checkpoint.Scalars = ReadScalars(reader);
        checkpoint.FixedActiveDomain = hasEnrollmentMetadata ? ReadFixedActiveDomain(reader, expectedParamCount) : null;

        int paramCount = reader.ReadInt32();
        if (paramCount < 0)
            throw new InvalidDataException($"Fused optimizer parameter-state count {paramCount} cannot be negative.");
        if (expectedParamCount is int expectedCount && paramCount != expectedCount)
            throw new InvalidDataException("Optimizer parameter-state count does not match the canonical parameter count.");
        if (checkpoint.FixedActiveDomain is { } domain && domain.Length != paramCount)
            throw new InvalidDataException("The fixed active domain must contain one entry per canonical parameter.");
        if (paramCount > MaxUntrustedParameterCount)
            throw new InvalidDataException($"Fused optimizer parameter-state count {paramCount} exceeds maximum {MaxUntrustedParameterCount}.");
        // Each parameter state writes its age (in v6+) and at least 12 array-length prefixes even when every
        // array is null, so bound the count against both the stream and the untrusted metadata budget before allocating.
        EnsureFitsBudget(
            reader,
            (long)paramCount * ((hasEnrollmentMetadata ? sizeof(long) : 0) + 12 * sizeof(int)),
            MaxUntrustedMetadataBytes,
            "parameter-state array");
        checkpoint.Parameters = new FusedOptimizerParameterCheckpoint[paramCount];
        for (int i = 0; i < paramCount; i++)
            checkpoint.Parameters[i] = ReadParameter(reader, hasEnrollmentMetadata, checkpoint.OptimizerStep, expectedParameterLengths?[i]);
        checkpoint.LbfgsHistory = hasEnrollmentMetadata ? ReadLbfgsHistory(reader, checkpoint, expectedTotalElements) : null;
        ValidateStateMetadata(checkpoint);
        return checkpoint;
    }

    private static void ValidateStateMetadata(FusedOptimizerCheckpoint checkpoint)
    {
        ValidateLbfgsHistory(checkpoint);
        if (checkpoint.OptimizerStep < 0)
            throw new InvalidDataException("The optimizer step cannot be negative.");
        bool requiresDomain = FusedOptimizerCheckpoint.RequiresFixedActiveDomain(checkpoint.OptimizerType);
        if (requiresDomain != (checkpoint.FixedActiveDomain is not null))
            throw new InvalidDataException("A fixed active domain is required exactly for coupled optimizer methods.");
        if (checkpoint.FixedActiveDomain is { } domain && (checkpoint.IsGrouped || domain.Length != checkpoint.Parameters.Length))
            throw new InvalidDataException("The fixed active domain must describe the ungrouped canonical parameter slots.");
        for (int i = 0; i < checkpoint.Parameters.Length; i++)
        {
            long age = checkpoint.Parameters[i].AcceptedUpdates;
            if (age < 0 || age > checkpoint.OptimizerStep)
                throw new InvalidDataException(
                    $"Parameter {i} accepted update count {age} must be between zero and optimizer step {checkpoint.OptimizerStep}.");
        }
    }

    internal static void ValidateLbfgsHistory(FusedOptimizerCheckpoint checkpoint)
    {
        var h = checkpoint.LbfgsHistory;
        if ((checkpoint.OptimizerType == OptimizerType.LBFGS && !checkpoint.IsGrouped) != (h is not null))
            throw new InvalidDataException("L-BFGS history is required exactly for an ungrouped L-BFGS optimizer.");
        if (h is null) return;
        int memory = Math.Max(1, checkpoint.Extras.LbfgsMemorySize);
        if (h.S is null || h.Y is null || h.Rho is null || h.PrevParam is null || h.PrevGrad is null
            || h.S.Length != memory || h.Y.Length != memory || h.Rho.Length != memory
            || h.PrevParam.Length != h.PrevGrad.Length || h.Count < 0 || h.Count > memory
            || h.Head < 0 || h.Head >= memory || (!h.HasPrev && h.Count != 0))
            throw new InvalidDataException("Invalid L-BFGS history dimensions or ring position.");
        ValidateFinite(h.PrevParam); ValidateFinite(h.PrevGrad); ValidateFinite(h.Rho);
        for (int i = 0; i < memory; i++)
        {
            if (h.S[i] is null || h.Y[i] is null || h.S[i].Length != h.PrevParam.Length || h.Y[i].Length != h.PrevParam.Length)
                throw new InvalidDataException("L-BFGS history vectors must have the canonical flattened width.");
            ValidateFinite(h.S[i]); ValidateFinite(h.Y[i]);
        }
        if (float.IsNaN(h.GammaNum) || float.IsInfinity(h.GammaNum)
            || float.IsNaN(h.GammaDen) || float.IsInfinity(h.GammaDen))
            throw new InvalidDataException("L-BFGS history scaling must be finite.");
    }

    private static void ValidateFinite(float[] values)
    {
        foreach (float value in values)
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw new InvalidDataException("L-BFGS history vectors must be finite.");
    }

    private static void WriteLbfgsHistory(BinaryWriter writer, FusedLbfgsHistoryCheckpoint? h)
    {
        writer.Write(h is not null);
        if (h is null) return;
        writer.Write(h.Count); writer.Write(h.Head); writer.Write(h.HasPrev);
        writer.Write(h.GammaNum); writer.Write(h.GammaDen);
        WriteFloatArray(writer, h.PrevParam); WriteFloatArray(writer, h.PrevGrad); WriteFloatArray(writer, h.Rho);
        writer.Write(h.S.Length);
        for (int i = 0; i < h.S.Length; i++) { WriteFloatArray(writer, h.S[i]); WriteFloatArray(writer, h.Y[i]); }
    }

    private static FusedLbfgsHistoryCheckpoint? ReadLbfgsHistory(
        BinaryReader reader,
        FusedOptimizerCheckpoint checkpoint,
        long? expectedTotalElements)
    {
        byte present = reader.ReadByte();
        if (present == 0) return null;
        if (present != 1) throw new InvalidDataException("Invalid L-BFGS history presence flag.");
        if (checkpoint.OptimizerType != OptimizerType.LBFGS || checkpoint.IsGrouped)
            throw new InvalidDataException("L-BFGS history is required exactly for an ungrouped L-BFGS optimizer.");
        int memory = Math.Max(1, checkpoint.Extras.LbfgsMemorySize);
        if (memory > MaxLbfgsMemorySize)
            throw new InvalidDataException($"L-BFGS history memory size {memory} exceeds maximum {MaxLbfgsMemorySize}.");
        var h = new FusedLbfgsHistoryCheckpoint
        {
            Count = reader.ReadInt32(), Head = reader.ReadInt32(),
        };
        byte hasPrev = reader.ReadByte();
        if (hasPrev > 1) throw new InvalidDataException("Invalid L-BFGS previous-iterate flag.");
        h.HasPrev = hasPrev == 1;
        h.GammaNum = reader.ReadSingle(); h.GammaDen = reader.ReadSingle();
        int maxVectorWidth = expectedTotalElements is long total && total <= int.MaxValue
            ? (int)total
            : (int)(MaxUntrustedParameterBufferBytes / sizeof(float));
        h.PrevParam = ReadFloatArray(reader, maxVectorWidth) ?? throw new InvalidDataException("Missing L-BFGS previous parameters.");
        if (expectedTotalElements is long exactTotal && h.PrevParam.LongLength != exactTotal)
            throw new InvalidDataException("L-BFGS history width does not match the plan's canonical parameter layout.");
        h.PrevGrad = ReadFloatArray(reader, h.PrevParam.Length) ?? throw new InvalidDataException("Missing L-BFGS previous gradient.");
        h.Rho = ReadFloatArray(reader, memory) ?? throw new InvalidDataException("Missing L-BFGS curvature reciprocals.");
        int count = reader.ReadInt32();
        if (count < 0) throw new InvalidDataException("Negative L-BFGS history size.");
        if (count != memory)
            throw new InvalidDataException("Invalid L-BFGS history dimensions or ring position.");
        EnsureFitsBudget(reader, (long)count * 2 * sizeof(int), MaxUntrustedMetadataBytes, "L-BFGS history");
        h.S = new float[count][]; h.Y = new float[count][];
        for (int i = 0; i < count; i++)
        {
            h.S[i] = ReadFloatArray(reader, h.PrevParam.Length) ?? throw new InvalidDataException("Missing L-BFGS displacement.");
            h.Y[i] = ReadFloatArray(reader, h.PrevParam.Length) ?? throw new InvalidDataException("Missing L-BFGS gradient difference.");
        }
        return h;
    }

    private static void WriteFixedActiveDomain(BinaryWriter writer, bool[]? domain)
    {
        writer.Write(domain?.Length ?? -1);
        if (domain is not null)
            foreach (bool active in domain) writer.Write((byte)(active ? 1 : 0));
    }

    private static bool[]? ReadFixedActiveDomain(BinaryReader reader, int? expectedParameterCount)
    {
        int count = reader.ReadInt32();
        if (count == -1) return null;
        if (count < 0) throw new InvalidDataException("The fixed active domain count cannot be negative.");
        if (expectedParameterCount is int expected && count != expected)
            throw new InvalidDataException("The fixed active domain must contain one entry per canonical parameter.");
        if (count > MaxUntrustedParameterCount)
            throw new InvalidDataException($"The fixed active domain count {count} exceeds maximum {MaxUntrustedParameterCount}.");
        EnsureFitsBudget(reader, count, MaxUntrustedMetadataBytes, "fixed active domain");
        var domain = new bool[count];
        for (int i = 0; i < count; i++)
        {
            byte active = reader.ReadByte();
            if (active > 1) throw new InvalidDataException("A fixed active domain entry must be zero or one.");
            domain[i] = active == 1;
        }
        return domain;
    }

    private static void WriteExtras(BinaryWriter writer, FusedOptimizerExtras extras)
    {
        writer.Write(extras.Momentum);
        writer.Write(extras.TrustCoefficient);
        writer.Write(extras.L1);
        writer.Write(extras.L2);
        writer.Write(extras.LrPower);
        writer.Write(extras.Lambd);
        writer.Write(extras.Alpha);
        writer.Write(extras.T0);
        writer.Write(extras.RpropEtaPlus);
        writer.Write(extras.RpropEtaMinus);
        writer.Write(extras.RpropStepMin);
        writer.Write(extras.RpropStepMax);
        writer.Write(extras.RpropInitialStep);
        writer.Write(extras.HyperLr);
        writer.Write(extras.D0);
        writer.Write(extras.DGrowthRate);
        writer.Write(extras.SfBeta);
        // Appended in the same PR that introduced them. Both change WHICH ALGORITHM runs rather than only its
        // constants — Nesterov vs classical momentum, and FTRL's per-coordinate learning-rate denominator — so
        // omitting them here would let a checkpoint round trip into a quietly different optimizer.
        writer.Write(extras.Nesterov);
        writer.Write(extras.FtrlBeta);
        // Same reason again: each of these selects behaviour, not just a constant. A checkpoint that
        // omitted them restored L-BFGS with the default memory depth, a trust region with radius 1, and
        // ADMM with rho 1 — a different optimizer from the one that was saved.
        writer.Write(extras.LbfgsMemorySize);
        writer.Write(extras.TrustRegionRadius);
        writer.Write(extras.AdmmRho);
        // LAMB's trust-ratio clip and bias-correction switch select the step LAMB takes, so a round trip that dropped
        // them would restore a different optimizer (format version 7).
        writer.Write(extras.LambMaxTrustRatio);
        writer.Write(extras.LambDisableBiasCorrection);
        // Selects decoupled (AdamW) vs L2 decay for AMSGrad - an algorithm choice, so it must round-trip.
        writer.Write(extras.DecoupledWeightDecay);
        // Format version 9: AMSGrad's bias-correction switch (Reddi et al. 2018, Algorithm 2 when set).
        writer.Write(extras.AmsgradDisableBiasCorrection);
        WriteNullableDouble(writer, extras.AmsgradExactBeta1);
        WriteNullableDouble(writer, extras.AmsgradExactBeta2);
        // Format version 14: SGD momentum whose velocity carries the learning rate - a different rule under a schedule.
        writer.Write(extras.VelocityIncludesLearningRate);
    }

    private static void WriteNullableDouble(BinaryWriter writer, double? value)
    {
        writer.Write(value.HasValue);
        writer.Write(value ?? 0.0);
    }

    private static double? ReadNullableDouble(BinaryReader reader)
    {
        bool present = reader.ReadBoolean();
        double value = reader.ReadDouble();
        return present ? value : null;
    }

    private static FusedOptimizerExtras ReadExtras(BinaryReader reader, bool hasVelocityIncludesLearningRate)
        => new FusedOptimizerExtras
        {
            Momentum = reader.ReadSingle(),
            TrustCoefficient = reader.ReadSingle(),
            L1 = reader.ReadSingle(),
            L2 = reader.ReadSingle(),
            LrPower = reader.ReadSingle(),
            Lambd = reader.ReadSingle(),
            Alpha = reader.ReadSingle(),
            T0 = reader.ReadSingle(),
            RpropEtaPlus = reader.ReadSingle(),
            RpropEtaMinus = reader.ReadSingle(),
            RpropStepMin = reader.ReadSingle(),
            RpropStepMax = reader.ReadSingle(),
            RpropInitialStep = reader.ReadSingle(),
            HyperLr = reader.ReadSingle(),
            D0 = reader.ReadSingle(),
            DGrowthRate = reader.ReadSingle(),
            SfBeta = reader.ReadSingle(),
            Nesterov = reader.ReadBoolean(),
            FtrlBeta = reader.ReadSingle(),
            LbfgsMemorySize = reader.ReadInt32(),
            TrustRegionRadius = reader.ReadSingle(),
            AdmmRho = reader.ReadSingle(),
            LambMaxTrustRatio = reader.ReadSingle(),
            LambDisableBiasCorrection = reader.ReadBoolean(),
            DecoupledWeightDecay = reader.ReadBoolean(),
            AmsgradDisableBiasCorrection = reader.ReadBoolean(),
            AmsgradExactBeta1 = ReadNullableDouble(reader),
            AmsgradExactBeta2 = ReadNullableDouble(reader),
            VelocityIncludesLearningRate = hasVelocityIncludesLearningRate && reader.ReadBoolean(),
        };

    private static void WriteLrSchedules(BinaryWriter writer, FusedLrScheduleCheckpoint[] schedules)
    {
        writer.Write(schedules.Length);
        for (int i = 0; i < schedules.Length; i++)
        {
            writer.Write((int)schedules[i].Kind);
            WriteDoubleArray(writer, schedules[i].Doubles);
            WriteIntArray(writer, schedules[i].Ints);
        }
    }

    private static FusedLrScheduleCheckpoint[] ReadLrSchedules(BinaryReader reader, int? expectedParameterCount)
    {
        int count = reader.ReadInt32();
        if (count < 0)
            throw new InvalidDataException($"Fused optimizer LR schedule count {count} cannot be negative.");
        int maxSchedules = expectedParameterCount is int pCount ? Math.Max(1, pCount) : MaxUntrustedScheduleCount;
        if (count > maxSchedules)
            throw new InvalidDataException($"Fused optimizer LR schedule count {count} exceeds maximum {maxSchedules}.");
        // Each schedule writes kind(4) + a double[] + an int[] (>= 4 bytes each) → >= 12 bytes.
        EnsureFitsBudget(reader, (long)count * (sizeof(int) + 2 * sizeof(int)), MaxUntrustedMetadataBytes, "LR schedule");
        var schedules = new FusedLrScheduleCheckpoint[count];
        for (int i = 0; i < count; i++)
        {
            schedules[i] = new FusedLrScheduleCheckpoint
            {
                Kind = (FusedLrScheduleKind)reader.ReadInt32(),
                Doubles = ReadDoubleArray(reader, MaxUntrustedScheduleElements) ?? System.Array.Empty<double>(),
                Ints = ReadIntArray(reader, MaxUntrustedScheduleElements) ?? System.Array.Empty<int>(),
            };
            // Reject unknown kinds / wrong arity now, so a malformed schedule fails at load rather
            // than throwing IndexOutOfRange deep inside GetLr mid-training.
            schedules[i].Validate();
        }
        return schedules;
    }

    private static void WriteScalars(BinaryWriter writer, FusedOptimizerScalarCheckpoint scalars)
    {
        writer.Write(scalars.HypergradientAdjustment);
        writer.Write(scalars.DAdaptationEstimate);
        writer.Write(scalars.DAdaptationRAccum);
        writer.Write(scalars.ScheduleFreeWeightSum);
        writer.Write(scalars.ConjugateGradientPrevGradNorm2);
    }

    private static FusedOptimizerScalarCheckpoint ReadScalars(BinaryReader reader)
        => new FusedOptimizerScalarCheckpoint
        {
            HypergradientAdjustment = reader.ReadSingle(),
            DAdaptationEstimate = reader.ReadSingle(),
            DAdaptationRAccum = reader.ReadSingle(),
            ScheduleFreeWeightSum = reader.ReadSingle(),
            ConjugateGradientPrevGradNorm2 = reader.ReadSingle(),
        };

    private static void WriteParameter(BinaryWriter writer, FusedOptimizerParameterCheckpoint p)
    {
        writer.Write(p.AcceptedUpdates);
        WriteFloatArray(writer, p.MFloat);
        WriteFloatArray(writer, p.VFloat);
        WriteFloatArray(writer, p.VMaxFloat);
        WriteDoubleArray(writer, p.MDouble);
        WriteDoubleArray(writer, p.VDouble);
        WriteDoubleArray(writer, p.VMaxDouble);
        WriteUShortArray(writer, p.MBFloat16);
        WriteUShortArray(writer, p.VBFloat16);
        WriteByteArray(writer, p.MQuantized);
        WriteByteArray(writer, p.VQuantized);
        WriteDoubleArray(writer, p.MScales);
        WriteDoubleArray(writer, p.VScales);
    }

    private static FusedOptimizerParameterCheckpoint ReadParameter(
        BinaryReader reader,
        bool hasEnrollmentMetadata,
        int defaultAcceptedUpdates,
        int? expectedParameterLength)
    {
        int maxFloatElems = expectedParameterLength ?? (int)(MaxUntrustedParameterBufferBytes / sizeof(float));
        int maxDoubleElems = expectedParameterLength ?? (int)(MaxUntrustedParameterBufferBytes / sizeof(double));
        int maxUShortElems = expectedParameterLength ?? (int)(MaxUntrustedParameterBufferBytes / sizeof(ushort));
        int maxByteElems = expectedParameterLength ?? (int)MaxUntrustedParameterBufferBytes;
        return new FusedOptimizerParameterCheckpoint
        {
            AcceptedUpdates = hasEnrollmentMetadata ? reader.ReadInt64() : defaultAcceptedUpdates,
            MFloat = ReadFloatArray(reader, maxFloatElems),
            VFloat = ReadFloatArray(reader, maxFloatElems),
            VMaxFloat = ReadFloatArray(reader, maxFloatElems),
            MDouble = ReadDoubleArray(reader, maxDoubleElems),
            VDouble = ReadDoubleArray(reader, maxDoubleElems),
            VMaxDouble = ReadDoubleArray(reader, maxDoubleElems),
            MBFloat16 = ReadUShortArray(reader, maxUShortElems),
            VBFloat16 = ReadUShortArray(reader, maxUShortElems),
            MQuantized = ReadByteArray(reader, maxByteElems),
            VQuantized = ReadByteArray(reader, maxByteElems),
            MScales = ReadDoubleArray(reader, maxDoubleElems),
            VScales = ReadDoubleArray(reader, maxDoubleElems),
        };
    }

    private static void WriteFloatArray(BinaryWriter writer, float[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static float[]? ReadFloatArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "float[]", sizeof(float), maxElements);
        if (length < 0) return null;
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = reader.ReadSingle();
        return values;
    }

    private static void WriteDoubleArray(BinaryWriter writer, double[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static double[]? ReadDoubleArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "double[]", sizeof(double), maxElements);
        if (length < 0) return null;
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = reader.ReadDouble();
        return values;
    }

    private static void WriteIntArray(BinaryWriter writer, int[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static int[]? ReadIntArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "int[]", sizeof(int), maxElements);
        if (length < 0) return null;
        var values = new int[length];
        for (int i = 0; i < length; i++) values[i] = reader.ReadInt32();
        return values;
    }

    private static void WriteOptimizerTypeArray(BinaryWriter writer, OptimizerType[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write((int)values[i]);
    }

    private static OptimizerType[]? ReadOptimizerTypeArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "OptimizerType[]", sizeof(int), maxElements);
        if (length < 0) return null;
        var values = new OptimizerType[length];
        for (int i = 0; i < length; i++) values[i] = (OptimizerType)reader.ReadInt32();
        return values;
    }

    private static void WriteUShortArray(BinaryWriter writer, ushort[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static ushort[]? ReadUShortArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "ushort[]", sizeof(ushort), maxElements);
        if (length < 0) return null;
        var values = new ushort[length];
        for (int i = 0; i < length; i++) values[i] = reader.ReadUInt16();
        return values;
    }

    private static void WriteByteArray(BinaryWriter writer, byte[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        writer.Write(values);
    }

    private static byte[]? ReadByteArray(BinaryReader reader, int maxElements)
    {
        int length = ReadArrayLength(reader, "byte[]", 1, maxElements);
        if (length < 0) return null;
        var values = reader.ReadBytes(length);
        if (values.Length != length)
            throw new InvalidDataException(
                $"Fused optimizer byte[] payload was truncated: expected {length} bytes, got {values.Length}.");
        return values;
    }

    private static int ReadArrayLength(BinaryReader reader, string field, int bytesPerElement, int maxElements)
    {
        int length = reader.ReadInt32();
        if (length < -1)
            throw new InvalidDataException($"Fused optimizer {field} length {length} is invalid.");
        if (length > maxElements)
            throw new InvalidDataException(
                $"Fused optimizer {field} length {length} exceeds maximum allowed {maxElements}.");
        if (length > 0)
            EnsureFitsBudget(reader, (long)length * bytesPerElement, MaxUntrustedParameterBufferBytes, field);
        return length;
    }

    /// <summary>
    /// Guards against a corrupt/truncated checkpoint claiming an element count whose payload exceeds
    /// either the remaining seekable stream bytes or the non-seekable allocation budget.
    /// </summary>
    private static void EnsureFitsBudget(BinaryReader reader, long requiredBytes, long nonSeekableBudgetBytes, string field)
    {
        var stream = reader.BaseStream;
        if (!stream.CanSeek)
        {
            if (requiredBytes > nonSeekableBudgetBytes)
                throw new InvalidDataException(
                    $"Fused optimizer {field} claims {requiredBytes} byte(s), exceeding the non-seekable budget of {nonSeekableBudgetBytes} byte(s).");
            return;
        }
        long remaining = stream.Length - stream.Position;
        if (requiredBytes > remaining)
            throw new InvalidDataException(
                $"Fused optimizer {field} claims {requiredBytes} byte(s) but only {remaining} remain " +
                "in the stream — corrupt or truncated checkpoint.");
    }
}
