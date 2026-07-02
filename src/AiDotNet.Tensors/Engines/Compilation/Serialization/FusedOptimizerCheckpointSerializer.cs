using System.IO;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

internal static class FusedOptimizerCheckpointSerializer
{
    internal static void Write(BinaryWriter writer, FusedOptimizerCheckpoint? checkpoint)
    {
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
        writer.Write(checkpoint.MaxGradNorm);
        WriteExtras(writer, checkpoint.Extras);
        WriteLrSchedules(writer, checkpoint.Schedules);
        WriteIntArray(writer, checkpoint.ParamToGroup);
        WriteScalars(writer, checkpoint.Scalars);

        writer.Write(checkpoint.Parameters.Length);
        for (int i = 0; i < checkpoint.Parameters.Length; i++)
            WriteParameter(writer, checkpoint.Parameters[i]);
    }

    internal static FusedOptimizerCheckpoint? Read(BinaryReader reader)
    {
        if (!reader.ReadBoolean()) return null;

        var checkpoint = new FusedOptimizerCheckpoint
        {
            OptimizerType = (OptimizerType)reader.ReadInt32(),
            IsGrouped = reader.ReadBoolean(),
            OptimizerStep = reader.ReadInt32(),
            Beta1 = reader.ReadSingle(),
            Beta2 = reader.ReadSingle(),
            Epsilon = reader.ReadSingle(),
            WeightDecay = reader.ReadSingle(),
            MomentStorageMode = (FusedMomentStorageMode)reader.ReadInt32(),
            Int8MomentBlockSize = reader.ReadInt32(),
            MaxGradNorm = reader.ReadDouble(),
            Extras = ReadExtras(reader),
            Schedules = ReadLrSchedules(reader),
            ParamToGroup = ReadIntArray(reader),
            Scalars = ReadScalars(reader),
        };

        int paramCount = reader.ReadInt32();
        if (paramCount < 0)
            throw new InvalidDataException($"Fused optimizer parameter-state count {paramCount} cannot be negative.");
        checkpoint.Parameters = new FusedOptimizerParameterCheckpoint[paramCount];
        for (int i = 0; i < paramCount; i++)
            checkpoint.Parameters[i] = ReadParameter(reader);
        return checkpoint;
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
    }

    private static FusedOptimizerExtras ReadExtras(BinaryReader reader)
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

    private static FusedLrScheduleCheckpoint[] ReadLrSchedules(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        if (count < 0)
            throw new InvalidDataException($"Fused optimizer LR schedule count {count} cannot be negative.");
        var schedules = new FusedLrScheduleCheckpoint[count];
        for (int i = 0; i < count; i++)
        {
            schedules[i] = new FusedLrScheduleCheckpoint
            {
                Kind = (FusedLrScheduleKind)reader.ReadInt32(),
                Doubles = ReadDoubleArray(reader) ?? System.Array.Empty<double>(),
                Ints = ReadIntArray(reader) ?? System.Array.Empty<int>(),
            };
        }
        return schedules;
    }

    private static void WriteScalars(BinaryWriter writer, FusedOptimizerScalarCheckpoint scalars)
    {
        writer.Write(scalars.HypergradientAdjustment);
        writer.Write(scalars.DAdaptationEstimate);
        writer.Write(scalars.DAdaptationRAccum);
        writer.Write(scalars.ScheduleFreeWeightSum);
    }

    private static FusedOptimizerScalarCheckpoint ReadScalars(BinaryReader reader)
        => new FusedOptimizerScalarCheckpoint
        {
            HypergradientAdjustment = reader.ReadSingle(),
            DAdaptationEstimate = reader.ReadSingle(),
            DAdaptationRAccum = reader.ReadSingle(),
            ScheduleFreeWeightSum = reader.ReadSingle(),
        };

    private static void WriteParameter(BinaryWriter writer, FusedOptimizerParameterCheckpoint p)
    {
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

    private static FusedOptimizerParameterCheckpoint ReadParameter(BinaryReader reader)
        => new FusedOptimizerParameterCheckpoint
        {
            MFloat = ReadFloatArray(reader),
            VFloat = ReadFloatArray(reader),
            VMaxFloat = ReadFloatArray(reader),
            MDouble = ReadDoubleArray(reader),
            VDouble = ReadDoubleArray(reader),
            VMaxDouble = ReadDoubleArray(reader),
            MBFloat16 = ReadUShortArray(reader),
            VBFloat16 = ReadUShortArray(reader),
            MQuantized = ReadByteArray(reader),
            VQuantized = ReadByteArray(reader),
            MScales = ReadDoubleArray(reader),
            VScales = ReadDoubleArray(reader),
        };

    private static void WriteFloatArray(BinaryWriter writer, float[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static float[]? ReadFloatArray(BinaryReader reader)
    {
        int length = ReadArrayLength(reader, "float[]");
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

    private static double[]? ReadDoubleArray(BinaryReader reader)
    {
        int length = ReadArrayLength(reader, "double[]");
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

    private static int[]? ReadIntArray(BinaryReader reader)
    {
        int length = ReadArrayLength(reader, "int[]");
        if (length < 0) return null;
        var values = new int[length];
        for (int i = 0; i < length; i++) values[i] = reader.ReadInt32();
        return values;
    }

    private static void WriteUShortArray(BinaryWriter writer, ushort[]? values)
    {
        if (values is null) { writer.Write(-1); return; }
        writer.Write(values.Length);
        for (int i = 0; i < values.Length; i++) writer.Write(values[i]);
    }

    private static ushort[]? ReadUShortArray(BinaryReader reader)
    {
        int length = ReadArrayLength(reader, "ushort[]");
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

    private static byte[]? ReadByteArray(BinaryReader reader)
    {
        int length = ReadArrayLength(reader, "byte[]");
        if (length < 0) return null;
        var values = reader.ReadBytes(length);
        if (values.Length != length)
            throw new InvalidDataException(
                $"Fused optimizer byte[] payload was truncated: expected {length} bytes, got {values.Length}.");
        return values;
    }

    private static int ReadArrayLength(BinaryReader reader, string field)
    {
        int length = reader.ReadInt32();
        if (length < -1)
            throw new InvalidDataException($"Fused optimizer {field} length {length} is invalid.");
        return length;
    }
}
