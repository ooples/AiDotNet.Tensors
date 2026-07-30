namespace AiDotNet.Tensors.Engines.Compilation;

internal enum FusedLrScheduleKind
{
    Constant = 1,
    NoamScaled = 2,
    Cosine = 3,
    OneCycleResolved = 4,
    Exponential = 5,
    Step = 6,
    Cyclic = 7,
    LinearWarmupCosine = 8,
    LinearWarmupDecay = 9
}

internal sealed class FusedLrScheduleCheckpoint
{
    public FusedLrScheduleKind Kind { get; set; }
    public double[] Doubles { get; set; } = System.Array.Empty<double>();
    public int[] Ints { get; set; } = System.Array.Empty<int>();

    public FusedLrScheduleCheckpoint()
    {
    }

    public FusedLrScheduleCheckpoint(FusedLrScheduleKind kind, double[] doubles, int[] ints)
    {
        Kind = kind;
        Doubles = doubles;
        Ints = ints;
    }

    public LrSchedule ToSchedule()
    {
        Validate();
        return new RestoredLrSchedule(Kind, Doubles, Ints);
    }

    /// <summary>
    /// Rejects an unknown <see cref="Kind"/> or a payload with too few Doubles/Ints for that kind,
    /// so a malformed checkpoint fails loudly at restore instead of throwing IndexOutOfRange deep
    /// inside <c>GetLr</c> mid-training (or silently restoring an invalid schedule).
    /// </summary>
    public void Validate()
    {
        (int reqD, int reqI) = Kind switch
        {
            FusedLrScheduleKind.Constant => (1, 0),
            FusedLrScheduleKind.NoamScaled => (2, 0),
            FusedLrScheduleKind.Cosine => (2, 1),
            FusedLrScheduleKind.OneCycleResolved => (3, 2),
            FusedLrScheduleKind.Exponential => (2, 0),
            FusedLrScheduleKind.Step => (2, 1),
            FusedLrScheduleKind.Cyclic => (2, 1),
            FusedLrScheduleKind.LinearWarmupCosine => (2, 2),
            FusedLrScheduleKind.LinearWarmupDecay => (3, 3),
            _ => throw new System.IO.InvalidDataException(
                $"Unknown serialized LR schedule kind {(int)Kind}."),
        };
        if (Doubles.Length < reqD || Ints.Length < reqI)
            throw new System.IO.InvalidDataException(
                $"Serialized LR schedule kind {Kind} requires at least {reqD} double(s) and {reqI} int(s); " +
                $"got {Doubles.Length} double(s) and {Ints.Length} int(s).");
    }

    private sealed class RestoredLrSchedule : LrSchedule
    {
        private readonly FusedLrScheduleKind _kind;
        private readonly double[] _d;
        private readonly int[] _i;

        public RestoredLrSchedule(FusedLrScheduleKind kind, double[] doubles, int[] ints)
        {
            _kind = kind;
            _d = doubles;
            _i = ints;
        }

        public override double GetLr(int step)
        {
            int s = step < 1 ? 1 : step;
            return _kind switch
            {
                FusedLrScheduleKind.Constant => _d[0],
                FusedLrScheduleKind.NoamScaled => Noam(s),
                FusedLrScheduleKind.Cosine => Cosine(s),
                FusedLrScheduleKind.OneCycleResolved => OneCycle(s),
                FusedLrScheduleKind.Exponential => _d[0] * System.Math.Pow(_d[1], System.Math.Max(0, s - 1)),
                FusedLrScheduleKind.Step => _d[0] * System.Math.Pow(_d[1], System.Math.Max(0, s - 1) / _i[0]),
                FusedLrScheduleKind.Cyclic => Cyclic(s),
                FusedLrScheduleKind.LinearWarmupCosine => LinearWarmupCosine(s),
                FusedLrScheduleKind.LinearWarmupDecay => LinearWarmupDecay(s),
                _ => throw new NotSupportedException($"Unsupported serialized LR schedule kind '{_kind}'."),
            };
        }

        // Faithful restore of LinearWarmupLr.GetLr — must stay bit-identical to that eager formula
        // (FusedLrScheduleMappingTests). _d = [lrMax, warmupInitLr, endLr]; _i = [warmupSteps,
        // totalSteps(normalized), (int)WarmupDecayMode].
        private double LinearWarmupDecay(int step)
        {
            double lrMax = _d[0], warmupInit = _d[1], endLr = _d[2];
            int warmup = _i[0];
            if (step <= 1) return warmup > 0 ? warmupInit : lrMax;
            double raw = LinearWarmupRaw(step - 1, lrMax, warmupInit, endLr, warmup, _i[1], (WarmupDecayMode)_i[2]);
            return raw > endLr ? raw : endLr;
        }

        private static double LinearWarmupRaw(
            int t, double lrMax, double warmupInit, double endLr, int warmupSteps, int totalSteps, WarmupDecayMode mode)
        {
            if (t < warmupSteps)
            {
                if (warmupSteps == 0) return lrMax;
                return warmupInit + (lrMax - warmupInit) * ((double)t / warmupSteps);
            }
            if (mode == WarmupDecayMode.Constant) return lrMax;
            int decaySteps = totalSteps - warmupSteps;
            int decayStep = t - warmupSteps;
            if (decayStep >= decaySteps) return endLr;
            double progress = (double)decayStep / decaySteps;
            if (mode == WarmupDecayMode.Linear)
                return lrMax - (lrMax - endLr) * progress;
            double cos = (1.0 + System.Math.Cos(System.Math.PI * progress)) / 2.0;
            return endLr + (lrMax - endLr) * cos;
        }

        internal override FusedLrScheduleCheckpoint? TryCaptureCheckpoint()
            => new FusedLrScheduleCheckpoint(
                _kind,
                (double[])_d.Clone(),
                (int[])_i.Clone());

        private double Noam(int step)
        {
            double invSqrtBranch = _d[0] / System.Math.Sqrt(step);
            double warmupBranch = _d[1] * step;
            return System.Math.Min(invSqrtBranch, warmupBranch);
        }

        private double Cosine(int step)
        {
            int total = _i[0];
            int s = step > total ? total : step;
            double progress = (s - 1) / (double)System.Math.Max(total - 1, 1);
            double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
            return _d[1] + (_d[0] - _d[1]) * cos;
        }

        private double OneCycle(int step)
        {
            int total = _i[0];
            int warmupEnd = _i[1];
            int s = step > total ? total : step;
            if (s <= warmupEnd)
            {
                double t = (s - 1) / (double)System.Math.Max(warmupEnd - 1, 1);
                return _d[1] + (_d[0] - _d[1]) * t;
            }

            double progress = (s - warmupEnd) / (double)System.Math.Max(total - warmupEnd, 1);
            double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
            return _d[2] + (_d[0] - _d[2]) * cos;
        }

        private double Cyclic(int step)
        {
            int s = System.Math.Max(0, step - 1);
            int period = 2 * _i[0];
            int phase = s % period;
            double t = phase < _i[0]
                ? phase / (double)_i[0]
                : 1.0 - (phase - _i[0]) / (double)_i[0];
            return _d[0] + (_d[1] - _d[0]) * t;
        }

        private double LinearWarmupCosine(int step)
        {
            int warmup = _i[0];
            int total = _i[1];
            int s = step > total ? total : step;
            if (warmup > 0 && s <= warmup)
                return _d[0] * s / (double)warmup;

            double progress = warmup == 0
                ? (s - 1) / (double)System.Math.Max(total - 1, 1)
                : (s - warmup) / (double)System.Math.Max(total - warmup, 1);
            double cos = 0.5 * (1.0 + System.Math.Cos(System.Math.PI * progress));
            return _d[1] + (_d[0] - _d[1]) * cos;
        }
    }
}

internal sealed class FusedOptimizerCheckpoint
{
    public OptimizerType OptimizerType { get; set; }
    public bool IsGrouped { get; set; }
    public int OptimizerStep { get; set; }
    public float Beta1 { get; set; }
    public float Beta2 { get; set; }
    public float Epsilon { get; set; }
    public float WeightDecay { get; set; }
    public FusedMomentStorageMode MomentStorageMode { get; set; }
    public int Int8MomentBlockSize { get; set; }
    public double MaxGradNorm { get; set; }
    public FusedOptimizerExtras Extras { get; set; } = new FusedOptimizerExtras();
    public FusedLrScheduleCheckpoint[] Schedules { get; set; } = System.Array.Empty<FusedLrScheduleCheckpoint>();
    public int[]? ParamToGroup { get; set; }
    public FusedOptimizerScalarCheckpoint Scalars { get; set; } = new FusedOptimizerScalarCheckpoint();
    public FusedOptimizerParameterCheckpoint[] Parameters { get; set; } = System.Array.Empty<FusedOptimizerParameterCheckpoint>();
}

internal sealed class FusedOptimizerScalarCheckpoint
{
    public float HypergradientAdjustment { get; set; }
    public float DAdaptationEstimate { get; set; }
    public float DAdaptationRAccum { get; set; }
    public float ScheduleFreeWeightSum { get; set; }
}

internal sealed class FusedOptimizerParameterCheckpoint
{
    public float[]? MFloat { get; set; }
    public float[]? VFloat { get; set; }
    public float[]? VMaxFloat { get; set; }
    public double[]? MDouble { get; set; }
    public double[]? VDouble { get; set; }
    public double[]? VMaxDouble { get; set; }
    public ushort[]? MBFloat16 { get; set; }
    public ushort[]? VBFloat16 { get; set; }
    public byte[]? MQuantized { get; set; }
    public byte[]? VQuantized { get; set; }
    public double[]? MScales { get; set; }
    public double[]? VScales { get; set; }
}
