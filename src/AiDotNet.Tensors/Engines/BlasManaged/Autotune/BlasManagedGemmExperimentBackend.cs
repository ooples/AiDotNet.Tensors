using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// First-party managed-GEMM experiment backend. Inputs are deterministic, candidate execution uses
/// the public typed plan controls, and correctness is checked against an independent scalar oracle.
/// </summary>
internal sealed class BlasManagedGemmExperimentBackend<T> :
    IKernelTuningExperimentBackend<BlasManagedGemmConfiguration>
    where T : unmanaged
{
    private readonly int _m;
    private readonly int _n;
    private readonly int _k;
    private readonly int _lda;
    private readonly int _ldb;
    private readonly bool _transA;
    private readonly bool _transB;
    private readonly bool _deterministic;
    private readonly T[] _a;
    private readonly T[] _b;
    private readonly T[] _output;
    private readonly double[] _reference;

    internal BlasManagedGemmExperimentBackend(
        int m,
        int n,
        int k,
        bool transA,
        bool transB,
        bool deterministic,
        int inputSeed)
    {
        _m = m;
        _n = n;
        _k = k;
        _transA = transA;
        _transB = transB;
        _deterministic = deterministic;
        _lda = transA ? m : k;
        _ldb = transB ? k : n;
        _a = new T[(transA ? k : m) * _lda];
        _b = new T[(transB ? n : k) * _ldb];
        _output = new T[m * n];
        _reference = new double[m * n];
        FillDeterministically(_a, unchecked((uint)inputSeed));
        FillDeterministically(_b, unchecked((uint)inputSeed ^ 0x9e3779b9u));
        ComputeReference();
    }

    public ValueTask PrepareAsync(
        BlasManagedGemmConfiguration configuration,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return default;
    }

    public ValueTask ExecuteAsync(
        BlasManagedGemmConfiguration configuration,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        BlasOptions<T> options = CreateOptions(configuration);
        BlasManaged.Gemm<T>(
            _a, _lda, _transA,
            _b, _ldb, _transB,
            _output, _n,
            _m, _n, _k,
            in options);
        return default;
    }

    public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return default;
    }

    public async ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
        BlasManagedGemmConfiguration configuration,
        CancellationToken cancellationToken = default)
    {
        await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
        double maximumAbsoluteError = 0d;
        double maximumRelativeError = 0d;
        for (int i = 0; i < _reference.Length; i++)
        {
            double actual = ToDouble(_output[i]);
            double expected = _reference[i];
            double absoluteError = Math.Abs(actual - expected);
            double relativeError = absoluteError / Math.Max(Math.Abs(expected), 1e-30d);
            maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.Max(maximumRelativeError, relativeError);
        }

        double absoluteTolerance = typeof(T) == typeof(float)
            ? 5e-5d * Math.Max(1d, _k / 64d)
            : 1e-12d * Math.Max(1d, _k / 64d);
        double relativeTolerance = typeof(T) == typeof(float) ? 5e-4d : 1e-11d;
        if (maximumAbsoluteError > absoluteTolerance && maximumRelativeError > relativeTolerance)
        {
            throw new KernelTuningValidationException(
                KernelTuningTrialStatus.OutputMismatch,
                "Managed GEMM output differs from the independent scalar oracle.");
        }
        return new KernelTuningCorrectnessEvidence(
            KernelTuningValidationScope.Output,
            maximumAbsoluteError,
            maximumRelativeError,
            absoluteTolerance,
            relativeTolerance);
    }

    public KernelTuningResourceUsage GetResourceUsage(BlasManagedGemmConfiguration configuration)
    {
        KernelTuningResourceMetric<long> workspace = configuration.PackingMode switch
        {
            PackingMode.Auto => KernelTuningResourceMetric<long>.Unavailable(),
            PackingMode.ForceStreaming => KernelTuningResourceMetric<long>.Measured(0),
            PackingMode.ForcePackAOnly => KernelTuningResourceMetric<long>.Measured(
                (long)configuration.Mc * configuration.Kc * Unsafe.SizeOf<T>()),
            PackingMode.ForcePackBoth => KernelTuningResourceMetric<long>.Measured(
                ((long)configuration.Mc * configuration.Kc +
                 (long)configuration.Kc * configuration.Nc) * Unsafe.SizeOf<T>()),
            _ => KernelTuningResourceMetric<long>.Unavailable()
        };
        return new KernelTuningResourceUsage(
            workspace,
            KernelTuningResourceMetric<double>.NotApplicable(),
            KernelTuningResourceMetric<int>.NotApplicable(),
            KernelTuningResourceMetric<TimeSpan>.Measured(TimeSpan.Zero),
            KernelTuningResourceMetric<int>.Measured(1));
    }

    private BlasOptions<T> CreateOptions(BlasManagedGemmConfiguration configuration) => new()
    {
        PackingMode = configuration.PackingMode,
        NumThreads = configuration.ThreadCount,
        ParallelismAxis = configuration.PackingMode == PackingMode.Auto
            ? null
            : configuration.ParallelismAxis,
        Mc = configuration.Mc,
        Nc = configuration.Nc,
        Kc = configuration.Kc,
        Mode = _deterministic ? BlasMode.Deterministic : BlasMode.Fast,
        BetaZero = false,
    };

    private void ComputeReference()
    {
        for (int i = 0; i < _m; i++)
        {
            for (int j = 0; j < _n; j++)
            {
                double sum = 0d;
                for (int p = 0; p < _k; p++)
                    sum += ToDouble(ReadA(i, p)) * ToDouble(ReadB(p, j));
                _reference[i * _n + j] = sum;
            }
        }
    }

    private T ReadA(int row, int column) =>
        _transA ? _a[column * _lda + row] : _a[row * _lda + column];

    private T ReadB(int row, int column) =>
        _transB ? _b[column * _ldb + row] : _b[row * _ldb + column];

    private static void FillDeterministically(T[] values, uint state)
    {
        for (int i = 0; i < values.Length; i++)
        {
            state = unchecked(state * 1664525u + 1013904223u);
            double value = ((state >> 8) / 16777216d - 0.5d) * 0.25d;
            values[i] = FromDouble(value);
        }
    }

    private static double ToDouble(T value)
    {
        if (typeof(T) == typeof(float)) return (float)(object)value;
        if (typeof(T) == typeof(double)) return (double)(object)value;
        throw new NotSupportedException($"Managed GEMM experiments do not support T={typeof(T).Name}.");
    }

    private static T FromDouble(double value)
    {
        if (typeof(T) == typeof(float)) return (T)(object)(float)value;
        if (typeof(T) == typeof(double)) return (T)(object)value;
        throw new NotSupportedException($"Managed GEMM experiments do not support T={typeof(T).Name}.");
    }
}
