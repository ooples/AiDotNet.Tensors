using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Evolves the generated FP32 GEMM (<see cref="PtxSgemmGenerator"/>) for one problem shape on one CUDA device with
/// AiDotNet.Evolution (#1263, phase C of #377), through the first-party tuning scaffold: every candidate is compiled,
/// checked against an fp64 oracle, then timed on CUDA events; finalists are replayed paired against the incumbent and
/// only a winner that clears the promotion policy is deployed or persisted.
/// </summary>
internal static class PtxGemmEvolution
{
    internal static readonly KernelId EvolutionKernelId = new("gemm", "cuda-ptx-typed-evolution");
    internal static readonly KernelSearchSpaceVersion SearchSpaceVersion = new(2);
    internal static readonly KernelBenchmarkProtocolVersion BenchmarkProtocolVersion = new(1);

    /// <summary>
    /// Hand-picked starting points, from small to large tiles, kept only where they admit the problem. The first is the
    /// shape's default (<see cref="DefaultFor"/>): the incumbent every evolved winner must beat in the paired replay, so
    /// it has to be a sensible kernel for this shape, not an arbitrary one.
    /// </summary>
    internal static IReadOnlyList<PtxGemmConfiguration> Seeds(PtxGemmLayout layout, int m, int n, int k, int multiprocessors = 68,
        int ccMajor = 8)
    {
        var candidates = new[]
        {
            // SM 8.0+: cp.async for the M/N-contiguous operand, register transposition for a K-contiguous one, split
            // fragments, grouped rasterization, two blocks per multiprocessor, register-pipelined fragments (the best
            // strict-FP32 kernel: 3-4% faster than without the fragment pipeline at 1024^3 to 4096^3 on an RTX 3080).
            new PtxGemmConfiguration(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2, 2, true, FragmentPipeline: true),
            new PtxGemmConfiguration(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2, 2, true),
            new PtxGemmConfiguration(128, 64, 16, 8, 4, 4, 4, false, true, 8, 2, 2, true, FragmentPipeline: true),
            new PtxGemmConfiguration(128, 64, 16, 8, 4, 4, 4, false, true, 8, 2, 2, true),
            new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 4, false, false, 8, 2, 3, true),
            // Any device: register staging.
            new PtxGemmConfiguration(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2),
            new PtxGemmConfiguration(128, 128, 8, 8, 8, 4, 4, true),
            new PtxGemmConfiguration(128, 64, 16, 8, 4, 4, 4, true),
            new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 4, true),
            new PtxGemmConfiguration(64, 128, 16, 4, 8, 4, 0, true),
            new PtxGemmConfiguration(32, 32, 16, 4, 4, 4, 0, true),
            new PtxGemmConfiguration(128, 128, 8, 8, 8, 1, 4, true),
            new PtxGemmConfiguration(64, 64, 16, 4, 4, 1, 4, true),
            new PtxGemmConfiguration(32, 32, 8, 4, 4, 1, 0, false),
        };
        var seeds = new List<PtxGemmConfiguration>();
        foreach (var candidate in candidates)
            if ((!candidate.IsAsync || ccMajor >= 8) && Admits(candidate, layout, m, n, k)) seeds.Add(candidate);
        if (seeds.Count == 0) throw new InvalidOperationException($"No GEMM seed admits M={m} N={n} K={k} ({layout.Name}).");
        PtxGemmConfiguration defaultSeed = DefaultFor(seeds, m, n, multiprocessors);
        seeds.Remove(defaultSeed);
        seeds.Insert(0, defaultSeed);
        return seeds;
    }

    /// <summary>
    /// The largest-tile admitted seed that still launches blocks for at least half the multiprocessors (a 128 x 128 tile
    /// gives a 128^3 problem a single block); with no such seed, the one launching the most blocks. Vector access breaks
    /// ties, then seed order. Half, not all: a larger tile does less redundant loading, and on an RTX 3080 (68 SMs) the
    /// 128 x 128 seed beat the 128 x 64 one at 768^3 to 1024^3 with 36 to 64 blocks (90 vs 103 us at 768^3, 119 vs
    /// 135 us at 1024^3), while at 640^3 (25 blocks) it lost (74 vs 47 us).
    /// </summary>
    internal static PtxGemmConfiguration DefaultFor(IReadOnlyList<PtxGemmConfiguration> admitted, int m, int n, int multiprocessors)
    {
        static long Blocks(PtxGemmConfiguration c, int m, int n) =>
            (long)((m + c.BlockM - 1) / c.BlockM) * ((n + c.BlockN - 1) / c.BlockN);
        PtxGemmConfiguration? best = null;
        foreach (var c in admitted)
        {
            if (2 * Blocks(c, m, n) < multiprocessors) continue;
            if (best is not { } b || c.BlockM * c.BlockN > b.BlockM * b.BlockN ||
                (c.BlockM * c.BlockN == b.BlockM * b.BlockN && c.VectorWidth > b.VectorWidth))
                best = c;
        }
        if (best is { } chosen) return chosen;
        PtxGemmConfiguration most = admitted[0];
        foreach (var c in admitted)
            if (Blocks(c, m, n) > Blocks(most, m, n) ||
                (Blocks(c, m, n) == Blocks(most, m, n) && c.VectorWidth > most.VectorWidth))
                most = c;
        return most;
    }

    internal static int MultiprocessorCount(DirectPtxRuntime runtime)
    {
        using var _ = runtime.Enter();
        if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return 68;
        return CuBlasNative.cuDeviceGetAttribute(out int count, (int)CudaDeviceAttribute.MultiprocessorCount, device) ==
               CudaResult.Success && count > 0 ? count : 68;
    }

    internal static bool Admits(PtxGemmConfiguration configuration, PtxGemmLayout layout, int m, int n, int k) =>
        PtxSgemmGenerator.Admits(configuration, layout, m, n, k,
            layout.TransposeA ? m : k, layout.TransposeB ? k : n, n, 0, 0, 0);

    internal static KernelTuningDeviceFingerprint Fingerprint(DirectPtxRuntime runtime)
    {
        string model = EvolutionHash.Combine(new[]
        {
            "cuda-model-v1", runtime.DeviceName,
            runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture),
            runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture),
        });
        string local = EvolutionHash.Combine(new[] { "cuda-device-v1", model, runtime.DeviceFingerprint });
        return new KernelTuningDeviceFingerprint(KernelTuningDeviceKind.NvidiaGpu, local, model);
    }

    /// <summary>Runs a fixed-budget evolutionary search for one shape and layout and returns the deployment record.</summary>
    internal static async Task<EvolutionKernelTuningResult<PtxGemmConfiguration>> TuneAsync(
        DirectPtxRuntime runtime, PtxGemmLayout layout, int m, int n, int k,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<PtxGemmConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<PtxGemmConfiguration>? store = null,
        Action<PtxGemmConfiguration, KernelTuningTrialResult>? onTrial = null,
        CancellationToken cancellationToken = default)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));

        IReadOnlyList<PtxGemmConfiguration> seeds = Seeds(layout, m, n, k, MultiprocessorCount(runtime), runtime.ComputeCapabilityMajor);
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
        using var backend = new ExperimentBackend<PtxGemmConfiguration>(runtime, layout, m, n, k,
            configuration =>
            {
                var kernel = new PtxSgemmKernel(runtime, configuration, layout);
                return new LoadedGemm(kernel, kernel.Audit,
                    (a, b, c) => kernel.Launch(a, b, c, m, n, k, lda, ldb, n, 1f, 0f));
            },
            configuration => configuration.Name);
        var experiment = new KernelTuningExperiment<PtxGemmConfiguration>(
            backend,
            new DeviceEventKernelTuningTimer(new CudaEventClock(runtime)),
            seeds[0],
            new KernelTuningWorkload(checked(2d * m * n * k), KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution);
        var identity = new KernelTuningIdentity(
            EvolutionKernelId,
            new ShapeProfile(m, n, k, (layout.TransposeA ? 1 : 0) | (layout.TransposeB ? 2 : 0)),
            Fingerprint(runtime),
            KernelTuningBackend.Cuda,
            SearchSpaceVersion,
            BenchmarkProtocolVersion);
        var tuner = new EvolutionKernelAutotuner<PtxGemmConfiguration>(
            identity,
            new Codec(),
            new Variation(layout, m, n, k, runtime.ComputeCapabilityMajor),
            async (configuration, context, token) =>
            {
                KernelTuningTrialResult result = await EvaluateAsync(configuration, context, token).ConfigureAwait(false);
                onTrial?.Invoke(configuration, result);
                return result;
            },
            experiment,
            engineOptions,
            tuningOptions,
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration =>
                configuration.Invalidity() is null && Admits(configuration, layout, m, n, k));
        async ValueTask<KernelTuningTrialResult> EvaluateAsync(
            PtxGemmConfiguration configuration, EvolutionEvaluationContext context, CancellationToken token)
        {
            string? invalid = configuration.Invalidity();
            if (invalid is not null)
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration, invalid);
            if (!Admits(configuration, layout, m, n, k))
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration,
                    "The configuration does not admit this problem's alignment.");
            // Compile here so a driver or resource-budget failure is reported as such, not as a benchmark failure.
            string? compileError = backend.TryCompile(configuration);
            if (compileError is not null)
                return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.CompilationFailed, compileError);
            return await experiment.EvaluateAsync(configuration, context, token).ConfigureAwait(false);
        }

        if (seeds.Count > tuner.MaximumProposals)
        {
            var trimmed = new List<PtxGemmConfiguration>();
            for (int i = 0; i < tuner.MaximumProposals; i++) trimmed.Add(seeds[i]);
            seeds = trimmed;
        }
        return await tuner.TuneAsync(seeds, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>CUDA events on the runtime's stream as the tuning scaffold's device clock.</summary>
    internal sealed class CudaEventClock : IKernelTuningDeviceClock
    {
        private readonly DirectPtxRuntime _runtime;

        internal CudaEventClock(DirectPtxRuntime runtime) => _runtime = runtime;

        // The scaffold awaits between calls, so a call may run on a thread-pool thread with no current CUDA context:
        // every driver call below enters the runtime's context first.
        public IKernelTuningDeviceTimestamp Record()
        {
            using var _ = _runtime.Enter();
            Check(CudaNativeBindings.cuEventCreate(out IntPtr handle, CudaNativeBindings.CU_EVENT_DEFAULT), "cuEventCreate");
            try
            {
                Check(CudaNativeBindings.cuEventRecord(handle, _runtime.Stream), "cuEventRecord");
            }
            catch
            {
                CudaNativeBindings.cuEventDestroy(handle);
                throw;
            }
            return new Timestamp(_runtime, handle);
        }

        public TimeSpan GetElapsed(IKernelTuningDeviceTimestamp start, IKernelTuningDeviceTimestamp end)
        {
            var first = start as Timestamp ?? throw new ArgumentException("Not a CUDA event timestamp.", nameof(start));
            var last = end as Timestamp ?? throw new ArgumentException("Not a CUDA event timestamp.", nameof(end));
            using var _ = _runtime.Enter();
            Check(CudaNativeBindings.cuEventElapsedTime(out float milliseconds, first.Handle, last.Handle), "cuEventElapsedTime");
            return TimeSpan.FromTicks((long)Math.Round(milliseconds * TimeSpan.TicksPerMillisecond));
        }

        private static void Check(CudaResult result, string call)
        {
            if (result != CudaResult.Success)
                throw new InvalidOperationException($"{call} failed: {result}.");
        }

        private sealed class Timestamp : IKernelTuningDeviceTimestamp
        {
            private readonly DirectPtxRuntime _runtime;

            internal Timestamp(DirectPtxRuntime runtime, IntPtr handle)
            {
                _runtime = runtime;
                Handle = handle;
            }

            internal IntPtr Handle { get; private set; }

            public ValueTask WaitAsync(CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var _ = _runtime.Enter();
                Check(CudaNativeBindings.cuEventSynchronize(Handle), "cuEventSynchronize");
                return default;
            }

            public void Dispose()
            {
                if (Handle == IntPtr.Zero) return;
                using (_runtime.Enter()) CudaNativeBindings.cuEventDestroy(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }

    /// <summary>A compiled candidate: what disposes it, its audit, and a launch of the experiment's problem.</summary>
    internal readonly record struct LoadedGemm(IDisposable Owner, DirectPtxKernelAudit Audit, Action<IntPtr, IntPtr, IntPtr> Launch);

    /// <summary>
    /// Owns the problem's device operands, the compiled candidates and the fp64 oracle, for any generated GEMM family
    /// (CUDA-core FP32, TF32 or FP16 Tensor Core). Operands are multiples of 1/256 in [-63/256, 63/256], exact in FP32,
    /// TF32 and FP16, so every family sees exactly the oracle's inputs; FP16 operands are uploaded as halves.
    /// </summary>
    internal sealed class ExperimentBackend<TConfiguration> : IKernelTuningExperimentBackend<TConfiguration>, IDisposable
        where TConfiguration : notnull
    {
        // FP32 accumulation over K terms in a different order than the fp64 oracle: relative to the output scale.
        private const double RelativeTolerance = 1e-4;
        private const int MaximumOracleOutputs = 4096;
        private readonly DirectPtxRuntime _runtime;
        private readonly int _m, _n;
        private readonly Func<TConfiguration, LoadedGemm> _compile;
        private readonly Func<TConfiguration, string> _name;
        private readonly DirectPtxBuffer _a, _b, _c;
        private readonly int[] _sampleIndices;
        private readonly double[] _expected;
        private readonly double _scale;
        private readonly Dictionary<TConfiguration, LoadedGemm> _kernels = new();
        private readonly Dictionary<TConfiguration, TimeSpan> _compileTimes = new();
        private readonly Dictionary<TConfiguration, string> _compileErrors = new();
        private bool _disposed;

        internal ExperimentBackend(DirectPtxRuntime runtime, PtxGemmLayout layout, int m, int n, int k,
            Func<TConfiguration, LoadedGemm> compile, Func<TConfiguration, string> name, bool halfOperands = false)
        {
            _runtime = runtime;
            _m = m;
            _n = n;
            _compile = compile;
            _name = name;
            int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
            float[] a = Values(checked(m * k), 17), b = Values(checked(k * n), 43);
            int operandBytes = halfOperands ? 2 : sizeof(float);
            _a = runtime.AllocateBytes((nuint)((long)a.Length * operandBytes));
            _b = runtime.AllocateBytes((nuint)((long)b.Length * operandBytes));
            _c = runtime.AllocateBytes((nuint)((long)m * n * sizeof(float)));
            Upload(_a, a, halfOperands);
            Upload(_b, b, halfOperands);

            // An fp64 oracle over every output for small problems, over a fixed sample (corners included) otherwise,
            // so the reference costs bounded host time at 4096^3.
            int outputs = checked(m * n);
            int samples = Math.Min(outputs, MaximumOracleOutputs);
            _sampleIndices = new int[samples];
            if (samples == outputs)
            {
                for (int i = 0; i < samples; i++) _sampleIndices[i] = i;
            }
            else
            {
                var random = new Random(1263);
                _sampleIndices[0] = 0;
                _sampleIndices[1] = n - 1;
                _sampleIndices[2] = (m - 1) * n;
                _sampleIndices[3] = outputs - 1;
                for (int i = 4; i < samples; i++) _sampleIndices[i] = random.Next(outputs);
            }
            _expected = new double[samples];
            double scale = 1e-30;
            for (int s = 0; s < samples; s++)
            {
                int row = _sampleIndices[s] / n, column = _sampleIndices[s] % n;
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    double av = layout.TransposeA ? a[p * lda + row] : a[row * lda + p];
                    double bv = layout.TransposeB ? b[column * ldb + p] : b[p * ldb + column];
                    sum += av * bv;
                }
                _expected[s] = sum;
                scale = Math.Max(scale, Math.Abs(sum));
            }
            _scale = scale;
        }

        // The values are q / 256 for an integer |q| <= 63 (see Values), so the FP16 encoding is exact and computed from q in
        // integers: with p the position of q's leading bit, the exponent is p - 8 and the fraction q's remaining bits
        // shifted to 10. No Half type, which netstandard2.0 and net462 lack, and no floating-point logarithm.
        private static ushort HalfBits(float value)
        {
            int q = (int)Math.Round(value * 256f);
            if (q == 0) return 0;
            if (Math.Abs(q) > 63 || q / 256f != value) throw new ArgumentOutOfRangeException(nameof(value));
            ushort sign = (ushort)(q < 0 ? 0x8000 : 0);
            int magnitude = Math.Abs(q), leading = 0;
            while ((magnitude >> (leading + 1)) != 0) leading++;
            int fraction = (magnitude - (1 << leading)) << (10 - leading);
            return (ushort)(sign | ((leading - 8 + 15) << 10) | fraction);
        }

        private static void Upload(DirectPtxBuffer buffer, float[] values, bool half)
        {
            if (!half)
            {
                buffer.Upload<float>(values);
                return;
            }
            var bits = new ushort[values.Length];
            for (int i = 0; i < values.Length; i++) bits[i] = HalfBits(values[i]);
            buffer.Upload<ushort>(bits);
        }

        /// <summary>Compiles the candidate once; returns the driver's or the budget's error, or null.</summary>
        internal string? TryCompile(TConfiguration configuration)
        {
            if (_kernels.ContainsKey(configuration)) return null;
            if (_compileErrors.TryGetValue(configuration, out string? known)) return known;
            var watch = Stopwatch.StartNew();
            try
            {
                _kernels[configuration] = _compile(configuration);
                _compileTimes[configuration] = watch.Elapsed;
                return null;
            }
            catch (Exception exception) when (exception is InvalidOperationException or ArgumentException or
                                                 PlatformNotSupportedException or InvalidDataException)
            {
                _compileErrors[configuration] = exception.Message;
                return exception.Message;
            }
        }

        public ValueTask PrepareAsync(TConfiguration configuration, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            string? error = TryCompile(configuration);
            if (error is not null) throw new InvalidOperationException("Compilation failed: " + error);
            return default;
        }

        public ValueTask ExecuteAsync(TConfiguration configuration, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            if (!_kernels.TryGetValue(configuration, out LoadedGemm kernel))
                throw new InvalidOperationException("The candidate was not prepared.");
            kernel.Launch(_a.Pointer, _b.Pointer, _c.Pointer);
            return default;
        }

        public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            _runtime.Synchronize();
            return default;
        }

        public async ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
            TConfiguration configuration, CancellationToken cancellationToken = default)
        {
            await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
            await SynchronizeAsync(cancellationToken).ConfigureAwait(false);
            var actual = new float[checked(_m * _n)];
            _c.Download<float>(actual);
            double maximumAbsolute = 0;
            for (int s = 0; s < _sampleIndices.Length; s++)
                maximumAbsolute = Math.Max(maximumAbsolute, Math.Abs(actual[_sampleIndices[s]] - _expected[s]));
            double relative = maximumAbsolute / _scale;
            double absoluteTolerance = RelativeTolerance * _scale;
            if (!(relative <= RelativeTolerance))
                throw new KernelTuningValidationException(KernelTuningTrialStatus.OutputMismatch,
                    $"PTX GEMM {_name(configuration)} exceeded tolerance: max abs {maximumAbsolute:R}, relative {relative:R}.");
            return new KernelTuningCorrectnessEvidence(KernelTuningValidationScope.Output,
                maximumAbsolute, relative, absoluteTolerance, RelativeTolerance);
        }

        public KernelTuningResourceUsage GetResourceUsage(TConfiguration configuration)
        {
            ThrowIfDisposed();
            if (!_kernels.TryGetValue(configuration, out LoadedGemm kernel))
                throw new InvalidOperationException("The candidate was not prepared.");
            DirectPtxKernelAudit audit = kernel.Audit;
            double occupancy = (double)audit.ActiveBlocksPerMultiprocessor * audit.BlockThreads /
                Math.Max(1, _runtime.MaxThreadsPerMultiprocessor);
            return new KernelTuningResourceUsage(
                KernelTuningResourceMetric<long>.Measured(0),
                KernelTuningResourceMetric<double>.Measured(Math.Min(1, occupancy)),
                KernelTuningResourceMetric<int>.Measured(audit.Function.RegistersPerThread),
                KernelTuningResourceMetric<TimeSpan>.Measured(_compileTimes[configuration]),
                KernelTuningResourceMetric<int>.Measured(1));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (LoadedGemm kernel in _kernels.Values) kernel.Owner.Dispose();
            _kernels.Clear();
            _c.Dispose();
            _b.Dispose();
            _a.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ExperimentBackend<TConfiguration>));
        }

        private static float[] Values(int length, int salt)
        {
            var values = new float[length];
            for (int i = 0; i < values.Length; i++) values[i] = (((i * 37 + salt * 17) % 127) - 63) / 256f;
            return values;
        }
    }

    /// <summary>The genome codec persisted records use (deployments and CUDA GEMM routes).</summary>
    internal static IEvolutionGenomeCodec<PtxGemmConfiguration> GenomeCodec { get; } = new Codec();

    private sealed class Codec : IEvolutionGenomeCodec<PtxGemmConfiguration>
    {
        public string Id => "cuda-ptx-gemm-typed";
        public string VersionHash => "3";

        public string Serialize(PtxGemmConfiguration genome) => string.Join("|", new[]
        {
            F(genome.BlockM), F(genome.BlockN), F(genome.BlockK), F(genome.ThreadM), F(genome.ThreadN),
            F(genome.VectorWidth), F(genome.SharedPad), genome.DoubleBuffer ? "1" : "0", genome.SplitTile ? "1" : "0",
            F(genome.Swizzle), F(genome.MinBlocksPerSM), F(genome.AsyncStages), genome.RegisterTranspose ? "1" : "0",
            genome.FragmentPipeline ? "1" : "0",
        });

        public PtxGemmConfiguration Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] v = payload.Split('|');
            // Eight fields is the version-1 genome (no fragment split, plain rasterization, no occupancy floor, register
            // staging); thirteen the version-2 genome (no fragment pipeline).
            if (v.Length is not (8 or 13 or 14)) throw new InvalidDataException("Invalid PTX GEMM genome field count.");
            bool v2 = v.Length >= 13, v3 = v.Length == 14;
            var genome = new PtxGemmConfiguration(P(v[0]), P(v[1]), P(v[2]), P(v[3]), P(v[4]), P(v[5]), P(v[6]), B(v[7]),
                v2 && B(v[8]), v2 ? P(v[9]) : 1, v2 ? P(v[10]) : 1, v2 ? P(v[11]) : 0, v2 && B(v[12]), v3 && B(v[13]));
            if (genome.Invalidity() is { } reason) throw new InvalidDataException("Invalid PTX GEMM genome: " + reason);
            return genome;
        }

        private static string F(int value) => value.ToString(CultureInfo.InvariantCulture);

        private static bool B(string value) =>
            value switch { "0" => false, "1" => true, _ => throw new InvalidDataException("Invalid Boolean.") };

        private static int P(string value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("Invalid integer in the PTX GEMM genome.");
    }

    /// <summary>One to three mutations of a parent (or an inspiration), repaired until the child admits the problem.</summary>
    private sealed class Variation : IVariationOperator<PtxGemmConfiguration>
    {
        private static readonly int[] Tiles = { 32, 64, 128, 256 };
        private static readonly int[] KTiles = { 8, 16, 32 };
        private static readonly int[] ThreadTiles = { 4, 8, 16 };
        private readonly PtxGemmLayout _layout;
        private readonly int _m, _n, _k, _ccMajor;

        internal Variation(PtxGemmLayout layout, int m, int n, int k, int ccMajor)
        {
            _ccMajor = ccMajor;
            _layout = layout;
            _m = m;
            _n = n;
            _k = k;
        }

        public string Id => "cuda-ptx-gemm-constrained-variation";
        public string VersionHash => "3";

        public ValueTask<PtxGemmConfiguration> ProposeAsync(
            EvolutionVariationContext<PtxGemmConfiguration> context, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            PtxGemmConfiguration parent = context.Inspirations.Count > 0 && context.Random.NextDouble() < 0.3d
                ? context.Inspirations[context.Random.NextInt(context.Inspirations.Count)].Candidate.CanonicalGenome.Genome
                : context.Parent.Candidate.CanonicalGenome.Genome;
            for (int attempt = 0; attempt < 64; attempt++)
            {
                PtxGemmConfiguration child = parent;
                int mutations = 1 + context.Random.NextInt(3);
                for (int i = 0; i < mutations; i++) child = Mutate(child, context.Random);
                if (child != parent && child.Invalidity() is null && (!child.IsAsync || _ccMajor >= 8) &&
                    Admits(child, _layout, _m, _n, _k))
                    return new ValueTask<PtxGemmConfiguration>(child);
            }
            return new ValueTask<PtxGemmConfiguration>(parent);
        }

        private static PtxGemmConfiguration Mutate(PtxGemmConfiguration value, StableRandom random) =>
            random.NextInt(14) switch
            {
                0 => value with { BlockM = Pick(Tiles, random) },
                1 => value with { BlockN = Pick(Tiles, random) },
                2 => value with { BlockK = Pick(KTiles, random) },
                3 => value with { ThreadM = Pick(ThreadTiles, random) },
                4 => value with { ThreadN = Pick(ThreadTiles, random) },
                5 => value with { VectorWidth = value.VectorWidth == 4 ? 1 : 4 },
                6 => value with { SharedPad = value.SharedPad == 0 ? 4 : 0 },
                7 => value with { DoubleBuffer = !value.DoubleBuffer },
                8 => value with { SplitTile = !value.SplitTile },
                9 => value with { Swizzle = Pick(PtxGemmConfiguration.Swizzles, random) },
                10 => value with { MinBlocksPerSM = value.MinBlocksPerSM == 1 ? 2 : 1 },
                // Switching to cp.async drops register double buffering (the pipeline replaces it).
                11 => value with { AsyncStages = Pick(PtxGemmConfiguration.AsyncStageCounts, random), DoubleBuffer = false },
                12 => value with { RegisterTranspose = !value.RegisterTranspose },
                13 => value with { FragmentPipeline = !value.FragmentPipeline },
                _ => throw new InvalidOperationException(),
            };

        private static int Pick(int[] values, StableRandom random) => values[random.NextInt(values.Length)];
    }
}
