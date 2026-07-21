using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #834 resident NVIDIA comparison for rectangular MHA/GQA/MQA cells.
/// This is intentionally separate from the historical square-MHA table: a v3
/// specialization cannot inherit a v2 performance result.
/// </summary>
internal static class DirectPtxAttentionFamilyExperiment
{
    private const int Dimension = 64;
    private const float Scale = 0.125f;
    private const int Warmups = 30;
    private const int Samples = 101;

    private readonly record struct Shape(
        string Name, int Batch, int QueryHeads, int KeyValueHeads,
        int QuerySequence, int KeyValueSequence);

    private readonly record struct Distribution(
        double Mean, double Median, double P95, double P99);

    private static readonly Shape[] Shapes =
    [
        new("rect-mha", 2, 8, 8, 32, 64),
        new("rect-gqa", 2, 8, 2, 32, 64),
        new("rect-mqa", 2, 8, 1, 32, 64),
        new("long-gqa", 1, 16, 4, 64, 128),
        new("q-long-gqa", 1, 16, 4, 128, 64)
    ];

    internal static void Run(int independentRuns = 3)
    {
        if (independentRuns <= 0) throw new ArgumentOutOfRangeException(nameof(independentRuns));
        Console.WriteLine(
            $"Direct-PTX v3 attention family: {independentRuns} independent run(s), " +
            $"{Warmups} warmups + {Samples} samples/cell");
        Console.WriteLine(
            $"{"Run",3} {"Shape",-12} {"Mode",-7} {"Method",-24} " +
            $"{"median us",10} {"p95 us",10} {"p99 us",10} {"mean us",10} " +
            $"{"TFLOPS",9} {"B/call",9} {"tmp MiB",9} {"max err",10} " +
            $"{"regs",5} {"shared",7} {"local",5} {"occ",4}");
        Console.WriteLine(new string('-', 159));

        for (int run = 1; run <= independentRuns; run++)
        {
            using var backend = new CudaBackend();
            if (!backend.IsDirectPtxAttentionEnabled)
                throw new InvalidOperationException(
                    "The direct-PTX attention gate and an Ampere NVIDIA GPU are required.");
            if (run == 1)
                Console.WriteLine($"GPU: {backend.DeviceName}; direct PTX enabled={backend.IsDirectPtxAttentionEnabled}");

            foreach (Shape shape in Shapes)
            foreach (bool causal in new[] { false, true })
            {
                // Current SDPA bottom-right alignment yields leading fully-masked
                // rows when Sq > Skv. That requires a separate zero-row PTX path.
                if (causal && shape.QuerySequence > shape.KeyValueSequence) continue;
                RunCell(backend, run, shape, causal);
            }
        }
    }

    private static void RunCell(CudaBackend backend, int run, Shape shape, bool causal)
    {
        int queryElements = checked(
            shape.Batch * shape.QueryHeads * shape.QuerySequence * Dimension);
        int keyValueElements = checked(
            shape.Batch * shape.KeyValueHeads * shape.KeyValueSequence * Dimension);
        int seed = 20260800 + run * 100_000 + queryElements + keyValueElements;
        ushort[] qHost = RandomHalf(new Random(seed + 1), queryElements);
        ushort[] kHost = RandomHalf(new Random(seed + 2), keyValueElements);
        ushort[] vHost = RandomHalf(new Random(seed + 3), keyValueElements);
        float[] qFloatHost = ToFloat(qHost);
        float[] kFloatHost = ToFloat(kHost);
        float[] vFloatHost = ToFloat(vHost);

        using var qFloat = backend.AllocateBuffer(qFloatHost);
        using var kFloat = backend.AllocateBuffer(kFloatHost);
        using var vFloat = backend.AllocateBuffer(vFloatHost);
        using var qHalf = backend.AllocateByteBuffer(queryElements * sizeof(ushort));
        using var kHalf = backend.AllocateByteBuffer(keyValueElements * sizeof(ushort));
        using var vHalf = backend.AllocateByteBuffer(keyValueElements * sizeof(ushort));
        using var directOutput = backend.AllocateBuffer(queryElements);
        using var currentOutput = backend.AllocateBuffer(queryElements);
        backend.ConvertToFp16Native(qFloat, qHalf, queryElements);
        backend.ConvertToFp16Native(kFloat, kHalf, keyValueElements);
        backend.ConvertToFp16Native(vFloat, vHalf, keyValueElements);

        void DirectLaunch()
        {
            if (!backend.TryDirectPtxOnlineAttentionFamily(
                qHalf, kHalf, vHalf, directOutput, null,
                shape.Batch, shape.QueryHeads, shape.KeyValueHeads,
                shape.QuerySequence, shape.KeyValueSequence,
                Scale, causal, emitSoftmaxStats: false,
                causalQueryOffset: causal ? shape.KeyValueSequence - shape.QuerySequence : 0))
                throw new InvalidOperationException(backend.DirectPtxLastError);
        }

        void CurrentLaunch() => backend.ScaledDotProductAttention(
            qFloat, kFloat, vFloat, currentOutput,
            attentionWeights: null, mask: null,
            shape.Batch, shape.QueryHeads, shape.QuerySequence, shape.KeyValueSequence,
            Dimension, Scale, causal, softcap: 0, numKVHeads: shape.KeyValueHeads);

        DirectLaunch();
        CurrentLaunch();
        backend.Synchronize();
        float[] expected = Oracle(qHost, kHost, vHost, shape, causal);
        float directError = MaximumError(backend.DownloadBuffer(directOutput), expected);
        float currentError = MaximumError(backend.DownloadBuffer(currentOutput), expected);

        Distribution directTime = Measure(backend, DirectLaunch);
        long directAllocation = MeasureAllocation(backend, DirectLaunch);
        if (!backend.TryGetDirectPtxAttentionKernelAuditFamily(
            shape.Batch, shape.QueryHeads, shape.KeyValueHeads,
            shape.QuerySequence, shape.KeyValueSequence,
            Scale, causal, fuseLayerNormGelu: false, epsilon: 1e-5f,
            emitSoftmaxStats: false,
            causalQueryOffset: causal ? shape.KeyValueSequence - shape.QuerySequence : 0,
            out DirectPtxKernelAudit audit))
            throw new InvalidOperationException("The measured direct-PTX module has no audit record.");
        Print(run, shape, causal, "Direct PTX v3 resident", directTime,
            Tflops(shape, directTime.Median), directAllocation, 0, directError,
            audit.Function.RegistersPerThread, audit.Function.StaticSharedBytes,
            audit.Function.LocalBytesPerThread, audit.ActiveBlocksPerMultiprocessor);

        Distribution currentTime = Measure(backend, CurrentLaunch);
        long currentAllocation = MeasureAllocation(backend, CurrentLaunch);
        Print(run, shape, causal, "AiDotNet NVRTC current", currentTime,
            Tflops(shape, currentTime.Median), currentAllocation, 0, currentError,
            -1, -1, -1, -1);
    }

    private static Distribution Measure(CudaBackend backend, Action action)
    {
        for (int i = 0; i < Warmups; i++) action();
        backend.Synchronize();
        var samples = new double[Samples];
        double tickToMicroseconds = 1_000_000.0 / Stopwatch.Frequency;
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            backend.Synchronize();
            samples[i] = (Stopwatch.GetTimestamp() - start) * tickToMicroseconds;
        }
        Array.Sort(samples);
        return new Distribution(
            samples.Average(), Percentile(samples, 0.50),
            Percentile(samples, 0.95), Percentile(samples, 0.99));
    }

    private static long MeasureAllocation(CudaBackend backend, Action action)
    {
        for (int i = 0; i < 8; i++) action();
        backend.Synchronize();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < Samples; i++)
        {
            action();
            backend.Synchronize();
        }
        long after = GC.GetAllocatedBytesForCurrentThread();
        return (after - before) / Samples;
    }

    private static float[] Oracle(
        ushort[] q, ushort[] k, ushort[] v, Shape shape, bool causal)
    {
        var output = new float[
            shape.Batch * shape.QueryHeads * shape.QuerySequence * Dimension];
        var scores = new float[shape.KeyValueSequence];
        int queriesPerKeyValue = shape.QueryHeads / shape.KeyValueHeads;
        for (int batch = 0; batch < shape.Batch; batch++)
        for (int queryHead = 0; queryHead < shape.QueryHeads; queryHead++)
        {
            int keyValueHead = queryHead / queriesPerKeyValue;
            int flatQueryHead = batch * shape.QueryHeads + queryHead;
            int flatKeyValueHead = batch * shape.KeyValueHeads + keyValueHead;
            for (int row = 0; row < shape.QuerySequence; row++)
            {
                int lastKey = causal
                    ? Math.Min(row + shape.KeyValueSequence - shape.QuerySequence,
                        shape.KeyValueSequence - 1)
                    : shape.KeyValueSequence - 1;
                int queryBase = (flatQueryHead * shape.QuerySequence + row) * Dimension;
                float maximum = float.NegativeInfinity;
                for (int column = 0; column <= lastKey; column++)
                {
                    int keyBase = (flatKeyValueHead * shape.KeyValueSequence + column) * Dimension;
                    float score = 0;
                    for (int d = 0; d < Dimension; d++)
                        score += Half(q[queryBase + d]) * Half(k[keyBase + d]);
                    scores[column] = score * Scale;
                    maximum = MathF.Max(maximum, scores[column]);
                }
                float sum = 0;
                for (int column = 0; column <= lastKey; column++)
                {
                    scores[column] = MathF.Exp(scores[column] - maximum);
                    sum += scores[column];
                }
                for (int d = 0; d < Dimension; d++)
                {
                    float value = 0;
                    for (int column = 0; column <= lastKey; column++)
                    {
                        int valueBase = (flatKeyValueHead * shape.KeyValueSequence + column) * Dimension;
                        value += scores[column] / sum * Half(v[valueBase + d]);
                    }
                    output[queryBase + d] = value;
                }
            }
        }
        return output;
    }

    private static double Tflops(Shape shape, double microseconds)
    {
        double flops = 4.0 * shape.Batch * shape.QueryHeads *
            shape.QuerySequence * shape.KeyValueSequence * Dimension;
        return flops / (microseconds * 1e-6) / 1e12;
    }

    private static void Print(
        int run, Shape shape, bool causal, string method, Distribution time,
        double tflops, long allocation, long temporaryBytes, float error,
        int registers, int sharedBytes, int localBytes, int occupancy)
    {
        string mode = causal ? "causal" : "plain";
        string regs = registers < 0 ? "n/a" : registers.ToString();
        string shared = sharedBytes < 0 ? "n/a" : sharedBytes.ToString();
        string local = localBytes < 0 ? "n/a" : localBytes.ToString();
        string occ = occupancy < 0 ? "n/a" : occupancy.ToString();
        Console.WriteLine(
            $"{run,3} {shape.Name,-12} {mode,-7} {method,-24} " +
            $"{time.Median,10:F2} {time.P95,10:F2} {time.P99,10:F2} {time.Mean,10:F2} " +
            $"{tflops,9:F3} {allocation,9} {temporaryBytes / 1048576.0,9:F3} {error,10:G4} " +
            $"{regs,5} {shared,7} {local,5} {occ,4}");
    }

    private static double Percentile(double[] sorted, double percentile)
    {
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static float MaximumError(float[] actual, float[] expected)
    {
        float error = 0;
        for (int i = 0; i < expected.Length; i++)
            error = MathF.Max(error, MathF.Abs(actual[i] - expected[i]));
        return error;
    }

    private static ushort[] RandomHalf(Random random, int length)
    {
        var result = new ushort[length];
        for (int i = 0; i < length; i++)
            result[i] = BitConverter.HalfToUInt16Bits(
                (Half)((random.NextSingle() - 0.5f) * 0.5f));
        return result;
    }

    private static float[] ToFloat(ushort[] source)
    {
        var result = new float[source.Length];
        for (int i = 0; i < result.Length; i++) result[i] = Half(source[i]);
        return result;
    }

    private static float Half(ushort bits) =>
        (float)BitConverter.UInt16BitsToHalf(bits);
}
