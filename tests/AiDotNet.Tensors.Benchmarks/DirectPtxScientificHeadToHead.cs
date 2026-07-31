// Copyright (c) AiDotNet. All rights reserved.

using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Compares every experimental scientific direct-PTX route with the shipped CUDA kernel that
/// the same public API calls when the experiment is disabled.
/// </summary>
/// <remarks>
/// The two sides deliberately share one <see cref="CudaBackend"/>. This removes context,
/// stream, allocation, and public-API differences from the comparison: the only switch is the
/// scientific experiment gate. The manifest coverage assertion makes a newly assigned kernel
/// fail closed until it has an incumbent comparison here.
/// </remarks>
internal static class DirectPtxScientificHeadToHead
{
    private const double WinThreshold = 1.10;

    internal static void Run(string[] args)
    {
        string? filter = args.FirstOrDefault(static a => !a.StartsWith("--", StringComparison.Ordinal));
        GpuBenchmarkEnvironment.RequireIdleGpu("direct-ptx-scientific-head-to-head-start");

        bool? previousTestOverride = DirectPtxFeatureGate.TestOverride;
        bool previousExperimentOverride = DirectPtxFeatureGate.ScientificExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        try
        {
            using var backend = new CudaBackend();
            if (!backend.IsAvailable || !backend.IsDirectPtxScientificEnabled)
            {
                Console.WriteLine("Validated NVIDIA direct-PTX scientific backend unavailable.");
                return;
            }

            List<Func<ScientificCase>> factories = Cases(backend);
            AssertCompleteManifestCoverage(factories);
            if (filter is not null)
                factories = factories.Where(f =>
                {
                    using ScientificCase item = f();
                    return item.Api.Contains(filter, StringComparison.OrdinalIgnoreCase);
                }).ToList();

            Console.WriteLine();
            Console.WriteLine("DIRECT PTX SCIENTIFIC - public API incumbent vs experimental route");
            Console.WriteLine("device: {0}", backend.DeviceName);
            Console.WriteLine("gate: same backend/context/stream; adjacent host-timed batches; 5% stability ceiling");
            Console.WriteLine();
            Console.WriteLine("{0,-43} {1,14} {2,14} {3,8}  {4}",
                "operator", "incumbent", "direct PTX", "ratio", "verdict");

            int wins = 0, ties = 0, losses = 0, rejected = 0;
            foreach (Func<ScientificCase> factory in factories)
            {
                using ScientificCase item = factory();
                CaseResult result = Measure(backend, item);
                switch (result.Verdict)
                {
                    case Verdict.Win: wins++; break;
                    case Verdict.Tie: ties++; break;
                    case Verdict.Loss: losses++; break;
                    default: rejected++; break;
                }

                Console.WriteLine("{0,-43} {1,14} {2,14} {3,8}  {4}",
                    item.Label,
                    result.Timing.A.Describe(), result.Timing.B.Describe(),
                    result.Timing.DescribeRatio(), Describe(result));
            }

            Console.WriteLine();
            Console.WriteLine("summary: {0} win, {1} tie, {2} loss, {3} rejected", wins, ties, losses, rejected);
            Console.WriteLine("ratio is incumbent/direct-PTX; a win requires equivalence, stable paired evidence, and >=1.10x.");
            GpuBenchmarkEnvironment.RequireNoForeignCompute("direct-ptx-scientific-head-to-head-end", afterSuite: true);
        }
        finally
        {
            DirectPtxFeatureGate.ScientificExperimentOverride = previousExperimentOverride;
            DirectPtxFeatureGate.TestOverride = previousTestOverride;
        }
    }

    private static CaseResult Measure(CudaBackend backend, ScientificCase item)
    {
        long before = backend.DirectPtxScientificDispatchCount;
        item.Launch(existing: true);
        backend.Synchronize();
        if (backend.DirectPtxScientificDispatchCount != before)
            return CaseResult.Rejected("incumbent unexpectedly dispatched direct PTX");
        double[] expected = item.Snapshot(existing: true);

        item.Launch(existing: false);
        backend.Synchronize();
        if (backend.DirectPtxScientificDispatchCount <= before)
            return CaseResult.Rejected("direct route did not dispatch: " + backend.DirectPtxLastError);
        double[] actual = item.Snapshot(existing: false);
        (double absolute, double relative) = Error(expected, actual);
        if (absolute > item.Tolerance && relative > item.Tolerance)
            return CaseResult.Rejected(string.Format(CultureInfo.InvariantCulture,
                "not equivalent (abs {0:E2}, rel {1:E2})", absolute, relative));

        StableTimer.PairResult timing = StableTimer.MeasureHostPair(
            () => item.Launch(existing: true), backend.Synchronize, item.WorkUnits,
            () => item.Launch(existing: false), backend.Synchronize, item.WorkUnits);
        if (!timing.Stable) return new(timing, Verdict.Rejected, "not measurable");
        Verdict verdict = timing.Ratio >= WinThreshold
            ? Verdict.Win
            : timing.Ratio < 1.0 / WinThreshold ? Verdict.Loss : Verdict.Tie;
        return new(timing, verdict, string.Format(CultureInfo.InvariantCulture,
            "{0}; abs {1:E1}, rel {2:E1}", verdict.ToString().ToUpperInvariant(), absolute, relative));
    }

    private static string Describe(CaseResult result) => result.Detail;

    private static (double Absolute, double Relative) Error(double[] expected, double[] actual)
    {
        if (expected.Length != actual.Length) return (double.PositiveInfinity, double.PositiveInfinity);
        double maxError = 0, maxMagnitude = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!double.IsFinite(expected[i]) || !double.IsFinite(actual[i]))
                return (double.PositiveInfinity, double.PositiveInfinity);
            maxError = Math.Max(maxError, Math.Abs(expected[i] - actual[i]));
            maxMagnitude = Math.Max(maxMagnitude, Math.Abs(expected[i]));
        }
        return (maxError, maxError / Math.Max(maxMagnitude, 1e-6));
    }

    private static List<Func<ScientificCase>> Cases(CudaBackend backend)
    {
        var random = new Random(20260731);
        float[] V(int n, float scale = 1.0f) => Values(random, n, scale);
        float[] P(int n)
        {
            float[] values = new float[n];
            for (int i = 0; i < n; i++) values[i] = 0.01f + (float)random.NextDouble();
            return values;
        }

        const int Count = 1 << 20;
        const int Batch = 1 << 13;
        const int Dim = 64;
        var cases = new List<Func<ScientificCase>>
        {
            () => FloatCase(backend, "CudaBackend.ComplexMultiply", "complex multiply 1M pairs",
                [V(Count * 2), V(Count * 2)], [Count * 2],
                static (b, i, o) => b.ComplexMultiply(i[0], i[1], o[0], Count), 12L * Count),
            () => FloatCase(backend, "CudaBackend.ComplexConjugate", "complex conjugate 1M pairs",
                [V(Count * 2)], [Count * 2],
                static (b, i, o) => b.ComplexConjugate(i[0], o[0], Count), 8L * Count, 0),
            () => FloatCase(backend, "CudaBackend.ComplexMagnitude", "complex magnitude 1M",
                [V(Count), V(Count)], [Count],
                static (b, i, o) => b.ComplexMagnitude(i[0], i[1], o[0], Count), 12L * Count),
            () => FloatCase(backend, "CudaBackend.ComplexPhase", "complex phase 1M",
                [V(Count), V(Count)], [Count],
                static (b, i, o) => b.ComplexPhase(i[0], i[1], o[0], Count), 12L * Count, 5e-3),

            () => FloatCase(backend, "CudaBackend.MobiusAdd", "Mobius add 8192x64",
                [V(Batch * Dim, 0.02f), V(Batch * Dim, 0.02f)], [Batch * Dim],
                static (b, i, o) => b.MobiusAdd(i[0], i[1], o[0], Batch, Dim, 0.5f), 16L * Batch * Dim, 4e-3),
            () => FloatCase(backend, "CudaBackend.PoincareProject", "Poincare project 8192x64",
                [V(Batch * Dim, 0.2f)], [Batch * Dim],
                static (b, i, o) => b.PoincareProject(i[0], o[0], Batch, Dim, 0.5f, 1e-5f), 8L * Batch * Dim, 4e-3),
            () => FloatCase(backend, "CudaBackend.PoincareExpMap", "Poincare exp-map 8192x64",
                [V(Batch * Dim, 0.02f), V(Batch * Dim, 0.005f)], [Batch * Dim],
                static (b, i, o) => b.PoincareExpMap(i[0], i[1], o[0], Batch, Dim, 0.5f), 16L * Batch * Dim, 5e-3),
            () => FloatCase(backend, "CudaBackend.PoincareDistance", "Poincare distance 32768x64",
                [V(32768 * Dim, 0.02f), V(32768 * Dim, 0.02f)], [32768],
                static (b, i, o) => b.PoincareDistance(i[0], i[1], o[0], 32768, Dim, 0.5f), 8L * 32768 * Dim, 5e-3),

            () => FloatCase(backend, "CudaBackend.OctonionMultiply", "octonion multiply 131072",
                [V((Count / 8) * 8), V((Count / 8) * 8)], [(Count / 8) * 8],
                static (b, i, o) => b.OctonionMultiply(i[0], i[1], o[0], Count / 8), 12L * Count, 3e-3),
            () => FloatCase(backend, "CudaBackend.OctonionAdd", "octonion add 131072",
                [V((Count / 8) * 8), V((Count / 8) * 8)], [(Count / 8) * 8],
                static (b, i, o) => b.OctonionAdd(i[0], i[1], o[0], Count / 8), 12L * Count, 0),

            () => FloatCase(backend, "CudaBackend.RbfForward", "RBF 4096x64x64",
                [V(4096 * 64), V(64 * 64), P(64)], [4096 * 64],
                static (b, i, o) => b.RbfForward(i[0], i[1], i[2], o[0], 4096, 64, 64), 4096L * 64 * 64),
            () => FloatCase(backend, "CudaBackend.PairwiseDistance", "pairwise L2 512x512x64",
                [V(512 * 64), V(512 * 64)], [512 * 512],
                static (b, i, o) => b.PairwiseDistance(i[0], i[1], o[0], 512, 512, 64), 512L * 512 * 64),
            () => FloatCase(backend, "CudaBackend.PairwiseDistanceSquared", "pairwise L2-squared 512x512x64",
                [V(512 * 64), V(512 * 64)], [512 * 512],
                static (b, i, o) => b.PairwiseDistanceSquared(i[0], i[1], o[0], 512, 512, 64), 512L * 512 * 64),
            () => FloatCase(backend, "CudaBackend.QuantumMeasurement", "quantum measurement 1M",
                [V(Count), V(Count)], [Count],
                static (b, i, o) => b.QuantumMeasurement(i[0], i[1], o[0], 4096, 256), 12L * Count),
            () => FloatCase(backend, "CudaBackend.ComplexMatVec", "complex matvec 4096x64",
                [V(Dim * Dim), V(Dim * Dim), V(4096 * Dim), V(4096 * Dim)], [4096 * Dim, 4096 * Dim],
                static (b, i, o) => b.ComplexMatVec(i[0], i[1], i[2], i[3], o[0], o[1], 4096, Dim), 4096L * Dim * Dim),

            () => FloatCase(backend, "CudaBackend.SphericalHarmonics", "spherical harmonics 32768x16x4",
                [V(32768 * 16 * 4, 0.2f), V(32768 * 3)], [32768 * 4],
                static (b, i, o) => b.SphericalHarmonics(i[0], i[1], o[0], 32768, 16, 4, 3, 0), 32768L * 16 * 4),
            () => FloatCase(backend, "CudaBackend.SphericalHarmonicsBackward", "spherical harmonics backward 8192x16x4",
                [V(8192 * 16 * 4, 0.2f), V(8192 * 3), V(8192 * 4)], [8192 * 16 * 4],
                static (b, i, o) => b.SphericalHarmonicsBackward(i[0], i[1], i[2], o[0], 8192, 16, 4, 3, 0), 8192L * 16 * 4),

            () => FloatCase(backend, "CudaBackend.CapsulePredictions", "capsule predictions 64x16x16x16x16",
                [V(64 * 16 * 16), V(16 * 16 * 16 * 16)], [64 * 16 * 16 * 16],
                static (b, i, o) => b.CapsulePredictions(i[0], i[1], o[0], 64, 16, 16, 16, 16), 64L * 16 * 16 * 16 * 16),
            () => FloatCase(backend, "CudaBackend.CapsuleTransform", "capsule transform 64x16x16x16x16",
                [V(64 * 16 * 16), V(16 * 16 * 16 * 16)], [64 * 16 * 16 * 16],
                static (b, i, o) => b.CapsuleTransform(i[0], i[1], o[0], 64, 16, 16, 16, 16), 64L * 16 * 16 * 16 * 16),
            () => FloatCase(backend, "CudaBackend.CapsuleWeightedSum", "capsule weighted sum 1024x16x16x16",
                [V(1024 * 16 * 16), V(1024 * 16 * 16 * 16)], [1024 * 16 * 16],
                static (b, i, o) => b.CapsuleWeightedSum(i[0], i[1], o[0], 1024, 16, 16, 16), 1024L * 16 * 16 * 16),
            () => FloatCase(backend, "CudaBackend.CapsuleAgreement", "capsule agreement 1024x16x16x16",
                [V(1024 * 16 * 16 * 16), V(1024 * 16 * 16)], [1024 * 16 * 16],
                static (b, i, o) => b.CapsuleAgreement(i[0], i[1], o[0], 1024, 16, 16, 16), 1024L * 16 * 16 * 16),
            () => FloatCase(backend, "CudaBackend.CosineSimilarity", "cosine similarity 16384x64",
                [V(16384 * 64), V(16384 * 64)], [16384],
                static (b, i, o) => b.CosineSimilarity(i[0], i[1], o[0], 16384, 64), 16384L * 64),
            () => FloatCase(backend, "CudaBackend.SphericalSoftmax", "spherical softmax 16384x64",
                [V(16384 * 64)], [16384 * 64],
                static (b, i, o) => b.SphericalSoftmax(i[0], o[0], 16384, 64), 16384L * 64, 4e-3),
            () => InPlaceCase(backend, "CudaBackend.NormalizeProbabilities", "normalize probabilities 4096x256",
                P(4096 * 256), static (b, x) => b.NormalizeProbabilities(x, 4096, 256), 4096L * 256, 4e-3),
            () => FloatCase(backend, "CudaBackend.MeasurementForward", "measurement forward 4096x256",
                [V(4096 * 256 * 2)], [4096 * 256],
                static (b, i, o) => b.MeasurementForward(i[0], o[0], 4096, 256), 4096L * 256, 4e-3),
            () => FloatCase(backend, "CudaBackend.QuantumRotation", "quantum rotation 8192x1024",
                [V(8192 * 1024), V(8192 * 1024), P(10)], [8192 * 1024, 8192 * 1024],
                static (b, i, o) => b.QuantumRotation(i[0], i[1], o[0], o[1], i[2], 10, 8192), 8192L * 1024 * 10, 5e-3),

            () => FloatCase(backend, "CudaBackend.AnnComputeDistances", "ANN distances 512x512x64",
                [V(512 * 64), V(512 * 64)], [512 * 512],
                static (b, i, o) => b.ComputeDistances(i[0], i[1], o[0], 512, 512, 64, AnnMetric.L2), 512L * 512 * 64),
            () => FloatCase(backend, "CudaBackend.AnnPqDistanceTables", "ANN PQ tables 1024x16x256x8",
                [V(1024 * 16 * 8), V(16 * 256 * 8)], [1024 * 16 * 256],
                static (b, i, o) => b.PqComputeDistanceTables(i[0], i[1], o[0], 1024, 16, 256, 8, AnnMetric.L2), 1024L * 16 * 256 * 8),
            () => IvfCase(backend, V(8192 * 64), V(64 * 64), 8192, 64, 64),
            () => AdcCase(backend, random, 1024, 4096, 16, 256),
            () => FloatCase(backend, "CudaBackend.HashGridEncodeLevel", "hash-grid encode 1M cells",
                [P((Count / 2) * 3), V(1 << 18)], [Count],
                static (b, i, o) => b.HashGridEncodeLevel(i[0], i[1], o[0], Count / 2, 64, 1 << 17, 2, 0, 2), 32L * Count),
            () => FloatCase(backend, "CudaBackend.HashGridEncodeLevelBackward", "hash-grid backward 4096x4096x2",
                [P(4096 * 3), V(4096 * 2)], [4096 * 2],
                static (b, i, o) => b.HashGridEncodeLevelBackward(i[0], i[1], o[0], 4096, 64, 4096, 2, 0, 2), 4096L * 4096),
            () => MeshCase(backend, random, 1024, 256)
        };
        return cases;
    }

    private static ScientificCase FloatCase(
        CudaBackend backend, string api, string label, float[][] inputs, int[] outputSizes,
        Action<CudaBackend, IGpuBuffer[], IGpuBuffer[]> launch, long workUnits, double tolerance = 3e-3)
    {
        IGpuBuffer[] incumbentInputs = inputs.Select(backend.AllocateBuffer).ToArray();
        IGpuBuffer[] directInputs = inputs.Select(backend.AllocateBuffer).ToArray();
        IGpuBuffer[] incumbentOutputs = outputSizes.Select(backend.AllocateBuffer).ToArray();
        IGpuBuffer[] directOutputs = outputSizes.Select(backend.AllocateBuffer).ToArray();
        var resources = incumbentInputs.Concat(directInputs).Concat(incumbentOutputs).Concat(directOutputs).ToArray();
        return new ScientificCase(api, label, workUnits, tolerance,
            () => launch(backend, incumbentInputs, incumbentOutputs),
            () => launch(backend, directInputs, directOutputs),
            () => Snapshot(backend, incumbentOutputs),
            () => Snapshot(backend, directOutputs), resources);
    }

    private static ScientificCase InPlaceCase(
        CudaBackend backend, string api, string label, float[] values,
        Action<CudaBackend, IGpuBuffer> launch, long workUnits, double tolerance)
    {
        IGpuBuffer incumbent = backend.AllocateBuffer(values);
        IGpuBuffer direct = backend.AllocateBuffer(values);
        return new ScientificCase(api, label, workUnits, tolerance,
            () => launch(backend, incumbent), () => launch(backend, direct),
            () => Snapshot(backend, [incumbent]), () => Snapshot(backend, [direct]),
            [incumbent, direct]);
    }

    private static ScientificCase IvfCase(
        CudaBackend backend, float[] vectors, float[] centroids,
        int vectorCount, int centroidCount, int dim)
    {
        IGpuBuffer iv = backend.AllocateBuffer(vectors), dv = backend.AllocateBuffer(vectors);
        IGpuBuffer ic = backend.AllocateBuffer(centroids), dc = backend.AllocateBuffer(centroids);
        IGpuBuffer io = backend.AllocateIntBuffer(vectorCount), directOutput = backend.AllocateIntBuffer(vectorCount);
        double[] Read(IGpuBuffer buffer)
        {
            var values = new int[vectorCount];
            backend.DownloadIntBuffer(buffer, values);
            return values.Select(static x => (double)x).ToArray();
        }
        return new ScientificCase("CudaBackend.AnnIvfAssign", $"ANN IVF assign {vectorCount}x{centroidCount}x{dim}",
            (long)vectorCount * centroidCount * dim, 0,
            () => backend.IvfAssign(iv, ic, io, vectorCount, centroidCount, dim, AnnMetric.L2),
            () => backend.IvfAssign(dv, dc, directOutput, vectorCount, centroidCount, dim, AnnMetric.L2),
            () => Read(io), () => Read(directOutput), [iv, dv, ic, dc, io, directOutput]);
    }

    private static ScientificCase AdcCase(
        CudaBackend backend, Random random, int queries, int codes, int m, int ksub)
    {
        var codeData = new byte[codes * m];
        random.NextBytes(codeData);
        float[] tables = Values(random, queries * m * ksub, 1.0f);
        IGpuBuffer ic = backend.AllocateByteBuffer(codeData.Length), dc = backend.AllocateByteBuffer(codeData.Length);
        backend.UploadByteBuffer(ic, codeData);
        backend.UploadByteBuffer(dc, codeData);
        IGpuBuffer it = backend.AllocateBuffer(tables), dt = backend.AllocateBuffer(tables);
        IGpuBuffer io = backend.AllocateBuffer(queries * codes), directOutput = backend.AllocateBuffer(queries * codes);
        return new ScientificCase("CudaBackend.AnnPqAdcScan", $"ANN PQ ADC {queries}x{codes}x{m}",
            (long)queries * codes * m, 3e-3,
            () => backend.PqAdcScan(ic, it, io, queries, codes, m, ksub),
            () => backend.PqAdcScan(dc, dt, directOutput, queries, codes, m, ksub),
            () => Snapshot(backend, [io]), () => Snapshot(backend, [directOutput]),
            [ic, dc, it, dt, io, directOutput]);
    }

    private static ScientificCase MeshCase(CudaBackend backend, Random random, int faceCount, int vertices)
    {
        var faces = new int[faceCount * 3];
        for (int i = 0; i < faces.Length; i++) faces[i] = random.Next(vertices);
        IGpuBuffer inputA = backend.AllocateIntBuffer(faces), inputB = backend.AllocateIntBuffer(faces);
        IGpuBuffer outputA = backend.AllocateBuffer(vertices * vertices), outputB = backend.AllocateBuffer(vertices * vertices);
        return new ScientificCase("CudaBackend.UniformMeshLaplacian", $"mesh Laplacian {vertices}x{vertices} over {faceCount} faces",
            (long)vertices * vertices * faceCount, 0,
            () => backend.UniformMeshLaplacian(inputA, outputA, faceCount, vertices),
            () => backend.UniformMeshLaplacian(inputB, outputB, faceCount, vertices),
            () => Snapshot(backend, [outputA]), () => Snapshot(backend, [outputB]),
            [inputA, inputB, outputA, outputB]);
    }

    private static double[] Snapshot(CudaBackend backend, IGpuBuffer[] outputs) =>
        outputs.SelectMany(output => backend.DownloadBuffer(output).Select(static x => (double)x)).ToArray();

    private static float[] Values(Random random, int count, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)((random.NextDouble() * 2.0 - 1.0) * scale);
        return values;
    }

    private static void AssertCompleteManifestCoverage(List<Func<ScientificCase>> factories)
    {
        string[] expected = DirectPtxScientificCoverageManifest.All
            .Where(static c => c.Status == DirectPtxScientificCoverageStatus.ExperimentalDirectPtx)
            .Select(static c => c.Api).OrderBy(static x => x, StringComparer.Ordinal).ToArray();
        string[] actual = factories.Select(f =>
        {
            using ScientificCase item = f();
            return item.Api;
        }).OrderBy(static x => x, StringComparer.Ordinal).ToArray();
        if (!expected.SequenceEqual(actual, StringComparer.Ordinal))
            throw new InvalidOperationException("Scientific head-to-head coverage does not match the experimental manifest. " +
                "Missing: " + string.Join(", ", expected.Except(actual, StringComparer.Ordinal)) + "; extra: " +
                string.Join(", ", actual.Except(expected, StringComparer.Ordinal)));
    }

    private enum Verdict { Win, Tie, Loss, Rejected }

    private readonly record struct CaseResult(StableTimer.PairResult Timing, Verdict Verdict, string Detail)
    {
        internal static CaseResult Rejected(string detail) => new(default, Verdict.Rejected, detail);
    }

    private sealed class ScientificCase : IDisposable
    {
        private readonly Action _incumbent;
        private readonly Action _direct;
        private readonly Func<double[]> _snapshotIncumbent;
        private readonly Func<double[]> _snapshotDirect;
        private readonly IDisposable[] _resources;

        internal ScientificCase(
            string api, string label, long workUnits, double tolerance,
            Action incumbent, Action direct,
            Func<double[]> snapshotIncumbent, Func<double[]> snapshotDirect,
            IDisposable[] resources)
        {
            Api = api;
            Label = label;
            WorkUnits = Math.Max(workUnits, 1);
            Tolerance = tolerance;
            _incumbent = incumbent;
            _direct = direct;
            _snapshotIncumbent = snapshotIncumbent;
            _snapshotDirect = snapshotDirect;
            _resources = resources;
        }

        internal string Api { get; }
        internal string Label { get; }
        internal long WorkUnits { get; }
        internal double Tolerance { get; }

        internal void Launch(bool existing)
        {
            DirectPtxFeatureGate.ScientificExperimentOverride = !existing;
            if (existing) _incumbent(); else _direct();
        }

        internal double[] Snapshot(bool existing) => existing ? _snapshotIncumbent() : _snapshotDirect();

        public void Dispose()
        {
            foreach (IDisposable resource in _resources) resource.Dispose();
        }
    }
}
