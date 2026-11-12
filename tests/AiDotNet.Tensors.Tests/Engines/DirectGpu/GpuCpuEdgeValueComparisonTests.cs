using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Comparisons, selections, min/max (element-wise and reduced), arg-extrema, sign, abs, clamp and the IEEE
/// classifiers must give the CPU engine's answer on every float, whatever math flags the GPU kernels are built with.
/// GPU fast math flushes subnormals and assumes no NaN/infinity, which made <c>TensorGreaterThan(t, 0)</c> return 0
/// for a resident 1.4e-45 on OpenCL while the CPU returned 1. The inputs here are GPU-resident tensors of the edge
/// values (signed zeros, the smallest and largest subnormals, the smallest normal, the largest finite value,
/// infinities and NaN) in every pairing, and each result must match the CPU engine bit for bit (any NaN matches NaN).
/// Run per backend with <c>AIDOTNET_DIRECTGPU_BACKENDS</c> (opencl, cuda, ...).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuCpuEdgeValueComparisonTests
{
    private static readonly float[] Edges =
    {
        0f,
        -0f,
        float.Epsilon,                                   // smallest subnormal
        -float.Epsilon,
        FromBits(0x007FFFFF),      // largest subnormal
        FromBits(unchecked((int)0x807FFFFF)),
        FromBits(0x00800000),      // smallest normal
        FromBits(unchecked((int)0x80800000)),
        1f,
        -1f,
        float.MaxValue,
        float.MinValue,
        float.PositiveInfinity,
        float.NegativeInfinity,
        float.NaN,
    };

    private static int N => Edges.Length;

    // BitConverter.Int32BitsToSingle / SingleToInt32Bits are not on .NET Framework.
    private static float FromBits(int bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
    private static int ToBits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    /// <summary>Every (a, b) pairing of the edge values: a varies fastest.</summary>
    private static (float[] A, float[] B) Pairs()
    {
        var a = new float[N * N];
        var b = new float[N * N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                a[i * N + j] = Edges[j];
                b[i * N + j] = Edges[i];
            }
        return (a, b);
    }

    /// <summary>Every (x, lo, hi) triple of the edge values: x varies fastest.</summary>
    private static (float[] X, float[] Lo, float[] Hi) Triples()
    {
        var x = new float[N * N * N];
        var lo = new float[N * N * N];
        var hi = new float[N * N * N];
        int k = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int m = 0; m < N; m++, k++)
                {
                    x[k] = Edges[m];
                    lo[k] = Edges[j];
                    hi[k] = Edges[i];
                }
        return (x, lo, hi);
    }

    /// <summary>
    /// Max-pool inputs: one channel per (a, b) pairing, laid out [1, N*N, 2, 4] with every row a, b, b, a, so a 2x2
    /// window holds a, b, a, b and the next one b, a, b, a (both orders of each pair, ties included).
    /// </summary>
    private static float[] PoolPlanes(float[] pa, float[] pb)
    {
        var planes = new float[N * N * 8];
        for (int c = 0; c < N * N; c++)
            for (int row = 0; row < 2; row++)
            {
                int o = c * 8 + row * 4;
                planes[o] = pa[c]; planes[o + 1] = pb[c]; planes[o + 2] = pb[c]; planes[o + 3] = pa[c];
            }
        return planes;
    }

    // An op runs its engine calls and returns a reader of its results, called only after the run, so that reading the
    // results back is not counted as a transfer of the op.
    private delegate Func<float[]> Op(IEngine engine, Func<float[], int[], Tensor<float>> make);

    private static Func<float[]> F(Tensor<float> t) => () => t.ToArray();
    private static Func<float[]> F(Tensor<Bit> t) => () => t.ToArray().Select(v => (bool)v ? 1f : 0f).ToArray();
    private static Func<float[]> F(Tensor<int> t) => () => t.ToArray().Select(v => (float)v).ToArray();

    // The readers of several results, run in order: the list is built eagerly, so every engine call happens now.
    private static Func<float[]> All(IEnumerable<Func<float[]>> readers)
    {
        var list = readers.ToList();
        return () => list.SelectMany(read => read()).ToArray();
    }

    private static IEnumerable<(string Name, Op Run, bool Resident)> Ops()
    {
        var (pa, pb) = Pairs();
        int[] pairShape = { N * N };
        int[] edgeShape = { N };

        // Tensor-tensor binary forms over every pairing.
        (string, Func<IEngine, Tensor<float>, Tensor<float>, Func<float[]>>)[] binary =
        {
            ("GreaterThan", (e, a, b) => F(e.TensorGreaterThan(a, b))),
            ("LessThan", (e, a, b) => F(e.TensorLessThan(a, b))),
            ("GreaterEqual", (e, a, b) => F(e.TensorGreaterEqual(a, b))),
            ("LessEqual", (e, a, b) => F(e.TensorLessEqual(a, b))),
            ("Equals", (e, a, b) => F(e.TensorEquals(a, b))),
            ("NotEquals", (e, a, b) => F(e.TensorNotEquals(a, b))),
            ("Eq", (e, a, b) => F(e.TensorEq(a, b))),
            ("Max", (e, a, b) => F(e.TensorMax(a, b))),
            ("Min", (e, a, b) => F(e.TensorMin(a, b))),
            ("Fmax", (e, a, b) => F(e.TensorFmax(a, b))),
            ("Fmin", (e, a, b) => F(e.TensorFmin(a, b))),
        };
        foreach (var (name, f) in binary)
            yield return (name, (e, make) => f(e, make(pa, pairShape), make(pb, pairShape)), true);

        // Where with a float condition: every edge value as the condition (true unless ±0, NaN included), selecting
        // between two edge-valued branches that must pass through bit for bit.
        var negatedB = pb.Select(v => -v).ToArray();
        yield return ("WhereCondition", (e, make) =>
            F(e.TensorWhere(make(pa, pairShape), make(pb, pairShape), make(negatedB, pairShape))), true);

        // Tensor-scalar forms: every edge tensor against every edge scalar.
        (string, Func<IEngine, Tensor<float>, float, Func<float[]>>)[] scalar =
        {
            ("GreaterThanScalar", (e, t, s) => F(e.TensorGreaterThan(t, s))),
            ("LessThanScalar", (e, t, s) => F(e.TensorLessThan(t, s))),
            ("EqualsScalar", (e, t, s) => F(e.TensorEquals(t, s))),
            ("NotEqualsScalar", (e, t, s) => F(e.TensorNotEquals(t, s))),
            ("EqScalar", (e, t, s) => F(e.TensorEqScalar(t, s))),
            ("MaxScalar", (e, t, s) => F(e.TensorMax(t, s))),
            ("MinScalar", (e, t, s) => F(e.TensorMin(t, s))),
            ("ClampMin", (e, t, s) => F(e.TensorClampMin(t, s))),
            ("ClampMax", (e, t, s) => F(e.TensorClampMax(t, s))),
        };
        foreach (var (name, f) in scalar)
            yield return (name, (e, make) =>
            {
                var t = make(Edges, edgeShape);
                return All(Edges.Select(s => f(e, t, s)));
            }, true);

        // Clamp(t, lo, hi) for every ordered, non-NaN bound pair, and the tensor-bound clamp over every pairing.
        yield return ("Clamp", (e, make) =>
        {
            var t = make(Edges, edgeShape);
            var bounds = Edges.Where(v => !float.IsNaN(v)).ToArray();
            return All(from lo in bounds from hi in bounds where lo <= hi select F(e.TensorClamp(t, lo, hi)));
        }, true);
        yield return ("ClampTensorMin", (e, make) => F(e.TensorClampTensor(make(pa, pairShape), make(pb, pairShape), null)), true);
        yield return ("ClampTensorMax", (e, make) => F(e.TensorClampTensor(make(pa, pairShape), null, make(pb, pairShape))), true);

        // Clamp with any scalar bounds: NaN bounds (ignored on the CPU) and inverted bounds (the upper bound wins).
        yield return ("ClampAnyBounds", (e, make) =>
        {
            var t = make(Edges, edgeShape);
            return All(from lo in Edges from hi in Edges select F(e.TensorClamp(t, lo, hi)));
        }, true);
        yield return ("Clip", (e, make) =>
        {
            var t = make(Edges, edgeShape);
            var bounds = Edges.Where(v => !float.IsNaN(v)).ToArray();
            return All(from lo in bounds from hi in bounds where lo <= hi select F(e.TensorClip(t, lo, hi)));
        }, true);

        // Tensor clamp with both bounds over every (x, lo, hi) triple of edge values.
        var (tx, tlo, thi) = Triples();
        int[] tripleFlatShape = { N * N * N };
        yield return ("ClampTensorBoth", (e, make) =>
            F(e.TensorClampTensor(make(tx, tripleFlatShape), make(tlo, tripleFlatShape), make(thi, tripleFlatShape))), true);
        yield return ("ReLU6", (e, make) => F(e.TensorReLU6(make(Edges, edgeShape))), true);

        // Unary forms.
        (string, Func<IEngine, Tensor<float>, Func<float[]>>)[] unary =
        {
            ("Abs", (e, t) => F(e.TensorAbs(t))),
            ("Negate", (e, t) => F(e.TensorNegate(t))),
            ("Sign", (e, t) => F(e.TensorSign(t))),
            ("IsNan", (e, t) => F(e.TensorIsNan(t))),
            ("IsInf", (e, t) => F(e.TensorIsInf(t))),
            ("IsFinite", (e, t) => F(e.TensorIsFinite(t))),
            ("NanToNum", (e, t) => F(e.TensorNanToNum(t))),
            ("NanToNumValues", (e, t) => F(e.TensorNanToNum(t, 7.0, 1.5, -2.5))),
        };
        foreach (var (name, f) in unary)
            yield return (name, (e, make) => f(e, make(Edges, edgeShape)), true);

        // Selection with a mask: the selected edge values must pass through bit for bit.
        yield return ("WhereBit", (e, make) =>
        {
            var mask = new Tensor<Bit>(Enumerable.Range(0, N * N).Select(i => (Bit)(i % 3 != 0)).ToArray(), pairShape);
            return F(e.TensorWhere(mask, make(pa, pairShape), make(pb, pairShape)));
        }, true);

        // Row reductions over every pair, and over every triple ending in each edge (a NaN or infinity mid-row).
        int[] rowShape = { N * N, 2 };
        var rows = new float[N * N * 2];
        for (int i = 0; i < N * N; i++) { rows[2 * i] = pa[i]; rows[2 * i + 1] = pb[i]; }
        var triples = new float[N * N * 3];
        for (int i = 0; i < N * N; i++) { triples[3 * i] = pa[i]; triples[3 * i + 1] = pb[i]; triples[3 * i + 2] = 1f; }
        int[] tripleShape = { N * N, 3 };
        (string, Func<IEngine, Tensor<float>, Func<float[]>>)[] rowOps =
        {
            ("ReduceMax", (e, t) => F(e.ReduceMax(t, new[] { 1 }, false))),
            ("ArgMax", (e, t) => F(e.TensorArgMax(t, 1))),
            ("ArgMin", (e, t) => F(e.TensorArgMin(t, 1))),
            ("Amin", (e, t) => F(e.TensorAmin(t, new[] { 1 }, false))),
        };
        foreach (var (name, f) in rowOps)
        {
            yield return (name + "Pairs", (e, make) => f(e, make(rows, rowShape)), true);
            yield return (name + "Triples", (e, make) => f(e, make(triples, tripleShape)), true);
        }

        // Running max / min along the last axis (pairs and triples) and along a leading axis.
        var columns = pa.Concat(pb).ToArray();
        int[] columnShape = { 2, N * N };
        (string, Func<IEngine, Tensor<float>, int, Func<float[]>>)[] scans =
        {
            ("CumMax", (e, t, axis) => F(e.TensorCumMax(t, axis))),
            ("CumMin", (e, t, axis) => F(e.TensorCumMin(t, axis))),
        };
        foreach (var (name, f) in scans)
        {
            yield return (name + "Pairs", (e, make) => f(e, make(rows, rowShape), 1), true);
            yield return (name + "Triples", (e, make) => f(e, make(triples, tripleShape), 1), true);
            yield return (name + "Axis0", (e, make) => f(e, make(columns, columnShape), 0), true);
        }

        // Max pooling: every window holds one (a, b) pairing in both orders.
        var planes = PoolPlanes(pa, pb);
        int[] planeShape = { 1, N * N, 2, 4 };
        (string, Func<IEngine, Func<int[], Tensor<float>>, Func<float[]>>)[] pools =
        {
            ("MaxPool2D", (e, p) => F(e.MaxPool2D(p(planeShape), 2))),
            ("MaxPool2DPadded", (e, p) => F(e.MaxPool2D(p(planeShape), 3, 1, 1))),
            ("TensorMaxPool2D", (e, p) => F(e.TensorMaxPool2D(p(planeShape), 2, 2, 0))),
            ("MaxPool2DWithIndices", (e, p) =>
            {
                var values = e.MaxPool2DWithTensorIndices(p(planeShape), new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
                return All(new[] { F(values), F(indices) });
            }),
            ("MaxPool3D", (e, p) => F(e.MaxPool3D(p(new[] { 1, N * N, 2, 2, 2 }), 2))),
            ("MaxPool1D", (e, p) => F(e.TensorMaxPool1D(p(new[] { 1, N * N, 8 }), 2, 2))),
            ("GlobalMaxPool2D", (e, p) => F(e.GlobalMaxPool2D(p(planeShape)))),
            ("AdaptiveMaxPool2D", (e, p) => F(e.TensorAdaptiveMaxPool2D(p(planeShape), new[] { 1, 2 }))),
            ("AdaptiveMaxPool2DGlobal", (e, p) => F(e.TensorAdaptiveMaxPool2D(p(planeShape), new[] { 1, 1 }))),
        };
        foreach (var (name, f) in pools)
            yield return (name, (e, make) => f(e, shape => make(planes, shape)), true);

        // Whole-tensor extrema of every pair (these return a host scalar, so the residency check does not apply).
        yield return ("MaxValue", (e, make) =>
        {
            var values = Enumerable.Range(0, N * N).Select(i => e.TensorMaxValue(make(new[] { pa[i], pb[i] }, new[] { 2 }))).ToArray();
            return () => values;
        }, false);
        yield return ("MinValue", (e, make) =>
        {
            var values = Enumerable.Range(0, N * N).Select(i => e.TensorMinValue(make(new[] { pa[i], pb[i] }, new[] { 2 }))).ToArray();
            return () => values;
        }, false);
    }

    public static TheoryData<string> OpNames()
    {
        var data = new TheoryData<string>();
        foreach (var (name, _, _) in Ops()) data.Add(name);
        return data;
    }

    /// <summary>
    /// Ops whose zero sign is not a single defined CPU answer. The whole-tensor extrema reduce in SIMD lanes and
    /// parallel chunks on the CPU and in a tree on the GPU, so which of two tied zeros wins depends on where the zeros
    /// sit. Everything else about these ops must still match exactly: subnormals, infinities and NaN.
    /// </summary>
    private static readonly HashSet<string> ZeroSignFree = new(StringComparer.Ordinal)
    {
        "MaxValue", "MinValue",
    };

    private static bool Same(float expected, float actual, bool zeroSignFree)
        => (float.IsNaN(expected) && float.IsNaN(actual))
           || ToBits(expected) == ToBits(actual)
           || (zeroSignFree && expected == 0f && actual == 0f);

    private static string Show(float v) => $"{v:R} (0x{ToBits(v):X8})";

    [SkippableTheory]
    [MemberData(nameof(OpNames))]
    public void EdgeValues_MatchTheCpuEngine_OnResidentInputs(string opName)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AssertMatchesCpu(gpu, opName, requireResident: true);
    }

    /// <summary>
    /// Runs <paramref name="opName"/> on the CPU engine and on <paramref name="gpu"/> over GPU-resident edge-value
    /// inputs and requires the same bits. With <paramref name="requireResident"/>, an op that returns device tensors
    /// must also run without reading anything back: a CPU fallback downloads its inputs.
    /// </summary>
    internal static void AssertMatchesCpu(DirectGpuTensorEngine gpu, string opName, bool requireResident)
    {
        var op = Ops().Single(o => o.Name == opName);
        var expected = op.Run(new CpuEngine(), (data, shape) => new Tensor<float>((float[])data.Clone(), shape))();
        Tensor<float> Upload(float[] data, int[] shape)
            => gpu.UploadToGpu(new Tensor<float>((float[])data.Clone(), shape), GpuTensorRole.General);

        // Warm the kernels once so the measured run reflects the steady state, not first-launch compilation.
        op.Run(gpu, Upload)();

        float[] actual;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            if (requireResident && op.Resident)
            {
                // The inputs are uploaded inside the scope, but only the engine calls run in it: the results are read
                // after it ends, so any download inside it is the op leaving the device.
                Func<float[]> read;
                using (var scope = GpuResidencyScope.Begin(captureOperations: true))
                {
                    read = op.Run(gpu, Upload);
                    Assert.True(scope.Downloads == 0,
                        $"{opName} on {gpu.Name} read data back to the host (a CPU fallback): " +
                        string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));
                }
                actual = read();
            }
            else
            {
                actual = op.Run(gpu, Upload)();
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }

        Assert.Equal(expected.Length, actual.Length);
        var mismatches = new List<string>();
        for (int i = 0; i < expected.Length; i++)
            if (!Same(expected[i], actual[i], ZeroSignFree.Contains(opName)))
                mismatches.Add($"[{i}] CPU {Show(expected[i])} GPU {Show(actual[i])}");
        Assert.True(mismatches.Count == 0,
            $"{opName} on {gpu.Name}: {mismatches.Count} of {expected.Length} differ from the CPU engine: " +
            string.Join("; ", mismatches.Take(12)));
    }
}

#if NET7_0_OR_GREATER
/// <summary>
/// The comparison and NaN-ignoring min/max ops of the FP32 PyTorch-parity kernels that Metal and WebGPU run
/// (TorchParityF32Kernels), as WGSL on Dawn, over the same edge values. Needs AIDOTNET_WEBGPU_NODE_DIR (see
/// WebGpuShaderCompilationTests); skipped otherwise.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuCpuEdgeValueComparisonDawnTests : IClassFixture<DawnTorchParityFixture>
{
    private readonly DawnTorchParityFixture _fixture;

    public GpuCpuEdgeValueComparisonDawnTests(DawnTorchParityFixture fixture) => _fixture = fixture;

    [SkippableTheory]
    [InlineData("GreaterEqual")]
    [InlineData("LessEqual")]
    [InlineData("Fmax")]
    [InlineData("Fmin")]
    public void EdgeValues_MatchTheCpuEngine_OnDawn(string opName)
    {
        Skip.If(!_fixture.IsAvailable, "needs Dawn (AIDOTNET_WEBGPU_NODE_DIR) and an OpenCL host device.");
        // The Dawn launcher runs out of process, so its data crosses the host by design: only the values are checked.
        GpuCpuEdgeValueComparisonTests.AssertMatchesCpu(_fixture.Engine!, opName, requireResident: false);
    }
}
#endif
