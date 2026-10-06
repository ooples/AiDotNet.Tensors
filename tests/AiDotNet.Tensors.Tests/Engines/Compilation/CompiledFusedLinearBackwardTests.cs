using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled training plan's specialized FusedLinear backward and its gradient-relevance pruning. The
/// specialization writes dW / db / dX straight into the plan's gradient buffers (overwrite) and skips dX when the
/// layer's input cannot reach a parameter; these tests pin the gradients it produces against a hand-computed
/// reference, including the cases where skipping or overwriting would be WRONG (an input derived from a parameter,
/// an input shared by two layers).
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledFusedLinearBackwardTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public CompiledFusedLinearBackwardTests()
    {
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _priorEngine;
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return new Tensor<float>(data, shape);
    }

    // Row-major helpers in double so the reference is independent of every float kernel under test.
    private static double[] MatMul(double[] a, double[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += av * b[p * n + j];
            }
        return c;
    }

    private static double[] Transpose(double[] a, int rows, int cols)
    {
        var t = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) t[c * rows + r] = a[r * cols + c];
        return t;
    }

    private static double[] ColumnSums(double[] a, int rows, int cols)
    {
        var s = new double[cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) s[c] += a[r * cols + c];
        return s;
    }

    private static double[] D(Tensor<float> t) => t.ToArray().Select(v => (double)v).ToArray();

    private static void AssertClose(double[] expected, Tensor<float> actual, string what)
    {
        var got = actual.ToArray();
        Assert.True(got.Length >= expected.Length, $"{what}: {got.Length} elements, expected {expected.Length}");
        double scale = expected.Max(v => Math.Abs(v)) + 1e-12;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(got[i] - expected[i]) <= 1e-4 * scale,
                $"{what}[{i}] = {got[i]:G7}, expected {expected[i]:G7}");
    }

    /// <summary>
    /// loss = sum((FusedLinear(ReLU(FusedLinear(X, W1, b1)), W2, b2) - T)^2). Layer 1 is large enough
    /// (64*256*128 >= 1M FMAs) to take the BLAS-routed GEMMs and layer 2 small enough to take the direct kernel,
    /// so both routes of the linear backward are pinned.
    /// </summary>
    [Fact]
    public void TwoLayerMlp_ParameterGradientsMatchReference()
    {
        const int rows = 64, inF = 256, hid = 128, outF = 8;
        var engine = new CpuEngine();
        var x = Rand([rows, inF], 1, 1f);
        var target = Rand([rows, outF], 2, 1f);
        var w1 = Rand([inF, hid], 3, 0.1f); var b1 = Rand([hid], 4, 0.1f);
        var w2 = Rand([hid, outF], 5, 0.1f); var b2 = Rand([outF], 6, 0.1f);
        var parameters = new[] { w1, b1, w2, b2 };

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.ReLU(engine.FusedLinear(x, w1, b1, FusedActivationType.None));
            var y = engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            var diff = engine.TensorSubtract(y, target);
            var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
            plan = scope.CompileTraining(parameters, loss);
        }

        // Reference.
        var z1 = MatMul(D(x), D(w1), rows, inF, hid);
        var b1d = D(b1);
        for (int r = 0; r < rows; r++) for (int j = 0; j < hid; j++) z1[r * hid + j] += b1d[j];
        var hRef = z1.Select(v => Math.Max(v, 0)).ToArray();
        var yRef = MatMul(hRef, D(w2), rows, hid, outF);
        var b2d = D(b2); var td = D(target);
        var dY = new double[rows * outF];
        for (int r = 0; r < rows; r++) for (int j = 0; j < outF; j++)
            dY[r * outF + j] = 2 * (yRef[r * outF + j] + b2d[j] - td[r * outF + j]);
        var dW2 = MatMul(Transpose(hRef, rows, hid), dY, hid, rows, outF);
        var db2 = ColumnSums(dY, rows, outF);
        var dH = MatMul(dY, Transpose(D(w2), hid, outF), rows, outF, hid);
        var dZ1 = dH.Select((g, i) => z1[i] > 0 ? g : 0).ToArray();
        var dW1 = MatMul(Transpose(D(x), rows, inF), dZ1, inF, rows, hid);
        var db1 = ColumnSums(dZ1, rows, hid);

        using (plan)
        {
            // Twice: the second step replays into buffers the first one already wrote (overwrite, not accumulate).
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                var g = plan.Gradients;
                AssertClose(dW1, g[0], $"step {step} dW1");
                AssertClose(db1, g[1], $"step {step} db1");
                AssertClose(dW2, g[2], $"step {step} dW2");
                AssertClose(db2, g[3], $"step {step} db2");
            }
        }
    }

    /// <summary>
    /// The layer's input is computed from a parameter (X = P * data), so its gradient is RELEVANT and dX must be
    /// produced: P's gradient is dX * data. Relevance pruning that skipped every first-layer dX would zero it.
    /// </summary>
    [Fact]
    public void InputDerivedFromParameter_GetsItsGradient()
    {
        const int rows = 16, inF = 24, outF = 8;
        var engine = new CpuEngine();
        var data = Rand([rows, inF], 11, 1f);
        var p = Rand([rows, inF], 12, 1f);
        var w = Rand([inF, outF], 13, 0.3f); var b = Rand([outF], 14, 0.3f);
        var parameters = new[] { p, w, b };

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var xin = engine.TensorMultiply(p, data);
            var y = engine.FusedLinear(xin, w, b, FusedActivationType.None);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, y), null);
            plan = scope.CompileTraining(parameters, loss);
        }

        var xd = D(p).Zip(D(data), (a, c) => a * c).ToArray();
        var yRef = MatMul(xd, D(w), rows, inF, outF);
        var bd = D(b);
        var dY = new double[rows * outF];
        for (int r = 0; r < rows; r++) for (int j = 0; j < outF; j++) dY[r * outF + j] = 2 * (yRef[r * outF + j] + bd[j]);
        var dX = MatMul(dY, Transpose(D(w), inF, outF), rows, outF, inF);
        var dP = dX.Zip(D(data), (g, c) => g * c).ToArray();

        using (plan)
        {
            plan.Step();
            AssertClose(dP, plan.Gradients[0], "dP");
            AssertClose(MatMul(Transpose(xd, rows, inF), dY, inF, rows, outF), plan.Gradients[1], "dW");
        }
    }

    /// <summary>
    /// One parameter-derived input feeds two linear layers: its gradient is the SUM of both layers' dX, so neither
    /// layer may overwrite it. P's gradient pins that the shared operand accumulates.
    /// </summary>
    [Fact]
    public void InputSharedByTwoLayers_AccumulatesBothContributions()
    {
        const int rows = 16, inF = 24, outF = 8;
        var engine = new CpuEngine();
        var data = Rand([rows, inF], 21, 1f);
        var p = Rand([rows, inF], 22, 1f);
        var wa = Rand([inF, outF], 23, 0.3f); var ba = Rand([outF], 24, 0.3f);
        var wb = Rand([inF, outF], 25, 0.3f); var bb = Rand([outF], 26, 0.3f);
        var parameters = new[] { p, wa, ba, wb, bb };

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var xin = engine.TensorMultiply(p, data);
            var ya = engine.FusedLinear(xin, wa, ba, FusedActivationType.None);
            var yb = engine.FusedLinear(xin, wb, bb, FusedActivationType.None);
            var loss = engine.ReduceSum(engine.TensorAdd(engine.TensorMultiply(ya, ya), yb), null);
            plan = scope.CompileTraining(parameters, loss);
        }

        var xd = D(p).Zip(D(data), (a, c) => a * c).ToArray();
        var yaRef = MatMul(xd, D(wa), rows, inF, outF);
        var bad = D(ba);
        var dYa = new double[rows * outF];
        for (int r = 0; r < rows; r++) for (int j = 0; j < outF; j++) dYa[r * outF + j] = 2 * (yaRef[r * outF + j] + bad[j]);
        var dYb = Enumerable.Repeat(1.0, rows * outF).ToArray();
        var dX = MatMul(dYa, Transpose(D(wa), inF, outF), rows, outF, inF)
            .Zip(MatMul(dYb, Transpose(D(wb), inF, outF), rows, outF, inF), (u, v) => u + v).ToArray();
        var dP = dX.Zip(D(data), (g, c) => g * c).ToArray();

        using (plan)
        {
            plan.Step();
            AssertClose(dP, plan.Gradients[0], "dP");
        }
    }

    /// <summary>
    /// A branch computed only from data (here a label normaliser |T| + |T|) cannot reach a parameter, so it
    /// contributes no backward actions: the plan is the same size as one whose normaliser is a precomputed constant.
    /// </summary>
    [Fact]
    public void DataOnlyBranch_AddsNoBackwardActions()
    {
        const int rows = 8, inF = 12, outF = 4;
        var engine = new CpuEngine();
        var x = Rand([rows, inF], 31, 1f);
        var t = Rand([rows, outF], 32, 1f);
        var w = Rand([inF, outF], 33, 0.3f); var b = Rand([outF], 34, 0.3f);
        var precomputed = engine.TensorAdd(engine.TensorAbs(t), engine.TensorAbs(t));

        int Count(bool normaliseFromData)
        {
            var parameters = new[] { w, b };
            using var scope = GraphMode.EnableTraining(parameters);
            var y = engine.FusedLinear(x, w, b, FusedActivationType.None);
            var weighted = engine.TensorMultiply(y, t);
            var norm = normaliseFromData
                ? engine.TensorAdd(engine.TensorAbs(t), engine.TensorAbs(t))
                : precomputed;
            var loss = engine.ReduceSum(engine.TensorDivide(weighted, norm), null);
            using var plan = scope.CompileTraining(parameters, loss);
            plan.Step();
            return plan.BackwardStepCount;
        }

        Assert.Equal(Count(normaliseFromData: false), Count(normaliseFromData: true));
    }

    /// <summary>
    /// A loss with an unspecialized (generic, accumulating) backward -- LogSoftmax -- in front of specialized dense
    /// layers. The step zeroes only the buffers an accumulating backward adds into; the logits gradient is one of
    /// them, so a step that skipped it would add this step's gradient onto the previous one. Stepping three times at
    /// fixed weights pins that every step's gradients equal the reference, not a multiple of it.
    /// </summary>
    [Fact]
    public void GenericLossBackward_RepeatedStepsDoNotAccumulate()
    {
        const int rows = 32, inF = 48, hid = 40, outF = 10;
        var engine = new CpuEngine();
        var x = Rand([rows, inF], 41, 1f);
        var target = Rand([rows, outF], 42, 1f);
        var w1 = Rand([inF, hid], 43, 0.2f); var b1 = Rand([hid], 44, 0.1f);
        var w2 = Rand([hid, outF], 45, 0.2f); var b2 = Rand([outF], 46, 0.1f);
        var parameters = new[] { w1, b1, w2, b2 };

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var h = engine.ReLU(engine.FusedLinear(x, w1, b1, FusedActivationType.None));
            var z = engine.FusedLinear(h, w2, b2, FusedActivationType.None);
            var loss = engine.ReduceSum(engine.TensorMultiply(engine.TensorLogSoftmax(z, 1), target), null);
            plan = scope.CompileTraining(parameters, loss);
        }

        // Reference: L = sum(logsoftmax(z) * T)  =>  dL/dz[r,j] = T[r,j] - softmax(z)[r,j] * sum_j T[r,j].
        var z1 = MatMul(D(x), D(w1), rows, inF, hid);
        var b1d = D(b1);
        for (int r = 0; r < rows; r++) for (int j = 0; j < hid; j++) z1[r * hid + j] += b1d[j];
        var hRef = z1.Select(v => Math.Max(v, 0)).ToArray();
        var zRef = MatMul(hRef, D(w2), rows, hid, outF);
        var b2d = D(b2); var td = D(target);
        var dZ = new double[rows * outF];
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < outF; j++) max = Math.Max(max, zRef[r * outF + j] + b2d[j]);
            double sumExp = 0, sumT = 0;
            for (int j = 0; j < outF; j++) { sumExp += Math.Exp(zRef[r * outF + j] + b2d[j] - max); sumT += td[r * outF + j]; }
            for (int j = 0; j < outF; j++)
                dZ[r * outF + j] = td[r * outF + j] - Math.Exp(zRef[r * outF + j] + b2d[j] - max) / sumExp * sumT;
        }
        var dW2 = MatMul(Transpose(hRef, rows, hid), dZ, hid, rows, outF);
        var db2 = ColumnSums(dZ, rows, outF);
        var dH = MatMul(dZ, Transpose(D(w2), hid, outF), rows, outF, hid);
        var dZ1 = dH.Select((g, i) => z1[i] > 0 ? g : 0).ToArray();
        var dW1 = MatMul(Transpose(D(x), rows, inF), dZ1, inF, rows, hid);
        var db1 = ColumnSums(dZ1, rows, hid);

        using (plan)
        {
            for (int step = 0; step < 3; step++)
            {
                plan.Step();
                var g = plan.Gradients;
                AssertClose(dW1, g[0], $"step {step} dW1");
                AssertClose(db1, g[1], $"step {step} db1");
                AssertClose(dW2, g[2], $"step {step} dW2");
                AssertClose(db2, g[3], $"step {step} db2");
            }
        }
    }
}
