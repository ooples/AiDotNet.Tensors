using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Autodiff.Transforms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Second derivatives through small MLPs, the computation physics-informed networks rest on: the Laplacian of a
/// tanh network's output with respect to its input. The Hessian from <see cref="TensorFunc{T}.Hessian"/> (a
/// create-graph backward differentiated again) and from <see cref="TensorFunc{T}.Hvp"/> must match the central
/// finite differences of the network's EXACT gradient — written out by hand below, independent of the tape — in
/// double precision.
/// </summary>
/// <remarks>
/// Before the activations' backward passes recorded differentiable ops under create-graph, the second backward saw
/// each activation's gradient as a constant: the Hessian of this network came back as the weights' outer product
/// terms alone (or zero), and a PINN's Laplacian residual trained against the wrong operator.
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class MlpHessianLaplacianTests : IDisposable
{
    private const int In = 3, Hidden1 = 5, Hidden2 = 4, Classes = 4;
    private const double Step = 1e-5, Tolerance = 1e-6;

    private readonly IEngine _previousEngine;
    private readonly IEngine _engine;

    public MlpHessianLaplacianTests()
    {
        _previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        _engine = AiDotNetEngine.Current;
    }

    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    private sealed class Weights
    {
        public double[,] W1 = new double[0, 0], W2 = new double[0, 0], W3 = new double[0, 0];
        public double[] B1 = Array.Empty<double>(), B2 = Array.Empty<double>(), B3 = Array.Empty<double>();
    }

    private static double[,] Matrix(Random rng, int rows, int cols, double scale)
    {
        var m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = (rng.NextDouble() * 2 - 1) * scale;
        return m;
    }

    private static double[] Vector(Random rng, int n, double scale)
    {
        var v = new double[n];
        for (int i = 0; i < n; i++) v[i] = (rng.NextDouble() * 2 - 1) * scale;
        return v;
    }

    private static Weights MakeWeights(int seed, int outputs)
    {
        var rng = new Random(seed);
        return new Weights
        {
            W1 = Matrix(rng, In, Hidden1, 0.9),
            B1 = Vector(rng, Hidden1, 0.3),
            W2 = Matrix(rng, Hidden1, Hidden2, 0.9),
            B2 = Vector(rng, Hidden2, 0.3),
            W3 = Matrix(rng, Hidden2, outputs, 0.9),
            B3 = Vector(rng, outputs, 0.3),
        };
    }

    private static Tensor<double> ToTensor(double[,] m)
    {
        int r = m.GetLength(0), c = m.GetLength(1);
        var t = new Tensor<double>(new[] { r, c });
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t[i * c + j] = m[i, j];
        return t;
    }

    private static Tensor<double> Row(double[] v)
    {
        var t = new Tensor<double>(new[] { 1, v.Length });
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    // Dense → Tanh → Dense → Tanh → Dense, as recorded engine ops: x [In] ↦ f(x) [1, 1].
    private Func<Tensor<double>, Tensor<double>> TanhMlp(Weights w)
    {
        var w1 = ToTensor(w.W1);
        var w2 = ToTensor(w.W2);
        var w3 = ToTensor(w.W3);
        var b1 = Row(w.B1);
        var b2 = Row(w.B2);
        var b3 = Row(w.B3);
        return x =>
        {
            var row = _engine.Reshape(x, new[] { 1, In });
            var h1 = _engine.Tanh(_engine.TensorAdd(_engine.TensorMatMul(row, w1), b1));
            var h2 = _engine.Tanh(_engine.TensorAdd(_engine.TensorMatMul(h1, w2), b2));
            return _engine.TensorAdd(_engine.TensorMatMul(h2, w3), b3);
        };
    }

    // Dense → Sigmoid → Dense → Softmax, contracted with fixed class weights c: x [In] ↦ Σ_k c_k p_k [1, 1].
    private Func<Tensor<double>, Tensor<double>> SigmoidSoftmaxMlp(Weights w, double[] c)
    {
        var w1 = ToTensor(w.W1);
        var w2 = ToTensor(w.W2);
        var w3 = ToTensor(w.W3);
        var b1 = Row(w.B1);
        var b2 = Row(w.B2);
        var b3 = Row(w.B3);
        var classWeights = new Tensor<double>(new[] { Classes, 1 });
        for (int k = 0; k < Classes; k++) classWeights[k] = c[k];
        return x =>
        {
            var row = _engine.Reshape(x, new[] { 1, In });
            var h1 = _engine.TensorSigmoid(_engine.TensorAdd(_engine.TensorMatMul(row, w1), b1));
            var h2 = _engine.Tanh(_engine.TensorAdd(_engine.TensorMatMul(h1, w2), b2));
            var p = _engine.Softmax(_engine.TensorAdd(_engine.TensorMatMul(h2, w3), b3), -1);
            return _engine.TensorMatMul(p, classWeights);
        };
    }

    private static double[] RowTimes(double[] v, double[,] m)
    {
        int r = m.GetLength(0), c = m.GetLength(1);
        var result = new double[c];
        for (int j = 0; j < c; j++)
        {
            double s = 0;
            for (int i = 0; i < r; i++) s += v[i] * m[i, j];
            result[j] = s;
        }
        return result;
    }

    // v Mᵀ
    private static double[] RowTimesTransposed(double[] v, double[,] m)
    {
        int r = m.GetLength(0), c = m.GetLength(1);
        var result = new double[r];
        for (int i = 0; i < r; i++)
        {
            double s = 0;
            for (int j = 0; j < c; j++) s += v[j] * m[i, j];
            result[i] = s;
        }
        return result;
    }

    private static double[] Add(double[] a, double[] b)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] + b[i];
        return r;
    }

    // The tanh network's exact input gradient, by hand-written backpropagation in double.
    private static double[] TanhMlpGradient(Weights w, double[] x)
    {
        var h1 = Add(RowTimes(x, w.W1), w.B1);
        for (int i = 0; i < h1.Length; i++) h1[i] = Math.Tanh(h1[i]);
        var h2 = Add(RowTimes(h1, w.W2), w.B2);
        for (int i = 0; i < h2.Length; i++) h2[i] = Math.Tanh(h2[i]);
        var dh2 = new double[Hidden2];
        for (int i = 0; i < Hidden2; i++) dh2[i] = w.W3[i, 0] * (1 - h2[i] * h2[i]);
        var dh1 = RowTimesTransposed(dh2, w.W2);
        for (int i = 0; i < Hidden1; i++) dh1[i] *= 1 - h1[i] * h1[i];
        return RowTimesTransposed(dh1, w.W1);
    }

    // The sigmoid/softmax network's exact input gradient, by hand.
    private static double[] SigmoidSoftmaxMlpGradient(Weights w, double[] c, double[] x)
    {
        var h1 = Add(RowTimes(x, w.W1), w.B1);
        for (int i = 0; i < h1.Length; i++) h1[i] = 1 / (1 + Math.Exp(-h1[i]));
        var h2 = Add(RowTimes(h1, w.W2), w.B2);
        for (int i = 0; i < h2.Length; i++) h2[i] = Math.Tanh(h2[i]);
        var z = Add(RowTimes(h2, w.W3), w.B3);
        double max = double.NegativeInfinity;
        foreach (var v in z) max = Math.Max(max, v);
        var p = new double[Classes];
        double sum = 0;
        for (int k = 0; k < Classes; k++) sum += p[k] = Math.Exp(z[k] - max);
        double cp = 0;
        for (int k = 0; k < Classes; k++) { p[k] /= sum; cp += c[k] * p[k]; }
        var dz = new double[Classes];
        for (int k = 0; k < Classes; k++) dz[k] = p[k] * (c[k] - cp);
        var dh2 = RowTimesTransposed(dz, w.W3);
        for (int i = 0; i < Hidden2; i++) dh2[i] *= 1 - h2[i] * h2[i];
        var dh1 = RowTimesTransposed(dh2, w.W2);
        for (int i = 0; i < Hidden1; i++) dh1[i] *= h1[i] * (1 - h1[i]);
        return RowTimesTransposed(dh1, w.W1);
    }

    // H[i, j] ≈ (∂_i f(x + h e_j) - ∂_i f(x - h e_j)) / 2h from the exact gradient.
    private static double[,] FiniteDifferenceHessian(Func<double[], double[]> gradient, double[] x)
    {
        var h = new double[In, In];
        for (int j = 0; j < In; j++)
        {
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[j] += Step;
            minus[j] -= Step;
            var gp = gradient(plus);
            var gm = gradient(minus);
            for (int i = 0; i < In; i++) h[i, j] = (gp[i] - gm[i]) / (2 * Step);
        }
        return h;
    }

    private static Tensor<double> Point() => new Tensor<double>(new[] { 0.3, -0.7, 0.45 }, new[] { In });

    private static double[] ToArray(Tensor<double> t)
    {
        var a = new double[t.Length];
        for (int i = 0; i < a.Length; i++) a[i] = t[i];
        return a;
    }

    private void AssertHessianAndLaplacian(Func<Tensor<double>, Tensor<double>> network, Func<double[], double[]> exactGradient)
    {
        var x = Point();
        var xs = ToArray(x);

        // The tape's first-order gradient is the hand-written one (sanity for the reference).
        var tapeGradient = ToArray(TensorFunc<double>.Grad(network)(x));
        var handGradient = exactGradient(xs);
        for (int i = 0; i < In; i++)
            Assert.True(Math.Abs(tapeGradient[i] - handGradient[i]) < 1e-12,
                $"gradient[{i}]: tape {tapeGradient[i]:G17} vs hand {handGradient[i]:G17}");

        var expected = FiniteDifferenceHessian(exactGradient, xs);
        var hessian = TensorFunc<double>.Hessian(network)(x);
        Assert.Equal(new[] { In, In }, hessian.Shape.ToArray());

        double laplacian = 0, expectedLaplacian = 0;
        for (int i = 0; i < In; i++)
        {
            for (int j = 0; j < In; j++)
            {
                double got = hessian[i * In + j];
                Assert.True(Math.Abs(got - expected[i, j]) < Tolerance,
                    $"H[{i},{j}]: double backward {got:G12} vs finite differences of the exact gradient {expected[i, j]:G12}");
            }
            laplacian += hessian[i * In + i];
            expectedLaplacian += expected[i, i];
        }
        Assert.True(Math.Abs(laplacian - expectedLaplacian) < Tolerance,
            $"Laplacian: {laplacian:G12} vs {expectedLaplacian:G12}");

        // The same columns through Hessian-vector products (one create-graph backward, then one more).
        double hvpLaplacian = 0;
        for (int j = 0; j < In; j++)
        {
            var e = new Tensor<double>(new[] { In });
            e[j] = 1;
            var column = TensorFunc<double>.Hvp(network, x, e);
            for (int i = 0; i < In; i++)
                Assert.True(Math.Abs(column[i] - expected[i, j]) < Tolerance,
                    $"Hvp(e_{j})[{i}]: {column[i]:G12} vs {expected[i, j]:G12}");
            hvpLaplacian += column[j];
        }
        Assert.True(Math.Abs(hvpLaplacian - expectedLaplacian) < Tolerance,
            $"Laplacian by Hvp: {hvpLaplacian:G12} vs {expectedLaplacian:G12}");
    }

    [Fact]
    public void TanhMlp_HessianAndLaplacian_MatchFiniteDifferencesOfTheExactGradient()
    {
        var w = MakeWeights(seed: 7, outputs: 1);
        AssertHessianAndLaplacian(TanhMlp(w), x => TanhMlpGradient(w, x));
    }

    [Fact]
    public void SigmoidSoftmaxMlp_HessianAndLaplacian_MatchFiniteDifferencesOfTheExactGradient()
    {
        var w = MakeWeights(seed: 11, outputs: Classes);
        var c = new[] { 0.8, -1.3, 0.4, 1.1 };
        AssertHessianAndLaplacian(SigmoidSoftmaxMlp(w, c), x => SigmoidSoftmaxMlpGradient(w, c, x));
    }

    [Fact]
    public void TanhMlp_LaplacianThroughGradientTapeCreateGraph_MatchesFiniteDifferences()
    {
        // The PINN pattern without TensorFunc: one create-graph backward for ∇f, then one backward per component.
        var w = MakeWeights(seed: 7, outputs: 1);
        var network = TanhMlp(w);
        var x = Point();
        var expected = FiniteDifferenceHessian(v => TanhMlpGradient(w, v), ToArray(x));
        double expectedLaplacian = expected[0, 0] + expected[1, 1] + expected[2, 2];

        double laplacian = 0;
        using (var tape = new GradientTape<double>())
        {
            var y = network(x);
            var grad = tape.ComputeGradients(y, new[] { x }, createGraph: true)[x];
            for (int i = 0; i < In; i++)
            {
                var selector = new Tensor<double>(new[] { In });
                selector[i] = 1;
                var component = _engine.ReduceSum(_engine.TensorMultiply(grad, selector), null);
                var second = tape.ComputeGradients(component, new[] { x }, createGraph: true)[x];
                laplacian += second[i];
            }
        }
        Assert.True(Math.Abs(laplacian - expectedLaplacian) < Tolerance,
            $"Laplacian: {laplacian:G12} vs {expectedLaplacian:G12}");
    }
}
