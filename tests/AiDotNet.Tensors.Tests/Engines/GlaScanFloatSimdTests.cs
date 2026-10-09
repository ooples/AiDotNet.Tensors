using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float32 SIMD GLA scan (forward and BPTT backward) against the double kernel on the same inputs: output and the
/// gradients of q, k, v and the gate, with a weighted upstream gradient, a non-unit gate, and head widths that leave
/// SIMD tails (13) or match the HRE Born-linear scan width (100).
/// </summary>
public class GlaScanFloatSimdTests
{
    private static readonly CpuEngine Engine = new();

    [Theory]
    [InlineData(2, 9, 3, 13)]
    [InlineData(2, 16, 2, 100)]
    public void FloatScan_MatchesDoubleKernel(int batch, int seq, int heads, int headDim)
    {
        int model = heads * headDim;
        var rng = new Random(batch * 101 + headDim);
        double[] R(int n, double lo, double hi) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = lo + (hi - lo) * rng.NextDouble(); return a; }
        var q = R(batch * seq * model, -0.5, 0.5); var k = R(batch * seq * model, -0.5, 0.5);
        var v = R(batch * seq * model, -0.5, 0.5); var g = R(batch * seq * heads, 0.6, 1.0);
        var w = R(batch * seq * model, -1, 1);
        int[] s3 = { batch, seq, model }, sg = { batch, seq, heads };
        float[] F(double[] a) => Array.ConvertAll(a, x => (float)x);

        var qf = new Tensor<float>(F(q), s3); var kf = new Tensor<float>(F(k), s3);
        var vf = new Tensor<float>(F(v), s3); var gf = new Tensor<float>(F(g), sg);
        float[] of; float[][] df;
        using (var tape = new GradientTape<float>())
        {
            var o = Engine.GlaScanForward(qf, kf, vf, gf, heads);
            of = o.ToArray();
            var loss = Engine.ReduceSum(Engine.TensorMultiply(o, new Tensor<float>(F(w), s3)), null);
            var gr = tape.ComputeGradients(loss, new[] { qf, kf, vf, gf });
            df = new[] { gr[qf].ToArray(), gr[kf].ToArray(), gr[vf].ToArray(), gr[gf].ToArray() };
        }

        var qd = new Tensor<double>(q, s3); var kd = new Tensor<double>(k, s3);
        var vd = new Tensor<double>(v, s3); var gd = new Tensor<double>(g, sg);
        double[] od; double[][] dd;
        using (var tape = new GradientTape<double>())
        {
            var o = Engine.GlaScanForward(qd, kd, vd, gd, heads);
            od = o.ToArray();
            var loss = Engine.ReduceSum(Engine.TensorMultiply(o, new Tensor<double>(w, s3)), null);
            var gr = tape.ComputeGradients(loss, new[] { qd, kd, vd, gd });
            dd = new[] { gr[qd].ToArray(), gr[kd].ToArray(), gr[vd].ToArray(), gr[gd].ToArray() };
        }

        AssertClose("output", of, od);
        string[] names = { "dQ", "dK", "dV", "dGate" };
        for (int i = 0; i < 4; i++) AssertClose(names[i], df[i], dd[i]);
    }

    private static void AssertClose(string what, float[] f, double[] d)
    {
        Assert.Equal(d.Length, f.Length);
        double maxErr = 0, maxRef = 0;
        for (int i = 0; i < d.Length; i++) { maxErr = Math.Max(maxErr, Math.Abs(f[i] - d[i])); maxRef = Math.Max(maxRef, Math.Abs(d[i])); }
        Assert.True(maxRef > 0, $"{what}: reference is all zero");
        Assert.True(maxErr <= 2e-5 * Math.Max(1, maxRef), $"{what}: max |error| {maxErr:G4} (max |ref| {maxRef:G4})");
    }
}
