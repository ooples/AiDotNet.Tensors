using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

// #679: core operations of the netstandard2.0 build of AiDotNet.Tensors against hand-computed references.
int failures = 0;
void Check(string what, bool ok, string detail = "")
{
    Console.WriteLine($"{(ok ? "PASS" : "FAIL")}  {what}{(ok || detail.Length == 0 ? "" : "  " + detail)}");
    if (!ok) failures++;
}
bool Close(float a, float b, float tol = 1e-4f) => Math.Abs(a - b) <= tol * (1 + Math.Abs(b));

var asm = typeof(Tensor<float>).Assembly;
string tfm = asm.GetCustomAttribute<System.Runtime.Versioning.TargetFrameworkAttribute>()?.FrameworkName ?? "?";
Console.WriteLine($"AiDotNet.Tensors target: {tfm}; runtime: {Environment.Version}");
Check("loaded the .NET Standard 2.0 asset", tfm.StartsWith(".NETStandard,Version=v2.0", StringComparison.Ordinal), tfm);

var engine = new CpuEngine();

// MatMul [2,3] x [3,2]
var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
var b = new Tensor<float>(new float[] { 7, 8, 9, 10, 11, 12 }, new[] { 3, 2 });
var c = engine.TensorMatMul(a, b).ToArray();
Check("TensorMatMul", c.SequenceEqual(new float[] { 58, 64, 139, 154 }), string.Join(",", c));

// Larger MatMul against a naive loop
var rng = new Random(3);
int m = 37, k = 53, n = 29;
var x = new Tensor<float>(Enumerable.Range(0, m * k).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray(), new[] { m, k });
var y = new Tensor<float>(Enumerable.Range(0, k * n).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray(), new[] { k, n });
var z = engine.TensorMatMul(x, y).ToArray();
var xa = x.ToArray(); var ya = y.ToArray();
bool mmOk = true;
for (int i = 0; i < m && mmOk; i++)
    for (int j = 0; j < n; j++)
    {
        double s = 0;
        for (int t = 0; t < k; t++) s += xa[i * k + t] * (double)ya[t * n + j];
        if (!Close(z[i * n + j], (float)s, 1e-3f)) { mmOk = false; break; }
    }
Check("TensorMatMul 37x53x29 vs naive", mmOk);

// Softmax rows sum to 1 and match exp(x)/sum
var sIn = new Tensor<float>(new float[] { 1, 2, 3, -1, 0, 1 }, new[] { 2, 3 });
var sm = engine.Softmax(sIn, -1).ToArray();
double e1 = Math.Exp(1), e2 = Math.Exp(2), e3 = Math.Exp(3);
Check("Softmax", Close(sm[0], (float)(e1 / (e1 + e2 + e3))) && Close(sm[2], (float)(e3 / (e1 + e2 + e3))) && Close(sm[3] + sm[4] + sm[5], 1f));

// ReduceSum / elementwise / activation
var sum = engine.ReduceSum(a, null).ToArray()[0];
Check("ReduceSum", sum == 21f, sum.ToString());
var added = engine.TensorAdd(a, a).ToArray();
Check("TensorAdd", added.SequenceEqual(new float[] { 2, 4, 6, 8, 10, 12 }));
var relu = engine.ReLU(new Tensor<float>(new float[] { -1, 0, 2 }, new[] { 3 })).ToArray();
Check("ReLU", relu.SequenceEqual(new float[] { 0, 0, 2 }));

// LayerNorm: each row normalized to mean 0, variance ~1
var ln = engine.LayerNorm(sIn, new Tensor<float>(new float[] { 1, 1, 1 }, new[] { 3 }), new Tensor<float>(new float[3], new[] { 3 }), 1e-5, out _, out _).ToArray();
Check("LayerNorm", Close(ln[0] + ln[1] + ln[2], 0f, 1e-3f) && Close(ln[0], -1.2247f, 1e-3f));

// Autodiff: d/dw sum(x . w) = column sums of x
var xi = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
var w = new Tensor<float>(new float[] { 0.5f, -1f }, new[] { 2, 1 });
using (var tape = new GradientTape<float>())
{
    var loss = engine.ReduceSum(engine.TensorMatMul(xi, w), null);
    var g = tape.ComputeGradients(loss, new[] { w })[w].ToArray();
    Check("GradientTape MatMul gradient", g.SequenceEqual(new float[] { 4, 6 }), string.Join(",", g));
}

Console.WriteLine(failures == 0 ? "ALL PASSED" : $"{failures} FAILED");
return failures == 0 ? 0 : 1;
