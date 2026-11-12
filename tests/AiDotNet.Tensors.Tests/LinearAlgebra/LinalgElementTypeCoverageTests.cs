using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Element-type coverage for the dense <see cref="Linalg"/> kernels that AiDotNet's generic
/// decompositions bind to (CholeskyEx, Det, Inv, Eigh, Eig, SolveEx, QR, LU, SolveTriangular,
/// Svd, Diagonal, CholeskySolve, CholeskyInverse, Solve, LuFactor, LuSolve).
/// <para>
/// Every real field type that has <c>INumericOperations&lt;T&gt;</c> (float, double, decimal,
/// Half, BFloat16) must run every method on the CPU without throwing and produce finite
/// output. Integer and bool element types are rejected with <see cref="NotSupportedException"/>
/// by every factorization (as <c>torch.linalg</c> rejects integer dtypes); only the purely
/// structural <see cref="Linalg.Diagonal{T}"/> accepts them.
/// </para>
/// </summary>
public class LinalgElementTypeCoverageTests
{
    private readonly ITestOutputHelper _output;

    public LinalgElementTypeCoverageTests(ITestOutputHelper output) => _output = output;

    private static readonly double[] SpdValues = { 4, 1, 2, 1, 5, 3, 2, 3, 6 };
    private static readonly double[] LowerValues = { 2, 0, 0, 1, 3, 0, 2, 1, 4 };
    private static readonly double[] RhsValues = { 1, 2, 2, -1, 3, 0.5 };

    private static Tensor<T> Make<T>(int[] shape, double[] values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < values.Length; i++) d[i] = ops.FromDouble(values[i]);
        return t;
    }

    private static void AssertFinite<T>(Tensor<T> t, string what)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var d = t.GetDataArray();
        for (int i = 0; i < t.Length; i++)
            Assert.False(ops.IsNaN(d[i]) || ops.IsInfinity(d[i]), $"{what}[{i}] is not finite: {d[i]}");
    }

    /// <summary>Calls each bound Linalg method once; returns method name to thrown exception (null = ok).</summary>
    private static List<(string Method, Exception? Error)> Probe<T>()
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var a = Make<T>(new[] { 3, 3 }, SpdValues);
        var l = Make<T>(new[] { 3, 3 }, LowerValues);
        var b = Make<T>(new[] { 3, 2 }, RhsValues);
        var identityPivots = new Tensor<int>(new[] { 3 });
        var pv = identityPivots.GetDataArray();
        for (int i = 0; i < 3; i++) pv[i] = i;

        var calls = new List<(string, Action)>
        {
            ("CholeskyEx", () =>
            {
                var (f, info) = Linalg.CholeskyEx(a);
                Assert.Equal(0, info.GetDataArray()[0]);
                AssertFinite(f, "L");
            }),
            ("Det", () => AssertFinite(Linalg.Det(a), "det")),
            ("Inv", () => AssertFinite(Linalg.Inv(a), "inv")),
            ("Eigh", () =>
            {
                var (w, v) = Linalg.Eigh(a);
                AssertFinite(w, "w"); AssertFinite(v, "v");
            }),
            ("Eig", () =>
            {
                var (w, v) = Linalg.Eig(a);
                AssertFinite(w, "w"); AssertFinite(v, "v");
            }),
            ("SolveEx", () =>
            {
                var (x, info) = Linalg.SolveEx(a, b);
                Assert.Equal(0, info.GetDataArray()[0]);
                AssertFinite(x, "x");
            }),
            ("QR", () =>
            {
                var (q, r) = Linalg.QR(a);
                AssertFinite(q, "Q"); AssertFinite(r, "R");
            }),
            ("LU", () =>
            {
                var (p, lo, u) = Linalg.LU(a);
                AssertFinite(p, "P"); AssertFinite(lo, "L"); AssertFinite(u, "U");
            }),
            ("SolveTriangular", () => AssertFinite(Linalg.SolveTriangular(l, b, upper: false), "x")),
            ("Svd", () =>
            {
                var (u, s, vh) = Linalg.Svd(a);
                AssertFinite(u, "U"); AssertFinite(s, "S"); AssertFinite(vh, "Vh");
            }),
            ("Diagonal", () => AssertFinite(Linalg.Diagonal(a), "diag")),
            ("CholeskySolve", () => AssertFinite(Linalg.CholeskySolve(b, l), "x")),
            ("CholeskyInverse", () => AssertFinite(Linalg.CholeskyInverse(l), "inv")),
            ("Solve", () => AssertFinite(Linalg.Solve(a, b), "x")),
            ("LuFactor", () =>
            {
                var (lu, piv) = Linalg.LuFactor(a);
                AssertFinite(lu, "LU");
                Assert.Equal(3, piv.Length);
            }),
            ("LuSolve", () => AssertFinite(Linalg.LuSolve(a, identityPivots, b), "x")),
        };

        var results = new List<(string, Exception?)>();
        foreach (var (name, call) in calls)
        {
            try { call(); results.Add((name, null)); }
            catch (Exception ex) { results.Add((name, ex)); }
        }
        return results;
    }

    private void AssertAllSucceed<T>()
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var results = Probe<T>();
        foreach (var (method, error) in results)
            _output.WriteLine($"{typeof(T).Name,-9} {method,-16} {(error is null ? "ok" : error.GetType().Name + ": " + error.Message)}");
        var failures = results.Where(r => r.Error is not null)
            .Select(r => $"{r.Method}: {r.Error?.GetType().Name} {r.Error?.Message}").ToList();
        Assert.True(failures.Count == 0,
            $"Linalg methods failed for {typeof(T).Name}:\n" + string.Join("\n", failures));
    }

    private void AssertRejected<T>()
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        var results = Probe<T>();
        foreach (var (method, error) in results)
        {
            _output.WriteLine($"{typeof(T).Name,-9} {method,-16} {(error is null ? "ok" : error.GetType().Name)}");
            if (method == "Diagonal")
                Assert.True(error is null, $"Diagonal must accept {typeof(T).Name}: {error}");
            else
                Assert.True(error is NotSupportedException,
                    $"{method} must reject {typeof(T).Name} with NotSupportedException, got {error?.GetType().Name ?? "success"}");
        }
    }

    [Fact] public void AllBoundMethods_RunFor_Float() => AssertAllSucceed<float>();
    [Fact] public void AllBoundMethods_RunFor_Double() => AssertAllSucceed<double>();
    [Fact] public void AllBoundMethods_RunFor_Decimal() => AssertAllSucceed<decimal>();
    [Fact] public void AllBoundMethods_RunFor_BFloat16() => AssertAllSucceed<BFloat16>();
#if NET5_0_OR_GREATER
    [Fact] public void AllBoundMethods_RunFor_Half() => AssertAllSucceed<Half>();
#endif

    [Fact] public void Factorizations_Reject_Int32() => AssertRejected<int>();
    [Fact] public void Factorizations_Reject_Int64() => AssertRejected<long>();
    [Fact] public void Factorizations_Reject_Int16() => AssertRejected<short>();
    [Fact] public void Factorizations_Reject_Byte() => AssertRejected<byte>();
}
