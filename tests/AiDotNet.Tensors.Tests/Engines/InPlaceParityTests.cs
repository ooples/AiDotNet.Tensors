using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The torch <c>op_</c> forms: every in-place op leaves the tensor holding exactly what its out-of-place op returns,
/// writes through views into the shared storage, and the layout ops change shape and strides, not values.
/// </summary>
public class InPlaceParityTests
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = new CpuEngine();

    public InPlaceParityTests(ITestOutputHelper output) => _output = output;

    // A view of t's storage with the given layout (the constructor every view op uses).
    private static Tensor<double> View(Tensor<double> t, int[] shape, int[] strides, int offset)
        => new Tensor<double>(t.DataVector, shape, strides, offset, t._storage);

    private static Tensor<double> Sample(int[] shape, int salt)
    {
        var t = new Tensor<double>(shape);
        var s = t.AsWritableSpan();
        // In (0.15, 0.85) and distinct: inside the domain of the logs, roots and inverse trig functions, away from the
        // integer edges of floor / round / fmod, and never equal to each other.
        for (int i = 0; i < s.Length; i++) s[i] = 0.15 + 0.7 * ((i * 37 + salt * 11) % 97) / 97.0;
        return t;
    }

    // Arguments for one parameter type; null when the harness has no value for it.
    private static object? Argument(ParameterInfo p, int[] shape, int salt)
    {
        var t = p.ParameterType;
        string n = p.Name ?? "";
        if (t == typeof(Tensor<double>)) return Sample(shape, salt);
        if (t == typeof(Tensor<Bit>))
        {
            var b = new Tensor<Bit>(shape);
            for (int i = 0; i < b.Length; i++) b[i] = (i % 3 == 0) ? Bit.True : Bit.False;
            return b;
        }
        if (t == typeof(Tensor<int>))
        {
            // indices along axis 1 (length 4), or an exponent tensor for ldexp
            if (n.Contains("exp")) { var e = new Tensor<int>(shape); e.Fill(2); return e; }
            return new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        }
        if (t == typeof(int)) return n is "axis" or "dim" or "spatialDims" ? 1 : n is "n" or "p" ? 2 : 1;
        if (t == typeof(int?)) return n == "seed" ? 1234 : (int?)null;
        if (t == typeof(double)) return n is "p" or "dropoutRate" ? 0.25 : n == "maxNorm" ? 1.0 : 0.5;
        if (t == typeof(double?)) return null;
        if (t == typeof(bool)) return n == "training" ? true : n == "accumulate" ? false : true;
        if (t == typeof(long)) return 3L;
        if (t.IsEnum) return Enum.GetValues(t).GetValue(0);
        if (t == typeof(Tensor<bool>))
        {
            var m = new Tensor<bool>(shape);
            for (int i = 0; i < m.Length; i++) m[i] = i % 2 == 0;
            return m;
        }
        if (t == typeof(bool[])) return Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => i % 2 == 0).ToArray();
        if (t == typeof(Tensor<int>[])) return new[] { new Tensor<int>(new[] { 0, 1 }, new[] { 2 }), new Tensor<int>(new[] { 1, 3 }, new[] { 2 }) };
        if (t.IsGenericParameter || t == typeof(double)) return 0.5;
        return null;
    }

    private static bool SameValues(Tensor<double> a, Tensor<double> b)
    {
        if (!a._shape.SequenceEqual(b._shape)) return false;
        var x = a.ToArray(); var y = b.ToArray();
        for (int i = 0; i < x.Length; i++)
        {
            if (double.IsNaN(x[i]) && double.IsNaN(y[i])) continue;
            if (x[i] != y[i] && Math.Abs(x[i] - y[i]) > 1e-12 * Math.Max(1, Math.Abs(y[i]))) return false;
        }
        return true;
    }

    [Fact]
    public void EveryGeneratedInPlaceOp_MatchesItsOutOfPlaceOp()
    {
        var methods = typeof(IEngine).GetMethods().Where(m => m.Name.EndsWith("InPlace", StringComparison.Ordinal)).ToList();
        int checkedOps = 0;
        var skipped = new List<string>();
        var failures = new List<string>();
        foreach (var inPlace in methods)
        {
            string baseName = inPlace.Name.Substring(0, inPlace.Name.Length - "InPlace".Length);
            var ipParams = inPlace.GetParameters();
            // The out-of-place op with exactly the same parameter types once constructed (the in-place form puts the tensor
            // first, so the order may differ). Comparing constructed types matters: Tensor<int> and Tensor<T> share a name.
            static Type[] Constructed(System.Reflection.MethodInfo m)
                => (m.IsGenericMethodDefinition ? m.MakeGenericMethod(typeof(double)) : m).GetParameters()
                    .Select(p => p.ParameterType).OrderBy(t => t.FullName).ToArray();
            if (!inPlace.IsGenericMethodDefinition && inPlace.GetParameters()[0].ParameterType != typeof(Tensor<Bit>)) continue;
            var ipTypes = Constructed(inPlace);
            var outOfPlace = typeof(IEngine).GetMethods().FirstOrDefault(m =>
                m.Name == baseName && m.GetParameters().Length == ipParams.Length && m.ReturnType.IsGenericType &&
                (m.IsGenericMethodDefinition ? m.GetGenericArguments().Length == 1 : true) &&
                Constructed(m).SequenceEqual(ipTypes));
            if (outOfPlace is null) continue;   // hand-written forms are covered by the tests below

            var ip = inPlace.IsGenericMethodDefinition ? inPlace.MakeGenericMethod(typeof(double)) : inPlace;
            var op = outOfPlace.IsGenericMethodDefinition ? outOfPlace.MakeGenericMethod(typeof(double)) : outOfPlace;
            bool bitOp = ip.GetParameters()[0].ParameterType == typeof(Tensor<Bit>);
            if (bitOp) continue;   // Tensor<Bit> logical ops: checked in LogicalInPlace_MatchesOutOfPlace

            foreach (var shape in new[] { new[] { 3, 4 }, new[] { 4, 4 } })
            {
                var args = ip.GetParameters().Select((p, i) => Argument(p, shape, i)).ToArray();
                if (args.Select((a, i) => a is null && !ip.GetParameters()[i].ParameterType.IsGenericType
                        && Nullable.GetUnderlyingType(ip.GetParameters()[i].ParameterType) is null).Any(missing => missing))
                {
                    skipped.Add($"{inPlace.Name}: no argument for a parameter"); break;
                }
                // Same arguments for the out-of-place op, in its own order, with the tensor as an untouched copy.
                var self = (Tensor<double>)args[0];
                var opArgs = new object?[op.GetParameters().Length];
                var used = new bool[args.Length];
                for (int j = 0; j < opArgs.Length; j++)
                {
                    var pj = op.GetParameters()[j];
                    int k = Enumerable.Range(0, args.Length).Where(i => !used[i] && ip.GetParameters()[i].ParameterType == pj.ParameterType).DefaultIfEmpty(-1).First();
                    if (k < 0) break;
                    used[k] = true;
                    opArgs[j] = k == 0 ? self.Clone() : args[k];
                }
                if (used.Any(x => !x)) { skipped.Add($"{inPlace.Name}: parameters do not pair with {baseName}"); break; }
                Tensor<double> expected;
                try { expected = (Tensor<double>)op.Invoke(_engine, opArgs)!; }
                catch (TargetInvocationException) { continue; }   // this shape is outside the op's contract; try the next
                if (!expected._shape.SequenceEqual(self._shape)) { skipped.Add($"{inPlace.Name}: shape changes"); break; }

                try { ip.Invoke(_engine, args); }
                catch (TargetInvocationException ex) { failures.Add($"{inPlace.Name} threw where {baseName} succeeded: {ex.InnerException?.Message}"); break; }
                if (!SameValues(self, expected))
                    failures.Add($"{inPlace.Name} on [{string.Join(",", shape)}]: in-place result differs from {baseName}");
                checkedOps++;
                break;
            }
        }
        foreach (var s in skipped) _output.WriteLine("skipped " + s);
        _output.WriteLine($"{checkedOps} in-place ops checked against their out-of-place op");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        Assert.True(checkedOps >= 100, $"only {checkedOps} in-place ops were exercised; the harness is not reaching them");
    }

    [Fact]
    public void LogicalInPlace_MatchesOutOfPlace()
    {
        Tensor<Bit> Bits(params bool[] v) { var t = new Tensor<Bit>(new[] { v.Length }); for (int i = 0; i < v.Length; i++) t[i] = v[i] ? Bit.True : Bit.False; return t; }
        var a = Bits(true, true, false, false);
        var b = Bits(true, false, true, false);
        var expected = _engine.TensorLogicalXor(a, b).ToArray();
        _engine.TensorLogicalXorInPlace(a, b);
        Assert.Equal(expected, a.ToArray());
    }

    [Fact]
    public void InPlaceOnAView_WritesTheSharedStorage()
    {
        var m = Sample(new[] { 4, 4 }, 0);
        var before = m.ToArray();
        var column = View(m, new[] { 4, 4 }, new[] { 1, 4 }, 0);   // m transposed: a strided view of its storage
        _engine.TensorExpInPlace(column);
        var after = m.ToArray();
        for (int i = 0; i < after.Length; i++) Assert.Equal(Math.Exp(before[i]), after[i], 12);
    }

    [Fact]
    public void FastPath_IsBitIdenticalToOutOfPlace()
    {
        foreach (var (name, op, inPlace) in new (string, Func<Tensor<double>, Tensor<double>>, Action<Tensor<double>>)[]
        {
            ("floor", _engine.TensorFloor, _engine.TensorFloorInPlace),
            ("ceil", _engine.TensorCeiling, _engine.TensorCeilingInPlace),
            ("asin", _engine.TensorAsin, _engine.TensorAsinInPlace),
            ("log1p", _engine.TensorLog1p, _engine.TensorLog1pInPlace),
            ("expm1", _engine.TensorExpm1, _engine.TensorExpm1InPlace),
            ("tan", _engine.TensorTan, _engine.TensorTanInPlace),
        })
        {
            // Large enough to take the parallel chunked path.
            var t = Sample(new[] { 300_000 }, 3);
            var expected = op(t.Clone()).ToArray();
            inPlace(t);
            Assert.True(expected.SequenceEqual(t.ToArray()), $"{name}_ fast path differs from {name}");
        }
    }

    [Fact]
    public void SeededRandomFills_MatchTheirOutOfPlaceGenerators()
    {
        var t = new Tensor<double>(new[] { 5, 6 });
        _engine.TensorUniformInPlace(t, -2, 3, seed: 7);
        Assert.Equal(_engine.TensorUniform<double>(new[] { 5, 6 }, -2, 3, 7).ToArray(), t.ToArray());
        _engine.TensorExponentialInPlace(t, 2.0, seed: 9);
        Assert.Equal(_engine.TensorExponential<double>(new[] { 5, 6 }, 2.0, 9).ToArray(), t.ToArray());
    }

    [Fact]
    public void LayoutOps_ChangeShapeAndStridesNotValues()
    {
        var m = Sample(new[] { 1, 3, 1, 4 }, 1);
        var values = m.ToArray();
        _engine.TensorSqueezeInPlace(m);
        Assert.Equal(_engine.TensorSqueeze(Sample(new[] { 1, 3, 1, 4 }, 1)).Shape.ToArray(), m.Shape.ToArray());
        Assert.Equal(values, m.ToArray());

        var a = Sample(new[] { 3, 4 }, 2);
        var expectedT = _engine.TensorSwapAxes(a.Clone(), 0, 1).ToArray();
        _engine.TensorTInPlace(a);
        Assert.Equal(new[] { 4, 3 }, a.Shape.ToArray());
        Assert.Equal(expectedT, a.ToArray());

        var u = Sample(new[] { 3, 4 }, 4);
        _engine.TensorUnsqueezeInPlace(u, 0);
        Assert.Equal(new[] { 1, 3, 4 }, u.Shape.ToArray());
    }

    [Fact]
    public void AsStridedInPlace_ViewsItsOwnStorage()
    {
        var t = new Tensor<double>(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), new[] { 12 });
        _engine.TensorAsStridedInPlace(t, new[] { 3, 2 }, new[] { 4, 1 }, 1);
        Assert.Equal(new double[] { 1, 2, 5, 6, 9, 10 }, t.ToArray());
    }

    [Fact]
    public void SetInPlace_SharesTheSourceStorage()
    {
        var src = Sample(new[] { 2, 3 }, 5);
        var t = new Tensor<double>(new[] { 7 });
        _engine.TensorSetInPlace(t, src);
        Assert.Equal(src.Shape.ToArray(), t.Shape.ToArray());
        src[0, 0] = 42;   // a write through either is seen by both
        Assert.Equal(42, t[0, 0]);
    }

    [Fact]
    public void ResizeInPlace_KeepsTheLeadingValuesAndZerosTheRest()
    {
        var t = new Tensor<double>(new double[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _engine.TensorResizeInPlace(t, new[] { 3, 2 });
        Assert.Equal(new double[] { 1, 2, 3, 4, 0, 0 }, t.ToArray());
        _engine.TensorResizeInPlace(t, new[] { 3 });
        Assert.Equal(new double[] { 1, 2, 3 }, t.ToArray());
    }

    [Fact]
    public void Fill_OnARowView_LeavesTheOtherRows()
    {
        var m = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });
        var row = View(m, new[] { 2 }, new[] { 1 }, 2);   // m[1]: contiguous, at a non-zero offset
        _engine.TensorFill(row, 9.0);
        Assert.Equal(new double[] { 1, 2, 9, 9, 5, 6 }, m.ToArray());
    }

    [Fact]
    public void Fill_OnAStridedView_WritesTheView()
    {
        var m = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 2 });
        var firstColumn = View(m, new[] { 3 }, new[] { 2 }, 0);   // m[:, 0]: strided
        _engine.TensorFill(firstColumn, 0.0);
        Assert.Equal(new double[] { 0, 2, 0, 4, 0, 6 }, m.ToArray());
    }

    [Fact]
    public void FillDiagonalInPlace_MatchesTorch()
    {
        var t = new Tensor<double>(new[] { 4, 3 });
        _engine.TensorFillDiagonalInPlace(t, 1.0);
        Assert.Equal(new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 }, t.ToArray());
        var w = new Tensor<double>(new[] { 7, 3 });
        _engine.TensorFillDiagonalInPlace(w, 1.0, wrap: true);
        // torch: the diagonal restarts after cols + 1 = 4 rows
        Assert.Equal(new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 }, w.ToArray());
    }

    [Fact]
    public void EmbeddingRenormInPlace_ScalesOnlyTheNamedRowsOverTheLimit()
    {
        var w = new Tensor<double>(new double[] { 3, 4, 0.3, 0.4, 6, 8 }, new[] { 3, 2 });
        _engine.TensorEmbeddingRenormInPlace(w, new Tensor<int>(new[] { 0, 1 }, new[] { 2 }), maxNorm: 1.0);
        var v = w.ToArray();
        Assert.Equal(1.0, Math.Sqrt(v[0] * v[0] + v[1] * v[1]), 6);   // row 0 (norm 5) scaled to 1
        Assert.Equal(new[] { 0.3, 0.4 }, new[] { v[2], v[3] });         // row 1 already within
        Assert.Equal(new[] { 6.0, 8.0 }, new[] { v[4], v[5] });         // row 2 not named
    }

    [Fact]
    public void DetachInPlace_StopsTheGradientOfLaterUses()
    {
        var x = new Tensor<double>(new double[] { 1, 2, 3 }, new[] { 3 });
        using var tape = new GradientTape<double>();
        var y = _engine.TensorMultiplyScalar(x, 2.0);
        var early = _engine.ReduceSum(y, null);            // recorded before the detach: still differentiable
        _engine.TensorDetachInPlace(y);
        var late = _engine.ReduceSum(_engine.TensorMultiply(y, y), null);
        var loss = _engine.TensorAdd(early, late);
        var gx = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
        Assert.Equal(new double[] { 2, 2, 2 }, gx);   // only `early` reaches x
    }
}
