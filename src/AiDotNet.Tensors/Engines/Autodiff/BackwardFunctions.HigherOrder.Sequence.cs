using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Small building blocks for writing an op's forward as a composition of recorded engine ops, which the sequence-scan
/// and spectral backwards differentiate under <c>createGraph</c> (see
/// <see cref="BackwardFunctions{T}.TryCompositeBackward"/>). Every helper uses only ops whose own backward is recorded,
/// so the composition is differentiable to any order. Element-wise helpers broadcast operands of equal rank
/// explicitly (a recorded broadcast), since only the multiply broadcasts on its own.
/// </summary>
internal static class RecordedOps<T>
{
    private static readonly Interfaces.INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The scalar <paramref name="value"/> in the element type.</summary>
    internal static T S(double value) => Ops.FromDouble(value);

    /// <summary>Slice <paramref name="index"/> of <paramref name="axis"/>, with that axis removed.</summary>
    internal static Tensor<T> At(IEngine e, Tensor<T> x, int axis, int index)
    {
        var shape = new int[x.Rank - 1];
        for (int d = 0, k = 0; d < x.Rank; d++) if (d != axis) shape[k++] = x._shape[d];
        return e.Reshape(e.TensorNarrow(x, axis, index, 1), shape);
    }

    /// <summary>Recorded reshape.</summary>
    internal static Tensor<T> R(IEngine e, Tensor<T> x, params int[] shape) => e.Reshape(x, shape);

    /// <summary>Stacks equal-shape tensors along a new <paramref name="axis"/>.</summary>
    internal static Tensor<T> Stack(IEngine e, IList<Tensor<T>> items, int axis)
    {
        var array = new Tensor<T>[items.Count];
        items.CopyTo(array, 0);
        return e.TensorStack(array, axis);
    }

    private static (Tensor<T>, Tensor<T>) Common(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != b.Rank)
            throw new ArgumentException($"RecordedOps: operands must have equal rank; got [{string.Join(",", a._shape)}] and [{string.Join(",", b._shape)}].");
        bool same = true;
        var shape = new int[a.Rank];
        for (int d = 0; d < shape.Length; d++)
        {
            shape[d] = Math.Max(a._shape[d], b._shape[d]);
            if (a._shape[d] != b._shape[d]) same = false;
        }
        if (same) return (a, b);
        return (Expand(e, a, shape), Expand(e, b, shape));
    }

    private static Tensor<T> Expand(IEngine e, Tensor<T> x, int[] shape)
    {
        for (int d = 0; d < shape.Length; d++)
            if (x._shape[d] != shape[d]) return e.TensorBroadcastTo(x, shape);
        return x;
    }

    /// <summary>a + b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Add(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorAdd(x, y);
    }

    /// <summary>a - b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Sub(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorSubtract(x, y);
    }

    /// <summary>a ⊙ b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Mul(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorMultiply(x, y);
    }

    /// <summary>a / b (equal rank, broadcast over size-1 axes).</summary>
    internal static Tensor<T> Div(IEngine e, Tensor<T> a, Tensor<T> b)
    {
        var (x, y) = Common(e, a, b);
        return e.TensorDivide(x, y);
    }

    /// <summary>x · s.</summary>
    internal static Tensor<T> Scale(IEngine e, Tensor<T> x, double s) => e.TensorMultiplyScalar(x, S(s));

    /// <summary>x + s.</summary>
    internal static Tensor<T> Shift(IEngine e, Tensor<T> x, double s) => e.TensorAddScalar(x, S(s));

    /// <summary>Σ over <paramref name="axis"/>, kept as size 1 or removed.</summary>
    internal static Tensor<T> Sum(IEngine e, Tensor<T> x, int axis, bool keep) => e.ReduceSum(x, new[] { axis }, keep);

    /// <summary>
    /// A constant (unrecorded) tensor f(x) element-wise, read from <paramref name="x"/>'s current values: the masks and
    /// signs that select a branch of a piecewise op (a clamp, a max, an absolute value), whose derivative within the
    /// selected branch the recorded ops then carry.
    /// </summary>
    [ScalarKernel("Builds a constant branch mask from a few host values; nothing here is recorded or differentiated.")]
    internal static Tensor<T> Constant(Tensor<T> x, Func<double, double> f)
    {
        var src = x.GetFlattenedData();
        var dst = new T[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = Ops.FromDouble(f(Ops.ToDouble(src[i])));
        return new Tensor<T>(dst, (int[])x._shape.Clone());
    }

    /// <summary>A constant zero tensor (records nothing).</summary>
    internal static Tensor<T> Zeros(params int[] shape) => new Tensor<T>(shape);

    /// <summary>A constant tensor of <paramref name="value"/> (records nothing).</summary>
    internal static Tensor<T> Full(double value, params int[] shape)
    {
        var t = new Tensor<T>(shape);
        t.Fill(S(value));
        return t;
    }
}

/// <summary>Recorded short-time Fourier analysis and synthesis for the spectral ops' create-graph backwards.</summary>
internal static partial class BackwardFunctions<T>
{
    // Phase tensors the ISTFT composite feeds its own ISTFT calls with: constants (0 or π/2), so those calls are
    // linear in their magnitude, and IstftBackward takes the linear path for them instead of recursing.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Tensor<T>, object> ConstantSynthesisPhases = new();

    /// <summary>
    /// The real (or imaginary) part of the centred STFT of <paramref name="waveform"/>, laid out like its magnitude
    /// spectrogram, recorded as a linear node: its backward is the analysis adjoint (<see cref="MagnitudeStftAdjoint"/>
    /// with phase 0, or π/2 for the imaginary part), itself recorded with this map as its adjoint, so every order of
    /// derivative stays connected.
    /// </summary>
    internal static Tensor<T> RecordedStftPart(IEngine engine, Tensor<T> waveform, bool imaginary, int nFft, int hopLength,
        Tensor<T> window, int origLength)
    {
        Tensor<T> magnitude, phase;
        using (new NoGradScope<T>())
            engine.STFT(waveform, nFft, hopLength, window, center: true, out magnitude, out phase);
        var part = PolarPart(magnitude, phase, imaginary);
        var waveformShape = (int[])waveform._shape.Clone();
        DifferentiableOps.RecordUnary(imaginary ? "StftImagPart" : "StftRealPart", part, waveform,
            (g, ins, _, _, eng, grads) =>
            {
                var gShape = (int[])g._shape.Clone();
                var adjoint = LinearGrad(imaginary ? "StftImagPartBackward" : "StftRealPartBackward", g,
                    () => MagnitudeStftAdjoint(g, RecordedOps<T>.Full(imaginary ? Math.PI / 2 : 0.0, gShape),
                        nFft, hopLength, window, origLength, waveformShape),
                    (e, w) => RecordedStftPart(e, w, imaginary, nFft, hopLength, window, origLength));
                DifferentiableOps.AccumulateGrad(grads, ins[0], adjoint, eng);
            });
        return part;
    }

    // m·cos(p) or m·sin(p) element-wise (constant values; the recorded node above carries the derivative).
    [ScalarKernel("Element-wise polar-to-Cartesian of a host spectrogram that seeds a recorded node.")]
    private static Tensor<T> PolarPart(Tensor<T> magnitude, Tensor<T> phase, bool imaginary)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var m = magnitude.GetFlattenedData();
        var p = phase.GetFlattenedData();
        var result = new T[m.Length];
        for (int i = 0; i < m.Length; i++)
        {
            double angle = ops.ToDouble(p[i]);
            result[i] = ops.FromDouble(ops.ToDouble(m[i]) * (imaginary ? Math.Sin(angle) : Math.Cos(angle)));
        }
        return new Tensor<T>(result, (int[])magnitude._shape.Clone());
    }

    /// <summary>
    /// ISTFT(m, p) as recorded ops: the synthesis is linear in the complex spectrum m·e^{ip}, so it is
    /// ISTFT(m cos p, 0) + ISTFT(m sin p, π/2), whose constant phases make each call linear in its magnitude.
    /// </summary>
    internal static Tensor<T> IstftComposite(IEngine engine, Tensor<T> magnitude, Tensor<T> phase, int nFft, int hopLength,
        Tensor<T> window, bool center, int outputLength)
    {
        var shape = (int[])magnitude._shape.Clone();
        var zero = RecordedOps<T>.Full(0.0, shape);
        var quarter = RecordedOps<T>.Full(Math.PI / 2, shape);
        ConstantSynthesisPhases.Add(zero, zero);
        ConstantSynthesisPhases.Add(quarter, quarter);
        var real = engine.ISTFT(engine.TensorMultiply(magnitude, engine.TensorCos(phase)), zero, nFft, hopLength, window, center, outputLength);
        var imag = engine.ISTFT(engine.TensorMultiply(magnitude, engine.TensorSin(phase)), quarter, nFft, hopLength, window, center, outputLength);
        return engine.TensorAdd(real, imag);
    }

    /// <summary>True when <paramref name="phase"/> is one of <see cref="IstftComposite"/>'s constant phases.</summary>
    internal static bool IsConstantSynthesisPhase(Tensor<T> phase) => ConstantSynthesisPhases.TryGetValue(phase, out _);

    /// <summary>
    /// AmplitudeToDB's input gradient as recorded ops: (20/ln 10)·(g ⊙ M + P·Σ(g ⊙ Cl)) / x, with M the elements the
    /// forward takes the log of unclipped, Cl the elements clipped to peak - topDb (their gradient belongs to the
    /// peak) and P the peak, all constant masks chosen exactly as <see cref="AmplitudeToDBBackward"/> chooses them.
    /// </summary>
    internal static Tensor<T> AmplitudeToDBGradRecorded(Tensor<T> gradOutput, Tensor<T> input, double minAmp, float? topDb, IEngine engine)
    {
        var (live, clipped, peak) = AmplitudeToDBMasks(input, minAmp, topDb);
        var allAxes = new int[input.Rank];
        for (int d = 0; d < allAxes.Length; d++) allAxes[d] = d;
        Tensor<T> routed = engine.TensorMultiply(gradOutput, live);
        if (topDb.HasValue)
        {
            var toPeak = engine.ReduceSum(engine.TensorMultiply(gradOutput, clipped), allAxes, keepDims: true);
            routed = engine.TensorAdd(routed, RecordedOps<T>.Mul(engine, peak, toPeak));
        }
        // x where the gradient is nonzero, 1 elsewhere (where the routed gradient is zero anyway).
        var used = RecordedOps<T>.Constant(engine.TensorAdd(live, peak), v => v > 0 ? 1.0 : 0.0);
        var safe = engine.TensorAdd(engine.TensorMultiply(input, used), BackwardFunctions<T>.OneMinus(engine, used));
        return RecordedOps<T>.Scale(engine, engine.TensorDivide(routed, safe), 20.0 / Math.Log(10.0));
    }

    [ScalarKernel("Branch masks of AmplitudeToDB's clamp and topDb clip, read once from the host values.")]
    private static (Tensor<T> Live, Tensor<T> Clipped, Tensor<T> Peak) AmplitudeToDBMasks(Tensor<T> input, double minAmp, float? topDb)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var src = input.GetFlattenedData();
        int n = src.Length;
        var live = new T[n];
        var clipped = new T[n];
        var peak = new T[n];
        int peakIndex = -1;
        double floor = double.NegativeInfinity;
        if (topDb.HasValue)
        {
            double best = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                double db = 20.0 * Math.Log10(Math.Max(ops.ToDouble(src[i]), minAmp));
                if (db > best) { best = db; peakIndex = i; }
            }
            floor = best - topDb.Value;
        }
        for (int i = 0; i < n; i++)
        {
            double x = ops.ToDouble(src[i]);
            if (topDb.HasValue && ops.ToDouble(ops.FromDouble(20.0 * Math.Log10(Math.Max(x, minAmp)))) < floor)
            {
                clipped[i] = ops.One;
                live[i] = ops.Zero;
            }
            else
            {
                clipped[i] = ops.Zero;
                live[i] = x > minAmp ? ops.One : ops.Zero;
            }
            peak[i] = ops.Zero;
        }
        if (peakIndex >= 0 && ops.ToDouble(src[peakIndex]) > minAmp) peak[peakIndex] = ops.One;
        var shape = (int[])input._shape.Clone();
        return (new Tensor<T>(live, shape), new Tensor<T>(clipped, (int[])shape.Clone()), new Tensor<T>(peak, (int[])shape.Clone()));
    }
}
