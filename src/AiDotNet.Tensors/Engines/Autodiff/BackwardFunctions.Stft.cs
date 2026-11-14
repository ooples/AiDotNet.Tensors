using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Backwards of the two outputs of <c>IEngine.STFT</c>, for either <c>center</c>, with respect to both the waveform and
/// the window (as torch.stft: learnable-window front ends train the window).
/// </summary>
/// <remarks>
/// <para>
/// First order uses the analysis adjoint: the windowed overlap-add of the unnormalised inverse DFT of the per-bin seed
/// gives the waveform gradient, and the same per-frame inverse transform times the analysed (padded) signal gives the
/// window gradient, <c>dL/dw[i] = Σ_frames dL/df[i] · xPadded[start + i]</c>.
/// </para>
/// <para>
/// Under <c>createGraph</c> the backward differentiates <see cref="StftCartesianComposite"/>, the same analysis written
/// with recorded engine ops (a gather that frames and reflection-pads the signal, the window product, and products with
/// constant DFT bases), so every order of derivative reaches the waveform and the window.
/// </para>
/// </remarks>
internal static partial class BackwardFunctions<T>
{
    /// <summary>Backward of the magnitude output of <c>IEngine.STFT</c>; inputs are the waveform and the window.</summary>
    internal static void StftMagnitudeBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var waveform = inputs[0];
        var window = inputs[1];
        RequireSavedState(nameof(StftMagnitudeBackward), savedState, 5);
        int nFft = SavedInt(nameof(StftMagnitudeBackward), savedState, 0, "nFft");
        int hopLength = SavedInt(nameof(StftMagnitudeBackward), savedState, 1, "hopLength");
        var phase = SavedTensor(nameof(StftMagnitudeBackward), savedState, 2, "phase");
        int origLength = SavedInt(nameof(StftMagnitudeBackward), savedState, 3, "origLength");
        bool center = SavedBool(nameof(StftMagnitudeBackward), savedState, 4, "center");

        if (TryCompositeBackward(gradOutput, new[] { waveform, window }, engine, grads,
                e => StftMagnitudeComposite(e, waveform, window, nFft, hopLength, center)))
            return;

        if (TryDeviceStftBackward(gradOutput, output, phase, phaseTarget: false, waveform, window, nFft, hopLength, center,
                engine, grads))
            return;

        var windowGradient = new double[nFft];
        var result = StftAnalysisAdjoint(
            gradOutput, magnitude: null, phase, StftAdjointTarget.Magnitude,
            nFft, hopLength, window, origLength, waveform._shape, center, waveform, windowGradient);
        DifferentiableOps.AccumulateGrad(grads, waveform, result, engine);
        DifferentiableOps.AccumulateGrad(grads, window, DoublesAsTensor(windowGradient, window._shape), engine);
    }

    // The same first-order adjoint on the GPU engine's backend (all six implement the three kernels), so a device
    // training step does not download the spectrogram gradient to run it on the host.
    private static bool TryDeviceStftBackward(Tensor<T> gradOutput, Tensor<T> magnitude, Tensor<T> phase, bool phaseTarget,
        Tensor<T> waveform, Tensor<T> window, int nFft, int hopLength, bool center, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (!engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            || !gpu.TryStftBackward(gradOutput, magnitude, phase, phaseTarget, waveform, window, nFft, hopLength, center,
                out var gradWaveform, out var gradWindow)
            || gradWaveform is null || gradWindow is null)
            return false;
        DifferentiableOps.AccumulateGrad(grads, waveform, gradWaveform, engine);
        DifferentiableOps.AccumulateGrad(grads, window, gradWindow, engine);
        return true;
    }

    /// <summary>Backward of the phase output of <c>IEngine.STFT</c>; inputs are the waveform and the window.</summary>
    internal static void StftPhaseOutputBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var waveform = inputs[0];
        var window = inputs[1];
        RequireSavedState(nameof(StftPhaseOutputBackward), savedState, 5);
        int nFft = SavedInt(nameof(StftPhaseOutputBackward), savedState, 0, "nFft");
        int hopLength = SavedInt(nameof(StftPhaseOutputBackward), savedState, 1, "hopLength");
        var magnitude = SavedTensor(nameof(StftPhaseOutputBackward), savedState, 2, "magnitude");
        int origLength = SavedInt(nameof(StftPhaseOutputBackward), savedState, 3, "origLength");
        bool center = SavedBool(nameof(StftPhaseOutputBackward), savedState, 4, "center");

        if (TryCompositeBackward(gradOutput, new[] { waveform, window }, engine, grads,
                e => StftPhaseComposite(e, waveform, window, nFft, hopLength, center)))
            return;

        if (TryDeviceStftBackward(gradOutput, magnitude, output, phaseTarget: true, waveform, window, nFft, hopLength, center,
                engine, grads))
            return;

        var windowGradient = new double[nFft];
        var result = StftAnalysisAdjoint(
            gradOutput, magnitude, output, StftAdjointTarget.Phase,
            nFft, hopLength, window, origLength, waveform._shape, center, waveform, windowGradient);
        DifferentiableOps.AccumulateGrad(grads, waveform, result, engine);
        DifferentiableOps.AccumulateGrad(grads, window, DoublesAsTensor(windowGradient, window._shape), engine);
    }

    private static Tensor<T> StftMagnitudeComposite(IEngine e, Tensor<T> waveform, Tensor<T> window, int nFft,
        int hopLength, bool center)
    {
        var (re, im) = StftCartesianComposite(e, waveform, window, nFft, hopLength, center);
        return RecordedComplexMagnitude(re, im);
    }

    private static Tensor<T> StftPhaseComposite(IEngine e, Tensor<T> waveform, Tensor<T> window, int nFft,
        int hopLength, bool center)
    {
        var (re, im) = StftCartesianComposite(e, waveform, window, nFft, hopLength, center);
        return RecordedComplexPhase(re, im);
    }

    /// <summary>
    /// <c>|re + i·im|</c> as a recorded node whose derivatives, of every order, are finite at an empty bin.
    /// </summary>
    /// <remarks>
    /// <c>sqrt(re² + im²)</c> differentiated at zero divides by zero, and the NaN survives the multiplication by the
    /// zero seed. Here the first derivative is <c>(re, im)/|C|</c> where the bin is not empty and, where it is, the
    /// direction <c>(cos p, sin p)</c> of the phase the forward reports there (<c>atan2(0, 0) = 0</c>), exactly as the
    /// first-order analysis adjoint seeds it. The reciprocal is computed over <c>re² + im² + (1 - mask)</c>, which is
    /// never zero, so neither it nor any further derivative of it divides by zero; the mask then zeroes the empty bins.
    /// </remarks>
    internal static Tensor<T> RecordedComplexMagnitude(Tensor<T> re, Tensor<T> im)
    {
        var polar = PolarOfCartesian(re, im, floor: 0.0);
        var magnitude = polar.Tensor(polar.Magnitude);
        var mask = polar.Tensor(polar.Mask);
        var fill = polar.Tensor(polar.Fill);
        var emptyCos = polar.Tensor(polar.EmptyCos);
        var emptySin = polar.Tensor(polar.EmptySin);
        DifferentiableOps.RecordBinary("ComplexMagnitude", magnitude, re, im,
            (g, ins, _, _, eng, grads) =>
            {
                var r = ins[0];
                var i = ins[1];
                var squared = eng.TensorAdd(eng.TensorAdd(eng.TensorMultiply(r, r), eng.TensorMultiply(i, i)), fill);
                var reciprocal = eng.TensorDivide(mask, eng.TensorSqrt(squared));
                DifferentiableOps.AccumulateGrad(grads, r,
                    eng.TensorMultiply(g, eng.TensorAdd(eng.TensorMultiply(r, reciprocal), emptyCos)), eng);
                DifferentiableOps.AccumulateGrad(grads, i,
                    eng.TensorMultiply(g, eng.TensorAdd(eng.TensorMultiply(i, reciprocal), emptySin)), eng);
            });
        return magnitude;
    }

    /// <summary>
    /// <c>atan2(im, re)</c> as a recorded node whose derivatives, of every order, are zero at bins no larger than
    /// <see cref="PhaseGradientMagnitudeFloor"/>, as the first-order analysis adjoint's are.
    /// </summary>
    /// <remarks>
    /// The derivative <c>(-im, re)/|C|²</c> is computed over <c>re² + im² + (1 - mask)</c>, which is never zero, so the
    /// floored bins divide nothing by zero before the mask zeroes them (a zero mask times an infinite quotient would
    /// still be NaN).
    /// </remarks>
    internal static Tensor<T> RecordedComplexPhase(Tensor<T> re, Tensor<T> im)
    {
        var polar = PolarOfCartesian(re, im, PhaseGradientMagnitudeFloor);
        var phase = polar.Tensor(polar.Phase);
        var mask = polar.Tensor(polar.Mask);
        var negativeMask = polar.Tensor(polar.NegativeMask);
        var fill = polar.Tensor(polar.Fill);
        DifferentiableOps.RecordBinary("ComplexPhase", phase, re, im,
            (g, ins, _, _, eng, grads) =>
            {
                var r = ins[0];
                var i = ins[1];
                var squared = eng.TensorAdd(eng.TensorAdd(eng.TensorMultiply(r, r), eng.TensorMultiply(i, i)), fill);
                DifferentiableOps.AccumulateGrad(grads, r,
                    eng.TensorMultiply(g, eng.TensorMultiply(i, eng.TensorDivide(negativeMask, squared))), eng);
                DifferentiableOps.AccumulateGrad(grads, i,
                    eng.TensorMultiply(g, eng.TensorMultiply(r, eng.TensorDivide(mask, squared))), eng);
            });
        return phase;
    }

    private sealed class PolarValues
    {
        public PolarValues(int length, int[] shape)
        {
            Shape = shape;
            Magnitude = new T[length]; Phase = new T[length]; Mask = new T[length]; NegativeMask = new T[length];
            Fill = new T[length]; EmptyCos = new T[length]; EmptySin = new T[length];
        }

        private int[] Shape { get; }
        public T[] Magnitude { get; }
        public T[] Phase { get; }
        // 1 where the bin's magnitude exceeds the floor, else 0; NegativeMask is its negation and Fill is 1 - Mask.
        public T[] Mask { get; }
        public T[] NegativeMask { get; }
        public T[] Fill { get; }
        // (cos p, sin p) of the reported phase where the bin is at or below the floor, else 0.
        public T[] EmptyCos { get; }
        public T[] EmptySin { get; }

        public Tensor<T> Tensor(T[] values) => new Tensor<T>(values, (int[])Shape.Clone());
    }

    [ScalarKernel("Element-wise polar values and constant masks of a host spectrum that seed a recorded node.")]
    private static PolarValues PolarOfCartesian(Tensor<T> re, Tensor<T> im, double floor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var r = re.GetFlattenedData();
        var i = im.GetFlattenedData();
        var values = new PolarValues(r.Length, (int[])re._shape.Clone());
        for (int k = 0; k < r.Length; k++)
        {
            double x = ops.ToDouble(r[k]), y = ops.ToDouble(i[k]);
            double m = Math.Sqrt(x * x + y * y);
            double p = Math.Atan2(y, x);
            bool live = m > floor;
            values.Magnitude[k] = ops.FromDouble(m);
            values.Phase[k] = ops.FromDouble(p);
            values.Mask[k] = ops.FromDouble(live ? 1.0 : 0.0);
            values.NegativeMask[k] = ops.FromDouble(live ? -1.0 : 0.0);
            values.Fill[k] = ops.FromDouble(live ? 0.0 : 1.0);
            values.EmptyCos[k] = ops.FromDouble(live ? 0.0 : Math.Cos(p));
            values.EmptySin[k] = ops.FromDouble(live ? 0.0 : Math.Sin(p));
        }
        return values;
    }

    /// <summary>
    /// The real and imaginary parts of the one-sided STFT, laid out like its magnitude (<c>[..., numFreqs,
    /// numFrames]</c>), written with recorded engine ops so they are differentiable to any order in both the waveform
    /// and the window: <c>Re[k] = Σ_i f[i] cos(2πki/N)</c>, <c>Im[k] = -Σ_i f[i] sin(2πki/N)</c> with
    /// <c>f[i] = xPadded[start + i] · w[i]</c>.
    /// </summary>
    internal static (Tensor<T> Real, Tensor<T> Imaginary) StftCartesianComposite(IEngine engine, Tensor<T> waveform,
        Tensor<T> window, int nFft, int hopLength, bool center)
    {
        int length = waveform._shape[^1];
        int batch = waveform.Length / length;
        int pad = center ? nFft / 2 : 0;
        int numFrames = (length + 2 * pad - nFft) / hopLength + 1;
        int numFreqs = nFft / 2 + 1;

        // One gather both frames the signal and applies the reflection padding: frame f, tap i reads padded position
        // f·hop + i, which is the source sample p (reflected about the first and last samples when outside).
        var indices = new int[numFrames * nFft];
        for (int f = 0; f < numFrames; f++)
        {
            for (int i = 0; i < nFft; i++)
            {
                int p = f * hopLength + i - pad;
                indices[f * nFft + i] = p < 0 ? -p : p >= length ? 2 * length - 2 - p : p;
            }
        }

        var flat = engine.Reshape(waveform, new[] { batch, length });
        var gathered = engine.TensorIndexSelect(flat, new Tensor<int>(indices, new[] { indices.Length }), 1);
        var frames = engine.Reshape(gathered, new[] { batch, numFrames, nFft });
        var windowed = engine.TensorMultiply(frames, engine.Reshape(window, new[] { 1, 1, nFft }));

        var re = engine.TensorPermute(engine.TensorMatMul(windowed, DftBasis(nFft, numFreqs, imaginary: false)), new[] { 0, 2, 1 });
        var im = engine.TensorPermute(engine.TensorMatMul(windowed, DftBasis(nFft, numFreqs, imaginary: true)), new[] { 0, 2, 1 });

        var spectralShape = new int[waveform.Rank + 1];
        for (int d = 0; d < waveform.Rank - 1; d++) spectralShape[d] = waveform._shape[d];
        spectralShape[^2] = numFreqs;
        spectralShape[^1] = numFrames;
        return (engine.Reshape(re, spectralShape), engine.Reshape(im, (int[])spectralShape.Clone()));
    }

    // [nFft, numFreqs]: cos(2πki/N) for the real part, -sin(2πki/N) for the imaginary part (a constant, no graph).
    [ScalarKernel("Builds a constant DFT basis from host doubles.")]
    private static Tensor<T> DftBasis(int nFft, int numFreqs, bool imaginary)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = new T[nFft * numFreqs];
        for (int i = 0; i < nFft; i++)
        {
            for (int k = 0; k < numFreqs; k++)
            {
                double angle = 2 * Math.PI * k * i / nFft;
                data[i * numFreqs + k] = ops.FromDouble(imaginary ? -Math.Sin(angle) : Math.Cos(angle));
            }
        }
        return new Tensor<T>(data, new[] { nFft, numFreqs });
    }

    // The waveform as doubles, once per backward, for the window gradient's analysed signal.
    [ScalarKernel("Reads a host waveform into doubles for the STFT window adjoint.")]
    private static double[] SignalAsDoubles(Tensor<T> waveform)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = waveform.GetFlattenedData();
        var result = new double[data.Length];
        for (int i = 0; i < data.Length; i++) result[i] = ops.ToDouble(data[i]);
        return result;
    }

    // Batch b of the analysed signal: the waveform itself, reflection-padded by pad on each side as the forward does.
    private static void FillAnalysedSignal(double[] signal, int batchIndex, int length, int pad, double[] analysed)
    {
        int offset = batchIndex * length;
        Array.Copy(signal, offset, analysed, pad, length);
        for (int i = 0; i < pad; i++)
        {
            analysed[i] = signal[offset + pad - i];
            analysed[pad + length + i] = signal[offset + length - 2 - i];
        }
    }

    [ScalarKernel("Wraps a host gradient accumulated in doubles as a tensor.")]
    private static Tensor<T> DoublesAsTensor(double[] values, int[] shape)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = new T[values.Length];
        for (int i = 0; i < values.Length; i++) data[i] = ops.FromDouble(values[i]);
        return new Tensor<T>(data, (int[])shape.Clone());
    }
}
