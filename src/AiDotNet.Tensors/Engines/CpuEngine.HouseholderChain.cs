// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Returns the default squared-norm floor for <see cref="HouseholderChain{T}"/> on element type <typeparamref name="T"/>.
    /// </summary>
    internal static double DefaultHouseholderNormFloor<T>()
    {
        string typeName = typeof(T).Name;
        return (typeof(T) == typeof(double) || typeof(T) == typeof(decimal))
            ? 1.1754943508222875e-38
            : (typeName == "Half" || typeName.StartsWith("Float8", StringComparison.Ordinal))
                ? 1e-2
                : 1e-30;
    }

    /// <summary>
    /// Applies a matrix-free chain of <c>K</c> Householder reflections <c>y = x H_0 H_1 ... H_{K-1}</c>
    /// (or reverse order when <paramref name="transpose"/> is <c>true</c>) to <paramref name="input"/> (<c>[..., D]</c>)
    /// using <paramref name="reflectors"/> (<c>[K, D]</c>) normalized with overflow-safe row scaling and squared-norm
    /// floor <paramref name="normFloor"/>.
    /// </summary>
    public virtual Tensor<T> HouseholderChain<T>(
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose = false,
        double normFloor = 0.0)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (reflectors is null) throw new ArgumentNullException(nameof(reflectors));
        if (reflectors.Rank != 2)
            throw new ArgumentException($"Reflectors must be rank-2 [K, D], got rank {reflectors.Rank}.", nameof(reflectors));

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        if (input.Rank < 1 || input.Shape[input.Rank - 1] != d)
            throw new ArgumentException($"Expected last dimension {d}, got [{string.Join(",", input._shape)}].", nameof(input));
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double) && typeof(T).Name != "Half")
            throw new NotSupportedException($"HouseholderChain supports floating-point element types (float, double, Half); got {typeof(T).Name}.");

        double effFloor = normFloor > 0.0 ? normFloor : DefaultHouseholderNormFloor<T>();
        int rows = d == 0 ? 0 : input.Length / d; // [..., 0] holds no rows

        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capIn = input;
            var capRef = reflectors;
            bool capTrans = transpose;
            double capFloor = effFloor;
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "HouseholderChain",
                new[] { input, reflectors },
                (int[])input._shape.Clone(),
                (eng, outT) =>
                {
                    var res = eng.HouseholderChain(capIn, capRef, capTrans, capFloor);
                    DirectGpuTensorEngine.CopyResultInto(eng, res, outT);
                },
                HouseholderChainBackward<T>,
                new object[] { transpose, effFloor });
        }

        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            return HouseholderChainComposed(this, input, reflectors, transpose, effFloor);
        }

        var output = new Tensor<T>((int[])input._shape.Clone());
        if (rows > 0)
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcRef = reflectors.IsContiguous ? reflectors : reflectors.Contiguous();
            if (typeof(T) == typeof(float))
            {
                HouseholderChainForwardFloat(
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcRef).AsSpan(),
                    ((Tensor<float>)(object)output).AsWritableSpan(),
                    rows, k, d, transpose, effFloor);
            }
            else
            {
                HouseholderChainForwardDouble(
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcRef).AsSpan(),
                    ((Tensor<double>)(object)output).AsWritableSpan(),
                    rows, k, d, transpose, effFloor);
            }
        }

        DifferentiableOps.RecordIfActive<T>(
            "HouseholderChain",
            output,
            new[] { input, reflectors },
            HouseholderChainBackward<T>,
            savedState: new object[] { transpose, effFloor });
        return output;
    }

    private static Tensor<T> HouseholderChainComposed<T>(
        IEngine engine,
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose,
        double normFloor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        int rows = d == 0 ? 0 : input.Length / d; // [..., 0] holds no rows
        // No rows, or no reflectors (the empty product of reflections is the identity): y = x.
        if (rows == 0 || k == 0)
            return engine.TensorMultiplyScalar(input, numOps.One);

        double sqrtFloor = Math.Sqrt(normFloor);
        var rowMaxAbs = engine.StopGradient(engine.ReduceMax(engine.TensorAbs(reflectors), new[] { 1 }, keepDims: true));
        var rowScale = engine.TensorMax(rowMaxAbs, numOps.FromDouble(sqrtFloor / 16.0));
        var rowScaleFull = engine.TensorBroadcastTo(rowScale, reflectors._shape);
        var scaled = engine.TensorDivide(reflectors, rowScaleFull);
        var scaledSq = engine.ReduceSum(engine.TensorMultiply(scaled, scaled), new[] { 1 }, keepDims: true);
        var scaledFloorRoot = engine.TensorMultiplyScalar(engine.TensorReciprocal(rowScale), numOps.FromDouble(sqrtFloor));
        var scaledFloor = engine.TensorMultiply(scaledFloorRoot, scaledFloorRoot);
        var denom = engine.TensorMax(scaledSq, scaledFloor);
        var sqrtDenomFull = engine.TensorBroadcastTo(engine.TensorSqrt(denom), reflectors._shape);
        var uHat = engine.TensorDivide(scaled, sqrtDenomFull);

        var cur = input.Rank == 2 ? input : engine.Reshape(input, new[] { rows, d });
        for (int s = 0; s < k; s++)
        {
            int j = transpose ? (k - 1 - s) : s;
            var uj = engine.TensorNarrow(uHat, 0, j, 1);
            var proj = engine.TensorMatMul(cur, engine.TensorTranspose(uj));
            var step = engine.TensorMatMul(proj, uj);
            cur = engine.TensorSubtract(engine.TensorSubtract(cur, step), step);
        }
        return input.Rank == 2 ? cur : engine.Reshape(cur, input._shape);
    }

    internal static void HouseholderChainBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        bool transpose = (bool)savedState[0];
        double normFloor = (double)savedState[1];
        var input = inputs[0];
        var reflectors = inputs[1];
        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        int rows = d == 0 ? 0 : input.Length / d; // [..., 0] holds no rows

        if (DifferentiableOps._isBackwardCreateGraph || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
        {
            HouseholderChainHigherOrderBackward(gradOutput, input, reflectors, transpose, normFloor, engine, grads);
            return;
        }

        if (rows > 0 && k > 0
            && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
            && gpu.TryHouseholderChainBackwardGpu(gradOutput, input, reflectors, transpose, normFloor) is { } gpuGrads)
        {
            DifferentiableOps.AccumulateGrad(grads, input, gpuGrads.GradInput, engine);
            DifferentiableOps.AccumulateGrad(grads, reflectors, gpuGrads.GradReflectors, engine);
            return;
        }

        var gradInput = new Tensor<T>((int[])input._shape.Clone());
        var gradReflectors = new Tensor<T>((int[])reflectors._shape.Clone());
        if (rows > 0)
        {
            var srcGrad = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcRef = reflectors.IsContiguous ? reflectors : reflectors.Contiguous();
            if (typeof(T) == typeof(float))
            {
                HouseholderChainBackwardFloat(
                    ((Tensor<float>)(object)srcGrad).AsSpan(),
                    ((Tensor<float>)(object)srcIn).AsSpan(),
                    ((Tensor<float>)(object)srcRef).AsSpan(),
                    ((Tensor<float>)(object)gradInput).AsWritableSpan(),
                    ((Tensor<float>)(object)gradReflectors).AsWritableSpan(),
                    rows, k, d, transpose, normFloor);
            }
            else
            {
                HouseholderChainBackwardDouble(
                    ((Tensor<double>)(object)srcGrad).AsSpan(),
                    ((Tensor<double>)(object)srcIn).AsSpan(),
                    ((Tensor<double>)(object)srcRef).AsSpan(),
                    ((Tensor<double>)(object)gradInput).AsWritableSpan(),
                    ((Tensor<double>)(object)gradReflectors).AsWritableSpan(),
                    rows, k, d, transpose, normFloor);
            }
        }
        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);
        DifferentiableOps.AccumulateGrad(grads, reflectors, gradReflectors, engine);
    }

    private static void HouseholderChainHigherOrderBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose,
        double normFloor,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        int rows = d == 0 ? 0 : input.Length / d; // [..., 0] holds no rows
        double sqrtFloor = Math.Sqrt(normFloor);

        // No rows or no reflectors: y = x, so dL/dx = dL/dy and nothing reaches the reflectors. Kept on the tape so a
        // higher-order pass still differentiates through it.
        if (rows == 0 || k == 0)
        {
            DifferentiableOps.AccumulateGrad(grads, input, engine.TensorMultiplyScalar(gradOutput, numOps.One), engine);
            DifferentiableOps.AccumulateGrad(grads, reflectors, engine.TensorMultiplyScalar(reflectors, numOps.Zero), engine);
            return;
        }

        var rowMaxAbs = engine.StopGradient(engine.ReduceMax(engine.TensorAbs(reflectors), new[] { 1 }, keepDims: true));
        var rowScale = engine.TensorMax(rowMaxAbs, numOps.FromDouble(sqrtFloor / 16.0));
        var rowScaleFull = engine.TensorBroadcastTo(rowScale, reflectors._shape);
        var scaled = engine.TensorDivide(reflectors, rowScaleFull);
        var scaledSq = engine.ReduceSum(engine.TensorMultiply(scaled, scaled), new[] { 1 }, keepDims: true);
        var scaledFloorRoot = engine.TensorMultiplyScalar(engine.TensorReciprocal(rowScale), numOps.FromDouble(sqrtFloor));
        var scaledFloor = engine.TensorMultiply(scaledFloorRoot, scaledFloorRoot);
        var denom = engine.TensorMax(scaledSq, scaledFloor);
        var sqrtDenomFull = engine.TensorBroadcastTo(engine.TensorSqrt(denom), reflectors._shape);
        var uHat = engine.TensorDivide(scaled, sqrtDenomFull);

        var x2d = input.Rank == 2 ? input : engine.Reshape(input, new[] { rows, d });
        var g2d = gradOutput.Rank == 2 ? gradOutput : engine.Reshape(gradOutput, new[] { rows, d });
        T negTwo = numOps.FromDouble(-2.0);

        var stages = new Tensor<T>[k];
        var cur = x2d;
        for (int s = 0; s < k; s++)
        {
            stages[s] = cur;
            int j = transpose ? (k - 1 - s) : s;
            var uj = engine.TensorNarrow(uHat, 0, j, 1);
            var ujCol = engine.TensorTranspose(uj);
            var proj = engine.TensorMatMul(cur, ujCol);
            var step = engine.TensorMatMul(proj, uj);
            cur = engine.TensorSubtract(engine.TensorSubtract(cur, step), step);
        }

        var dUHatRows = new Tensor<T>[k];
        var gCur = g2d;
        for (int s = k - 1; s >= 0; s--)
        {
            int j = transpose ? (k - 1 - s) : s;
            var uj = engine.TensorNarrow(uHat, 0, j, 1);
            var ujCol = engine.TensorTranspose(uj);
            var p = engine.TensorMatMul(stages[s], ujCol);
            var q = engine.TensorMatMul(gCur, ujCol);
            var term1 = engine.TensorMatMul(engine.TensorTranspose(q), stages[s]);
            var term2 = engine.TensorMatMul(engine.TensorTranspose(p), gCur);
            var duRow = engine.TensorMultiplyScalar(engine.TensorAdd(term1, term2), negTwo);
            dUHatRows[j] = duRow;
            var stepQ = engine.TensorMatMul(q, uj);
            gCur = engine.TensorSubtract(engine.TensorSubtract(gCur, stepQ), stepQ);
        }

        var dUHat = k == 1 ? dUHatRows[0] : engine.TensorConcatenate(dUHatRows, axis: 0);
        var dot = engine.ReduceSum(engine.TensorMultiply(dUHat, uHat), new[] { 1 }, keepDims: true);
        var active = engine.StopGradient(engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorSign(engine.TensorSubtract(scaledSq, scaledFloor)), numOps.One),
            numOps.FromDouble(0.5)));
        var activeDotFull = engine.TensorBroadcastTo(engine.TensorMultiply(active, dot), reflectors._shape);
        var dScaled = engine.TensorDivide(
            engine.TensorSubtract(dUHat, engine.TensorMultiply(activeDotFull, uHat)),
            sqrtDenomFull);
        var dRef = engine.TensorDivide(dScaled, rowScaleFull);
        var gradIn = input.Rank == 2 ? gCur : engine.Reshape(gCur, input._shape);
        DifferentiableOps.AccumulateGrad(grads, input, gradIn, engine);
        DifferentiableOps.AccumulateGrad(grads, reflectors, dRef, engine);
    }

    private static void NormalizeReflectorRowsFloat(
        ReadOnlySpan<float> reflectors,
        int k,
        int d,
        double normFloor,
        double[] uHat,
        double[] invNorm,
        double[] active)
    {
        double sqrtFloor = Math.Sqrt(normFloor);
        double minScale = sqrtFloor * 0.0625;
        for (int j = 0; j < k; j++)
        {
            int off = j * d;
            double maxAbs = 0.0;
            for (int i = 0; i < d; i++)
            {
                double a = Math.Abs((double)reflectors[off + i]);
                if (a > maxAbs) maxAbs = a;
            }
            double rowScale = Math.Max(maxAbs, minScale);
            double scaledSq = 0.0;
            for (int i = 0; i < d; i++)
            {
                double sc = reflectors[off + i] / rowScale;
                scaledSq += sc * sc;
            }
            double sfr = sqrtFloor / rowScale;
            double scaledFloor = sfr * sfr;
            double denom = Math.Max(scaledSq, scaledFloor);
            double r = 1.0 / Math.Sqrt(denom);
            for (int i = 0; i < d; i++)
                uHat[off + i] = (reflectors[off + i] / rowScale) * r;
            invNorm[j] = r / rowScale;
            active[j] = scaledSq > scaledFloor ? 1.0 : (scaledSq == scaledFloor ? 0.5 : 0.0);
        }
    }

    private static void NormalizeReflectorRowsDoubleCore(
        ReadOnlySpan<double> reflectors,
        int k,
        int d,
        double normFloor,
        double[] uHat,
        double[] invNorm,
        double[] active)
    {
        double sqrtFloor = Math.Sqrt(normFloor);
        double minScale = sqrtFloor * 0.0625;
        for (int j = 0; j < k; j++)
        {
            int off = j * d;
            double maxAbs = 0.0;
            for (int i = 0; i < d; i++)
            {
                double a = Math.Abs(reflectors[off + i]);
                if (a > maxAbs) maxAbs = a;
            }
            double rowScale = Math.Max(maxAbs, minScale);
            double scaledSq = 0.0;
            for (int i = 0; i < d; i++)
            {
                double sc = reflectors[off + i] / rowScale;
                scaledSq += sc * sc;
            }
            double sfr = sqrtFloor / rowScale;
            double scaledFloor = sfr * sfr;
            double denom = Math.Max(scaledSq, scaledFloor);
            double r = 1.0 / Math.Sqrt(denom);
            for (int i = 0; i < d; i++)
                uHat[off + i] = (reflectors[off + i] / rowScale) * r;
            invNorm[j] = r / rowScale;
            active[j] = scaledSq > scaledFloor ? 1.0 : (scaledSq == scaledFloor ? 0.5 : 0.0);
        }
    }

    private static void HouseholderChainForwardFloat(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> reflectors,
        Span<float> output,
        int rows,
        int k,
        int d,
        bool transpose,
        double normFloor)
    {
        var uHat = new double[k * d];
        var invNorm = new double[k];
        var active = new double[k];
        NormalizeReflectorRowsFloat(reflectors, k, d, normFloor, uHat, invNorm, active);

        var rowBuf = new double[d];
        for (int m = 0; m < rows; m++)
        {
            int mOff = m * d;
            for (int i = 0; i < d; i++) rowBuf[i] = input[mOff + i];
            for (int s = 0; s < k; s++)
            {
                int j = transpose ? (k - 1 - s) : s;
                int uOff = j * d;
                double dot = 0.0;
                for (int i = 0; i < d; i++) dot += rowBuf[i] * uHat[uOff + i];
                for (int i = 0; i < d; i++)
                {
                    double step = dot * uHat[uOff + i];
                    rowBuf[i] = (rowBuf[i] - step) - step;
                }
            }
            for (int i = 0; i < d; i++) output[mOff + i] = (float)rowBuf[i];
        }
    }

    private static void HouseholderChainForwardDouble(
        ReadOnlySpan<double> input,
        ReadOnlySpan<double> reflectors,
        Span<double> output,
        int rows,
        int k,
        int d,
        bool transpose,
        double normFloor)
    {
        var uHat = new double[k * d];
        var invNorm = new double[k];
        var active = new double[k];
        NormalizeReflectorRowsDoubleCore(reflectors, k, d, normFloor, uHat, invNorm, active);

        var rowBuf = new double[d];
        for (int m = 0; m < rows; m++)
        {
            int mOff = m * d;
            for (int i = 0; i < d; i++) rowBuf[i] = input[mOff + i];
            for (int s = 0; s < k; s++)
            {
                int j = transpose ? (k - 1 - s) : s;
                int uOff = j * d;
                double dot = 0.0;
                for (int i = 0; i < d; i++) dot += rowBuf[i] * uHat[uOff + i];
                for (int i = 0; i < d; i++)
                {
                    double step = dot * uHat[uOff + i];
                    rowBuf[i] = (rowBuf[i] - step) - step;
                }
            }
            for (int i = 0; i < d; i++) output[mOff + i] = rowBuf[i];
        }
    }

    private static void HouseholderChainBackwardFloat(
        ReadOnlySpan<float> gradOutput,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> reflectors,
        Span<float> gradInput,
        Span<float> gradReflectors,
        int rows,
        int k,
        int d,
        bool transpose,
        double normFloor)
    {
        var uHat = new double[k * d];
        var invNorm = new double[k];
        var active = new double[k];
        NormalizeReflectorRowsFloat(reflectors, k, d, normFloor, uHat, invNorm, active);

        var duHat = new double[k * d];
        var rx = new double[d];
        var rg = new double[d];
        var p = new double[k];
        var stages = new double[k * d];

        for (int m = 0; m < rows; m++)
        {
            int mOff = m * d;
            for (int i = 0; i < d; i++)
            {
                rx[i] = input[mOff + i];
                rg[i] = gradOutput[mOff + i];
            }
            AccumulateRowBackwardDouble(uHat, duHat, rx, rg, p, stages, k, d, transpose);
            for (int i = 0; i < d; i++) gradInput[mOff + i] = (float)rg[i];
        }

        for (int j = 0; j < k; j++)
        {
            int uOff = j * d;
            double dot = 0.0;
            for (int i = 0; i < d; i++) dot += duHat[uOff + i] * uHat[uOff + i];
            double inv = invNorm[j];
            double act = active[j];
            for (int i = 0; i < d; i++)
                gradReflectors[uOff + i] = (float)(inv * (duHat[uOff + i] - act * dot * uHat[uOff + i]));
        }
    }

    private static void HouseholderChainBackwardDouble(
        ReadOnlySpan<double> gradOutput,
        ReadOnlySpan<double> input,
        ReadOnlySpan<double> reflectors,
        Span<double> gradInput,
        Span<double> gradReflectors,
        int rows,
        int k,
        int d,
        bool transpose,
        double normFloor)
    {
        var uHat = new double[k * d];
        var invNorm = new double[k];
        var active = new double[k];
        NormalizeReflectorRowsDoubleCore(reflectors, k, d, normFloor, uHat, invNorm, active);

        var duHat = new double[k * d];
        var rx = new double[d];
        var rg = new double[d];
        var p = new double[k];
        var stages = new double[k * d];

        for (int m = 0; m < rows; m++)
        {
            int mOff = m * d;
            for (int i = 0; i < d; i++)
            {
                rx[i] = input[mOff + i];
                rg[i] = gradOutput[mOff + i];
            }
            AccumulateRowBackwardDouble(uHat, duHat, rx, rg, p, stages, k, d, transpose);
            for (int i = 0; i < d; i++) gradInput[mOff + i] = rg[i];
        }

        for (int j = 0; j < k; j++)
        {
            int uOff = j * d;
            double dot = 0.0;
            for (int i = 0; i < d; i++) dot += duHat[uOff + i] * uHat[uOff + i];
            double inv = invNorm[j];
            double act = active[j];
            for (int i = 0; i < d; i++)
                gradReflectors[uOff + i] = inv * (duHat[uOff + i] - act * dot * uHat[uOff + i]);
        }
    }

    private static void AccumulateRowBackwardDouble(
        double[] uHat,
        double[] duHat,
        double[] rx,
        double[] rg,
        double[] p,
        double[] stages,
        int k,
        int d,
        bool transpose)
    {
        for (int s = 0; s < k; s++)
        {
            int sOff = s * d;
            for (int i = 0; i < d; i++) stages[sOff + i] = rx[i];
            int j = transpose ? (k - 1 - s) : s;
            int uOff = j * d;
            double dot = 0.0;
            for (int i = 0; i < d; i++) dot += rx[i] * uHat[uOff + i];
            p[s] = dot;
            for (int i = 0; i < d; i++)
            {
                double step = dot * uHat[uOff + i];
                rx[i] = (rx[i] - step) - step;
            }
        }

        for (int s = k - 1; s >= 0; s--)
        {
            int sOff = s * d;
            int j = transpose ? (k - 1 - s) : s;
            int uOff = j * d;
            double ps = p[s];

            double q = 0.0;
            for (int i = 0; i < d; i++) q += rg[i] * uHat[uOff + i];

            for (int i = 0; i < d; i++)
            {
                double duTerm = q * stages[sOff + i] + ps * rg[i];
                duHat[uOff + i] = (duHat[uOff + i] - duTerm) - duTerm;
            }

            for (int i = 0; i < d; i++)
            {
                double stepQ = q * uHat[uOff + i];
                rg[i] = (rg[i] - stepQ) - stepQ;
            }
        }
    }
}
