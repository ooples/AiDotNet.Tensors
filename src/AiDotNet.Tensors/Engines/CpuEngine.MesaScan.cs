using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> MesaScanForward<T>(
        Tensor<T> q,
        Tensor<T> k,
        Tensor<T> v,
        Tensor<T> initialWeights,
        T regularization,
        int numHeads)
    {
        ValidateMesaScan(q, k, v, initialWeights, regularization, numHeads,
            out int batch, out int time, out int model, out int headDim);

        var output = new Tensor<T>(new[] { batch, time, model });
        if (typeof(T) == typeof(float))
        {
            MesaForwardFloatStable(
                (float[])(object)q.GetDataArray()!, (float[])(object)k.GetDataArray()!,
                (float[])(object)v.GetDataArray()!, (float[])(object)initialWeights.GetDataArray()!,
                (float[])(object)output.GetDataArray()!, (float)(object)regularization!,
                batch, time, model, numHeads, headDim);
        }
        else
        {
            MesaForwardCore(
                q.GetDataArray()!, k.GetDataArray()!, v.GetDataArray()!,
                initialWeights.GetDataArray()!, output.GetDataArray()!, regularization,
                batch, time, model, numHeads, headDim);
        }

        DifferentiableOps.RecordIfActive<T>(
            "MesaScan", output,
            new[] { q, k, v, initialWeights },
            MesaScanBackward<T>,
            savedState: new object[] { regularization!, numHeads });

        return output;
    }

    private static void ValidateMesaScan<T>(
        Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> initialWeights,
        T regularization, int numHeads,
        out int batch, out int time, out int model, out int headDim)
    {
        if (q is null) throw new ArgumentNullException(nameof(q));
        if (k is null) throw new ArgumentNullException(nameof(k));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (initialWeights is null) throw new ArgumentNullException(nameof(initialWeights));
        if (q.Rank != 3)
            throw new ArgumentException("q must be rank-3 [batch,time,model].", nameof(q));
        EnsureSameShape(q, k, nameof(k));
        EnsureSameShape(q, v, nameof(v));
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");

        batch = q.Shape[0];
        time = q.Shape[1];
        model = q.Shape[2];
        if (batch <= 0 || time <= 0 || model <= 0 || model % numHeads != 0)
            throw new ArgumentException(
                $"q dimensions must be positive and model ({model}) divisible by numHeads ({numHeads}).",
                nameof(q));
        headDim = model / numHeads;

        if (initialWeights.Rank != 3 || initialWeights.Shape[0] != numHeads ||
            initialWeights.Shape[1] != headDim || initialWeights.Shape[2] != headDim)
            throw new ArgumentException(
                $"initialWeights must have shape [{numHeads},{headDim},{headDim}].",
                nameof(initialWeights));

        var ops = MathHelper.GetNumericOperations<T>();
        if (!ops.GreaterThan(regularization, ops.Zero))
            throw new ArgumentOutOfRangeException(nameof(regularization), "regularization must be positive.");
    }

    private static void MesaForwardCore<T>(
        T[] q, T[] k, T[] v, T[] w0, T[] output, T regularization,
        int batch, int time, int model, int heads, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T invLambda = ops.Divide(ops.One, regularization);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                int matrixSize = dim * dim;
                int w0Base = h * matrixSize;
                var weights = new T[matrixSize];
                var covariance = new T[matrixSize];
                Array.Copy(w0, w0Base, weights, 0, matrixSize);
                for (int i = 0; i < dim; i++) covariance[i * dim + i] = invLambda;

                var pk = new T[dim];
                var error = new T[dim];
                var row = new T[dim];
                for (int t = 0; t < time; t++)
                {
                    int vectorBase = (b * time + t) * model + h * dim;
                    for (int i = 0; i < dim; i++)
                    {
                        T sum = ops.Zero;
                        for (int j = 0; j < dim; j++)
                            sum = ops.Add(sum, ops.Multiply(covariance[i * dim + j], k[vectorBase + j]));
                        pk[i] = sum;
                    }

                    T denominator = ops.One;
                    for (int i = 0; i < dim; i++)
                        denominator = ops.Add(denominator, ops.Multiply(k[vectorBase + i], pk[i]));
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            covariance[i * dim + j] = ops.Subtract(
                                covariance[i * dim + j],
                                ops.Divide(ops.Multiply(pk[i], pk[j]), denominator));

                    for (int i = 0; i < dim; i++)
                    {
                        T prediction = ops.Zero;
                        for (int j = 0; j < dim; j++)
                            prediction = ops.Add(prediction, ops.Multiply(weights[i * dim + j], k[vectorBase + j]));
                        error[i] = ops.Subtract(prediction, v[vectorBase + i]);
                    }
                    for (int j = 0; j < dim; j++)
                    {
                        T sum = ops.Zero;
                        for (int i = 0; i < dim; i++)
                            sum = ops.Add(sum, ops.Multiply(k[vectorBase + i], covariance[i * dim + j]));
                        row[j] = sum;
                    }
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            weights[i * dim + j] = ops.Subtract(
                                weights[i * dim + j], ops.Multiply(error[i], row[j]));

                    for (int i = 0; i < dim; i++)
                    {
                        T sum = ops.Zero;
                        for (int j = 0; j < dim; j++)
                            sum = ops.Add(sum, ops.Multiply(weights[i * dim + j], q[vectorBase + j]));
                        output[vectorBase + i] = sum;
                    }
                }
            }
        }
    }

    private static void MesaForwardFloatStable(
        float[] q, float[] k, float[] v, float[] w0, float[] output, float regularization,
        int batch, int time, int model, int heads, int dim)
    {
        // Recursive least squares is especially sensitive when lambda is small (MesaNet defaults
        // to 0.01). Accumulating P/W updates in float can lose enough precision that a subsequent
        // projection amplifies the VJP error by orders of magnitude. Keep the public storage and
        // result float32, but perform the sequential state evolution in float64, like mixed-precision
        // optimizers keep their master state in FP32 for FP16 parameters.
        int matrixSize = dim * dim;
        double invLambda = 1.0 / regularization;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
            {
                int w0Base = h * matrixSize;
                var weights = new double[matrixSize];
                var covariance = new double[matrixSize];
                for (int i = 0; i < matrixSize; i++) weights[i] = w0[w0Base + i];
                for (int i = 0; i < dim; i++) covariance[i * dim + i] = invLambda;
                var pk = new double[dim];
                var error = new double[dim];
                var row = new double[dim];
                for (int t = 0; t < time; t++)
                {
                    int vectorBase = (b * time + t) * model + h * dim;
                    for (int i = 0; i < dim; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < dim; j++) sum += covariance[i * dim + j] * k[vectorBase + j];
                        pk[i] = sum;
                    }
                    double denominator = 1;
                    for (int i = 0; i < dim; i++) denominator += k[vectorBase + i] * pk[i];
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            covariance[i * dim + j] -= pk[i] * pk[j] / denominator;
                    for (int i = 0; i < dim; i++)
                    {
                        double prediction = 0;
                        for (int j = 0; j < dim; j++) prediction += weights[i * dim + j] * k[vectorBase + j];
                        error[i] = prediction - v[vectorBase + i];
                    }
                    for (int j = 0; j < dim; j++)
                    {
                        double sum = 0;
                        for (int i = 0; i < dim; i++) sum += k[vectorBase + i] * covariance[i * dim + j];
                        row[j] = sum;
                    }
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++) weights[i * dim + j] -= error[i] * row[j];
                    for (int i = 0; i < dim; i++)
                    {
                        double sum = 0;
                        for (int j = 0; j < dim; j++) sum += weights[i * dim + j] * q[vectorBase + j];
                        output[vectorBase + i] = (float)sum;
                    }
                }
            }
    }

    protected static void MesaScanBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var q = inputs[0];
        var k = inputs[1];
        var v = inputs[2];
        var w0 = inputs[3];
        T regularization = (T)savedState[0];
        int heads = (int)savedState[1];
        int batch = q.Shape[0], time = q.Shape[1], model = q.Shape[2], dim = model / heads;

        var dq = new Tensor<T>(q.Shape.ToArray());
        var dk = new Tensor<T>(k.Shape.ToArray());
        var dv = new Tensor<T>(v.Shape.ToArray());
        var dw0 = new Tensor<T>(w0.Shape.ToArray());
        if (typeof(T) == typeof(float))
        {
            MesaBackwardFloatStable(
                (float[])(object)gradOutput.GetDataArray()!, (float[])(object)q.GetDataArray()!,
                (float[])(object)k.GetDataArray()!, (float[])(object)v.GetDataArray()!,
                (float[])(object)w0.GetDataArray()!, (float[])(object)dq.GetDataArray()!,
                (float[])(object)dk.GetDataArray()!, (float[])(object)dv.GetDataArray()!,
                (float[])(object)dw0.GetDataArray()!, (float)(object)regularization!,
                batch, time, model, heads, dim);
        }
        else
        {
            MesaBackwardCore(
                gradOutput.GetDataArray()!, q.GetDataArray()!, k.GetDataArray()!, v.GetDataArray()!,
                w0.GetDataArray()!, dq.GetDataArray()!, dk.GetDataArray()!, dv.GetDataArray()!,
                dw0.GetDataArray()!, regularization, batch, time, model, heads, dim);
        }

        DifferentiableOps.AccumulateGrad(grads, q, dq, engine);
        DifferentiableOps.AccumulateGrad(grads, k, dk, engine);
        DifferentiableOps.AccumulateGrad(grads, v, dv, engine);
        DifferentiableOps.AccumulateGrad(grads, w0, dw0, engine);
    }

    private static void MesaBackwardCore<T>(
        T[] dy, T[] q, T[] k, T[] v, T[] w0,
        T[] dq, T[] dk, T[] dv, T[] dw0, T regularization,
        int batch, int time, int model, int heads, int dim)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T invLambda = ops.Divide(ops.One, regularization);
        int matrixSize = dim * dim;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                int w0Base = h * matrixSize;
                var weightTrajectory = new T[(time + 1) * matrixSize];
                var covarianceTrajectory = new T[(time + 1) * matrixSize];
                Array.Copy(w0, w0Base, weightTrajectory, 0, matrixSize);
                for (int i = 0; i < dim; i++) covarianceTrajectory[i * dim + i] = invLambda;

                var pk = new T[dim];
                var error = new T[dim];
                var row = new T[dim];
                for (int t = 0; t < time; t++)
                {
                    int vectorBase = (b * time + t) * model + h * dim;
                    int prev = t * matrixSize, current = prev + matrixSize;
                    for (int i = 0; i < dim; i++)
                    {
                        T sum = ops.Zero;
                        for (int j = 0; j < dim; j++)
                            sum = ops.Add(sum, ops.Multiply(covarianceTrajectory[prev + i * dim + j], k[vectorBase + j]));
                        pk[i] = sum;
                    }
                    T denominator = ops.One;
                    for (int i = 0; i < dim; i++) denominator = ops.Add(denominator, ops.Multiply(k[vectorBase + i], pk[i]));
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            covarianceTrajectory[current + i * dim + j] = ops.Subtract(
                                covarianceTrajectory[prev + i * dim + j],
                                ops.Divide(ops.Multiply(pk[i], pk[j]), denominator));
                    for (int i = 0; i < dim; i++)
                    {
                        T prediction = ops.Zero;
                        for (int j = 0; j < dim; j++)
                            prediction = ops.Add(prediction, ops.Multiply(weightTrajectory[prev + i * dim + j], k[vectorBase + j]));
                        error[i] = ops.Subtract(prediction, v[vectorBase + i]);
                    }
                    for (int j = 0; j < dim; j++)
                    {
                        T sum = ops.Zero;
                        for (int i = 0; i < dim; i++)
                            sum = ops.Add(sum, ops.Multiply(k[vectorBase + i], covarianceTrajectory[current + i * dim + j]));
                        row[j] = sum;
                    }
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            weightTrajectory[current + i * dim + j] = ops.Subtract(
                                weightTrajectory[prev + i * dim + j], ops.Multiply(error[i], row[j]));
                }

                var adjW = new T[matrixSize];
                var adjP = new T[matrixSize];
                var adjPk = new T[dim];
                var adjError = new T[dim];
                var adjRow = new T[dim];
                for (int t = time - 1; t >= 0; t--)
                {
                    int vectorBase = (b * time + t) * model + h * dim;
                    int prev = t * matrixSize, current = prev + matrixSize;
                    Array.Clear(adjPk, 0, dim);
                    Array.Clear(adjError, 0, dim);
                    Array.Clear(adjRow, 0, dim);

                    // y = W_t q
                    for (int i = 0; i < dim; i++)
                    {
                        T g = dy[vectorBase + i];
                        for (int j = 0; j < dim; j++)
                        {
                            adjW[i * dim + j] = ops.Add(adjW[i * dim + j], ops.Multiply(g, q[vectorBase + j]));
                            dq[vectorBase + j] = ops.Add(dq[vectorBase + j], ops.Multiply(weightTrajectory[current + i * dim + j], g));
                        }
                    }

                    // Recompute forward intermediates for this step.
                    for (int i = 0; i < dim; i++)
                    {
                        T pkValue = ops.Zero;
                        T prediction = ops.Zero;
                        for (int j = 0; j < dim; j++)
                        {
                            pkValue = ops.Add(pkValue, ops.Multiply(covarianceTrajectory[prev + i * dim + j], k[vectorBase + j]));
                            prediction = ops.Add(prediction, ops.Multiply(weightTrajectory[prev + i * dim + j], k[vectorBase + j]));
                        }
                        pk[i] = pkValue;
                        error[i] = ops.Subtract(prediction, v[vectorBase + i]);
                    }
                    T denominator = ops.One;
                    for (int i = 0; i < dim; i++) denominator = ops.Add(denominator, ops.Multiply(k[vectorBase + i], pk[i]));
                    for (int j = 0; j < dim; j++)
                    {
                        T rowValue = ops.Zero;
                        for (int i = 0; i < dim; i++)
                            rowValue = ops.Add(rowValue, ops.Multiply(k[vectorBase + i], covarianceTrajectory[current + i * dim + j]));
                        row[j] = rowValue;
                    }

                    // W_t = W_(t-1) - error row
                    var adjWPrev = new T[matrixSize];
                    Array.Copy(adjW, adjWPrev, matrixSize);
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                        {
                            adjError[i] = ops.Subtract(adjError[i], ops.Multiply(adjW[i * dim + j], row[j]));
                            adjRow[j] = ops.Subtract(adjRow[j], ops.Multiply(adjW[i * dim + j], error[i]));
                        }

                    // row = k^T P_t
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                        {
                            dk[vectorBase + i] = ops.Add(dk[vectorBase + i],
                                ops.Multiply(covarianceTrajectory[current + i * dim + j], adjRow[j]));
                            adjP[i * dim + j] = ops.Add(adjP[i * dim + j],
                                ops.Multiply(k[vectorBase + i], adjRow[j]));
                        }

                    // error = W_(t-1) k - v
                    for (int i = 0; i < dim; i++)
                    {
                        dv[vectorBase + i] = ops.Subtract(dv[vectorBase + i], adjError[i]);
                        for (int j = 0; j < dim; j++)
                        {
                            adjWPrev[i * dim + j] = ops.Add(adjWPrev[i * dim + j],
                                ops.Multiply(adjError[i], k[vectorBase + j]));
                            dk[vectorBase + j] = ops.Add(dk[vectorBase + j],
                                ops.Multiply(weightTrajectory[prev + i * dim + j], adjError[i]));
                        }
                    }

                    // P_t = P_(t-1) - pk pk^T / denominator
                    var adjPPrev = new T[matrixSize];
                    Array.Copy(adjP, adjPPrev, matrixSize);
                    T adjDenominator = ops.Zero;
                    T denominatorSquared = ops.Multiply(denominator, denominator);
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                        {
                            T g = adjP[i * dim + j];
                            T adjOuter = ops.Divide(ops.Negate(g), denominator);
                            adjPk[i] = ops.Add(adjPk[i], ops.Multiply(adjOuter, pk[j]));
                            adjPk[j] = ops.Add(adjPk[j], ops.Multiply(adjOuter, pk[i]));
                            adjDenominator = ops.Add(adjDenominator,
                                ops.Divide(ops.Multiply(g, ops.Multiply(pk[i], pk[j])), denominatorSquared));
                        }

                    // denominator = 1 + k^T pk
                    for (int i = 0; i < dim; i++)
                    {
                        dk[vectorBase + i] = ops.Add(dk[vectorBase + i], ops.Multiply(adjDenominator, pk[i]));
                        adjPk[i] = ops.Add(adjPk[i], ops.Multiply(adjDenominator, k[vectorBase + i]));
                    }

                    // pk = P_(t-1) k
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                        {
                            adjPPrev[i * dim + j] = ops.Add(adjPPrev[i * dim + j],
                                ops.Multiply(adjPk[i], k[vectorBase + j]));
                            dk[vectorBase + j] = ops.Add(dk[vectorBase + j],
                                ops.Multiply(covarianceTrajectory[prev + i * dim + j], adjPk[i]));
                        }

                    adjW = adjWPrev;
                    adjP = adjPPrev;
                }

                for (int i = 0; i < matrixSize; i++)
                    dw0[w0Base + i] = ops.Add(dw0[w0Base + i], adjW[i]);
            }
        }
    }

    private static void MesaBackwardFloatStable(
        float[] dy, float[] q, float[] k, float[] v, float[] w0,
        float[] dq, float[] dk, float[] dv, float[] dw0, float regularization,
        int batch, int time, int model, int heads, int dim)
    {
        static double[] ToDouble(float[] source)
        {
            var result = new double[source.Length];
            for (int i = 0; i < source.Length; i++) result[i] = source[i];
            return result;
        }

        var dq64 = new double[dq.Length];
        var dk64 = new double[dk.Length];
        var dv64 = new double[dv.Length];
        var dw064 = new double[dw0.Length];
        MesaBackwardCore(
            ToDouble(dy), ToDouble(q), ToDouble(k), ToDouble(v), ToDouble(w0),
            dq64, dk64, dv64, dw064, (double)regularization,
            batch, time, model, heads, dim);
        for (int i = 0; i < dq.Length; i++) dq[i] = (float)dq64[i];
        for (int i = 0; i < dk.Length; i++) dk[i] = (float)dk64[i];
        for (int i = 0; i < dv.Length; i++) dv[i] = (float)dv64[i];
        for (int i = 0; i < dw0.Length; i++) dw0[i] = (float)dw064[i];
    }
}
