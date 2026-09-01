using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> ComplexDiagonalSsmScanForward<T>(
        Tensor<T> input,
        Tensor<T> transitionReal,
        Tensor<T> transitionImag,
        Tensor<T> inputMapReal,
        Tensor<T> inputMapImag,
        Tensor<T> outputMapReal,
        Tensor<T> outputMapImag,
        Tensor<T> skip)
    {
        ValidateComplexDiagonalSsmScan(
            input, transitionReal, transitionImag, inputMapReal, inputMapImag,
            outputMapReal, outputMapImag, skip,
            out int batch, out int time, out int groups, out int width, out int state);

        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var capturedInput = input;
            var capturedTransitionReal = transitionReal;
            var capturedTransitionImag = transitionImag;
            var capturedInputMapReal = inputMapReal;
            var capturedInputMapImag = inputMapImag;
            var capturedOutputMapReal = outputMapReal;
            var capturedOutputMapImag = outputMapImag;
            var capturedSkip = skip;
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "ComplexDiagonalSsmScan",
                new[]
                {
                    input, transitionReal, transitionImag, inputMapReal, inputMapImag,
                    outputMapReal, outputMapImag, skip
                },
                new[] { batch, time, groups, width },
                (eng, output) =>
                {
                    var result = eng.ComplexDiagonalSsmScanForward(
                        capturedInput, capturedTransitionReal, capturedTransitionImag,
                        capturedInputMapReal, capturedInputMapImag, capturedOutputMapReal,
                        capturedOutputMapImag, capturedSkip);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                ComplexDiagonalSsmScanBackward<T>,
                savedState: null);
        }

        var output = new Tensor<T>(new[] { batch, time, groups, width });
        ComplexDiagonalSsmScanForwardCore(
            input.GetDataArray()!, transitionReal.GetDataArray()!, transitionImag.GetDataArray()!,
            inputMapReal.GetDataArray()!, inputMapImag.GetDataArray()!,
            outputMapReal.GetDataArray()!, outputMapImag.GetDataArray()!, skip.GetDataArray()!,
            output.GetDataArray()!, batch, time, groups, width, state);

        DifferentiableOps.RecordIfActive<T>(
            "ComplexDiagonalSsmScan", output,
            new[]
            {
                input, transitionReal, transitionImag, inputMapReal, inputMapImag,
                outputMapReal, outputMapImag, skip
            },
            ComplexDiagonalSsmScanBackward<T>,
            savedState: null);

        return output;
    }

    private static void ValidateComplexDiagonalSsmScan<T>(
        Tensor<T> input,
        Tensor<T> transitionReal,
        Tensor<T> transitionImag,
        Tensor<T> inputMapReal,
        Tensor<T> inputMapImag,
        Tensor<T> outputMapReal,
        Tensor<T> outputMapImag,
        Tensor<T> skip,
        out int batch,
        out int time,
        out int groups,
        out int width,
        out int state)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (transitionReal is null) throw new ArgumentNullException(nameof(transitionReal));
        if (transitionImag is null) throw new ArgumentNullException(nameof(transitionImag));
        if (inputMapReal is null) throw new ArgumentNullException(nameof(inputMapReal));
        if (inputMapImag is null) throw new ArgumentNullException(nameof(inputMapImag));
        if (outputMapReal is null) throw new ArgumentNullException(nameof(outputMapReal));
        if (outputMapImag is null) throw new ArgumentNullException(nameof(outputMapImag));
        if (skip is null) throw new ArgumentNullException(nameof(skip));

        if (input.Rank != 4)
            throw new ArgumentException(
                $"input must be rank-4 [batch,time,group,width]; got rank {input.Rank}.", nameof(input));

        batch = input.Shape[0];
        time = input.Shape[1];
        groups = input.Shape[2];
        width = input.Shape[3];
        if (batch < 1 || time < 1 || groups < 1 || width < 1)
            throw new ArgumentException("All input dimensions must be positive.", nameof(input));

        if (transitionReal.Rank != 2 || transitionReal.Shape[0] != groups || transitionReal.Shape[1] < 1)
            throw new ArgumentException(
                $"transitionReal must be [group={groups},state>0].", nameof(transitionReal));
        state = transitionReal.Shape[1];
        EnsureShape(transitionImag, new[] { groups, state }, nameof(transitionImag));
        EnsureShape(inputMapReal, new[] { groups, state, width }, nameof(inputMapReal));
        EnsureShape(inputMapImag, new[] { groups, state, width }, nameof(inputMapImag));
        EnsureShape(outputMapReal, new[] { groups, width, state }, nameof(outputMapReal));
        EnsureShape(outputMapImag, new[] { groups, width, state }, nameof(outputMapImag));
        EnsureShape(skip, new[] { groups, width }, nameof(skip));
    }

    private static void EnsureShape<T>(Tensor<T> tensor, int[] expected, string paramName)
    {
        if (tensor.Rank != expected.Length)
            throw new ArgumentException(
                $"{paramName} must have shape [{string.Join(",", expected)}]; got rank {tensor.Rank}.", paramName);
        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
                throw new ArgumentException(
                    $"{paramName} must have shape [{string.Join(",", expected)}]; " +
                    $"dimension {i} was {tensor.Shape[i]}.", paramName);
        }
    }

    private static void ComplexDiagonalSsmScanForwardCore<T>(
        T[] x, T[] ar, T[] ai, T[] br, T[] bi, T[] cr, T[] ci, T[] d, T[] y,
        int batch, int time, int groups, int width, int state)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var hr = new T[state];
        var hi = new T[state];

        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < groups; g++)
            {
                Array.Clear(hr, 0, state);
                Array.Clear(hi, 0, state);
                int aBase = g * state;
                int mapBase = g * state * width;
                int outMapBase = g * width * state;
                int dBase = g * width;

                for (int t = 0; t < time; t++)
                {
                    int xBase = ((b * time + t) * groups + g) * width;
                    for (int n = 0; n < state; n++)
                    {
                        T oldR = hr[n];
                        T oldI = hi[n];
                        T nextR = ops.Subtract(
                            ops.Multiply(ar[aBase + n], oldR),
                            ops.Multiply(ai[aBase + n], oldI));
                        T nextI = ops.Add(
                            ops.Multiply(ar[aBase + n], oldI),
                            ops.Multiply(ai[aBase + n], oldR));
                        int bn = mapBase + n * width;
                        for (int w = 0; w < width; w++)
                        {
                            T xv = x[xBase + w];
                            nextR = ops.Add(nextR, ops.Multiply(br[bn + w], xv));
                            nextI = ops.Add(nextI, ops.Multiply(bi[bn + w], xv));
                        }
                        hr[n] = nextR;
                        hi[n] = nextI;
                    }

                    for (int w = 0; w < width; w++)
                    {
                        T value = ops.Multiply(d[dBase + w], x[xBase + w]);
                        int cn = outMapBase + w * state;
                        for (int n = 0; n < state; n++)
                        {
                            value = ops.Add(value, ops.Subtract(
                                ops.Multiply(cr[cn + n], hr[n]),
                                ops.Multiply(ci[cn + n], hi[n])));
                        }
                        y[xBase + w] = value;
                    }
                }
            }
        }
    }

    private static void ComplexDiagonalSsmScanBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var xTensor = inputs[0];
        int batch = xTensor.Shape[0];
        int time = xTensor.Shape[1];
        int groups = xTensor.Shape[2];
        int width = xTensor.Shape[3];
        int state = inputs[1].Shape[1];

        var gradients = new Tensor<T>[8];
        for (int i = 0; i < gradients.Length; i++)
            gradients[i] = new Tensor<T>(inputs[i].Shape.ToArray());

        ComplexDiagonalSsmScanBackwardCore(
            gradOutput.GetDataArray()!,
            inputs[0].GetDataArray()!, inputs[1].GetDataArray()!, inputs[2].GetDataArray()!,
            inputs[3].GetDataArray()!, inputs[4].GetDataArray()!, inputs[5].GetDataArray()!,
            inputs[6].GetDataArray()!, inputs[7].GetDataArray()!,
            gradients[0].GetDataArray()!, gradients[1].GetDataArray()!, gradients[2].GetDataArray()!,
            gradients[3].GetDataArray()!, gradients[4].GetDataArray()!, gradients[5].GetDataArray()!,
            gradients[6].GetDataArray()!, gradients[7].GetDataArray()!,
            batch, time, groups, width, state);

        for (int i = 0; i < gradients.Length; i++)
            DifferentiableOps.AccumulateGrad(grads, inputs[i], gradients[i], engine);
    }

    private static void ComplexDiagonalSsmScanBackwardCore<T>(
        T[] dy, T[] x, T[] ar, T[] ai, T[] br, T[] bi, T[] cr, T[] ci, T[] d,
        T[] dx, T[] dar, T[] dai, T[] dbr, T[] dbi, T[] dcr, T[] dci, T[] dd,
        int batch, int time, int groups, int width, int state)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var statesR = new T[(time + 1) * state];
        var statesI = new T[(time + 1) * state];
        var adjR = new T[state];
        var adjI = new T[state];

        for (int b = 0; b < batch; b++)
        {
            for (int g = 0; g < groups; g++)
            {
                Array.Clear(statesR, 0, statesR.Length);
                Array.Clear(statesI, 0, statesI.Length);
                Array.Clear(adjR, 0, state);
                Array.Clear(adjI, 0, state);
                int aBase = g * state;
                int mapBase = g * state * width;
                int outMapBase = g * width * state;
                int dBase = g * width;

                // Recompute states once; keeping only this group bounds memory at O(time*state).
                for (int t = 0; t < time; t++)
                {
                    int xBase = ((b * time + t) * groups + g) * width;
                    int prev = t * state;
                    int next = prev + state;
                    for (int n = 0; n < state; n++)
                    {
                        T nextR = ops.Subtract(
                            ops.Multiply(ar[aBase + n], statesR[prev + n]),
                            ops.Multiply(ai[aBase + n], statesI[prev + n]));
                        T nextI = ops.Add(
                            ops.Multiply(ar[aBase + n], statesI[prev + n]),
                            ops.Multiply(ai[aBase + n], statesR[prev + n]));
                        int bn = mapBase + n * width;
                        for (int w = 0; w < width; w++)
                        {
                            nextR = ops.Add(nextR, ops.Multiply(br[bn + w], x[xBase + w]));
                            nextI = ops.Add(nextI, ops.Multiply(bi[bn + w], x[xBase + w]));
                        }
                        statesR[next + n] = nextR;
                        statesI[next + n] = nextI;
                    }
                }

                for (int t = time - 1; t >= 0; t--)
                {
                    int xBase = ((b * time + t) * groups + g) * width;
                    int prev = t * state;
                    int current = prev + state;

                    for (int w = 0; w < width; w++)
                    {
                        T grad = dy[xBase + w];
                        dd[dBase + w] = ops.Add(dd[dBase + w], ops.Multiply(grad, x[xBase + w]));
                        dx[xBase + w] = ops.Add(dx[xBase + w], ops.Multiply(grad, d[dBase + w]));
                        int cn = outMapBase + w * state;
                        for (int n = 0; n < state; n++)
                        {
                            adjR[n] = ops.Add(adjR[n], ops.Multiply(cr[cn + n], grad));
                            adjI[n] = ops.Subtract(adjI[n], ops.Multiply(ci[cn + n], grad));
                            dcr[cn + n] = ops.Add(dcr[cn + n], ops.Multiply(grad, statesR[current + n]));
                            dci[cn + n] = ops.Subtract(dci[cn + n], ops.Multiply(grad, statesI[current + n]));
                        }
                    }

                    for (int n = 0; n < state; n++)
                    {
                        T gr = adjR[n];
                        T gi = adjI[n];
                        T oldR = statesR[prev + n];
                        T oldI = statesI[prev + n];
                        int an = aBase + n;
                        dar[an] = ops.Add(dar[an], ops.Add(ops.Multiply(gr, oldR), ops.Multiply(gi, oldI)));
                        dai[an] = ops.Add(dai[an], ops.Subtract(ops.Multiply(gi, oldR), ops.Multiply(gr, oldI)));

                        int bn = mapBase + n * width;
                        for (int w = 0; w < width; w++)
                        {
                            T xv = x[xBase + w];
                            dbr[bn + w] = ops.Add(dbr[bn + w], ops.Multiply(gr, xv));
                            dbi[bn + w] = ops.Add(dbi[bn + w], ops.Multiply(gi, xv));
                            dx[xBase + w] = ops.Add(dx[xBase + w], ops.Add(
                                ops.Multiply(br[bn + w], gr), ops.Multiply(bi[bn + w], gi)));
                        }

                        adjR[n] = ops.Add(ops.Multiply(ar[an], gr), ops.Multiply(ai[an], gi));
                        adjI[n] = ops.Subtract(ops.Multiply(ar[an], gi), ops.Multiply(ai[an], gr));
                    }
                }
            }
        }
    }
}
