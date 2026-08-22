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
    public virtual Tensor<T> RoutedDiagonalSsmScanForward<T>(
        Tensor<T> input, Tensor<T> activeMask, Tensor<T> transition,
        Tensor<T> inputMap, Tensor<T> outputMap, Tensor<T> skip)
    {
        ValidateRoutedDiagonalSsm(input, activeMask, transition, inputMap, outputMap, skip,
            out int batch, out int time, out int model, out int experts, out int state);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var capturedInput = input;
            var capturedActiveMask = activeMask;
            var capturedTransition = transition;
            var capturedInputMap = inputMap;
            var capturedOutputMap = outputMap;
            var capturedSkip = skip;
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "RoutedDiagonalSsmScanForward",
                new[] { input, activeMask, transition, inputMap, outputMap, skip },
                new[] { batch, time, experts, model },
                (eng, output) =>
                {
                    var result = eng.RoutedDiagonalSsmScanForward(
                        capturedInput, capturedActiveMask, capturedTransition, capturedInputMap,
                        capturedOutputMap, capturedSkip);
                    DirectGpuTensorEngine.CopyResultInto(eng, result, output);
                },
                RoutedDiagonalSsmBackward<T>,
                savedState: null);
        }
        var output = new Tensor<T>(new[] { batch, time, experts, model });
        RoutedDiagonalSsmForwardCore(
            input.GetDataArray()!, activeMask.GetDataArray()!, transition.GetDataArray()!,
            inputMap.GetDataArray()!, outputMap.GetDataArray()!, skip.GetDataArray()!,
            output.GetDataArray()!, batch, time, model, experts, state, null);
        DifferentiableOps.RecordIfActive(
            "RoutedDiagonalSsmScanForward", output,
            new[] { input, activeMask, transition, inputMap, outputMap, skip },
            RoutedDiagonalSsmBackward<T>);
        return output;
    }

    private static void ValidateRoutedDiagonalSsm<T>(
        Tensor<T> input, Tensor<T> mask, Tensor<T> transition,
        Tensor<T> inputMap, Tensor<T> outputMap, Tensor<T> skip,
        out int batch, out int time, out int model, out int experts, out int state)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (transition is null) throw new ArgumentNullException(nameof(transition));
        if (inputMap is null) throw new ArgumentNullException(nameof(inputMap));
        if (outputMap is null) throw new ArgumentNullException(nameof(outputMap));
        if (skip is null) throw new ArgumentNullException(nameof(skip));
        if (input.Rank != 3) throw new ArgumentException("input must be [batch,time,model].", nameof(input));
        batch = input.Shape[0]; time = input.Shape[1]; model = input.Shape[2];
        if (mask.Rank != 3 || mask.Shape[0] != batch || mask.Shape[1] != time)
            throw new ArgumentException("activeMask must be [batch,time,experts].", nameof(mask));
        experts = mask.Shape[2];
        if (transition.Rank != 2 || transition.Shape[0] != experts)
            throw new ArgumentException("transition must be [experts,state].", nameof(transition));
        state = transition.Shape[1];
        if (batch <= 0 || time <= 0 || model <= 0 || experts <= 0 || state <= 0)
            throw new ArgumentException("Routed diagonal SSM dimensions must be positive.");
        if (inputMap.Rank != 3 || inputMap.Shape[0] != experts || inputMap.Shape[1] != state || inputMap.Shape[2] != model)
            throw new ArgumentException("inputMap must be [experts,state,model].", nameof(inputMap));
        if (outputMap.Rank != 3 || outputMap.Shape[0] != experts || outputMap.Shape[1] != model || outputMap.Shape[2] != state)
            throw new ArgumentException("outputMap must be [experts,model,state].", nameof(outputMap));
        if (skip.Rank != 2 || skip.Shape[0] != experts || skip.Shape[1] != model)
            throw new ArgumentException("skip must be [experts,model].", nameof(skip));
    }

    private static void RoutedDiagonalSsmForwardCore<T>(
        T[] input, T[] mask, T[] transition, T[] inputMap, T[] outputMap, T[] skip,
        T[]? output, int batch, int time, int model, int experts, int state, T[]? trajectory)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int b = 0; b < batch; b++)
            for (int e = 0; e < experts; e++)
            {
                var h = new T[state];
                int trajectoryBase = (b * experts + e) * (time + 1) * state;
                for (int t = 0; t < time; t++)
                {
                    int xBase = (b * time + t) * model;
                    int maskIndex = (b * time + t) * experts + e;
                    T active = mask[maskIndex];
                    int aBase = e * state, bBase = e * state * model;
                    for (int s = 0; s < state; s++)
                    {
                        T next = ops.Multiply(transition[aBase + s], h[s]);
                        for (int d = 0; d < model; d++)
                            next = ops.Add(next, ops.Multiply(inputMap[bBase + s * model + d], input[xBase + d]));
                        // State gating is multiplicative on the WHOLE recurrence:
                        // h[s] = active * (A_s*h[s] + Bx). An inactive expert therefore
                        // CLEARS its state and breaks the chain - it does not preserve
                        // the previous h. All six GPU kernels use this same form, and
                        // parity depends on every backend keeping it.
                        h[s] = ops.Multiply(active, next);
                        if (trajectory is not null) trajectory[trajectoryBase + (t + 1) * state + s] = h[s];
                    }
                    int yBase = ((b * time + t) * experts + e) * model;
                    int cBase = e * model * state, dBase = e * model;
                    for (int d = 0; d < model; d++)
                    {
                        T y = ops.Multiply(skip[dBase + d], input[xBase + d]);
                        for (int s = 0; s < state; s++)
                            y = ops.Add(y, ops.Multiply(outputMap[cBase + d * state + s], h[s]));
                        if (output is not null) output[yBase + d] = ops.Multiply(active, y);
                    }
                }
            }
    }

    protected static void RoutedDiagonalSsmBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output, object[] savedState,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0]; var mask = inputs[1]; var transition = inputs[2];
        var inputMap = inputs[3]; var outputMap = inputs[4]; var skip = inputs[5];
        int batch = input.Shape[0], time = input.Shape[1], model = input.Shape[2];
        int experts = mask.Shape[2], state = transition.Shape[1];
        var dx = new Tensor<T>(input.Shape.ToArray()); var dm = new Tensor<T>(mask.Shape.ToArray());
        var da = new Tensor<T>(transition.Shape.ToArray()); var db = new Tensor<T>(inputMap.Shape.ToArray());
        var dc = new Tensor<T>(outputMap.Shape.ToArray()); var dd = new Tensor<T>(skip.Shape.ToArray());
        var trajectory = new T[batch * experts * (time + 1) * state];
        RoutedDiagonalSsmForwardCore(
            input.GetDataArray()!, mask.GetDataArray()!, transition.GetDataArray()!,
            inputMap.GetDataArray()!, outputMap.GetDataArray()!, skip.GetDataArray()!,
            // The backward replays the forward only to rebuild `trajectory`; the
            // outputs are discarded, so do not allocate a batch*time*experts*model
            // array to throw away.
            null, batch, time, model, experts, state, trajectory);
        RoutedDiagonalSsmBackwardCore(
            gradOutput.GetDataArray()!, input.GetDataArray()!, mask.GetDataArray()!, transition.GetDataArray()!,
            inputMap.GetDataArray()!, outputMap.GetDataArray()!, skip.GetDataArray()!, trajectory,
            dx.GetDataArray()!, dm.GetDataArray()!, da.GetDataArray()!, db.GetDataArray()!, dc.GetDataArray()!, dd.GetDataArray()!,
            batch, time, model, experts, state);
        Tensor<T>[] computed = { dx, dm, da, db, dc, dd };
        for (int i = 0; i < inputs.Length; i++) DifferentiableOps.AccumulateGrad(grads, inputs[i], computed[i], engine);
    }

    private static void RoutedDiagonalSsmBackwardCore<T>(
        T[] dy, T[] input, T[] mask, T[] transition, T[] inputMap, T[] outputMap, T[] skip, T[] trajectory,
        T[] dx, T[] dMask, T[] dTransition, T[] dInputMap, T[] dOutputMap, T[] dSkip,
        int batch, int time, int model, int experts, int state)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int b = 0; b < batch; b++)
            for (int e = 0; e < experts; e++)
            {
                var adjH = new T[state];
                int trajectoryBase = (b * experts + e) * (time + 1) * state;
                int aBase = e * state, bBase = e * state * model, cBase = e * model * state, dBase = e * model;
                for (int t = time - 1; t >= 0; t--)
                {
                    int xBase = (b * time + t) * model, maskIndex = (b * time + t) * experts + e;
                    int yBase = ((b * time + t) * experts + e) * model;
                    T active = mask[maskIndex], adjMask = ops.Zero;
                    for (int d = 0; d < model; d++)
                    {
                        T yUnmasked = ops.Multiply(skip[dBase + d], input[xBase + d]);
                        for (int s = 0; s < state; s++)
                            yUnmasked = ops.Add(yUnmasked, ops.Multiply(outputMap[cBase + d * state + s], trajectory[trajectoryBase + (t + 1) * state + s]));
                        T g = dy[yBase + d];
                        adjMask = ops.Add(adjMask, ops.Multiply(g, yUnmasked));
                        T scaled = ops.Multiply(g, active);
                        dSkip[dBase + d] = ops.Add(dSkip[dBase + d], ops.Multiply(scaled, input[xBase + d]));
                        dx[xBase + d] = ops.Add(dx[xBase + d], ops.Multiply(scaled, skip[dBase + d]));
                        for (int s = 0; s < state; s++)
                        {
                            dOutputMap[cBase + d * state + s] = ops.Add(dOutputMap[cBase + d * state + s],
                                ops.Multiply(scaled, trajectory[trajectoryBase + (t + 1) * state + s]));
                            adjH[s] = ops.Add(adjH[s], ops.Multiply(scaled, outputMap[cBase + d * state + s]));
                        }
                    }
                    var adjPrevious = new T[state];
                    for (int s = 0; s < state; s++)
                    {
                        T preMask = ops.Multiply(transition[aBase + s], trajectory[trajectoryBase + t * state + s]);
                        for (int d = 0; d < model; d++)
                            preMask = ops.Add(preMask, ops.Multiply(inputMap[bBase + s * model + d], input[xBase + d]));
                        adjMask = ops.Add(adjMask, ops.Multiply(adjH[s], preMask));
                        T g = ops.Multiply(adjH[s], active);
                        dTransition[aBase + s] = ops.Add(dTransition[aBase + s],
                            ops.Multiply(g, trajectory[trajectoryBase + t * state + s]));
                        adjPrevious[s] = ops.Multiply(g, transition[aBase + s]);
                        for (int d = 0; d < model; d++)
                        {
                            dInputMap[bBase + s * model + d] = ops.Add(dInputMap[bBase + s * model + d],
                                ops.Multiply(g, input[xBase + d]));
                            dx[xBase + d] = ops.Add(dx[xBase + d], ops.Multiply(g, inputMap[bBase + s * model + d]));
                        }
                    }
                    dMask[maskIndex] = ops.Add(dMask[maskIndex], adjMask);
                    adjH = adjPrevious;
                }
            }
    }
}
