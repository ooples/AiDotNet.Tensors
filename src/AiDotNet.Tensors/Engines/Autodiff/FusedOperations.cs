using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Fused forward+backward operations that share intermediate values
/// between forward and backward passes for optimal performance.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public static class FusedOperations<T>
{
    /// <summary>
    /// Fused linear layer forward: output = input @ weight + bias.
    /// Records a single tape entry with all three parameters for
    /// combined gradient computation.
    /// </summary>
    /// <param name="input">Input tensor [batch, inputSize].</param>
    /// <param name="weight">Weight matrix [inputSize, outputSize].</param>
    /// <param name="bias">Optional bias vector [outputSize].</param>
    /// <returns>Output tensor [batch, outputSize].</returns>
    public static Tensor<T> Linear(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias = null)
    {
        var engine = AiDotNetEngine.Current;

        // Forward: output = input @ weight
        var output = engine.TensorMatMul(input, weight);

        // Add bias if present
        if (bias is not null)
        {
            output = engine.TensorBroadcastAdd(output, bias);
        }

        // Record as a single fused op (instead of separate MatMul + Add)
        var inputs = bias is not null
            ? new[] { input, weight, bias }
            : new[] { input, weight };

        DifferentiableOps.RecordIfActive("FusedLinear", output, inputs,
            (gradOutput, inp, outp, savedState, eng, grads) =>
            {
                var x = inp[0];
                var w = inp[1];

                // dWeight = input^T @ gradOutput
                var inputT = eng.TensorTranspose(x);
                var gradWeight = eng.TensorMatMul(inputT, gradOutput);
                DifferentiableOps.AccumulateGrad(grads, w, gradWeight, eng);

                // dInput = gradOutput @ weight^T
                var weightT = eng.TensorTranspose(w);
                var gradInput = eng.TensorMatMul(gradOutput, weightT);
                DifferentiableOps.AccumulateGrad(grads, x, gradInput, eng);

                // dBias = sum(gradOutput, axis=0)
                if (inp.Length > 2)
                {
                    var b = inp[2];
                    var gradBias = eng.ReduceSum(gradOutput, new[] { 0 }, keepDims: false);
                    DifferentiableOps.AccumulateGrad(grads, b, gradBias, eng);
                }
            });

        return output;
    }

    /// <summary>
    /// Fused linear + ReLU: output = ReLU(input @ weight + bias).
    /// Records a single tape entry combining linear and activation.
    /// </summary>
    public static Tensor<T> LinearReLU(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias = null)
    {
        var engine = AiDotNetEngine.Current;

        // Forward
        var preActivation = engine.TensorMatMul(input, weight);
        if (bias is not null)
            preActivation = engine.TensorBroadcastAdd(preActivation, bias);
        var output = engine.ReLU(preActivation);

        // Record fused op
        var inputs = bias is not null
            ? new[] { input, weight, bias }
            : new[] { input, weight };

        DifferentiableOps.RecordIfActive("FusedLinearReLU", output, inputs,
            (gradOutput, inp, outp, savedState, eng, grads) =>
            {
                // ReLU backward: mask where preActivation > 0
                var reluGrad = eng.ReluBackward(gradOutput, (Tensor<T>)savedState![0]);

                var x = inp[0];
                var w = inp[1];

                var inputT = eng.TensorTranspose(x);
                var gradWeight = eng.TensorMatMul(inputT, reluGrad);
                DifferentiableOps.AccumulateGrad(grads, w, gradWeight, eng);

                var weightT = eng.TensorTranspose(w);
                var gradInput = eng.TensorMatMul(reluGrad, weightT);
                DifferentiableOps.AccumulateGrad(grads, x, gradInput, eng);

                if (inp.Length > 2)
                {
                    var gradBias = eng.ReduceSum(reluGrad, new[] { 0 }, keepDims: false);
                    DifferentiableOps.AccumulateGrad(grads, inp[2], gradBias, eng);
                }
            },
            savedState: new object[] { preActivation });

        return output;
    }
}
