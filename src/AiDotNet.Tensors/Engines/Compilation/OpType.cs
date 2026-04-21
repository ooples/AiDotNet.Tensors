namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Type-safe enum for all tensor operations, replacing error-prone string comparisons
/// in compiled plans. Each value maps 1:1 to the string OpName used in DifferentiableOps.Record*.
/// </summary>
internal enum OpType : byte
{
    Unknown = 0,

    // Elementwise binary
    TensorAdd,
    TensorSubtract,
    TensorMultiply,
    TensorDivide,
    TensorMax,

    // Broadcast binary
    TensorBroadcastAdd,
    TensorBroadcastSubtract,
    TensorBroadcastMultiply,

    // Unary math
    TensorExp,
    TensorLog,
    TensorSqrt,
    TensorAbs,
    TensorPower,
    TensorNegate,

    // Activations
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    Mish,
    ELU,
    SELU,
    Softmax,
    LogSoftmax,
    Softplus,
    HardSwish,
    HardSigmoid,

    // Unary element ops
    Sin,
    Cos,
    Sign,
    Reciprocal,
    Floor,
    Ceiling,
    Round,
    Clamp,

    // Reductions
    ReduceSum,
    ReduceMax,
    Mean,

    // Linear algebra
    TensorMatMul,
    TensorTranspose,

    // Convolution
    Conv2D,
    DepthwiseConv2D,
    ConvTranspose2D,

    // Normalization
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,

    // Pooling
    MaxPool2D,
    AvgPool2D,

    // Attention
    ScaledDotProductAttention,
    BatchMatMul,

    // Fused
    FusedLinear,

    // Loss
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropy,
}

internal static class OpTypeParser
{
    /// <summary>
    /// Parses a string operation name to its OpType enum value.
    /// Returns OpType.Unknown for unrecognized names.
    /// </summary>
    internal static OpType Parse(string opName) => opName switch
    {
        "TensorAdd" => OpType.TensorAdd,
        "TensorSubtract" => OpType.TensorSubtract,
        "TensorMultiply" => OpType.TensorMultiply,
        "TensorDivide" => OpType.TensorDivide,
        "TensorMax" => OpType.TensorMax,
        "TensorBroadcastAdd" => OpType.TensorBroadcastAdd,
        "TensorBroadcastSubtract" => OpType.TensorBroadcastSubtract,
        "TensorBroadcastMultiply" => OpType.TensorBroadcastMultiply,
        "TensorExp" => OpType.TensorExp,
        "TensorLog" => OpType.TensorLog,
        "TensorSqrt" => OpType.TensorSqrt,
        "TensorAbs" => OpType.TensorAbs,
        "TensorPower" => OpType.TensorPower,
        "TensorNegate" => OpType.TensorNegate,
        "ReLU" => OpType.ReLU,
        "LeakyReLU" => OpType.LeakyReLU,
        "Sigmoid" => OpType.Sigmoid,
        "Tanh" => OpType.Tanh,
        "GELU" => OpType.GELU,
        "Swish" => OpType.Swish,
        "Mish" => OpType.Mish,
        "ELU" => OpType.ELU,
        "SELU" => OpType.SELU,
        "Softmax" => OpType.Softmax,
        "LogSoftmax" => OpType.LogSoftmax,
        "Softplus" => OpType.Softplus,
        "HardSwish" => OpType.HardSwish,
        "HardSigmoid" => OpType.HardSigmoid,
        "Sin" => OpType.Sin,
        "Cos" => OpType.Cos,
        "Sign" => OpType.Sign,
        "Reciprocal" => OpType.Reciprocal,
        "Floor" => OpType.Floor,
        "Ceiling" => OpType.Ceiling,
        "Round" => OpType.Round,
        "Clamp" => OpType.Clamp,
        "ReduceSum" => OpType.ReduceSum,
        "ReduceMax" => OpType.ReduceMax,
        "Mean" or "ReduceMean" => OpType.Mean,
        "TensorMatMul" => OpType.TensorMatMul,
        "TensorTranspose" => OpType.TensorTranspose,
        "Conv2D" => OpType.Conv2D,
        "DepthwiseConv2D" => OpType.DepthwiseConv2D,
        "ConvTranspose2D" => OpType.ConvTranspose2D,
        "BatchNorm" or "BatchNormInference" => OpType.BatchNorm,
        "LayerNorm" => OpType.LayerNorm,
        "GroupNorm" => OpType.GroupNorm,
        "InstanceNorm" => OpType.InstanceNorm,
        "MaxPool2D" or "MaxPool2DWithIndices" => OpType.MaxPool2D,
        "AvgPool2D" => OpType.AvgPool2D,
        "ScaledDotProductAttention" or "TensorScaledDotProductAttention" => OpType.ScaledDotProductAttention,
        "BatchMatMul" or "TensorBatchMatMul" => OpType.BatchMatMul,
        "FusedLinear" => OpType.FusedLinear,
        "MSELoss" => OpType.MSELoss,
        "TensorCrossEntropyLoss" or "CrossEntropyLoss" => OpType.CrossEntropyLoss,
        "TensorBinaryCrossEntropy" => OpType.BinaryCrossEntropy,
        _ => OpType.Unknown,
    };
}
