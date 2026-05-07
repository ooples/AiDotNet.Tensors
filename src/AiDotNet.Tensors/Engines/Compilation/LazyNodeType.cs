namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Types of operations in the lazy computation graph.
/// Maps to both CPU engine methods and GPU kernel types.
/// </summary>
internal enum LazyNodeType : byte
{
    // Linear algebra
    MatMul,
    BatchMatMul,
    Transpose,

    // Unary elementwise
    Negate,

    // Elementwise binary
    Add,
    Subtract,
    Multiply,
    Divide,

    // Broadcast binary
    BroadcastAdd,
    BroadcastSubtract,
    BroadcastMultiply,
    BroadcastDivide,

    // Scalar operations
    MultiplyScalar,
    DivideScalar,
    AddScalar,

    // Activations
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    LeakyReLU,
    ELU,
    Softmax,

    // Normalization
    LayerNorm,
    BatchNorm,
    GroupNorm,

    // Reductions
    ReduceSum,
    ReduceMean,
    Sum,
    Mean,

    // Convolution / Pooling
    Conv2D,
    DepthwiseConv2D,
    MaxPool2D,

    // Regularization
    Dropout,

    // Fused operations (already optimized — no further fusion needed)
    FusedLinear,
    FusedLinearReLU,
    FusedLinearGELU,
    FusedLinearSigmoid,
    FusedConvBNReLU,

    // Issue #301 fused kernels — emitted by graph-mode IFusionPattern
    // chain walkers for LoRA, DDIM, and sparse linear paths.
    FusedLoRA,
    FusedDDIMStep,
    FusedSparseLinear,

    // Issue #302 VSA primitives — content-addressable retrieval and
    // HRR algebra. Standalone ops, not fusion targets.
    HopfieldRetrieve,
    HrrBind,
    HrrUnbind,

    // Loss functions
    MSELoss,
    CrossEntropyLoss,

    // Other
    Reshape,
    Concat,
    Slice,
    Custom
}
