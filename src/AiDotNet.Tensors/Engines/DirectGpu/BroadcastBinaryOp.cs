namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Element-wise binary operation of a broadcast routed through the engine's rank-generic path.</summary>
internal enum BroadcastBinaryOp
{
    /// <summary>a + b.</summary>
    Add,
    /// <summary>a - b.</summary>
    Subtract,
    /// <summary>a * b.</summary>
    Multiply,
    /// <summary>a / b.</summary>
    Divide,
}
