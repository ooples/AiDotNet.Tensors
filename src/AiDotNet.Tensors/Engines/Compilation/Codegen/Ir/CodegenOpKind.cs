// Copyright (c) AiDotNet. All rights reserved.
// Codegen IR op taxonomy. Each emitter dispatches on CodegenOpKind to
// produce its target language — the enum is the cross-emitter
// contract, and any op that doesn't appear here cannot be codegen'd
// (the lowering pass emits a passthrough CompiledStep for unsupported
// ops, preserving correctness at the cost of not fusing them).

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Primitive op kinds the codegen IR can represent. Grouped into
/// four semantic categories that map to four different emission
/// strategies:
/// <list type="bullet">
/// <item><b>Pointwise</b> — one output element per input element,
/// no cross-element dependency. Fuses trivially into a single loop
/// (or a single SIMD pass, or a single GPU thread-per-element
/// kernel).</item>
/// <item><b>Reduction</b> — collapses one or more dimensions. Fuses
/// with upstream pointwise into a map-reduce loop; fuses with
/// downstream pointwise into an epilogue.</item>
/// <item><b>Matmul</b> — <c>C = A · B</c> (+ transpose variants).
/// Emitters produce tiled GEMM kernels; bias + activation fold
/// into the epilogue.</item>
/// <item><b>Attention</b> — softmax(Q · Kᵀ / √d) · V. Its own
/// category because the memory traffic is qualitatively different
/// (flash-attention / splitKV).</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>How this relates to
/// <see cref="LazyNodeType"/>:</b></para>
/// <para>
/// <see cref="LazyNodeType"/> enumerates the engine's op surface
/// (<c>Add</c>, <c>ReLU</c>, <c>MatMul</c>, …) — a 1:1 correspondence
/// with user-visible operations. <see cref="CodegenOpKind"/> is the
/// emitter's dispatch axis, richer because it annotates the op with
/// the fusion strategy. The lowering pass (Codegen/Lowering/)
/// translates a LazyNodeType into one or more CodegenOpKind entries.
/// </para>
/// </remarks>
public enum CodegenOpKind
{
    // ─── Pointwise arithmetic ─────────────────────────────────────────

    /// <summary>Load the i-th element from a graph input tensor.</summary>
    LoadInput = 0,
    /// <summary>Write the i-th accumulator to a graph output tensor.</summary>
    StoreOutput = 1,
    /// <summary>Materialize a compile-time scalar constant.</summary>
    Constant = 2,

    // ─── Binary arithmetic (elementwise) ──────────────────────────────

    Add = 3,
    Sub = 4,
    Mul = 5,
    Div = 6,
    Max = 7,
    Min = 8,
    Pow = 9,

    // ─── Unary arithmetic (elementwise) ───────────────────────────────

    Negate = 10,
    Reciprocal = 11,
    Sqrt = 12,
    Rsqrt = 13,
    Exp = 14,
    Log = 15,
    Sin = 16,
    Cos = 17,
    Tan = 18,
    Tanh = 19,
    Abs = 20,
    Floor = 21,
    Ceil = 22,
    Round = 23,

    // ─── Activation functions (elementwise, often fused as epilogue) ──

    ReLU = 24,
    Sigmoid = 25,
    GELU = 26,
    Swish = 27,
    LeakyReLU = 28,
    ELU = 29,
    SoftPlus = 30,
    HardSwish = 31,

    // ─── Comparison (elementwise, produces Bool) ──────────────────────

    Equal = 32,
    NotEqual = 33,
    Greater = 34,
    GreaterEqual = 35,
    Less = 36,
    LessEqual = 37,

    // ─── Reductions ───────────────────────────────────────────────────

    /// <summary>Σ along one or more axes.</summary>
    ReduceSum = 38,
    /// <summary>arithmetic mean along one or more axes.</summary>
    ReduceMean = 39,
    /// <summary>max along one or more axes (non-differentiable at ties).</summary>
    ReduceMax = 40,
    /// <summary>min along one or more axes (non-differentiable at ties).</summary>
    ReduceMin = 41,
    /// <summary>Σ xᵢ²</summary>
    ReduceSumOfSquares = 42,

    // ─── Linear algebra ───────────────────────────────────────────────

    /// <summary>C = A · B. Emitters produce tiled GEMM.</summary>
    MatMul = 43,
    /// <summary>C = Aᵀ · B. Lowered to MatMul with transpose flag.</summary>
    MatMulTransposeA = 44,
    /// <summary>C = A · Bᵀ.</summary>
    MatMulTransposeB = 45,
    /// <summary>Batched MatMul: <c>C[b] = A[b] · B[b]</c>.</summary>
    BatchMatMul = 46,

    // ─── Convolution ──────────────────────────────────────────────────
    //
    // Convolution had no op kind at all, so it arrived as Opaque or not at all. That put
    // the whole catalog of measured convolution kernels out of reach of every graph: the
    // index-map layer expresses convolution exactly and CodegenAdjoint already derives
    // its backward maps, but nothing upstream could ASK for one.
    //
    // The attribute is a CodegenConvAttributes. Operands are [input, weights].

    /// <summary>
    /// Dense 2-D cross-correlation, <c>out[n,k,oh,ow] = Σ_{c,kh,kw} in[n,c,·,·]·w[k,c,kh,kw]</c>.
    /// </summary>
    Conv2D = 56,

    /// <summary>
    /// Depthwise 2-D convolution: one filter per channel, no summation over channels.
    /// </summary>
    DepthwiseConv2D = 57,

    /// <summary>
    /// Transposed ("fractionally strided") 2-D convolution — the adjoint of
    /// <see cref="Conv2D"/> with respect to its input.
    /// </summary>
    ConvTranspose2D = 58,

    // ─── Softmax (reduction + pointwise, fused as one op for numerical stability) ──

    /// <summary>Numerically stable <c>softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))</c>.</summary>
    Softmax = 47,
    /// <summary>Log-softmax — same as <see cref="Softmax"/> with a log epilogue.</summary>
    LogSoftmax = 48,

    // ─── Attention ────────────────────────────────────────────────────

    /// <summary>Scaled dot-product attention. Emitters produce
    /// flash-attention-style tiled kernels.</summary>
    ScaledDotProductAttention = 49,

    // ─── Layout / movement (usually pre-lowered away by the optimizer) ───

    Transpose = 50,
    Reshape = 51,
    Concat = 52,
    Split = 53,
    Broadcast = 54,

    // ─── Escape hatch ─────────────────────────────────────────────────

    /// <summary>
    /// An op the codegen IR doesn't know how to represent natively —
    /// the lowering pass wraps the original <c>CompiledStep</c>
    /// <c>Execute</c> delegate as an opaque node. Emitters skip
    /// opaque nodes; the surrounding pattern matcher treats them as
    /// fusion boundaries so we can still fuse everything around them.
    /// </summary>
    Opaque = 55,
}

/// <summary>
/// Which semantic category the op belongs to, for dispatch from the
/// fusion pass matcher and the emitter selector.
/// </summary>
public enum CodegenOpCategory
{
    /// <summary>One-to-one map — fuses trivially.</summary>
    Pointwise,
    /// <summary>Collapses dimensions — fuses as map-reduce.</summary>
    Reduction,
    /// <summary>Matrix multiplication — tiled GEMM kernel.</summary>
    Matmul,
    /// <summary>Flash-attention-style — custom kernel per emitter.</summary>
    Attention,
    /// <summary>Sliding-window contraction — a reduction with windowed index maps.</summary>
    Convolution,
    /// <summary>Layout/movement ops that don't need codegen (handled by optimizer).</summary>
    Movement,
    /// <summary>Escape hatch.</summary>
    Opaque,
}

/// <summary>
/// Category classification + per-op predicates used by the fusion
/// pass matcher. Kept as a static helper (not extension methods on
/// the enum) because emitters can specialise on patterns that cross
/// category boundaries (e.g. "a pointwise epilogue on a matmul").
/// </summary>
public static class CodegenOpKinds
{
    /// <summary>
    /// Returns the semantic category of an op. Used by the fusion
    /// matcher to decide which strategy to apply.
    /// </summary>
    public static CodegenOpCategory Categorize(CodegenOpKind kind) => kind switch
    {
        CodegenOpKind.LoadInput or
        CodegenOpKind.StoreOutput or
        CodegenOpKind.Constant or
        CodegenOpKind.Add or CodegenOpKind.Sub or CodegenOpKind.Mul or CodegenOpKind.Div or
        CodegenOpKind.Max or CodegenOpKind.Min or CodegenOpKind.Pow or
        CodegenOpKind.Negate or CodegenOpKind.Reciprocal or CodegenOpKind.Sqrt or CodegenOpKind.Rsqrt or
        CodegenOpKind.Exp or CodegenOpKind.Log or
        CodegenOpKind.Sin or CodegenOpKind.Cos or CodegenOpKind.Tan or CodegenOpKind.Tanh or
        CodegenOpKind.Abs or CodegenOpKind.Floor or CodegenOpKind.Ceil or CodegenOpKind.Round or
        CodegenOpKind.ReLU or CodegenOpKind.Sigmoid or CodegenOpKind.GELU or
        CodegenOpKind.Swish or CodegenOpKind.LeakyReLU or CodegenOpKind.ELU or
        CodegenOpKind.SoftPlus or CodegenOpKind.HardSwish or
        CodegenOpKind.Equal or CodegenOpKind.NotEqual or
        CodegenOpKind.Greater or CodegenOpKind.GreaterEqual or
        CodegenOpKind.Less or CodegenOpKind.LessEqual
            => CodegenOpCategory.Pointwise,

        CodegenOpKind.ReduceSum or CodegenOpKind.ReduceMean or
        CodegenOpKind.ReduceMax or CodegenOpKind.ReduceMin or
        CodegenOpKind.ReduceSumOfSquares or
        CodegenOpKind.Softmax or CodegenOpKind.LogSoftmax
            => CodegenOpCategory.Reduction,

        CodegenOpKind.MatMul or CodegenOpKind.MatMulTransposeA or
        CodegenOpKind.MatMulTransposeB or CodegenOpKind.BatchMatMul
            => CodegenOpCategory.Matmul,

        CodegenOpKind.Conv2D or CodegenOpKind.DepthwiseConv2D or
        CodegenOpKind.ConvTranspose2D
            => CodegenOpCategory.Convolution,

        CodegenOpKind.ScaledDotProductAttention
            => CodegenOpCategory.Attention,

        CodegenOpKind.Transpose or CodegenOpKind.Reshape or
        CodegenOpKind.Concat or CodegenOpKind.Split or CodegenOpKind.Broadcast
            => CodegenOpCategory.Movement,

        CodegenOpKind.Opaque
            => CodegenOpCategory.Opaque,

        _ => CodegenOpCategory.Opaque,
    };

    /// <summary>Returns true if the op is a pointwise primitive that
    /// fuses trivially into a single SIMD loop / GPU thread-per-element kernel.</summary>
    public static bool IsPointwise(CodegenOpKind kind)
        => Categorize(kind) == CodegenOpCategory.Pointwise
        && kind != CodegenOpKind.LoadInput
        && kind != CodegenOpKind.StoreOutput;

    /// <summary>Returns true if the op produces a boolean tensor.
    /// Used by the emitter to pick the result dtype.</summary>
    public static bool IsComparison(CodegenOpKind kind)
        => kind is CodegenOpKind.Equal or CodegenOpKind.NotEqual
                or CodegenOpKind.Greater or CodegenOpKind.GreaterEqual
                or CodegenOpKind.Less or CodegenOpKind.LessEqual;

    /// <summary>Returns true for unary pointwise ops (one input producer).</summary>
    public static bool IsUnaryPointwise(CodegenOpKind kind)
        => kind is CodegenOpKind.Negate or CodegenOpKind.Reciprocal
                or CodegenOpKind.Sqrt or CodegenOpKind.Rsqrt
                or CodegenOpKind.Exp or CodegenOpKind.Log
                or CodegenOpKind.Sin or CodegenOpKind.Cos
                or CodegenOpKind.Tan or CodegenOpKind.Tanh
                or CodegenOpKind.Abs or CodegenOpKind.Floor
                or CodegenOpKind.Ceil or CodegenOpKind.Round
                or CodegenOpKind.ReLU or CodegenOpKind.Sigmoid
                or CodegenOpKind.GELU or CodegenOpKind.Swish
                or CodegenOpKind.LeakyReLU or CodegenOpKind.ELU
                or CodegenOpKind.SoftPlus or CodegenOpKind.HardSwish;
}
