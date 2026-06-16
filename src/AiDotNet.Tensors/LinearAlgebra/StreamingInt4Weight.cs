// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// No-upcast int4 group-quant form of a streaming weight, fed straight to the int4 weight-only
/// GEMM (<c>SimdGemm.SgemmWithInt4GroupScaled</c>) without ever materializing the full fp32
/// weight. <see cref="Data"/> is the weight in the kernel's <c>[N, K]</c> (= <c>[out, in]</c>)
/// row-major layout (Wᵀ of a logical <c>[in, out]</c> Linear weight), and <see cref="GroupScales"/>
/// holds one fp32 scale per contiguous <see cref="GroupSize"/>-element group of that flat array —
/// the exact layout <see cref="StreamingStoreCodec"/> produces when it stores int4. Keeping the
/// weight 4-bit through the matmul is what lets an int4-resident model hold its ~8x-smaller
/// footprint at compute time (the memory half of L5a / #1622).
/// </summary>
internal sealed class StreamingInt4Weight
{
    public StreamingInt4Weight(sbyte[] data, float[] groupScales, int groupSize, int rows, int k, bool transposedFromLogical)
    {
        Data = data;
        GroupScales = groupScales;
        GroupSize = groupSize;
        Rows = rows;
        K = k;
        TransposedFromLogical = transposedFromLogical;
    }

    /// <summary>Sign-extended int4 weights (one <see cref="sbyte"/> in [-7,7] per element) in the
    /// kernel's [Rows, K] (= [out, in]) row-major layout.</summary>
    public sbyte[] Data { get; }

    /// <summary>Per-group fp32 scales over the FLAT <see cref="Data"/> array (group of flat index
    /// <c>i</c> is <c>i / GroupSize</c>).</summary>
    public float[] GroupScales { get; }

    /// <summary>Quantization group size (e.g. 128).</summary>
    public int GroupSize { get; }

    /// <summary>Rows of <see cref="Data"/> = output channels (the matmul's N).</summary>
    public int Rows { get; }

    /// <summary>Columns of <see cref="Data"/> = input features (the matmul's K).</summary>
    public int K { get; }

    /// <summary>True when <see cref="Data"/> is the TRANSPOSE of the tensor's logical layout
    /// (a Linear weight logically [in, out] stored here as [out, in] so it feeds the GEMM
    /// directly). The fp32 lazy fallback must transpose back to the logical [in, out].</summary>
    public bool TransposedFromLogical { get; }
}
