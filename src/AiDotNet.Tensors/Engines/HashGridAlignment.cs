namespace AiDotNet.Tensors.Engines;

/// <summary>
/// How a multiresolution hash encoding (Müller et al. 2022, "Instant Neural Graphics Primitives with a Multiresolution
/// Hash Encoding") places each level's grid over the unit cube.
/// </summary>
/// <remarks>
/// <para>
/// Both variants hash with tiny-cuda-nn's coherent prime hash (factors 1, 2654435761, 805459861), and index a level
/// densely, without hashing, when its whole grid fits the table. They differ only in the per-level scale and offset.
/// </para>
/// </remarks>
public enum HashGridAlignment
{
    /// <summary>
    /// tiny-cuda-nn's grid (include/tiny-cuda-nn/common_device.h, <c>grid_scale</c> and <c>pos_fract</c>):
    /// <c>scale = 2^(l * log2(b)) * N_min - 1</c> and <c>pos = x * scale + 0.5</c>, so N_min counts grid vertices and
    /// the half-cell offset staggers the levels (paper, appendix A). Each axis has <c>ceil(scale) + 1</c> vertices. This
    /// is the reference implementation, and what models trained with tiny-cuda-nn expect.
    /// </summary>
    TinyCudaNn = 0,

    /// <summary>
    /// The paper's equations 2 and 3: <c>N_l = floor(N_min * b^l)</c> and <c>pos = x * N_l</c>, with no offset, so
    /// N_min counts cells and each axis has <c>N_l + 1</c> vertices.
    /// </summary>
    Paper = 1,
}
