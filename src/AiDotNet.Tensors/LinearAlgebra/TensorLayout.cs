namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Authoritative physical layout of a 4-D (or promoted 5-D) tensor.
/// Unlike <see cref="TensorLayoutHint"/> which is a Flags enum recording
/// multiple informational properties, <c>TensorLayout</c> names ONE
/// specific physical arrangement so the engine's op dispatcher can route
/// each consumer to the matching kernel family.
///
/// <para>The typical ResNet/BERT inference path used <see cref="Nchw"/>
/// exclusively until we added channel-packed variants to close the gap
/// vs oneDNN / MKL: packing <c>cBlock</c> consecutive channels into the
/// innermost dimension turns every per-channel op (Conv, BN, bias, pool,
/// pointwise) into a contiguous SIMD load + FMA, which is the single
/// biggest win on modern CPUs.</para>
///
/// <para><b>cBlock sizing.</b> <see cref="Nchwc8"/> packs 8 channels —
/// matches <c>Vector256&lt;float&gt;</c> (AVX2+FMA, universal on x86_64
/// since 2013) and <c>Vector128&lt;double&gt;</c> pairs. <see cref="Nchwc16"/>
/// packs 16 channels — matches <c>Vector512&lt;float&gt;</c> (AVX-512,
/// Skylake-SP / Zen 4 / Ice Lake+) and unlocks the twice-wider FMA path
/// the B-track microkernels target.</para>
///
/// <para><b>Reorder policy.</b> The layout planner walks the compiled
/// plan's step list and identifies contiguous regions of NCHWc-eligible
/// ops. A single <c>ReorderToNchwc</c> is inserted at region entry and a
/// matching <c>ReorderToNchw</c> at region exit, so the cost of layout
/// conversion is amortized across every op in the region. Ops whose
/// channel count is not divisible by <c>cBlock</c> force a region exit;
/// ops whose semantic is layout-invariant (elementwise arithmetic with
/// matching shapes) extend the region. See <c>TensorLayoutExtensions</c>
/// for the <see cref="CBlock"/> accessor that maps each layout to its
/// channel-block size.</para>
/// </summary>
public enum TensorLayout
{
    /// <summary>
    /// Standard NCHW row-major (batch, channels, height, width). Default
    /// for every tensor constructed outside the NCHWc fast path; every
    /// ONNX model input starts in this layout.
    /// </summary>
    Nchw = 0,

    /// <summary>
    /// NCHWc with <c>cBlock = 8</c>. Shape encoded as
    /// <c>[batch, channels / 8, height, width, 8]</c>. Matches AVX2
    /// 256-bit float lanes.
    /// </summary>
    Nchwc8 = 1,

    /// <summary>
    /// NCHWc with <c>cBlock = 16</c>. Shape encoded as
    /// <c>[batch, channels / 16, height, width, 16]</c>. Matches AVX-512
    /// 512-bit float lanes. Populated by the B-track kernels; unused on
    /// AVX2-only hardware.
    /// </summary>
    Nchwc16 = 2,
}

/// <summary>
/// Utility helpers for <see cref="TensorLayout"/>.
/// </summary>
public static class TensorLayoutExtensions
{
    /// <summary>
    /// Returns the channel-block size for a given layout, or 1 for
    /// <see cref="TensorLayout.Nchw"/> (meaning channels are not packed).
    /// </summary>
    public static int CBlock(this TensorLayout layout) => layout switch
    {
        TensorLayout.Nchwc8 => 8,
        TensorLayout.Nchwc16 => 16,
        _ => 1,
    };

    /// <summary>
    /// True when the layout is any of the channel-packed variants.
    /// Used by layout-aware op translators to decide whether to take the
    /// NCHWc fast path or fall back to NCHW kernels.
    /// </summary>
    public static bool IsChannelPacked(this TensorLayout layout) =>
        layout == TensorLayout.Nchwc8 || layout == TensorLayout.Nchwc16;
}
