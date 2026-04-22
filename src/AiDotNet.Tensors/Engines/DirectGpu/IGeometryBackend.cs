// Copyright (c) AiDotNet. All rights reserved.
// Optional capability interface for backends that ship native kernels for
// the geometry / sampling ops added by Issue #217 (Interpolate, PadNd,
// GridSample, AffineGrid3D). DirectGpuTensorEngine type-tests against
// this and falls through to the CpuEngine reference when absent or when
// the op requires a shape/mode combination the backend doesn't support.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Native kernels for the geometry / sampling family (Issue #217).
    /// <para>Signatures are scoped to the most common 4D NCHW / NHWC
    /// shapes — rank-3 1D and rank-5 3D Interpolate, or rank ≠ 4 PadNd,
    /// fall through to the CpuEngine reference.</para>
    /// <para>Mode and padding-mode ints map one-to-one onto
    /// <see cref="InterpolateMode"/>, <see cref="PadMode"/>,
    /// <see cref="GridSampleMode"/> and <see cref="GridSamplePadding"/>.</para>
    /// </summary>
    public interface IGeometryBackend
    {
        /// <summary>
        /// 2D interpolation on NCHW float tensors. Supports
        /// nearest / bilinear / area / bicubic (modes 0, 2, 5, 3 per the
        /// <see cref="InterpolateMode"/> enum order). 1D / 3D variants
        /// fall through to the CPU reference.
        /// </summary>
        /// <param name="input">NCHW float input.</param>
        /// <param name="output">NCHW float output (caller-allocated).</param>
        void Interpolate2D(IGpuBuffer input, IGpuBuffer output,
            int N, int C, int Hin, int Win, int Hout, int Wout,
            int mode, bool alignCorners);

        /// <summary>
        /// 4D tensor pad along all four axes simultaneously. Any of the
        /// four <see cref="PadMode"/> options (constant / reflect /
        /// replicate / circular). <paramref name="padValue"/> is only
        /// consulted in constant mode.
        /// </summary>
        void Pad4D(IGpuBuffer input, IGpuBuffer output,
            int N, int C, int Hin, int Win,
            int padN0, int padN1, int padC0, int padC1,
            int padH0, int padH1, int padW0, int padW1,
            int mode, float padValue);

        /// <summary>
        /// 2D GridSample on NHWC float tensors. Input
        /// <c>[N, H, W, C]</c>, grid <c>[N, outH, outW, 2]</c>, output
        /// <c>[N, outH, outW, C]</c>.
        /// </summary>
        void GridSample2D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
            int N, int H, int W, int C, int outH, int outW,
            int mode, int padding, bool alignCorners);

        /// <summary>
        /// 3D affine-grid generator — produces a <c>[N, D, H, W, 3]</c>
        /// sampling grid from a <c>[N, 3, 4]</c> affine matrix.
        /// </summary>
        void AffineGrid3D(IGpuBuffer theta, IGpuBuffer grid,
            int N, int D, int H, int W, bool alignCorners);
    }
}
