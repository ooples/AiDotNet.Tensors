// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CUDA convolution kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for convolution operations.
    /// </summary>
    internal static class CudaConvolutionKernels
    {
        public static string GetSource()
        {
            return KernelSourceResources.Read("CudaConvolution.main.c") + ConvEngineCore + KernelSourceResources.Read("CudaConvolution.tail.c");
        }

        /// <summary>
        /// #1325: the implicit-GEMM conv engine's shared device code (CE_BK, ce_mainloop and the forward body
        /// conv2d_fprop_ig_body with its prologue / epilogue fusion hooks), CUDA C with no extern "C" entry points.
        /// Prepend it to another NVRTC module to instantiate the engine with fused operand transforms.
        /// </summary>
        public static string ConvEngineCore => KernelSourceResources.Read("CudaConvolution.engine_core.c");

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "im2col",
                "col2im",
                "conv2d_direct",
                "conv2d_backward_input",
                "conv2d_backward_kernel",
                "depthwise_conv2d",
                "conv_transpose2d",
                "conv_transpose2d_backward_input",
                "conv_transpose2d_backward_kernel",
                "conv2d_tiled",
                "conv2d_winograd_f2x2_3x3",
                "conv3d_direct",
                "trilinear_interpolate",
                "trilinear_interpolate_backward",
                "conv_transpose3d",
                "conv_transpose3d_backward_input",
                "conv_transpose3d_backward_weights",
                "spiral_conv",
                "spiral_conv_backward_input",
                "spiral_conv_backward_weights",
                "depthwise_conv2d_backward_input",
                "depthwise_conv2d_backward_weights",
                "conv2d_wgrad_implicit_gemm",
                "conv2d_wgrad_splitk_reduce", "conv2d_dgrad_phase_reduce", "conv2d_dgrad_weight_tapmajor",
                "conv2d_dgrad_implicit_gemm",
                "conv2d_dgrad_implicit_gemm_c16",
                "conv2d_dgrad_implicit_gemm_c16_phase",
                "conv2d_fprop_ig128_tap",
                "conv2d_fprop_ig128_gen",
                "conv2d_fprop_ig64_tap",
                "conv2d_fprop_ig64_gen",
                "conv2d_wgrad_ce128",
                "conv2d_wgrad_ce64",
                "conv2d_dgrad_ce128",
                "conv2d_dgrad_ce64",
                "conv2d_dgrad_ce128_phase",
                "conv2d_dgrad_ce64_phase",
                "conv2d_fprop_ig128_tap_x3",
                "conv2d_fprop_ig128_gen_x3",
                "conv2d_fprop_ig64_tap_x3",
                "conv2d_fprop_ig64_gen_x3",
                "conv2d_wgrad_ce128_x3",
                "conv2d_wgrad_ce64_x3",
                "conv2d_dgrad_ce128_x3",
                "conv2d_dgrad_ce128_phase_x3",
                "conv2d_dgrad_ce64_x3",
                "conv2d_dgrad_ce64_phase_x3",
                "conv2d_fprop_ig128_tap_tf32",
                "conv2d_fprop_ig128_gen_tf32",
                "conv2d_fprop_ig64_tap_tf32",
                "conv2d_fprop_ig64_gen_tf32",
                "conv2d_wgrad_ce128_tf32",
                "conv2d_wgrad_ce64_tf32",
                "conv2d_dgrad_ce128_tf32",
                "conv2d_dgrad_ce128_phase_tf32",
                "conv2d_dgrad_ce64_tf32",
                "conv2d_dgrad_ce64_phase_tf32"
            };
        }
    }
}
