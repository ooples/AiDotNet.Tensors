// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Generated row / column kernel variants for OpenCL, served through the tuned-kernel registry (#1114).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL instances of the registry's row-wise and column-wise variants (CUDA: CudaTunedRowKernels). The definition
    /// lives in <see cref="TunedRowKernelTemplate"/>, which emits the same bodies for Metal (#1116), so these kernels and
    /// Metal's run identical arithmetic.
    /// </summary>
    /// <remarks>
    /// OpenCL 1.2 has no portable sub-group shuffle, so each lane group reduces through local memory with the same
    /// halving tree as CUDA's shuffle-down reduction.
    /// </remarks>
    internal static class OpenClTunedRowKernels
    {
        internal const int GroupSize = TunedRowKernelTemplate.GroupSize;

        /// <summary>Lane counts each row-kernel family is instantiated for.</summary>
        internal static int[] RowLanes => TunedRowKernelTemplate.RowLanes;

        /// <summary>Row-lane counts the column-reduction family is instantiated for.</summary>
        internal static int[] ColumnRowLanes => TunedRowKernelTemplate.ColumnRowLanes;

        internal static string SoftmaxName(int lanes) => TunedRowKernelTemplate.SoftmaxName(lanes);
        internal static string SoftmaxBackwardName(int lanes) => TunedRowKernelTemplate.SoftmaxBackwardName(lanes);
        internal static string LayerNormName(int lanes) => TunedRowKernelTemplate.LayerNormName(lanes);
        internal static string LayerNormBackwardName(int lanes) => TunedRowKernelTemplate.LayerNormBackwardName(lanes);
        internal static string LayerNormGradParamsName(int rowLanes) => TunedRowKernelTemplate.LayerNormGradParamsName(rowLanes);
        internal static string RmsNormGradGammaName(int rowLanes) => TunedRowKernelTemplate.RmsNormGradGammaName(rowLanes);

        /// <summary>The im2col variant writing [P, B * L] for the single-GEMM convolution candidate.</summary>
        internal const string Im2ColName = TunedRowKernelTemplate.Im2ColName;

        public static string GetSource() => TunedRowKernelTemplate.GetSource(TunedKernelDialect.OpenCl);

        public static string[] GetKernelNames() => TunedRowKernelTemplate.GetKernelNames();
    }
}
