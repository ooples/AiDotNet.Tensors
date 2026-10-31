// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The statistic conversions of ILayerNormBackwardPartsBackend on OpenCL: a training BatchNorm or LayerNorm whose kernel
    // saves 1/sqrt(var + eps) reports the variance, and its backward turns that back. Composed, each direction was three
    // elementwise launches (Multiply, Reciprocal, SubScalar; AddScalar, Sqrt, Reciprocal); here each is one, with the same
    // arithmetic under strict math. The input-gradient parts are not offered: their callers keep the composed backward.
    public sealed partial class OpenClBackend : ILayerNormBackwardPartsBackend
    {
        private const string StatisticConversionSource = @"
__kernel void norm_inverse_std(__global const float* variance, __global float* inverseStd, int n, float epsilon)
{
    int i = get_global_id(0);
    if (i < n) inverseStd[i] = 1.0f / sqrt(variance[i] + epsilon);
}

__kernel void norm_variance_from_inverse_std(__global float* values, int n, float epsilon)
{
    int i = get_global_id(0);
    if (i < n)
    {
        float s = values[i];
        values[i] = 1.0f / (s * s) - epsilon;
    }
}
";

        // 0 not tried, 1 compiled, 2 unavailable.
        private int _statisticConversionState;

        private bool StatisticConversionsReady()
        {
            int state = System.Threading.Volatile.Read(ref _statisticConversionState);
            if (state != 0) return state == 1;
            lock (_statisticConversionLock)
            {
                if (_statisticConversionState != 0) return _statisticConversionState == 1;
                var context = _context;
                bool ok = false;
                if (context is not null)
                {
                    try
                    {
                        var program = CompileOrLoadCached(StatisticConversionSource, OpenClBuildOptions.SafeMathFlags,
                            "normalization statistic conversions");
                        _programs.Add(program);
                        _kernelCache["norm_inverse_std"] = new DirectOpenClKernel(context, program, "norm_inverse_std");
                        _kernelCache["norm_variance_from_inverse_std"] =
                            new DirectOpenClKernel(context, program, "norm_variance_from_inverse_std");
                        ok = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL normalization statistic conversions unavailable: {ex.Message}");
                    }
                }
                System.Threading.Volatile.Write(ref _statisticConversionState, ok ? 1 : 2);
                return ok;
            }
        }

        private readonly object _statisticConversionLock = new object();

        private int ElementwiseGroupSize(int n) => (int)Math.Min(256UL, Math.Max(1UL, Math.Min(_maxWorkGroupSize, (ulong)n)));

        bool ILayerNormBackwardPartsBackend.TryLayerNormInverseStd(IGpuBuffer variance, IGpuBuffer inverseStd, int rows, float epsilon)
        {
            if (rows <= 0 || variance.Size < rows || inverseStd.Size < rows || !StatisticConversionsReady()) return false;
            var kernel = _kernelCache["norm_inverse_std"];
            kernel.SetArg(0, BufferHandle(variance));
            kernel.SetArg(1, BufferHandle(inverseStd));
            kernel.SetArg(2, rows);
            kernel.SetArg(3, epsilon);
            int local = ElementwiseGroupSize(rows);
            kernel.Execute1D((rows + local - 1) / local * local, local);
            return true;
        }

        bool ILayerNormBackwardPartsBackend.TryLayerNormVarianceFromInverseStd(IGpuBuffer values, int rows, float epsilon)
        {
            if (rows <= 0 || values.Size < rows || !StatisticConversionsReady()) return false;
            var kernel = _kernelCache["norm_variance_from_inverse_std"];
            kernel.SetArg(0, BufferHandle(values));
            kernel.SetArg(1, rows);
            kernel.SetArg(2, epsilon);
            int local = ElementwiseGroupSize(rows);
            kernel.Execute1D((rows + local - 1) / local * local, local);
            return true;
        }

        bool ILayerNormBackwardPartsBackend.TryLayerNormInputGradient(IGpuBuffer gradOutput, IGpuBuffer input,
            IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer gradInput, int rows, int cols) => false;

        bool ILayerNormBackwardPartsBackend.SupportsLayerNormInputGradientAdd => false;

        bool ILayerNormBackwardPartsBackend.TryLayerNormInputGradientAdd(IGpuBuffer gradOutput, IGpuBuffer input,
            IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer gradInput, int rows, int cols) => false;

        int ILayerNormBackwardPartsBackend.LayerNormRowBackwardScratchFloats(int rows, int cols) => 0;

        bool ILayerNormBackwardPartsBackend.TryLayerNormRowBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
            IGpuBuffer mean, IGpuBuffer variance, IGpuBuffer scratch, IGpuBuffer gradInput, IGpuBuffer gradGamma,
            IGpuBuffer gradBeta, int rows, int cols, float epsilon, bool addInput, bool addGamma, bool addBeta) => false;

        int ILayerNormBackwardPartsBackend.LayerNormAffineGradientScratchFloats(int rows, int cols) => 0;

        bool ILayerNormBackwardPartsBackend.TryLayerNormAffineGradients(IGpuBuffer gradOutput, IGpuBuffer input,
            IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer scratch, IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows,
            int cols, bool addGamma, bool addBeta) => false;
    }
}
