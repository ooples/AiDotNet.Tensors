// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Nearest and linear-family (linear / bilinear / trilinear) Interpolate, forward and backward, over per-axis source
/// tables built once per call and run in parallel over the (N, C) planes.
/// </summary>
/// <remarks>
/// <para>
/// The kernels they replace recomputed every source coordinate with a division and a floor for every element of every
/// plane, read and wrote through <c>INumericOperations</c>, allocated three arrays per output element in the linear
/// backward, and ran on one thread. The arithmetic here is the same in the same order - a corner's weight is the
/// product of its per-axis weights from axis 0, the corners accumulate in a double in bit order, and the backward
/// adds into the gradient in output order per plane - so the results are bit-identical to those kernels.
/// </para>
/// <para>
/// A plane only ever reads and writes itself, so the parallel backward needs no synchronization and is deterministic.
/// </para>
/// </remarks>
internal static class InterpolateKernels
{
    private interface IConverter<T>
    {
        double ToDouble(T value);
        T FromDouble(double value);
        T Add(T a, T b);
    }

    private readonly struct FloatConverter : IConverter<float>
    {
        public double ToDouble(float value) => value;
        public float FromDouble(double value) => (float)value;
        public float Add(float a, float b) => a + b;
    }

    private readonly struct DoubleConverter : IConverter<double>
    {
        public double ToDouble(double value) => value;
        public double FromDouble(double value) => value;
        public double Add(double a, double b) => a + b;
    }

    private readonly struct GenericConverter<T> : IConverter<T>
    {
        private readonly INumericOperations<T> _ops;
        public GenericConverter(INumericOperations<T> ops) => _ops = ops;
        public double ToDouble(T value) => _ops.ToDouble(value);
        public T FromDouble(double value) => _ops.FromDouble(value);
        public T Add(T a, T b) => _ops.Add(a, b);
    }

    /// <summary>Per spatial axis: the low and high source offsets (already times the axis stride) and the high weight.</summary>
    private sealed class Tables
    {
        public int Spatial;
        public int[] DstDims = Array.Empty<int>();
        public int SrcPlane;
        public int DstPlane;
        public int[][] Lo = Array.Empty<int[]>();
        public int[][] Hi = Array.Empty<int[]>();
        public double[][] Frac = Array.Empty<double[]>();
    }

    /// <summary>True for the modes these kernels cover.</summary>
    public static bool Handles(InterpolateMode mode)
        => mode is InterpolateMode.Nearest or InterpolateMode.Linear or InterpolateMode.Bilinear or InterpolateMode.Trilinear;

    /// <summary>Writes the interpolation of <paramref name="input"/> into <paramref name="output"/>; false when either has no CPU array.</summary>
    public static bool TryForward<T>(Tensor<T> input, Tensor<T> output, InterpolateMode mode, bool alignCorners)
    {
        var source = input.IsContiguous ? input : input.Contiguous();
        var src = source.GetCpuBackingForStridedRead(out int srcBase);
        var dst = output.GetCpuBackingForContiguousWrite(out int dstBase);
        if (src is null || dst is null) return false;

        var tables = Build(source._shape, output._shape, mode, alignCorners);
        int planes = source._shape[0] * source._shape[1];
        if (src is float[] fs && dst is float[] fd) Forward(fs, srcBase, fd, dstBase, planes, tables, default(FloatConverter));
        else if (src is double[] ds && dst is double[] dd) Forward(ds, srcBase, dd, dstBase, planes, tables, default(DoubleConverter));
        else Forward(src, srcBase, dst, dstBase, planes, tables, new GenericConverter<T>(MathHelper.GetNumericOperations<T>()));
        return true;
    }

    /// <summary>The gradient of <see cref="TryForward{T}"/> with respect to its input; null when either tensor has no CPU array.</summary>
    public static Tensor<T>? TryBackward<T>(Tensor<T> gradOutput, Tensor<T> input, InterpolateMode mode, bool alignCorners)
    {
        var gradient = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        var gradInput = new Tensor<T>(input._shape);
        var gout = gradient.GetCpuBackingForStridedRead(out int goutBase);
        var gin = gradInput.GetCpuBackingForContiguousWrite(out int ginBase);
        if (gout is null || gin is null) return null;

        var tables = Build(input._shape, gradient._shape, mode, alignCorners);
        int planes = input._shape[0] * input._shape[1];
        if (gout is float[] fo && gin is float[] fi) Backward(fo, goutBase, fi, ginBase, planes, tables, default(FloatConverter));
        else if (gout is double[] dout && gin is double[] din) Backward(dout, goutBase, din, ginBase, planes, tables, default(DoubleConverter));
        else Backward(gout, goutBase, gin, ginBase, planes, tables, new GenericConverter<T>(MathHelper.GetNumericOperations<T>()));
        return gradInput;
    }

    private static Tables Build(int[] srcShape, int[] dstShape, InterpolateMode mode, bool alignCorners)
    {
        int spatial = srcShape.Length - 2;
        var t = new Tables
        {
            Spatial = spatial,
            DstDims = new int[spatial],
            Lo = new int[spatial][],
            Hi = new int[spatial][],
            Frac = new double[spatial][],
            SrcPlane = 1,
            DstPlane = 1,
        };
        var srcStride = new int[spatial];
        srcStride[spatial - 1] = 1;
        for (int i = spatial - 2; i >= 0; i--) srcStride[i] = srcStride[i + 1] * srcShape[3 + i];
        bool nearest = mode == InterpolateMode.Nearest;

        for (int i = 0; i < spatial; i++)
        {
            int src = srcShape[2 + i], dst = dstShape[2 + i];
            t.DstDims[i] = dst;
            t.SrcPlane *= src;
            t.DstPlane *= dst;
            var lo = t.Lo[i] = new int[dst];
            var hi = t.Hi[i] = new int[dst];
            var frac = t.Frac[i] = new double[dst];
            for (int d = 0; d < dst; d++)
            {
                if (nearest)
                {
                    // floor(d * in / out), as PyTorch's mode='nearest'.
                    double s = dst > 1 ? (double)d * src / dst : 0.0;
                    int si = (int)Math.Floor(s);
                    if (si >= src) si = src - 1;
                    lo[d] = hi[d] = si * srcStride[i];
                    continue;
                }

                double c = dst <= 1 ? 0.0
                    : alignCorners ? (double)d * (src - 1) / (dst - 1)
                    : ((d + 0.5) * src / dst) - 0.5;
                int l = (int)Math.Floor(c);
                if (l < 0) l = 0;
                int h = l + 1;
                if (h >= src) { h = src - 1; l = Math.Min(l, h); }
                double f = c - l;
                if (f < 0) f = 0;
                if (f > 1) f = 1;
                lo[d] = l * srcStride[i];
                hi[d] = h * srcStride[i];
                frac[d] = f;
            }
        }

        if (nearest) t.Frac = Array.Empty<double[]>();
        return t;
    }

    private static void Forward<T, TConv>(T[] src, int srcBase, T[] dst, int dstBase, int planes, Tables t, TConv conv)
        where TConv : struct, IConverter<T>
    {
        bool nearest = t.Frac.Length == 0;
        int corners = 1 << t.Spatial;
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * t.DstPlane * (nearest ? 1 : corners), plane =>
        {
            int inBase = srcBase + (plane * t.SrcPlane);
            int outBase = dstBase + (plane * t.DstPlane);
            Span<int> idx = stackalloc int[t.Spatial];
            for (int k = 0; k < t.DstPlane; k++)
            {
                if (nearest)
                {
                    int off = 0;
                    for (int i = 0; i < t.Spatial; i++) off += t.Lo[i][idx[i]];
                    dst[outBase + k] = src[inBase + off];
                }
                else
                {
                    double acc = 0.0;
                    for (int corner = 0; corner < corners; corner++)
                    {
                        int off = 0;
                        double weight = 1.0;
                        for (int i = 0; i < t.Spatial; i++)
                        {
                            bool takeHi = ((corner >> i) & 1) == 1;
                            double f = t.Frac[i][idx[i]];
                            off += takeHi ? t.Hi[i][idx[i]] : t.Lo[i][idx[i]];
                            weight *= takeHi ? f : (1.0 - f);
                        }

                        acc += weight * conv.ToDouble(src[inBase + off]);
                    }

                    dst[outBase + k] = conv.FromDouble(acc);
                }

                Advance(idx, t.DstDims);
            }
        }, deterministicSafe: true);
    }

    private static void Backward<T, TConv>(T[] gout, int goutBase, T[] gin, int ginBase, int planes, Tables t, TConv conv)
        where TConv : struct, IConverter<T>
    {
        bool nearest = t.Frac.Length == 0;
        int corners = 1 << t.Spatial;
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * t.DstPlane * (nearest ? 1 : corners), plane =>
        {
            int inBase = ginBase + (plane * t.SrcPlane);
            int outBase = goutBase + (plane * t.DstPlane);
            Span<int> idx = stackalloc int[t.Spatial];
            for (int k = 0; k < t.DstPlane; k++, Advance(idx, t.DstDims))
            {
                double g = conv.ToDouble(gout[outBase + k]);
                if (g == 0.0) continue;
                if (nearest)
                {
                    int off = inBase;
                    for (int i = 0; i < t.Spatial; i++) off += t.Lo[i][idx[i]];
                    gin[off] = conv.Add(gin[off], conv.FromDouble(g));
                    continue;
                }

                for (int corner = 0; corner < corners; corner++)
                {
                    int off = inBase;
                    double weight = 1.0;
                    for (int i = 0; i < t.Spatial; i++)
                    {
                        bool takeHi = ((corner >> i) & 1) == 1;
                        double f = t.Frac[i][idx[i]];
                        off += takeHi ? t.Hi[i][idx[i]] : t.Lo[i][idx[i]];
                        weight *= takeHi ? f : (1.0 - f);
                    }

                    gin[off] = conv.Add(gin[off], conv.FromDouble(weight * g));
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>Steps a row-major multi-index by one.</summary>
    private static void Advance(Span<int> idx, int[] dims)
    {
        for (int i = idx.Length - 1; i >= 0; i--)
        {
            if (++idx[i] < dims[i]) return;
            idx[i] = 0;
        }
    }
}
