using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorGridSample3D<T>(Tensor<T> input, Tensor<T> grid, GridSampleMode mode = GridSampleMode.Bilinear,
        GridSamplePadding padding = GridSamplePadding.Zeros, bool alignCorners = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (mode == GridSampleMode.Bicubic) throw new ArgumentException("bicubic sampling is 2-D only, as in PyTorch.", nameof(mode));
        if (input.Rank != 5 || grid.Rank != 5 || grid._shape[4] != 3 || grid._shape[0] != input._shape[0])
            throw new ArgumentException("input must be [N, C, D, H, W] and grid [N, D', H', W', 3] with the same N.");
        var ops = MathHelper.GetNumericOperations<T>();
        var x = ToDoubles(input, ops);
        var g = ToDoubles(grid, ops);
        int n = input._shape[0], c = input._shape[1];
        var size = new[] { input._shape[4], input._shape[3], input._shape[2] };   // W, H, D: the grid's x, y, z order
        int points = grid._shape[1] * grid._shape[2] * grid._shape[3];
        var result = new double[n * c * points];
        var sample = new GridSampler3D(size, mode, padding, alignCorners);
        for (int b = 0; b < n; b++)
            for (int q = 0; q < points; q++)
            {
                int gi = (b * points + q) * 3;
                sample.Locate(g[gi], g[gi + 1], g[gi + 2]);
                for (int ch = 0; ch < c; ch++)
                {
                    int plane = (b * c + ch) * sample.Volume;
                    double acc = 0;
                    for (int t = 0; t < sample.TapCount; t++) acc += sample.Weight[t] * x[plane + sample.Offset[t]];
                    result[(b * c + ch) * points + q] = acc;
                }
            }
        var outShape = new[] { n, c, grid._shape[1], grid._shape[2], grid._shape[3] };
        var output = FromDoubles<T>(outShape, i => result[i]);
        DifferentiableOps.RecordBinary("TensorGridSample3D", output, input, grid, GridSample3DBackward<T>,
            new object[] { mode, padding, alignCorners });
        return output;
    }

    private static double[] ToDoubles<T>(Tensor<T> t, Interfaces.INumericOperations<T> ops)
        => (t.IsContiguous ? t : t.Contiguous()).AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();

    private static void GridSample3DBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> input = inputs[0], grid = inputs[1];
        var x = ToDoubles(input, ops);
        var g = ToDoubles(grid, ops);
        var go = ToDoubles(gradOutput, ops);
        int n = input._shape[0], c = input._shape[1];
        var size = new[] { input._shape[4], input._shape[3], input._shape[2] };
        int points = grid._shape[1] * grid._shape[2] * grid._shape[3];
        var dx = new double[x.Length];
        var dg = new double[g.Length];
        var sample = new GridSampler3D(size, (GridSampleMode)savedState[0], (GridSamplePadding)savedState[1], (bool)savedState[2]);
        for (int b = 0; b < n; b++)
            for (int q = 0; q < points; q++)
            {
                int gi = (b * points + q) * 3;
                sample.Locate(g[gi], g[gi + 1], g[gi + 2]);
                for (int ch = 0; ch < c; ch++)
                {
                    int plane = (b * c + ch) * sample.Volume;
                    double up = go[(b * c + ch) * points + q];
                    for (int t = 0; t < sample.TapCount; t++)
                    {
                        dx[plane + sample.Offset[t]] += sample.Weight[t] * up;
                        double v = x[plane + sample.Offset[t]] * up;
                        for (int a = 0; a < 3; a++) dg[gi + a] += v * sample.WeightGrad[t, a];
                    }
                }
            }
        DifferentiableOps.AccumulateGrad(grads, input, FromDoubles<T>((int[])input._shape.Clone(), i => dx[i]), engine);
        DifferentiableOps.AccumulateGrad(grads, grid, FromDoubles<T>((int[])grid._shape.Clone(), i => dg[i]), engine);
    }

    // Per-point taps of a 3-D grid sample: up to 8 voxels with their weights and each weight's derivative with
    // respect to the normalized grid coordinates (following PyTorch's GridSampler: unnormalize, then clip or
    // reflect, then trilinear or nearest, skipping out-of-range voxels).
    private sealed class GridSampler3D
    {
        private readonly int[] _size;
        private readonly GridSampleMode _mode;
        private readonly GridSamplePadding _padding;
        private readonly bool _align;
        public readonly int[] Offset = new int[8];
        public readonly double[] Weight = new double[8];
        public readonly double[,] WeightGrad = new double[8, 3];
        public int TapCount;
        public int Volume => _size[0] * _size[1] * _size[2];

        public GridSampler3D(int[] size, GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
        {
            _size = size; _mode = mode; _padding = padding; _align = alignCorners;
        }

        public void Locate(double gx, double gy, double gz)
        {
            var coord = new double[3];
            var slope = new double[3];
            var normalized = new[] { gx, gy, gz };
            for (int a = 0; a < 3; a++) coord[a] = SourceIndex(normalized[a], _size[a], out slope[a]);
            TapCount = 0;
            if (_mode == GridSampleMode.Nearest)
            {
                // nearbyint: round half to even.
                int ix = (int)Math.Round(coord[0]), iy = (int)Math.Round(coord[1]), iz = (int)Math.Round(coord[2]);
                if (Inside(ix, 0) && Inside(iy, 1) && Inside(iz, 2))
                {
                    Offset[0] = (iz * _size[1] + iy) * _size[0] + ix; Weight[0] = 1;
                    WeightGrad[0, 0] = WeightGrad[0, 1] = WeightGrad[0, 2] = 0;
                    TapCount = 1;
                }
                return;
            }
            int x0 = (int)Math.Floor(coord[0]), y0 = (int)Math.Floor(coord[1]), z0 = (int)Math.Floor(coord[2]);
            double fx = coord[0] - x0, fy = coord[1] - y0, fz = coord[2] - z0;
            for (int oz = 0; oz < 2; oz++)
                for (int oy = 0; oy < 2; oy++)
                    for (int ox = 0; ox < 2; ox++)
                    {
                        int ix = x0 + ox, iy = y0 + oy, iz = z0 + oz;
                        if (!Inside(ix, 0) || !Inside(iy, 1) || !Inside(iz, 2)) continue;
                        double wx = ox == 1 ? fx : 1 - fx, wy = oy == 1 ? fy : 1 - fy, wz = oz == 1 ? fz : 1 - fz;
                        int t = TapCount++;
                        Offset[t] = (iz * _size[1] + iy) * _size[0] + ix;
                        Weight[t] = wx * wy * wz;
                        WeightGrad[t, 0] = (ox == 1 ? 1 : -1) * wy * wz * slope[0];
                        WeightGrad[t, 1] = (oy == 1 ? 1 : -1) * wx * wz * slope[1];
                        WeightGrad[t, 2] = (oz == 1 ? 1 : -1) * wx * wy * slope[2];
                    }
        }

        private bool Inside(int i, int axis) => i >= 0 && i < _size[axis];

        // The voxel coordinate for a normalized one, and d(voxel)/d(normalized).
        private double SourceIndex(double normalized, int size, out double slope)
        {
            double coord = _align ? (normalized + 1) / 2 * (size - 1) : ((normalized + 1) * size - 1) / 2;
            slope = _align ? (size - 1) / 2.0 : size / 2.0;
            if (_padding == GridSamplePadding.Border) return Clip(coord, size, ref slope);
            if (_padding == GridSamplePadding.Reflection)
            {
                coord = _align ? Reflect(coord, 0, 2 * (size - 1), ref slope) : Reflect(coord, -1, 2 * size - 1, ref slope);
                return Clip(coord, size, ref slope);
            }
            return coord;
        }

        private static double Clip(double coord, int size, ref double slope)
        {
            if (coord <= 0) { slope = 0; return 0; }
            if (coord >= size - 1) { slope = 0; return size - 1; }
            return coord;
        }

        private static double Reflect(double coord, int twiceLow, int twiceHigh, ref double slope)
        {
            if (twiceLow == twiceHigh) { slope = 0; return 0; }
            double min = twiceLow / 2.0, span = (twiceHigh - twiceLow) / 2.0, sign = 1;
            coord -= min;
            if (coord < 0) { sign = -1; coord = -coord; }
            double extra = coord % span, flips = Math.Floor(coord / span);
            if (flips % 2 == 0) { slope *= sign; return extra + min; }
            slope *= -sign;
            return span - extra + min;
        }
    }
}
