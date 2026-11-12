using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>GridSample bicubic's padded integer tap (-1 for a zero-padded tap), for the recorded composite.</summary>
    internal static int GridSampleBicubicTap(int i, int size, GridSamplePadding padding, bool alignCorners)
        => BicubicTap(i, size, padding, alignCorners);

    /// <summary>
    /// The exact adjoint of <see cref="InterpolateBicubic2D{T}"/>: each output gradient is scattered back to the same
    /// clamped 4x4 taps with the same Catmull-Rom weights the forward read them with. The interpolate backward used to
    /// treat bicubic like area interpolation (overlap weights), which is a different function.
    /// </summary>
    internal static Tensor<T> InterpolateBicubic2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, bool alignCorners)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gradInput = new Tensor<T>((int[])inputShape.Clone());
        int N = inputShape[0], C = inputShape[1], H = inputShape[2], W = inputShape[3];
        int outH = gradOutput._shape[2], outW = gradOutput._shape[3];
        var g = RoIAlignPlan.ToDoubles(gradOutput);
        var acc = new double[gradInput.Length];
        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        {
            int srcBase = (n * C + c) * H * W;
            int dstBase = (n * C + c) * outH * outW;
            for (int y = 0; y < outH; y++)
            {
                double sy = SourceCoordinate(y, outH, H, alignCorners);
                int y0 = (int)System.Math.Floor(sy);
                double[] wy = CubicWeights(sy - y0);
                for (int x = 0; x < outW; x++)
                {
                    double gv = g[dstBase + y * outW + x];
                    if (gv == 0.0) continue;
                    double sx = SourceCoordinate(x, outW, W, alignCorners);
                    int x0 = (int)System.Math.Floor(sx);
                    double[] wx = CubicWeights(sx - x0);
                    for (int yy = 0; yy < 4; yy++)
                    {
                        int yi = ClampInt(y0 - 1 + yy, 0, H - 1);
                        for (int xx = 0; xx < 4; xx++)
                        {
                            int xi = ClampInt(x0 - 1 + xx, 0, W - 1);
                            acc[srcBase + yi * W + xi] += wy[yy] * wx[xx] * gv;
                        }
                    }
                }
            }
        }
        return RoIAlignPlan.FromDoubles<T>(acc, (int[])inputShape.Clone());
    }
}
