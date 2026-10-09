using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorConvolution<T>(Tensor<T> input, Tensor<T> weight, Tensor<T>? bias, int[] stride, int[] padding,
        int[] dilation, bool transposed = false, int[]? outputPadding = null, int groups = 1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        int dims = input.Rank - 2;
        if (dims < 1 || dims > 3 || weight.Rank != input.Rank)
            throw new ArgumentException("input must be [batch, channels, 1-3 spatial axes] and weight of the same rank.", nameof(input));
        int[] Axis(int[]? v, string name, int fill)
        {
            if (v == null) return Enumerable.Repeat(fill, dims).ToArray();
            if (v.Length == 1) return Enumerable.Repeat(v[0], dims).ToArray();
            if (v.Length != dims) throw new ArgumentException($"{name} needs 1 or {dims} entries.", name);
            return v;
        }
        var s = Axis(stride, nameof(stride), 1); var p = Axis(padding, nameof(padding), 0);
        var d = Axis(dilation, nameof(dilation), 1); var op = Axis(outputPadding, nameof(outputPadding), 0);
        for (int a = 0; a < dims; a++)
        {
            if (s[a] < 1) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be at least 1.");
            if (d[a] < 1) throw new ArgumentOutOfRangeException(nameof(dilation), "dilation must be at least 1.");
            if (p[a] < 0) throw new ArgumentOutOfRangeException(nameof(padding), "padding must not be negative.");
            // As PyTorch: output padding only resolves the transposed output size, so it is below the stride or the
            // dilation, and a plain convolution takes none.
            if (op[a] < 0 || (transposed ? op[a] >= Math.Max(s[a], d[a]) : op[a] != 0))
                throw new ArgumentOutOfRangeException(nameof(outputPadding), transposed
                    ? "outputPadding must be non-negative and smaller than the stride or the dilation."
                    : "outputPadding applies only to a transposed convolution.");
        }
        if (groups < 1 || input._shape[1] % groups != 0 || weight._shape[0] % groups != 0)
            throw new ArgumentException("groups must divide the input channels and weight's first axis.", nameof(groups));
        var k = weight._shape.Skip(2).ToArray();
        var x = input;
        if (transposed)
        {
            // A transposed convolution is a stride-1 convolution of the input spread out by the stride, padded by
            // d(k-1)-p on each side (plus the output padding on the far side), with the kernel flipped and its
            // channel axes swapped.
            var spatial = input._shape.Skip(2).ToArray();
            var lead = new int[dims];
            var spread = new int[dims];
            for (int a = 0; a < dims; a++)
            {
                lead[a] = d[a] * (k[a] - 1) - p[a];
                spread[a] = (spatial[a] - 1) * s[a] + 1 + 2 * lead[a] + op[a];
                if (spread[a] < d[a] * (k[a] - 1) + 1) throw new ArgumentException("the transposed convolution's output would be empty.");
            }
            int planes = input._shape[0] * input._shape[1], inPlane = spatial.Aggregate(1, (a, b) => a * b);
            int outPlane = spread.Aggregate(1, (a, b) => a * b);
            var source = new int[planes * outPlane];
            var at = new int[dims];
            for (int q = 0; q < outPlane; q++)
            {
                int flat = 0, rest = q;
                bool hit = true;
                for (int a = dims - 1; a >= 0; a--) { at[a] = rest % spread[a]; rest /= spread[a]; }
                for (int a = 0; a < dims && hit; a++)
                {
                    int shifted = at[a] - lead[a];
                    if (shifted < 0 || shifted % s[a] != 0 || shifted / s[a] >= spatial[a]) hit = false;
                    else flat = flat * spatial[a] + shifted / s[a];
                }
                for (int plane = 0; plane < planes; plane++) source[plane * outPlane + q] = hit ? plane * inPlane + flat : -1;
            }
            x = IndexMap("TensorConvolution", input, input._shape.Take(2).Concat(spread).ToArray(), source);
            // [in, out/g, k...] -> per group [out/g, in/g, k...] flipped, regrouped to [out, in/g, k...].
            int inPerGroup = weight._shape[0] / groups, outPerGroup = weight._shape[1];
            var w = Reshape(weight, new[] { groups, inPerGroup, outPerGroup }.Concat(k).ToArray());
            w = TensorPermute(w, new[] { 0, 2, 1 }.Concat(Enumerable.Range(3, dims)).ToArray());
            w = TensorFlip(w, Enumerable.Range(3, dims).ToArray());
            weight = Reshape(w, new[] { groups * outPerGroup, inPerGroup }.Concat(k).ToArray());
            s = Enumerable.Repeat(1, dims).ToArray();
            p = new int[dims];
        }
        int inGroup = x._shape[1] / groups, outGroup = weight._shape[0] / groups;
        if (weight._shape[1] != inGroup) throw new ArgumentException($"weight expects {weight._shape[1]} channels per group, input has {inGroup}.", nameof(weight));
        var outputs = new Tensor<T>[groups];
        for (int g = 0; g < groups; g++)
        {
            var xg = groups == 1 ? x : TensorNarrow(x, 1, g * inGroup, inGroup);
            var wg = groups == 1 ? weight : TensorNarrow(weight, 0, g * outGroup, outGroup);
            outputs[g] = ConvolveSpatial(xg, wg, s, p, d);
        }
        var y = groups == 1 ? outputs[0] : TensorConcatenate(outputs, 1);
        if (bias == null) return y;
        if (bias.Length != y._shape[1])
            throw new ArgumentException($"bias has {bias.Length} entries; the convolution has {y._shape[1]} output channels.", nameof(bias));
        var biasShape = new[] { 1, bias.Length }.Concat(Enumerable.Repeat(1, dims)).ToArray();
        return TensorAdd(y, TensorBroadcastTo(Reshape(bias, biasShape), (int[])y._shape.Clone()));
    }

    // 1-D runs as 2-D over a unit height, so every rank gets per-axis stride, padding and dilation.
    private Tensor<T> ConvolveSpatial<T>(Tensor<T> x, Tensor<T> w, int[] s, int[] p, int[] d)
    {
        switch (s.Length)
        {
            case 1:
                var y = Conv2D(Reshape(x, new[] { x._shape[0], x._shape[1], 1, x._shape[2] }),
                    Reshape(w, new[] { w._shape[0], w._shape[1], 1, w._shape[2] }),
                    new[] { 1, s[0] }, new[] { 0, p[0] }, new[] { 1, d[0] });
                return Reshape(y, new[] { y._shape[0], y._shape[1], y._shape[3] });
            case 2: return Conv2D(x, w, s, p, d);
            default: return Conv3D(x, w, s, p, d);
        }
    }
}
