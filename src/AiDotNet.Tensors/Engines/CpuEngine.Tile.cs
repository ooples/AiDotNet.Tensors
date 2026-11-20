using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // Repeating one axis repeats whole contiguous slabs. For [B,1,C] -> [B,L,C] the
    // slab has C elements. For [1,L,C] -> [B,L,C] it contains the whole input.
    internal static bool TryGetTileSlab(int[] input, int[] output, out int width, out int repeats)
    {
        width = 1;
        repeats = 1;
        if (input.Length != output.Length) return false;
        for (int d = input.Length - 1; d >= 0; d--)
        {
            if (input[d] <= 0 || output[d] <= 0) return false;
            if (repeats == 1) width *= input[d];
            if (input[d] == output[d]) continue;
            if (repeats != 1 || output[d] % input[d] != 0) return false;
            repeats = output[d] / input[d];
        }
        return true;
    }

    private static bool TryCopyTileInto<T>(Tensor<T> source, Tensor<T> destination, int[] expectedShape)
    {
        if (source.IsSparse || destination.IsSparse || !source.IsContiguous || !destination.IsContiguous
            || !destination._shape.AsSpan().SequenceEqual(expectedShape)
            || !TryGetTileSlab(source._shape, expectedShape, out int width, out int repeats)) return false;

        using var output = destination.LeaseWritable();
        using var input = source.Lease();
        if (input.Span.Overlaps(output.Span)) return false;
        int targetWidth = width * repeats;
        for (int src = 0, dst = 0; src < source.Length; src += width, dst += targetWidth)
        {
            var block = output.Span.Slice(dst, targetWidth);
            input.Span.Slice(src, width).CopyTo(block);
            for (int filled = width; filled < targetWidth;)
            {
                int count = Math.Min(filled, targetWidth - filled);
                block.Slice(0, count).CopyTo(block.Slice(filled));
                filled += count;
            }
        }
        destination.IncrementVersion();
        return true;
    }
}
