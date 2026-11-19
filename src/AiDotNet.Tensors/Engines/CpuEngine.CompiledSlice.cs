using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // A traced slice already owns a fixed destination. Replaying the public allocating slice then copying that
    // result creates another payload on every step. Copy canonical dense layouts directly, retaining the existing
    // allocating implementation for sparse/strided layouts, empty boxes, overlap and custom-engine semantics.
    internal static bool TryCopyCompiledSliceInto<T>(Tensor<T> input, int[] start, int[] length, Tensor<T> output)
    {
        int rank = input.Rank;
        if (rank == 0 || start.Length != rank || length.Length != rank
            || !input.IsContiguous || !output.IsContiguous || output.Length == 0
            || !output._shape.AsSpan().SequenceEqual(length)) return false;

        for (int axis = 0; axis < rank; axis++)
        {
            if (start[axis] < 0 || start[axis] >= input._shape[axis])
                throw new ArgumentOutOfRangeException(nameof(start),
                    $"Start index {start[axis]} out of range for axis {axis} with size {input._shape[axis]}");
            if (length[axis] < 1 || length[axis] > input._shape[axis] - start[axis])
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Slice length {length[axis]} starting at {start[axis]} exceeds axis {axis} size {input._shape[axis]}");
        }

        using var keepInput = new KeepAliveScope(input);
        using var keepOutput = new KeepAliveScope(output);
        // Establish writable ownership first: a copy-on-write detach may rebind related views. Read the source
        // from its current storage afterward, rather than retain an unmarked span across that transition.
        Span<T> destination = output.DataUnmarked.Span.Slice(0, output.Length);
        ReadOnlySpan<T> source = input.ReadOnlyDataUnmarked.Span.Slice(0, input.Length);
        // A multi-row copy into overlapping storage can clobber a later source row. The existing slice snapshots
        // before publication, so leave this exceptional case on that path rather than alter its alias behavior.
        if (source.Overlaps(destination)) return false;

        int sourceStart = 0;
        int stride = 1;
        bool partialAxisSeen = false;
        bool singleBlock = true;
        for (int axis = rank - 1; axis >= 0; axis--)
        {
            sourceStart += start[axis] * stride;
            if (partialAxisSeen && length[axis] != 1) singleBlock = false;
            if (length[axis] != input._shape[axis]) partialAxisSeen = true;
            stride *= input._shape[axis];
        }
        if (singleBlock)
        {
            source.Slice(sourceStart, output.Length).CopyTo(destination);
            output.IncrementVersion();
            return true;
        }

        int rowWidth = length[rank - 1];
        int rows = output.Length / rowWidth;
        for (int row = 0; row < rows; row++)
        {
            int remainder = row;
            int inputOffset = start[rank - 1];
            stride = input._shape[rank - 1];
            for (int axis = rank - 2; axis >= 0; axis--)
            {
                int coordinate = remainder % length[axis];
                remainder /= length[axis];
                inputOffset += (start[axis] + coordinate) * stride;
                stride *= input._shape[axis];
            }
            source.Slice(inputOffset, rowWidth).CopyTo(destination.Slice(row * rowWidth, rowWidth));
        }
        output.IncrementVersion();
        return true;
    }
}
