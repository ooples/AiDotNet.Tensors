using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

[Collection(nameof(TensorArenaPinnedTests))]
public sealed class ArenaIssuedLinearAlgebraLifetimeTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Matrix_and_vector_results_kept_after_a_tape_arena_are_not_reused_by_later_arenas(bool matrix)
    {
        bool previousPooling = TensorPool.Enabled;
        TensorPool.Enabled = true;
        TensorArena.ClearPersistentPool();
        try
        {
            // Large enough for cross-arena pooling, with identical sizes for both allocation APIs.
            const int side = 256;
            Func<float[]> readKept;
            using (var arena = TensorArena.Create())
            {
                arena.OwnedByTape = true;
                if (matrix)
                {
                    var kept = MatrixAllocator.Rent<float>(side, side);
                    for (int r = 0; r < side; r++)
                        for (int c = 0; c < side; c++) kept[r, c] = r * 0.25f - c * 0.5f;
                    readKept = () => kept.AsSpan().ToArray();
                }
                else
                {
                    var kept = VectorAllocator.Rent<float>(side * side);
                    for (int i = 0; i < kept.Length; i++) kept[i] = (i / side) * 0.25f - (i % side) * 0.5f;
                    readKept = () => kept.AsSpan().ToArray();
                }
            }
            // Mimic subsequent steps which aggressively overwrite rents of exactly the retained result's size.
            for (int step = 0; step < 5; step++)
            {
                using var arena = TensorArena.Create();
                arena.OwnedByTape = true;
                MatrixAllocator.Rent<float>(side, side).AsWritableSpan().Fill(1000f + step);
                VectorAllocator.Rent<float>(side * side).AsWritableSpan().Fill(-1000f - step);
            }
            var values = readKept();
            Assert.Equal(side * side, values.Length);
            for (int i = 0; i < values.Length; i++)
                Assert.Equal((i / side) * 0.25f - (i % side) * 0.5f, values[i]);
        }
        finally
        {
            TensorPool.Enabled = previousPooling;
            TensorArena.ClearPersistentPool();
        }
    }
}
