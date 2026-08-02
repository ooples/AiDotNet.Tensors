using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Gradient-correctness guards for Gather's backward when indices REPEAT.
///
/// Gather backward must scatter-ADD: position k of the gathered result came from source slice
/// indices[k], so a slice selected k times must receive k contributions. Accumulation is not an
/// optimization - repeated indices are the norm rather than the exception:
///   * embedding lookups, where the same token appears many times in a batch
///   * relative-position-bias tables (Swin), where many (i, j) window positions map to the same
///     relative-position bucket
///
/// The original ScatterAddBackward looped over the SOURCE dim and did
///     gradInput[d] = gradOutput[indices[d % len]]
/// which walked the wrong axis, ASSIGNED instead of accumulating, and bounds-checked the target
/// index against the OUTPUT extent although it indexes the SOURCE. With an all-ones upstream
/// gradient it wrote 1 into every slot it touched - including never-selected slices, which must be
/// exactly 0. Downstream that produced analytical gradients too small by a non-uniform factor and
/// with signs that need not match the true sum (AiDotNet SwinTransformerBlockLayer disagreed with
/// finite differences on 6/6 sampled scalars).
/// </summary>
public class GatherBackwardRepeatedIndicesTests
{
    [Fact]
    public void GatherBackward_WithRepeatedIndices_AccumulatesPerOccurrence()
    {
        var engine = AiDotNetEngine.Current;

        // 4 source rows, gather 3 of them as [0, 0, 1]. The source deliberately has MORE rows than
        // the gather count so grad-wrt-source [4,2] and grad-wrt-output [3,2] have DIFFERENT
        // shapes: with equal shapes an all-ones result is ambiguous, since it is also exactly
        // d(loss)/d(gathered).
        var table = new Tensor<double>(new[] { 4, 2 });
        for (int i = 0; i < table.Length; i++) table[i] = i + 1;

        var indices = new Tensor<int>(new[] { 3 });
        indices[0] = 0;
        indices[1] = 0;
        indices[2] = 1;

        using var tape = new GradientTape<double>();
        var gathered = engine.TensorGather(table, indices, axis: 0);
        var allAxes = new int[gathered.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
        var loss = engine.ReduceSum(gathered, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(loss, new[] { table });
        Assert.True(grads.TryGetValue(table, out var grad) && grad is not null,
            "no gradient was produced for the gathered source table");

        Assert.Equal(new[] { 4, 2 }, grad!.Shape.ToArray());

        // loss = sum(gathered) => d(loss)/d(table[r, :]) == number of times row r was selected.
        double[] expectedPerRow = { 2.0, 1.0, 0.0, 0.0 };
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(expectedPerRow[r], grad[r * 2 + c], precision: 10);
            }
        }
    }

    [Fact]
    public void GatherBackward_WithoutRepeats_IsUnchanged()
    {
        var engine = AiDotNetEngine.Current;

        // Regression guard for the non-repeating case the old code happened to get right, so the
        // accumulation fix cannot silently double-count.
        var table = new Tensor<double>(new[] { 4, 2 });
        for (int i = 0; i < table.Length; i++) table[i] = i + 1;

        var indices = new Tensor<int>(new[] { 2 });
        indices[0] = 2;
        indices[1] = 0;

        using var tape = new GradientTape<double>();
        var gathered = engine.TensorGather(table, indices, axis: 0);
        var allAxes = new int[gathered.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++) allAxes[i] = i;
        var loss = engine.ReduceSum(gathered, allAxes, keepDims: false);

        var grads = tape.ComputeGradients(loss, new[] { table });
        Assert.True(grads.TryGetValue(table, out var grad) && grad is not null,
            "no gradient was produced for the gathered source table");

        double[] expectedPerRow = { 1.0, 0.0, 1.0, 0.0 };
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                Assert.Equal(expectedPerRow[r], grad![r * 2 + c], precision: 10);
            }
        }
    }
}
