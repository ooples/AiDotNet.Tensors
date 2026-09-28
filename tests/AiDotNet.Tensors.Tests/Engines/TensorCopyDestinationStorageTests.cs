using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// TensorCopy must write into the destination's live storage, whatever that storage looks like.
/// </summary>
/// <remarks>
/// It used to write into <c>destination.GetDataArray()</c>, which returns the backing array only when it is
/// exactly the tensor (offset 0, storage length equal to Length). A pooled tensor whose rented array is longer,
/// or a view at a non-zero offset, got a fresh copy instead, and the whole write was silently discarded. That is
/// how AiDotNet's FTRL and ASGD optimizers, which assign their result to a parameter with TensorCopy, never
/// updated pooled model parameters at all.
/// </remarks>
public class TensorCopyDestinationStorageTests
{
    internal static Tensor<float> Filled(int[] shape, float start)
    {
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = start + i;
        return t;
    }

    [Fact]
    public void TensorCopy_IntoARowViewAtANonZeroOffset_WritesTheSharedStorage()
    {
        var engine = new CpuEngine();
        var matrix = Filled(new[] { 3, 4 }, 100f);
        var row = matrix.Slice(1);                       // a view: elements 4..7 of the matrix's storage
        Assert.Equal(4, row.Length);

        var source = Filled(new[] { 4 }, 1f);
        long versionBefore = row.Version;
        engine.TensorCopy(source, row);

        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, row.AsSpan().ToArray());
        Assert.Equal(new[] { 100f, 101f, 102f, 103f, 1f, 2f, 3f, 4f, 108f, 109f, 110f, 111f }, matrix.AsSpan().ToArray());
        Assert.True(row.Version > versionBefore, "a write through TensorCopy must bump the destination's version");
    }

    /// <summary>
    /// A write through one view reaches every alias's mutation epoch. The per-tensor Version is local to the
    /// object written, so a cache keyed on an alias's Version never saw the write.
    /// </summary>
    [Fact]
    public void TensorCopy_IntoAView_AdvancesTheParentsStorageMutationVersion()
    {
        var engine = new CpuEngine();
        var matrix = Filled(new[] { 3, 4 }, 100f);
        var row = matrix.Slice(1);
        int parentEpoch = matrix.StorageMutationVersion;
        long parentVersion = matrix.Version;

        engine.TensorCopy(Filled(new[] { 4 }, 1f), row);

        Assert.True(matrix.StorageMutationVersion > parentEpoch, "the parent shares the written storage, so its epoch must move");
        Assert.Equal(parentVersion, matrix.Version);
    }
}

/// <summary>The fused MatMul chain is opt-in (AIDOTNET_CROSS_LAYER_FUSION), a process-wide switch, so this runs serialized with the other compilation-state tests.</summary>
[Collection("CompilationGlobalState")]
public class TensorCopyAliasCompiledPlanTests
{
    /// <summary>
    /// The compiled plan fuses x·W1·W2 into a cached W1·W2 product and refreshed it only when W1.Version or
    /// W2.Version changed. Overwriting W1 through an alias left both unchanged, so the plan kept training on
    /// the pre-copy weights. It now keys on the storage epoch.
    /// </summary>
    [Fact]
    public void CompiledPlan_FusedMatMulChain_SeesAWeightOverwrittenThroughAnAlias()
    {
        var prior = AiDotNetEngine.Current;
        string? priorFusion = Environment.GetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION");
        Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", "1");
        var priorOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
            new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableDataflowFusion = true });
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var x = TensorCopyDestinationStorageTests.Filled(new[] { 2, 3 }, 0.1f);
            var w1 = TensorCopyDestinationStorageTests.Filled(new[] { 3, 4 }, 0.2f);
            var w2 = TensorCopyDestinationStorageTests.Filled(new[] { 4, 5 }, 0.3f);
            Tensor<float> Loss() => engine.ReduceSum(engine.TensorMatMul(engine.TensorMatMul(x, w1), w2), null);

            // Prove the fused W1*W2 chain is what gets compiled: the same graph without cross-layer fusion must have
            // more forward steps. Otherwise the loss comparison below could pass on two ordinary MatMuls.
            Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", null);
            int unfusedSteps;
            using (var scope = GraphMode.Enable())
            {
                Loss();
                using var unfused = scope.CompileTraining(new[] { w1, w2 });
                unfusedSteps = unfused.ForwardStepCount;
            }

            Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", "1");
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Loss();
                plan = scope.CompileTraining(new[] { w1, w2 });
            }

            Assert.True(plan.ForwardStepCount < unfusedSteps,
                $"cross-layer fusion did not fuse the MatMul chain ({plan.ForwardStepCount} forward steps, {unfusedSteps} unfused)");

            try
            {
                plan.Step();
                var alias = w1.Reshape(new[] { 12 });         // same storage, a different tensor object
                engine.TensorCopy(TensorCopyDestinationStorageTests.Filled(new[] { 12 }, -1f), alias);

                float expected = Loss().GetFlattenedData()[0];
                float compiled = plan.Step()[0];
                Assert.True(Math.Abs(expected - compiled) <= 1e-3f * Math.Max(1f, Math.Abs(expected)),
                    $"compiled loss {compiled} after overwriting W1 through an alias; the eager loss is {expected}");
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            Environment.SetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION", priorFusion);
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(priorOptions);
        }
    }
}
