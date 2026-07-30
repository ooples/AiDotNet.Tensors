using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Ground-truth repro for the BatchNormalizationLayer running-stat staleness
/// behind the #7 clone-bucket failures (WebRTCVad / TOTEM / TableTransformer /
/// Zamba2 Clone_AfterTraining_ShouldPreserveLearnedWeights).
///
/// The layer updates running mean/variance with an IN-PLACE EMA on a PERSISTENT
/// tensor (BatchNormalizationLayer.cs:694-700):
///     runningVar *= momentum
///     runningVar += (1-momentum) * batchVar
/// Under GraphMode these in-place ops are recorded as deferred LazyNodes
/// (LazyTensorScope.RecordInPlace) so CompiledTrainingPlan replay re-applies
/// them per Step (#350). The running-stat update is a SIDE EFFECT — it is NOT
/// reachable from the loss output — so the question these tests pin down is:
/// after a compiled training Step (and after the scope disposes), is the
/// persistent running-stat tensor MATERIALIZED to its post-EMA value, or does
/// it read stale (pre-EMA) until some later lazy read forces realization?
///
/// If stale: the trained model's first Predict reads stale running stats while
/// a deserialized clone (whose serialization already read/materialized them)
/// reads fresh — producing the probe-0-only divergence the clone test reports.
/// </summary>
[Collection("CompilationGlobalState")]
public class RunningStatInPlaceMaterializationTests
{
    /// <summary>
    /// After a compiled training Step that records an in-place EMA on a
    /// persistent running-variance tensor, the persistent tensor must hold the
    /// post-EMA value. momentum=0.9, batchVar=0.5, init=1.0
    /// => expected = 0.9*1.0 + 0.1*0.5 = 0.95.
    /// </summary>
    [Fact]
    public void PersistentRunningStat_InPlaceEMA_MaterializedAfterCompiledStep()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;
        const double momentum = 0.9;
        const double batchVarConst = 0.5;
        const double expected = momentum * 1.0 + (1.0 - momentum) * batchVarConst; // 0.95

        var rng = new System.Random(42);
        var inputData = new double[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;

        // Persistent layer state — survives across Steps, init to 1.0 (the BN default).
        var runningVar = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) runningVar[i] = 1.0;

        // A constant standing in for the per-batch variance feeding the EMA.
        var batchVar = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) batchVar[i] = batchVarConst;

        // Trainable params so CompileTraining has something to optimize.
        var input = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < input.Length; i++) input[i] = inputData[i];
        var gamma = new Tensor<double>(new[] { features });
        var beta = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gamma[i] = 1.0; beta[i] = 0.0; }

        ICompiledTrainingPlan<double> plan;
        bool srcSetAfterCompile;
        double rawAfterCompile;
        using (var scope = GraphMode.Enable())
        {
            // Trainable forward producing a loss (gamma/beta are the params).
            var bn = engine.BatchNorm(input, gamma, beta, epsilon, out _, out _);
            engine.ReduceSum(bn, null);

            // Side-effect running-stat EMA — exactly the layer's pattern.
            engine.TensorMultiplyScalarInPlace(runningVar, momentum);
            var scaled = engine.TensorMultiplyScalar(batchVar, 1.0 - momentum);
            engine.TensorAddInPlace(runningVar, scaled);

            plan = scope.CompileTraining(new[] { gamma, beta });
        }
        // LazySource non-null after compile => the in-place node was NOT
        // scheduled (dropped from `optimized`), so the plan won't execute it.
        srcSetAfterCompile = runningVar.LazySource is not null;
        rawAfterCompile = ((double[])(object)runningVar.GetDataArray())[0];

        double rawAfterStep, idxAfterStep;
        bool srcSetAfterStep;
        int fwdStepCount = plan.ForwardStepCount;
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
            // RAW backing FIRST (bypasses EnsureMaterialized), THEN indexer
            // (triggers EnsureMaterialized → may realize a pending LazySource).
            rawAfterStep = ((double[])(object)runningVar.GetDataArray())[0];
            srcSetAfterStep = runningVar.LazySource is not null;
            idxAfterStep = runningVar[0];
        }
        double afterDispose = runningVar[0];

        Assert.True(
            System.Math.Abs(rawAfterStep - expected) < 1e-9,
            $"Persistent running-stat materialization probe:\n" +
            $"  LazySource set after Compile = {srcSetAfterCompile}  (true => in-place node DROPPED from plan)\n" +
            $"  raw backing  after Compile   = {rawAfterCompile:F12}\n" +
            $"  plan ForwardStepCount        = {fwdStepCount}\n" +
            $"  LazySource set after Step    = {srcSetAfterStep}\n" +
            $"  RAW backing[0] after Step    = {rawAfterStep:F12}\n" +
            $"  indexer[0]   after Step      = {idxAfterStep:F12}\n" +
            $"  indexer[0]   after Dispose   = {afterDispose:F12}\n" +
            $"  expected (post-EMA)          = {expected:F12}\n" +
            $"  (raw==1.0 && idx==0.95 => in-place side-effect left DEFERRED/stale)");
        Assert.True(
            System.Math.Abs(idxAfterStep - expected) < 1e-9,
            $"Indexer probe after Step diverged from expected:\n" +
            $"  indexer[0]   after Step      = {idxAfterStep:F12}\n" +
            $"  expected (post-EMA)          = {expected:F12}\n" +
            $"  (divergence => clone-divergence path: indexer triggers materialization but raw backing is stale)");
        Assert.True(
            System.Math.Abs(afterDispose - expected) < 1e-9,
            $"Indexer probe after Dispose diverged from expected:\n" +
            $"  indexer[0]   after Dispose   = {afterDispose:F12}\n" +
            $"  expected (post-EMA)          = {expected:F12}\n" +
            $"  (divergence => post-disposal value changed unexpectedly)");
    }

    /// <summary>
    /// Eager-tape control: the SAME in-place EMA outside GraphMode (under a
    /// plain GradientTape, which is how non-fused training runs) must update
    /// the persistent tensor immediately. Pins that the staleness is specific
    /// to the compiled/graph path, not the in-place ops themselves.
    /// </summary>
    [Fact]
    public void PersistentRunningStat_InPlaceEMA_EagerTape_UpdatesImmediately()
    {
        var engine = new CpuEngine();
        const int features = 4;
        const double momentum = 0.9;
        const double batchVarConst = 0.5;
        const double expected = momentum * 1.0 + (1.0 - momentum) * batchVarConst; // 0.95

        var runningVar = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) runningVar[i] = 1.0;
        var batchVar = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) batchVar[i] = batchVarConst;

        using (var tape = new GradientTape<double>())
        {
            engine.TensorMultiplyScalarInPlace(runningVar, momentum);
            var scaled = engine.TensorMultiplyScalar(batchVar, 1.0 - momentum);
            engine.TensorAddInPlace(runningVar, scaled);
        }

        Assert.True(
            System.Math.Abs(runningVar[0] - expected) < 1e-9,
            $"Eager-tape in-place EMA did not update immediately: runningVar[0]={runningVar[0]:F12}, expected {expected:F12}");
    }
}
