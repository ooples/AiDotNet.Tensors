using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A compiled plan is keyed by op structure and output shapes. That is sound for parameters, which
/// the plan binds by identity, and for constants that never change. It is NOT sound for a constant
/// the step computes FROM ITS INPUT: structure and shapes are identical on every step, so the plan
/// is reused while the baked-in value goes stale, and the result is silently wrong.
/// </summary>
/// <remarks>
/// This is not hypothetical. A cross-entropy whose ignore_index reduction divided by a
/// supervised-example count materialised from the target tensor computed exactly the right loss
/// eagerly (2.196550 for a uniform 9-class head, every gradient finite) and produced NaN on step 1
/// when replayed from a compiled plan.
/// </remarks>
[Collection("EngineCurrentGlobalState")]
public class DataDerivedConstantHashTests : IDisposable
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public DataDerivedConstantHashTests()
    {
        AutoTrainingCompiler.ResetState();
        AutoTrainingCompiler.Enabled = true;
        AutoTrainingCompiler.TestMinForwardElementsOverride = 1;
    }

    public void Dispose()
    {
        AutoTrainingCompiler.TestMinForwardElementsOverride = null;
        AutoTrainingCompiler.ResetState();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Builds a tape whose graph multiplies a parameter by a constant leaf, mirroring a loss that
    /// derives a scale from its data. Returns the hash the compiler would key a plan on.
    /// </summary>
    private long HashFor(Tensor<float> parameter, Tensor<float> derivedConstant, out bool compilable)
    {
        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

        var scaled = _engine.TensorMultiply(parameter, derivedConstant);
        var loss = _engine.ReduceSum(scaled, null);

        compilable = AutoTrainingCompiler.TryComputeStructureHash(
            tape.Entries, tape.EntryCount, new[] { parameter }, out long hash);

        GC.KeepAlive(loss);
        return hash;
    }

    [Fact]
    public void ConstantDerivedFromData_ChangesTheHash_SoAStalePlanCannotBeReused()
    {
        var parameter = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < parameter.Length; i++) parameter[i] = 1.0f;

        // Same shape, same op structure, DIFFERENT value - exactly what a per-batch derived
        // constant looks like to the compiler.
        var firstStep = new Tensor<float>(new[] { 2, 2 });
        var secondStep = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < firstStep.Length; i++)
        {
            firstStep[i] = 2.0f;
            secondStep[i] = 8.0f;
        }

        long hashA = HashFor(parameter, firstStep, out bool compilableA);
        long hashB = HashFor(parameter, secondStep, out bool compilableB);

        Assert.True(compilableA);
        Assert.True(compilableB);
        Assert.True(
            hashA != hashB,
            "A constant derived from the step's data must change the plan key. Hashing only op " +
            "names and output shapes makes both steps look identical, so the plan is replayed " +
            "with the first step's value baked in.");
    }

    [Fact]
    public void ParameterValuesDoNotChangeTheHash_SoTrainingStillReusesItsPlan()
    {
        var parameter = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < parameter.Length; i++) parameter[i] = 1.0f;

        var constant = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < constant.Length; i++) constant[i] = 3.0f;

        long before = HashFor(parameter, constant, out _);

        // Weights move on every step; if that invalidated the plan, nothing would ever be reused
        // and compiling would be pointless.
        for (int i = 0; i < parameter.Length; i++) parameter[i] = 42.0f;

        long after = HashFor(parameter, constant, out _);

        Assert.True(
            before == after,
            "Parameter values must NOT participate in the plan key - the plan binds them by " +
            "identity and they change every step by design.");
    }

    [Fact]
    public void ConstantTooLargeToHash_IsRefusedRatherThanAssumedStable()
    {
        var parameter = new Tensor<float>(new[] { 1, 8192 });
        for (int i = 0; i < parameter.Length; i++) parameter[i] = 1.0f;

        // Past the hashing cap: the compiler cannot afford to fingerprint it every step, and
        // assuming it never changes is the very bug this guards. Refusing is the safe answer.
        var oversized = new Tensor<float>(new[] { 1, 8192 });
        for (int i = 0; i < oversized.Length; i++) oversized[i] = 1.0f;

        HashFor(parameter, oversized, out bool compilable);

        Assert.False(
            compilable,
            "A non-parameter leaf larger than the hashing cap must make the tape uncompilable, " +
            "so no plan is stored or reused for it.");
    }

    [Fact]
    public void WithoutSources_HashIsStructureOnly_PreservingTheRecordStepFastPath()
    {
        var parameter = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < parameter.Length; i++) parameter[i] = 1.0f;

        var constant = new Tensor<float>(new[] { 2, 2 });
        for (int i = 0; i < constant.Length; i++) constant[i] = 5.0f;

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var loss = _engine.ReduceSum(_engine.TensorMultiply(parameter, constant), null);

        // RecordStep hashes without sources: it only needs to notice that the step SHAPE repeated,
        // and it cannot tell parameters from constants without the source list. That path must keep
        // its cheap structure-only behaviour.
        long structureOnly = AutoTrainingCompiler.ComputeStructureHash(tape.Entries, tape.EntryCount);

        Assert.NotEqual(0L, structureOnly);
        GC.KeepAlive(loss);
    }
}
