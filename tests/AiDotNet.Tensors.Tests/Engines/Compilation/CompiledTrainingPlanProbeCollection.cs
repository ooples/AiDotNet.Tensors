using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// xUnit collection definition for tests that mutate
/// <c>CompiledTrainingPlan&lt;T&gt;.StepProbe</c>, a process-wide static
/// diagnostic delegate. Concurrent tests setting/reading this static would
/// race, producing flaky probe logs and false failures. Tests that touch
/// the probe reference this collection via
/// <c>[Collection("CompiledTrainingPlanProbeSerial")]</c> so they run
/// serially. Mirrors the WeightRegistry collection convention.
/// </summary>
[CollectionDefinition("CompiledTrainingPlanProbeSerial", DisableParallelization = true)]
public sealed class CompiledTrainingPlanProbeCollection
{
}
