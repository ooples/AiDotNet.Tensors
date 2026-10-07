using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The plan census (AIDOTNET_PLAN_CENSUS) reports per-step action counts and the cost of each op, aggregated by
/// name. Warm-up steps are excluded from the timings.
/// </summary>
public class PlanCensusTests
{
    [Fact]
    public void Report_CountsActionsAndAggregatesCostByName_IgnoringWarmup()
    {
        var census = new PlanCensus(
            new[] { "generic:Reshape", "specialized:MatMul", "generic:Reshape" },
            new[] { "specialized:MatMul", "generic:Reshape" },
            forwardStepCount: 4);
        Assert.Equal(3, census.ForwardActionCount);
        Assert.Equal(2, census.BackwardActionCount);

        // Warm-up steps: nothing they record may reach the report.
        for (int s = 0; s < PlanCensus.WarmupSteps; s++)
        {
            Assert.False(census.Collecting);
            census.AddForward(0, 1_000_000);
            census.EndStep(1_000_000);
        }
        Assert.True(census.Collecting);

        // Timed steps short of the report: each Reshape costs 2 ticks forward, the MatMul 5.
        for (int s = 0; s < PlanCensus.ReportAfterSteps - 1; s++)
        {
            census.AddForward(0, 2);
            census.AddForward(1, 5);
            census.AddForward(2, 2);
            census.AddBackward(0, 7);
            census.AddBackward(1, 1);
            census.EndStep(20);
        }
        Assert.True(census.Collecting);

        string report = census.BuildReport();
        Assert.Contains("forward steps=4 forward actions=3 backward actions=2 per-step actions=5", report);
        // Both Reshapes land on one line with count 2; the MatMul is listed separately.
        Assert.Matches(@"\n\s+2\s+[0-9.]+\s+generic:Reshape", report);
        Assert.Matches(@"\n\s+1\s+[0-9.]+\s+specialized:MatMul", report);
    }
}
