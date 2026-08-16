using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public sealed class GotoGemmRoutingTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(47)]
    public void IsPreferredForThreadBudget_RejectsSmallAndMidSizedMachines(int threadBudget)
    {
        Assert.False(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }

    [Theory]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(128)]
    public void IsPreferredForThreadBudget_AllowsManyCoreMachines(int threadBudget)
    {
        Assert.True(GotoGemmFp32.IsPreferredForThreadBudget(threadBudget));
    }
}
