using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>Checks the real fixture's setup/teardown under disagreeing global and thread-local modes.</summary>
[Collection("BlasManaged-Stats-Serial")]
public sealed class DeterministicStrategyFixtureIsolationTests
{
    [Theory]
    [InlineData(false, null)]
    [InlineData(true, null)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void FixtureRestoresGlobalAndThreadLocalModesIndependently(bool globalMode, bool? localMode)
    {
        bool? priorLocal = BlasProvider.GetThreadLocalDeterministicMode();
        BlasProvider.SetThreadLocalDeterministicMode(null);
        bool priorGlobal = BlasProvider.IsDeterministicMode;
        try
        {
            BlasProvider.SetDeterministicMode(globalMode);
            BlasProvider.SetThreadLocalDeterministicMode(localMode);
            using (var fixture = new DeterministicStrategySelectionTests())
            {
                Assert.True(BlasProvider.IsDeterministicMode);
                Assert.Null(BlasProvider.GetThreadLocalDeterministicMode());
            }

            Assert.Equal(localMode, BlasProvider.GetThreadLocalDeterministicMode());
            BlasProvider.SetThreadLocalDeterministicMode(null);
            Assert.Equal(globalMode, BlasProvider.IsDeterministicMode);
        }
        finally
        {
            BlasProvider.SetThreadLocalDeterministicMode(null);
            BlasProvider.SetDeterministicMode(priorGlobal);
            BlasProvider.SetThreadLocalDeterministicMode(priorLocal);
        }
    }
}
