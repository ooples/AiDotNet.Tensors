using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

// #1572 acceptance probe (not for merge): a GPU-namespace test that always skips. The CUDA lane must fail on it.
public class DeliberateSkipProbeTests
{
    [SkippableFact]
    public void AlwaysSkips() => Skip.If(true, "deliberate skip: proves the CUDA lane fails on a skipped GPU test");
}
