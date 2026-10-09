using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every OpenCL kernel name must come from exactly one compiled program. The kernel cache used to let a second program
/// replace the first's kernel of the same name, so a host call written for one signature launched another: mse_loss
/// (4 arguments set on a 5-argument kernel), hardtanh, contrastive_loss and others failed only on a real device.
/// </summary>
public class OpenClKernelNameUniquenessTests
{
    [SkippableFact]
    public void NoKernelNameIsRegisteredByTwoPrograms()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var duplicates = backend.ReregisteredKernelNames.OrderBy(n => n, System.StringComparer.Ordinal).ToList();
        Assert.True(duplicates.Count == 0,
            $"{duplicates.Count} kernel names were registered by more than one program: {string.Join(", ", duplicates)}");
    }
}
