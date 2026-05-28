using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #375: the background autotuner is default-ON in production, but its concurrent
/// measurement load perturbs timing-sensitive tests. Disable it for the whole test
/// assembly at load time; the tests that exercise it re-enable locally inside the
/// serialized BlasManaged-Stats-Serial collection (so the worker never overlaps others).
/// </summary>
internal static class BackgroundAutotunerTestSwitch
{
    [ModuleInitializer]
    internal static void DisableForTests() => BackgroundAutotuner.Enabled = false;
}
