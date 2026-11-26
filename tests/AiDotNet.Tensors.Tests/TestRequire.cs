using Xunit.Sdk;

namespace AiDotNet.Tensors.Tests;

/// <summary>
/// A value a test needs in order to go on. A missing one fails the test with its name, where a null-forgiving
/// operator would fail later with a NullReferenceException that names nothing.
/// </summary>
internal static class TestRequire
{
    internal static TValue NotNull<TValue>(TValue? value, string what)
        where TValue : class
        => value ?? throw new XunitException($"The test needs {what}, which is null.");
}
