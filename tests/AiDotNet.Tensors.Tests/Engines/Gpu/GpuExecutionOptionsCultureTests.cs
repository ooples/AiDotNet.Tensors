using System;
using System.Globalization;
using System.Threading;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// AIDOTNET_GPU_* variables are machine configuration, so they parse in the invariant culture (#1557). Under de-DE the
/// current-culture parse read "0.5" as 5 ('.' is the group separator there), and MAX_MEMORY clamped it to 1.0.
/// </summary>
public class GpuExecutionOptionsCultureTests
{
    private const string Prefix = "AIDOTNET_GPU_";

    [Theory]
    [InlineData("de-DE")]
    [InlineData("fr-FR")]
    [InlineData("en-US")]
    public void ADecimalVariableParsesTheSameInEveryCulture(string culture)
    {
        const string Name = "TEST_1557_DOUBLE";
        var previous = CultureInfo.CurrentCulture;
        Environment.SetEnvironmentVariable(Prefix + Name, "0.5");
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.True(GpuExecutionOptions.TryGetEnvDouble(Name, out double value));
            Assert.Equal(0.5, value);
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
            Environment.SetEnvironmentVariable(Prefix + Name, null);
        }
    }

    [Fact]
    public void AnUnsetOrUnparsableDecimalVariableIsNotRead()
    {
        const string Name = "TEST_1557_DOUBLE_ABSENT";
        Environment.SetEnvironmentVariable(Prefix + Name, null);
        Assert.False(GpuExecutionOptions.TryGetEnvDouble(Name, out double unset));
        Assert.Equal(0.0, unset);
        Environment.SetEnvironmentVariable(Prefix + Name, "half");
        try
        {
            Assert.False(GpuExecutionOptions.TryGetEnvDouble(Name, out _));
        }
        finally
        {
            Environment.SetEnvironmentVariable(Prefix + Name, null);
        }
    }

    [Theory]
    [InlineData("de-DE")]
    [InlineData("en-US")]
    public void AnIntegerVariableRejectsGroupSeparatorsInEveryCulture(string culture)
    {
        const string Name = "TEST_1557_INT";
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Environment.SetEnvironmentVariable(Prefix + Name, "4096");
            Assert.True(GpuExecutionOptions.TryGetEnvInt(Name, out int value));
            Assert.Equal(4096, value);
            // "4.096" is a thousands-grouped 4096 only under de-DE; a configuration value must not change meaning there.
            Environment.SetEnvironmentVariable(Prefix + Name, "4.096");
            Assert.False(GpuExecutionOptions.TryGetEnvInt(Name, out _));
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
            Environment.SetEnvironmentVariable(Prefix + Name, null);
        }
    }
}
