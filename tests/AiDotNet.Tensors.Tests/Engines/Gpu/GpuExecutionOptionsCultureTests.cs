using System;
using System.Globalization;
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

    // Sets a variable and the culture for the scope, and restores both, the variable to whatever it held before.
    private sealed class Scope : IDisposable
    {
        private readonly string _variable;
        private readonly string? _previousValue;
        private readonly CultureInfo _previousCulture;

        public Scope(string name, string culture)
        {
            _variable = Prefix + name;
            _previousValue = Environment.GetEnvironmentVariable(_variable);
            _previousCulture = CultureInfo.CurrentCulture;
            CultureInfo.CurrentCulture = new CultureInfo(culture);
        }

        public void Set(string? value) => Environment.SetEnvironmentVariable(_variable, value);

        public void Dispose()
        {
            CultureInfo.CurrentCulture = _previousCulture;
            Environment.SetEnvironmentVariable(_variable, _previousValue);
        }
    }

    [Theory]
    [InlineData("de-DE")]
    [InlineData("fr-FR")]
    [InlineData("en-US")]
    public void ADecimalVariableParsesTheSameInEveryCulture(string culture)
    {
        const string Name = "TEST_1557_DOUBLE";
        using var scope = new Scope(Name, culture);
        scope.Set("0.5");
        Assert.True(GpuExecutionOptions.TryGetEnvDouble(Name, out double value));
        Assert.Equal(0.5, value);
    }

    [Fact]
    public void AnUnsetOrUnparsableDecimalVariableIsNotRead()
    {
        const string Name = "TEST_1557_DOUBLE_ABSENT";
        using var scope = new Scope(Name, "en-US");
        scope.Set(null);
        Assert.False(GpuExecutionOptions.TryGetEnvDouble(Name, out double unset));
        Assert.Equal(0.0, unset);
        scope.Set("half");
        Assert.False(GpuExecutionOptions.TryGetEnvDouble(Name, out _));
    }

    [Theory]
    [InlineData("de-DE")]
    [InlineData("en-US")]
    public void AnIntegerVariableRejectsGroupSeparatorsInEveryCulture(string culture)
    {
        const string Name = "TEST_1557_INT";
        using var scope = new Scope(Name, culture);
        scope.Set("4096");
        Assert.True(GpuExecutionOptions.TryGetEnvInt(Name, out int value));
        Assert.Equal(4096, value);
        // "4.096" is a thousands-grouped 4096 only under de-DE; a configuration value must not change meaning there.
        scope.Set("4.096");
        Assert.False(GpuExecutionOptions.TryGetEnvInt(Name, out _));
    }
}
