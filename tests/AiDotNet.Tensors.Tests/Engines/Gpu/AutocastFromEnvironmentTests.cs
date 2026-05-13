using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: validates the AIDOTNET_AUTOCAST env-var entry point that
/// lets consumer code (AiDotNet's NeuralNetworkBase.Train) opt into
/// mixed-precision training without a build-time T-parameter change to
/// the model.
/// </summary>
public class AutocastFromEnvironmentTests
{
    private static System.IDisposable SetEnv(string? value)
    {
        var prior = System.Environment.GetEnvironmentVariable("AIDOTNET_AUTOCAST");
        System.Environment.SetEnvironmentVariable("AIDOTNET_AUTOCAST", value);
        return new EnvScope(prior);
    }

    private sealed class EnvScope : System.IDisposable
    {
        private readonly string? _prior;
        public EnvScope(string? prior) { _prior = prior; }
        public void Dispose() => System.Environment.SetEnvironmentVariable("AIDOTNET_AUTOCAST", _prior);
    }

    [Fact]
    public void EnableFromEnvironment_Unset_ReturnsNull()
    {
        using (SetEnv(null))
        {
            var scope = AutocastScope.EnableFromEnvironment();
            Assert.Null(scope);
        }
    }

    [Fact]
    public void EnableFromEnvironment_EmptyString_ReturnsNull()
    {
        using (SetEnv(""))
        {
            var scope = AutocastScope.EnableFromEnvironment();
            Assert.Null(scope);
        }
    }

    [Fact]
    public void EnableFromEnvironment_Fp16_ReturnsActiveScope()
    {
        using (SetEnv("fp16"))
        {
            using var scope = AutocastScope.EnableFromEnvironment();
            Assert.NotNull(scope);
            Assert.Equal(PrecisionMode.Float16, scope!.Precision);
            Assert.True(AutocastScope.IsEnabled);
            Assert.Equal(PrecisionMode.Float16, AutocastScope.ActivePrecision);
        }
        // Scope disposes — autocast no longer active.
        Assert.False(AutocastScope.IsEnabled);
    }

    [Fact]
    public void EnableFromEnvironment_Float16_CaseInsensitive()
    {
        using (SetEnv("FLOAT16"))
        {
            using var scope = AutocastScope.EnableFromEnvironment();
            Assert.NotNull(scope);
            Assert.Equal(PrecisionMode.Float16, scope!.Precision);
        }
    }

    [Fact]
    public void EnableFromEnvironment_Half_RecognizedAlias()
    {
        using (SetEnv("half"))
        {
            using var scope = AutocastScope.EnableFromEnvironment();
            Assert.NotNull(scope);
            Assert.Equal(PrecisionMode.Float16, scope!.Precision);
        }
    }

    [Fact]
    public void EnableFromEnvironment_Fp8_ReturnsFloat8E5M2Scope()
    {
        using (SetEnv("fp8"))
        {
            using var scope = AutocastScope.EnableFromEnvironment();
            Assert.NotNull(scope);
            Assert.Equal(PrecisionMode.Float8E5M2, scope!.Precision);
        }
    }

    [Fact]
    public void EnableFromEnvironment_UnrecognizedValue_ReturnsNull()
    {
        using (SetEnv("garbage-value"))
        {
            var scope = AutocastScope.EnableFromEnvironment();
            Assert.Null(scope);
        }
    }

    [Fact]
    public void EnvironmentRequestedPrecision_Unset_ReturnsFloat32()
    {
        using (SetEnv(null))
        {
            Assert.Equal(PrecisionMode.Float32, AutocastScope.EnvironmentRequestedPrecision());
        }
    }

    [Fact]
    public void EnvironmentRequestedPrecision_Fp16_ReturnsFloat16()
    {
        using (SetEnv("fp16"))
        {
            Assert.Equal(PrecisionMode.Float16, AutocastScope.EnvironmentRequestedPrecision());
        }
    }
}
