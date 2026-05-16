using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ScalarKernelTests
{
    [Fact]
    public void Gemm_StubExistsButNotImplemented_ThrowsNotImplemented()
    {
        // Span<T> and readonly ref struct cannot be captured by lambdas (CS8175),
        // so we call Gemm directly and catch the expected NotImplementedException.
        var threw = false;
        try
        {
            double[] cArr = new double[4];
            var options = new BlasOptions<double>();
            BlasManagedLib.Gemm<double>(
                new ReadOnlySpan<double>(new double[2]), 2, false,
                new ReadOnlySpan<double>(new double[2]), 2, false,
                cArr.AsSpan(), 2, 1, 2, 2, options);
        }
        catch (NotImplementedException)
        {
            threw = true;
        }
        Assert.True(threw, "Expected NotImplementedException from BlasManaged.Gemm stub.");
    }

    [Fact]
    public void PackingMode_HasFiveValues()
    {
        var values = Enum.GetValues(typeof(PackingMode));
        Assert.Equal(5, values.Length);
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.Auto));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForcePackBoth));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForcePackAOnly));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.ForceStreaming));
        Assert.True(Enum.IsDefined(typeof(PackingMode), PackingMode.DisableAutotune));
    }

    [Fact]
    public void BlasOptions_DefaultValues_AreSafe()
    {
        var options = new BlasOptions<double>();
        Assert.Equal(PackingMode.Auto, options.PackingMode);
        Assert.Equal(FusedActivationType.None, options.Epilogue.Activation);
        Assert.True(options.Epilogue.BiasN.IsEmpty);
        Assert.True(options.Epilogue.SkipMxN.IsEmpty);
        Assert.Equal(0u, options.Epilogue.DropoutMask);
        Assert.Equal(0.0, options.Epilogue.OutputScale);  // T=double; default(T)==0.0
        Assert.Equal(0, options.NumThreads);
        Assert.Equal(0UL, options.AutotuneKey);
        Assert.Equal(0L, options.MaxJitCacheBytes);
        Assert.Null(options.PackedA);
        Assert.Null(options.PackedB);
        Assert.True(options.Workspace.IsEmpty);
    }

    [Fact]
    public void BlasOptions_CanSetAllFields()
    {
        Span<byte> ws = new byte[64];
        Span<double> bias = new double[4];
        var options = new BlasOptions<double>
        {
            PackingMode = PackingMode.ForcePackBoth,
            Epilogue = new Epilogue<double>
            {
                BiasN = bias,
                Activation = FusedActivationType.ReLU,
                OutputScale = 2.0,
            },
            Workspace = ws,
            NumThreads = -1,
            AutotuneKey = 42UL,
            MaxJitCacheBytes = 1024L * 1024,
        };
        Assert.Equal(PackingMode.ForcePackBoth, options.PackingMode);
        Assert.Equal(FusedActivationType.ReLU, options.Epilogue.Activation);
        Assert.Equal(2.0, options.Epilogue.OutputScale);
        Assert.Equal(-1, options.NumThreads);
    }
}
