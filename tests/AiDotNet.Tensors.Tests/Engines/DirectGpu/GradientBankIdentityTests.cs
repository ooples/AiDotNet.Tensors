using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GradientBankIdentityTests
{
    [Theory]
    [InlineData(64, false)]
    [InlineData(16, false)]
    [InlineData(64, true)]
    [InlineData(16, true)]
    public void CompilerPrivateBankDoesNotBecomeAnEscapedFlatParameter(int length, bool enabled)
    {
        var previous = DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled;
        try
        {
            DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = enabled;
            var array = new float[64];
            var owner = ResultBufferTracker.TrackFreshPrivateArray(array, 0)!;
            using var tensor = Tensor<float>.FromMemory(array.AsMemory(0, length), new[] { length });
            Assert.False(owner.State.Escaped);
            Assert.Equal(!enabled && length < array.Length, DirectGpuTensorEngine.IsFlatParameterView(tensor));
            Assert.Equal(!enabled, owner.State.Escaped);
        }
        finally { DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = previous; }
    }

    [Fact]
    public void CallerExposedPrivateArrayKeepsExistingViewBehavior()
    {
        var previous = DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled;
        try
        {
            DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = true;
            var array = new float[64];
            var owner = ResultBufferTracker.TrackFreshPrivateArray(array, 0)!;
            using var tensor = Tensor<float>.FromMemory(array.AsMemory(0, 16), new[] { 16 });
            Assert.Equal(0f, tensor.AsSpan()[0]);
            Assert.True(owner.State.Escaped);
            Assert.True(DirectGpuTensorEngine.IsFlatParameterView(tensor));
            Assert.True(owner.State.Escaped);
        }
        finally { DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = previous; }
    }

    [Fact]
    public void GenuineParameterBufferAndCallerOwnedViewsKeepTheirFlatIdentity()
    {
        var previous = DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled;
        try
        {
            DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = true;
            var buffer = new ParameterBuffer<float>(new[] { new[] { 8 }, new[] { 16 } });
            foreach (var view in buffer.CreateAllViews())
            {
                using (view) Assert.True(DirectGpuTensorEngine.IsFlatParameterView(view));
            }
            using var ordinary = Tensor<float>.FromMemory(new float[64].AsMemory(0, 16), new[] { 16 });
            Assert.True(DirectGpuTensorEngine.IsFlatParameterView(ordinary));
        }
        finally { DirectGpuTensorEngine.CompiledGradientBankIdentityEnabled = previous; }
    }

    [Fact]
    public void IdentityQueryNeitherAddsWrappersNorExposesArrayContents()
    {
        var array = new float[64];
        var owner = ResultBufferTracker.TrackFreshPrivateArray(array, 0)!;
        using var tensor = new Tensor<float>(array, new[] { 64 });
        int wrappers = owner.State.Wrappers;
        for (int i = 0; i < 16; i++) Assert.True(ResultBufferTracker.IsUnexposedPrivateArray(array));
        Assert.Equal(wrappers, owner.State.Wrappers);
        Assert.False(owner.State.Escaped);
    }
}
