using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue B (#370) task B.1: verifies the <see cref="BlasMode"/> enum and its
/// plumbing through <see cref="BlasOptions{T}"/> and <see cref="BlasManagedLib.DefaultMode"/>.
///
/// <para>
/// Default everywhere must be <see cref="BlasMode.Deterministic"/> — the spec
/// (issue #368) requires bit-exact reproducibility unless the caller explicitly
/// opts into <see cref="BlasMode.Fast"/>.
/// </para>
/// </summary>
public class BlasModeTest
{
    [Fact]
    public void Default_BlasOptions_Mode_Is_Deterministic()
    {
        BlasOptions<float> opts = default;
        Assert.Equal(BlasMode.Deterministic, opts.Mode);
    }

    [Fact]
    public void BlasManaged_DefaultMode_Is_Deterministic()
    {
        Assert.Equal(BlasMode.Deterministic, BlasManagedLib.DefaultMode);
    }

    [Fact]
    public void Caller_Can_Set_Fast_Mode_Via_BlasOptions()
    {
        var opts = new BlasOptions<float> { Mode = BlasMode.Fast };
        Assert.Equal(BlasMode.Fast, opts.Mode);
    }

    [Fact]
    public void BlasMode_Enum_Has_Expected_Members()
    {
        // Deterministic = 0 (so default(BlasMode) lands here)
        Assert.Equal(0, (int)BlasMode.Deterministic);
        Assert.Equal(1, (int)BlasMode.Fast);
    }
}
