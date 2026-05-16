using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;

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
}
