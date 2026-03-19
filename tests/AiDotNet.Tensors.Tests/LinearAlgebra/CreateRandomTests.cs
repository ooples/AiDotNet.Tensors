using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

public class CreateRandomTests
{
    [Fact]
    public void Vector_SameSeed_ProducesSameValues()
    {
        var v1 = Vector<float>.CreateRandom(new Random(42), 100);
        var v2 = Vector<float>.CreateRandom(new Random(42), 100);

        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(v1[i], v2[i]);
        }
    }

    [Fact]
    public void Vector_DifferentSeeds_ProducesDifferentValues()
    {
        var v1 = Vector<float>.CreateRandom(new Random(42), 100);
        var v2 = Vector<float>.CreateRandom(new Random(99), 100);

        bool anyDifferent = false;
        for (int i = 0; i < 100; i++)
        {
            if (v1[i] != v2[i])
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Vectors with different seeds should differ.");
    }

    [Fact]
    public void Matrix_SameSeed_ProducesSameValues()
    {
        var m1 = Matrix<float>.CreateRandom(new Random(42), 10, 10);
        var m2 = Matrix<float>.CreateRandom(new Random(42), 10, 10);

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                Assert.Equal(m1[i, j], m2[i, j]);
            }
        }
    }

    [Fact]
    public void Matrix_DifferentSeeds_ProducesDifferentValues()
    {
        var m1 = Matrix<float>.CreateRandom(new Random(42), 10, 10);
        var m2 = Matrix<float>.CreateRandom(new Random(99), 10, 10);

        bool anyDifferent = false;
        for (int i = 0; i < 10 && !anyDifferent; i++)
        {
            for (int j = 0; j < 10 && !anyDifferent; j++)
            {
                if (m1[i, j] != m2[i, j])
                    anyDifferent = true;
            }
        }

        Assert.True(anyDifferent, "Matrices with different seeds should differ.");
    }

    [Fact]
    public void Tensor_SameSeed_ProducesSameValues()
    {
        var t1 = Tensor<float>.CreateRandom(new Random(42), 4, 5, 3);
        var t2 = Tensor<float>.CreateRandom(new Random(42), 4, 5, 3);

        for (int i = 0; i < t1.Length; i++)
        {
            Assert.Equal(t1[i], t2[i]);
        }
    }

    [Fact]
    public void Tensor_DifferentSeeds_ProducesDifferentValues()
    {
        var t1 = Tensor<float>.CreateRandom(new Random(42), 4, 5, 3);
        var t2 = Tensor<float>.CreateRandom(new Random(99), 4, 5, 3);

        bool anyDifferent = false;
        for (int i = 0; i < t1.Length; i++)
        {
            if (t1[i] != t2[i])
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Tensors with different seeds should differ.");
    }

    [Fact]
    public void Parameterless_CreateRandom_IsNonDeterministic()
    {
        var v1 = Vector<float>.CreateRandom(10);
        var v2 = Vector<float>.CreateRandom(10);

        bool anyDifferent = false;
        for (int i = 0; i < 10; i++)
        {
            if (v1[i] != v2[i])
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent, "Parameterless CreateRandom should produce different values across calls.");
    }
}
