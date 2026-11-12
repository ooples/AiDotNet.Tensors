using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The normalized mesh Laplacian is D^(-1/2)·L·D^(-1/2) with D the diagonal of the cotangent Laplacian. It used to take
/// D from −L, which is negative, so every scale was 0 and the normalized Laplacian was identically zero.
/// </summary>
public class NormalizedMeshLaplacianTests
{
    [Fact]
    public void Normalized_IsTheSymmetricallyScaledCotangentLaplacian()
    {
        var engine = new CpuEngine();
        // A tetrahedron with acute faces: every cotangent weight is positive, so every degree is too.
        var vertices = new Tensor<double>(new[]
        {
            0.0, 0.0, 0.0,
            1.0, 0.1, 0.0,
            0.4, 0.9, 0.1,
            0.5, 0.4, 0.95,
        }, new[] { 4, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3 }, new[] { 4, 3 });

        var cot = engine.ComputeMeshLaplacian(vertices, faces, LaplacianType.Cotangent);
        var normalized = engine.ComputeMeshLaplacian(vertices, faces, LaplacianType.Normalized);

        for (int i = 0; i < 4; i++)
        {
            Assert.True(cot[i * 4 + i] > 0, $"degree {i} should be positive, got {cot[i * 4 + i]}");
            Assert.Equal(1.0, normalized[i * 4 + i], 12);
            for (int j = 0; j < 4; j++)
            {
                double expected = cot[i * 4 + j] / Math.Sqrt(cot[i * 4 + i] * cot[j * 4 + j]);
                Assert.Equal(expected, normalized[i * 4 + j], 12);
            }
        }
    }
}
