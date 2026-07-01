// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class SparseOptimizerBackendParityTests
{
    private static readonly string[] SparseBackendMethods =
    {
        "SparseSgdUpdate",
        "SparseSgdMomentumUpdate",
        "SparseAdamUpdate",
        "SparseAdamWUpdate",
        "SparseRmspropUpdate",
        "SparseAdagradUpdate",
        "SparseNagUpdate",
        "SparseAdadeltaUpdate",
        "SparseAmsgradUpdate",
        "SparseAdamaxUpdate",
        "SparseLionUpdate",
        "SparseNadamUpdate",
        "SparseFtrlUpdate",
        "SparseProximalL1Update"
    };

    private static readonly string[] SparseKernelNames =
    {
        "sparse_sgd_update",
        "sparse_sgd_momentum_update",
        "sparse_adam_update",
        "sparse_adamw_update",
        "sparse_rmsprop_update",
        "sparse_adagrad_update",
        "sparse_nag_update",
        "sparse_adadelta_update",
        "sparse_amsgrad_update",
        "sparse_adamax_update",
        "sparse_lion_update",
        "sparse_nadam_update",
        "sparse_ftrl_update",
        "sparse_proximal_l1_update"
    };

    [Fact]
    public void SparseOptimizerReference_DecodesNumericAndBitPackedIndices()
    {
        Assert.Equal(0, SparseOptimizerReference.DecodeIndex(0f, 4));
        Assert.Equal(2, SparseOptimizerReference.DecodeIndex(2f, 4));
        Assert.Equal(1, SparseOptimizerReference.DecodeIndex(Int32BitsToSingle(1), 4));
        Assert.Equal(2, SparseOptimizerReference.DecodeIndex(Int32BitsToSingle(2), 4));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SparseOptimizerReference.DecodeIndex(Int32BitsToSingle(6), 4));
    }

    [Fact]
    public void SparseReferenceUpdatesOnlyIndexedSlots_WithBitPackedIndices()
    {
        var param = new[] { 10f, 20f, 30f, 40f };
        var indices = new[] { Int32BitsToSingle(2), Int32BitsToSingle(1) };
        var values = new[] { 3f, 4f };

        SparseOptimizerReference.SparseSgdUpdate(param, indices, values, nnz: 2, learningRate: 0.5f, weightDecay: 0f);

        Assert.Equal(10f, param[0]);
        Assert.Equal(18f, param[1]);
        Assert.Equal(28.5f, param[2]);
        Assert.Equal(40f, param[3]);
    }

    [Fact]
    public void SparseBackends_ImplementEveryDirectBackendMethod()
    {
        string[] backendFiles =
        {
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA", "CudaBackend.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP", "HipBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "OpenCL", "OpenClBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Metal", "MetalBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Vulkan", "VulkanBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "WebGpuBackend.SparseOptimizer.cs")
        };

        foreach (string file in backendFiles)
        {
            string source = File.ReadAllText(file);
            foreach (string method in SparseBackendMethods)
                Assert.Contains("void " + method + "(", source);
        }
    }

    [Fact]
    public void NonCudaSparseBackends_DoNotContainSparseStubs()
    {
        string[] backendFiles =
        {
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP", "HipBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "OpenCL", "OpenClBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Metal", "MetalBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "Vulkan", "VulkanBackend.SparseOptimizer.cs"),
            SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "WebGpuBackend.SparseOptimizer.cs")
        };
        string[] forbidden =
        {
            "NotSupportedException",
            "SparseNotImpl",
            "CUDA-only",
            "does not yet ship",
            "throw Sparse"
        };

        foreach (string file in backendFiles)
        {
            string source = File.ReadAllText(file);
            foreach (string token in forbidden)
            {
                Assert.True(
                    source.IndexOf(token, StringComparison.OrdinalIgnoreCase) < 0,
                    $"{Path.GetFileName(file)} still contains sparse stub marker: {token}");
            }
        }
    }

    [Fact]
    public void NativeSparseOptimizerKernelRegistrations_AreComplete()
    {
        string cuda = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA", "Kernels", "CudaOptimizerKernels.cs"));
        string hip = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP", "Kernels", "HipOptimizerKernels.cs"));
        string openCl = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "OpenCL", "Kernels", "OptimizerKernels.cs"));
        string webGpuKernels = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "WebGpuKernels.cs"));
        string webGpuBackend = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "WebGpuBackend.SparseOptimizer.cs"));

        foreach (string name in SparseKernelNames)
        {
            Assert.Contains("void " + name + "(", cuda);
            Assert.Contains("\"" + name + "\"", cuda);
            Assert.Contains("void " + name + "(", hip);
            Assert.Contains("\"" + name + "\"", hip);
            Assert.Contains("void " + name + "(", openCl);
            Assert.Contains("\"" + name + "\"", openCl);
            Assert.Contains("fn " + name + "(", webGpuKernels);
            Assert.Contains("\"" + name + "\"", webGpuBackend);
        }
    }

    [Fact]
    public void NativeSparseIndexDecoders_AreBoundsChecked()
    {
        string cuda = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA", "Kernels", "CudaOptimizerKernels.cs"));
        string hip = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "HIP", "Kernels", "HipOptimizerKernels.cs"));
        string openCl = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "OpenCL", "Kernels", "OptimizerKernels.cs"));
        string webGpu = File.ReadAllText(SourcePath("src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "WebGpuKernels.cs"));

        Assert.Contains("decode_sparse_index(float raw, int param_size)", cuda);
        Assert.Contains("decode_sparse_index(float raw, int param_size)", hip);
        Assert.Contains("decode_sparse_index(float raw, int param_size)", openCl);
        Assert.Contains("if (i < 0) return;", cuda);
        Assert.Contains("if (i < 0) return;", hip);
        Assert.Contains("if (i < 0) return;", openCl);
        Assert.Contains("param_size: u32", webGpu);
        Assert.Contains("if (i >= opt_params.param_size) { return; }", webGpu);
    }

    private static string SourcePath(params string[] relativeParts)
        => Path.Combine(FindRepoRoot(), Path.Combine(relativeParts));

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string marker = Path.Combine(dir.FullName, "src", "AiDotNet.Tensors", "AiDotNet.Tensors.csproj");
            if (File.Exists(marker))
                return dir.FullName;
            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate repository root from test output directory.");
    }

    private static unsafe float Int32BitsToSingle(int value) => *(float*)&value;
}
