using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Source-parity guard for the #775 Gaussian-splat kernels (covariance R*S^2*R^T + spherical-harmonics
/// color eval/backward) across the backends that carry them. Only OpenCL is runtime-validated here;
/// CUDA/HIP are source-validated, so this asserts their kernel source uses the SAME quaternion->rotation
/// entries and SH basis constants as the OpenCL reference, catching a divergent blind port.
/// </summary>
public sealed class GaussianSplatKernelSourceTests
{
    private const string CudaSpatial = "AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels.CudaSpatialTransformerKernels";
    private const string HipSpatial = "AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipSpatialTransformerKernels";
    private const string OpenClSpatial = "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels.SpatialTransformerKernels";
    private const string MetalExt = "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalExtendedConvKernels";

    [Theory]
    [InlineData(CudaSpatial, "GetSource")]
    [InlineData(HipSpatial, "GetSource")]
    [InlineData(OpenClSpatial, "GetSource")]
    [InlineData(MetalExt, "Source")]
    public void QuaternionRotationEntry_MatchesAcrossBackends(string typeName, string memberName)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains("float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);", source, StringComparison.Ordinal);
        Assert.Contains("float r12 = 2.0f * (qy * qz - qw * qx);", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(CudaSpatial, "GetSource")]
    [InlineData(HipSpatial, "GetSource")]
    [InlineData(OpenClSpatial, "GetSource")]
    [InlineData(MetalExt, "Source")]
    public void SphericalHarmonicsBasisConstants_MatchAcrossBackends(string typeName, string memberName)
    {
        string source = GetStaticString(typeName, memberName);
        Assert.Contains("basis[0] = 0.282095f;", source, StringComparison.Ordinal);
        Assert.Contains("basis[6] = 0.315392f * (3.0f * dz * dz - 1.0f);", source, StringComparison.Ordinal);
    }

    // CUDA/HIP register these in GetKernelNames so their module compilation picks them up; OpenCL
    // registers them via an inline name array in OpenClBackend (not a reflected GetKernelNames), so it is
    // excluded here — its coverage is proven by the OpenCL runtime parity tests instead.
    [Theory]
    [InlineData(CudaSpatial)]
    [InlineData(HipSpatial)]
    public void AllThreeKernelNames_AreRegistered(string typeName)
    {
        string[] names = GetStaticStringArray(typeName, "GetKernelNames");
        Assert.Contains("gaussian_covariance", names);
        Assert.Contains("spherical_harmonics", names);
        Assert.Contains("spherical_harmonics_backward", names);
    }

    private static string GetStaticString(string typeName, string memberName) =>
        (string)(InvokeStatic(typeName, memberName)
            ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

    private static string[] GetStaticStringArray(string typeName, string memberName) =>
        (string[])(InvokeStatic(typeName, memberName)
            ?? throw new InvalidOperationException($"{memberName} returned null for {typeName}"));

    private static object? InvokeStatic(string typeName, string memberName)
    {
        Type type = typeof(DirectGpuTensorEngine).Assembly.GetType(typeName)
            ?? throw new InvalidOperationException($"Kernel source type not found: {typeName}");
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
        MethodInfo? method = type.GetMethod(memberName, flags, binder: null, Type.EmptyTypes, modifiers: null);
        if (method is not null) return method.Invoke(null, null);
        FieldInfo? field = type.GetField(memberName, flags);
        if (field is not null) return field.GetValue(null);
        PropertyInfo? property = type.GetProperty(memberName, flags);
        if (property is not null) return property.GetValue(null);
        throw new InvalidOperationException($"Static member not found: {typeName}.{memberName}");
    }
}
