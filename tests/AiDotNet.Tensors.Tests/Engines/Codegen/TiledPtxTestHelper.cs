// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

internal static class TiledPtxTestHelper
{
    internal static double[][] CreateInputs(CodegenKernelSpec spec, out float[][] host)
    {
        var inputs = new double[spec.Inputs.Count][];
        host = new float[spec.Inputs.Count][];
        for (int i = 0; i < spec.Inputs.Count; i++)
        {
            host[i] = new float[spec.Inputs[i].ElementCount];
            inputs[i] = new double[host[i].Length];
            for (int e = 0; e < host[i].Length; e++)
            {
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
                inputs[i][e] = host[i][e];
            }
        }
        return inputs;
    }

    internal static void AssertClose(
        double[] expected, float[] actual, double tolerance, string label,
        bool relative = false)
    {
        double worst = 0;
        int at = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double difference = System.Math.Abs(expected[i] - actual[i]);
            if (relative)
                difference /= System.Math.Max(1.0, System.Math.Abs(expected[i]));
            if (difference > worst) { worst = difference; at = i; }
        }
        Assert.True(worst < tolerance,
            $"{label} differs by {worst:E3} at {at}: " +
            $"expected {expected[at]}, actual {actual[at]}");
    }

    internal static void AssertClose(
        float[] expected, float[] actual, double tolerance, string label,
        bool relative = false)
    {
        var widened = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++) widened[i] = expected[i];
        AssertClose(widened, actual, tolerance, label, relative);
    }

    internal static unsafe void LaunchThree(
        DirectPtxModule module, IntPtr function,
        IntPtr first, IntPtr second, IntPtr output,
        uint blocks, uint blockX, uint blockY)
    {
        IntPtr p0 = first, p1 = second, p2 = output;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        module.Launch(function, blocks, 1, 1, blockX, blockY, 1, 0, arguments);
    }

    internal static unsafe void LaunchFour(
        DirectPtxModule module, IntPtr function,
        IntPtr first, IntPtr second, IntPtr third, IntPtr output,
        uint blocks, uint blockX, uint blockY)
    {
        IntPtr p0 = first, p1 = second, p2 = third, p3 = output;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        module.Launch(function, blocks, 1, 1, blockX, blockY, 1, 0, arguments);
    }

    internal static CodegenAxis[] CopyAxes(CodegenKernelSpec spec)
    {
        var axes = new CodegenAxis[spec.Space.Axes.Count];
        for (int i = 0; i < axes.Length; i++) axes[i] = spec.Space.Axes[i];
        return axes;
    }

    internal static CodegenTensorBinding[] CopyInputs(CodegenKernelSpec spec)
    {
        var inputs = new CodegenTensorBinding[spec.Inputs.Count];
        for (int i = 0; i < inputs.Length; i++) inputs[i] = spec.Inputs[i];
        return inputs;
    }

    internal static int[] CopyProductInputs(CodegenKernelSpec spec)
    {
        var inputs = new int[spec.ProductInputs.Count];
        for (int i = 0; i < inputs.Length; i++) inputs[i] = spec.ProductInputs[i];
        return inputs;
    }

    internal static CodegenTensorBinding WithShapeDimension(
        CodegenTensorBinding binding, int dimension, int extent)
    {
        var shape = new int[binding.Shape.Count];
        var map = new CodegenAffineExpr[binding.Map.Count];
        var indirect = new CodegenIndirectIndex?[binding.Indirect.Count];
        for (int i = 0; i < shape.Length; i++) shape[i] = binding.Shape[i];
        for (int i = 0; i < map.Length; i++) map[i] = binding.Map[i];
        for (int i = 0; i < indirect.Length; i++) indirect[i] = binding.Indirect[i];
        shape[dimension] = extent;
        return new CodegenTensorBinding(
            binding.ParameterIndex, binding.Name, shape, map,
            binding.IsOutput, binding.ElementType, indirect);
    }
}
