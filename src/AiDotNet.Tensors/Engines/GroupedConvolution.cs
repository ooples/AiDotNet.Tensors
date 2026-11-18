using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TensorLayout = AiDotNet.Tensors.LinearAlgebra.TensorLayout;

namespace AiDotNet.Tensors.Engines;

/// <summary>Internal ordinary grouped convolution without changing the public engine ABI.</summary>
internal static class GroupedConvolution
{
    private static CpuEngine? Canonical(IEngine engine)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_GROUPED_CONV2D") == "0") return null;
        var unwrapped = engine.Unwrap();
        if (!ReferenceEquals(engine, unwrapped)) return null;
        if (engine.Unwrap().GetType() == typeof(DirectGpuTensorEngine) && Gpu.AutocastScope.IsEnabled) return null;
        return engine.Unwrap().GetType() == typeof(CpuEngine) || engine.Unwrap().GetType() == typeof(DirectGpuTensorEngine)
            ? engine.Cast<CpuEngine>() : null;
    }

    internal static GroupedConv2DGeometry Validate<T>(Tensor<T> input, Tensor<T> kernel,
        int[] stride, int[] padding, int[] dilation, int groups)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4 || kernel.Rank != 4) throw new ArgumentException("Grouped convolution requires rank-four input and kernel.");
        if (stride is null || padding is null || dilation is null || stride.Length != 2 || padding.Length != 2 || dilation.Length != 2)
            throw new ArgumentException("Convolution geometry must contain exactly two axes.");
        var result = new GroupedConv2DGeometry(input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3],
            kernel.Shape[0], kernel.Shape[2], kernel.Shape[3], stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups);
        if (kernel.Shape[1] != result.InChannelsPerGroup)
            throw new ArgumentException("Compact kernel input channels must equal input channels divided by groups.", nameof(kernel));
        return result;
    }

    internal static Tensor<T> Forward<T>(IEngine engine, Tensor<T> input, Tensor<T> kernel,
        int[] stride, int[] padding, int[] dilation, int groups)
    {
        var geometry = Validate(input, kernel, stride, padding, dilation, groups);
        var canonical = Canonical(engine);
        if (canonical is null || input.Layout != TensorLayout.Nchw || kernel.Layout != TensorLayout.Nchw)
            return SplitForward(engine, input, kernel, geometry);
        if (!NoGradScope<T>.IsSuppressed) GradientTape<T>.Current?.BindEngineIfUnset(engine);
        var savedStride = geometry.Stride; var savedPadding = geometry.Padding; var savedDilation = geometry.Dilation;
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            return scope.RecordBinary(LazyNodeType.Custom, "GroupedConv2D", input, kernel, geometry.OutputShape,
                (current, output) => ForwardInto(current, output, input, kernel, geometry),
                BackwardFunctions<T>.GroupedConv2DBackward, new object[] { savedStride, savedPadding, savedDilation, groups });
        }
        Tensor<T> result;
        if (canonical is DirectGpuTensorEngine gpu) result = gpu.GroupedConv2DOutput(input, kernel, geometry);
        else
        {
            result = TensorAllocator.RentUninitialized<T>(geometry.OutputShape);
            ForwardInto(engine, result, input, kernel, geometry);
        }
        DifferentiableOps.RecordBinary("GroupedConv2D", result, input, kernel,
            BackwardFunctions<T>.GroupedConv2DBackward, new object[] { savedStride, savedPadding, savedDilation, groups });
        return result;
    }

    internal static void RequireShape<T>(Tensor<T> tensor, int[] shape, string name)
    {
        if (tensor is null) throw new ArgumentNullException(name);
        if (!tensor._shape.AsSpan().SequenceEqual(shape)) throw new ArgumentException("Grouped convolution tensor shape mismatch.", name);
    }

    internal static void ForwardInto<T>(IEngine engine, Tensor<T> output, Tensor<T> input,
        Tensor<T> kernel, in GroupedConv2DGeometry geometry)
    {
        RequireShape(input, geometry.InputShape, nameof(input));
        RequireShape(kernel, geometry.KernelShape, nameof(kernel));
        RequireShape(output, geometry.OutputShape, nameof(output));
        if (input.Layout == TensorLayout.Nchw && kernel.Layout == TensorLayout.Nchw && output.Layout == TensorLayout.Nchw
            && Canonical(engine) is { } cpu) cpu.GroupedConv2DInto(output, input, kernel, geometry);
        else DirectGpuTensorEngine.CopyResultInto(engine, SplitForward(engine, input, kernel, geometry), output);
    }

    internal static void BackwardInto<T>(IEngine engine, Tensor<T> gradOutput, Tensor<T> input, Tensor<T> kernel,
        Tensor<T>? gradInput, Tensor<T>? gradKernel, in GroupedConv2DGeometry geometry,
        bool accumulateInput, bool accumulateKernel)
    {
        RequireShape(input, geometry.InputShape, nameof(input));
        RequireShape(kernel, geometry.KernelShape, nameof(kernel));
        RequireShape(gradOutput, geometry.OutputShape, nameof(gradOutput));
        if (gradInput is not null) RequireShape(gradInput, geometry.InputShape, nameof(gradInput));
        if (gradKernel is not null) RequireShape(gradKernel, geometry.KernelShape, nameof(gradKernel));
        if (input.Layout == TensorLayout.Nchw && kernel.Layout == TensorLayout.Nchw && gradOutput.Layout == TensorLayout.Nchw
            && (gradInput is null || gradInput.Layout == TensorLayout.Nchw)
            && (gradKernel is null || gradKernel.Layout == TensorLayout.Nchw) && Canonical(engine) is { } cpu)
        {
            cpu.GroupedConv2DBackwardInto(gradOutput, input, kernel, gradInput, gradKernel, geometry,
                accumulateInput, accumulateKernel);
            return;
        }
        // A custom engine keeps every existing Conv2D derivative override.
        SplitBackwardInto(engine, gradOutput, input, kernel, gradInput, gradKernel, geometry, accumulateInput, accumulateKernel);
    }

    internal static Tensor<T> SplitForward<T>(IEngine engine, Tensor<T> input, Tensor<T> kernel,
        in GroupedConv2DGeometry g)
    {
        if (g.Groups == 1) return engine.Conv2D(input, kernel, g.Stride, g.Padding, g.Dilation);
        var parts = new Tensor<T>[g.Groups];
        for (int group = 0; group < g.Groups; ++group)
        {
            var x = engine.TensorSlice(input, new[] { 0, group * g.InChannelsPerGroup, 0, 0 },
                new[] { g.Batch, g.InChannelsPerGroup, g.InHeight, g.InWidth });
            var w = engine.TensorSlice(kernel, new[] { group * g.OutChannelsPerGroup, 0, 0, 0 },
                new[] { g.OutChannelsPerGroup, g.InChannelsPerGroup, g.KernelH, g.KernelW });
            parts[group] = engine.Conv2D(x, w, g.Stride, g.Padding, g.Dilation);
        }
        return engine.TensorConcatenate(parts, 1);
    }

    private static void SplitBackwardInto<T>(IEngine engine, Tensor<T> gradOutput, Tensor<T> input, Tensor<T> kernel,
        Tensor<T>? gradInput, Tensor<T>? gradKernel, in GroupedConv2DGeometry g, bool addInput, bool addKernel)
    {
        var inputs = gradInput is null ? null : new Tensor<T>[g.Groups];
        var kernels = gradKernel is null ? null : new Tensor<T>[g.Groups];
        for (int group = 0; group < g.Groups; ++group)
        {
            var dy = engine.TensorSlice(gradOutput, new[] { 0, group * g.OutChannelsPerGroup, 0, 0 },
                new[] { g.Batch, g.OutChannelsPerGroup, g.OutHeight, g.OutWidth });
            if (inputs is not null)
            {
                var w = engine.TensorSlice(kernel, new[] { group * g.OutChannelsPerGroup, 0, 0, 0 },
                    new[] { g.OutChannelsPerGroup, g.InChannelsPerGroup, g.KernelH, g.KernelW });
                inputs[group] = engine.Conv2DBackwardInput(dy, w,
                    new[] { g.Batch, g.InChannelsPerGroup, g.InHeight, g.InWidth }, g.Stride, g.Padding, g.Dilation);
            }
            if (kernels is not null)
            {
                var x = engine.TensorSlice(input, new[] { 0, group * g.InChannelsPerGroup, 0, 0 },
                    new[] { g.Batch, g.InChannelsPerGroup, g.InHeight, g.InWidth });
                kernels[group] = engine.Conv2DBackwardKernel(dy, x,
                    new[] { g.OutChannelsPerGroup, g.InChannelsPerGroup, g.KernelH, g.KernelW }, g.Stride, g.Padding, g.Dilation);
            }
        }
        void Commit(Tensor<T> target, Tensor<T>[] pieces, int axis, bool add)
        {
            var value = engine.TensorConcatenate(pieces, axis);
            if (add) value = engine.TensorAdd(target, value);
            DirectGpuTensorEngine.CopyResultInto(engine, value, target);
        }
        if (gradInput is not null && inputs is not null) Commit(gradInput, inputs, 1, addInput);
        if (gradKernel is not null && kernels is not null) Commit(gradKernel, kernels, 0, addKernel);
    }
}
