using System;
using System.Buffers;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using TensorLayout = AiDotNet.Tensors.LinearAlgebra.TensorLayout;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // These math-only entry points are recorded once by GroupedConvolution. Direct array offsets and
    // destination spans preserve parent ownership; no independent FromMemory wrapper resets its epoch.
    internal virtual void GroupedConv2DInto<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel,
        in GroupedConv2DGeometry geometry)
    {
        RequireGroupedSources(input, kernel, geometry);
        GroupedConvolution.RequireShape(output, geometry.OutputShape, nameof(output));
        RequireGroupedNchw(output);
        using var noGrad = new NoGradScope<T>();
        using var keepInput = new KeepAliveScope(input);
        using var keepKernel = new KeepAliveScope(kernel);
        using var keepOutput = new KeepAliveScope(output);
        var owned = new List<Tensor<T>>();
        try
        {
            output.PrepareForInPlaceWrite();
            var source = GroupedReadSource(input, output, null, owned);
            var weights = GroupedReadSource(kernel, output, null, owned);
            var destination = GroupedDestination(output, owned);
            if (typeof(T) == typeof(float))
                GroupedForwardFloat((Tensor<float>)(object)destination, (Tensor<float>)(object)source,
                    (Tensor<float>)(object)weights, geometry);
            else GroupedForwardGeneric(destination, source, weights, geometry);
            if (!ReferenceEquals(destination, output)) CommitGroupedResult(output, destination);
            else output.IncrementVersion();
        }
        finally { foreach (var tensor in owned) tensor.Dispose(); }
    }

    internal virtual void GroupedConv2DBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> kernel,
        Tensor<T>? gradInput, Tensor<T>? gradKernel, in GroupedConv2DGeometry geometry,
        bool addInput, bool addKernel)
    {
        RequireGroupedSources(input, kernel, geometry);
        GroupedConvolution.RequireShape(gradOutput, geometry.OutputShape, nameof(gradOutput));
        RequireGroupedNchw(gradOutput);
        if (gradInput is not null) GroupedConvolution.RequireShape(gradInput, geometry.InputShape, nameof(gradInput));
        if (gradKernel is not null) GroupedConvolution.RequireShape(gradKernel, geometry.KernelShape, nameof(gradKernel));
        if (gradInput is not null) RequireGroupedNchw(gradInput);
        if (gradKernel is not null) RequireGroupedNchw(gradKernel);
        if (gradInput is null && gradKernel is null) return;
        using var noGrad = new NoGradScope<T>();
        using var keepInput = new KeepAliveScope(input);
        using var keepKernel = new KeepAliveScope(kernel);
        using var keepUpstream = new KeepAliveScope(gradOutput);
        using var keepGradInput = new KeepAliveScope(gradInput);
        using var keepGradKernel = new KeepAliveScope(gradKernel);
        var owned = new List<Tensor<T>>();
        try
        {
            gradInput?.PrepareForInPlaceWrite();
            gradKernel?.PrepareForInPlaceWrite();
            // Snapshot only actual primal/upstream overlap before either target is written. This also covers
            // distinct Memory wrappers of the same array. Target commits are deliberately dX, then dW.
            var source = gradKernel is null ? input : GroupedReadSource(input, gradInput, gradKernel, owned);
            var weights = gradInput is null ? kernel : GroupedReadSource(kernel, gradInput, gradKernel, owned);
            var upstream = GroupedReadSource(gradOutput, gradInput, gradKernel, owned);
            if (gradInput is not null)
            {
                var destination = GroupedDestination(gradInput, owned);
                if (!ReferenceEquals(destination, gradInput) && addInput) CopyGroupedLogical(gradInput, destination);
                if (typeof(T) == typeof(float))
                    GroupedInputGradientFloat((Tensor<float>)(object)destination, (Tensor<float>)(object)upstream,
                        (Tensor<float>)(object)weights, geometry, addInput);
                else GroupedInputGradientGeneric(destination, upstream, weights, geometry, addInput);
                if (!ReferenceEquals(destination, gradInput)) CommitGroupedResult(gradInput, destination);
                else gradInput.IncrementVersion();
            }
            if (gradKernel is not null)
            {
                var destination = GroupedDestination(gradKernel, owned);
                // Read an accumulating overlapping destination AFTER the complete dX commit.
                if (!ReferenceEquals(destination, gradKernel) && addKernel) CopyGroupedLogical(gradKernel, destination);
                if (typeof(T) == typeof(float))
                    GroupedKernelGradientFloat((Tensor<float>)(object)destination, (Tensor<float>)(object)source,
                        (Tensor<float>)(object)upstream, geometry, addKernel);
                else GroupedKernelGradientGeneric(destination, source, upstream, geometry, addKernel);
                if (!ReferenceEquals(destination, gradKernel)) CommitGroupedResult(gradKernel, destination);
                else gradKernel.IncrementVersion();
            }
        }
        finally { foreach (var tensor in owned) tensor.Dispose(); }
    }

    private static void RequireGroupedSources<T>(Tensor<T> input, Tensor<T> kernel, in GroupedConv2DGeometry geometry)
    {
        if (geometry.Groups <= 0) throw new ArgumentException("A constructed grouped-convolution geometry is required.");
        GroupedConvolution.RequireShape(input, geometry.InputShape, nameof(input));
        GroupedConvolution.RequireShape(kernel, geometry.KernelShape, nameof(kernel));
        if (input.IsSparse || kernel.IsSparse) throw new NotSupportedException("Grouped convolution requires dense tensors.");
        RequireGroupedNchw(input);
        RequireGroupedNchw(kernel);
    }

    private static void RequireGroupedNchw<T>(Tensor<T> tensor)
    {
        if (tensor.Layout != TensorLayout.Nchw)
            throw new NotSupportedException("The grouped CPU core requires canonical NCHW storage; other layouts retain the split engine path.");
    }

    private static bool GroupedStorageOverlap<T>(Tensor<T> source, Tensor<T>? destination)
    {
        if (destination is null || source.Length == 0 || destination.Length == 0) return false;
        if (source.IsContiguous && destination.IsContiguous)
            return source.AsSpanUnmarked().Overlaps(destination.AsSpanUnmarked());
        // A strided destination can use an independent MemoryManager over the same physical address;
        // storage/vector/array identity cannot disprove that overlap. This exceptional path snapshots sources.
        return true;
    }

    private static Tensor<T> GroupedReadSource<T>(Tensor<T> source, Tensor<T>? firstWrite, Tensor<T>? secondWrite,
        List<Tensor<T>> owned)
    {
        // This internal read accessor materializes deferred storage and synchronizes an existing
        // CPU backing array without exposing a writable/escaped alias or recording another op.
        _ = source.GetCpuBackingForStridedRead(out _);
        if (!source.IsContiguous)
        {
            var contiguous = source.Contiguous();
            owned.Add(contiguous);
            source = contiguous;
        }
        if (GroupedStorageOverlap(source, firstWrite) || GroupedStorageOverlap(source, secondWrite))
        {
            var snapshot = new Tensor<T>(source.AsSpanUnmarked().ToArray(), (int[])source._shape.Clone());
            owned.Add(snapshot);
            source = snapshot;
        }
        return source;
    }

    private static Tensor<T> GroupedDestination<T>(Tensor<T> destination, List<Tensor<T>> owned)
    {
        if (destination.IsSparse) throw new NotSupportedException("Grouped convolution destinations must be dense.");
        if (destination.IsContiguous) return destination;
        var scratch = new Tensor<T>((int[])destination._shape.Clone());
        owned.Add(scratch);
        return scratch;
    }

    private static Tensor<T> GroupedImageView<T>(Tensor<T> source, int image, int firstChannel, int channels)
    {
        var shape = new[] { 1, channels, source._shape[2], source._shape[3] };
        int offset = checked(source._storageOffset + image * source._strides[0] + firstChannel * source._strides[1]);
        return new Tensor<T>((Vector<T>)source.GraphVectorIdentity, shape, (int[])source._strides.Clone(), offset, source._storage)
            { Layout = source.Layout };
    }

    private static Tensor<T> GroupedKernelView<T>(Tensor<T> source, int group, in GroupedConv2DGeometry geometry)
    {
        var shape = new[] { geometry.OutChannelsPerGroup, geometry.InChannelsPerGroup, geometry.KernelH, geometry.KernelW };
        int offset = checked(source._storageOffset + group * geometry.OutChannelsPerGroup * source._strides[0]);
        return new Tensor<T>((Vector<T>)source.GraphVectorIdentity, shape, (int[])source._strides.Clone(), offset, source._storage)
            { Layout = source.Layout };
    }

    // Ordinary Conv2D dispatch has array-only fallbacks which copy an offset destination. Stream bounded
    // columns directly into the parent array instead; the large waveform route is independently offset-aware.
    private static void GroupedForwardFloat(Tensor<float> destination, Tensor<float> input,
        Tensor<float> kernel, in GroupedConv2DGeometry g)
    {
        if (g.Batch == 0) return;
        var output = destination.GetCpuBackingForContiguousWrite(out int outputOffset);
        var source = input.GetCpuBackingForStridedRead(out int inputOffset);
        var weights = kernel.GetCpuBackingForStridedRead(out int kernelOffset);
        if (output is null || source is null || weights is null)
        {
            GroupedForwardGeneric(destination, input, kernel, g);
            return;
        }
        int k = checked(g.InChannelsPerGroup * g.KernelH * g.KernelW);
        int spatial = checked(g.OutHeight * g.OutWidth), columns = Math.Min(256, spatial);
        float[]? panel = null;
        try
        {
            for (int group = 0; group < g.Groups; group++)
                for (int image = 0; image < g.Batch; image++)
                {
                    int x = checked(inputOffset + (image * g.InChannels + group * g.InChannelsPerGroup) * g.InHeight * g.InWidth);
                    int w = checked(kernelOffset + group * g.OutChannelsPerGroup * k);
                    int y = checked(outputOffset + (image * g.OutChannels + group * g.OutChannelsPerGroup) * spatial);
                    if (TryConv1DColumnPanels(source, weights, output, 1, g.InChannelsPerGroup, g.InHeight, g.InWidth,
                        g.OutChannelsPerGroup, g.KernelH, g.KernelW, g.StrideH, g.StrideW, g.PadH, g.PadW,
                        g.DilationH, g.DilationW, g.OutHeight, g.OutWidth, inputStorageOffset: x,
                        outputStorageOffset: y, kernelStorageOffset: w)) continue;
                    panel ??= CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(k * columns), "workspace.grouped_conv_forward_columns");
                    for (int from = 0; from < spatial; from += columns)
                    {
                        int count = Math.Min(columns, spatial - from);
                        FillGroupedColumnPanel(source, x, panel, from, count, g);
                        BlasProvider.GemmEx(g.OutChannelsPerGroup, count, k, weights, w, k, false,
                            panel, 0, count, false, output, y + from, spatial);
                    }
                }
        }
        finally { if (panel is not null) ArrayPool<float>.Shared.Return(panel); }
    }

    private static void GroupedInputGradientFloat(Tensor<float> destination, Tensor<float> upstream,
        Tensor<float> kernel, in GroupedConv2DGeometry g, bool accumulate)
    {
        var target = destination.AsWritableSpanUnmarked();
        if (!accumulate) target.Clear();
        if (g.Batch == 0) return;
        var weights = kernel.GetCpuBackingForStridedRead(out int kernelOffset);
        var gradient = upstream.GetCpuBackingForStridedRead(out int gradientOffset);
        if (weights is null || gradient is null)
        {
            GroupedInputGradientGeneric(destination, upstream, kernel, g, accumulate);
            return;
        }
        int k = checked(g.InChannelsPerGroup * g.KernelH * g.KernelW);
        int spatial = checked(g.OutHeight * g.OutWidth), columns = Math.Min(256, spatial);
        bool waveform = g.InHeight == 1 && g.KernelH == 1 && g.OutHeight == 1 && g.StrideH == 1
            && g.StrideW == 1 && g.PadH == 0 && g.KernelW > 1
            && (long)g.OutChannelsPerGroup * g.KernelW * g.InWidth >= 1_048_576;
        float[]? panel = null;
        try
        {
            for (int group = 0; group < g.Groups; group++)
                for (int image = 0; image < g.Batch; image++)
                {
                    if (waveform)
                    {
                        using var weightView = GroupedKernelView(kernel, group, g);
                        using var upstreamView = GroupedImageView(upstream, image, group * g.OutChannelsPerGroup, g.OutChannelsPerGroup);
                        using var destinationView = GroupedImageView(destination, image, group * g.InChannelsPerGroup, g.InChannelsPerGroup);
                        if (TryConv1DBackwardInputColumnPanels(destinationView, upstreamView, weightView, 1,
                            g.InChannelsPerGroup, g.InHeight, g.InWidth, g.OutChannelsPerGroup, g.KernelH, g.KernelW,
                            g.StrideH, g.StrideW, g.PadH, g.PadW, g.DilationH, g.DilationW, g.OutHeight, g.OutWidth, accumulate)) continue;
                    }
                    panel ??= CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(k * columns), "workspace.grouped_conv_dx_columns");
                    int w = checked(kernelOffset + group * g.OutChannelsPerGroup * k);
                    int dy = checked(gradientOffset + (image * g.OutChannels + group * g.OutChannelsPerGroup) * spatial);
                    int x = checked((image * g.InChannels + group * g.InChannelsPerGroup) * g.InHeight * g.InWidth);
                    for (int from = 0; from < spatial; from += columns)
                    {
                        int count = Math.Min(columns, spatial - from);
                        BlasProvider.GemmEx(k, count, g.OutChannelsPerGroup, weights, w, k, true,
                            gradient, dy + from, spatial, false, panel, 0, count);
                        AddGroupedColumnGradient(target, x, panel, from, count, g);
                    }
                }
        }
        finally { if (panel is not null) ArrayPool<float>.Shared.Return(panel); }
    }

    private static void AddGroupedColumnGradient(Span<float> destination, int offset, float[] panel,
        int from, int count, in GroupedConv2DGeometry g)
    {
        int row = 0;
        for (int channel = 0; channel < g.InChannelsPerGroup; channel++)
            for (int kh = 0; kh < g.KernelH; kh++)
                for (int kw = 0; kw < g.KernelW; kw++, row++)
                    for (int column = 0; column < count; column++)
                    {
                        int position = from + column;
                        int ih = position / g.OutWidth * g.StrideH - g.PadH + kh * g.DilationH;
                        int iw = position % g.OutWidth * g.StrideW - g.PadW + kw * g.DilationW;
                        if ((uint)ih < (uint)g.InHeight && (uint)iw < (uint)g.InWidth)
                            destination[offset + (channel * g.InHeight + ih) * g.InWidth + iw] += panel[row * count + column];
                    }
    }

    [ScalarKernel("Only exceptional strided destinations need logical tensor writes; contiguous grouped outputs are written in place.")]
    private static void CommitGroupedResult<T>(Tensor<T> destination, Tensor<T> source)
    {
        var values = source.AsSpanUnmarked();
        for (int index = 0; index < values.Length; index++) destination[index] = values[index];
    }

    [ScalarKernel("Only the strided accumulating-target fallback reads logical tensor elements; ordinary destinations use zero-copy spans.")]
    private static void CopyGroupedLogical<T>(Tensor<T> source, Tensor<T> destination)
    {
        var values = destination.AsWritableSpanUnmarked();
        for (int index = 0; index < values.Length; index++) values[index] = source[index];
    }

    // The weight VJP streams spatial panels from the original NCHW arrays, avoiding GetFlattenedData and
    // offset-unsafe output leases. One panel and one compact partial are reused across ALL images/groups.
    private static void GroupedKernelGradientFloat(Tensor<float> destination, Tensor<float> input,
        Tensor<float> upstream, in GroupedConv2DGeometry geometry, bool accumulate)
    {
        var target = destination.AsWritableSpanUnmarked();
        if (geometry.Batch == 0)
        {
            if (!accumulate) target.Clear();
            return;
        }
        int channels = geometry.InChannelsPerGroup, outputs = geometry.OutChannelsPerGroup;
        int k = checked(channels * geometry.KernelH * geometry.KernelW);
        int spatial = checked(geometry.OutHeight * geometry.OutWidth);
        int columns = Math.Min(256, spatial), weights = checked(outputs * k);
        var source = input.GetCpuBackingForStridedRead(out int inputOffset)
            ?? throw new InvalidOperationException("Contiguous grouped input needs managed backing.");
        var gradient = upstream.GetCpuBackingForStridedRead(out int gradientOffset)
            ?? throw new InvalidOperationException("Contiguous grouped gradient needs managed backing.");
        var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(checked(k * columns), "workspace.grouped_conv_columns");
        float[]? partial = null, sum = null;
        try
        {
            partial = CpuStorageAllocationDiagnostics.RentWorkspace<float>(weights, "workspace.grouped_conv_dw_partial");
            if (accumulate) sum = CpuStorageAllocationDiagnostics.RentWorkspace<float>(weights, "workspace.grouped_conv_dw_sum");
            for (int group = 0; group < geometry.Groups; group++)
            {
                var total = sum is null ? target.Slice(group * weights, weights) : sum.AsSpan(0, weights);
                total.Clear();
                for (int image = 0; image < geometry.Batch; image++)
                    for (int from = 0; from < spatial; from += columns)
                    {
                        int count = Math.Min(columns, spatial - from);
                        int x = checked(inputOffset + (image * geometry.InChannels + group * channels) * geometry.InHeight * geometry.InWidth);
                        int dy = checked(gradientOffset + (image * geometry.OutChannels + group * outputs) * spatial + from);
                        FillGroupedColumnPanel(source, x, panel, from, count, geometry);
                        BlasProvider.GemmEx(outputs, k, count, gradient, dy, spatial, false, panel, 0, count, true, partial, 0, k);
                        AddGroupedFloat(total, partial.AsSpan(0, weights));
                    }
                if (sum is not null) AddGroupedFloat(target.Slice(group * weights, weights), total);
            }
        }
        finally
        {
            if (sum is not null) ArrayPool<float>.Shared.Return(sum);
            if (partial is not null) ArrayPool<float>.Shared.Return(partial);
            ArrayPool<float>.Shared.Return(panel);
        }
    }

    private static void FillGroupedColumnPanel(float[] input, int offset, float[] panel, int from, int count,
        in GroupedConv2DGeometry g)
    {
        int row = 0;
        for (int channel = 0; channel < g.InChannelsPerGroup; channel++)
            for (int kh = 0; kh < g.KernelH; kh++)
                for (int kw = 0; kw < g.KernelW; kw++, row++)
                    for (int column = 0; column < count; column++)
                    {
                        int position = from + column;
                        int ih = position / g.OutWidth * g.StrideH - g.PadH + kh * g.DilationH;
                        int iw = position % g.OutWidth * g.StrideW - g.PadW + kw * g.DilationW;
                        panel[row * count + column] = (uint)ih < (uint)g.InHeight && (uint)iw < (uint)g.InWidth
                            ? input[offset + (channel * g.InHeight + ih) * g.InWidth + iw] : 0f;
                    }
    }

    private static void AddGroupedFloat(Span<float> destination, ReadOnlySpan<float> source)
    {
#if NET5_0_OR_GREATER
        int width = System.Numerics.Vector<float>.Count, index = 0;
        for (; index <= source.Length - width; index += width)
            (new System.Numerics.Vector<float>(destination.Slice(index, width))
                + new System.Numerics.Vector<float>(source.Slice(index, width))).CopyTo(destination.Slice(index, width));
#else
        int index = 0;
#endif
        for (; index < source.Length; index++) destination[index] += source[index];
    }

    [ScalarKernel("Non-FP32 grouped convolution uses generic numeric operations over zero-copy spans; FP32 uses bounded offset-aware panels.")]
    private static void GroupedForwardGeneric<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel, in GroupedConv2DGeometry g)
    {
        var x = input.AsSpanUnmarked(); var w = kernel.AsSpanUnmarked(); var y = output.AsWritableSpanUnmarked();
        var ops = MathHelper.GetNumericOperations<T>();
        int ci = g.InChannelsPerGroup, co = g.OutChannelsPerGroup;
        for (int n = 0; n < g.Batch; n++) for (int group = 0; group < g.Groups; group++)
            for (int oc = 0; oc < co; oc++) for (int oh = 0; oh < g.OutHeight; oh++) for (int ow = 0; ow < g.OutWidth; ow++)
            {
                T sum = ops.Zero;
                for (int ic = 0; ic < ci; ic++) for (int kh = 0; kh < g.KernelH; kh++) for (int kw = 0; kw < g.KernelW; kw++)
                {
                    int ih = oh * g.StrideH - g.PadH + kh * g.DilationH, iw = ow * g.StrideW - g.PadW + kw * g.DilationW;
                    if ((uint)ih >= (uint)g.InHeight || (uint)iw >= (uint)g.InWidth) continue;
                    int xi = ((n * g.InChannels + group * ci + ic) * g.InHeight + ih) * g.InWidth + iw;
                    int wi = (((group * co + oc) * ci + ic) * g.KernelH + kh) * g.KernelW + kw;
                    sum = ops.Add(sum, ops.Multiply(x[xi], w[wi]));
                }
                y[((n * g.OutChannels + group * co + oc) * g.OutHeight + oh) * g.OutWidth + ow] = sum;
            }
    }

    [ScalarKernel("Non-FP32 grouped input VJP uses generic numeric operations; FP32 reuses the offset-aware optimized input-gradient dispatch.")]
    private static void GroupedInputGradientGeneric<T>(Tensor<T> output, Tensor<T> upstream, Tensor<T> kernel,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        var dy = upstream.AsSpanUnmarked(); var w = kernel.AsSpanUnmarked(); var dx = output.AsWritableSpanUnmarked();
        var ops = MathHelper.GetNumericOperations<T>(); int ci = g.InChannelsPerGroup, co = g.OutChannelsPerGroup;
        for (int n = 0; n < g.Batch; n++) for (int group = 0; group < g.Groups; group++)
            for (int ic = 0; ic < ci; ic++) for (int ih = 0; ih < g.InHeight; ih++) for (int iw = 0; iw < g.InWidth; iw++)
            {
                T sum = ops.Zero;
                for (int oc = 0; oc < co; oc++) for (int kh = 0; kh < g.KernelH; kh++) for (int kw = 0; kw < g.KernelW; kw++)
                {
                    int h = ih + g.PadH - kh * g.DilationH, v = iw + g.PadW - kw * g.DilationW;
                    if (h < 0 || v < 0 || h % g.StrideH != 0 || v % g.StrideW != 0) continue;
                    int oh = h / g.StrideH, ow = v / g.StrideW;
                    if ((uint)oh >= (uint)g.OutHeight || (uint)ow >= (uint)g.OutWidth) continue;
                    int wi = (((group * co + oc) * ci + ic) * g.KernelH + kh) * g.KernelW + kw;
                    int di = ((n * g.OutChannels + group * co + oc) * g.OutHeight + oh) * g.OutWidth + ow;
                    sum = ops.Add(sum, ops.Multiply(dy[di], w[wi]));
                }
                int target = ((n * g.InChannels + group * ci + ic) * g.InHeight + ih) * g.InWidth + iw;
                dx[target] = accumulate ? ops.Add(dx[target], sum) : sum;
            }
    }

    [ScalarKernel("Non-FP32 grouped kernel VJP uses generic numeric operations over spans; FP32 uses bounded im2col panels and BLAS.")]
    private static void GroupedKernelGradientGeneric<T>(Tensor<T> output, Tensor<T> input, Tensor<T> upstream,
        in GroupedConv2DGeometry g, bool accumulate)
    {
        var x = input.AsSpanUnmarked(); var dy = upstream.AsSpanUnmarked(); var dw = output.AsWritableSpanUnmarked();
        var ops = MathHelper.GetNumericOperations<T>(); int ci = g.InChannelsPerGroup, co = g.OutChannelsPerGroup;
        for (int group = 0; group < g.Groups; group++) for (int oc = 0; oc < co; oc++)
            for (int ic = 0; ic < ci; ic++) for (int kh = 0; kh < g.KernelH; kh++) for (int kw = 0; kw < g.KernelW; kw++)
            {
                T sum = ops.Zero;
                for (int n = 0; n < g.Batch; n++) for (int oh = 0; oh < g.OutHeight; oh++) for (int ow = 0; ow < g.OutWidth; ow++)
                {
                    int ih = oh * g.StrideH - g.PadH + kh * g.DilationH, iw = ow * g.StrideW - g.PadW + kw * g.DilationW;
                    if ((uint)ih >= (uint)g.InHeight || (uint)iw >= (uint)g.InWidth) continue;
                    int xi = ((n * g.InChannels + group * ci + ic) * g.InHeight + ih) * g.InWidth + iw;
                    int di = ((n * g.OutChannels + group * co + oc) * g.OutHeight + oh) * g.OutWidth + ow;
                    sum = ops.Add(sum, ops.Multiply(x[xi], dy[di]));
                }
                int target = (((group * co + oc) * ci + ic) * g.KernelH + kh) * g.KernelW + kw;
                dw[target] = accumulate ? ops.Add(dw[target], sum) : sum;
            }
    }
}
