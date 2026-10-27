using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for complex construction, nonzero_static, unravel_index, unique_dim and grid_sample 3-D (#1112 tier 8).
/// Argument errors stay with the CPU implementation (these return null so it raises them); per-element range errors in
/// unravel_index come back through the validation flag. unique_dim's output size depends on the data, so it reads back
/// one count, as any implementation must.
/// </summary>
public partial class DirectGpuTensorEngine
{
    // complex(real, imag): the two planes copied into one split-complex device buffer.
    private Tensor<Complex<T>>? TryComplexOnDevice<T>(Tensor<T> real, Tensor<T> imag)
    {
        if (real is null || imag is null || real.Length == 0 || !real._shape.AsSpan().SequenceEqual(imag._shape)
            || IsTapeActive<T>("TensorComplex") || !TryGetBackend(out var backend) || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            using var re = GetOrAllocateBuffer(backend, real.IsContiguous ? real : real.Contiguous());
            using var im = GetOrAllocateBuffer(backend, imag.IsContiguous ? imag : imag.Contiguous());
            return PackAndDeferSplitComplexResult<T>(backend, re.Buffer, im.Buffer, real.Length, (int[])real._shape.Clone());
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorComplex", ex);
            return null;
        }
    }

    // view_as_complex: the interleaved [..., 2] pairs deinterleaved on the device into a split-complex buffer.
    private Tensor<Complex<T>>? TryViewAsComplexOnDevice<T>(Tensor<T> input)
    {
        if (input is null || input.Rank == 0 || input._shape[input.Rank - 1] != 2 || input.Length == 0
            || IsTapeActive<T>("TensorViewAsComplex") || !TryGetBackend(out var backend) || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = input.Length / 2;
            using var pairs = GetOrAllocateBuffer(backend, input.IsContiguous ? input : input.Contiguous());
            using var re = AllocateOutputBuffer(backend, n);
            using var im = AllocateOutputBuffer(backend, n);
            backend.DeinterleaveComplex(pairs.Buffer, re.Buffer, im.Buffer, n);
            var shape = new int[input.Rank - 1];
            Array.Copy(input._shape, shape, shape.Length);
            return PackAndDeferSplitComplexResult<T>(backend, re.Buffer, im.Buffer, n, shape);
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorViewAsComplex", ex);
            return null;
        }
    }

    // nonzero_static: a device non-zero indicator, its inclusive scan (exact in float below 2^24 elements), and a
    // scatter of the first `size` coordinates into a fill-valued [size, rank] result.
    private Tensor<int>? TryNonzeroStaticOnDevice<T>(Tensor<T> tensor, int size, int fillValue)
    {
        if (tensor is null || tensor.Rank == 0 || tensor.Rank > 8 || size <= 0 || tensor.Length == 0 || tensor.Length >= (1 << 24)
            || Math.Abs((long)fillValue) >= (1 << 24) || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        try
        {
            int n = tensor.Length, rank = tensor.Rank;
            using var x = GetOrAllocateBuffer(backend, tensor.IsContiguous ? tensor : tensor.Contiguous());
            using var flags = AllocateOutputBuffer(backend, n);
            using var positions = AllocateOutputBuffer(backend, n);
            kernels.TorchParityUnary(TorchParityUnaryOp.NonZero, x.Buffer, flags.Buffer, n, 0f, 0f);
            kernels.TorchParityInclusiveScan(flags.Buffer, positions.Buffer, n);
            int count = checked(size * rank);
            var output = AllocateOutputBuffer(backend, count);
            bool ownsOutput = true;
            try
            {
                backend.Fill(output.Buffer, fillValue, count);
                kernels.TorchParityNonzeroStaticScatter(x.Buffer, positions.Buffer, output.Buffer, n, size, tensor._shape);
                var data = FinishGpuOp<int>(backend, output, count);
                ownsOutput = false;
                return new Tensor<int>(data, new[] { size, rank });
            }
            finally
            {
                if (ownsOutput) output.Dispose();
            }
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorNonzeroStatic", ex);
            return null;
        }
    }

    private Tensor<T>[]? TryUnravelIndexOnDevice<T>(Tensor<T> indices, int[] shape)
    {
        if (indices is null || shape is null || indices.Length == 0 || shape.Length == 0) return null;
        long total = 1;
        foreach (int d in shape) { if (d <= 0) return null; total *= d; }
        if (!TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        var result = new Tensor<T>[shape.Length];
        ulong flagValue;
        try
        {
            int n = indices.Length;
            using var x = GetOrAllocateBuffer(backend, indices.IsContiguous ? indices : indices.Contiguous());
            using var flag = AllocateOutputBuffer(backend, 2);
            kernels.TorchParityFlagReset(flag.Buffer);
            long stride = 1;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                long axisStride = stride, extent = shape[d];
                bool validate = d == shape.Length - 1;
                result[d] = DispatchDeferredGpuOp<T>(backend, n, (int[])indices._shape.Clone(),
                    output => kernels.TorchParityUnravelAxis(x.Buffer, output, n, total, axisStride, extent, validate, flag.Buffer));
                stride *= shape[d];
            }
            flagValue = ReadParityFlag(backend, flag.Buffer);
        }
        catch (NotSupportedException ex) {
            foreach (var t in result) t?.Dispose();
            GpuLaunchProbe.OnFallback("TensorUnravelIndex", ex);
            return null;
        }
        if (flagValue == NoParityError) return result;
        foreach (var t in result) t?.Dispose();
        long index = (long)((flagValue >> 3) & ((1UL << 58) - 1));
        double bad = Math.Truncate(MathHelper.GetNumericOperations<T>().ToDouble(indices[(int)index]));
        throw new ArgumentOutOfRangeException(nameof(indices), $"index {bad} is out of range for shape [{string.Join(", ", shape)}].");
    }

    // unique_dim: slices along dim moved to rows, bitonic-sorted by the CPU's slice order, first-of-run flags scanned to
    // output rows, the unique slices gathered in sorted order, and the axis moved back.
    private Tensor<T>? TryUniqueDimOnDevice<T>(Tensor<T> tensor, int dim)
    {
        if (tensor is null || tensor.Rank == 0 || tensor.Length == 0) return null;
        int d = dim < 0 ? dim + tensor.Rank : dim;
        if (d < 0 || d >= tensor.Rank || typeof(T) != typeof(float) || IsTapeActive<T>("TensorUniqueDim")
            || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels)
            return null;
        int extent = tensor._shape[d], outer = 1, inner = 1;
        for (int k = 0; k < d; k++) outer *= tensor._shape[k];
        for (int k = d + 1; k < tensor.Rank; k++) inner *= tensor._shape[k];
        int length = outer * inner;
        if (extent >= (1 << 24)) return null;
        try
        {
            using var x = GetOrAllocateBuffer(backend, tensor.IsContiguous ? tensor : tensor.Contiguous());
            using var slices = AllocateFullyWrittenOutputBuffer(backend, tensor.Length);
            if (outer == 1) backend.Copy(x.Buffer, slices.Buffer, tensor.Length);
            else backend.Permute(x.Buffer, slices.Buffer, new[] { outer, extent, inner }, new[] { 1, 0, 2 });
            int padded = 1;
            while (padded < extent) padded <<= 1;
            using var order = AllocateOutputBuffer(backend, padded);
            backend.IotaPad(order.Buffer, extent, padded, 1);
            for (int k = 2; k <= padded; k <<= 1)
                for (int j = k >> 1; j > 0; j >>= 1)
                    kernels.TorchParitySliceBitonicStep(slices.Buffer, order.Buffer, padded, k, j, length, extent);
            using var flags = AllocateOutputBuffer(backend, extent);
            using var positions = AllocateOutputBuffer(backend, extent);
            kernels.TorchParitySliceFirstFlags(slices.Buffer, order.Buffer, flags.Buffer, extent, length);
            kernels.TorchParityInclusiveScan(flags.Buffer, positions.Buffer, extent);
            int unique = (int)DownloadScalar(backend, positions.Buffer, extent - 1);
            using var rows = AllocateOutputBuffer(backend, unique * length);
            kernels.TorchParitySliceGatherUnique(slices.Buffer, order.Buffer, flags.Buffer, positions.Buffer, rows.Buffer, extent, length, unique);
            var shape = (int[])tensor._shape.Clone();
            shape[d] = unique;
            int count = unique * length;
            return DispatchDeferredGpuOp<T>(backend, count, shape, output =>
            {
                if (outer == 1) backend.Copy(rows.Buffer, output, count);
                else backend.Permute(rows.Buffer, output, new[] { unique, outer, inner }, new[] { 1, 0, 2 });
            });
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorUniqueDim", ex);
            return null;
        }
    }

    private Tensor<T>? TryGridSample3DOnDevice<T>(Tensor<T> input, Tensor<T> grid, GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (input is null || grid is null || mode == GridSampleMode.Bicubic || input.Rank != 5 || grid.Rank != 5 || grid._shape[4] != 3
            || grid._shape[0] != input._shape[0] || input.Length == 0 || grid.Length == 0
            || Compilation.GraphMode.IsActive || !TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels
            || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        Tensor<T> result;
        try
        {
            int n = input._shape[0], c = input._shape[1];
            int points = grid._shape[1] * grid._shape[2] * grid._shape[3];
            using var x = GetOrAllocateBuffer(backend, input.IsContiguous ? input : input.Contiguous());
            using var g = GetOrAllocateBuffer(backend, grid.IsContiguous ? grid : grid.Contiguous());
            int modeCode = mode == GridSampleMode.Nearest ? 1 : 0;
            int paddingCode = padding == GridSamplePadding.Border ? 1 : padding == GridSamplePadding.Reflection ? 2 : 0;
            var outShape = new[] { n, c, grid._shape[1], grid._shape[2], grid._shape[3] };
            result = DispatchDeferredGpuOp<T>(backend, n * c * points, outShape, output => kernels.TorchParityGridSample3D(
                x.Buffer, g.Buffer, output, n, c, input._shape[2], input._shape[3], input._shape[4], points, modeCode, paddingCode, alignCorners));
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorGridSample3D", ex);
            return null;
        }
        // The CPU's node and saved state, with a backward that computes both adjoints on the device.
        Autodiff.DifferentiableOps.RecordBinary("TensorGridSample3D", result, input, grid, GridSample3DDeviceBackward<T>,
            new object[] { mode, padding, alignCorners });
        return result;
    }

    private static void GridSample3DDeviceBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (engine is DirectGpuTensorEngine gpu
            && gpu.TryGridSample3DAdjointsOnDevice(gradOutput, inputs[0], inputs[1], (GridSampleMode)savedState[0],
                (GridSamplePadding)savedState[1], (bool)savedState[2]) is { } adjoints)
        {
            Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[0], adjoints.Input, engine);
            Autodiff.DifferentiableOps.AccumulateGrad(grads, inputs[1], adjoints.Grid, engine);
            return;
        }
        if (engine is DirectGpuTensorEngine) GpuLaunchProbe.OnFallback("TensorGridSample3D backward: host adjoint", null);
        GridSample3DBackward(gradOutput, inputs, output, savedState, engine, grads);
    }

    private (Tensor<T> Input, Tensor<T> Grid)? TryGridSample3DAdjointsOnDevice<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> grid,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (!TryGetBackend(out var backend) || backend is not ITorchParityElementwiseBackend kernels || DirectGpuEngine.ShouldFallbackForPrecision<T>())
            return null;
        OwnedBuffer dx = default, dg = default;
        bool ownsDx = false, ownsDg = false;
        try
        {
            int n = input._shape[0], c = input._shape[1];
            int points = grid._shape[1] * grid._shape[2] * grid._shape[3];
            using var x = GetOrAllocateBuffer(backend, input.IsContiguous ? input : input.Contiguous());
            using var g = GetOrAllocateBuffer(backend, grid.IsContiguous ? grid : grid.Contiguous());
            using var dy = GetOrAllocateBuffer(backend, gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous());
            dx = AllocateOutputBuffer(backend, input.Length);
            ownsDx = true;
            dg = AllocateOutputBuffer(backend, grid.Length);
            ownsDg = true;
            backend.Fill(dx.Buffer, 0f, input.Length);
            int modeCode = mode == GridSampleMode.Nearest ? 1 : 0;
            int paddingCode = padding == GridSamplePadding.Border ? 1 : padding == GridSamplePadding.Reflection ? 2 : 0;
            kernels.TorchParityGridSample3DBackward(x.Buffer, g.Buffer, dy.Buffer, dx.Buffer, dg.Buffer, n, c, input._shape[2],
                input._shape[3], input._shape[4], points, modeCode, paddingCode, alignCorners);
            var inputGrad = new Tensor<T>(FinishGpuOp<T>(backend, dx, input.Length), (int[])input._shape.Clone());
            ownsDx = false;
            var gridGrad = new Tensor<T>(FinishGpuOp<T>(backend, dg, grid.Length), (int[])grid._shape.Clone());
            ownsDg = false;
            return (inputGrad, gridGrad);
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback("TensorGridSample3D backward", ex);
            return null;
        }
        finally
        {
            if (ownsDx) dx.Dispose();
            if (ownsDg) dg.Dispose();
        }
    }

    // #1135: integer ops on long, ulong, short, ushort, sbyte, byte and uint tensors run on the raw int64 device layout
    // (two words per element, the element's int64 widening). A result stays resident in that layout, so a chain of
    // integer ops never leaves the device; the host sees it narrowed back to the element type.
    private Tensor<T>? TryInt64OnDevice<T>(string opName, Tensor<T> a, Tensor<T>? b, int op)
    {
        int typeCode = Int64TypeCode<T>();
        if (typeCode < 0 || a is null || a.Length == 0 || (b is not null && !a._shape.AsSpan().SequenceEqual(b._shape))
            || (op >= 0 && b is null) || IsTapeActive<T>(opName) || !TryGetBackend(out var backend)
            || backend is not ITorchParityElementwiseBackend kernels)
            return null;
        IGpuBuffer? output = null;
        ulong flagValue = NoParityError;
        try
        {
            int n = a.Length;
            if (n > int.MaxValue / 2) return null;
            using var x = GetOrUploadRawInt64(backend, a);
            output = backend.AllocateIntBuffer(2 * n);
            if (op < 0) kernels.TorchParityInt64Not(x.Buffer, output, n, typeCode);
            else
            {
                using var y = GetOrUploadRawInt64(backend, b!);
                using var flag = AllocateOutputBuffer(backend, 2);
                kernels.TorchParityFlagReset(flag.Buffer);
                kernels.TorchParityInt64Binary(op, x.Buffer, y.Buffer, output, n, typeCode, flag.Buffer);
                if (op >= 5) flagValue = ReadParityFlag(backend, flag.Buffer);   // gcd, lcm, floor divide can throw
            }
            if (flagValue != NoParityError)
            {
                if ((flagValue & 7) == 1) throw new DivideByZeroException($"{opName}: integer division by zero.");
                throw new OverflowException($"{opName}: the result overflows Int64 (as Math.Abs / division on the CPU).");
            }
            var result = Tensor<T>.FromGpuBuffer(backend, output, (int[])a._shape.Clone(), Gpu.GpuTensorRole.Intermediate,
                ownsBuffer: true, bufferContainsRawInt64: true);
            output = null;
            return result;
        }
        catch (Exception ex) when (ex is DivideByZeroException || ex is OverflowException)
        {
            throw;
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
        finally
        {
            output?.Dispose();
        }
    }

    private static int Int64TypeCode<T>()
    {
        if (typeof(T) == typeof(long)) return 0;
        if (typeof(T) == typeof(ulong)) return 1;
        if (typeof(T) == typeof(short)) return 2;
        if (typeof(T) == typeof(ushort)) return 3;
        if (typeof(T) == typeof(sbyte)) return 4;
        if (typeof(T) == typeof(byte)) return 5;
        if (typeof(T) == typeof(uint)) return 6;
        return -1;
    }

    // The tensor's raw int64 device buffer: its own when a previous integer op left it resident in that layout,
    // otherwise its host values widened and uploaded.
    private OwnedBuffer GetOrUploadRawInt64<T>(IDirectGpuBackend backend, Tensor<T> tensor)
    {
        if (tensor._gpuBufferContainsRawInt64 && tensor.IsContiguous && tensor._gpuBuffer is { } resident
            && ReferenceEquals(tensor._gpuBackend, backend) && resident.Size >= 2 * tensor.Length)
            return new OwnedBuffer(resident, ownsBuffer: false);
        var contiguous = tensor.IsContiguous ? tensor : tensor.Contiguous();
        var longs = Tensor<T>.ToRawInt64(contiguous.GetDataArray());
        var words = new int[2 * longs.Length];
        Buffer.BlockCopy(longs, 0, words, 0, longs.Length * sizeof(long));
        return new OwnedBuffer(backend.AllocateIntBuffer(words), ownsBuffer: true);
    }

    // Integer ops on int32 tensors run on raw int32 device storage (the same storage resident argsort indices use);
    // other integer widths keep the CPU path. op as in ITorchParityElementwiseBackend.TorchParityIntBinary; -1 is not.
    private Tensor<T>? TryInt32OnDevice<T>(string opName, Tensor<T> a, Tensor<T>? b, int op)
    {
        if (typeof(T) != typeof(int) || a is null || a.Length == 0 || (b is not null && !a._shape.AsSpan().SequenceEqual(b._shape))
            || (op >= 0 && b is null) || IsTapeActive<T>(opName) || !TryGetBackend(out var backend)
            || backend is not ITorchParityElementwiseBackend kernels)
            return null;
        IGpuBuffer? output = null;
        ulong flagValue = NoParityError;
        try
        {
            int n = a.Length;
            var ai = (Tensor<int>)(object)(a.IsContiguous ? a : a.Contiguous());
            using var x = GetOrAllocateInt32IndexBuffer(backend, ai);
            output = backend.AllocateIntBuffer(n);
            if (op < 0) kernels.TorchParityIntNot(x.Buffer, output, n);
            else
            {
                var bi = (Tensor<int>)(object)(b!.IsContiguous ? b : b.Contiguous());
                using var y = GetOrAllocateInt32IndexBuffer(backend, bi);
                using var flag = AllocateOutputBuffer(backend, 2);
                kernels.TorchParityFlagReset(flag.Buffer);
                kernels.TorchParityIntBinary(op, x.Buffer, y.Buffer, output, n, flag.Buffer);
                if (op == 7) flagValue = ReadParityFlag(backend, flag.Buffer);
            }
            if (flagValue != NoParityError)
                throw new DivideByZeroException("TensorFloorDivide: integer division by zero.");
            var result = Tensor<int>.FromGpuBuffer(backend, output, (int[])a._shape.Clone(), Gpu.GpuTensorRole.Intermediate,
                ownsBuffer: true, bufferContainsRawInt32: true);
            output = null;
            return (Tensor<T>)(object)result;
        }
        catch (DivideByZeroException)
        {
            throw;
        }
        catch (NotSupportedException ex) {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
        finally
        {
            output?.Dispose();
        }
    }
}
