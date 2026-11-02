using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>Elements per parallel chunk of the fused add + ReLU (a multiple of every vector width).</summary>
    private const int AddReluChunkAlignment = 16;

    /// <summary><c>activation(a + b)</c>, the residual join of a ResNet block; see
    /// <see cref="TensorAddActivationEngineExtensions.TensorAddActivation{T}"/>.</summary>
    /// <remarks>
    /// A float ReLU over two contiguous tensors of one shape and layout runs in one pass: <c>max(a + b, +0)</c>, the
    /// same arithmetic as <see cref="TensorAdd{T}"/> followed by <see cref="ReLU{T}(Tensor{T})"/> (a NaN sum stays
    /// NaN, a -0 sum becomes +0, as on every GPU backend), without writing and re-reading the sum. The tape keeps only the output, which the ReLU mask is
    /// rebuilt from in the backward, so training holds one activation per residual join instead of two. Any other
    /// type, activation, shape pairing or a lazy graph composes the two recorded ops.
    /// </remarks>
    public virtual Tensor<T> TensorAddActivation<T>(Tensor<T> a, Tensor<T> b, FusedActivationType activation)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (activation == FusedActivationType.None) return TensorAdd(a, b);
        if (activation == FusedActivationType.ReLU && a is Tensor<float> af && b is Tensor<float> bf
            && !GraphMode.IsActive && ShapesMatch(a._shape, b._shape) && a.IsContiguous && b.IsContiguous
            && a.Layout == b.Layout)
        {
            var output = TensorAllocator.RentUninitialized<T>(a._shape);
            output.Layout = a.Layout;
            if (output is Tensor<float> result)
            {
                AddReluFloat(af, bf, result);
                DifferentiableOps.RecordBinary("TensorAddRelu", output, a, b, BackwardFunctions<T>.AddReluBackward);
                return output;
            }
        }
        return ApplyActivationRecorded(TensorAdd(a, b), activation);
    }

    private static unsafe void AddReluFloat(Tensor<float> a, Tensor<float> b, Tensor<float> result)
    {
        int length = a.Length;
        if (length == 0) return;
        using var aKeep = new KeepAliveScope(a);
        using var bKeep = new KeepAliveScope(b);
        using var rKeep = new KeepAliveScope(result);
        // The logical memory, pinned (as TensorAdd does): a contiguous tensor need not have a managed backing array (a
        // DLPack import lives in native memory).
        using var pinA = AsFloatReadOnlyMemory(a.ReadOnlyDataUnmarked).Pin();
        using var pinB = AsFloatReadOnlyMemory(b.ReadOnlyDataUnmarked).Pin();
        using var pinR = AsFloatMemory(result.DataUnmarked).Pin();
        IntPtr ipA = (IntPtr)pinA.Pointer, ipB = (IntPtr)pinB.Pointer, ipR = (IntPtr)pinR.Pointer;
        int chunks = Math.Max(1, CpuParallelSettings.ElementwiseChunkCount(length));
        int chunkSize = ((length + chunks - 1) / chunks + AddReluChunkAlignment - 1) & ~(AddReluChunkAlignment - 1);
        CpuParallelSettings.ParallelForOrSerial(0, chunks, length, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int chunk) =>
        {
            int start = chunk * chunkSize;
            int count = Math.Min(chunkSize, length - start);
            if (count <= 0) return;
            AddReluSpan((float*)ipA + start, (float*)ipB + start, (float*)ipR + start, count);

        });
    }

    /// <summary><c>r[i] = relu(a[i] + b[i])</c> with the ReLU kernel's NaN contract: a NaN sum stays NaN (it read +0
    /// here while every GPU backend propagated it), a -0 sum becomes +0.</summary>
    [MethodImpl(Compatibility.MethodImplHelper.Hot)]
    private static unsafe void AddReluSpan(float* a, float* b, float* r, int count)
    {
        int i = 0;
#if NET5_0_OR_GREATER
        if (Avx.IsSupported)
        {
            for (; i + 32 <= count; i += 32)
            {
                Avx.Store(r + i, Simd.SimdKernels.ReluNaN256(Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i))));
                Avx.Store(r + i + 8, Simd.SimdKernels.ReluNaN256(Avx.Add(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8))));
                Avx.Store(r + i + 16, Simd.SimdKernels.ReluNaN256(Avx.Add(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16))));
                Avx.Store(r + i + 24, Simd.SimdKernels.ReluNaN256(Avx.Add(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24))));
            }
            for (; i + 8 <= count; i += 8)
                Avx.Store(r + i, Simd.SimdKernels.ReluNaN256(Avx.Add(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i))));
        }
#endif
        for (; i < count; i++)
        {
            r[i] = Simd.SimdKernels.ReluNaN(a[i] + b[i]);
        }
    }
}
