// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// <see cref="CpuEngine.Conv2D{T}(Tensor{T}, Tensor{T}, int, int, int)"/> and
    /// <see cref="CpuEngine.Conv2DInto{T}(Tensor{T}, Tensor{T}, Tensor{T}, int, int, int)"/> must
    /// compute the same function, bit for bit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// They dispatched differently for float: the allocating overload used the SIMD-direct /
    /// Winograd cascade, the in-place one used im2col-GEMM. The difference was assumed to be
    /// confined to summation order, and at float32 it is not -- on a 3x3 stride-1 conv over
    /// [1,3,16,16], 1442 of 2048 outputs differed, and on a 16-to-32 channel 14x14 conv 5465 of
    /// 6272 did.
    /// </para>
    /// <para>
    /// A caller cannot see which entry point it reached, and the two are not interchangeable in
    /// practice: the in-place variant bypasses the gradient tape, so a layer that wants gradients
    /// must call the allocating one. AiDotNet's ConvolutionalLayer therefore picked between them by
    /// whether a tape was recording, which made the same layer with the same weights compute a
    /// different function in training than at inference. That is this dispatch's problem, not the
    /// layer's: PyTorch chooses a convolution algorithm from shape and hardware, never from grad
    /// mode, and <c>no_grad</c> governs whether a graph is built rather than the values in it.
    /// </para>
    /// <para>
    /// Bit-for-bit is the right assertion rather than a tolerance. A tolerance would pass while the
    /// two kernels drifted apart, which is exactly the state this test exists to prevent, and the
    /// two are supposed to be the same computation reaching two different destinations.
    /// </para>
    /// </remarks>
    public class Conv2DEntryPointParityTests
    {
        public static IEnumerable<object[]> Shapes()
        {
            // kernel, stride, padding, inChannels, outChannels, spatial
            yield return new object[] { 3, 1, 1, 3, 8, 16 };     // the reported case
            yield return new object[] { 3, 1, 1, 16, 32, 14 };   // the shape the GEMM route was tuned on
            yield return new object[] { 3, 1, 1, 64, 64, 32 };
            yield return new object[] { 3, 1, 1, 256, 128, 8 };  // high channel count, separate routing
            yield return new object[] { 3, 2, 1, 3, 8, 16 };
            yield return new object[] { 1, 1, 0, 3, 8, 16 };
            yield return new object[] { 4, 1, 1, 3, 8, 16 };
            yield return new object[] { 4, 2, 1, 16, 32, 14 };
            yield return new object[] { 5, 1, 2, 3, 8, 16 };
            yield return new object[] { 7, 1, 3, 3, 8, 16 };
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void Conv2DAndConv2DInto_ProduceIdenticalFloatResults(
            int kernelSize, int stride, int padding, int inChannels, int outChannels, int spatial)
        {
            var engine = new CpuEngine();
            var rng = new Random(11);

            var input = new Tensor<float>(new[] { 1, inChannels, spatial, spatial });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var kernel = new Tensor<float>(new[] { outChannels, inChannels, kernelSize, kernelSize });
            for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() * 2 - 1);

            var allocating = engine.Conv2D(input, kernel, stride, padding, 1);

            var inPlace = new Tensor<float>(allocating.Shape.ToArray());
            engine.Conv2DInto(inPlace, input, kernel, stride, padding, 1);

            int differing = 0;
            int firstIndex = -1;
            for (int i = 0; i < allocating.Length; i++)
            {
                if (allocating[i] == inPlace[i]) continue;
                differing++;
                if (firstIndex < 0) firstIndex = i;
            }

            Assert.True(
                differing == 0,
                $"Conv2D and Conv2DInto disagree on {differing} of {allocating.Length} outputs for "
                    + $"k{kernelSize} s{stride} p{padding} {inChannels}->{outChannels} @{spatial}x{spatial}. "
                    + (firstIndex >= 0
                        ? $"First at [{firstIndex}]: allocating={allocating[firstIndex]:G9}, "
                          + $"inPlace={inPlace[firstIndex]:G9}. "
                        : string.Empty)
                    + "These entry points must select the same kernel: callers cannot observe which "
                    + "one they reached, and choosing between them by whether a gradient tape is "
                    + "recording would make a layer compute a different function in training than "
                    + "at inference.");
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void Conv2DAndConv2DInto_ProduceIdenticalDoubleResults(
            int kernelSize, int stride, int padding, int inChannels, int outChannels, int spatial)
        {
            var engine = new CpuEngine();
            var rng = new Random(11);

            var input = new Tensor<double>(new[] { 1, inChannels, spatial, spatial });
            for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

            var kernel = new Tensor<double>(new[] { outChannels, inChannels, kernelSize, kernelSize });
            for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() * 2 - 1;

            var allocating = engine.Conv2D(input, kernel, stride, padding, 1);

            var inPlace = new Tensor<double>(allocating.Shape.ToArray());
            engine.Conv2DInto(inPlace, input, kernel, stride, padding, 1);

            int differing = 0;
            for (int i = 0; i < allocating.Length; i++)
                if (allocating[i] != inPlace[i]) differing++;

            Assert.True(
                differing == 0,
                $"Conv2D and Conv2DInto disagree on {differing} of {allocating.Length} double outputs "
                    + $"for k{kernelSize} s{stride} p{padding} {inChannels}->{outChannels} "
                    + $"@{spatial}x{spatial}.");
        }
    }
}
