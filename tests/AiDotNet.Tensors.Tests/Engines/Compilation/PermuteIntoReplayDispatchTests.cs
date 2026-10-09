using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A recorded TensorPermuteInto replayed on a device engine permuted out of place and copied the result through host
/// spans, a download that aborts a CUDA graph capture. It must replay through the engine's own TensorPermuteInto, which a
/// device engine runs on the device.
/// </summary>
public class PermuteIntoReplayDispatchTests
{
    private sealed class DeviceLikeEngine : CpuEngine
    {
        public int ReplayedPermuteIntoCalls;

        public override bool SupportsGpu => true;

        public override void TensorPermuteInto<T>(Tensor<T> output, Tensor<T> tensor, int[] axes)
        {
            if (!GraphMode.IsActive) ReplayedPermuteIntoCalls++;
            base.TensorPermuteInto(output, tensor, axes);
        }
    }

    [Fact]
    public void ReplayedPermuteInto_DispatchesToTheEnginesOwnWriteInto()
    {
        var engine = new DeviceLikeEngine();
        var data = new float[2 * 3 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = i;
        var input = new Tensor<float>(data, new[] { 2, 3, 4 });
        var axes = new[] { 2, 0, 1 };

        var expected = new Tensor<float>(new[] { 4, 2, 3 });
        new CpuEngine().TensorPermuteInto(expected, input, axes);

        var lazy = new Tensor<float>(new[] { 4, 2, 3 });
        using (GraphMode.Enable())
        {
            engine.TensorPermuteInto(lazy, input, axes);
            Assert.NotNull(lazy.LazySource);
            Assert.Equal(0, engine.ReplayedPermuteIntoCalls);   // recorded, not run
        }

        Assert.Equal(expected.ToArray(), lazy.ToArray());
        Assert.Equal(1, engine.ReplayedPermuteIntoCalls);
    }
}
