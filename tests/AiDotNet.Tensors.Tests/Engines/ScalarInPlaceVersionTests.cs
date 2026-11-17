using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class ScalarInPlaceVersionTests
{
    [Fact] public void FloatWritesInvalidateTheTensorAndItsViews() => Check<float>();
    [Fact] public void DoubleWritesInvalidateTheTensorAndItsViews() => Check<double>();
    [Fact] public void GenericWritesInvalidateTheTensorAndItsViews() => Check<int>();

    private static void Check<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = new CpuEngine();
        var tensor = new Tensor<T>(new[] { 2, 3 }, new Vector<T>(
            Enumerable.Range(1, 6).Select(i => ops.FromDouble(i))));
        var view = engine.Reshape(tensor, new[] { 6 });
        int before = tensor.Version, viewBefore = view.Version;
        engine.TensorMultiplyScalarInPlace(view, ops.FromDouble(2));
        Assert.True(tensor.Version > before, "The source storage still appears unchanged after an in-place view write.");
        Assert.True(view.Version > viewBefore);
        for (int i = 0; i < tensor.Length; i++) Assert.Equal(2 * (i + 1), ops.ToDouble(tensor.GetFlat(i)));
    }

    [Fact]
    public void RecordingTheWriteDoesNotAdvanceTheVersionBeforeExecution()
    {
        var engine = new CpuEngine();
        var tensor = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 1f, 3f }));
        int before = tensor.Version;
        using (GraphMode.Enable())
        {
            engine.TensorMultiplyScalarInPlace(tensor, 2f);
            Assert.Equal(before, tensor.Version);
        }
        Assert.True(tensor.Version > before);
        Assert.Equal(new[] { 2f, 6f }, tensor.ToArray());
    }
}
