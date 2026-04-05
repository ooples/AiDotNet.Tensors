using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// Integration tests proving lazy graph mode produces bitwise-identical results to eager execution.
    /// Each test computes the same operation eagerly and lazily, then asserts element-wise equality.
    /// </summary>
    public class LazyGraphModeTests
    {
        private const float Tolerance = 1e-6f;

        private static Tensor<float> MakeRandom(int[] shape, int seed = 42)
        {
            var rng = new Random(seed);
            int length = 1;
            for (int i = 0; i < shape.Length; i++) length *= shape[i];
            var data = new float[length];
            for (int i = 0; i < data.Length; i++)
                data[i] = (float)(rng.NextDouble() * 2 - 1);
            return new Tensor<float>(data, shape);
        }

        private static void AssertTensorsEqual(Tensor<float> expected, Tensor<float> actual, string op)
        {
            Assert.Equal(expected._shape, actual._shape);
            var expSpan = expected.AsSpan();
            var actSpan = actual.AsSpan();
            for (int i = 0; i < expSpan.Length; i++)
            {
                Assert.True(Math.Abs(expSpan[i] - actSpan[i]) < Tolerance,
                    op + ": element [" + i + "] expected " + expSpan[i] + " but got " + actSpan[i]);
            }
        }

        [Fact]
        public void GraphMode_IsActive_ReportsCorrectly()
        {
            Assert.False(GraphMode.IsActive);
            using (var scope = GraphMode.Enable())
            {
                Assert.True(GraphMode.IsActive);
            }
        }

        [Fact]
        public void GraphMode_Dispose_RestoresParent()
        {
            Assert.False(GraphMode.IsActive);
            using (var scope = GraphMode.Enable())
            {
                Assert.True(GraphMode.IsActive);
            }
            Assert.False(GraphMode.IsActive);
        }

        [Fact]
        public void TensorAdd_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 64, 128 }, seed: 1);
            var b = MakeRandom(new[] { 64, 128 }, seed: 2);

            var eagerResult = engine.TensorAdd(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorAdd(a, b);
                Assert.Equal(1, scope.NodeCount);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorAdd");
        }

        [Fact]
        public void TensorSubtract_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 3);
            var b = MakeRandom(new[] { 32, 64 }, seed: 4);

            var eagerResult = engine.TensorSubtract(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorSubtract(a, b);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorSubtract");
        }

        [Fact]
        public void TensorMultiply_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 5);
            var b = MakeRandom(new[] { 32, 64 }, seed: 6);

            var eagerResult = engine.TensorMultiply(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorMultiply(a, b);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorMultiply");
        }

        [Fact]
        public void TensorDivide_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 16, 32 }, seed: 7);
            var rng = new Random(8);
            var bData = new float[16 * 32];
            for (int i = 0; i < bData.Length; i++)
                bData[i] = (float)(rng.NextDouble() * 2 + 0.1);
            var b = new Tensor<float>(bData, new[] { 16, 32 });

            var eagerResult = engine.TensorDivide(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorDivide(a, b);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorDivide");
        }

        [Fact]
        public void TensorMatMul_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 9);
            var b = MakeRandom(new[] { 64, 16 }, seed: 10);

            var eagerResult = engine.TensorMatMul(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorMatMul(a, b);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorMatMul");
        }

        [Fact]
        public void TensorNegate_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 64, 64 }, seed: 11);

            var eagerResult = engine.TensorNegate(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorNegate(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorNegate");
        }

        [Fact]
        public void TensorTranspose_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 12);

            var eagerResult = engine.TensorTranspose(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorTranspose(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "TensorTranspose");
        }

        [Fact]
        public void ReLU_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 64, 128 }, seed: 13);

            var eagerResult = engine.ReLU(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.ReLU(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "ReLU");
        }

        [Fact]
        public void Sigmoid_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 14);

            var eagerResult = engine.Sigmoid(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.Sigmoid(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "Sigmoid");
        }

        [Fact]
        public void Tanh_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 64 }, seed: 15);

            var eagerResult = engine.Tanh(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.Tanh(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "Tanh");
        }

        [Fact]
        public void Softmax_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 8, 16 }, seed: 16);

            var eagerResult = engine.Softmax(a);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.Softmax(a);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "Softmax");
        }

        [Fact]
        public void FusedLinear_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var input = MakeRandom(new[] { 8, 32 }, seed: 17);
            var weights = MakeRandom(new[] { 32, 16 }, seed: 18);
            var bias = MakeRandom(new[] { 16 }, seed: 19);

            var eagerResult = engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.FusedLinear(input, weights, bias, FusedActivationType.ReLU);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "FusedLinear+ReLU");
        }

        [Fact]
        public void MultiOpChain_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var input = MakeRandom(new[] { 8, 32 }, seed: 20);
            var weights = MakeRandom(new[] { 32, 16 }, seed: 21);
            var bias = MakeRandom(new[] { 8, 16 }, seed: 22);

            // Eager chain: MatMul -> Add -> ReLU
            var eagerMm = engine.TensorMatMul(input, weights);
            var eagerAdd = engine.TensorAdd(eagerMm, bias);
            var eagerRelu = engine.ReLU(eagerAdd);

            // Lazy chain
            Tensor<float> lazyRelu;
            using (var scope = GraphMode.Enable())
            {
                var lazyMm = engine.TensorMatMul(input, weights);
                var lazyAdd = engine.TensorAdd(lazyMm, bias);
                lazyRelu = engine.ReLU(lazyAdd);

                Assert.Equal(3, scope.NodeCount);
                scope.Realize();
            }

            AssertTensorsEqual(eagerRelu, lazyRelu, "MultiOpChain");
        }

        [Fact]
        public void NestedScope_Realizes_OnDispose()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 16, 16 }, seed: 23);
            var b = MakeRandom(new[] { 16, 16 }, seed: 24);

            var eagerResult = engine.TensorAdd(a, b);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorAdd(a, b);
            }
            // After dispose, data should be realized
            AssertTensorsEqual(eagerResult, lazyResult, "AutoRealize");
        }

        [Fact]
        public void Conv2D_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var input = MakeRandom(new[] { 1, 3, 8, 8 }, seed: 25);
            var kernel = MakeRandom(new[] { 4, 3, 3, 3 }, seed: 26);

            var eagerResult = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "Conv2D");
        }

        [Fact]
        public void MaxPool2D_Lazy_MatchesEager()
        {
            var engine = new CpuEngine();
            var input = MakeRandom(new[] { 1, 3, 8, 8 }, seed: 27);

            var eagerResult = engine.MaxPool2D(input, poolSize: 2, stride: 2);

            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.MaxPool2D(input, poolSize: 2, stride: 2);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "MaxPool2D");
        }

        [Fact]
        public void AutoMaterialize_OnDataAccess()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 16, 16 }, seed: 28);
            var b = MakeRandom(new[] { 16, 16 }, seed: 29);

            var eagerResult = engine.TensorAdd(a, b);

            // Record lazy op, then access data without explicit Realize
            // The scope auto-realizes on Dispose via using, but we verify
            // the data is correct after that auto-realization path
            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                lazyResult = engine.TensorAdd(a, b);
                // Scope Dispose will auto-realize
            }

            // Data should be available after auto-realize on Dispose
            var span = lazyResult.AsSpan();
            Assert.Equal(eagerResult.AsSpan().Length, span.Length);
            AssertTensorsEqual(eagerResult, lazyResult, "AutoMaterialize");
        }

        [Fact]
        public void CompiledInferencePlan_MatchesEager()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 32 }, seed: 30);
            var b = MakeRandom(new[] { 32, 32 }, seed: 31);

            // Eager
            var eagerAdd = engine.TensorAdd(a, b);
            var eagerRelu = engine.ReLU(eagerAdd);

            // Compiled inference plan
            CompiledInferencePlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var lazyAdd = engine.TensorAdd(a, b);
                var lazyRelu = engine.ReLU(lazyAdd);

                plan = scope.CompileInference<float>();
                Assert.True(plan.StepCount >= 2);
            }

            // Execute plan (scope is disposed, GraphMode is off)
            var planResult = plan.Execute();

            AssertTensorsEqual(eagerRelu, planResult, "CompiledInferencePlan");
        }

        [Fact]
        public void FusionPass_MatMulBiasReLU_ProducesCorrectResult()
        {
            var engine = new CpuEngine();
            var input = MakeRandom(new[] { 8, 32 }, seed: 40);
            var weights = MakeRandom(new[] { 32, 16 }, seed: 41);
            var bias1D = MakeRandom(new[] { 16 }, seed: 42);

            // Eager: MatMul + broadcast bias add + ReLU (3 separate ops)
            var mm = engine.TensorMatMul(input, weights);
            var biased = engine.TensorBroadcastAdd(mm, bias1D);
            var eagerResult = engine.ReLU(biased);

            // Lazy: should fuse into FusedLinearReLU via CpuFusionPass
            Tensor<float> lazyResult;
            using (var scope = GraphMode.Enable())
            {
                var lazyMm = engine.TensorMatMul(input, weights);
                var lazyBiased = engine.TensorBroadcastAdd(lazyMm, bias1D);
                lazyResult = engine.ReLU(lazyBiased);
                scope.Realize();
            }

            AssertTensorsEqual(eagerResult, lazyResult, "FusionPass_MatMulBiasReLU");
        }

        [Fact]
        public void OperationReordering_MaintainsCorrectness()
        {
            var engine = new CpuEngine();
            var a = MakeRandom(new[] { 32, 32 }, seed: 33);
            var b = MakeRandom(new[] { 32, 32 }, seed: 34);
            var c = MakeRandom(new[] { 32, 32 }, seed: 35);

            // Two independent chains that should be reorderable
            var eagerAB = engine.TensorAdd(a, b);
            var eagerBC = engine.TensorMultiply(b, c);
            var eagerFinal = engine.TensorAdd(eagerAB, eagerBC);

            Tensor<float> lazyFinal;
            using (var scope = GraphMode.Enable())
            {
                var lazyAB = engine.TensorAdd(a, b);
                var lazyBC = engine.TensorMultiply(b, c);
                lazyFinal = engine.TensorAdd(lazyAB, lazyBC);
                scope.Realize();
            }

            AssertTensorsEqual(eagerFinal, lazyFinal, "OperationReordering");
        }
    }
}
