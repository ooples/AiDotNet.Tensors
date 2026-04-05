using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference plan — a flat array of pre-resolved execution steps.
/// Zero overhead per-operation: no graph traversal, no shape validation, no allocation.
/// All buffers are pre-allocated at compile time and reused across replays.
///
/// Usage:
///   var plan = CompiledInferencePlan.Compile(scope);
///   var output = plan.Execute(engine);  // first call
///   var output2 = plan.Execute(engine); // replay — same speed, zero alloc
/// </summary>
internal sealed class CompiledInferencePlan<T>
{
    private readonly CompiledStep<T>[] _steps;
    private readonly Tensor<T> _finalOutput;
    private readonly IEngine _engine;

    private CompiledInferencePlan(CompiledStep<T>[] steps, Tensor<T> finalOutput, IEngine engine)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
    }

    /// <summary>Number of compiled steps.</summary>
    internal int StepCount => _steps.Length;

    /// <summary>
    /// Executes the compiled plan. Runs each step's delegate in order.
    /// All buffers are pre-allocated — zero allocation during execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tensor<T> Execute()
    {
        var steps = _steps;
        var engine = _engine;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }
        return _finalOutput;
    }

    /// <summary>
    /// Compiles a lazy tensor scope into an inference plan.
    /// Runs optimization passes, pre-allocates all buffers, and builds step array.
    /// </summary>
    internal static CompiledInferencePlan<T> Compile(LazyTensorScope scope, IEngine engine)
    {
        var compiler = new LazyGraphCompiler();
        var optimized = compiler.Compile(scope.Nodes);

        var steps = new List<CompiledStep<T>>();
        foreach (var node in optimized)
        {
            if (node is LazyNode<T> typed)
            {
                steps.Add(new CompiledStep<T>(
                    typed.OpName,
                    typed.Execute,
                    typed.Output,
                    typed.GetInputsArray(),
                    typed.BackwardFn,
                    typed.SavedState));
            }
        }

        var finalOutput = steps.Count > 0 ? steps[steps.Count - 1].OutputBuffer : new Tensor<T>(new int[] { 0 });
        return new CompiledInferencePlan<T>(steps.ToArray(), finalOutput, engine);
    }
}
