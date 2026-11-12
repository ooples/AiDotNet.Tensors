using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Base class for custom autograd functions with user-defined forward and backward passes.
/// Equivalent to PyTorch's torch.autograd.Function.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// Subclass this to define operations with custom gradient computation:
/// <code>
/// public class MySquare : AutogradFunction&lt;float&gt;
/// {
///     public override Tensor&lt;float&gt; Forward(AutogradContext ctx, params Tensor&lt;float&gt;[] inputs)
///     {
///         ctx.SaveForBackward(inputs[0]);
///         var engine = AiDotNetEngine.Current;
///         return engine.TensorMultiply(inputs[0], inputs[0]);
///     }
///
///     public override void Backward(AutogradContext ctx, Tensor&lt;float&gt; gradOutput,
///         IEngine engine, Dictionary&lt;Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt; grads)
///     {
///         var x = ctx.GetSaved&lt;float&gt;(0);
///         var two = engine.TensorMultiplyScalar(x, 2f);
///         var grad = engine.TensorMultiply(gradOutput, two);
///         DifferentiableOps.AccumulateGrad(grads, x, grad, engine);
///     }
/// }
/// </code>
/// </remarks>
public abstract class AutogradFunction<T>
{
    /// <summary>
    /// Performs the forward computation. Save tensors needed for backward using ctx.
    /// </summary>
    public abstract Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs);

    /// <summary>
    /// Computes the gradient of each input given the gradient of the output, as torch.autograd.Function.backward
    /// does: element <c>i</c> is the gradient of <c>inputs[i]</c>, or null when that input gets none. Override this
    /// overload or the accumulating one, not both.
    /// </summary>
    public virtual Tensor<T>?[] Backward(AutogradContext ctx, Tensor<T> gradOutput) =>
        throw new NotImplementedException(
            $"{GetType().Name} must override Backward(AutogradContext, Tensor<T>) or the accumulating Backward overload.");

    /// <summary>
    /// Computes gradients w.r.t. inputs given the gradient of the output and adds them into <paramref name="grads"/>.
    /// The default implementation calls <see cref="Backward(AutogradContext, Tensor{T})"/> and accumulates the
    /// gradient it returns for each input.
    /// </summary>
    public virtual void Backward(AutogradContext ctx, Tensor<T> gradOutput,
        IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var inputs = ctx.Inputs as Tensor<T>[] ?? throw new InvalidOperationException("The autograd context has no recorded inputs.");
        var perInput = Backward(ctx, gradOutput);
        if (perInput.Length != inputs.Length)
            throw new InvalidOperationException(
                $"{GetType().Name}.Backward returned {perInput.Length} gradient(s) for {inputs.Length} input(s).");
        for (int i = 0; i < inputs.Length; i++)
            if (perInput[i] is { } g) DifferentiableOps.AccumulateGrad(grads, inputs[i], g, engine);
    }

    /// <summary>
    /// Applies this custom function: runs forward and registers backward on the active tape.
    /// </summary>
    public Tensor<T> Apply(params Tensor<T>[] inputs)
    {
        var ctx = new AutogradContext { Inputs = (Tensor<T>[])inputs.Clone() };

        // Suppress tape recording during Forward to prevent double-recording:
        // The custom function's forward ops would otherwise record to the tape,
        // AND we record the custom function itself — resulting in double gradients.
        Tensor<T> output;
        using (GradientTape<T>.NoGrad())
        {
            output = Forward(ctx, inputs);
        }

        // Record the custom function as a single opaque op on the tape
        DifferentiableOps.RecordIfActive(GetType().Name, output, (Tensor<T>[])inputs.Clone(),
            (gradOutput, inp, outp, savedState, engine, grads) =>
            {
                var savedCtx = (AutogradContext)savedState[0];
                Backward(savedCtx, gradOutput, engine, grads);
            },
            savedState: new object[] { ctx });

        return output;
    }
}

/// <summary>
/// Context object for saving tensors between forward and backward passes.
/// </summary>
public sealed class AutogradContext : ISavedStateTensorContainer
{
    private readonly List<object> _saved = [];

    /// <summary>Saves a tensor for use in the backward pass.</summary>
    /// <summary>The inputs <see cref="AutogradFunction{T}.Apply"/> was called with (typed as the function's T).</summary>
    internal System.Array? Inputs { get; set; }

    public void SaveForBackward<TItem>(Tensor<TItem> tensor) => _saved.Add(tensor);

    /// <summary>Saves multiple tensors.</summary>
    public void SaveForBackward<TItem>(params Tensor<TItem>[] tensors)
    {
        foreach (var t in tensors) _saved.Add(t);
    }

    /// <summary>Saves an arbitrary value.</summary>
    public void Save(object value) => _saved.Add(value);

    /// <summary>Retrieves a saved tensor by index.</summary>
    public Tensor<TItem> GetSaved<TItem>(int index) => (Tensor<TItem>)_saved[index];

    /// <summary>Retrieves a saved value by index.</summary>
    public TValue Get<TValue>(int index) => (TValue)_saved[index];

    /// <summary>Gets the number of saved items.</summary>
    public int SavedCount => _saved.Count;

    IReadOnlyList<object> ISavedStateTensorContainer.SavedStateValues => _saved;
}
