using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Rehydrates previously-serialized compiled plans from a byte stream.
/// The static counterpart of <see cref="ICompiledPlan{T}.SaveAsync"/> and
/// <see cref="ICompiledTrainingPlan{T}.SaveAsync"/> — placed in a static class
/// because .NET Framework 4.7.1 does not support static interface members.
///
/// <para><b>Compatibility:</b> returns <c>null</c> (never throws) when the
/// stream's version stamp doesn't match the current runtime. Callers
/// interpret null as "recompile from scratch" — the zero-warmup path falls
/// back to one-time compilation on the first request.</para>
/// </summary>
public static class CompiledPlanLoader
{
    /// <summary>
    /// Rehydrates a previously-saved inference plan. Returns <c>null</c> if
    /// the stream's format version, tensor-codec version, hardware fingerprint,
    /// or element type doesn't match the current runtime — this forces
    /// recompile rather than silently mis-replaying an incompatible plan.
    /// </summary>
    /// <param name="stream">The stream containing the serialized plan.</param>
    /// <param name="engine">The engine to use for reconstructing ops.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The rehydrated plan, or null if incompatible.</returns>
    public static Task<ICompiledPlan<T>?> LoadInferenceAsync<T>(
        Stream stream,
        IEngine engine,
        CancellationToken cancellationToken = default)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (engine is null) throw new ArgumentNullException(nameof(engine));

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            var plan = InferencePlanReader.Read<T>(stream, engine);
            return Task.FromResult<ICompiledPlan<T>?>(plan);
        }
        catch (InvalidDataException ex)
        {
            // Corruption, truncation, unreadable format, or version mismatch —
            // treat as a cache miss so the caller recompiles. Log the cause
            // via Trace so operators can diagnose why warm-start isn't
            // working (silently returning null would hide genuine file
            // corruption behind a "just recompile" path).
            Trace.WriteLine($"[CompiledPlanLoader] Inference plan load failed, recompiling: {ex.Message}");
            return Task.FromResult<ICompiledPlan<T>?>(null);
        }
    }

    /// <summary>
    /// Rehydrates a previously-saved training plan. Returns <c>null</c> if
    /// incompatible. The caller must supply the same parameter tensors that
    /// were used during the original compilation — the loaded plan rebinds
    /// its forward/backward closures to these parameters.
    /// </summary>
    /// <param name="stream">The stream containing the serialized plan.</param>
    /// <param name="engine">The engine to use for reconstructing ops.</param>
    /// <param name="parameters">Model parameters — same tensors used during
    /// original compilation. The plan's backward closures capture these by
    /// reference for in-place gradient accumulation.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The rehydrated plan, or null if incompatible.</returns>
    public static Task<ICompiledTrainingPlan<T>?> LoadTrainingAsync<T>(
        Stream stream,
        IEngine engine,
        Tensor<T>[] parameters,
        CancellationToken cancellationToken = default)
    {
        if (stream is null)     throw new ArgumentNullException(nameof(stream));
        if (engine is null)     throw new ArgumentNullException(nameof(engine));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        cancellationToken.ThrowIfCancellationRequested();

        try
        {
            var plan = TrainingPlanReader.Read<T>(stream, engine, parameters);
            return Task.FromResult<ICompiledTrainingPlan<T>?>(plan);
        }
        catch (InvalidDataException ex)
        {
            Trace.WriteLine($"[CompiledPlanLoader] Training plan load failed, recompiling: {ex.Message}");
            return Task.FromResult<ICompiledTrainingPlan<T>?>(null);
        }
    }

    /// <summary>
    /// Convenience overload: loads from a file path. Returns null if the
    /// file doesn't exist or the plan is incompatible.
    /// </summary>
    public static async Task<ICompiledPlan<T>?> LoadInferenceAsync<T>(
        string filePath,
        IEngine engine,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath)) return null;
        using var stream = File.OpenRead(filePath);
        return await LoadInferenceAsync<T>(stream, engine, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Convenience overload: loads training plan from a file path.
    /// </summary>
    public static async Task<ICompiledTrainingPlan<T>?> LoadTrainingAsync<T>(
        string filePath,
        IEngine engine,
        Tensor<T>[] parameters,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath)) return null;
        using var stream = File.OpenRead(filePath);
        return await LoadTrainingAsync<T>(stream, engine, parameters, cancellationToken).ConfigureAwait(false);
    }
}
