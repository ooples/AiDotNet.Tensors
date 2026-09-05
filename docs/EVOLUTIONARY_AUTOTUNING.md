# Evolutionary autotuning

`AiDotNet.Tensors` uses `AiDotNet.Evolution` only where a configuration space is large enough that a fixed sweep is
no longer economical. Small finite spaces remain exhaustive. In particular, convolution tile selection currently
has at most four candidates and deliberately does not use evolution.

## Safety boundary

Every searchable configuration is a typed immutable value. Algorithm families are enums and launch parameters are
numeric fields. Strings and dictionaries are confined to the `IEvolutionGenomeCodec<T>` and legacy
`AutotuneCache` serialization boundaries; production selection never branches on a diagnostic label.

The evaluation delegate must run the real backend and return:

- at least three post-warmup timing samples, from which median and P95 are derived;
- output correctness evidence and, when applicable, gradient correctness evidence;
- workspace, occupancy, register, compilation, and launch resource data; and
- measured throughput.

Invalid geometry, compilation failures, resource-limit failures, and numerical mismatches cannot enter the
MAP-Elites archive. A candidate replaces the active deployment only when it clears the configured measured gain
threshold. The default threshold is five percent, so noise-level changes do not rewrite a stable winner.
The same typed deployment invariant is applied before evaluation, after selection, and while hydrating persisted
state, so a malformed evaluator or stale cache row cannot publish a configuration that the current backend rejects.

The identity includes kernel, shape, physical device and driver, search-space version, and benchmark-protocol
version. Persisted payloads are decoded, re-encoded canonically, and checked against their stored hash before use.
Community configurations are proposals only: the local resource and correctness gates still apply before they can
be evaluated or deployed.

## Runtime cost

Evolution runs only in an explicit offline, startup, or caller-admitted idle workflow. Background APIs require an
`IKernelTuningIdleGate`, and searches targeting the same physical device are serialized. Serving code receives a
pre-resolved `KernelTuningDeployment<TConfiguration>`; its hit path is one volatile reference read and a typed
assignment, with no filesystem access, parsing, reflection, hashing, or search.

## Integrated domains

- `GemmAutoTuner.CreateEvolutionTuner` searches actual typed `GemmConfig` code-generation and launch fields. The
  `GemmKernelTemplate` enum, not `KernelName`, selects generated source. Heuristic, Bayesian, prewarm, and community
  candidates are locally validated and deduplicated before evaluation. `TuneWithEvolutionAsync` installs the active
  typed winner into the existing GEMM dispatch cache, including a compatible winner hydrated from disk.
- `BlasManagedEvolutionAutotuner` searches packing, blocking, parallel axis, and thread count. It rejects semantic
  aliases and nondeterministic reduction axes before benchmarking, then publishes a promoted result into the
  existing managed-BLAS dispatch memo so GEMM does not gain another hot-path lookup.
- `EinsumEvolutionAutotuner` searches typed pairwise contraction orders. `EinsumPathOptimizer` now persists and
  reconstructs the actual pair sequence instead of ignoring cache hits and rerunning greedy planning. Its bounded
  in-memory cache keeps filesystem work off repeated execution. Cache identity includes the exact execution device,
  search-space version, and benchmark protocol; callers targeting a GPU use the typed `Optimize` overload so a GPU
  winner can never leak into the default CPU path.
- `CodegenTiledContractionEvolutionExplorer` explores arbitrary valid PTX tile geometry offline. Explorer output is
  isolated by the existing canonical spec and emitted-search-space fingerprints as well as device and target.
  Explorer output is intentionally not production dispatch evidence: a useful discovery must be added to the finite
  codegen schedule catalog and pass the existing full correctness, stability, and direct-finalist championship first.

The evolution engine's evaluation memo and Tensors' deployment caches have separate roles. The first avoids
evaluating the same canonical candidate twice inside a run. The second decides whether measured evidence is valid
for a later process and provides the pre-resolved winner used by runtime dispatch.

For local development before the package is published, build with:

```text
-p:UseLocalEvolution=true -p:EvolutionProjectPath=<path-to-AiDotNet.Evolution.csproj>
```
