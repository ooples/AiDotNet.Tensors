# Evolutionary autotuning

`AiDotNet.Tensors` uses `AiDotNet.Evolution` only where a configuration space is large enough that a fixed sweep is
no longer economical. Small finite spaces remain exhaustive. In particular, convolution tile selection currently
has at most four candidates and deliberately does not use evolution.

## Safety boundary

Every searchable configuration is a typed immutable value. Algorithm families are enums and launch parameters are
numeric fields. Strings and dictionaries are confined to the `IEvolutionGenomeCodec<T>` and legacy
`AutotuneCache` serialization boundaries; production selection never branches on a diagnostic label.

The first-party `KernelTuningExperiment<TConfiguration>` scaffold owns preparation, independent-oracle validation,
warmup, timing, control calibration, and finalist replay. A backend supplies executable operations and typed resource
evidence; it does not supply a precomputed score. Custom evaluators remain an extension trust boundary.

Each accepted search measurement carries:

- at least three post-warmup timing samples, from which median and P95 are derived;
- output correctness evidence and, when applicable, gradient correctness evidence;
- a typed workload and timing scope; and
- resource values whose state is explicitly `Measured`, `NotApplicable`, or `Unavailable`.

Latency is primary. Work rate is derived from the declared work per operation and median latency; GFLOP/s exists only
for workloads declared as floating-point operations. CPU experiments use `StopwatchKernelTuningTimer`. GPU users can
combine `DeviceEventKernelTuningTimer` with `GpuKernelTuningDeviceClock`, which records native timing events on the
same CUDA, HIP, or OpenCL stream as the tuned operation. A GPU backend/queue must have timing or profiling enabled;
hardware lanes are required before making device-performance claims.

Invalid geometry, compilation failures, missing required descriptors, resource-limit failures, and numerical
mismatches cannot enter the MAP-Elites archive. Evolution nominates a finalist; it does not authorize deployment.
The finalist is replayed directly against the current production incumbent with alternating execution order, raw
paired holdout samples, and a separate incumbent/incumbent control replay. Promotion requires the configured median
gain, a lower empirical speedup bound of at least one, and the configured P95 limit, all above the calibrated noise
floor. The default median threshold is five percent.
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
  aliases and nondeterministic reduction axes before benchmarking. Its built-in path uses deterministic inputs, a
  scalar reference oracle, real managed GEMM execution, paired replay against the production `Auto` policy, and
  exhaustive search when the complete canonical space fits both budgets. A partial enumeration is discarded and
  never reported as exhaustive. Promoted plans replay the exact strategy, axis, blocks, and thread cap. The current
  benchmark identity covers the plain default GEMM context only, so calls with workspaces, prepacked operands,
  epilogues, beta-zero mode, or explicit controls retain their established dispatch until separately tuned.
- `EinsumEvolutionAutotuner` searches typed pairwise contraction orders. `EinsumPathOptimizer` now persists and
  reconstructs the actual pair sequence instead of ignoring cache hits and rerunning greedy planning. Its bounded
  in-memory cache keeps filesystem work off repeated execution. Cache identity includes the exact execution device,
  search-space version, and benchmark protocol; callers targeting a GPU use the typed `Optimize` overload so a GPU
  winner can never leak into the default CPU path.
- `CodegenTiledContractionEvolutionExplorer` explores arbitrary valid PTX tile geometry offline. Explorer output is
  isolated by the existing canonical spec and emitted-search-space fingerprints as well as device and target.
  Explorer output is intentionally not production dispatch evidence: a useful discovery must be added to the finite
  codegen schedule catalog and pass the existing full correctness, stability, and direct-finalist championship first.

`TuneExhaustiveAsync` is the generic proof path for a declared small finite domain. It rejects duplicate canonical
identities or a domain larger than either configured budget, disables variation, and evaluates exactly the complete
domain before applying the same finalist gate. `TuneAsync` remains the bounded evolutionary path for large spaces.

The evolution engine's evaluation memo and Tensors' deployment caches have separate roles. The first avoids
evaluating the same canonical candidate twice inside a run. The second decides whether measured evidence is valid
for a later process and provides the pre-resolved winner used by runtime dispatch.

For local development before the package is published, build with:

```text
-p:UseLocalEvolution=true -p:EvolutionProjectPath=<path-to-AiDotNet.Evolution.csproj>
```
