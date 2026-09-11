# Streaming tape transaction validation

## Validated snapshot and scope

Final local validation on 2026-09-10, at merge commit
`2b48d9c6f50736cdc6ff75b46a3ce844c2f66358` on
`fix/streaming-tape-transaction`: fix commit `63fdc84e6` plus upstream
`5d22aa7f4`. All declared library/test target frameworks were rebuilt after the
merge. Binary hashes below identify that final build. The expanded post-merge
run selects every current `Autodiff` test class, not only the historical fixture
set used for the initial controlled comparison.

The machine has a Radeon RX 5500 XT, driver `32.0.21037.1004`. Physical GPU checks
used OpenCL. This is not CUDA/Metal runtime validation, an AiDotNet model-shard
result, or a throughput benchmark. Other target frameworks were built, not run.

## Results

| Check | Before | After |
| --- | --- | --- |
| Historical autodiff/COW/streaming family, before upstream merge | 1,078 passed, 29 skipped, 2 failed; 1,109 total | 1,089 passed, same 29 skipped, 0 failed; 1,118 total |
| Expanded post-merge autodiff/COW/streaming/precision/GEMM family | Broader selection, not the same test count | 1,508 passed, 30 skipped, 0 failed; 1,538 total |
| Five physical journal/cache invalidation repros | 0 passed, 5 failed | Same five passed, included in the GPU set below |
| AMD/OpenCL GPU/COW/journal/lifetime/precision set | Initial comparison: 200 passed, 0 skipped, 0 failed | Expanded set before merge: 208 passed; same set after merge: 208 passed; both 0 skipped, 0 failed |
| Library build: net10.0, net8.0, net471 | Not a before/after benchmark | Passed, 0 warnings, 0 errors |
| Test-project build: net10.0, net471 | Not a before/after benchmark | Passed, 195 warnings, 0 errors after upstream merge |

The initial CPU-family total increases by nine: eight array-access contract cases
plus one causal attention regression. The passed count increases by eleven because
the two existing MaxPool1D/attention failures also become passes. The GPU-set increase is the same eight array-access
cases. The expanded post-merge selection additionally passed all 18 inverse-trig,
14 ISTFT, 5 STFT-phase, 4 exact-type-precision, and 5 cross-thread GEMM cases.
The broad name filter also selected 371 GPU/CPU autodifferential cases and three
other matching cases. The one added skip is the existing
`Issue471CpuMatmulCrashRepro.Gpu_autodiff_training_stress` diagnostic; all 29
original skips remain unchanged. The expanded run used the CPU configuration;
its broader differential coverage is not counted as strict physical GPU proof.
The suites overlap; their totals must not be summed as unique test coverage.
Final run durations were approximately 1 minute 41 seconds for the expanded
family and 11 seconds for the strict AMD/OpenCL set.

The 208-test set is **not 208 physical kernel tests**. It includes these 41
explicitly hardware-required cases:

| Fixture | Hardware-required cases | Numerical/device-data-path cases |
| --- | ---: | ---: |
| `TensorGpuCachePhysicalTests` | 20 | 20 |
| Physical cases in `StreamingTensorJournalTests` | 5 | 5 |
| `GradientAccumulationPhysicalGpuTests` | 1 | 1 |
| `OpenClFp16NativeOpTests` | 9 | 8 |
| `OpenClHalfPrecisionGemmTests` | 6 | 5 |
| Total | 41 | 39 |

The other two cases validate argument guards with an initialized physical
backend. The 26 cache/journal/accumulation cases enable
`ThrowOnGpuKernelFallback`; the 13 OpenCL numerical cases call the physical
backend directly. Five additional nonempty DirectGpu COW cases ran with an
available GPU, but do not all forbid CPU fallback, so they are not counted as
strict numerical-kernel proof. The remaining coverage includes CPU contracts,
mock-buffer lifetimes, precision policy, and empty-input guards.

## Concrete before/after evidence

- Physical cache restore previously returned `[2, 4, 6, 8]` instead of
  `[20, 40, 60, 80]`. All five original physical cache repros now pass. Twenty
  additional physical cache cases cover late-created views, in-place mutation
  order, inference, independent raw-array/Vector wrappers, COW detachment, and
  retention of the same persistent allocation during repeated in-place work.
- Causal attention previously produced `dQ[1049] = 0.0015641242` instead of
  `0.041369293` for `(B=1,H=2,S=64,Dh=16)`, exceeding the unchanged `1e-3` tolerance.
  It also failed in isolation. The same test now passes; all 12 attention cases
  pass, including the new masked-first-query gradient/input-nonmutation canary.
- MaxPool1D previously produced gradient `0` instead of numerical `0.999451` at
  index 1. The original numerical-gradient test now passes.
- The actual AMD accumulation output was
  `expected=1.0005; inherited=1.0009766 (error=0.00047656250000005507); FP32=1.0005 (error=3.623962396837044E-08)`.
  This compares inherited half accumulation and explicit FP32 accumulation on
  the same physical backend, not simulated policy selection.

The attention/pooling writes now acquire writable spans and notify mutation.
The original full-contiguous-view backing-array contract is retained rather
than globally banning views. Eight array-access tests cover zero-copy reads,
COW write isolation, full-view write-through, offset/noncontiguous snapshots,
deferred materialization, and lazy evaluation. These establish allocation and
ownership contracts, not a measured end-to-end speedup.

## Commands used

Run from the repository root in PowerShell, with dependencies already restored.
These build all declared target frameworks:

```powershell
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore -p:GeneratePackageOnBuild=false --nologo -v quiet -clp:ErrorsOnly
dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --no-restore -p:GeneratePackageOnBuild=false --nologo -v quiet -clp:ErrorsOnly
```

These are the final post-merge commands, with the TRX-derived fixture lists
expanded into literals so they also work without the historical local artifacts.
The resulting filter strings were verified identical to those executed. The CPU
command deliberately uses broad `FullyQualifiedName~Autodiff` instead of the old
autodiff class list. No Tensors build ran concurrently with these tests.

CPU family:

```powershell
$ErrorActionPreference = 'Stop'
$report = 'pr2088-postmerge-after-autodiff-20260910.trx'
if (Test-Path -LiteralPath ('tests/AiDotNet.Tensors.Tests/TestResults/' + $report)) {
    throw 'Refusing to overwrite an existing proof report.'
}
$classes = @(
    'AiDotNet.Tensors.Tests.Engines.GemmDeterminismAcrossThreadsTests'
    'AiDotNet.Tensors.Tests.Engines.Gpu.ExactTypePrecisionTests'
    'AiDotNet.Tensors.Tests.Engines.Simd.AdamMomentKernelsParityTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.StreamingTensorPoolTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorArrayAccessContractTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorCowCloneTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorCowInferenceReadPathTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.WeightLifetimeIntegrationTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.WeightStreamingTransparentAccessTests'
)
$filter = (@('FullyQualifiedName~Autodiff') + @($classes | Sort-Object -Unique | ForEach-Object { 'FullyQualifiedName~' + $_ })) -join '|'
$env:AIDOTNET_FORCE_CPU = '1'
Remove-Item Env:AIDOTNET_REQUIRE_GPU_TESTS -ErrorAction SilentlyContinue
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -f net10.0 -c Release --no-build --no-restore --filter $filter --logger ('trx;LogFileName=' + $report) --logger 'console;verbosity=minimal' --nologo
```

Strict AMD/OpenCL set, in a separate PowerShell process:

```powershell
$ErrorActionPreference = 'Stop'
$report = 'pr2088-postmerge-after-gpu-cow-20260910.trx'
if (Test-Path -LiteralPath ('tests/AiDotNet.Tensors.Tests/TestResults/' + $report)) {
    throw 'Refusing to overwrite an existing proof report.'
}
$classes = @(
    'AiDotNet.Tensors.Tests.Engines.Autodiff.GradientAccumulationPhysicalGpuTests'
    'AiDotNet.Tensors.Tests.Engines.Compilation.CompiledTrainingCowIsolationTests'
    'AiDotNet.Tensors.Tests.Engines.DirectGpu.ActivationCacheEvictionLifetimeTests'
    'AiDotNet.Tensors.Tests.Engines.DirectGpu.InvalidateActivationCacheEntryLifetimeTests'
    'AiDotNet.Tensors.Tests.Engines.DirectGpu.OpenClFp16NativeOpTests'
    'AiDotNet.Tensors.Tests.Engines.DirectGpu.OpenClHalfPrecisionGemmTests'
    'AiDotNet.Tensors.Tests.Engines.Gpu.GpuPrecisionExecutionTests'
    'AiDotNet.Tensors.Tests.Engines.Gpu.GpuPrecisionPolicyTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.StreamingTensorJournalTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorArrayAccessContractTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorCowCloneTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorCowInferenceReadPathTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorGpuCachePhysicalTests'
    'AiDotNet.Tensors.Tests.LinearAlgebra.TensorGpuCacheVersionTests'
)
$filter = (@($classes | Sort-Object -Unique | ForEach-Object { 'FullyQualifiedName~' + $_ }) -join '|')
$env:AIDOTNET_DIRECTGPU_BACKENDS = 'opencl'
$env:AIDOTNET_REQUIRE_GPU_TESTS = '1'
Remove-Item Env:AIDOTNET_FORCE_CPU -ErrorAction SilentlyContinue
dotnet test tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -f net10.0 -c Release --no-build --no-restore --filter $filter --logger ('trx;LogFileName=' + $report) --logger 'console;verbosity=minimal' --nologo
```

## Preserved artifacts and SHA-256

TRXs are local test artifacts under
`tests/AiDotNet.Tensors.Tests/TestResults/`; they are not checked-in files or
published CI evidence. Preserve the original files rather than rerunning with
the same names. The isolated attention repro is also retained as
`pr2088-attention-isolation.trx`.

| TRX | SHA-256 |
| --- | --- |
| `pr2088-gpu-cache-before.trx` | `2CAEB7BD748FFA7AC83CECBBE35F9AC951C5173D3224C90AB66426C79EA1BAC2` |
| `pr2088-autodiff-family.trx` | `FDD5FF756304FC76B2042A20C76EE2D8F835AEB69A7E8DD0DFA11307D3B189D6` |
| `pr2088-final-after-autodiff-20260910.trx` | `BB3BC8C8C7C9724D8647699FF46ECAE8E42A623475C8498C3B5276A89B5B6FB9` |
| `pr2088-final-after-gpu-cow-20260910.trx` | `E9C317146E937B45FF188427AF65630D5CB8A234B6C208E82F81C8751B55F3C5` |
| `pr2088-postmerge-after-autodiff-20260910.trx` | `58B9BD8A51A7C2E54A2649C502035C1537B8C8C73642426C92DAD393378FE417` |
| `pr2088-postmerge-after-gpu-cow-20260910.trx` | `DCFC1182F378378BA51316FC06CFFB9BB2F3BDEDE956E1D258A9AB0C84CF7B2E` |

| Release binary | SHA-256 |
| --- | --- |
| Library `net10.0/AiDotNet.Tensors.dll` | `D81257F6E1ADDD2939ED463329EF46FA1ABD0C6C6DA1B76D377A73321823E7A6` |
| Library `net8.0/AiDotNet.Tensors.dll` | `57A9E24D3305769309927962617C6600EB1D3CF32028ACEDB4EB2A26FD48A537` |
| Library `net471/AiDotNet.Tensors.dll` | `C8F3DAF1DF7CF0404360FA8D7FF1ECFE0479F1C8391CE709E111D54A90136BB4` |
| Test `net10.0/AiDotNet.Tensors.Tests.dll` | `3E6FC73F501606CEDC2D02E32C63D018D7CC378AE4570C6CA89E345C4FA39D5C` |

The copied `net10.0` library in the test output has the same hash as the library
build, verified again after both runs. All 3,922 added C# lines relative to
`origin/main` were checked for null-forgiving operators: none found.
`git diff --check origin/main -- src tests` was clean. Local `.token-optimizer`
hook-generated files are unrelated and must be excluded from the companion PR.

## Stable-package release gate (not yet satisfied)

An additional, actual stable-version pack check on 2026-09-11 failed:

```powershell
dotnet pack src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release `
    --no-build --no-restore -o artifacts/pr1029-package-validation `
    --nologo -v minimal -p:Version=0.130.4
```

The result was `NU5104` for each of the three target frameworks: a stable Tensors
package depends on `AiDotNet.Evolution [0.1.0-preview.1, )`. A normal local build
does not expose this because the project's default version is itself prerelease.
The all-framework rebuild immediately before this check still had zero warnings
and zero errors. The package failure is not counted as successful validation.

The Evolution reference is already present in the merged `main` baseline; it is
not introduced by the storage fixes. Nevertheless, it blocks publishing those
fixes as a stable Tensors update. At this check the official NuGet feed contained
only the prerelease Evolution package. Stable Evolution publication followed by
a normal stable package reference is required; suppressing `NU5104` would not
prove that release path. `0.130.4` above is only the local candidate version used
to exercise the gate, not a published version or a reserved release number.

AiDotNet also needs its existing Evolution package-ownership migration (#2092)
before consuming this dependency graph normally. Its local source-reference
alias is a diagnostic aid, not a solution for downstream package consumers.


## PR #1029 review-fix validation (2026-09-11)

This section validates the review-fix working tree based on
`230dbb53ac0ee650753ed3bd5d65e2b8f298deb2`; the earlier sections describe
historical snapshots. No package reference or warning suppression was changed.
The stable-package release gate above remains unsatisfied.

| Check | Actual result |
| --- | --- |
| Controlled original-source repro, `pr1029-review-autodiff-before.trx` | 1 passed, 3 failed: strided MaxPool1D gradient; compiled and optimized FP32 policy propagation |
| Focused CPU review cases, `pr1029-review-cpu-final.trx` | 100 passed, 27 explicit GPU skips, 0 failed |
| Final broad autodiff/COW/streaming/precision family, `pr1029-review-autodiff-final.trx` | 1,166 passed, 429 explicit skips, 0 failed; 1,595 total |
| Final strict AMD/OpenCL set, `pr1029-review-opencl-final.trx` | 226 passed, 0 skipped, 0 failed |
| Library build, net10.0/net8.0/net471 | 0 warnings, 0 errors |
| Test build, net10.0/net471 | 191 warnings, 0 errors |

The broad CPU command explicitly disables GPU backends. Its 429 skips include
the 371 GPU autodifferential cases selected by the broad `Autodiff` name filter,
plus other GPU-only and pre-existing skipped diagnostics. It is not the same
hardware configuration as the historical 1,508-passed run; the lower passed count
does not represent failed tests. The final broad CPU run took 79 seconds, and the
strict AMD run took 10 seconds. These overlapping suites must not be summed.

The strict selection retains all 208 historical cases and adds 18: eight OpenCL
context-ownership cases, three persistent-cache ownership cases, three journal
boundary/fault cases, three operation-kind fallback cases, and one repeated
explicit-invalidation case. Of the 226 total, **50 require hardware**: 21 physical
cache, five journal, one accumulation, nine native-FP16, six half-GEMM, and eight
OpenCL context cases. Forty-two exercise numerical/device data paths; eight
exercise argument/context rejection. Mock-buffer and CPU tests are not counted
as physical kernels.

Concrete proof from this review pass:

- The original strided MaxPool1D case threw on `AsSpan()`; it now produces
  `[0,1,0,3,0,2,0,4]` while preserving the noncontiguous input gradient.
- Both original compiled-policy failures observed FP16 fan-out addition despite
  a configured FP32 policy. Six current tape/compiled/optimized cases prove
  configured FP32 and inherited policy, unchanged FP16 backward kernels,
  numerical gradient `1.0005`, and restoration of a conflicting outer policy.
- AMD accumulation still reports
  `expected=1.0005; inherited=1.0009766 (error=0.00047656250000005507); FP32=1.0005 (error=3.623962396837044E-08)`.
- Repeated raw-array mutation without an epoch increment followed by explicit
  invalidation produces real GPU values `[20,4,6,8]` then `[22,4,6,8]`.
  Separate tracking-backend cases verify exactly one disposal per allocation
  after successful, failed, and eight concurrent invalidations.
- Six actual OpenCL cases reject foreign-context float/byte wrappers in each
  operand position; two same-context control cases execute GEMM successfully.
- Injected journal write failure preserves committed records when range cleanup
  succeeds; injected cleanup failure invalidates every record and poisons all
  subsequent operations. Oversized deferred payload rejection does not
  materialize the backing array.

Two intermediate failures in the newly added physical invalidation test were
test-contract mistakes, not hidden successes: OpenCL returns disposed allocations
to its pool without zeroing their native handle, and eager reads may clear a
tensor-local shortcut while retaining its persistent cache entry. The final test
tracks published borrowed allocations and device values; exact disposal remains
covered by the tracking backend. The failed intermediate reports
`pr1029-review-opencl-after.trx` and `pr1029-review-opencl-complete.trx` are
preserved. A new OpenCL fixture also initially failed the net471 test build until
it adopted the existing fixture's `NET6_0_OR_GREATER` guard.

### Review checklist

These are source/test responses awaiting independent review, not claims that
GitHub threads have already been resolved.

| Review comment ID | Response and coverage |
| --- | --- |
| 3985388622 | Distinguish initial 200-case GPU baseline from expanded 208-case pre/post-merge runs. |
| 3985388625 | Explain nine new CPU cases plus two existing failures becoming passes. |
| 3985388633 | Materialize only noncontiguous MaxPool1D gradients before read-only span access; controlled red/green regression. |
| 3985388665 | Carry accumulation policy into compiled and optimized backward execution; six policy/scope cases. |
| 3985388673 | Persistent-weight comments now describe storage `GpuCacheVersion`. |
| 3985388681 | Reject foreign OpenCL contexts before precision dispatch; six rejection and two same-context cases. |
| 3985388689 | Remove the unused private untracked-memory overload and require an epoch at every persistent-cache insertion; preserve existing tracked-cache fast paths. |
| 3985388698 | Serialize all persistent replacement publishers under the existing lock and retire the displaced allocation; failure/concurrency ownership and real GPU mutation cases. |
| 3985388704 | Preserve typed operation kind in every CPU precision fallback; three planner cases. |
| 3985388711 | Reject unsupported byte length before deferred materialization; allocation-free boundary case. |
| 3985388713 | Typed internal file-operation fault seam proves recoverable append failure versus poisoned cleanup; diagnostic wording now names append cleanup. |
| 3985388721 | Continue bulk allocator cleanup after individual failures and remove registrations exactly once; reset and dead-owner-prune cases. |
| 3985388726 | Accumulation tests dynamically skip missing hardware/support unless hardware is explicitly required. |
| 3985388731 | Pin the default FP32 accumulation policy. |
| 3985388738 | CPU engine observations test the actual tape option-to-fan-out path and backward kernel precision, not only scope helpers. |
| 3985388746 | Bound readiness/start/completion waits and signal readiness even when worker setup fails. |
| 3985388752 | State `ReleaseAfterBackward` explicitly in the saved-state release regression. |
| 3985388759 | All five hardware-dependent journal cases report explicit skips without hardware. |
| 3985388764 | Physical cache cases report explicit skips without hardware; the strict AMD run executes all 21. |

The untracked-cache review concern was addressed without accepting version-blind
cache hits: repository-wide call-site inspection found that only an unused private
`ReadOnlyMemory<T>` overload could publish an untracked entry. Removing that
overload and the sentinel makes the epoch mandatory at compile time instead of
weakening stale-value detection. Cache ownership changes affect replacement and
invalidation, not the unchanged lock-free read path.

### Reproduction commands for this review snapshot

From the repository root, after ordinary restore:

```powershell
dotnet build tests/AiDotNet.Tensors.Tests/AiDotNet.Tensors.Tests.csproj -c Release --no-restore --nologo -v quiet
dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release --no-restore --nologo -v quiet
```

CPU selection (the controlled before run used only
`FullyQualifiedName~BackwardReviewRegressionTests` before applying the fixes):

```powershell
$names = @(
    'Autodiff'
    'GemmDeterminismAcrossThreadsTests'
    'ExactTypePrecisionTests'
    'AdamMomentKernelsParityTests'
    'StreamingTensorPoolTests'
    'TensorArrayAccessContractTests'
    'TensorCowCloneTests'
    'TensorCowInferenceReadPathTests'
    'WeightLifetimeIntegrationTests'
    'WeightStreamingTransparentAccessTests'
    'StreamingTensorJournalTests'
    'GpuPrecisionPolicyTests'
    'PersistentCacheOwnershipTests'
)
$filter = ($names | ForEach-Object { 'FullyQualifiedName~' + $_ }) -join '|'
$report = 'pr1029-review-autodiff-final.trx'
if (Test-Path -LiteralPath ('tests/AiDotNet.Tensors.Tests/TestResults/' + $report)) {
    throw 'Proof report already exists; choose a new name.'
}
$env:AIDOTNET_FORCE_CPU = '1'
$env:AIDOTNET_DIRECTGPU_BACKENDS = 'none'
Remove-Item Env:AIDOTNET_REQUIRE_GPU_TESTS -ErrorAction SilentlyContinue
dotnet vstest tests/AiDotNet.Tensors.Tests/bin/Release/net10.0/AiDotNet.Tensors.Tests.dll ("/TestCaseFilter:" + $filter) ("/Logger:trx;LogFileName=" + $report) /ResultsDirectory:tests/AiDotNet.Tensors.Tests/TestResults '/Logger:console;Verbosity=quiet'
```

Strict AMD selection, in a separate process:

```powershell
$names = @(
    'GradientAccumulationPhysicalGpuTests'
    'CompiledTrainingCowIsolationTests'
    'ActivationCacheEvictionLifetimeTests'
    'InvalidateActivationCacheEntryLifetimeTests'
    'OpenClFp16NativeOpTests'
    'OpenClHalfPrecisionGemmTests'
    'GpuPrecisionExecutionTests'
    'GpuPrecisionPolicyTests'
    'StreamingTensorJournalTests'
    'TensorArrayAccessContractTests'
    'TensorCowCloneTests'
    'TensorCowInferenceReadPathTests'
    'TensorGpuCachePhysicalTests'
    'TensorGpuCacheVersionTests'
    'OpenClPrecisionBufferOwnershipTests'
    'PersistentCacheOwnershipTests'
)
$filter = ($names | ForEach-Object { 'FullyQualifiedName~' + $_ }) -join '|'
$report = 'pr1029-review-opencl-final.trx'
if (Test-Path -LiteralPath ('tests/AiDotNet.Tensors.Tests/TestResults/' + $report)) {
    throw 'Proof report already exists; choose a new name.'
}
$env:AIDOTNET_DIRECTGPU_BACKENDS = 'opencl'
$env:AIDOTNET_REQUIRE_GPU_TESTS = '1'
Remove-Item Env:AIDOTNET_FORCE_CPU -ErrorAction SilentlyContinue
dotnet vstest tests/AiDotNet.Tensors.Tests/bin/Release/net10.0/AiDotNet.Tensors.Tests.dll ("/TestCaseFilter:" + $filter) ("/Logger:trx;LogFileName=" + $report) /ResultsDirectory:tests/AiDotNet.Tensors.Tests/TestResults '/Logger:console;Verbosity=quiet'
```

### Final review artifacts

The final broad CPU and strict GPU runs used the same unchanged binaries.
Reports remain local artifacts under `tests/AiDotNet.Tensors.Tests/TestResults/`.

| Artifact | SHA-256 |
| --- | --- |
| Controlled before TRX | `B79D7696BAEDEE9962F444D289BF7E9169959E96F0964FB5E61325798B6C0D59` |
| Final broad CPU TRX | `9CC95BD2FC4D3EDAD6732D6222682782804FFE961E06ABA5F508BF090EB27A7D` |
| Final strict GPU TRX | `0CA833E03B725653C6DA2A1D97DEDD27BD7A08E228E2EE0F2933BFDAC7F0479C` |
| Library net10.0, including copied test dependency | `8585288EFB410BD932B06138945CD0588BAA8F6749E8625872087EA5129080A2` |
| Library net8.0 | `E11FBF4AB63EFB170D6BA1D8B2D5AC06436835DAB7E0920C256BA00B0D970F15` |
| Library net471 | `5CD93510BD559A81DAFF15FBDD6882E98BB953845AAACA03E0F73C6610053930` |
| Final net10.0 test binary | `849031DB58DE2C62B7CA7892F09F40CF702BC3E7A6D1F1195E9D8DB35DF3D426` |

Added tracked C# lines and all three new C# files were scanned for null-forgiving
operators: none found. `git diff --check` is clean. No CUDA/Metal runtime or
throughput claim is made by these checks. The unrelated `.token-optimizer`
hook files remain excluded from review-fix source changes.

### Independent cleanup follow-up and final source identity

Independent review found two additional secondary-failure paths: a user-supplied
trace listener could interrupt bulk reclamation after an allocator failure, and
replacement-buffer disposal could throw before a failed invalidation evicted its
stale cache metadata. The focused failure-first run
`pr1029-root-double-fault-before.trx` reproduced **3 failures and 3 passing
controls**. Diagnostics are now best effort, and stale-entry cleanup is in a
`finally` block, even when replacement disposal also fails. No lock was added to
ordinary cache reads.

Final independently executed results:

- `pr1029-root-double-fault-after.trx`: 17 passed, no failures/skips.
- `pr1029-root-broad-after.trx`: 1,170 passed, 429 explicit GPU skips, no failures.
- `pr1029-root-strict-opencl-after.trx`: 228 passed, no failures/skips. The two
  additions to the earlier 226 are mocked double-failure ownership controls;
  the number of hardware-required cases remains 50, not 228 physical kernels.
- `pr1029-root-cleanup-net471-after.trx`: 17 passed, no failures/skips.
- Rebuilt the library on all three targets: zero warnings/errors. Rebuilt both
  test targets: 191 existing warnings, zero errors. A temporary trace-listener
  annotation mismatch was removed; the final test binary is byte-identical to
  the one used by the first three final reports above.

Use the preceding reproduction commands with the new report names; the CPU and
GPU selection sets are unchanged. The final net10 library SHA-256 is
`5A62A0BF737909AEFC81F038293AA668CA6C355DC38029BF37C20DF580D6E624`;
the final net10 test binary is
`3EA8B926BA28CF7D16A48C82C43A55433F31B8E17FC81D7AD174491A07F308E9`.
The earlier hashes describe their explicitly named earlier snapshots.

The stable-package `NU5104` prerequisite described above still applies. Passing
local runtime checks does not claim that an unavailable stable Evolution package
has been published, that stable packing succeeds, or that this PR is merge-ready.
