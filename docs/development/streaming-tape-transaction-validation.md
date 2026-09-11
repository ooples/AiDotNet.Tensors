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
| AMD/OpenCL GPU/COW/journal/lifetime/precision set | Earlier comparison set: 200 passed | 208 passed, 0 skipped, 0 failed, both before and after upstream merge |
| Library build: net10.0, net8.0, net471 | Not a before/after benchmark | Passed, 0 warnings, 0 errors |
| Test-project build: net10.0, net471 | Not a before/after benchmark | Passed, 195 warnings, 0 errors after upstream merge |

The initial CPU-family increase is eight array-access contract cases plus one
causal attention regression. The GPU-set increase is the same eight array-access
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
