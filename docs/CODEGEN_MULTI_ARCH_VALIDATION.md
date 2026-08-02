# Codegen multi-architecture validation

`--kernel-arch-validate --require-ptxas` emits every catalog kernel for SM80, SM86,
SM89, and SM90, assembles every PTX program with `ptxas`, and writes
`artifacts/codegen-architecture-validation.tsv`. Each row carries the kernel, target,
PTX SHA-256, assembly result, and current protocol tag. `AIDOTNET_PTXAS_PATH` or
`--ptxas <path>` selects a toolkit explicitly.

This closes the portable-code check; it does not pretend one GPU is four GPUs. Release
performance still needs two physical lanes:

1. the primary Ampere lane runs correctness, paired head-to-head, autotune, competitor,
   limiter, and release evidence;
2. a second Ada or Hopper lane reruns correctness, head-to-head, and autotune and publishes
   its device-keyed artifacts.

Autotune rows include device UUID, compute target, driver, spec hash, and emitter build
fingerprint, so an Ampere winner cannot be consumed on the second lane. Until that physical
runner exists, the second-architecture performance gate is external and remains unsatisfied;
the ptxas matrix is assembly portability evidence only.
