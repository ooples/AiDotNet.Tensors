# Sharing one machine between trainings

Two training processes can run on one machine at once. Each one can be given its own share of GPU memory and its own
set of CPU cores. Two settings do this; both are read from the environment, so a launcher can apply them without changing
the program.

| Setting | What it does |
|---|---|
| `AIDOTNET_GPU_MEMORY_FRACTION=0.5` | Caps the GPU memory this process reserves on each device at that fraction of the device's memory. |
| `AIDOTNET_CPU_CORES=0-31` | Binds this process's worker pool to those physical cores, and runs one thread per core by default. |

In code, `GpuMemoryBudget.SetPerProcessMemoryFraction(0.5, "cuda:0")` or `GpuMemoryBudget.SetMemoryLimit(bytes, "cuda:0")`
sets the GPU budget for one device. It takes precedence over the environment variable. This is the same feature as
PyTorch's `torch.cuda.set_per_process_memory_fraction`.

## Example: two jobs on one GPU and one CPU

On a 64-core machine with one GPU:

```
AIDOTNET_GPU_MEMORY_FRACTION=0.45 AIDOTNET_CPU_CORES=0-31  train-a
AIDOTNET_GPU_MEMORY_FRACTION=0.45 AIDOTNET_CPU_CORES=32-63 train-b
```

Leave part of the GPU unbudgeted. The CUDA context, cuBLAS/cuDNN handles and the display use memory that the budget does
not count.

## GPU memory budget

- **What counts.** The budget counts *reserved* memory, as PyTorch's does: tensors in use plus the memory the library's
  pools keep for reuse, including CUDA's stream-ordered pool. On CUDA it also counts the cuDNN workspace, the training
  step arena's blocks, and the scratch buffers the library allocates for cuBLASLt algorithm selection, cuRAND, the
  direct-PTX runtime and NCCL staging.
- **Over the budget.** A new device allocation that would pass the budget first releases cached and unreachable memory,
  the same release a driver out-of-memory triggers. If it is still over, it fails with the backend's out-of-memory error
  (`GpuOutOfMemoryException` on CUDA, OpenCL, Vulkan and WebGPU; `OutOfMemoryException` on Metal; `hipErrorOutOfMemory` on
  HIP). Code that already handles a driver out-of-memory, such as a kernel that falls back to needing no workspace,
  handles a budget refusal the same way.
- **Reuse is not checked.** Reusing pooled memory is not a new reservation, so the budget never refuses it. On CUDA,
  a stream-ordered request is charged only for what the pool's free memory does not cover, and is judged afterwards
  only by what it added to the pool: a request the pool served without growing always stands, also under a budget set
  below what the pool already holds. If the pool grew past the budget (fragmentation), the allocation is returned and
  retried once after the reclaim, then refused. Inside a stream capture nothing can be reclaimed, so such a request is
  refused at once; the capture is abandoned and the step runs eagerly.
- **Concurrent allocations.** A request is held against the budget from the moment it is admitted until it is counted
  as reserved, so threads allocating at the same moment cannot together pass the budget.
- **A budget of zero.** A fraction smaller than one byte of the device is a budget that admits nothing, not the absence
  of one.
- **Why it matters on Windows.** Without a budget, a process that outgrows the GPU does not fail. The Windows driver backs
  the overflow with system memory, and both processes slow down severalfold instead.
- **Device names.** Devices are named as `GpuMemoryStats` names them: `cuda:0`, `hip:0`, `opencl:0`, `vulkan:0`,
  `metal:0`, `webgpu:0`.
- **Fractions.** A fraction is of the memory the backend reports. For Metal that is the recommended working-set size. WebGPU
  reports no memory size, so only a byte limit applies there.

## CPU core set

- **What it fixes.** The worker pool binds each worker to its own physical core. Without a core set, every process binds
  its workers to the first cores of its processor group. Two processes started in the same group then run on the same
  cores and slow each other down.
- **Format.** Physical cores are numbered in machine order, across processor groups. The list takes indexes and ranges:
  `0-15,32,40-47`.
- **What is confined.**
  - When every listed core is in the process's own processor group, the whole process is confined to those cores and
    their SMT siblings, so the threads that dispatch work stay there too.
  - Otherwise only the pool's workers are bound. A process's affinity mask can name only its own group.
- **Thread count.** `AIDOTNET_CPU_THREADS`, when set, still sets the thread count.
- **Platforms.**
  - Windows: as above.
  - Linux: every thread of the process is confined to the listed cores and their SMT siblings (read from sysfs), and
    threads started later inherit that. Workers are not bound one by one. A core whose logical processor number is 64
    or higher cannot be named this way; use `taskset` or `numactl` for those.
  - macOS has no processor affinity. The setting sets the thread count only, and a warning is traced.
- **Failures.** A core the machine does not have, or a confinement the operating system refuses, throws an error naming
  `AIDOTNET_CPU_CORES` on the first CPU work of the process, whether or not that work uses the pool (a single listed
  core runs everything on the calling thread). The process is never left sized for cores it is not confined to.
- **Library assets.** Core selection is in the `net8.0` and `net10.0` assets. The `net471`, `net462` and
  `netstandard2.0` assets ignore `AIDOTNET_CPU_CORES`: set `AIDOTNET_CPU_THREADS` there and confine the process from
  its launcher (`start /affinity`, `taskset`, `numactl`).

## What is not isolated

Neither setting isolates speed on the GPU. On consumer GPUs (no MIG, and no MPS on Windows), kernels from two processes
take turns on the device. Each training runs slower than it would alone, and timings taken on a shared GPU are not
benchmark numbers.
