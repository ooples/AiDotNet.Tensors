# Run AiDotNet against Python PyTorch on 11 model families

This end-to-end benchmark measures how changes to AiDotNet.Tensors affect training and inference. The C# models live in AiDotNet. This launcher fetches the exact AiDotNet source used for a recorded run, builds it, and runs it on your machine against Python PyTorch with `torch.compile`, on CPU and on an NVIDIA GPU.

| Recorded run | Results you can read on GitHub | Raw reports |
| --- | --- | --- |
| **November 21–26, current** (AiDotNet `9dc1c828`, Tensors 0.151.0 from NuGet): main comparison and large datasets | [Results and how to read them](https://github.com/ooples/AiDotNet/blob/d831cfbf75b45739632bbd8070b9f740fda15a0c/benchmarks/AiDotNet.PyTorchParity/README.md#results-from-november-2126-2026) · [Large-data results](https://github.com/ooples/AiDotNet/blob/d831cfbf75b45739632bbd8070b9f740fda15a0c/benchmarks/AiDotNet.PyTorchParity/README.md#large-datasets---scale) | [Every report, receipt and log](https://github.com/ooples/AiDotNet/blob/d831cfbf75b45739632bbd8070b9f740fda15a0c/benchmarks/AiDotNet.PyTorchParity/evidence) |
| November 20, superseded (AiDotNet `37597f19`): PyTorch's LSTM and GRU ran eagerly | [Summary table](evidence/2026-11-20.md) · [Summary CSV](evidence/2026-11-20-summary.csv) · [Every launch](evidence/2026-11-20.csv) | [Every report, receipt and log](evidence/2026-11-20.zip) |
| November 13, archived | [Training and inference CSV](evidence/2026-11-13.csv) | [88 reports and run settings](evidence/2026-11-13.zip) |
| November 19, archived | [Training and inference CSV](evidence/2026-11-19.csv) | [88 reports, process receipts, fingerprints and step-one checks](evidence/2026-11-19.zip) |

Use the November 21–26 runs for any performance claim. The snapshot pins `9dc1c828`, the measured code; their results and raw evidence were published one commit later, in AiDotNet `d831cfbf`, which also merges later AiDotNet master changes that these runs did not measure. The November 20 run compared the LSTM and GRU only against eager PyTorch, because `torch.compile` skips recurrent modules unless `allow_rnn` is set. [Why the two archived runs are not evidence](#the-archived-runs) explains the problems with the older ones. The [17-second LSTM video benchmark](../lstm-video/README.md) has its own training code and recordings.

## Try one model

Install Git, the .NET 10 SDK and Python 3.13. On Windows, open the x64 **Developer PowerShell for Visual Studio** with the C++ tools, because `torch.compile` needs `cl.exe`. On Linux, install a C++ compiler. For GPU runs, use an NVIDIA driver that supports your PyTorch wheel.

From the root of this repo, create and activate a Python virtual environment and install the versions the recorded runs used. For the CPU-only wheels, change the index suffix from `cu126` to `cpu`.

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install psutil==7.2.2 numpy==2.5.4
```

PyTorch publishes Triton only for Linux. For Windows CUDA, also install the community build: `python -m pip install triton-windows==3.9.0.post30`. Then try a short CPU run:

```shell
python parity/model-families/run.py --models lstm --devices cpu --smoke
```

The first run fetches the pinned AiDotNet source and builds it in Release. The printed directory contains each process's report, command, log and receipt, along with the runtime versions, assembly hashes and the summary. Smoke mode only checks that everything runs.

## Run the full comparison

```shell
python parity/model-families/run.py
```

This reproduces the November 21–26 runs (`--snapshot 2026-11-26`, the default; `--snapshot 2026-11-20` reproduces the superseded one). It runs the pinned revision's own `run_matrix.py`, so the protocol is the one in that revision: five launches; AiDotNet against every applicable PyTorch configuration, including ones that compile the LSTM and GRU; the comparison against the fastest configuration for each row; and a first-update correctness check before any timing. The pinned [runner](https://github.com/ooples/AiDotNet/blob/9dc1c8287faa0ea46236985b9dd5e268a09d51bf/benchmarks/AiDotNet.PyTorchParity/run_matrix.py) defines the configurations, flags and outputs, and the AiDotNet benchmark README describes them. Use `--devices cpu` or `--devices cuda` for one device, `--models cnn resnet18` for a subset, and `--launches N` to change the number of launches. Use `--work PATH` to put checkouts, builds, compiler caches and results on a drive with room. The runner refuses to start on battery power. Close other GPU and CPU-heavy programs: the summary flags rows whose launches started on a busy machine.

## Large datasets

The 2026-11-26 snapshot adds a large-data mode. Every model trains whole epochs over real data (EMNIST Digits and CINIC-10) and over 1,000,000-sample synthetic datasets, and both frameworks report test accuracy after every epoch. The datasets take about 5.5 GB and are built and checksum-verified on the first run:

```shell
python parity/model-families/run.py --devices cuda --launches 3 --scale D:/parity-data
```

A full CUDA run takes about 20 hours on a GTX 1660 Ti. If your GPU computes wrong values at its boost clocks, as the recording laptop's does, lock the clock first (`nvidia-smi -lgc 1200,1200` as administrator) and add `--gpu-clock-lock 1200`. The runner then refuses to start unless the lock is in effect. `--scale-torch-from SUMMARY_CSV` limits each row to the PyTorch configuration that was fastest in a main-comparison summary. The [AiDotNet benchmark README](https://github.com/ooples/AiDotNet/blob/d831cfbf75b45739632bbd8070b9f740fda15a0c/benchmarks/AiDotNet.PyTorchParity/README.md#large-datasets---scale) describes the protocol, the results, the ViT fix the runs led to, and the GPU clock lock.

To rebuild the November 20 tables from its raw reports (Python standard library and the pinned AiDotNet source):

```shell
python -m zipfile -e parity/model-families/evidence/2026-11-20.zip archive-20
python parity/model-families/prepare.py --snapshot 2026-11-20
python <printed directory>/AiDotNet/benchmarks/AiDotNet.PyTorchParity/run_matrix.py --summarize archive-20
```

## What the superseded November 20 run showed

These are the November 20 numbers. The current results are linked at the top of this page.

The machine was a Ryzen 7 4800H laptop on AC power, with a 6 GB GTX 1660 Ti, NVIDIA driver 617.42, Windows 11, .NET 10 and PyTorch 2.14.1+cu126. Each row compares AiDotNet with PyTorch's fastest configuration for that row. GRU, ViT and MobileNetV2 carry ⚑ on every row because their parameter counts differ slightly (`parameters-differ`).

**CUDA**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 6.53 (5.92–7.33) | 1.43 (0.92–1.81) ⚑ | 1.67 (1.60–1.78) | 1.63 (0.69–1.75) ⚑ |
| lstm | 6.55 (6.14–6.67) | 1.48 (1.01–1.60) ⚑ | 1.91 (1.75–1.97) | 1.83 (1.75–2.70) ⚑ |
| transformer | 2.06 (1.85–2.35) | 0.40 (0.34–0.54) ⚑ | 0.57 (0.53–0.67) | 0.71 (0.66–1.08) ⚑ |
| cnn | 2.37 (2.14–2.56) | 0.87 (0.30–1.30) ⚑ | 1.41 (1.34–2.12) ⚑ | 1.50 (1.25–1.58) |
| resnet18 | 1.30 (1.18–1.54) | 2.12 (1.95–2.51) | 1.99 (0.93–2.12) ⚑ | 2.00 (1.38–2.10) |
| resnet50 | 1.71 (1.51–1.80) | 1.42 (1.33–1.47) | 1.48 (1.33–1.51) | 1.32 (1.30–1.36) |
| vgg11 | 1.18 (1.16–1.18) | 2.21 (2.13–2.33) | 1.69 (1.54–1.77) | 1.41 (1.39–1.47) |
| vit | 1.02 (1.02–1.02) ⚑ | 0.62 (0.55–0.66) ⚑ | 1.01 (0.83–1.04) ⚑ | 0.93 (0.92–0.95) ⚑ |
| gru | 8.73 (7.99–10.06) ⚑ | 1.73 (1.47–2.49) ⚑ | 2.29 (2.24–3.17) ⚑ | 2.65 (1.67–2.82) ⚑ |
| mobilenetv2 | 0.61 (0.58–0.62) ⚑ | 1.77 (1.26–1.79) ⚑ | 0.96 (0.93–2.90) ⚑ | 0.86 (0.81–0.89) ⚑ |
| densenet121 | 1.00 (0.99–1.01) | 1.34 (1.27–1.49) | 1.70 (1.69–1.71) | 1.31 (1.30–1.31) |

**CPU**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.

| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| mlp | 2.65 (2.55–3.00) | 3.95 (3.73–4.10) | 4.28 (3.54–4.95) | 4.72 (4.17–4.82) |
| lstm | 2.28 (2.10–2.30) | 3.97 (3.85–4.09) | 1.58 (1.46–2.30) | 1.93 (1.75–2.09) |
| transformer | 1.65 (1.34–1.73) | 2.18 (2.11–2.28) | 3.00 (2.89–3.11) | 2.77 (2.31–3.05) |
| cnn | 1.14 (0.98–1.59) | 3.86 (3.49–5.18) | 2.53 (2.44–2.91) | 1.67 (1.36–2.52) ⚑ |
| resnet18 | 4.67 (4.02–4.67) | 2.18 (1.96–2.59) | 1.95 (1.72–2.05) | 1.63 (1.45–1.78) |
| resnet50 | 3.06 (2.86–3.22) | 1.68 (1.33–1.98) ⚑ | 1.55 (1.42–1.97) ⚑ | 1.60 (1.40–1.60) |
| vgg11 | 1.43 (1.39–1.50) | 2.01 (1.95–2.26) | 2.88 (2.74–3.48) | 1.26 (1.23–1.50) |
| vit | 1.24 (1.13–1.41) ⚑ | 1.28 (1.19–1.92) ⚑ | 1.09 (0.96–1.23) ⚑ | 1.34 (1.22–1.46) ⚑ |
| gru | 2.35 (1.90–2.56) ⚑ | 15.40 (11.49–16.09) ⚑ | 6.59 (6.47–7.33) ⚑ | 4.77 (4.66–5.77) ⚑ |
| mobilenetv2 | 1.73 (1.63–1.80) ⚑ | 2.54 (2.22–2.67) ⚑ | 1.94 (1.84–2.09) ⚑ | 1.36 (1.27–1.50) ⚑ |
| densenet121 | 1.24 (1.08–1.31) | 1.39 (1.09–1.58) | 1.21 (0.91–1.36) | 1.29 (1.17–1.49) |

Where AiDotNet took longer, or the result is too close to call:

- **CUDA training.** MobileNetV2 is slower (0.61). DenseNet-121 (1.00) and ViT (1.02) are even.
- **CUDA inference.** The Transformer is slower at every batch size (0.40–0.71). ViT is slower at batch 1 (0.62) and batch 16 (0.93). MobileNetV2 is slower at batches 8 and 16 (0.96, 0.86), and the CNN at batch 1 (0.87).
- **CPU.** AiDotNet's median is lower on every row. On some rows a single launch went the other way: CNN training (0.98), ViT at batch 8 (0.96) and DenseNet-121 at batch 8 (0.91).

Most `unstable` flags are on CUDA inference calls that take less than a millisecond. At that scale, one launch of either framework can take up to three times as long as another. The range column shows how far the launches spread. No row was flagged `battery`, `busy-cpu`, `busy-gpu`, `ci-job` or `missing-launches`. The run was interrupted once by a switch to battery power. Its one battery-flagged process is kept under `superseded/` in the ZIP, and the process was rerun on AC power.

The fastest PyTorch configuration for most CUDA rows was `torch-cudagraphs-fused` (10 of 11 training rows and 28 of 33 inference rows). On CPU it was `torch-compile-fused` for every training row.

These are results for one Windows laptop. On Windows, PyTorch's GPU compiler uses the community `triton-windows` build, and Windows display-driver scheduling adds GPU launch overhead that Linux does not have. We have not run this on Linux yet, so these numbers do not show how the two frameworks compare there. The scripts are written to run on Linux, but this has not been tested. A Linux run on the same GPU would show how much of the CUDA inference result comes from the platform.

## The archived runs

The November 13 and 19 runs came before the current protocol. Their reports are preserved exactly, and this launcher still reproduces them (`--snapshot 2026-11-13` or `--snapshot 2026-11-19`, two launches by default, with `--verify-step1` for the first-update check). They are not evidence:

- PyTorch ran in one configuration: default `torch.compile` with PyTorch's default AdamW. Inference at batch sizes 8 and 16 used the dynamic-shape graph that `torch.compile` creates after it sees a second batch size.
- Each run had two launches, and AiDotNet always ran first. Power, load and AiDotNet's GPU backend were not recorded for each process. When rerun now, the launcher alternates the order, pins the CUDA backend and refuses battery power.
- **November 19** used local merge commits of then-open pull requests that never shipped. The laptop was on battery. Its first CUDA launch was throttled, with both frameworks 5–10× slower than in launch 2: PyTorch's ResNet-18 took 339 ms against 59 ms, and AiDotNet's ViT took 342 ms against 37 ms. Its LSTM training was 14× slower than on November 13 on CPU and 7–28× slower on CUDA, with identical losses. This was a regression in that build, and the November 20 code does not have it.
- GRU, ViT and MobileNetV2 differ from their PyTorch versions in parameter count by up to 1% (the current summary flags them `parameters-differ`). On November 13 the two GRUs also applied the reset gate at different points.

To regenerate an archived CSV from its reports (Python standard library only):

```shell
python parity/model-families/report.py parity/model-families/evidence/2026-11-19.zip
```

## Source and evidence

The [revision manifest](sources/revisions.json) pins each snapshot's commits and Git trees. The November 20 snapshot pins one public AiDotNet commit and uses the Tensors NuGet package that commit references. The November 19 result came from local integration commits. Two small Git bundles under `sources/` contain its merge commits and conflict resolutions. `prepare.py` fetches the public prerequisite commits, verifies the bundles, and checks out and verifies each exact revision. To inspect a snapshot's model and optimizer code without running anything, run `python parity/model-families/prepare.py --snapshot NAME` and open `AiDotNet/benchmarks/AiDotNet.PyTorchParity` in the printed directory.
