# Run AiDotNet against Python PyTorch on 11 model families

This is the end-to-end benchmark used to test how changes to AiDotNet.Tensors affect training and inference. The C# models live in AiDotNet. The runner fetches their source, builds them against the exact Tensors source used for the chosen run, and runs each PyTorch model in a separate Python process with `torch.compile`. Both sides use the same device.

| Recorded run | Results you can read on GitHub | Download the individual reports |
| --- | --- | --- |
| November 13 baseline, public mainline commits | [Training and inference CSV](evidence/2026-11-13.csv) | [88 reports and run settings](evidence/2026-11-13.zip) |
| November 19 experiment, combined PR revisions | [Training and inference CSV](evidence/2026-11-19.csv) | [88 reports, process receipts, source and binary fingerprints, and step-one checks](evidence/2026-11-19.zip) |

Each CSV includes both launches, across all 11 models, on CPU and CUDA. The 17-second LSTM video was a separate experiment with different source and a different training task.

You can read both implementations of the November 13 baseline directly on GitHub. [Python models and training loop](https://github.com/ooples/AiDotNet/blob/8e1066a7454f95b1f2441229b8e736032fb6111c/benchmarks/AiDotNet.PyTorchParity/pytorch/benchmark.py) · [C# benchmark](https://github.com/ooples/AiDotNet/tree/8e1066a7454f95b1f2441229b8e736032fb6111c/benchmarks/AiDotNet.PyTorchParity)

## Try one model

Install Git, the .NET 10 SDK, and Python 3.13. On Windows open **Developer PowerShell for Visual Studio** with the x64 C++ tools in your environment. PyTorch's compiler needs those tools. On Linux install a C++ compiler. For GPU runs use an NVIDIA driver and CUDA development libraries compatible with your PyTorch wheel and GPU.

From the root of this repo, create and activate a Python virtual environment. The original machine used Python 3.13.3, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, and the versions below. For the CPU-only wheels, change the index suffix from `cu126` to `cpu`.

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install psutil==7.2.2 numpy==2.5.4
```

For Windows CUDA, also install `python -m pip install triton-windows==3.9.0.post30`. Linux CUDA wheels install the matching Triton dependency. Then try a short CPU run.

```shell
python parity/model-families/run.py --models lstm --devices cpu --smoke
```

The first run fetches the sources and builds both projects in Release. The path printed at the end contains the two original JSON reports, their commands and logs, the runtime versions, assembly hashes and a CSV. Smoke mode checks that everything runs. Use the full protocol for timings.

## Run the full comparison

```shell
python parity/model-families/run.py --snapshot 2026-11-19 --verify-step1
python parity/model-families/run.py --snapshot 2026-11-13
```

Each command runs 88 processes for the timing matrix (11 models × 2 devices × 2 frameworks × 2 launches). Use `--devices cpu` or `--devices cuda` for one device, or `--models cnn resnet18` for a subset. Use `--work PATH` to put checkouts, builds, compiler caches and results on a drive with room. Each invocation gets its own results directory. Run one benchmark at a time and keep a laptop plugged in.

Both frameworks use eight CPU threads, batch size 64, seed 1234, and three epochs of 20 training updates. The training value is the reported median steady-state epoch time divided by 20, in milliseconds per update. Inference measures median latency at batch sizes 1, 8 and 16 with 100 timed iterations. The protocol requests 10 warm-up iterations and the harnesses' default 1,000 ms minimum warm-up. Reports retain the individual epoch times, loss, memory measurements and inference percentiles. A ratio above 1 in the CSV means AiDotNet took less time on that row. Each launch stays visible.

To regenerate a CSV from the archived JSON reports (Python standard library only):

```shell
python parity/model-families/report.py parity/model-families/evidence/2026-11-19.zip
```

## What the runs showed

The table gives the November 19 **training** ratios, PyTorch milliseconds divided by AiDotNet milliseconds, in launch order. The machine was a Ryzen 7 4800H laptop with a 6 GB GTX 1660 Ti, NVIDIA driver 617.42, and .NET runtime 10.0.12. The laptop was observed on battery during this experiment and power state was not continuously recorded. A few rows swung substantially between launches.

| Model | CPU, launch 1 / 2 | CUDA, launch 1 / 2 |
| --- | ---: | ---: |
| MLP | 2.81 / 3.71 | 4.04 / 9.08 |
| LSTM | 0.17 / 0.14 | 0.37 / 0.96 |
| Transformer | 2.07 / 1.39 | 1.41 / 4.28 |
| CNN | 3.07 / 3.22 | 0.96 / 3.27 |
| ResNet-18 | 4.54 / 4.80 | 6.25 / 1.62 |
| ResNet-50 | 2.74 / 3.26 | 3.28 / 1.67 |
| VGG-11 | 1.60 / 1.97 | 1.58 / 1.46 |
| ViT | 1.35 / 1.21 | 1.05 / 1.04 |
| GRU | 2.39 / 2.07 | 3.49 / 6.27 |
| MobileNetV2 | 1.78 / 1.74 | 0.90 / 1.93 |
| DenseNet-121 | 1.24 / 1.18 | 1.20 / 1.02 |

For November 13, AiDotNet took at least 10% less time to train in both launches for seven models on CUDA and ten on CPU. CUDA VGG-11, GRU and DenseNet-121 favored PyTorch. CUDA ViT was close, and the CPU DenseNet-121 launches disagreed. DenseNet-121 ran near the machine's memory limit.

`--verify-step1` exports weights and a batch from Python, then checks the C# result on each requested device. The archived November 19 checks pass for MLP, CNN, LSTM and Transformer. The other seven are timing comparisons. Inspect each model's parameter counts in the CSV and PyTorch's model-specific notes in its report. For example, the November 13 GRUs used different reset-gate formulas. By November 19 they used the same formula but still had different bias counts. ViT and MobileNetV2 also differ in parameter count. First-step agreement measures one update and does not establish accuracy after training.

## Source and evidence

[Revision manifest](sources/revisions.json) pins both repository commits and Git trees. The November 19 result came from local integration commits. Two small Git bundles under `sources/` contain the merge commits and conflict resolutions. `prepare.py` fetches their prerequisite public GitHub commits, verifies the bundles, and checks out and verifies both exact revisions. To inspect the model and optimizer code without running anything, run `python parity/model-families/prepare.py` and open `AiDotNet/benchmarks/AiDotNet.PyTorchParity` in the printed directory. The Python implementation is in `pytorch/benchmark.py`.

Each archive includes the original report hashes and launcher, with the host's home directory replaced in the launcher. The November 19 C# reports store the report filename in place of the original machine's output directory. The November 13 harness did not embed the command protocol or produce per-process receipts. Its archived environment, reports and original launcher settings identify the baseline. New runs record the full command and exit status for each process. The PyTorch launcher requests the default `torch.compile` mode. The archived reports contain no per-operation compiler trace or comparison against other PyTorch modes.
