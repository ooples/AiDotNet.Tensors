"""#1353: torch's per-block time for one ResNet basic block training forward (batch statistics), batch 8, fp32.

conv3x3 -> BatchNorm2d (train) -> ReLU -> conv3x3 -> BatchNorm2d (train) -> + x -> ReLU, at the ResNet-18 stage shapes
64@32, 128@16, 256@8, 512@4. Modes: eager and torch.compile (default and reduce-overhead), cudnn.benchmark on, TF32
off. Each mode: 20 warm-up blocks, then the best of 7 rounds of 50 back-to-back blocks timed with CUDA events. Prints
microseconds per block; the best mode per shape is the bar the fused kernel is compared against.
"""
import sys

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + x)


def per_block_us(fn, x):
    with torch.no_grad():
        for _ in range(20):
            fn(x)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                fn(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end) * 1000.0 / 50)
    return best


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 2
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    print(f"{'shape':<10}{'eager':>10}{'compile':>10}{'reduce-oh':>11}{'best':>10}")
    for c, hw in ((64, 32), (128, 16), (256, 8), (512, 4)):
        torch.manual_seed(c + hw)
        block = BasicBlock(c).cuda().train()
        x = torch.randn(8, c, hw, hw, device="cuda")
        results = {"eager": per_block_us(block, x)}
        for mode in ("default", "reduce-overhead"):
            try:
                compiled = torch.compile(block, mode=mode)
                results[mode] = per_block_us(compiled, x)
            except Exception as exc:  # inductor needs triton, which may be missing on Windows
                print(f"  torch.compile({mode}) unavailable: {type(exc).__name__}", file=sys.stderr)
                results[mode] = float("nan")
        best = min(v for v in results.values() if v == v)
        print(f"{f'{c}@{hw}':<10}{results['eager']:>10.1f}{results['default']:>10.1f}{results['reduce-overhead']:>11.1f}{best:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
