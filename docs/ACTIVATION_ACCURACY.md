# Activation accuracy, measured

The emitter previously had two activations — `None` and `ReLU` — and emitted **zero**
transcendental instructions. That blocked four open PRs on a missing capability rather than
on wiring: LayerNorm+GELU (#863), GLU (#864), and the softmax family (#868, #884).

`Sigmoid`, `Tanh`, `Swish` and `Gelu` now exist end to end: in the fp64 oracle, in the PTX
emitter, and in the front end so a graph carrying them translates.

## These are the first kernels that cannot be exact

PTX has no `exp`. It has `ex2.approx.f32`, so every exponential is `ex2(x · log₂e)`, and
reciprocals use `rcp.approx.f32`. Both are **approximate** — about 2 ulp and 1 ulp — so an
activation kernel cannot reach the `0.000E+000` the affine kernels hit against the fp64
oracle. The deviation is therefore measured, not assumed.

Measured on device, relative to the fp64 interpretation of the same spec, on a
`conv 1x1 + bias + activation` graph of 100,352 elements:

| activation | implementation | max relative deviation |
|---|---|---|
| `ReLU` | `max.f32` | **0.000E+000** (exact — `max` is not an approximation) |
| `Swish` | `ex2` + `rcp` | 8.450E-008 |
| `Gelu` | `ex2` + `rcp` | 9.368E-008 |
| `Sigmoid` | `ex2` + `rcp` | 9.574E-008 |
| `Tanh` | `ex2` + `rcp` | 1.576E-007 |

All four sit at ~1e-7, consistent with two approximate instructions composed.

## The native tanh instruction was the weak link, so it is not used

The first implementation used `tanh.approx.f32`, which sm_75+ provides natively. Measuring
it against the oracle showed it is by far the least accurate thing in the set:

| activation | with `tanh.approx.f32` | with `ex2`-derived tanh | improvement |
|---|---|---|---|
| `Tanh` | 7.268E-006 | **1.576E-007** | **46×** |
| `Gelu` | 2.437E-006 | **9.368E-008** | **26×** |

So tanh is built from the same primitives the two accurate activations use:

```
tanh(x) = sign(x) · (1 − 2 / (exp(2|x|) + 1))
```

Taking the absolute value first is what makes it stable. For large `|x|` the exponential
saturates to infinity, the reciprocal underflows to zero, and the result lands on exactly
±1 with no special case — whereas evaluating the raw ratio `(e^{2x}−1)/(e^{2x}+1)` would
produce `inf/inf`. `copysign.f32` restores the sign.

A test asserts `tanh.approx` does **not** appear in the emitted PTX for either `Tanh` or
`Gelu`, so the faster-but-worse instruction cannot quietly return.

## GELU is the tanh form, and that is part of the definition

```
gelu(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

Not the erf form. The two differ by roughly 1e-3 near |x| = 2 — three orders of magnitude
above any floating-point concern — so this is a choice about *which operator* is being
compiled, not an implementation detail. The oracle and the emitter evaluate the same
formula, and a test pins the gap to the erf form so an "equivalent formula" edit cannot
silently change the operator.

## What is deliberately not mapped

`LeakyReLU` and `ELU` carry a slope parameter that `CodegenKernelSpec` has nowhere to
store. Mapping them at a default slope would compile a different operator than the graph
asked for, so the front end declines them by name.

## Reproducing

```
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -f net10.0 -- --frontend-check
```

The four activation rows report their own deviation. 17 of 17 rows pass, the thirteen
affine ones at exactly `0.000E+000` and the four activation ones at ~1e-7.
