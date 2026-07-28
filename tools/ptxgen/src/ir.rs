//! Index-map IR — the Rust half of the codegen bake-off.
//!
//! Mirrors `Codegen/Ir/CodegenAffine.cs` + `CodegenKernelSpec.cs` so the two
//! toolchains can be compared on identical ground. The invariants are the same and
//! exist for the same reason: a hand-recomputed bounds guard in the C# PTX kernels
//! dropped a factor the launch grid still had, and half a gradient was silently
//! never written.
//!
//! Where Rust differs materially from the C# version is called out in comments —
//! that difference is the point of the bake-off.

/// One `coefficient * axis` term of an affine index expression.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub axis: usize,
    pub coeff: i64,
}

/// A quasi-affine index expression: `(sum(coeff*axis) + constant) / divisor`.
///
/// Unlike the C# version, out-of-range is encoded in the return type rather than
/// an `out bool` parameter, so a caller physically cannot read an index without
/// having handled the invalid case.
#[derive(Clone, Debug)]
pub struct AffineExpr {
    pub terms: Vec<Term>,
    pub constant: i64,
    pub divisor: i64,
    pub requires_exact: bool,
}

impl AffineExpr {
    pub fn axis(a: usize) -> Self {
        Self { terms: vec![Term { axis: a, coeff: 1 }], constant: 0, divisor: 1, requires_exact: false }
    }

    /// Direct-convolution window: `axis*stride + tap - padding`.
    pub fn window(spatial: usize, tap: usize, stride: i64, padding: i64) -> Self {
        Self {
            terms: vec![Term { axis: spatial, coeff: stride }, Term { axis: tap, coeff: 1 }],
            constant: -padding,
            divisor: 1,
            requires_exact: false,
        }
    }

    /// Transposed convolution: `(axis + padding - tap) / stride`, exact-division only.
    pub fn transposed_window(spatial: usize, tap: usize, stride: i64, padding: i64) -> Self {
        Self {
            terms: vec![Term { axis: spatial, coeff: 1 }, Term { axis: tap, coeff: -1 }],
            constant: padding,
            divisor: stride,
            requires_exact: stride != 1,
        }
    }

    /// Evaluates for concrete axis values. `None` means the access is invalid, so
    /// the type system forces the caller to handle zero-padding.
    pub fn eval(&self, axes: &[i64]) -> Option<i64> {
        let mut num = self.constant;
        for t in &self.terms {
            num += t.coeff * axes[t.axis];
        }
        if self.divisor == 1 {
            return Some(num);
        }
        if num < 0 {
            return None;
        }
        if self.requires_exact && num % self.divisor != 0 {
            return None;
        }
        Some(num / self.divisor)
    }

    /// True when the map can address outside the tensor and therefore needs a guard.
    pub fn can_escape(&self) -> bool {
        self.constant < 0
            || self.divisor != 1
            || self.terms.len() > 1
            || self.terms.iter().any(|t| t.coeff < 0)
    }
}

/// One axis of the iteration space.
#[derive(Clone, Debug)]
pub struct Axis {
    pub name: &'static str,
    pub extent: i64,
    pub reduction: bool,
}

impl Axis {
    pub fn parallel(name: &'static str, extent: i64) -> Self {
        Self { name, extent, reduction: false }
    }
    pub fn reduce(name: &'static str, extent: i64) -> Self {
        Self { name, extent, reduction: true }
    }
}

/// The axis list, and the single authority on the launch grid.
#[derive(Clone, Debug)]
pub struct IterationSpace {
    pub axes: Vec<Axis>,
}

impl IterationSpace {
    pub fn new(axes: Vec<Axis>) -> Self {
        assert!(axes.iter().any(|a| !a.reduction), "need at least one parallel axis");
        Self { axes }
    }

    pub fn parallel_axes(&self) -> Vec<usize> {
        self.axes.iter().enumerate().filter(|(_, a)| !a.reduction).map(|(i, _)| i).collect()
    }

    pub fn reduction_axes(&self) -> Vec<usize> {
        self.axes.iter().enumerate().filter(|(_, a)| a.reduction).map(|(i, _)| i).collect()
    }

    /// Single source of truth: the host grid and the in-kernel guard both use this.
    pub fn total_threads(&self) -> i64 {
        self.axes.iter().filter(|a| !a.reduction).map(|a| a.extent).product()
    }

    pub fn reduction_trips(&self) -> i64 {
        self.axes.iter().filter(|a| a.reduction).map(|a| a.extent).product::<i64>().max(1)
    }
}

/// Binds a tensor parameter to the iteration space.
#[derive(Clone, Debug)]
pub struct Binding {
    pub param: usize,
    pub name: &'static str,
    pub shape: Vec<i64>,
    pub map: Vec<AffineExpr>,
    pub is_output: bool,
}

impl Binding {
    pub fn new(
        param: usize,
        name: &'static str,
        shape: Vec<i64>,
        map: Vec<AffineExpr>,
        is_output: bool,
    ) -> Self {
        assert_eq!(shape.len(), map.len(), "{name}: rank mismatch between shape and index map");
        Self { param, name, shape, map, is_output }
    }

    pub fn stride(&self, dim: usize) -> i64 {
        self.shape[dim + 1..].iter().product::<i64>().max(1)
    }

    pub fn elements(&self) -> i64 {
        self.shape.iter().product()
    }

    /// Flat offset, or `None` when any dimension is out of range. The derived
    /// predicate lives here and nowhere else.
    pub fn resolve(&self, axes: &[i64]) -> Option<i64> {
        let mut off = 0;
        for (d, e) in self.map.iter().enumerate() {
            let idx = e.eval(axes)?;
            if idx < 0 || idx >= self.shape[d] {
                return None;
            }
            off += idx * self.stride(d);
        }
        Some(off)
    }
}

/// How reduction axes combine.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Reduce {
    None,
    Sum,
    Max,
}

/// Epilogue activation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Activation {
    None,
    ReLU,
}

/// A complete kernel: iteration space + bindings + reduce-and-epilogue body.
#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub name: String,
    pub space: IterationSpace,
    pub inputs: Vec<Binding>,
    pub output: Binding,
    pub product_inputs: Vec<usize>,
    pub reduce: Reduce,
    pub bias_input: Option<usize>,
    pub scale_input: Option<usize>,
    pub activation: Activation,
}

impl KernelSpec {
    pub fn param_count(&self) -> usize {
        self.inputs.len() + 1
    }

    /// fp64 CPU reference execution — the semantic definition, no GPU required.
    pub fn interpret(&self, data: &[Vec<f64>]) -> Vec<f64> {
        let axes_meta = &self.space.axes;
        let parallel = self.space.parallel_axes();
        let reduction = self.space.reduction_axes();
        let mut values = vec![0i64; axes_meta.len()];
        let mut out = vec![0f64; self.output.elements() as usize];

        for tid in 0..self.space.total_threads() {
            let mut rest = tid;
            for &p in parallel.iter().rev() {
                let e = axes_meta[p].extent;
                values[p] = rest % e;
                rest /= e;
            }

            let mut acc = if self.reduce == Reduce::Max { f64::NEG_INFINITY } else { 0.0 };
            for t in 0..self.space.reduction_trips() {
                let mut r = t;
                for &ra in reduction.iter().rev() {
                    let e = axes_meta[ra].extent;
                    values[ra] = r % e;
                    r /= e;
                }

                // `?`-style propagation: an out-of-range tap yields the additive
                // identity, and the type system will not let us forget the case.
                let product = self
                    .product_inputs
                    .iter()
                    .try_fold(1.0f64, |acc, &i| {
                        let off = self.inputs[i].resolve(&values)?;
                        Some(acc * data[i][off as usize])
                    })
                    .unwrap_or(0.0);

                acc = match self.reduce {
                    Reduce::Sum => acc + product,
                    Reduce::Max => acc.max(product),
                    Reduce::None => product,
                };
            }

            if let Some(b) = self.bias_input {
                if let Some(off) = self.inputs[b].resolve(&values) {
                    acc += data[b][off as usize];
                }
            }
            if let Some(s) = self.scale_input {
                if let Some(off) = self.inputs[s].resolve(&values) {
                    acc *= data[s][off as usize];
                }
            }
            if self.activation == Activation::ReLU && acc < 0.0 {
                acc = 0.0;
            }
            if let Some(off) = self.output.resolve(&values) {
                out[off as usize] = acc;
            }
        }
        out
    }

    /// The bake-off target: depthwise Conv2D 3x3 + bias + ReLU.
    pub fn depthwise_conv2d_3x3_bias_relu(n: i64, c: i64, h: i64, w: i64) -> Self {
        const N: usize = 0;
        const C: usize = 1;
        const OH: usize = 2;
        const OW: usize = 3;
        const KH: usize = 4;
        const KW: usize = 5;

        let space = IterationSpace::new(vec![
            Axis::parallel("n", n),
            Axis::parallel("c", c),
            Axis::parallel("oh", h),
            Axis::parallel("ow", w),
            Axis::reduce("kh", 3),
            Axis::reduce("kw", 3),
        ]);

        let input = Binding::new(
            0,
            "input",
            vec![n, c, h, w],
            vec![
                AffineExpr::axis(N),
                AffineExpr::axis(C),
                AffineExpr::window(OH, KH, 1, 1),
                AffineExpr::window(OW, KW, 1, 1),
            ],
            false,
        );
        let weights = Binding::new(
            1,
            "weights",
            vec![c, 3, 3],
            vec![AffineExpr::axis(C), AffineExpr::axis(KH), AffineExpr::axis(KW)],
            false,
        );
        let bias = Binding::new(2, "bias", vec![c], vec![AffineExpr::axis(C)], false);
        let output = Binding::new(
            3,
            "output",
            vec![n, c, h, w],
            vec![
                AffineExpr::axis(N),
                AffineExpr::axis(C),
                AffineExpr::axis(OH),
                AffineExpr::axis(OW),
            ],
            true,
        );

        Self {
            name: format!("aidotnet_gen_dwconv2d3x3_n{n}_c{c}_h{h}_w{w}_relu"),
            space,
            inputs: vec![input, weights, bias],
            output,
            product_inputs: vec![0, 1],
            reduce: Reduce::Sum,
            bias_input: Some(2),
            scale_input: None,
            activation: Activation::ReLU,
        }
    }
}
