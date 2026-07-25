//! ptxgen — Rust half of the AiDotNet codegen bake-off.
//!
//! Emits SM86 PTX for the bake-off kernel and self-checks its interpreter against
//! an independent fp64 reference, so correctness is established with no GPU.
//!
//! Usage: ptxgen <out.ptx> [n c h w]

mod emit;
mod ir;

use emit::Emitter;
use ir::KernelSpec;

fn oracle(n: i64, c: i64, h: i64, w: i64, input: &[f64], weights: &[f64], bias: &[f64]) -> Vec<f64> {
    let mut out = vec![0f64; (n * c * h * w) as usize];
    for bn in 0..n {
        for bc in 0..c {
            for oh in 0..h {
                for ow in 0..w {
                    let mut acc = 0f64;
                    for kh in 0..3i64 {
                        for kw in 0..3i64 {
                            let ih = oh + kh - 1;
                            let iw = ow + kw - 1;
                            if ih < 0 || ih >= h || iw < 0 || iw >= w {
                                continue;
                            }
                            acc += input[(((bn * c + bc) * h + ih) * w + iw) as usize]
                                * weights[((bc * 3 + kh) * 3 + kw) as usize];
                        }
                    }
                    acc += bias[bc as usize];
                    out[(((bn * c + bc) * h + oh) * w + ow) as usize] = acc.max(0.0);
                }
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "generated.ptx".to_string());
    let n: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let c: i64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let h: i64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let w: i64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);

    let spec = KernelSpec::depthwise_conv2d_3x3_bias_relu(n, c, h, w);

    // Same deterministic generators as the C# test, so both sides see identical data.
    let input: Vec<f64> = (0..(n * c * h * w))
        .map(|i| (((i * 37 % 97) - 48) as f64 / 64.0) as f32 as f64)
        .collect();
    let weights: Vec<f64> = (0..(c * 9))
        .map(|i| (((i * 53 % 89) - 44) as f64 / 128.0) as f32 as f64)
        .collect();
    let bias: Vec<f64> = (0..c)
        .map(|i| (((i * 29 % 71) - 35) as f64 / 256.0) as f32 as f64)
        .collect();

    // Gate 1: the IR's semantics, checked on CPU with no GPU involved.
    let expected = oracle(n, c, h, w, &input, &weights, &bias);
    let actual = spec.interpret(&[input.clone(), weights.clone(), bias.clone()]);
    let worst = expected
        .iter()
        .zip(actual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    if worst > 1e-12 {
        eprintln!("FAIL: interpreter deviates from fp64 oracle by {worst:E}");
        std::process::exit(2);
    }

    // Gate 2: emit PTX.
    let mut em = Emitter::default();
    match em.emit(&spec, 8, 6) {
        Ok(ptx) => {
            std::fs::write(&path, &ptx).expect("write ptx");
            println!("interpreter vs fp64 oracle : PASS (max abs {worst:E})");
            println!("entry                      : {}", spec.name);
            println!("threads (single source)    : {}", spec.space.total_threads());
            println!("grid blocks                : {}", Emitter::grid_blocks(&spec));
            println!("emitted loads              : {}", em.emitted_loads);
            println!("guards elided (interval)   : {}", em.elided_guards);
            println!("ptx lines                  : {}", ptx.lines().count());
            println!("written                    : {path}");
        }
        Err(e) => {
            eprintln!("emit declined: {e}");
            std::process::exit(3);
        }
    }
}
