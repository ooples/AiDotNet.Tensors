"""Generate IEngine.InPlace.cs / CpuEngine.InPlace.Generated.cs for the torch `op_` in-place variants.

Every in-place op runs its out-of-place engine op through ApplyInPlace (GraphMode, autograd, views, GPU dispatch via
IEngine). Ops listed in FAST get a true in-place path that writes the tensor's own buffer when nothing records.
Ops in MANUAL are written by hand in CpuEngine.InPlace.cs (random fills, metadata ops, ops whose base has an out
parameter or returns a different element type). Writes coverage updates to inplace_coverage.json.
"""
import json, re, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BENCH = os.path.join(ROOT, 'parity', 'build')
sigs = json.load(open(BENCH + '/sigs.json', encoding='utf-8'))

# torch name -> engine base op, for gaps whose mapped out-of-place symbol is a Vector overload or misnamed.
BASE_OVERRIDE = {
    'acosh_': 'TensorAcosh', 'arccosh_': 'TensorAcosh', 'asinh_': 'TensorAsinh', 'arcsinh_': 'TensorAsinh',
    'exp2_': 'TensorExp2', 'expm1_': 'TensorExpm1', 'log10_': 'TensorLog10', 'log1p_': 'TensorLog1p',
    'log2_': 'TensorLog2', 'rsqrt_': 'TensorRsqrt', 'tan_': 'TensorTan', 'trunc_': 'TensorTrunc', 'fix_': 'TensorTrunc',
}
# Signatures of the new ElementwiseExtra ops (not in sigs.json, which was extracted before they existed).
NEW_UNARY = {'TensorAcosh', 'TensorAsinh', 'TensorExp2', 'TensorExpm1', 'TensorLog10', 'TensorLog1p', 'TensorLog2',
             'TensorRsqrt', 'TensorTan', 'TensorTrunc'}

# Hand-written in CpuEngine.InPlace.cs (torch name -> IEngine symbol it maps to).
MANUAL = {
    'detach_': 'TensorDetachInPlace', 'dropout_': 'DropoutInPlace',
    'zero_': 'TensorZeroInPlace', 'copy_': 'TensorCopyInPlace', 'fill_': None,  # fill_ maps to the existing TensorFill
    'bernoulli_': 'TensorBernoulliInPlace', 'cauchy_': 'TensorCauchyInPlace', 'exponential_': 'TensorExponentialInPlace',
    'geometric_': 'TensorGeometricInPlace', 'log_normal_': 'TensorLogNormalInPlace', 'normal_': 'TensorNormalInPlace',
    'uniform_': 'TensorUniformInPlace', 'random_': 'TensorRandomIntegersInPlace',
    'squeeze_': 'TensorSqueezeInPlace', 'unsqueeze_': 'TensorUnsqueezeInPlace', 'transpose_': 'TensorTransposeInPlace',
    't_': 'TensorTInPlace', 'swapaxes_': 'TensorSwapAxesInPlace', 'swapdims_': 'TensorSwapAxesInPlace',
    'as_strided_': 'TensorAsStridedInPlace', 'resize_': 'TensorResizeInPlace', 'resize_as_': 'TensorResizeAsInPlace',
    'set_': 'TensorSetInPlace', 'fill_diagonal_': 'TensorFillDiagonalInPlace', 'requires_grad_': 'TensorRequiresGradInPlace',
    'apply_': 'TensorApplyInPlace', 'map_': 'TensorMapInPlace', 'map2_': 'TensorMap2InPlace',
    'embedding_renorm_': 'TensorEmbeddingRenormInPlace',
    # left for later batches: rename_, share_memory_, resize_as_sparse_, sparse_resize_, sparse_resize_and_clear_
}
LATER = {'rename_', 'share_memory_', 'resize_as_sparse_', 'sparse_resize_', 'sparse_resize_and_clear_'}

# Overloads that cannot be in place (output shape differs from self).
SKIP_OVERLOAD = [('ScatterAdd', 'int dim = 0, int? outputSize = null')]

# True in-place fast paths: base op -> (scalar C# expression over double `x` or None, float kernel, double kernel).
# Only where the result is bit-identical to the out-of-place op: the same scalar function the shared unary helper
# applies, or an exact / correctly rounded operation (floor, ceil, trunc, negate, abs, sign, square, sqrt).
FAST = {
    'TensorAsin': ('Math.Asin', None, None), 'TensorAcos': ('Math.Acos', None, None), 'TensorAtan': ('Math.Atan', None, None),
    'TensorAtanh': ('AtanhScalar', None, None), 'TensorAcosh': ('AcoshScalar', None, None),
    'TensorAsinh': ('AsinhScalar', None, None), 'TensorExp2': ('x => Math.Pow(2.0, x)', None, None),
    'TensorExpm1': ('Expm1Scalar', None, None), 'TensorLog10': ('Math.Log10', None, None),
    'TensorLog1p': ('Log1pScalar', None, None), 'TensorLog2': ('Log2Scalar', None, None),
    'TensorRsqrt': ('x => 1.0 / Math.Sqrt(x)', None, None), 'TensorTan': ('Math.Tan', None, None),
    'TensorTrunc': ('Math.Truncate', None, None),
    'TensorFloor': (None, 'FloorUnsafe', 'FloorUnsafe'), 'TensorCeiling': (None, 'CeilingUnsafe', 'CeilingUnsafe'),
}


def split_params(ps):
    out, depth, cur = [], 0, ''
    for ch in ps:
        if ch in '<([': depth += 1
        if ch in '>)]': depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out


def pname(p):
    p = p.split('=')[0].strip()
    return p.split()[-1]


def ptype(p):
    p = p.split('=')[0].strip()
    return ' '.join(p.split()[:-1])


decls, impls, coverage = [], [], {}
emitted = set()
for torch, d in sigs.items():
    if torch in MANUAL or torch in LATER:
        continue
    base = BASE_OVERRIDE.get(torch, d['engineName'])
    if base in NEW_UNARY:
        overloads = [{'gen': '<T>', 'plist': ['Tensor<T> tensor'], 'self': 0, 'ret': 'Tensor<T>'}]
    elif torch == 'eq_':
        # TensorEq returns Tensor<Bit>; TensorEquals is the same comparison as 0/1 in the tensor's own type, which is
        # what torch's eq_ writes.
        base = 'TensorEquals'
        overloads = [{'gen': '<T>', 'plist': ['Tensor<T> a', 'Tensor<T> b'], 'self': 0, 'ret': 'Tensor<T>'},
                     {'gen': '<T>', 'plist': ['Tensor<T> tensor', 'T value'], 'self': 0, 'ret': 'Tensor<T>'}]
    else:
        overloads = d['overloads']
    if not overloads:
        raise SystemExit(f'unresolved and not manual: {torch}')
    syms = set()
    for o in overloads:
        params = ' '.join(o['params'].split()) if 'params' in o else ', '.join(o['plist'])
        if any(base == b and tail in params for b, tail in SKIP_OVERLOAD):
            continue
        plist = list(o['plist'])
        self_p = plist.pop(o['self'])
        elem = re.match(r'(?:ReadOnly)?Tensor<(.+)>$', ptype(self_p)).group(1)
        ret_elem = re.match(r'Tensor<(.+)>$', o['ret']).group(1)
        if ret_elem != elem:
            raise SystemExit(f'{torch}: {base} returns Tensor<{ret_elem}> for Tensor<{elem}> input; needs MANUAL')
        if any(p.startswith(('out ', 'ref ')) for p in plist):
            raise SystemExit(f'{torch}: {base} has out/ref; needs MANUAL')
        name = base + 'InPlace'
        gen = o['gen']
        self_name = pname(self_p)
        sig_params = ', '.join([self_p] + plist)
        key = (name, tuple(ptype(p) for p in [self_p] + plist))
        syms.add(name)
        if key in emitted:
            continue
        emitted.add(key)
        # call of the out-of-place op with the snapshot `x` in self's position
        call_args = [pname(p) for p in o['plist']]
        # An operand that IS the target (a.addcmul_(a, a)) must read the snapshot too: the target itself is about to be
        # overwritten and to become the write's output, so recording it as an input would put a cycle in the graph.
        for j, p in enumerate(o['plist']):
            if j != o['self'] and ptype(p) == ptype(o['plist'][o['self']]):
                call_args[j] = f'ReferenceEquals({pname(p)}, {self_name}) ? snapshot : {pname(p)}'
        call_args[o['self']] = 'snapshot'
        others = [pname(p) for p in plist if ptype(p) == f'Tensor<{elem}>']
        others_arg = (', ' + ', '.join(others)) if others else ''
        decls.append(
            f'    /// <summary>In-place <see cref="{base}"/> (torch <c>{torch}</c>): overwrites <paramref name="{self_name}"/> with the result.</summary>\n'
            f'    void {name}{gen}({sig_params});\n')
        fast = ''
        unsafe_kw = ''
        if base in FAST and elem == 'T' and not plist:
            scalar, fk, dk = FAST[base]
            if scalar:
                fast = f'        if (TryUnaryInPlaceFast({self_name}, {scalar})) return;\n'
            else:
                unsafe_kw = 'unsafe '
                fast = (f'        if (TryUnaryInPlaceFast({self_name}, null, '
                        f'static (s, d, n) => Simd.SimdKernels.{fk}(s, d, n), static (s, d, n) => Simd.SimdKernels.{dk}(s, d, n))) return;\n')
        lam_args = ', '.join(call_args)
        impls.append(
            f'    /// <inheritdoc/>\n'
            f'    public virtual {unsafe_kw}void {name}{gen}({sig_params})\n'
            f'    {{\n'
            f'{fast}'
            f'        ApplyInPlace({self_name}, "{name}", (e, snapshot) => e.{base}({lam_args}){others_arg});\n'
            f'    }}\n')
    for cat in d['cats']:
        coverage[f'{cat}:{torch}'] = sorted(syms)[0]

for torch, sym in MANUAL.items():
    for cat in sigs[torch]['cats']:
        coverage[f'{cat}:{torch}'] = sym or 'TensorFill'

hdr = 'using System;\nusing AiDotNet.Tensors.LinearAlgebra;\n\nnamespace AiDotNet.Tensors.Engines;\n\n'
open(ROOT + '/src/AiDotNet.Tensors/Engines/IEngine.InPlace.Generated.cs', 'w', encoding='utf-8').write(
    '// <auto-generated> by parity/gen_inplace.py from the out-of-place IEngine signatures. </auto-generated>\n' + hdr +
    '/// <summary>In-place forms (torch <c>op_</c>) of the out-of-place engine ops.</summary>\npublic partial interface IEngine\n{\n'
    + '\n'.join(decls) + '}\n')
open(ROOT + '/src/AiDotNet.Tensors/Engines/CpuEngine.InPlace.Generated.cs', 'w', encoding='utf-8').write(
    '// <auto-generated> by parity/gen_inplace.py; the mechanism is in CpuEngine.InPlace.cs. </auto-generated>\n' + hdr +
    'public partial class CpuEngine\n{\n' + '\n'.join(impls) + '}\n')
json.dump(coverage, open(BENCH + '/inplace_coverage.json', 'w'), indent=1)
print(len(decls), 'in-place overloads;', len(coverage), 'coverage entries;', len(LATER), 'items left for later batches')
