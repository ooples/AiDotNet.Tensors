# GUARD_OK: emits operator API tests (scaled_mm decode scales), not model-weight compression.
import json
import pathlib

# Every path is relative to this script, so the generator runs from any checkout and any working directory.
here = pathlib.Path(__file__).resolve().parent
repo = here.parents[1]
with open(here / 'torch_mm_ref.json', encoding='utf-8') as f:
    ref = json.load(f)
def farr(v): return 'new float[] { ' + ', '.join(repr(float(x)) + 'f' for x in v) + ' }'
def iarr(v): return 'new int[] { ' + ', '.join(str(int(x)) for x in v) + ' }'
def T(d, k): return f'T({farr(d[k])}, {iarr(d[k + "_shape"])})'
lines = []
for name in ['g_2d3d', 'g_3d3d', 'g_2d2d', 'g_3d2d']:
    d = ref[name]
    offs = f'new Tensor<int>({iarr(d["offs"])}, new[] {{ {len(d["offs"])} }})' if 'offs' in d else 'null'
    lines.append(f'    [Fact]\n    public void GroupedMatMul_{name[2:]}_MatchesTorch()\n        => AssertClose({T(d,"out")}, _engine.TensorGroupedMatMul({T(d,"a")}, {T(d,"b")}, {offs}), 1e-5);\n')
d = ref['s_tensorwise']
lines.append(f'    [Fact]\n    public void ScaledMatMul_TensorWise_MatchesTorch()\n        => AssertClose({T(d,"out")}, _engine.TensorScaledMatMul({T(d,"a")}, {T(d,"b")}, {T(d,"sa")}, ScalingRecipe.TensorWise, {T(d,"sb")}, ScalingRecipe.TensorWise), 1e-3);\n')
d = ref['s_rowwise']
lines.append(f'    [Fact]\n    public void ScaledMatMul_RowWiseWithBias_MatchesTorch()\n        => AssertClose({T(d,"out")}, _engine.TensorScaledMatMul({T(d,"a")}, {T(d,"b")}, {T(d,"sa")}, ScalingRecipe.RowWise, {T(d,"sb")}, ScalingRecipe.RowWise, bias: {T(d,"bias")}), 1e-3);\n')
tmpl = (here / 'mm_tests_template.cs').read_text(encoding='utf-8')
(repo / 'tests' / 'AiDotNet.Tensors.Tests' / 'Engines' / 'GroupedScaledMatMulTests.cs').write_text(
    tmpl.replace('//@@TORCH_CASES@@', '\n'.join(lines)), encoding='utf-8')
print('ok')
