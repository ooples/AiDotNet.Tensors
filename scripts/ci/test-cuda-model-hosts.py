"""Exercise the workflow's model process loop without a CUDA device (requires PyYAML and pwsh)."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/gpu-cuda-selfhosted.yml').read_text())
STEPS = next(iter(WORKFLOW['jobs'].values()))['steps']
MODEL_STEP = next(s['run'] for s in STEPS if s.get('name', '').startswith('AiDotNet model families'))
LOOP = MODEL_STEP[MODEL_STEP.index('$modelDeadline ='):]
LOOP = re.sub(r'\$\{\{.*?\}\}', 'test-temp', LOOP)


class ModelHosts(unittest.TestCase):
    def run_loop(self, scenario):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'calls.jsonl'
            script = Path(folder) / 'run.ps1'
            prefix = r'''
$ErrorActionPreference = 'Stop'
$env:MODEL_FAMILIES = 'mlp,cnn,densenet121'
$script:clockCalls = 0
function Get-Date {
  $script:clockCalls++
  $minutes = if ($env:SCENARIO -eq 'expired' -and $script:clockCalls -gt 1) { 21 } else { $script:clockCalls }
  [datetime]::new(2026, 1, 1).AddMinutes($minutes)
}
function Start-Process {
  param($FilePath, $ArgumentList, [switch]$NoNewWindow, [switch]$PassThru)
  @{args=$ArgumentList} | ConvertTo-Json -Compress | Add-Content $env:CALL_LOG
  $code = if ($env:SCENARIO -eq 'failure') { 7 } else { 0 }
  $process = [pscustomobject]@{Handle=1; Id=123; ExitCode=$code}
  $process | Add-Member ScriptMethod WaitForExit {
    param($milliseconds)
    if ($milliseconds -le 0 -or $milliseconds -gt 1200000) { throw 'Invalid watchdog budget' }
    return $true
  }
  return $process
}
'''
            script.write_text(prefix + LOOP)
            env = dict(os.environ, SCENARIO=scenario, CALL_LOG=str(log))
            result = subprocess.run(['pwsh', '-NoProfile', '-File', str(script)],
                                    env=env, capture_output=True, text=True, timeout=30)
            calls = [json.loads(line)['args'] for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_fresh_process_per_family_and_reuse_build(self):
        result, calls = self.run_loop('success')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 3)
        for model, args in zip(('mlp', 'cnn', 'densenet121'), calls):
            self.assertIn(f'--models {model} --device cuda', args)
            self.assertIn(f'aidotnet-cuda-{model}.json', args)
            self.assertIn('--batch-size 64', args)
        self.assertNotIn('--no-build', calls[0])
        self.assertTrue(all('--no-build' in args for args in calls[1:]))

    def test_model_failure_fails_the_step(self):
        result, calls = self.run_loop('failure')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)

    def test_expired_shared_deadline_does_not_launch_another_host(self):
        result, calls = self.run_loop('expired')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_powershell_steps_parse(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, step in enumerate(STEPS):
                if 'run' in step:
                    Path(folder, f'{index}.ps1').write_text(
                        re.sub(r'\$\{\{.*?\}\}', 'placeholder', step['run']))
            parser = Path(folder, 'parse.ps1')
            parser.write_text('''
$failed = $false
Get-ChildItem $PSScriptRoot -Filter *.ps1 | ForEach-Object {
  $tokens = $null; $errors = $null
  $null = [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$errors)
  if ($errors.Count) { $errors | Out-String | Write-Output; $failed = $true }
}
if ($failed) { exit 1 }
''')
            result = subprocess.run(['pwsh', '-NoProfile', '-File', str(parser)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
