#!/usr/bin/env python3
"""Serial IX CT-single comparison on one explicitly selected physical GPU.

This is performance screening only. Use run_tests.py for the accuracy gate.
Environment overrides are fixed at process startup; each trial gets a fresh
process so native and filesystem kernel caches cannot cross policy settings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--shapes', default='2048,1048576')
    p.add_argument('--apis', default='c2c,c2r,r2c')
    p.add_argument('--variants', default='baseline,default')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--directions', default='forward,inverse')
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--iters', type=int, default=300)
    p.add_argument('--variant-config', action='append', default=[], metavar='NAME=JSON',
                   help='Named startup environment overrides; baseline/default remain available')
    a = p.parse_args()
    configs = {'baseline': {'FLAGFFT_IX_CT_SINGLE': '0'},
               'default': {'FLAGFFT_IX_CT_SINGLE': '1'}}
    for spec in a.variant_config:
        name, value = spec.split('=', 1)
        override = json.loads(value)
        if not isinstance(override, dict) or any(
            not (k.startswith('FLAGFFT_IX_') or k == 'FLAGFFT_PACKED_REAL')
            or not isinstance(v, str) for k, v in override.items()
        ):
            p.error('Variant overrides must be string-valued IX or PACKED_REAL settings')
        configs[name] = {'FLAGFFT_IX_CT_SINGLE': '1', **override}
    variants = a.variants.split(',')
    if any(v not in configs for v in variants):
        p.error('Unknown variant; provide --variant-config NAME=JSON')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    rows = []
    for repeat in range(a.repeats):
        # Alternate order across repetitions to expose temporal drift.
        for variant in variants if repeat % 2 == 0 else variants[::-1]:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('FLAGFFT_IX_') and k != 'FLAGFFT_PACKED_REAL'}
            env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), IX_VISIBLE_DEVICES=str(a.gpu),
                       FLAGFFT_TUNE_DISABLE='1', **configs[variant])
            env['PYTHONPATH'] = str(root / 'python') + os.pathsep + env.get('PYTHONPATH', '')
            for n in a.shapes.split(','):
                for api in a.apis.split(','):
                    directions = a.directions.split(',') if api == 'c2c' else ['inverse' if api == 'c2r' else 'forward']
                    for direction in directions:
                        name = f'{variant}_{n}_{api}_{direction}_{repeat}'
                        cmd = [a.binary, 'bench', '--api', api, '--rank', '1', '--shape', n,
                               '--batch', '1', '--direction', direction, '--warmup', str(a.warmup),
                               '--iters', str(a.iters), '--json', '--print-path']
                        row = dict(variant=variant, n=int(n), api=api, direction=direction,
                                   repeat=repeat, git_commit=commit, command=cmd, gpu=a.gpu,
                                   overrides=configs[variant])
                        proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, start_new_session=True)
                        try:
                            stdout, stderr = proc.communicate(timeout=a.timeout)
                            (out / f'{name}.json').write_text(stdout)
                            (out / f'{name}.err').write_text(stderr)
                            row['exit_code'] = proc.returncode
                            if proc.returncode == 0:
                                row.update(json.loads(stdout)['cases'][0]['timing'])
                        except subprocess.TimeoutExpired:
                            # Stop only this trial's process group, including its JIT child.
                            os.killpg(proc.pid, signal.SIGKILL)
                            stdout, stderr = proc.communicate()
                            (out / f'{name}.json').write_text(stdout)
                            (out / f'{name}.err').write_text(stderr)
                            row['timeout'] = a.timeout
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        (out / 'sweep.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
