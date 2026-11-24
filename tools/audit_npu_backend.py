#!/usr/bin/env python3
"""Verify that FlagFFT owns only Triton NPU kernels and SiP stays a test oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

REFERENCE_SOURCE = Path('src/adaptor/backend/npu/test_adaptor.cpp')
PATTERNS = {
    'Ascend C kernel syntax': r'__aicore__|__gm__|kernel_operator\.h|AscendC::',
    'Ascend C build or launch': r'ascendc_library\s*\(|launch_ascendc_',
    'production SiP execution': r'AsdSip::|asdFftExec|sip_plan_execute',
}


def scan_source(root):
    root = Path(root)
    issues = []
    paths = [root/'CMakeLists.txt'] + list((root/'src').rglob('*'))
    checked = 0
    for p in paths:
        if not p.is_file() or (p.suffix not in ('.cpp','.hpp','.h','.cmake') and p.name!='CMakeLists.txt'):
            continue
        rel = p.relative_to(root)
        if rel == REFERENCE_SOURCE:
            continue
        checked += 1
        if p.name.startswith('ascendc_'):
            issues.append(dict(file=str(rel),reason='retained Ascend C implementation file'))
        for reason, pattern in PATTERNS.items():
            for match in re.finditer(pattern,p.read_text()):
                issues.append(dict(file=str(rel),line=p.read_text().count('\n',0,match.start())+1,reason=reason))
    return dict(passed=not issues,checked_files=checked,issues=issues,reference_allowlist=[str(REFERENCE_SOURCE)])


def audit(root,build):
    report = scan_source(root)
    build = Path(build)
    cache = (build/'CMakeCache.txt').read_text()
    commands = json.loads((build/'compile_commands.json').read_text())
    bad = [row['file'] for row in commands if 'ascendc_' in Path(row['file']).name]
    defines = [row['file'] for row in commands if '-DFLAGFFT_NPU_ENABLE_SIP_EXECUTION=1' in row.get('command','')]
    lib = build/'libflagfft.so'
    symbols = subprocess.check_output(['nm','-DC','--undefined-only',str(lib)],text=True)
    forbidden = [line for line in symbols.splitlines() if re.search(r'launch_ascendc_|AsdSip::|asdFftExec',line)]
    dynamic = subprocess.check_output(['readelf','-d',str(lib)],text=True)
    needed = re.findall(r'NEEDED.*?\[(.*?)\]',dynamic)
    report.update(ascendc_compilation_units=bad,production_sip_defines=defines,forbidden_symbols=forbidden,
                  direct_dependencies=needed,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                  sip_option_disabled='FLAGFFT_NPU_ENABLE_SIP_EXECUTION:BOOL=OFF' in cache)
    report['passed'] = (report['passed'] and not bad and not defines and not forbidden
                        and report['sip_option_disabled'] and not any('asdsip' in s for s in needed))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    report = audit(args.source_dir,args.build_dir)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: Triton-only FlagFFT NPU execution; SiP reference isolated' if report['passed'] else 'FAIL: forbidden implementation or dependency found')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
