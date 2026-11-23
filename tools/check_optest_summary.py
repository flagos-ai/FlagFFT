# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check coverage and pytest verdicts in a FlagFFT OpTest result directory."""

import argparse
import json
from pathlib import Path

import yaml

OPERATORS = Path(__file__).resolve().parents[1] / "conf" / "operators.yaml"


def _ids(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def check_summary(summary, expected_ops):
    results = summary["result"]
    errors = []
    if missing := expected_ops - results.keys():
        errors.append(f"Missing operators: {', '.join(sorted(missing))}")
    if unexpected := results.keys() - expected_ops:
        errors.append(f"Unexpected operators: {', '.join(sorted(unexpected))}")
    for op, phases in results.items():
        for phase in ("accuracy", "performance"):
            result = phases.get(phase, {})
            status = result.get("status")
            exit_code = result.get("exit_code")
            acceptable = status == "Passed"
            if phase == "performance" and status == "Skipped":
                # The only policy skips emitted by the benchmark are explicit
                # SiP reference limits. A pytest setup error still fails CI.
                acceptable = "SiP" in (result.get("reason") or "")
            if not acceptable or exit_code != 0:
                errors.append(f"{op} {phase}: {status or 'missing'} (pytest exit {exit_code})")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--ops", default="all")
    parser.add_argument("--exclude-ops", default="")
    args = parser.parse_args()
    with OPERATORS.open(encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)["ops"]
    if args.ops == "all":
        expected = {op["id"] for op in catalog if any(
            stage in op["stages"][-1] for stage in ("alpha", "beta", "stable")
        )}
    else:
        expected = _ids(args.ops)
    expected -= _ids(args.exclude_ops)
    if not expected:
        parser.error("expected at least one operator")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    errors = check_summary(summary, expected)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"OpTest checks passed for {len(expected)} FlagFFT operators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
