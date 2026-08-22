"""Run a reproducible Krea-2 offload backend benchmark matrix.

The matrix is JSON so each case can supply its own launcher.  This matters for
AIMDO: it must initialize before the inference script imports PyTorch.

Example matrix::

  {
    "cases": {
      "stream": ["uv", "run", "python", "krea2/inference.py", "--offload", ...],
      "nvme-python": ["uv", "run", "python", "krea2/inference.py", "--offload", "--offload-nvme", "8", "--offload-nvme-path", "F:/nvme/k2.bin", ...],
      "nvme-aimdo": ["ramtorch-aimdo", "--devices", "0", "krea2/inference.py", "--offload", "--offload-nvme", "8", "--offload-nvme-path", "F:/nvme/k2.bin", "--offload-nvme-io", "aimdo", ...]
    }
  }

Each command must include a fixed prompt, seed, resolution, and sampling-step
count.  Process-level timings deliberately include checkpoint/model setup and
therefore measure the end-to-end cold-start cost users actually see.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def _run(command: list[str]) -> dict:
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True)
    return {
        "seconds": time.perf_counter() - started,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="JSON matrix file")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded process runs per case (default 1)")
    parser.add_argument("--runs", type=int, default=5,
                        help="measured process runs per case (default 5)")
    parser.add_argument("--output", type=Path, default=Path("profiles/offload_matrix.json"))
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cases = matrix.get("cases")
    if not isinstance(cases, dict) or not cases:
        parser.error("matrix must contain a non-empty object at 'cases'")

    report = {"matrix": str(args.matrix), "warmup": args.warmup, "runs": args.runs, "cases": {}}
    for name, command in cases.items():
        if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
            parser.error(f"case {name!r} must be a list of command strings")
        print(f"\n[bench] {name}: {' '.join(command)}")
        for _ in range(args.warmup):
            result = _run(command)
            if result["returncode"]:
                raise RuntimeError(f"warmup failed for {name}:\n{result['stderr']}")
        runs = []
        for index in range(args.runs):
            result = _run(command)
            if result["returncode"]:
                raise RuntimeError(f"run {index + 1} failed for {name}:\n{result['stderr']}")
            runs.append(result)
            print(f"[bench] {name} run {index + 1}/{args.runs}: {result['seconds']:.3f}s")
        seconds = [result["seconds"] for result in runs]
        report["cases"][name] = {
            "median_seconds": statistics.median(seconds),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
            "runs": runs,
        }
        print(f"[bench] {name} median: {report['cases'][name]['median_seconds']:.3f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[bench] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
