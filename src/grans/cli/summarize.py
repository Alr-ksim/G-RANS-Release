"""Create a compact aggregate report from a full G-RANS evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import _default_summary_path, compact_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", help="Full evaluation JSON")
    parser.add_argument("--output", default=None, help="Compact JSON output path")
    args = parser.parse_args()

    source = Path(args.evaluation)
    payload = json.loads(source.read_text(encoding="utf-8"))
    compact = compact_evaluation(payload)
    output = Path(args.output) if args.output else _default_summary_path(source)
    if output.resolve() == source.resolve():
        raise ValueError("summary output must be different from the full evaluation file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(f"saved compact summary to {output}")


if __name__ == "__main__":
    main()
