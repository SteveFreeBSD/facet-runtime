"""Command-line entry point for Facet."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from facet_runtime import models
from facet_runtime.benchmark import CASES, run_benchmark
from facet_runtime.errors import FacetRuntimeError
from facet_runtime.image_pipeline import inspect_image
from facet_runtime.runtime import BACKENDS, run_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facet")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run one prompt on one compute backend"
    )
    run_parser.add_argument("prompt")
    run_parser.add_argument(
        "--backend",
        choices=(*BACKENDS, "auto"),
        default="auto",
        help="compute backend; auto prefers the measured-fastest available device",
    )
    image_parser = commands.add_parser(
        "inspect-image", help="transcribe one image independently on NPU and GPU"
    )
    image_parser.add_argument("image")
    commands.add_parser(
        "models", help="show the model assigned to each backend and why"
    )
    bench_parser = commands.add_parser(
        "bench", help="measure the assigned model on each backend"
    )
    bench_parser.add_argument(
        "--backend",
        default=",".join(BACKENDS),
        help="comma-separated backends to measure",
    )
    bench_parser.add_argument(
        "--repeat", type=int, default=2, help="iterations per backend and case"
    )
    bench_parser.add_argument(
        "--case",
        default=",".join(case.name for case in CASES),
        help="comma-separated benchmark cases",
    )
    return parser


def _selected_cases(names: str) -> tuple:
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    known = {case.name: case for case in CASES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"unknown benchmark case: {', '.join(unknown)}")
    return tuple(known[name] for name in wanted)


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            payload = run_prompt(args.prompt, args.backend).to_dict()
        elif args.command == "inspect-image":
            payload = inspect_image(args.image).to_dict()
        elif args.command == "models":
            payload = {"assignments": models.report()}
        else:
            backends = [
                name.strip() for name in args.backend.split(",") if name.strip()
            ]
            payload = run_benchmark(
                backends, cases=_selected_cases(args.case), repeat=args.repeat
            ).to_dict()
    except (FacetRuntimeError, FileNotFoundError, ValueError) as error:
        print(
            json.dumps({"error": str(error), "command": args.command}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
