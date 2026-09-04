"""Command-line entry point for Facet."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from facet_runtime.errors import FacetRuntimeError
from facet_runtime.image_pipeline import inspect_image
from facet_runtime.runtime import run_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facet")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run one prompt on one compute backend"
    )
    run_parser.add_argument("prompt")
    run_parser.add_argument(
        "--backend",
        choices=("cpu", "gpu", "npu", "auto"),
        default="auto",
        help="compute backend; auto currently uses a temporary fixed order",
    )
    image_parser = commands.add_parser(
        "inspect-image", help="transcribe one image independently on NPU and GPU"
    )
    image_parser.add_argument("image")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = run_prompt(args.prompt, args.backend)
        else:
            result = inspect_image(args.image)
    except (FacetRuntimeError, FileNotFoundError, ValueError) as error:
        print(
            json.dumps({"error": str(error), "command": args.command}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
