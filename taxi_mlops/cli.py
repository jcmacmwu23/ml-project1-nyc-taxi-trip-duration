"""Command line entry points for the local lifecycle."""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from .batch import score
from .data import ingest
from .models import evaluate, get_decision, held_out, release, train
from .monitor import monitor


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("--root", type=Path, default=Path(".local"), help="local lifecycle state directory")
    commands = main.add_subparsers(dest="command", required=True)

    command = commands.add_parser("ingest")
    command.add_argument("--offline", action="store_true")

    command = commands.add_parser("train")
    command.add_argument("snapshot")
    command.add_argument("--weak", action="store_true", help="train a deliberately bad demo candidate")

    command = commands.add_parser("evaluate")
    command.add_argument("snapshot")
    command.add_argument("--candidate", type=int, required=True)
    command.add_argument("--baseline", type=int, required=True)

    command = commands.add_parser("test-report")
    command.add_argument("decision")

    for action in ("bootstrap", "promote", "rollback"):
        command = commands.add_parser(action)
        if action == "promote":
            command.add_argument("decision")
        else:
            command.add_argument("version", type=int)
        command.add_argument("--reason", required=True)

    command = commands.add_parser("score")
    command.add_argument("snapshot")
    command.add_argument("date")
    command.add_argument("--run-id")
    command.add_argument("--fail", choices=("after_stage", "after_commit"))

    command = commands.add_parser("backfill")
    command.add_argument("snapshot")
    command.add_argument("start")
    command.add_argument("end", help="inclusive")
    command.add_argument("--version", type=int, required=True)
    command.add_argument("--run-id", required=True)

    command = commands.add_parser("monitor")
    command.add_argument("date")
    command.add_argument("--as-of", required=True)
    command.add_argument("--decision", required=True)
    return main


def dispatch(args):
    root = args.root.resolve()
    command = args.command
    if command == "ingest":
        return {"snapshot": ingest(root, args.offline)}
    if command == "train":
        return train(root, args.snapshot, args.weak)
    if command == "evaluate":
        return evaluate(root, args.snapshot, args.candidate, args.baseline)
    if command == "test-report":
        return held_out(root, args.decision)
    if command in {"bootstrap", "promote", "rollback"}:
        return release(root, command, getattr(args, "version", None), getattr(args, "decision", None), args.reason)
    if command == "score":
        return score(root, args.snapshot, args.date, args.run_id, fail=args.fail)
    if command == "backfill":
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if start > end:
            raise ValueError("Backfill start is after end")
        return [
            score(root, args.snapshot, (start + timedelta(days=offset)).isoformat(),
                  args.run_id, args.version, replace=True)
            for offset in range((end - start).days + 1)
        ]
    return monitor(root, args.date, args.as_of, get_decision(root, args.decision))


def main():
    args = parser().parse_args()
    try:
        print(json.dumps(dispatch(args), indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
