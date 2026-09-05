#!/usr/bin/env python3
"""Run a GFSS benchmark and append emitted solver records to JSONL.

The benchmark keeps its ordinary console output. Any line beginning with
GFSS_RECORD_JSON= is parsed, minimally validated, enriched with timestamp/git
provenance when missing, and appended as one canonical JSON object per line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import uuid
from typing import Any

PREFIX = "GFSS_RECORD_JSON="


def git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_minimal(record: dict[str, Any]) -> None:
    if record.get("schema") != "gfss.solver_record":
        raise ValueError("schema must be gfss.solver_record")
    version = str(record.get("schema_version", ""))
    if not version.startswith("1."):
        raise ValueError(f"unsupported schema_version {version!r}")
    for key in ("benchmark", "solver", "problem", "correctness", "timing", "memory"):
        if key not in record:
            raise ValueError(f"missing required field {key}")
    if not record["solver"].get("family") or not record["solver"].get("variant"):
        raise ValueError("solver.family and solver.variant are required")
    if not record["problem"].get("name") or int(record["problem"].get("fine_dofs", 0)) <= 0:
        raise ValueError("problem.name and positive problem.fine_dofs are required")
    if float(record["memory"].get("peak_bytes_per_dof", 0.0)) <= 0.0:
        raise ValueError("memory.peak_bytes_per_dof must be positive")


def enrich(record: dict[str, Any], commit: str | None, branch: str | None) -> dict[str, Any]:
    if not record.get("timestamp_utc"):
        record["timestamp_utc"] = (
            dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    if not record.get("run_id"):
        problem = record.get("problem", {}).get("name", "problem")
        variant = record.get("solver", {}).get("variant", "solver")
        stamp = record["timestamp_utc"].replace(":", "").replace("-", "")
        record["run_id"] = f"{stamp}-{problem}-{variant}-{uuid.uuid4().hex[:8]}"
    git = record.setdefault("git", {})
    if not git.get("commit") and commit:
        git["commit"] = commit
    if not git.get("branch") and branch:
        git["branch"] = branch
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GFSS benchmark and append GFSS Solver Record lines to JSONL."
    )
    parser.add_argument(
        "--out",
        default="results/solver_records.jsonl",
        help="append destination (default: results/solver_records.jsonl)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="do not fail when the command emits no solver records",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run; place it after --",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = parse_args()
    commit = git_value(["rev-parse", "HEAD"])
    branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])

    try:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"capture error: {exc}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        print(raw_line, end="")
        line = raw_line.rstrip("\r\n")
        if not line.startswith(PREFIX):
            continue
        try:
            record = json.loads(line[len(PREFIX) :])
            if not isinstance(record, dict):
                raise ValueError("record JSON must be an object")
            validate_minimal(record)
            records.append(enrich(record, commit, branch))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"capture error: invalid solver record: {exc}", file=sys.stderr)
            process.kill()
            process.wait()
            return 3

    return_code = process.wait()
    if return_code != 0:
        return return_code

    if not records:
        message = "capture: command completed but emitted no GFSS Solver Record"
        if args.allow_empty:
            print(message)
            return 0
        print(message, file=sys.stderr)
        return 4

    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")

    print(f"capture: appended {len(records)} record(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
