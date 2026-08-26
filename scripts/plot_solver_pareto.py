#!/usr/bin/env python3
"""Plot GFSS solver time/capacity Pareto frontiers from canonical JSONL.

Default axes follow the upper-left convention:
  x = time to verified tolerance (ms), lower is better
  y = solver-state capacity (MDOF/GiB), higher is better

The script refuses to mix different problems or target tolerances unless the
user filters them explicitly. It can also export the flattened plotted points
to CSV for external analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Iterable

PREFIX = "GFSS_RECORD_JSON="
GIB = float(1 << 30)


@dataclass
class Point:
    record: dict[str, Any]
    run_id: str
    family: str
    variant: str
    problem: str
    target: float
    final_residual: float
    valid: bool
    x_ms: float
    capacity: float
    bpd: float
    fine_dofs: int
    fme: float | None
    digits: float | None
    pareto: bool = False


def nested(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            if text.startswith(PREFIX):
                text = text[len(PREFIX) :]
            if not text.startswith("{"):
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict) and value.get("schema") == "gfss.solver_record":
                records.append(value)
    return records


def capacity_from_record(record: dict[str, Any], bpd: float) -> float:
    stored = as_float(nested(record, "memory", "capacity_mdof_per_gib"))
    if stored is not None and stored > 0.0:
        return stored
    return GIB / bpd / 1.0e6


def residual_digits(record: dict[str, Any]) -> float | None:
    stored = as_float(nested(record, "derived", "residual_digits_removed"))
    if stored is not None:
        return stored
    r0 = as_float(nested(record, "correctness", "initial_true_relative_residual"))
    r = as_float(nested(record, "correctness", "true_relative_residual"))
    if r0 is None or r is None or r0 <= 0.0 or r <= 0.0:
        return None
    return max(0.0, -math.log10(r / r0))


def point_from_record(record: dict[str, Any], reuse: int) -> Point | None:
    if str(record.get("schema_version", "")).split(".")[0] != "1":
        return None

    family = str(nested(record, "solver", "family", default="unknown"))
    variant = str(nested(record, "solver", "variant", default="unknown"))
    problem = str(nested(record, "problem", "name", default="unknown"))
    fine_dofs = int(nested(record, "problem", "fine_dofs", default=0) or 0)
    target = as_float(nested(record, "correctness", "target_relative_residual"))
    final = as_float(nested(record, "correctness", "true_relative_residual"))
    converged = bool(nested(record, "correctness", "converged", default=False))
    breakdown = nested(record, "correctness", "breakdown")
    setup = as_float(nested(record, "timing", "setup_ms"))
    solve = as_float(nested(record, "timing", "solve_ms"))
    bpd = as_float(nested(record, "memory", "peak_bytes_per_dof"))

    if target is None or final is None or setup is None or solve is None or bpd is None:
        return None
    if target <= 0.0 or bpd <= 0.0 or fine_dofs <= 0:
        return None

    x_ms = setup / float(reuse) + solve
    capacity = capacity_from_record(record, bpd)
    valid = converged and final <= target and not breakdown
    fme = as_float(nested(record, "work", "fine_matvec_equivalents"))

    return Point(
        record=record,
        run_id=str(record.get("run_id", "")),
        family=family,
        variant=variant,
        problem=problem,
        target=target,
        final_residual=final,
        valid=valid,
        x_ms=x_ms,
        capacity=capacity,
        bpd=bpd,
        fine_dofs=fine_dofs,
        fme=fme,
        digits=residual_digits(record),
    )


def same_tolerance(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1.0e-12, abs_tol=0.0)


def mark_upper_left_frontier(points: list[Point]) -> list[Point]:
    valid = [p for p in points if p.valid]
    # At equal x, process highest capacity first. Moving right, a point joins
    # the upper-left frontier only if it exceeds every capacity seen so far.
    ordered = sorted(valid, key=lambda p: (p.x_ms, -p.capacity))
    best_capacity = -math.inf
    frontier: list[Point] = []
    for point in ordered:
        if point.capacity > best_capacity + 1.0e-12 * max(1.0, abs(best_capacity)):
            point.pareto = True
            frontier.append(point)
            best_capacity = point.capacity
    return frontier


def export_csv(path: pathlib.Path, points: list[Point], reuse: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "solver_family",
        "solver_variant",
        "problem",
        "target_relative_residual",
        "true_relative_residual",
        "valid",
        "pareto",
        "rhs_reuse",
        "plot_time_ms",
        "capacity_mdof_per_gib",
        "peak_bytes_per_dof",
        "fine_dofs",
        "fine_matvec_equivalents",
        "residual_digits_removed",
        "git_commit",
        "benchmark",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in sorted(points, key=lambda q: (q.x_ms, -q.capacity)):
            writer.writerow(
                {
                    "run_id": p.run_id,
                    "solver_family": p.family,
                    "solver_variant": p.variant,
                    "problem": p.problem,
                    "target_relative_residual": f"{p.target:.17g}",
                    "true_relative_residual": f"{p.final_residual:.17g}",
                    "valid": str(p.valid).lower(),
                    "pareto": str(p.pareto).lower(),
                    "rhs_reuse": reuse,
                    "plot_time_ms": f"{p.x_ms:.17g}",
                    "capacity_mdof_per_gib": f"{p.capacity:.17g}",
                    "peak_bytes_per_dof": f"{p.bpd:.17g}",
                    "fine_dofs": p.fine_dofs,
                    "fine_matvec_equivalents": "" if p.fme is None else f"{p.fme:.17g}",
                    "residual_digits_removed": "" if p.digits is None else f"{p.digits:.17g}",
                    "git_commit": nested(p.record, "git", "commit", default="") or "",
                    "benchmark": p.record.get("benchmark", ""),
                }
            )


def print_summary(points: list[Point], frontier: list[Point], reuse: int) -> None:
    valid = sum(1 for p in points if p.valid)
    print(f"records_selected={len(points)} valid={valid} rhs_reuse={reuse}")
    print("pareto_frontier_upper_left:")
    if not frontier:
        print("  <none>")
        return
    for point in frontier:
        print(
            f"  {point.family}/{point.variant}: "
            f"time_ms={point.x_ms:.6f} "
            f"capacity_MDOF_per_GiB={point.capacity:.6f} "
            f"B_per_DOF={point.bpd:.6f} "
            f"residual={point.final_residual:.6e}"
        )


def plot(points: list[Point], frontier: list[Point], output: pathlib.Path, reuse: int, annotate: bool, show_invalid: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting; install it with: python -m pip install matplotlib"
        ) from exc

    families = sorted({p.family for p in points})
    cmap = plt.get_cmap("tab10")
    family_color = {family: cmap(i % 10) for i, family in enumerate(families)}

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for family in families:
        group = [p for p in points if p.family == family and p.valid]
        if not group:
            continue
        sizes = []
        fmes = [p.fme for p in group if p.fme is not None and p.fme > 0.0]
        if fmes:
            lo, hi = min(fmes), max(fmes)
            for p in group:
                if p.fme is None or p.fme <= 0.0 or hi <= lo:
                    sizes.append(70.0)
                else:
                    sizes.append(45.0 + 100.0 * (p.fme - lo) / (hi - lo))
        else:
            sizes = [70.0] * len(group)
        ax.scatter(
            [p.x_ms for p in group],
            [p.capacity for p in group],
            s=sizes,
            alpha=0.82,
            label=family,
            color=family_color[family],
        )

    if show_invalid:
        invalid = [p for p in points if not p.valid]
        if invalid:
            ax.scatter(
                [p.x_ms for p in invalid],
                [p.capacity for p in invalid],
                marker="x",
                s=55,
                alpha=0.55,
                label="invalid / not converged",
            )

    if frontier:
        ordered = sorted(frontier, key=lambda p: p.x_ms)
        ax.plot(
            [p.x_ms for p in ordered],
            [p.capacity for p in ordered],
            linewidth=1.8,
            alpha=0.9,
            label="Pareto frontier",
        )

    if annotate:
        for point in points:
            if not point.valid:
                continue
            ax.annotate(
                point.variant,
                (point.x_ms, point.capacity),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    problem = points[0].problem if points else "unknown"
    target = points[0].target if points else math.nan
    ax.set_title(f"GFSS solver Pareto frontier — {problem}, target {target:.1e}, RHS reuse {reuse}")
    ax.set_xlabel("Time to verified tolerance (ms) — lower is better")
    ax.set_ylabel("Solver-state capacity (MDOF/GiB) — higher is better")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(f"plot_written={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GFSS upper-left solver Pareto frontier.")
    parser.add_argument("input", help="canonical JSONL file or text log containing GFSS_RECORD_JSON lines")
    parser.add_argument("--problem", help="select one problem name")
    parser.add_argument("--tolerance", type=float, help="select target true relative residual")
    parser.add_argument("--family", action="append", help="solver family to include; repeatable")
    parser.add_argument("--reuse", type=int, default=1, help="RHS reuse count for setup amortization (default: 1)")
    parser.add_argument("--output", default="results/pareto.png", help="plot path (default: results/pareto.png)")
    parser.add_argument("--csv-out", help="optional flattened CSV export")
    parser.add_argument("--no-plot", action="store_true", help="only print/export the frontier")
    parser.add_argument("--annotate", action="store_true", help="label valid points with solver.variant")
    parser.add_argument("--show-invalid", action="store_true", help="draw failed/not-converged points as crosses")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reuse <= 0:
        print("error: --reuse must be positive", file=sys.stderr)
        return 2

    try:
        records = load_records(pathlib.Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    points = [p for record in records if (p := point_from_record(record, args.reuse)) is not None]

    if args.problem:
        points = [p for p in points if p.problem == args.problem]
    if args.tolerance is not None:
        points = [p for p in points if same_tolerance(p.target, args.tolerance)]
    if args.family:
        allowed = set(args.family)
        points = [p for p in points if p.family in allowed]

    if not points:
        print("error: no solver records match the requested filters", file=sys.stderr)
        return 3

    problems = sorted({p.problem for p in points})
    if len(problems) != 1:
        print(
            "error: records contain multiple problems; use --problem. Available: " + ", ".join(problems),
            file=sys.stderr,
        )
        return 4

    tolerances = sorted({p.target for p in points})
    if len(tolerances) > 1 and args.tolerance is None:
        print(
            "error: records contain multiple target tolerances; use --tolerance. Available: "
            + ", ".join(f"{value:.17g}" for value in tolerances),
            file=sys.stderr,
        )
        return 5

    frontier = mark_upper_left_frontier(points)
    print_summary(points, frontier, args.reuse)

    if args.csv_out:
        export_csv(pathlib.Path(args.csv_out), points, args.reuse)
        print(f"csv_written={args.csv_out}")

    if not args.no_plot:
        try:
            plot(
                points,
                frontier,
                pathlib.Path(args.output),
                args.reuse,
                args.annotate,
                args.show_invalid,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
