#!/usr/bin/env python3
"""Plot GFSS upper-left solver Pareto frontiers from Solver Record JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

PREFIX = "GFSS_RECORD_JSON="
GIB = float(1 << 30)


@dataclass
class Point:
    record: dict[str, Any]
    family: str
    variant: str
    problem: str
    target: float
    residual: float
    valid: bool
    time_ms: float
    capacity: float
    bpd: float
    fine_dofs: int
    fme: float | None
    digits: float | None
    pareto: bool = False


def get(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            if text.startswith(PREFIX):
                text = text[len(PREFIX):]
            if not text.startswith("{"):
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(record, dict) and record.get("schema") == "gfss.solver_record":
                records.append(record)
    return records


def make_point(record: dict[str, Any], reuse: int) -> Point | None:
    if str(record.get("schema_version", "")).split(".")[0] != "1":
        return None
    target = finite(get(record, "correctness", "target_relative_residual"))
    residual = finite(get(record, "correctness", "true_relative_residual"))
    setup = finite(get(record, "timing", "setup_ms"))
    solve = finite(get(record, "timing", "solve_ms"))
    bpd = finite(get(record, "memory", "peak_bytes_per_dof"))
    fine_dofs = int(get(record, "problem", "fine_dofs", default=0) or 0)
    if None in (target, residual, setup, solve, bpd) or fine_dofs <= 0:
        return None
    assert target is not None and residual is not None and setup is not None
    assert solve is not None and bpd is not None
    if target <= 0.0 or bpd <= 0.0:
        return None

    capacity = finite(get(record, "memory", "capacity_mdof_per_gib"))
    if capacity is None or capacity <= 0.0:
        capacity = GIB / bpd / 1.0e6

    breakdown = get(record, "correctness", "breakdown")
    valid = bool(get(record, "correctness", "converged", default=False)) and residual <= target and not breakdown

    digits = finite(get(record, "derived", "residual_digits_removed"))
    if digits is None:
        r0 = finite(get(record, "correctness", "initial_true_relative_residual"))
        if r0 is not None and r0 > 0.0 and residual > 0.0:
            digits = max(0.0, -math.log10(residual / r0))

    return Point(
        record=record,
        family=str(get(record, "solver", "family", default="unknown")),
        variant=str(get(record, "solver", "variant", default="unknown")),
        problem=str(get(record, "problem", "name", default="unknown")),
        target=target,
        residual=residual,
        valid=valid,
        time_ms=setup / reuse + solve,
        capacity=capacity,
        bpd=bpd,
        fine_dofs=fine_dofs,
        fme=finite(get(record, "work", "fine_matvec_equivalents")),
        digits=digits,
    )


def mark_frontier(points: list[Point]) -> list[Point]:
    ordered = sorted((p for p in points if p.valid), key=lambda p: (p.time_ms, -p.capacity))
    frontier: list[Point] = []
    best: float | None = None
    for point in ordered:
        if best is None or point.capacity > best + 1.0e-12 * max(1.0, abs(best)):
            point.pareto = True
            frontier.append(point)
            best = point.capacity
    return frontier


def marker_size_map(points: list[Point]) -> tuple[dict[int, float], bool]:
    values = [p.fme for p in points if p.valid and p.fme is not None and p.fme > 0.0]
    if len(values) < 2 or math.isclose(min(values), max(values), rel_tol=1.0e-12):
        return ({id(p): 70.0 for p in points}, False)
    lo = math.log1p(min(values))
    hi = math.log1p(max(values))
    sizes: dict[int, float] = {}
    for p in points:
        if p.fme is None or p.fme <= 0.0:
            sizes[id(p)] = 70.0
        else:
            u = (math.log1p(p.fme) - lo) / (hi - lo)
            sizes[id(p)] = 45.0 + 125.0 * u
    return sizes, True


def export_csv(path: pathlib.Path, points: list[Point], reuse: int) -> None:
    fields = [
        "run_id", "solver_family", "solver_variant", "problem",
        "target_relative_residual", "true_relative_residual", "valid", "pareto",
        "rhs_reuse", "plot_time_ms", "capacity_mdof_per_gib", "peak_bytes_per_dof",
        "fine_dofs", "fine_matvec_equivalents", "residual_digits_removed",
        "git_commit", "benchmark"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in sorted(points, key=lambda q: (q.time_ms, -q.capacity)):
            writer.writerow({
                "run_id": p.record.get("run_id", ""),
                "solver_family": p.family,
                "solver_variant": p.variant,
                "problem": p.problem,
                "target_relative_residual": f"{p.target:.17g}",
                "true_relative_residual": f"{p.residual:.17g}",
                "valid": str(p.valid).lower(),
                "pareto": str(p.pareto).lower(),
                "rhs_reuse": reuse,
                "plot_time_ms": f"{p.time_ms:.17g}",
                "capacity_mdof_per_gib": f"{p.capacity:.17g}",
                "peak_bytes_per_dof": f"{p.bpd:.17g}",
                "fine_dofs": p.fine_dofs,
                "fine_matvec_equivalents": "" if p.fme is None else f"{p.fme:.17g}",
                "residual_digits_removed": "" if p.digits is None else f"{p.digits:.17g}",
                "git_commit": get(p.record, "git", "commit", default="") or "",
                "benchmark": p.record.get("benchmark", ""),
            })


def print_summary(points: list[Point], frontier: list[Point], reuse: int) -> None:
    print(f"records_selected={len(points)} valid={sum(p.valid for p in points)} rhs_reuse={reuse}")
    print("pareto_frontier_upper_left:")
    for p in frontier:
        work = "" if p.fme is None else f" FME={p.fme:.3f}"
        print(
            f"  {p.family}/{p.variant}: time_ms={p.time_ms:.6f} "
            f"capacity_MDOF_per_GiB={p.capacity:.6f} B_per_DOF={p.bpd:.6f} "
            f"residual={p.residual:.6e}{work}"
        )
    if not frontier:
        print("  <none>")


def draw(points: list[Point], frontier: list[Point], output: pathlib.Path, reuse: int,
         annotate: bool, show_invalid: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: python -m pip install matplotlib") from exc

    families = sorted({p.family for p in points})
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(i % 10) for i, family in enumerate(families)}
    sizes, sizes_encode_work = marker_size_map(points)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for family in families:
        group = [p for p in points if p.family == family and p.valid]
        if group:
            ax.scatter(
                [p.time_ms for p in group], [p.capacity for p in group],
                s=[sizes[id(p)] for p in group], alpha=0.82,
                label=family, color=colors[family]
            )
    if show_invalid:
        group = [p for p in points if not p.valid]
        if group:
            ax.scatter([p.time_ms for p in group], [p.capacity for p in group],
                       marker="x", s=55, alpha=0.55, label="invalid / not converged")
    if frontier:
        ordered = sorted(frontier, key=lambda p: p.time_ms)
        ax.plot([p.time_ms for p in ordered], [p.capacity for p in ordered],
                linewidth=1.8, alpha=0.9, label="Pareto frontier")
    if annotate:
        for p in points:
            if p.valid:
                ax.annotate(p.variant, (p.time_ms, p.capacity), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)

    ax.set_title(
        f"GFSS solver Pareto frontier — {points[0].problem}, "
        f"target {points[0].target:.1e}, RHS reuse {reuse}"
    )
    ax.set_xlabel("Time to verified tolerance (ms) — lower is better")
    ax.set_ylabel("Solver-state capacity (MDOF/GiB) — higher is better")
    if sizes_encode_work:
        ax.text(0.99, 0.01, "Marker area encodes fine-matvec-equivalent work",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(f"plot_written={output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot GFSS upper-left solver Pareto frontier.")
    parser.add_argument("input", help="Solver Record JSONL or console log with GFSS_RECORD_JSON lines")
    parser.add_argument("--problem")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--family", action="append")
    parser.add_argument("--reuse", type=int, default=1, help="RHS reuse count for setup amortization")
    parser.add_argument("--output", default="results/pareto.png")
    parser.add_argument("--csv-out")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--show-invalid", action="store_true")
    args = parser.parse_args()
    if args.reuse <= 0:
        parser.error("--reuse must be positive")

    try:
        records = load_records(pathlib.Path(args.input))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    points = [p for r in records if (p := make_point(r, args.reuse)) is not None]
    if args.problem:
        points = [p for p in points if p.problem == args.problem]
    if args.tolerance is not None:
        points = [p for p in points if math.isclose(p.target, args.tolerance, rel_tol=1e-12)]
    if args.family:
        allowed = set(args.family)
        points = [p for p in points if p.family in allowed]
    if not points:
        print("error: no comparable solver records match the filters", file=sys.stderr)
        return 3

    problems = sorted({p.problem for p in points})
    if len(problems) != 1:
        print("error: multiple problems present; use --problem. Available: " + ", ".join(problems), file=sys.stderr)
        return 4
    tolerances = sorted({p.target for p in points})
    if len(tolerances) != 1:
        print("error: multiple target tolerances present; use --tolerance. Available: " +
              ", ".join(f"{v:.17g}" for v in tolerances), file=sys.stderr)
        return 5

    frontier = mark_frontier(points)
    print_summary(points, frontier, args.reuse)
    if args.csv_out:
        export_csv(pathlib.Path(args.csv_out), points, args.reuse)
        print(f"csv_written={args.csv_out}")
    if not args.no_plot:
        try:
            draw(points, frontier, pathlib.Path(args.output), args.reuse, args.annotate, args.show_invalid)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
