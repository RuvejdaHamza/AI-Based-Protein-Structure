"""Command-line interface for 2D HP protein folding optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Sequence

from .evaluation_budget import EvaluationBudget
from .genetic_algorithm import GeneticAlgorithmConfig, GeneticAlgorithmResult, run_genetic_algorithm
from .hybrid_optimizer import HybridConfig, HybridResult, run_hybrid_optimizer
from .sequence_processing import ProcessedSequence, process_sequence
from .simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingResult,
    run_simulated_annealing,
)
from .structure import (
    ProteinStructure,
    StructureConstraints,
    generate_random_valid_structure,
    validate_structure,
)


_ALGORITHM_NAMES = {
    "sa": "simulated_annealing",
    "ga": "genetic_algorithm",
    "hybrid": "hybrid_ga_sa",
}


@dataclass(frozen=True)
class RunRecord:
    algorithm: str
    status: str
    result: SimulatedAnnealingResult | GeneticAlgorithmResult | HybridResult | None
    runtime_seconds: float
    peak_memory_bytes: int
    generated_candidates: int
    valid_candidates: int
    invalid_candidates: int
    energy_evaluations: int
    error: str | None = None


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _negative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed >= 0:
        raise argparse.ArgumentTypeError("must be a negative finite contact energy")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a validated amino-acid sequence with the 2D HP model."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Standard amino-acid sequence, for example ACDEFGHIKLMNPQRSTVWY.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("sa", "ga", "hybrid", "all"),
        default="all",
        help="Optimizer to run (default: all).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=_positive_integer,
        default=200,
        help="Maximum objective evaluations per algorithm (default: 200).",
    )
    parser.add_argument(
        "--time-limit",
        type=_positive_finite_float,
        default=None,
        help="Optional wall-clock limit in seconds for each algorithm.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--hh-energy",
        type=_negative_finite_float,
        default=-1.0,
        help="Energy contribution per H-H contact; must be negative (default: -1.0).",
    )
    parser.add_argument(
        "--constraints",
        type=Path,
        default=None,
        help="Optional JSON file containing hard structural constraints.",
    )
    parser.add_argument(
        "--png",
        nargs="?",
        const=Path("fold.png"),
        type=Path,
        default=None,
        help="Optionally save a structure plot (default path: fold.png).",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def load_constraints(
    path: Path | str | None,
    sequence_length: int,
) -> StructureConstraints | None:
    """Load and validate zero-based structural constraints from JSON."""
    if path is None:
        return None
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise ValueError(f"Cannot read constraints file {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid constraints JSON in {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("Constraints JSON must contain a top-level object.")
    allowed = {"fixed_contacts", "forbidden_positions", "fixed_directions"}
    unknown = set(document) - allowed
    if unknown:
        raise ValueError("Unknown constraint key(s): " + ", ".join(sorted(unknown)))

    fixed_contacts = document.get("fixed_contacts", [])
    forbidden_positions = document.get("forbidden_positions", [])
    fixed_directions = document.get("fixed_directions", {})
    if not isinstance(fixed_contacts, list):
        raise ValueError("fixed_contacts must be a list of residue-index pairs.")
    if not isinstance(forbidden_positions, list):
        raise ValueError("forbidden_positions must be a list of coordinate pairs.")
    if not isinstance(fixed_directions, dict):
        raise ValueError("fixed_directions must be a JSON object of step indices.")

    parsed_directions: dict[int, str] = {}
    for raw_index, direction in fixed_directions.items():
        if not isinstance(raw_index, str) or not raw_index.isdecimal():
            raise ValueError("fixed_directions keys must be non-negative integer strings.")
        parsed_directions[int(raw_index)] = direction

    try:
        constraints = StructureConstraints(
            fixed_contacts=tuple(tuple(pair) for pair in fixed_contacts),
            forbidden_positions=frozenset(
                tuple(position) for position in forbidden_positions
            ),
            fixed_directions=parsed_directions,
        )
        constraints.validate_for_length(sequence_length)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid structural constraints: {error}") from error
    return constraints


def _optimizer_configs(
    sequence_length: int,
    max_evaluations: int,
    seed: int,
) -> tuple[SimulatedAnnealingConfig, GeneticAlgorithmConfig, HybridConfig]:
    population_size = min(20, max_evaluations)
    ga = GeneticAlgorithmConfig(
        population_size=population_size,
        max_generations=max_evaluations,
        tournament_size=min(3, population_size),
        crossover_rate=0.8,
        mutation_rate=1.0 / max(1, sequence_length - 1),
        elitism_count=1,
        max_initial_attempts=100,
        max_offspring_attempts=max(100, max_evaluations * 10),
        seed=seed,
    )
    sa = SimulatedAnnealingConfig(
        initial_temperature=10.0,
        cooling_rate=0.995,
        minimum_temperature=0.0,
        max_iterations=max_evaluations,
        seed=seed,
    )
    hybrid = HybridConfig(
        genetic=ga,
        local_search=sa,
        local_search_frequency=1,
        local_search_fraction=0.2,
        seed=seed,
    )
    return sa, ga, hybrid


def _counts(
    result: SimulatedAnnealingResult | GeneticAlgorithmResult | HybridResult,
) -> tuple[int, int, int, int]:
    if isinstance(result, SimulatedAnnealingResult):
        generated = result.local_move_attempts
        invalid = result.invalid_candidate_structures
        return (
            generated,
            max(0, generated - invalid),
            invalid,
            result.energy_evaluations,
        )
    if isinstance(result, GeneticAlgorithmResult):
        return (
            result.generated_candidates,
            result.valid_candidates,
            result.invalid_candidates,
            result.energy_evaluations,
        )
    generated = result.generated_candidates + result.local_search_move_attempts
    invalid = result.invalid_candidates + result.local_search_invalid_candidates
    valid = result.valid_candidates + result.local_search_move_attempts - result.local_search_invalid_candidates
    return generated, valid, invalid, result.energy_evaluations


def run_algorithm(
    algorithm: str,
    sequence: ProcessedSequence,
    *,
    max_evaluations: int = 200,
    time_limit: float | None = None,
    seed: int = 0,
    hh_energy: float = -1.0,
    constraints: StructureConstraints | None = None,
) -> RunRecord:
    """Run one algorithm with independent budget, deadline, and memory trace."""
    if algorithm not in _ALGORITHM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if not isinstance(sequence, ProcessedSequence):
        raise TypeError("sequence must be a validated ProcessedSequence.")
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer.")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be a positive finite number or None.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if (
        isinstance(hh_energy, bool)
        or not isinstance(hh_energy, (int, float))
        or not math.isfinite(hh_energy)
        or hh_energy >= 0
    ):
        raise ValueError("hh_energy must be a negative finite number.")
    if constraints is not None:
        if not isinstance(constraints, StructureConstraints):
            raise TypeError("constraints must be a StructureConstraints instance or None.")
        constraints.validate_for_length(sequence.length)
    if tracemalloc.is_tracing():
        raise RuntimeError("Cannot measure peak memory while tracemalloc is already active.")

    sa_config, ga_config, hybrid_config = _optimizer_configs(
        sequence.length, max_evaluations, seed
    )
    epsilon = abs(hh_energy)
    budget = EvaluationBudget(max_evaluations)
    tracemalloc.start()
    start = time.perf_counter()
    deadline = None if time_limit is None else start + time_limit
    result = None
    status = "completed"
    error_message = None
    try:
        if algorithm == "sa":
            initial = generate_random_valid_structure(
                sequence.length,
                constraints=constraints,
                deadline=deadline,
                max_restarts=100,
                max_steps_per_restart=max(10_000, sequence.length * 100),
            )
            result = run_simulated_annealing(
                sequence.hp_sequence,
                initial,
                sa_config,
                epsilon=epsilon,
                evaluation_budget=budget,
                constraints=constraints,
                deadline=deadline,
            )
        elif algorithm == "ga":
            result = run_genetic_algorithm(
                sequence.hp_sequence,
                ga_config,
                epsilon=epsilon,
                evaluation_budget=budget,
                constraints=constraints,
                deadline=deadline,
            )
        else:
            result = run_hybrid_optimizer(
                sequence.hp_sequence,
                hybrid_config,
                epsilon=epsilon,
                evaluation_budget=budget,
                constraints=constraints,
                deadline=deadline,
            )
        if result.termination_reason == "time_limit":
            status = "time_limit"
    except TimeoutError as error:
        status = "time_limit"
        error_message = str(error)
    except Exception as error:
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
    runtime = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if result is None:
        return RunRecord(
            algorithm,
            status,
            None,
            runtime,
            peak_memory,
            0,
            0,
            0,
            budget.evaluations,
            error_message,
        )
    generated, valid, invalid, evaluations = _counts(result)
    return RunRecord(
        algorithm,
        status,
        result,
        runtime,
        peak_memory,
        generated,
        valid,
        invalid,
        evaluations,
        error_message,
    )


def _ascii_fold(hp_sequence: str, structure: ProteinStructure) -> str:
    """Render a readable lattice sketch using actual coordinates and bonds."""
    min_x = min(x for x, _ in structure.coordinates)
    max_x = max(x for x, _ in structure.coordinates)
    min_y = min(y for _, y in structure.coordinates)
    max_y = max(y for _, y in structure.coordinates)
    labels = [f"{index}:{residue}" for index, residue in enumerate(hp_sequence, 1)]
    cell_width = max(7, max(map(len, labels), default=1) + 2)
    row_height = 3
    width = (max_x - min_x + 1) * cell_width
    height = (max_y - min_y + 1) * row_height
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    def center(coordinate: tuple[int, int]) -> tuple[int, int]:
        x, y = coordinate
        return (
            (x - min_x) * cell_width + cell_width // 2,
            (max_y - y) * row_height + row_height // 2,
        )

    for first, second in zip(structure.coordinates, structure.coordinates[1:]):
        x1, y1 = center(first)
        x2, y2 = center(second)
        symbol = "-" if y1 == y2 else "|"
        if y1 == y2:
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                canvas[y1][x] = symbol
        else:
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                canvas[y][x1] = symbol

    for index, (coordinate, label) in enumerate(
        zip(structure.coordinates, labels)
    ):
        x, y = center(coordinate)
        start = max(0, x - len(label) // 2)
        canvas[y][start : start + len(label)] = label
    return "\n".join("".join(row).rstrip() for row in canvas[::-1]).rstrip()


def _format_energy(value: float | None) -> str:
    if value is None:
        return "N/A"
    return "0" if value == 0 else f"{value:g}"


def _print_record(
    record: RunRecord,
    sequence: ProcessedSequence,
    hh_energy: float,
    constraints: StructureConstraints | None = None,
    png_path: Path | None = None,
) -> None:
    print(f"\n=== {record.algorithm.upper()} [{record.status}] ===")
    result = record.result
    if result is None:
        print(f"Stopped before producing a valid structure: {record.error or 'time limit reached'}")
    else:
        structure = result.best_structure
        validation = validate_structure(
            structure,
            expected_length=sequence.length,
            constraints=constraints,
        )
        if not validation.valid:
            raise RuntimeError("Optimizer returned an invalid structure: " + "; ".join(validation.issues))
        directions = "".join(direction.value for direction in structure.directions) or "(none)"
        print(f"Directions: {directions}")
        print("Coordinates:")
        for index, coordinate in enumerate(structure.coordinates):
            print(f"  residue {index + 1}: {coordinate}")

        if isinstance(result, SimulatedAnnealingResult):
            final_energy = result.final_energy
            best_energy = result.best_energy
            termination = result.termination_reason
        elif isinstance(result, GeneticAlgorithmResult):
            final_energy = result.best_energy
            best_energy = result.best_energy
            termination = result.termination_reason
        else:
            final_energy = result.final_population_best_energy
            best_energy = result.best_energy
            termination = result.termination_reason
        print(f"Final energy: {_format_energy(final_energy)}")
        print(f"Best energy: {_format_energy(best_energy)}")
        if isinstance(result, SimulatedAnnealingResult):
            print("Displayed structure: best-so-far (final energy is terminal state).")
        print(f"Termination: {termination}")
        print("ASCII fold:")
        print(_ascii_fold(sequence.hp_sequence, structure))
        if png_path is not None:
            _save_png(sequence, structure, abs(hh_energy), png_path)
            print(f"PNG saved: {png_path}")
    print(
        "Candidates: "
        f"generated={record.generated_candidates}, "
        f"valid={record.valid_candidates}, invalid={record.invalid_candidates}"
    )
    print(f"Energy evaluations: {record.energy_evaluations}")
    print(f"Runtime: {record.runtime_seconds:.6f} seconds")
    print(f"Peak traced memory: {record.peak_memory_bytes} bytes")
    if record.error:
        print(f"Note: {record.error}")


def _save_png(
    sequence: ProcessedSequence,
    structure: ProteinStructure,
    epsilon: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt

        from .visualization import plot_protein_structure
    except ImportError as error:
        raise RuntimeError(
            "PNG output requires Matplotlib; install the project's requirements."
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    plot_protein_structure(
        sequence.amino_acid_sequence,
        sequence.hp_sequence,
        structure,
        ax=axis,
        epsilon=epsilon,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _print_comparison(records: list[RunRecord]) -> None:
    print("\n=== Algorithm comparison ===")
    headers = ("Algorithm", "Status", "Final E", "Best E", "Evaluations", "Runtime (s)", "Peak memory")
    rows = []
    for record in records:
        result = record.result
        if isinstance(result, SimulatedAnnealingResult):
            final_energy, best_energy = result.final_energy, result.best_energy
        elif isinstance(result, GeneticAlgorithmResult):
            final_energy = best_energy = result.best_energy
        elif isinstance(result, HybridResult):
            final_energy, best_energy = (
                result.final_population_best_energy,
                result.best_energy,
            )
        else:
            final_energy = best_energy = None
        rows.append(
            (
                record.algorithm,
                record.status,
                _format_energy(final_energy),
                _format_energy(best_energy),
                str(record.energy_evaluations),
                f"{record.runtime_seconds:.4f}",
                str(record.peak_memory_bytes),
            )
        )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        sequence = process_sequence(arguments.sequence)
        constraints = load_constraints(arguments.constraints, sequence.length)
    except (ValueError, OSError) as error:
        build_argument_parser().error(str(error))

    algorithms = (
        tuple(_ALGORITHM_NAMES)
        if arguments.algorithm == "all"
        else (arguments.algorithm,)
    )
    records: list[RunRecord] = []
    for algorithm in algorithms:
        record = run_algorithm(
            algorithm,
            sequence,
            max_evaluations=arguments.max_evaluations,
            time_limit=arguments.time_limit,
            seed=arguments.seed,
            hh_energy=arguments.hh_energy,
            constraints=constraints,
        )
        records.append(record)
        png_path = arguments.png
        if png_path is not None and len(algorithms) > 1:
            png_path = png_path.with_name(
                f"{png_path.stem}_{algorithm}{png_path.suffix or '.png'}"
            )
        _print_record(
            record,
            sequence,
            arguments.hh_energy,
            constraints,
            png_path,
        )

    if arguments.algorithm == "all":
        _print_comparison(records)
    return 1 if any(record.status == "failed" for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
