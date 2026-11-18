"""Reproducible, budgeted experiments for SA, GA, and the hybrid optimizer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
import tracemalloc
from typing import Iterable

import pandas as pd

from .energy import calculate_energy
from .evaluation_budget import EvaluationBudget
from .genetic_algorithm import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmResult,
    run_genetic_algorithm,
)
from .hybrid_optimizer import HybridConfig, HybridResult, run_hybrid_optimizer
from .sequence_processing import process_sequence
from .simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingResult,
    run_simulated_annealing,
)
from .structure import (
    generate_random_valid_structure,
    validate_structure,
)


PROTEIN_LENGTHS = (8, 12, 16, 20, 30)
BUDGET_STUDY_LENGTHS = (12, 20, 30, 40, 50)
BUDGET_STUDY_BUDGETS = (200, 1_000, 5_000, 20_000)
DEFAULT_ROOT_SEEDS = (
    1101,
    2027,
    3037,
    4051,
    5059,
    6067,
    7079,
    8081,
    9091,
    10007,
)
ALGORITHMS = (
    "simulated_annealing",
    "genetic_algorithm",
    "hybrid_ga_sa",
)

_REAL_AA_PREFIXES = {
    "real_HBB": (
        "P68871",
        "SV2",
        "human hemoglobin subunit beta",
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALG",
    ),
    "real_INS": (
        "P01308",
        "SV1",
        "human insulin precursor",
        "MALWMRLLPLLALLALWGPDPAAAFVNQHL",
    ),
    "real_LYZ": (
        "P61626",
        "SV1",
        "human lysozyme C precursor",
        "MKALIVLGLVLLSVTVQGKVFERCELARTL",
    ),
    "real_CYCS": (
        "P99999",
        "SV2",
        "human cytochrome c",
        "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTG",
    ),
}
_BUDGET_STUDY_REAL_AA_PREFIXES = {
    "real_HBB": (
        "P68871",
        "SV2",
        "human hemoglobin subunit beta",
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLS",
    ),
    "real_INS": (
        "P01308",
        "SV1",
        "human insulin precursor",
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY",
    ),
    "real_LYZ": (
        "P61626",
        "SV1",
        "human lysozyme C precursor",
        "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLA",
    ),
    "real_CYCS": (
        "P99999",
        "SV2",
        "human cytochrome c",
        "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYT",
    ),
}


@dataclass(frozen=True)
class ExperimentCase:
    """One exact amino-acid or synthetic HP input for the experiment."""

    case_id: str
    length: int
    sequence: str
    hp_sequence: str
    sequence_type: str
    source: str
    source_version: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty.")
        if type(self.length) is not int or self.length < 1:
            raise ValueError("length must be a positive integer.")
        if len(self.sequence) != self.length or len(self.hp_sequence) != self.length:
            raise ValueError("sequence and hp_sequence must match length.")
        if set(self.hp_sequence) - {"H", "P"}:
            raise ValueError("hp_sequence must contain only H and P.")
        if self.sequence_type not in {"synthetic_hp", "real_amino_acid"}:
            raise ValueError("Unsupported sequence_type.")


def build_experiment_cases() -> tuple[ExperimentCase, ...]:
    """Build 20 synthetic and 20 UniProt-derived prefix test cases."""
    cases: list[ExperimentCase] = []
    for length in PROTEIN_LENGTHS:
        h_rich = math.ceil(0.7 * length)
        p_rich = math.ceil(0.7 * length)
        half_h = length // 2
        synthetic = {
            "syn_H70": "H" * h_rich + "P" * (length - h_rich),
            "syn_P70": "P" * p_rich + "H" * (length - p_rich),
            "syn_balanced_blocks": "H" * half_h + "P" * (length - half_h),
            "syn_alternating": ("HP" * ((length + 1) // 2))[:length],
        }
        for case_name, hp_sequence in synthetic.items():
            cases.append(
                ExperimentCase(
                    case_id=f"{case_name}_N{length}",
                    length=length,
                    sequence=hp_sequence,
                    hp_sequence=hp_sequence,
                    sequence_type="synthetic_hp",
                    source=case_name,
                )
            )

        for case_name, (
            accession,
            version,
            _,
            aa_prefix,
        ) in _REAL_AA_PREFIXES.items():
            amino_acids = aa_prefix[:length]
            processed = process_sequence(amino_acids)
            cases.append(
                ExperimentCase(
                    case_id=f"{case_name}_N{length}",
                    length=length,
                    sequence=processed.amino_acid_sequence,
                    hp_sequence=processed.hp_sequence,
                    sequence_type="real_amino_acid",
                    source=f"UniProtKB:{accession}",
                    source_version=version,
                )
            )
    return tuple(cases)


def build_budget_study_cases() -> tuple[ExperimentCase, ...]:
    """Build 20 synthetic and 20 real-protein prefix cases for the study."""
    cases: list[ExperimentCase] = []
    for length in BUDGET_STUDY_LENGTHS:
        h_rich = math.ceil(0.7 * length)
        p_rich = math.ceil(0.7 * length)
        half_h = length // 2
        synthetic = {
            "syn_H70": "H" * h_rich + "P" * (length - h_rich),
            "syn_P70": "P" * p_rich + "H" * (length - p_rich),
            "syn_balanced_blocks": "H" * half_h + "P" * (length - half_h),
            "syn_alternating": ("HP" * ((length + 1) // 2))[:length],
        }
        for case_name, hp_sequence in synthetic.items():
            cases.append(
                ExperimentCase(
                    case_id=f"{case_name}_N{length}",
                    length=length,
                    sequence=hp_sequence,
                    hp_sequence=hp_sequence,
                    sequence_type="synthetic_hp",
                    source=case_name,
                )
            )

        for case_name, (
            accession,
            version,
            _,
            aa_prefix,
        ) in _BUDGET_STUDY_REAL_AA_PREFIXES.items():
            amino_acids = aa_prefix[:length]
            processed = process_sequence(amino_acids)
            cases.append(
                ExperimentCase(
                    case_id=f"{case_name}_N{length}",
                    length=length,
                    sequence=processed.amino_acid_sequence,
                    hp_sequence=processed.hp_sequence,
                    sequence_type="real_amino_acid",
                    source=f"UniProtKB:{accession}",
                    source_version=version,
                )
            )
    return tuple(cases)


def derive_seed(
    root_seed: int,
    case_id: str,
    algorithm: str,
    replicate: int,
) -> int:
    """Derive a stable 32-bit seed independent of Python's hash randomization."""
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer.")
    if not case_id or not algorithm or type(replicate) is not int or replicate < 0:
        raise ValueError("case_id, algorithm, and replicate must be valid.")
    payload = f"{root_seed}|{case_id}|{algorithm}|{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _algorithm_config(
    seed: int,
    energy_budget: int,
    population_size: int,
    protein_length: int,
) -> tuple[
    SimulatedAnnealingConfig,
    GeneticAlgorithmConfig,
    HybridConfig,
]:
    sa_config = SimulatedAnnealingConfig(
        initial_temperature=10.0,
        cooling_rate=0.995,
        minimum_temperature=0.0,
        max_iterations=energy_budget,
        no_improvement_limit=None,
        target_energy=None,
        move_types=("end", "corner", "crankshaft"),
        seed=seed,
    )
    ga_config = GeneticAlgorithmConfig(
        population_size=population_size,
        max_generations=energy_budget,
        tournament_size=min(3, population_size),
        crossover_rate=0.8,
        mutation_rate=1.0 / max(1, protein_length - 1),
        elitism_count=1,
        max_initial_attempts=100,
        max_offspring_attempts=max(100, energy_budget * 10),
        target_energy=None,
        no_improvement_limit=None,
        symmetry_reduction=False,
        seed=seed,
    )
    hybrid_config = HybridConfig(
        genetic=ga_config,
        local_search=sa_config,
        local_search_frequency=5,
        local_search_fraction=0.2,
        seed=seed,
    )
    return sa_config, ga_config, hybrid_config


def _history(result: object, name: str) -> list[float]:
    value = getattr(result, name, ())
    return [float(item) for item in value]


def _capture_run_measurements(
    start_wall: float,
    start_cpu: float,
    baseline_memory: int,
) -> tuple[float, float, int]:
    _, peak_memory = tracemalloc.get_traced_memory()
    return (
        time.perf_counter() - start_wall,
        time.process_time() - start_cpu,
        max(0, peak_memory - baseline_memory),
    )


def _convergence_evaluations(
    result: object,
    best_energy: float,
) -> int:
    energy_history = _history(result, "best_energy_history")
    evaluation_history = tuple(
        int(value)
        for value in getattr(result, "energy_evaluations_history", ())
    )
    if not energy_history or len(energy_history) != len(evaluation_history):
        raise RuntimeError(
            "Best-energy and evaluation-count histories must be non-empty "
            "and have matching lengths."
        )
    for energy, evaluations in zip(energy_history, evaluation_history):
        if math.isclose(energy, best_energy, abs_tol=1e-12):
            return evaluations
    raise RuntimeError("The final best energy is missing from its history.")


def _run_one(
    case: ExperimentCase,
    algorithm: str,
    replicate: int,
    root_seed: int,
    energy_budget: int,
    population_size: int,
) -> dict[str, object]:
    seed = derive_seed(root_seed, case.case_id, algorithm, replicate)
    rng = random.Random(seed)
    budget = EvaluationBudget(energy_budget)
    sa_config, ga_config, hybrid_config = _algorithm_config(
        seed, energy_budget, population_size, case.length
    )
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "Stop existing tracemalloc tracing before measuring an individual run."
        )
    tracemalloc.start()
    baseline_memory = 0

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    result: SimulatedAnnealingResult | GeneticAlgorithmResult | HybridResult | None = None
    config: (
        SimulatedAnnealingConfig | GeneticAlgorithmConfig | HybridConfig
    ) = sa_config
    initial_energy: float | None = None
    status = "completed"
    error_type: str | None = None
    error_message: str | None = None
    valid_best: bool | None = None
    final_energy: float | None = None
    best_energy: float | None = None
    runtime_seconds: float | None = None
    cpu_seconds: float | None = None
    peak_memory_bytes: int | None = None
    convergence_evaluations: int | None = None
    convergence: dict[str, object] = {}
    details: dict[str, object] = {}

    try:
        if algorithm == "simulated_annealing":
            config = sa_config
            initial_structure = generate_random_valid_structure(
                case.length, rng=rng
            )
            result = run_simulated_annealing(
                case.hp_sequence,
                initial_structure,
                sa_config,
                rng=rng,
                evaluation_budget=budget,
            )
            runtime_seconds, cpu_seconds, peak_memory_bytes = (
                _capture_run_measurements(
                    start_wall, start_cpu, baseline_memory
                )
            )
            initial_energy = float(result.initial_energy)
            best_energy = float(result.best_energy)
            final_energy = float(result.final_energy)
            valid_best = validate_structure(
                result.best_structure, expected_length=case.length
            ).valid
            convergence = {
                "best_energy_history": _history(result, "best_energy_history"),
                "current_energy_history": _history(
                    result, "current_energy_history"
                ),
                "energy_evaluations_history": list(
                    result.energy_evaluations_history
                ),
            }
            details = {
                "iterations": result.iterations,
                "generated_candidates": result.local_move_attempts,
                "valid_candidates": (
                    result.local_move_attempts
                    - result.invalid_candidate_structures
                ),
                "valid_candidate_structures": (
                    result.valid_candidate_structures
                ),
                "accepted_moves": result.accepted_moves,
                "rejected_moves": result.rejected_moves,
                "local_move_attempts": result.local_move_attempts,
                "duplicate_candidates_avoided": (
                    result.duplicate_candidates_avoided
                ),
                "termination_reason": result.termination_reason,
                "energy_evaluations": result.energy_evaluations,
                "invalid_candidates": result.invalid_candidate_structures,
            }
        elif algorithm == "genetic_algorithm":
            config = ga_config
            result = run_genetic_algorithm(
                case.hp_sequence,
                ga_config,
                rng=rng,
                evaluation_budget=budget,
            )
            runtime_seconds, cpu_seconds, peak_memory_bytes = (
                _capture_run_measurements(
                    start_wall, start_cpu, baseline_memory
                )
            )
            initial_energy = float(result.best_energy_history[0])
            best_energy = float(result.best_energy)
            final_energy = best_energy
            valid_best = validate_structure(
                result.best_structure, expected_length=case.length
            ).valid
            convergence = {
                "best_energy_history": _history(result, "best_energy_history"),
                "average_energy_history": _history(
                    result, "average_energy_history"
                ),
                "diversity_history": _history(result, "diversity_history"),
                "energy_evaluations_history": list(
                    result.energy_evaluations_history
                ),
            }
            details = {
                "generations": result.generations,
                "initial_population_average_energy": float(
                    result.average_energy_history[0]
                ),
                "generated_candidates": result.generated_candidates,
                "early_collision_rejections": result.early_collision_rejections,
                "duplicate_candidates": result.duplicate_candidates,
                "termination_reason": result.termination_reason,
                "energy_evaluations": result.energy_evaluations,
                "valid_candidates": result.valid_candidates,
                "invalid_candidates": result.invalid_candidates,
            }
        elif algorithm == "hybrid_ga_sa":
            config = hybrid_config
            result = run_hybrid_optimizer(
                case.hp_sequence,
                hybrid_config,
                rng=rng,
                evaluation_budget=budget,
            )
            runtime_seconds, cpu_seconds, peak_memory_bytes = (
                _capture_run_measurements(
                    start_wall, start_cpu, baseline_memory
                )
            )
            initial_energy = float(result.initial_population_best_energy)
            best_energy = float(result.best_energy)
            final_energy = float(result.final_population_best_energy)
            valid_best = validate_structure(
                result.best_structure, expected_length=case.length
            ).valid
            convergence = {
                "best_energy_history": _history(result, "best_energy_history"),
                "average_energy_history": _history(
                    result, "average_energy_history"
                ),
                "pre_local_search_best_history": _history(
                    result, "pre_local_search_best_history"
                ),
                "diversity_history": _history(result, "diversity_history"),
                "energy_evaluations_history": list(
                    result.energy_evaluations_history
                ),
            }
            details = {
                "generations": result.generations,
                "initial_population_average_energy": (
                    result.initial_population_average_energy
                ),
                "final_population_average_energy": (
                    result.final_population_average_energy
                ),
                "generated_candidates": (
                    result.generated_candidates
                    + result.local_search_move_attempts
                ),
                "valid_candidate_structures": (
                    result.valid_candidates
                    + result.local_search_valid_candidates
                ),
                "local_search_calls": result.local_search_calls,
                "local_search_move_attempts": (
                    result.local_search_move_attempts
                ),
                "local_search_replacements": result.local_search_replacements,
                "genetic_energy_evaluations": (
                    result.genetic_energy_evaluations
                ),
                "local_search_energy_evaluations": (
                    result.local_search_energy_evaluations
                ),
                "termination_reason": result.termination_reason,
                "energy_evaluations": result.energy_evaluations,
                "valid_candidates": (
                    result.valid_candidates
                    + result.local_search_move_attempts
                    - result.local_search_invalid_candidates
                ),
                "invalid_candidates": (
                    result.invalid_candidates
                    + result.local_search_invalid_candidates
                ),
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        if best_energy is None:
            raise RuntimeError("Optimizer did not return a best energy.")
        convergence_evaluations = _convergence_evaluations(result, best_energy)
        if not valid_best:
            raise RuntimeError("Optimizer returned an invalid best structure.")
        independent_energy = calculate_energy(
            case.hp_sequence, result.best_structure
        )
        if not math.isclose(
            independent_energy.energy, best_energy, abs_tol=1e-12
        ):
            raise RuntimeError(
                "Reported best energy does not match independent recomputation."
            )
        details["independent_contact_count"] = independent_energy.contact_count
        details["best_directions"] = [
            direction.value for direction in result.best_structure.directions
        ]
    except Exception as error:
        status = "failed"
        error_type = type(error).__name__
        error_message = str(error)
        valid_best = False
        if runtime_seconds is None:
            runtime_seconds, cpu_seconds, peak_memory_bytes = (
                _capture_run_measurements(
                    start_wall, start_cpu, baseline_memory
                )
            )
    tracemalloc.stop()

    result_evaluations = (
        int(getattr(result, "energy_evaluations", budget.evaluations))
        if result is not None
        else budget.evaluations
    )
    valid_candidates = details.get("valid_candidates")
    invalid_candidates = details.get("invalid_candidates")
    return {
        "case_id": case.case_id,
        "sequence_type": case.sequence_type,
        "source": case.source,
        "source_version": case.source_version,
        "protein_length": case.length,
        "sequence": case.sequence,
        "hp_sequence": case.hp_sequence,
        "hydrophobic_count": case.hp_sequence.count("H"),
        "polar_count": case.hp_sequence.count("P"),
        "algorithm": algorithm,
        "replicate": replicate,
        "root_seed": root_seed,
        "random_seed": seed,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "best_energy": best_energy,
        "runtime_seconds": runtime_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "generated_candidates": details.get("generated_candidates"),
        "valid_best_structure": valid_best,
        "valid_candidates": valid_candidates,
        "invalid_candidates": invalid_candidates,
        "energy_evaluations": result_evaluations,
        "convergence_evaluations_to_best": convergence_evaluations,
        "energy_budget_limit": energy_budget,
        "energy_budget_used": budget.evaluations,
        "convergence_json": json.dumps(convergence, separators=(",", ":")),
        "details_json": json.dumps(details, separators=(",", ":")),
        "configuration_json": json.dumps(asdict(config), default=str, sort_keys=True),
    }


def summarize_results(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level outcomes by case and algorithm without hiding failures."""
    if runs.empty:
        return pd.DataFrame(
            columns=(
                "case_id",
                "algorithm",
                "runs",
                "successful_runs",
                "failed_runs",
            )
        )
    grouped = runs.groupby(["case_id", "algorithm"], dropna=False)
    summary = grouped.agg(
        runs=("status", "size"),
        successful_runs=("status", lambda values: int((values == "completed").sum())),
        failed_runs=("status", lambda values: int((values != "completed").sum())),
        best_energy_mean=("best_energy", "mean"),
        best_energy_median=("best_energy", "median"),
        best_energy_std=("best_energy", "std"),
        final_energy_mean=("final_energy", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        peak_memory_bytes_mean=("peak_memory_bytes", "mean"),
        generated_candidates_mean=("generated_candidates", "mean"),
        valid_candidates_mean=("valid_candidates", "mean"),
        invalid_candidates_mean=("invalid_candidates", "mean"),
        energy_evaluations_mean=("energy_evaluations", "mean"),
        convergence_evaluations_to_best_mean=(
            "convergence_evaluations_to_best",
            "mean",
        ),
    ).reset_index()
    summary["success_rate"] = summary["successful_runs"] / summary["runs"]
    return summary


def run_experiments(
    cases: Iterable[ExperimentCase] | None = None,
    *,
    run_count: int = 10,
    root_seeds: Iterable[int] = DEFAULT_ROOT_SEEDS,
    energy_budget: int = 200,
    population_size: int = 20,
    output_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all algorithms and export run-level and summary CSV files.

    The common budget counts objective evaluations, including initial scoring.
    For the hybrid, GA and nested SA share one `EvaluationBudget`. Runtime and
    invalid-candidate counts are also recorded because equal objective-call
    counts do not imply identical Python CPU work.
    """
    if type(run_count) is not int or run_count < 1:
        raise ValueError("run_count must be a positive integer.")
    seeds = tuple(root_seeds)
    if len(seeds) != run_count:
        raise ValueError("Provide exactly one root seed per independent run.")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("All root seeds must be non-negative integers.")
    if type(population_size) is not int or population_size < 2:
        raise ValueError("population_size must be an integer of at least two.")
    if type(energy_budget) is not int or energy_budget < population_size:
        raise ValueError(
            "energy_budget must be at least population_size so GA and hybrid "
            "can score their initial populations."
        )
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "Stop existing tracemalloc tracing before running experiments."
        )

    experiment_cases = tuple(cases if cases is not None else build_experiment_cases())
    if not experiment_cases:
        raise ValueError("At least one experiment case is required.")

    records: list[dict[str, object]] = []
    for case in experiment_cases:
        for replicate, root_seed in enumerate(seeds, start=1):
            for algorithm in ALGORITHMS:
                records.append(
                    _run_one(
                        case,
                        algorithm,
                        replicate,
                        root_seed,
                        energy_budget,
                        population_size,
                    )
                )

    runs = pd.DataFrame.from_records(records)
    summary = summarize_results(runs)
    export_results(runs, summary, output_dir)
    return runs, summary


def export_results(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path = "results",
) -> tuple[Path, Path]:
    """Write all individual runs and grouped summaries as UTF-8 CSV files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs_path = destination / "experiment_runs.csv"
    summary_path = destination / "experiment_summary.csv"
    runs.to_csv(runs_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    return runs_path, summary_path


def run_budget_sensitivity_study(
    cases: Iterable[ExperimentCase] | None = None,
    *,
    budgets: Iterable[int] | None = None,
    run_count: int | None = None,
    root_seeds: Iterable[int] | None = None,
    population_size: int | None = None,
    output_dir: str | Path | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the budget-sensitivity matrix and save measured CSVs and plots.

    Full defaults are 40 cases, four objective-evaluation budgets, ten seeded
    replicates, and three algorithms (4,800 runs). Smoke mode uses one
    synthetic and one real case at length 12, budgets 20 and 40, one replicate,
    and a six-member population unless explicitly overridden.
    """
    if type(smoke_test) is not bool:
        raise ValueError("smoke_test must be a boolean.")
    study_budgets = tuple(
        budgets
        if budgets is not None
        else ((20, 40) if smoke_test else BUDGET_STUDY_BUDGETS)
    )
    if not study_budgets or any(
        type(budget) is not int or budget < 1 for budget in study_budgets
    ):
        raise ValueError("budgets must contain positive integers.")
    if len(set(study_budgets)) != len(study_budgets):
        raise ValueError("budgets cannot contain duplicates.")

    if cases is None:
        if smoke_test:
            all_cases = build_budget_study_cases()
            experiment_cases = tuple(
                case
                for case in all_cases
                if case.length == 12
                and case.case_id in {"syn_H70_N12", "real_HBB_N12"}
            )
        else:
            experiment_cases = build_budget_study_cases()
    else:
        experiment_cases = tuple(cases)
    if not experiment_cases:
        raise ValueError("At least one budget-study case is required.")

    replicates = run_count if run_count is not None else (1 if smoke_test else 10)
    if type(replicates) is not int or replicates < 1:
        raise ValueError("run_count must be a positive integer.")
    if root_seeds is None:
        if replicates > len(DEFAULT_ROOT_SEEDS):
            raise ValueError(
                f"At most {len(DEFAULT_ROOT_SEEDS)} default root seeds are available."
            )
        seeds = DEFAULT_ROOT_SEEDS[:replicates]
    else:
        seeds = tuple(root_seeds)
    if len(seeds) != replicates:
        raise ValueError("Provide exactly one root seed per independent run.")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("All root seeds must be non-negative integers.")

    min_budget = min(study_budgets)
    population = (
        population_size
        if population_size is not None
        else (min(6, min_budget) if smoke_test else 20)
    )
    if type(population) is not int or population < 2:
        raise ValueError("population_size must be an integer of at least two.")
    if min_budget < population:
        raise ValueError(
            "Every study budget must be at least population_size so GA and "
            "hybrid can score their initial populations."
        )
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "Stop existing tracemalloc tracing before running experiments."
        )

    destination = Path(
        output_dir
        if output_dir is not None
        else (
            "results/budget_study/smoke_test"
            if smoke_test
            else "results/budget_study"
        )
    )
    records: list[dict[str, object]] = []
    for case in experiment_cases:
        for budget in study_budgets:
            for replicate, root_seed in enumerate(seeds, start=1):
                for algorithm in ALGORITHMS:
                    record = _run_one(
                        case,
                        algorithm,
                        replicate,
                        root_seed,
                        budget,
                        population,
                    )
                    record["budget"] = budget
                    record["study_mode"] = "smoke_test" if smoke_test else "full"
                    records.append(record)

    runs = pd.DataFrame.from_records(records)
    destination.mkdir(parents=True, exist_ok=True)
    runs.to_csv(destination / "budget_study_runs.csv", index=False, encoding="utf-8")

    from .experiment_analysis import analyze_budget_sensitivity_results

    analysis = analyze_budget_sensitivity_results(runs, destination)
    return runs, analysis.budget_summary


def main(argv: list[str] | None = None) -> None:
    """Run the primary experiment by default or the optional budget study."""
    parser = argparse.ArgumentParser(
        description="Run protein-folding experiments with reproducible budgets."
    )
    parser.add_argument(
        "--budget-study",
        action="store_true",
        help="Run the separate budget-sensitivity study.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small synthetic/real budget-study matrix for validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for run CSVs and analysis plots.",
    )
    arguments = parser.parse_args(argv)
    if arguments.smoke_test and not arguments.budget_study:
        parser.error("--smoke-test requires --budget-study.")

    if arguments.budget_study:
        runs, summary = run_budget_sensitivity_study(
            output_dir=arguments.output_dir,
            smoke_test=arguments.smoke_test,
        )
        mode = "smoke test" if arguments.smoke_test else "full budget study"
        print(
            f"Recorded {len(runs)} {mode} runs in "
            f"{arguments.output_dir or 'results/budget_study'}"
        )
        print(summary.to_string(index=False))
        return

    runs, summary = run_experiments()
    print(f"Recorded {len(runs)} runs in results/experiment_runs.csv")
    print("Summary saved to results/experiment_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
