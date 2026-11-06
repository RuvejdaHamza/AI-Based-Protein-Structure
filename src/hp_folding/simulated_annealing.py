"""Simulated Annealing for the 2D HP lattice protein-folding model.

Each iteration builds the valid one-move neighborhood, selects one candidate
uniformly, and applies the Metropolis acceptance rule. For N residues, a
neighborhood has O(N) proposals; validating each proposal costs O(N), so an
iteration costs O(N^2) in this straightforward implementation. Energy
evaluation is O(N). The total time for I iterations is O(I*N^2), and working
space is O(N^2) in the worst case because the neighborhood is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from time import perf_counter

from .evaluation_budget import EvaluationBudget
from .search_space import IncrementalEnergyTracker
from .local_moves import MoveStatistics, generate_neighborhood
from .structure import (
    ProteinStructure,
    StructureConstraints,
    _check_deadline,
    validate_structure,
)


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    initial_temperature: float = 10.0
    cooling_rate: float = 0.995
    minimum_temperature: float = 1e-3
    max_iterations: int = 10_000
    no_improvement_limit: int | None = None
    target_energy: float | None = None
    move_types: tuple[str, ...] = ("end", "corner", "crankshaft")
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_temperature, bool)
            or not isinstance(self.initial_temperature, (int, float))
            or not math.isfinite(self.initial_temperature)
            or self.initial_temperature <= 0
        ):
            raise ValueError("initial_temperature must be a positive finite number.")
        if (
            isinstance(self.cooling_rate, bool)
            or not isinstance(self.cooling_rate, (int, float))
            or not math.isfinite(self.cooling_rate)
            or not 0 < self.cooling_rate < 1
        ):
            raise ValueError("cooling_rate must be finite and strictly between 0 and 1.")
        if (
            isinstance(self.minimum_temperature, bool)
            or not isinstance(self.minimum_temperature, (int, float))
            or not math.isfinite(self.minimum_temperature)
            or self.minimum_temperature < 0
        ):
            raise ValueError("minimum_temperature must be a non-negative finite number.")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("max_iterations must be a non-negative integer.")
        if self.no_improvement_limit is not None and (
            type(self.no_improvement_limit) is not int
            or self.no_improvement_limit < 1
        ):
            raise ValueError("no_improvement_limit must be a positive integer or None.")
        if self.target_energy is not None and (
            isinstance(self.target_energy, bool)
            or not isinstance(self.target_energy, (int, float))
            or not math.isfinite(self.target_energy)
        ):
            raise ValueError("target_energy must be a finite number or None.")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None.")
        if (
            not isinstance(self.move_types, tuple)
            or not self.move_types
            or set(self.move_types) - {"end", "corner", "crankshaft"}
        ):
            raise ValueError(
                "move_types must be a non-empty tuple of end, corner, and/or crankshaft."
            )


@dataclass(frozen=True)
class SimulatedAnnealingResult:
    algorithm: str
    best_structure: ProteinStructure
    best_energy: float
    initial_energy: float
    final_energy: float
    iterations: int
    valid_candidate_structures: int
    invalid_candidate_structures: int
    accepted_moves: int
    rejected_moves: int
    energy_evaluations: int
    local_move_attempts: int
    duplicate_candidates_avoided: int
    restricted_operator_opportunities: int
    incremental_neighbor_checks: int
    full_scan_neighbor_checks_estimate: int
    execution_time_seconds: float
    best_energy_history: tuple[float, ...]
    current_energy_history: tuple[float, ...]
    energy_evaluations_history: tuple[int, ...]
    termination_reason: str
    seed: int | None


def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Return the Metropolis probability for a proposed energy difference."""
    if not math.isfinite(delta_energy):
        raise ValueError("delta_energy must be finite.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number.")
    if delta_energy <= 0:
        return 1.0
    return math.exp(-delta_energy / temperature)


def run_simulated_annealing(
    hp_sequence: str,
    initial_structure: ProteinStructure,
    config: SimulatedAnnealingConfig | None = None,
    *,
    epsilon: float = 1.0,
    rng: random.Random | None = None,
    evaluation_budget: EvaluationBudget | None = None,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
) -> SimulatedAnnealingResult:
    """Optimize a valid initial structure with a geometric cooling schedule.

    The temperature used at iteration ``i`` is
    ``initial_temperature * cooling_rate ** i``. A candidate is selected
    uniformly from the valid one-move neighborhood. Invalid local proposals
    are counted by the move statistics and never scored or silently repaired.

    `valid_candidate_structures` counts distinct valid neighbors generated.
    `invalid_candidate_structures` counts invalid move proposals encountered
    while generating those neighborhoods. Accepted/rejected move counts refer
    only to valid candidates evaluated by the Metropolis rule.
    """
    settings = config if config is not None else SimulatedAnnealingConfig()
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    normalized_hp_sequence = "".join(hp_sequence.split())
    initial_validation = validate_structure(
        initial_structure,
        expected_length=len(normalized_hp_sequence),
        constraints=constraints,
    )
    if not initial_validation.valid:
        raise ValueError(
            "initial_structure must be valid and match hp_sequence length: "
            + "; ".join(initial_validation.issues)
        )

    if evaluation_budget is not None and not evaluation_budget.try_consume():
        raise ValueError("Evaluation budget is exhausted before initial scoring.")
    energy_tracker = IncrementalEnergyTracker(hp_sequence, initial_structure, epsilon)
    initial_energy = energy_tracker.energy
    generator = rng if rng is not None else random.Random(settings.seed)
    current_structure = initial_structure
    current_energy = initial_energy
    best_structure = initial_structure
    best_energy = initial_energy
    current_history = [current_energy]
    best_history = [best_energy]
    iterations = 0
    valid_candidates = 0
    invalid_candidates = 0
    accepted_moves = 0
    rejected_moves = 0
    energy_evaluations = 1
    energy_evaluations_history = [energy_evaluations]
    consecutive_without_improvement = 0
    move_statistics = MoveStatistics()
    start_time = perf_counter()
    termination_reason = (
        "time_limit"
        if deadline is not None and start_time >= deadline
        else "maximum_iterations"
    )

    if settings.target_energy is not None and best_energy <= settings.target_energy:
        termination_reason = "target_energy"
    elif termination_reason == "time_limit":
        pass
    else:
        while iterations < settings.max_iterations:
            if deadline is not None and perf_counter() >= deadline:
                termination_reason = "time_limit"
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                termination_reason = "evaluation_budget"
                break
            temperature = settings.initial_temperature * (
                settings.cooling_rate**iterations
            )
            if temperature <= settings.minimum_temperature:
                termination_reason = "minimum_temperature"
                break

            rejected_before = move_statistics.rejected
            accepted_before = move_statistics.accepted
            duplicates_before = move_statistics.duplicate_candidates
            try:
                neighborhood = generate_neighborhood(
                    current_structure,
                    move_statistics,
                    move_types=settings.move_types,
                    constraints=constraints,
                    deadline=deadline,
                )
            except TimeoutError:
                invalid_candidates += move_statistics.rejected - rejected_before
                valid_candidates += max(
                    0,
                    move_statistics.accepted
                    - accepted_before
                    - move_statistics.duplicate_candidates
                    + duplicates_before,
                )
                termination_reason = "time_limit"
                break
            invalid_candidates += move_statistics.rejected - rejected_before
            valid_candidates += len(neighborhood)
            if deadline is not None and perf_counter() >= deadline:
                iterations += 1
                current_history.append(current_energy)
                best_history.append(best_energy)
                energy_evaluations_history.append(energy_evaluations)
                termination_reason = "time_limit"
                break
            iterations += 1

            if not neighborhood:
                termination_reason = "no_valid_neighbors"
                current_history.append(current_energy)
                best_history.append(best_energy)
                energy_evaluations_history.append(energy_evaluations)
                break

            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                termination_reason = "evaluation_budget"
                break
            candidate = generator.choice(neighborhood)
            if (
                evaluation_budget is not None
                and not evaluation_budget.try_consume()
            ):
                termination_reason = "evaluation_budget"
                break
            energy_delta = energy_tracker.evaluate_validated(candidate)
            candidate_energy = energy_delta.energy
            energy_evaluations += 1
            delta_energy = energy_delta.delta_energy

            probability = acceptance_probability(delta_energy, temperature)

            if delta_energy <= 0 or generator.random() < probability:
                energy_tracker.commit(candidate, energy_delta)
                current_structure = candidate
                current_energy = candidate_energy
                accepted_moves += 1
            else:
                rejected_moves += 1

            if current_energy < best_energy:
                best_structure = current_structure
                best_energy = current_energy
                consecutive_without_improvement = 0
            else:
                consecutive_without_improvement += 1

            current_history.append(current_energy)
            best_history.append(best_energy)
            energy_evaluations_history.append(energy_evaluations)

            if settings.target_energy is not None and best_energy <= settings.target_energy:
                termination_reason = "target_energy"
                break
            if (
                settings.no_improvement_limit is not None
                and consecutive_without_improvement >= settings.no_improvement_limit
            ):
                termination_reason = "no_improvement_limit"
                break

    elapsed = perf_counter() - start_time
    return SimulatedAnnealingResult(
        algorithm="simulated_annealing",
        best_structure=best_structure,
        best_energy=best_energy,
        initial_energy=initial_energy,
        final_energy=current_energy,
        iterations=iterations,
        valid_candidate_structures=valid_candidates,
        invalid_candidate_structures=invalid_candidates,
        accepted_moves=accepted_moves,
        rejected_moves=rejected_moves,
        energy_evaluations=energy_evaluations,
        local_move_attempts=int(move_statistics.snapshot()["attempted"]),
        duplicate_candidates_avoided=int(move_statistics.snapshot()["duplicate_candidates"]),
        restricted_operator_opportunities=int(
            move_statistics.snapshot()["restricted_operator_opportunities"]
        ),
        incremental_neighbor_checks=energy_tracker.neighbor_checks,
        full_scan_neighbor_checks_estimate=(
            4 * normalized_hp_sequence.upper().count("H") * energy_evaluations
        ),
        execution_time_seconds=elapsed,
        best_energy_history=tuple(best_history),
        current_energy_history=tuple(current_history),
        energy_evaluations_history=tuple(energy_evaluations_history),
        termination_reason=termination_reason,
        seed=settings.seed if rng is None else None,
    )
