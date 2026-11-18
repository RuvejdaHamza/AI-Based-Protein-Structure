"""Reproducible baseline-versus-optimized benchmarks for the HP implementation.

The reference paths intentionally use full validation, per-candidate coordinate
maps, and the original allocate/validate local-neighborhood pattern. The
optimized paths are the production implementations. Run with:

    python benchmarks/benchmark_performance.py
"""

from __future__ import annotations

import random
import statistics
import tracemalloc
from time import perf_counter

import hp_folding.energy as energy
import hp_folding.local_moves as local_moves
from hp_folding.energy import calculate_energy
from hp_folding.genetic_algorithm import (
    GeneticAlgorithmConfig,
    _Candidate,
    _Counters,
    _evaluate_genome,
    run_genetic_algorithm,
)
from hp_folding.search_space import symmetry_key
from hp_folding.simulated_annealing import (
    SimulatedAnnealingConfig,
    acceptance_probability,
    run_simulated_annealing,
)
from hp_folding.structure import (
    DIRECTION_VECTORS,
    Direction,
    ProteinStructure,
    generate_random_valid_structure,
    validate_structure,
)


def _baseline_neighborhood(
    structure: ProteinStructure,
) -> tuple[ProteinStructure, ...]:
    """Reference neighborhood using public operators and full validation."""
    occupied = set(structure.coordinates)
    candidates: dict[tuple[tuple[int, int], ...], ProteinStructure] = {}
    source_key = symmetry_key(structure)

    def retain(candidate: ProteinStructure | None) -> None:
        if candidate is None:
            return
        key = symmetry_key(candidate)
        if key != source_key and key not in candidates:
            candidates[key] = candidate

    if structure.length >= 2:
        for terminal, anchor_index in (
            (0, 1),
            (structure.length - 1, structure.length - 2),
        ):
            x, y = structure.coordinates[anchor_index]
            for dx, dy in DIRECTION_VECTORS.values():
                retain(local_moves.end_move(structure, terminal, (x + dx, y + dy)))

    for index in range(1, structure.length - 1):
        retain(local_moves.corner_move(structure, index))

    for start in range(1, structure.length - 2):
        pair = (structure.coordinates[start], structure.coordinates[start + 1])
        first_anchor = structure.coordinates[start - 1]
        second_anchor = structure.coordinates[start + 2]
        moving = set(pair)
        fixed = occupied - moving
        valid_proposals = set(local_moves._crankshaft_proposals(structure, start))
        first_neighbors = tuple(
            (first_anchor[0] + dx, first_anchor[1] + dy)
            for dx, dy in DIRECTION_VECTORS.values()
        )
        second_neighbors = tuple(
            (second_anchor[0] + dx, second_anchor[1] + dy)
            for dx, dy in DIRECTION_VECTORS.values()
        )
        for first in first_neighbors:
            for second in second_neighbors:
                if (
                    first == second
                    or abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
                    or (first, second) == pair
                    or first in fixed
                    or second in fixed
                    or (first, second) not in valid_proposals
                ):
                    continue
                coordinates = list(structure.coordinates)
                coordinates[start] = first
                coordinates[start + 1] = second
                retain(
                    local_moves._candidate_or_reject(
                        coordinates, "crankshaft", statistics=None
                    )
                )
    return tuple(candidates.values())


def _baseline_score_genomes(
    hp_sequence: str,
    genomes: tuple[tuple[Direction, ...], ...],
    epsilon: float = 1.0,
) -> tuple[list[_Candidate], int, int]:
    """Reference GA scoring: decode, validate, then call validated public energy."""
    candidates: list[_Candidate] = []
    valid = invalid = 0
    for genome in genomes:
        structure = ProteinStructure.from_directions(genome)
        validation = validate_structure(structure, expected_length=len(hp_sequence))
        if not validation.valid:
            invalid += 1
            continue
        valid += 1
        energy_value = calculate_energy(hp_sequence, structure, epsilon).energy
        candidates.append(_Candidate(genome, structure, energy_value))
    return candidates, valid, invalid


def _optimized_score_genomes(
    hp_sequence: str,
    genomes: tuple[tuple[Direction, ...], ...],
    epsilon: float = 1.0,
) -> tuple[list[_Candidate], _Counters]:
    counters = _Counters()
    candidates = [
        candidate
        for genome in genomes
        if (
            candidate := _evaluate_genome(
                hp_sequence, genome, epsilon, counters
            )
        )
        is not None
    ]
    return candidates, counters


def _baseline_sa(
    hp_sequence: str,
    initial: ProteinStructure,
    config: SimulatedAnnealingConfig,
    seed: int,
    epsilon: float = 1.0,
) -> tuple[float, int]:
    """Reference SA path with full neighborhood and energy recomputation."""
    rng = random.Random(seed)
    current = initial
    current_energy = calculate_energy(hp_sequence, current, epsilon).energy
    best_energy = current_energy
    evaluations = 1
    no_improvement = 0
    for iteration in range(config.max_iterations):
        temperature = config.initial_temperature * config.cooling_rate**iteration
        if temperature <= config.minimum_temperature:
            break
        neighborhood = _baseline_neighborhood(current)
        if not neighborhood:
            break
        candidate = rng.choice(neighborhood)
        new_energy = calculate_energy(hp_sequence, candidate, epsilon).energy
        evaluations += 1
        delta = new_energy - current_energy
        if delta <= 0 or rng.random() < acceptance_probability(delta, temperature):
            current, current_energy = candidate, new_energy
        if current_energy < best_energy:
            best_energy = current_energy
            no_improvement = 0
        else:
            no_improvement += 1
        if (
            config.no_improvement_limit is not None
            and no_improvement >= config.no_improvement_limit
        ):
            break
    return best_energy, evaluations


def _measure(function, repeats: int = 5) -> tuple[float, int]:
    times: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        times.append(perf_counter() - start)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(times), peak


def run_benchmarks() -> None:
    sequence = "HPPHHPHHPPHH"
    seed_set = (11, 23, 37, 41, 59)
    structure = generate_random_valid_structure(
        48, rng=random.Random(921), max_restarts=30
    )

    # Use a same-length HP sequence for a fair energy comparison.
    hp_sequence = "".join("H" if index % 3 else "P" for index in range(structure.length))
    baseline_energy = lambda: calculate_energy(hp_sequence, structure)
    optimized_energy = lambda: energy._calculate_energy_validated(
        hp_sequence, structure, 1.0
    )

    genomes = tuple(
        tuple(random.Random(seed).choice(tuple(Direction)) for _ in range(len(hp_sequence) - 1))
        for seed in range(250)
    )

    print("Component benchmarks (median runtime over 5 runs; peak Python allocation)")
    for label, fn in (
        ("energy/reference validated", baseline_energy),
        ("energy/optimized trusted-valid", optimized_energy),
        ("neighborhood/reference", lambda: _baseline_neighborhood(structure)),
        ("neighborhood/optimized", lambda: local_moves.generate_neighborhood(structure)),
        (
            "GA scoring/reference",
            lambda: _baseline_score_genomes(hp_sequence, genomes),
        ),
        (
            "GA scoring/optimized",
            lambda: _optimized_score_genomes(hp_sequence, genomes),
        ),
    ):
        elapsed, peak = _measure(fn)
        print(f"{label}: {elapsed:.6f}s, {peak} bytes peak")

    baseline_population, valid, invalid = _baseline_score_genomes(
        hp_sequence, genomes
    )
    optimized_population, optimized_counters = _optimized_score_genomes(
        hp_sequence, genomes
    )
    print(
        "GA scoring parity:",
        f"baseline_valid={valid}",
        f"optimized_valid={optimized_counters.valid}",
        f"baseline_invalid={invalid}",
        f"optimized_invalid={optimized_counters.invalid}",
        f"baseline_energy_evaluations={valid}",
        f"optimized_energy_evaluations={optimized_counters.evaluations}",
        f"same_best_energy="
        f"{min(item.energy for item in baseline_population) == min(item.energy for item in optimized_population)}",
    )

    baseline_neighbors = _baseline_neighborhood(structure)
    optimized_neighbors = local_moves.generate_neighborhood(structure)
    baseline_keys = {symmetry_key(candidate) for candidate in baseline_neighbors}
    optimized_keys = {symmetry_key(candidate) for candidate in optimized_neighbors}
    print(
        "Neighborhood parity:",
        f"baseline={len(baseline_keys)}",
        f"optimized={len(optimized_keys)}",
        f"same_states={baseline_keys == optimized_keys}",
    )

    ga_config = GeneticAlgorithmConfig(
        population_size=18,
        max_generations=14,
        tournament_size=3,
        elitism_count=1,
        max_offspring_attempts=1500,
    )
    ga_rows: list[tuple[float, float, int, int]] = []
    for seed in seed_set:
        result = run_genetic_algorithm(
            sequence,
            GeneticAlgorithmConfig(
                **{**ga_config.__dict__, "seed": seed, "symmetry_reduction": False}
            ),
        )
        ga_rows.append(
            (
                result.best_energy,
                result.execution_time_seconds,
                result.energy_evaluations,
                result.generated_candidates,
            )
        )
    print(
        "GA optimized:",
        f"mean_best_energy={statistics.mean(row[0] for row in ga_rows):.3f}",
        f"best_energy={min(row[0] for row in ga_rows):.3f}",
        f"mean_runtime={statistics.mean(row[1] for row in ga_rows):.6f}s",
        f"mean_energy_evaluations={statistics.mean(row[2] for row in ga_rows):.1f}",
        f"mean_generated={statistics.mean(row[3] for row in ga_rows):.1f}",
    )

    sa_config = SimulatedAnnealingConfig(
        initial_temperature=2.0,
        cooling_rate=0.97,
        minimum_temperature=0.01,
        max_iterations=35,
    )
    reference_quality: list[float] = []
    optimized_quality: list[float] = []
    reference_evaluations: list[int] = []
    optimized_evaluations: list[int] = []
    reference_times: list[float] = []
    optimized_times: list[float] = []
    for seed in seed_set:
        initial = generate_random_valid_structure(
            len(sequence), rng=random.Random(seed)
        )
        start = perf_counter()
        baseline_best, baseline_evals = _baseline_sa(
            sequence, initial, sa_config, seed
        )
        reference_times.append(perf_counter() - start)
        result = run_simulated_annealing(
            sequence,
            initial,
            SimulatedAnnealingConfig(**{**sa_config.__dict__, "seed": seed}),
        )
        reference_quality.append(baseline_best)
        optimized_quality.append(result.best_energy)
        reference_evaluations.append(baseline_evals)
        optimized_evaluations.append(result.energy_evaluations)
        optimized_times.append(result.execution_time_seconds)

    print(
        "SA reference:",
        f"mean_best_energy={statistics.mean(reference_quality):.3f}",
        f"best_energy={min(reference_quality):.3f}",
        f"mean_runtime={statistics.mean(reference_times):.6f}s",
        f"mean_energy_evaluations={statistics.mean(reference_evaluations):.1f}",
    )
    print(
        "SA optimized:",
        f"mean_best_energy={statistics.mean(optimized_quality):.3f}",
        f"best_energy={min(optimized_quality):.3f}",
        f"mean_runtime={statistics.mean(optimized_times):.6f}s",
        f"mean_energy_evaluations={statistics.mean(optimized_evaluations):.1f}",
    )


if __name__ == "__main__":
    run_benchmarks()
