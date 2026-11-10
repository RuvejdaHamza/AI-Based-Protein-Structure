"""Genetic Algorithm for 2D HP lattice folding, implemented from scratch.

Individuals are relative lattice direction strings. Crossover and mutation may
produce walks with collisions; every offspring is explicitly decoded and
validated. Invalid offspring are rejected (never silently repaired) and
bounded retries are used to refill the population.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import heapq
import random
from time import perf_counter

from .evaluation_budget import EvaluationBudget
from .energy import _calculate_energy_validated
from .search_space import symmetry_key
from .structure import (
    DIRECTION_VECTORS,
    Direction,
    ProteinStructure,
    StructureConstraints,
    StructureError,
    _check_deadline,
    generate_random_valid_structure,
    validate_structure,
)


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 50
    max_generations: int = 200
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elitism_count: int = 1
    max_initial_attempts: int = 100
    max_offspring_attempts: int = 100
    target_energy: float | None = None
    no_improvement_limit: int | None = None
    symmetry_reduction: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "population_size",
            "max_generations",
            "tournament_size",
            "elitism_count",
            "max_initial_attempts",
            "max_offspring_attempts",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "max_generations" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.elitism_count > self.population_size:
            raise ValueError("elitism_count cannot exceed population_size.")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size.")
        for name in ("crossover_rate", "mutation_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite value between 0 and 1.")
        if self.target_energy is not None and (
            isinstance(self.target_energy, bool)
            or not isinstance(self.target_energy, (int, float))
            or not math.isfinite(self.target_energy)
        ):
            raise ValueError("target_energy must be a finite number or None.")
        if self.no_improvement_limit is not None and (
            type(self.no_improvement_limit) is not int
            or self.no_improvement_limit < 1
        ):
            raise ValueError("no_improvement_limit must be a positive integer or None.")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None.")
        if type(self.symmetry_reduction) is not bool:
            raise ValueError("symmetry_reduction must be a boolean.")


@dataclass(frozen=True)
class GeneticAlgorithmResult:
    algorithm: str
    best_individual: tuple[Direction, ...]
    best_structure: ProteinStructure
    best_energy: float
    generations: int
    generated_candidates: int
    valid_candidates: int
    invalid_candidates: int
    early_collision_rejections: int
    coordinate_steps_checked: int
    coordinate_steps_avoided: int
    duplicate_candidates: int
    symmetry_duplicate_candidates: int
    elite_individuals_copied: int
    energy_evaluations: int
    execution_time_seconds: float
    best_energy_history: tuple[float, ...]
    average_energy_history: tuple[float, ...]
    diversity_history: tuple[float, ...]
    energy_evaluations_history: tuple[int, ...]
    termination_reason: str
    seed: int | None


@dataclass
class _Candidate:
    genome: tuple[Direction, ...]
    structure: ProteinStructure
    energy: float


@dataclass
class _Counters:
    generated: int = 0
    valid: int = 0
    invalid: int = 0
    early_collision_rejections: int = 0
    coordinate_steps_checked: int = 0
    duplicates: int = 0
    symmetry_duplicates: int = 0
    evaluations: int = 0


def _decode(
    genome: tuple[Direction, ...],
    counters: _Counters,
    constraints: StructureConstraints | None = None,
) -> ProteinStructure | None:
    """Build a walk and stop immediately when a coordinate is revisited."""
    coordinates = [(0, 0)]
    occupied = {(0, 0)}
    for step_index, direction in enumerate(genome):
        counters.coordinate_steps_checked += 1
        if (
            constraints is not None
            and constraints.fixed_directions.get(step_index) not in (None, direction)
        ):
            return None
        dx, dy = DIRECTION_VECTORS[direction]
        x, y = coordinates[-1]
        coordinate = (x + dx, y + dy)
        if coordinate in occupied:
            counters.early_collision_rejections += 1
            return None
        if (
            constraints is not None
            and coordinate in constraints.forbidden_positions
        ):
            return None
        coordinates.append(coordinate)
        occupied.add(coordinate)
    structure = ProteinStructure(tuple(coordinates))
    if constraints is not None and not validate_structure(
        structure, constraints=constraints
    ).valid:
        return None
    return structure


def _evaluate_genome(
    hp_sequence: str,
    genome: tuple[Direction, ...],
    epsilon: float,
    counters: _Counters,
    seen_keys: dict[object, _Candidate] | None = None,
    symmetry_reduction: bool = True,
    evaluation_budget: EvaluationBudget | None = None,
    constraints: StructureConstraints | None = None,
) -> _Candidate | None:
    counters.generated += 1
    if len(genome) != len(hp_sequence) - 1:
        counters.invalid += 1
        return None
    structure = _decode(genome, counters, constraints)
    if structure is None:
        counters.invalid += 1
        return None
    counters.valid += 1
    if seen_keys is not None:
        key: object = symmetry_key(structure) if symmetry_reduction else genome
        if key in seen_keys:
            counters.duplicates += 1
            if (
                symmetry_reduction
                and seen_keys[key].genome != genome
            ):
                counters.symmetry_duplicates += 1
            return seen_keys[key]
    if evaluation_budget is not None and not evaluation_budget.try_consume():
        return None
    energy = _calculate_energy_validated(hp_sequence, structure, epsilon).energy
    counters.evaluations += 1
    candidate = _Candidate(genome, structure, energy)
    if seen_keys is not None:
        seen_keys[key] = candidate
    return candidate


def _tournament_select(
    population: list[_Candidate],
    tournament_size: int,
    rng: random.Random,
) -> _Candidate:
    contestants = rng.sample(population, tournament_size)
    return min(contestants, key=lambda candidate: candidate.energy)


def _crossover(
    parent_a: tuple[Direction, ...],
    parent_b: tuple[Direction, ...],
    rng: random.Random,
) -> tuple[Direction, ...]:
    """Single-point direction-string crossover; feasibility is checked later."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parent genomes must have equal lengths.")
    if len(parent_a) < 2:
        return parent_a
    cut = rng.randrange(1, len(parent_a))
    return parent_a[:cut] + parent_b[cut:]


def _mutate(
    genome: tuple[Direction, ...],
    mutation_rate: float,
    rng: random.Random,
) -> tuple[Direction, ...]:
    """Independently replace selected directions with one of the four moves."""
    directions = tuple(DIRECTION_VECTORS)
    mutated = list(genome)
    for index in range(len(mutated)):
        if rng.random() < mutation_rate:
            alternatives = [direction for direction in directions if direction != mutated[index]]
            mutated[index] = rng.choice(alternatives)
    return tuple(mutated)


def _make_offspring_genome(
    population: list[_Candidate],
    config: GeneticAlgorithmConfig,
    rng: random.Random,
) -> tuple[Direction, ...]:
    parent_a = _tournament_select(population, config.tournament_size, rng)
    parent_b = _tournament_select(population, config.tournament_size, rng)
    if rng.random() < config.crossover_rate:
        genome = _crossover(parent_a.genome, parent_b.genome, rng)
    else:
        genome = parent_a.genome
    return _mutate(genome, config.mutation_rate, rng)


def _diversity(population: list[_Candidate]) -> float:
    """Fraction of distinct genomes in the current population."""
    return len({candidate.genome for candidate in population}) / len(population)


def run_genetic_algorithm(
    hp_sequence: str,
    config: GeneticAlgorithmConfig | None = None,
    *,
    epsilon: float = 1.0,
    rng: random.Random | None = None,
    evaluation_budget: EvaluationBudget | None = None,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
) -> GeneticAlgorithmResult:
    """Run a minimization GA, rejecting invalid offspring with bounded retries.

    Fitness is represented by energy directly, so tournament selection chooses
    the lowest-energy individual. Elites are copied unchanged. Initial and
    offspring candidates are generated and validated explicitly. Offspring
    attempts that form self-intersections are counted as invalid and discarded;
    if retries cannot fill the next generation, the run stops with a clear
    error rather than inserting an invalid or silently repaired structure.
    """
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    sequence = "".join(hp_sequence.split()).upper()
    if not sequence or set(sequence) - {"H", "P"}:
        raise ValueError("hp_sequence must be non-empty and contain only H and P.")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite number.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number.")
    if constraints is not None:
        if not isinstance(constraints, StructureConstraints):
            raise TypeError("constraints must be a StructureConstraints instance or None.")
        constraints.validate_for_length(len(sequence))

    settings = config if config is not None else GeneticAlgorithmConfig()
    symmetry_reduction = settings.symmetry_reduction and constraints is None
    generator = rng if rng is not None else random.Random(settings.seed)
    counters = _Counters()
    start_time = perf_counter()

    population: list[_Candidate] = []
    initial_seen: dict[object, _Candidate] = {}
    initialization_timed_out = False
    for _ in range(settings.population_size):
        individual_created = False
        for _ in range(settings.max_initial_attempts):
            if deadline is not None and perf_counter() >= deadline:
                initialization_timed_out = True
                break
            try:
                structure = generate_random_valid_structure(
                    len(sequence),
                    rng=generator,
                    max_restarts=1,
                    max_steps_per_restart=max(100, len(sequence) * 100),
                    constraints=constraints,
                    deadline=deadline,
                )
            except TimeoutError:
                initialization_timed_out = True
                break
            except StructureError:
                continue
            candidate = _evaluate_genome(
                sequence,
                structure.directions,
                float(epsilon),
                counters,
                initial_seen,
                symmetry_reduction,
                evaluation_budget,
                constraints,
            )
            if candidate is not None:
                population.append(candidate)
                individual_created = True
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                break
        if initialization_timed_out:
            break
        if not individual_created:
            raise RuntimeError(
                "Could not create the requested initial population after "
                f"{settings.max_initial_attempts} attempts per individual."
            )
        if (
            evaluation_budget is not None
            and evaluation_budget.exhausted
            and len(population) < settings.population_size
        ):
            raise RuntimeError(
                "Evaluation budget was exhausted before the initial population "
                "could be completed."
            )

    if not population:
        if initialization_timed_out:
            raise TimeoutError(
                "Time limit reached before a valid initial individual was generated."
            )
        raise RuntimeError("Could not generate an initial valid population.")

    best = min(population, key=lambda candidate: candidate.energy)
    best_history = [best.energy]
    average_history = [
        sum(candidate.energy for candidate in population) / len(population)
    ]
    diversity_history = [_diversity(population)]
    energy_evaluations_history = [counters.evaluations]
    generations = 0
    generations_without_improvement = 0
    termination_reason = "time_limit" if initialization_timed_out else "maximum_generations"

    if settings.target_energy is not None and best.energy <= settings.target_energy:
        termination_reason = "target_energy"
    elif initialization_timed_out:
        pass
    else:
        while generations < settings.max_generations:
            if deadline is not None and perf_counter() >= deadline:
                termination_reason = "time_limit"
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                termination_reason = "evaluation_budget"
                break
            next_population = heapq.nsmallest(
                settings.elitism_count,
                population,
                key=lambda candidate: candidate.energy,
            )
            generation_seen: dict[object, _Candidate] = {
                (
                    symmetry_key(candidate.structure)
                    if symmetry_reduction
                    else candidate.genome
                ): candidate
                for candidate in next_population
            }
            attempts = 0
            generation_timed_out = False
            while (
                len(next_population) < settings.population_size
                and attempts < settings.max_offspring_attempts
                and not (
                    evaluation_budget is not None
                    and evaluation_budget.exhausted
                )
            ):
                if deadline is not None and perf_counter() >= deadline:
                    generation_timed_out = True
                    break
                attempts += 1
                genome = _make_offspring_genome(population, settings, generator)
                child = _evaluate_genome(
                    sequence,
                    genome,
                    float(epsilon),
                    counters,
                    generation_seen,
                    symmetry_reduction,
                    evaluation_budget,
                    constraints,
                )
                if child is not None:
                    next_population.append(child)

            attempt_limit_reached = (
                len(next_population) != settings.population_size
                and attempts >= settings.max_offspring_attempts
            )
            if len(next_population) != settings.population_size:
                interrupted = generation_timed_out or attempt_limit_reached or (
                    evaluation_budget is not None
                    and evaluation_budget.exhausted
                )
                if interrupted:
                    if generation_timed_out:
                        termination_reason = "time_limit"
                    elif evaluation_budget is not None and evaluation_budget.exhausted:
                        termination_reason = "evaluation_budget"
                    else:
                        termination_reason = "offspring_attempt_limit"
                    next_population.extend(
                        population[: settings.population_size - len(next_population)]
                    )
                else:
                    raise RuntimeError(
                        "Could not fill the next generation with valid offspring: "
                        f"created {len(next_population)} of {settings.population_size} "
                        f"individuals after {settings.max_offspring_attempts} attempts. "
                        "Increase the attempt limit or adjust crossover/mutation rates."
                    )

            population = next_population
            generations += 1
            generation_best = min(population, key=lambda candidate: candidate.energy)
            if generation_best.energy < best.energy:
                best = generation_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            best_history.append(best.energy)
            average_history.append(
                sum(candidate.energy for candidate in population) / len(population)
            )
            diversity_history.append(_diversity(population))
            energy_evaluations_history.append(counters.evaluations)

            if settings.target_energy is not None and best.energy <= settings.target_energy:
                termination_reason = "target_energy"
                break
            if (
                settings.no_improvement_limit is not None
                and generations_without_improvement >= settings.no_improvement_limit
            ):
                termination_reason = "no_improvement_limit"
                break
            if generation_timed_out:
                termination_reason = "time_limit"
                break
            if attempt_limit_reached:
                termination_reason = "offspring_attempt_limit"
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                termination_reason = "evaluation_budget"
                break

    elapsed = perf_counter() - start_time
    return GeneticAlgorithmResult(
        algorithm="genetic_algorithm",
        best_individual=best.genome,
        best_structure=best.structure,
        best_energy=best.energy,
        generations=generations,
        generated_candidates=counters.generated,
        valid_candidates=counters.valid,
        invalid_candidates=counters.invalid,
        early_collision_rejections=counters.early_collision_rejections,
        coordinate_steps_checked=counters.coordinate_steps_checked,
        coordinate_steps_avoided=(
            counters.generated * max(0, len(sequence) - 1)
            - counters.coordinate_steps_checked
        ),
        duplicate_candidates=counters.duplicates,
        symmetry_duplicate_candidates=counters.symmetry_duplicates,
        elite_individuals_copied=generations * settings.elitism_count,
        energy_evaluations=counters.evaluations,
        execution_time_seconds=elapsed,
        best_energy_history=tuple(best_history),
        average_energy_history=tuple(average_history),
        diversity_history=tuple(diversity_history),
        energy_evaluations_history=tuple(energy_evaluations_history),
        termination_reason=termination_reason,
        seed=settings.seed if rng is None else None,
    )
