"""Hybrid Genetic Algorithm + Simulated Annealing for 2D HP folding.

The GA supplies global population exploration. At a configurable generation
frequency, the best fraction of the offspring population is refined with the
existing Simulated Annealing implementation and replaced only when refinement
improves energy. Every individual remains a validated self-avoiding walk.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from time import perf_counter

from .evaluation_budget import EvaluationBudget
from .genetic_algorithm import (
    GeneticAlgorithmConfig,
    _Candidate,
    _Counters,
    _diversity,
    _evaluate_genome,
    _make_offspring_genome,
)
from .search_space import symmetry_key
from .simulated_annealing import (
    SimulatedAnnealingConfig,
    run_simulated_annealing,
)
from .structure import (
    Direction,
    ProteinStructure,
    StructureConstraints,
    StructureError,
    generate_random_valid_structure,
)


@dataclass(frozen=True)
class HybridConfig:
    genetic: GeneticAlgorithmConfig = GeneticAlgorithmConfig()
    local_search: SimulatedAnnealingConfig = SimulatedAnnealingConfig(
        max_iterations=100,
        initial_temperature=2.0,
        cooling_rate=0.95,
        minimum_temperature=0.01,
    )
    local_search_frequency: int = 1
    local_search_fraction: float = 0.2
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.genetic, GeneticAlgorithmConfig):
            raise TypeError("genetic must be a GeneticAlgorithmConfig.")
        if not isinstance(self.local_search, SimulatedAnnealingConfig):
            raise TypeError("local_search must be a SimulatedAnnealingConfig.")
        if (
            type(self.local_search_frequency) is not int
            or self.local_search_frequency < 1
        ):
            raise ValueError("local_search_frequency must be a positive integer.")
        if (
            isinstance(self.local_search_fraction, bool)
            or not isinstance(self.local_search_fraction, (int, float))
            or not math.isfinite(self.local_search_fraction)
            or not 0 <= self.local_search_fraction <= 1
        ):
            raise ValueError("local_search_fraction must be between 0 and 1.")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None.")


@dataclass(frozen=True)
class HybridResult:
    """Run statistics; candidate counters are split between GA and SA work.

    `generated_candidates`, `valid_candidates`, and `invalid_candidates`
    describe GA genomes. SA neighborhood candidates and invalid proposals are
    reported in their dedicated local-search fields. Energy evaluations count
    both parts of the hybrid.
    """

    algorithm: str
    best_individual: tuple[Direction, ...]
    best_structure: ProteinStructure
    best_energy: float
    initial_population_best_energy: float
    initial_population_average_energy: float
    final_population_best_energy: float
    final_population_average_energy: float
    improvement_vs_initial_best: float
    generations: int
    local_search_calls: int
    local_search_replacements: int
    local_search_frequency: int
    local_search_fraction: float
    generated_candidates: int
    local_search_move_attempts: int
    valid_candidates: int
    invalid_candidates: int
    early_collision_rejections: int
    coordinate_steps_checked: int
    coordinate_steps_avoided: int
    duplicate_candidates: int
    symmetry_duplicate_candidates: int
    elite_individuals_copied: int
    energy_evaluations: int
    genetic_energy_evaluations: int
    local_search_energy_evaluations: int
    local_search_candidate_structures: int
    local_search_valid_candidates: int
    local_search_invalid_candidates: int
    local_search_duplicate_replacements_avoided: int
    local_search_execution_time_seconds: float
    execution_time_seconds: float
    best_energy_history: tuple[float, ...]
    average_energy_history: tuple[float, ...]
    pre_local_search_best_history: tuple[float, ...]
    diversity_history: tuple[float, ...]
    energy_evaluations_history: tuple[int, ...]
    termination_reason: str
    seed: int | None


def _validate_hp_sequence(hp_sequence: str) -> str:
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    sequence = "".join(hp_sequence.split()).upper()
    if not sequence or set(sequence) - {"H", "P"}:
        raise ValueError("hp_sequence must be non-empty and contain only H and P.")
    return sequence


def _initial_population(
    hp_sequence: str,
    config: GeneticAlgorithmConfig,
    epsilon: float,
    rng: random.Random,
    counters: _Counters,
    evaluation_budget: EvaluationBudget | None = None,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
    symmetry_reduction: bool = True,
) -> tuple[list[_Candidate], bool]:
    population: list[_Candidate] = []
    initial_seen: dict[object, _Candidate] = {}
    timed_out = False
    for _ in range(config.population_size):
        for _ in range(config.max_initial_attempts):
            if deadline is not None and perf_counter() >= deadline:
                timed_out = True
                break
            try:
                structure = generate_random_valid_structure(
                    len(hp_sequence),
                    rng=rng,
                    max_restarts=1,
                    max_steps_per_restart=max(100, len(hp_sequence) * 100),
                    constraints=constraints,
                    deadline=deadline,
                )
            except TimeoutError:
                timed_out = True
                break
            except StructureError:
                continue
            individual = _evaluate_genome(
                hp_sequence,
                structure.directions,
                epsilon,
                counters,
                initial_seen,
                symmetry_reduction,
                evaluation_budget,
                constraints,
            )
            if individual is not None:
                population.append(individual)
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                break
        else:
            raise RuntimeError(
                "Could not create the requested initial population after "
                f"{config.max_initial_attempts} attempts per individual."
            )
        if (
            evaluation_budget is not None
            and evaluation_budget.exhausted
            and len(population) < config.population_size
        ):
            raise RuntimeError(
                "Evaluation budget was exhausted before the initial population "
                "could be completed."
            )
        if timed_out:
            break
    if not population and timed_out:
        raise TimeoutError(
            "Time limit reached before a valid initial individual was generated."
        )
    return population, timed_out


def run_hybrid_optimizer(
    hp_sequence: str,
    config: HybridConfig | None = None,
    *,
    epsilon: float = 1.0,
    rng: random.Random | None = None,
    evaluation_budget: EvaluationBudget | None = None,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
) -> HybridResult:
    """Run GA generations with periodic SA refinement of promising individuals.

    The baseline is the initial valid GA population. Per generation, the
    pre-refinement best energy is recorded; after local search, improved SA
    results replace their corresponding individuals. Elitism is enforced by
    carrying the best-so-far individual into each next generation and restoring
    it if necessary.
    """
    sequence = _validate_hp_sequence(hp_sequence)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite number.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number.")
    if constraints is not None:
        if not isinstance(constraints, StructureConstraints):
            raise TypeError("constraints must be a StructureConstraints instance or None.")
        constraints.validate_for_length(len(sequence))
    settings = config if config is not None else HybridConfig()
    ga_config = settings.genetic
    symmetry_reduction = ga_config.symmetry_reduction and constraints is None
    generator = rng if rng is not None else random.Random(
        settings.seed if settings.seed is not None else ga_config.seed
    )
    counters = _Counters()
    start_time = perf_counter()

    population, initialization_timed_out = _initial_population(
        sequence,
        ga_config,
        float(epsilon),
        generator,
        counters,
        evaluation_budget,
        constraints,
        deadline,
        symmetry_reduction,
    )
    global_best = min(population, key=lambda individual: individual.energy)
    initial_best_energy = global_best.energy
    initial_average_energy = sum(item.energy for item in population) / len(population)
    best_history = [global_best.energy]
    average_history = [initial_average_energy]
    diversity_history = [_diversity(population)]
    energy_evaluations_history = [counters.evaluations]
    pre_local_history: list[float] = []
    generations = 0
    generations_without_improvement = 0
    local_calls = 0
    local_replacements = 0
    local_search_move_attempts = 0
    local_energy_evaluations = 0
    local_candidate_structures = 0
    local_valid_candidates = 0
    local_invalid_candidates = 0
    local_duplicate_replacements_avoided = 0
    local_execution_time = 0.0
    termination_reason = "time_limit" if initialization_timed_out else "maximum_generations"

    if ga_config.target_energy is not None and global_best.energy <= ga_config.target_energy:
        termination_reason = "target_energy"
    elif initialization_timed_out:
        pass
    else:
        while generations < ga_config.max_generations:
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
                ga_config.elitism_count,
                population,
                key=lambda individual: individual.energy,
            )
            generation_seen: dict[object, _Candidate] = {
                (
                    symmetry_key(individual.structure)
                    if symmetry_reduction
                    else individual.genome
                ): individual
                for individual in next_population
            }
            attempts = 0
            generation_timed_out = False
            while (
                len(next_population) < ga_config.population_size
                and attempts < ga_config.max_offspring_attempts
                and not (
                    evaluation_budget is not None
                    and evaluation_budget.exhausted
                )
            ):
                if deadline is not None and perf_counter() >= deadline:
                    generation_timed_out = True
                    break
                attempts += 1
                genome = _make_offspring_genome(population, ga_config, generator)
                offspring = _evaluate_genome(
                    sequence,
                    genome,
                    float(epsilon),
                    counters,
                    generation_seen,
                    symmetry_reduction,
                    evaluation_budget,
                    constraints,
                )
                if offspring is not None:
                    next_population.append(offspring)
            attempt_limit_reached = (
                len(next_population) != ga_config.population_size
                and attempts >= ga_config.max_offspring_attempts
            )
            if len(next_population) != ga_config.population_size:
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
                        population[: ga_config.population_size - len(next_population)]
                    )
                else:
                    raise RuntimeError(
                        "Could not fill the next generation with valid offspring: "
                        f"created {len(next_population)} of {ga_config.population_size} "
                        f"individuals after {ga_config.max_offspring_attempts} attempts."
                    )

            population = next_population
            generations += 1
            pre_local_best = min(population, key=lambda individual: individual.energy)
            pre_local_history.append(pre_local_best.energy)

            if (
                settings.local_search_fraction > 0
                and generations % settings.local_search_frequency == 0
            ):
                count = max(
                    1,
                    math.ceil(ga_config.population_size * settings.local_search_fraction),
                )
                selected = sorted(
                    range(len(population)),
                    key=lambda index: population[index].energy,
                )[:count]
                for index in selected:
                    if (
                        evaluation_budget is not None
                        and evaluation_budget.exhausted
                    ):
                        break
                    if deadline is not None and perf_counter() >= deadline:
                        break
                    original = population[index]
                    local_start = perf_counter()
                    local_result = run_simulated_annealing(
                        sequence,
                        original.structure,
                        settings.local_search,
                        epsilon=float(epsilon),
                        rng=generator,
                        evaluation_budget=evaluation_budget,
                        constraints=constraints,
                        deadline=deadline,
                    )
                    local_execution_time += perf_counter() - local_start
                    local_calls += 1
                    local_search_move_attempts += local_result.local_move_attempts
                    local_energy_evaluations += local_result.energy_evaluations
                    local_candidate_structures += (
                        local_result.valid_candidate_structures
                    )
                    local_valid_candidates += (
                        local_result.valid_candidate_structures
                    )
                    local_invalid_candidates += (
                        local_result.invalid_candidate_structures
                    )
                    if local_result.best_energy < original.energy:
                        refined = _Candidate(
                            genome=local_result.best_structure.directions,
                            structure=local_result.best_structure,
                            energy=local_result.best_energy,
                        )
                        key = (
                            symmetry_key(refined.structure)
                            if symmetry_reduction
                            else refined.genome
                        )
                        if any(
                            other_index != index
                            and (
                                symmetry_key(individual.structure)
                                if symmetry_reduction
                                else individual.genome
                            ) == key
                            for other_index, individual in enumerate(population)
                        ):
                            local_duplicate_replacements_avoided += 1
                            continue
                        population[index] = refined
                        local_replacements += 1

            generation_best = min(population, key=lambda individual: individual.energy)
            if generation_best.energy < global_best.energy:
                global_best = generation_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Keep the global best explicitly in the population.
            if not any(item.genome == global_best.genome for item in population):
                worst_index = max(
                    range(len(population)),
                    key=lambda index: population[index].energy,
                )
                population[worst_index] = global_best

            best_history.append(global_best.energy)
            average_history.append(
                sum(item.energy for item in population) / len(population)
            )
            diversity_history.append(_diversity(population))
            energy_evaluations_history.append(
                counters.evaluations + local_energy_evaluations
            )

            if (
                ga_config.target_energy is not None
                and global_best.energy <= ga_config.target_energy
            ):
                termination_reason = "target_energy"
                break
            if (
                ga_config.no_improvement_limit is not None
                and generations_without_improvement
                >= ga_config.no_improvement_limit
            ):
                termination_reason = "no_improvement_limit"
                break
            if (
                evaluation_budget is not None
                and evaluation_budget.exhausted
            ):
                termination_reason = "evaluation_budget"
                break
            if generation_timed_out:
                termination_reason = "time_limit"
                break
            if attempt_limit_reached:
                termination_reason = "offspring_attempt_limit"
                break
            if deadline is not None and perf_counter() >= deadline:
                termination_reason = "time_limit"
                break

    elapsed = perf_counter() - start_time
    final_best = min(population, key=lambda individual: individual.energy)
    final_average = sum(item.energy for item in population) / len(population)
    return HybridResult(
        algorithm="hybrid_genetic_algorithm_simulated_annealing",
        best_individual=global_best.genome,
        best_structure=global_best.structure,
        best_energy=global_best.energy,
        initial_population_best_energy=initial_best_energy,
        initial_population_average_energy=initial_average_energy,
        final_population_best_energy=final_best.energy,
        final_population_average_energy=final_average,
        improvement_vs_initial_best=initial_best_energy - global_best.energy,
        generations=generations,
        local_search_calls=local_calls,
        local_search_replacements=local_replacements,
        local_search_frequency=settings.local_search_frequency,
        local_search_fraction=settings.local_search_fraction,
        generated_candidates=counters.generated,
        local_search_move_attempts=local_search_move_attempts,
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
        elite_individuals_copied=generations * ga_config.elitism_count,
        energy_evaluations=counters.evaluations + local_energy_evaluations,
        genetic_energy_evaluations=counters.evaluations,
        local_search_energy_evaluations=local_energy_evaluations,
        local_search_candidate_structures=local_candidate_structures,
        local_search_valid_candidates=local_valid_candidates,
        local_search_invalid_candidates=local_invalid_candidates,
        local_search_duplicate_replacements_avoided=local_duplicate_replacements_avoided,
        local_search_execution_time_seconds=local_execution_time,
        execution_time_seconds=elapsed,
        best_energy_history=tuple(best_history),
        average_energy_history=tuple(average_history),
        pre_local_search_best_history=tuple(pre_local_history),
        diversity_history=tuple(diversity_history),
        energy_evaluations_history=tuple(energy_evaluations_history),
        termination_reason=termination_reason,
        seed=(
            settings.seed if settings.seed is not None else ga_config.seed
        ) if rng is None else None,
    )
