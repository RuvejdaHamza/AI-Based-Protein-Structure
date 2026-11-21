import unittest
from unittest.mock import patch

from hp_folding import genetic_algorithm
from hp_folding.energy import calculate_energy
from hp_folding.genetic_algorithm import GeneticAlgorithmConfig, run_genetic_algorithm
from hp_folding.hybrid_optimizer import HybridConfig, run_hybrid_optimizer
from hp_folding.search_space import IncrementalEnergyTracker
from hp_folding.simulated_annealing import (
    SimulatedAnnealingConfig,
    run_simulated_annealing,
)
from hp_folding.structure import ProteinStructure, validate_structure


class AlgorithmInvariantTests(unittest.TestCase):
    def assert_result_is_valid_and_consistent(self, hp_sequence, result):
        validation = validate_structure(
            result.best_structure, expected_length=len(hp_sequence)
        )
        self.assertTrue(validation.valid, validation.issues)
        recomputed = calculate_energy(hp_sequence, result.best_structure)
        self.assertEqual(result.best_energy, recomputed.energy)
        self.assertEqual(result.best_energy, min(result.best_energy_history))

    def test_all_optimizers_preserve_and_return_the_best_valid_structure(self):
        hp_sequence = "HPPHHPPH"
        initial = ProteinStructure.from_directions(["R"] * (len(hp_sequence) - 1))

        sa_result = run_simulated_annealing(
            hp_sequence,
            initial,
            SimulatedAnnealingConfig(
                initial_temperature=2.0,
                cooling_rate=0.8,
                minimum_temperature=0.0,
                max_iterations=8,
                seed=41,
            ),
        )
        ga_result = run_genetic_algorithm(
            hp_sequence,
            GeneticAlgorithmConfig(
                population_size=8,
                max_generations=3,
                tournament_size=2,
                elitism_count=1,
                max_offspring_attempts=300,
                seed=41,
            ),
        )
        hybrid_result = run_hybrid_optimizer(
            hp_sequence,
            HybridConfig(
                genetic=GeneticAlgorithmConfig(
                    population_size=8,
                    max_generations=2,
                    tournament_size=2,
                    elitism_count=1,
                    max_offspring_attempts=300,
                    seed=41,
                ),
                local_search=SimulatedAnnealingConfig(
                    initial_temperature=2.0,
                    cooling_rate=0.8,
                    minimum_temperature=0.0,
                    max_iterations=4,
                ),
                local_search_frequency=1,
                local_search_fraction=0.25,
                seed=41,
            ),
        )

        for result in (sa_result, ga_result, hybrid_result):
            with self.subTest(algorithm=result.algorithm):
                self.assert_result_is_valid_and_consistent(hp_sequence, result)

        self.assertLessEqual(sa_result.best_energy, sa_result.initial_energy)
        self.assertLessEqual(ga_result.best_energy, ga_result.best_energy_history[0])
        self.assertLessEqual(
            hybrid_result.best_energy,
            hybrid_result.initial_population_best_energy,
        )
        for result in (ga_result, hybrid_result):
            self.assertTrue(
                all(
                    current <= previous
                    for previous, current in zip(
                        result.best_energy_history,
                        result.best_energy_history[1:],
                    )
                )
            )

    def test_invalid_structures_never_reach_algorithm_energy_scoring(self):
        hp_sequence = "H" * 8
        scored_ga_structures = []
        scored_local_structures = []
        original_ga_score = genetic_algorithm._calculate_energy_validated
        original_local_score = IncrementalEnergyTracker.evaluate_validated

        def checked_ga_score(sequence, structure, epsilon):
            validation = validate_structure(
                structure, expected_length=len(sequence)
            )
            self.assertTrue(validation.valid, validation.issues)
            scored_ga_structures.append(structure)
            return original_ga_score(sequence, structure, epsilon)

        def checked_local_score(tracker, candidate):
            validation = validate_structure(
                candidate, expected_length=len(tracker.hp_sequence)
            )
            self.assertTrue(validation.valid, validation.issues)
            scored_local_structures.append(candidate)
            return original_local_score(tracker, candidate)

        with (
            patch(
                "hp_folding.genetic_algorithm._calculate_energy_validated",
                side_effect=checked_ga_score,
            ),
            patch.object(
                IncrementalEnergyTracker,
                "evaluate_validated",
                autospec=True,
                side_effect=checked_local_score,
            ),
        ):
            ga_result = run_genetic_algorithm(
                hp_sequence,
                GeneticAlgorithmConfig(
                    population_size=8,
                    max_generations=3,
                    tournament_size=2,
                    elitism_count=1,
                    max_offspring_attempts=500,
                    seed=0,
                ),
            )
            self.assertGreater(ga_result.invalid_candidates, 0)
            self.assertEqual(
                len(scored_ga_structures), ga_result.energy_evaluations
            )

            local_calls_before_sa = len(scored_local_structures)
            sa_result = run_simulated_annealing(
                hp_sequence,
                ProteinStructure.from_directions(["R"] * (len(hp_sequence) - 1)),
                SimulatedAnnealingConfig(
                    initial_temperature=2.0,
                    cooling_rate=0.8,
                    minimum_temperature=0.0,
                    max_iterations=6,
                    seed=11,
                ),
            )
            self.assertEqual(
                len(scored_local_structures) - local_calls_before_sa,
                sa_result.energy_evaluations - 1,
            )

            ga_calls_before_hybrid = len(scored_ga_structures)
            local_calls_before_hybrid = len(scored_local_structures)
            hybrid_result = run_hybrid_optimizer(
                hp_sequence,
                HybridConfig(
                    genetic=GeneticAlgorithmConfig(
                        population_size=8,
                        max_generations=2,
                        tournament_size=2,
                        elitism_count=1,
                        max_offspring_attempts=500,
                        seed=13,
                    ),
                    local_search=SimulatedAnnealingConfig(
                        initial_temperature=2.0,
                        cooling_rate=0.8,
                        minimum_temperature=0.0,
                        max_iterations=4,
                    ),
                    local_search_frequency=1,
                    local_search_fraction=0.25,
                    seed=13,
                ),
            )

        self.assertEqual(
            len(scored_ga_structures) - ga_calls_before_hybrid,
            hybrid_result.genetic_energy_evaluations,
        )
        self.assertEqual(
            len(scored_local_structures) - local_calls_before_hybrid,
            hybrid_result.local_search_energy_evaluations
            - hybrid_result.local_search_calls,
        )
        for result in (ga_result, sa_result, hybrid_result):
            self.assert_result_is_valid_and_consistent(hp_sequence, result)

    def test_single_residue_is_supported_by_all_optimizers(self):
        hp_sequence = "H"
        initial = ProteinStructure(((0, 0),))
        sa_config = SimulatedAnnealingConfig(max_iterations=2, seed=7)
        ga_config = GeneticAlgorithmConfig(
            population_size=4,
            max_generations=1,
            tournament_size=2,
            elitism_count=1,
            max_offspring_attempts=10,
            seed=7,
        )
        results = (
            run_simulated_annealing(hp_sequence, initial, sa_config),
            run_genetic_algorithm(hp_sequence, ga_config),
            run_hybrid_optimizer(
                hp_sequence,
                HybridConfig(
                    genetic=ga_config,
                    local_search=sa_config,
                    local_search_fraction=1.0,
                    seed=7,
                ),
            ),
        )

        for result in results:
            with self.subTest(algorithm=result.algorithm):
                self.assert_result_is_valid_and_consistent(hp_sequence, result)
                self.assertEqual(result.best_structure, initial)
                self.assertEqual(result.best_energy, 0.0)


if __name__ == "__main__":
    unittest.main()
