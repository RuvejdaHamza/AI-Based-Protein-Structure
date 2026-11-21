import unittest
import random

from hp_folding.genetic_algorithm import (
    GeneticAlgorithmConfig,
    _crossover,
    _mutate,
    run_genetic_algorithm,
)
from hp_folding.structure import Direction, ProteinStructure, validate_structure


class GeneticAlgorithmTests(unittest.TestCase):
    def test_direction_string_crossover_keeps_genome_length(self):
        parent_a = (Direction.RIGHT, Direction.RIGHT, Direction.UP, Direction.LEFT)
        parent_b = (Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT)
        child = _crossover(parent_a, parent_b, random.Random(2))
        self.assertEqual(len(child), len(parent_a))
        self.assertTrue(all(isinstance(direction, Direction) for direction in child))

    def test_mutation_preserves_genome_length_and_direction_alphabet(self):
        import random

        parent = (Direction.RIGHT,) * 10
        child = _mutate(parent, 1.0, random.Random(5))
        self.assertEqual(len(child), len(parent))
        self.assertTrue(all(direction in Direction for direction in child))
        self.assertNotEqual(child, parent)

    def test_ga_returns_valid_best_individual_and_tracks_histories(self):
        result = run_genetic_algorithm(
            "HHHHHH",
            GeneticAlgorithmConfig(
                population_size=12,
                max_generations=20,
                tournament_size=3,
                crossover_rate=0.8,
                mutation_rate=0.2,
                elitism_count=1,
                max_offspring_attempts=500,
                seed=42,
            ),
        )
        self.assertTrue(validate_structure(result.best_structure, expected_length=6).valid)
        self.assertEqual(len(result.best_individual), 5)
        self.assertEqual(result.best_energy, min(result.best_energy_history))
        self.assertEqual(len(result.best_energy_history), result.generations + 1)
        self.assertEqual(len(result.average_energy_history), result.generations + 1)
        self.assertEqual(len(result.diversity_history), result.generations + 1)
        self.assertEqual(
            len(result.energy_evaluations_history),
            len(result.best_energy_history),
        )
        self.assertEqual(
            result.energy_evaluations_history[-1], result.energy_evaluations
        )
        self.assertEqual(result.elite_individuals_copied, result.generations)
        self.assertLessEqual(
            result.coordinate_steps_checked,
            result.generated_candidates * (6 - 1),
        )
        self.assertGreaterEqual(result.execution_time_seconds, 0)
        self.assertEqual(result.valid_candidates + result.invalid_candidates, result.generated_candidates)
        self.assertLessEqual(result.best_energy_history[-1], result.best_energy_history[0])

    def test_ga_is_reproducible_with_fixed_seed(self):
        config = GeneticAlgorithmConfig(
            population_size=8,
            max_generations=5,
            tournament_size=2,
            max_offspring_attempts=300,
            seed=19,
        )
        first = run_genetic_algorithm("HPPHHPPH", config)
        second = run_genetic_algorithm("HPPHHPPH", config)
        self.assertEqual(first.best_individual, second.best_individual)
        self.assertEqual(first.best_energy_history, second.best_energy_history)
        self.assertEqual(first.average_energy_history, second.average_energy_history)

    def test_invalid_offspring_are_counted_and_rejected(self):
        result = run_genetic_algorithm(
            "H" * 10,
            GeneticAlgorithmConfig(
                population_size=12,
                max_generations=4,
                tournament_size=3,
                max_offspring_attempts=500,
                seed=0,
            ),
        )
        self.assertGreater(result.invalid_candidates, 0)
        self.assertEqual(
            result.generated_candidates,
            result.valid_candidates + result.invalid_candidates,
        )
        self.assertGreater(result.early_collision_rejections, 0)
        self.assertLess(
            result.coordinate_steps_checked,
            result.generated_candidates * (10 - 1),
        )
        self.assertEqual(
            result.energy_evaluations,
            result.valid_candidates - result.duplicate_candidates,
        )
        self.assertGreater(result.duplicate_candidates, 0)
        self.assertTrue(validate_structure(result.best_structure, expected_length=10).valid)

    def test_zero_generations_returns_best_initial_individual(self):
        result = run_genetic_algorithm(
            "HPPH",
            GeneticAlgorithmConfig(
                population_size=5,
                max_generations=0,
                tournament_size=2,
                seed=3,
            ),
        )
        self.assertEqual(result.generations, 0)
        self.assertEqual(len(result.best_energy_history), 1)
        self.assertEqual(result.termination_reason, "maximum_generations")

    def test_partial_generation_keeps_valid_best_candidate_on_attempt_limit(self):
        result = run_genetic_algorithm(
            "HPPHH",
            GeneticAlgorithmConfig(
                population_size=6,
                max_generations=5,
                tournament_size=2,
                max_offspring_attempts=1,
                seed=13,
            ),
        )
        self.assertEqual(result.termination_reason, "offspring_attempt_limit")
        self.assertTrue(validate_structure(result.best_structure, expected_length=5).valid)
        self.assertLessEqual(result.best_energy, result.best_energy_history[0])

    def test_target_energy_can_stop_before_first_generation(self):
        result = run_genetic_algorithm(
            "HHH",
            GeneticAlgorithmConfig(
                population_size=4,
                max_generations=5,
                tournament_size=2,
                target_energy=0,
                seed=1,
            ),
        )
        self.assertEqual(result.generations, 0)
        self.assertEqual(result.termination_reason, "target_energy")

    def test_invalid_hp_sequence_is_rejected(self):
        for sequence in ("", "HXP", "H-P"):
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    run_genetic_algorithm(sequence)

    def test_config_rejects_inconsistent_or_invalid_parameters(self):
        for kwargs in (
            {"population_size": 0},
            {"max_generations": -1},
            {"tournament_size": 0},
            {"population_size": 2, "tournament_size": 3},
            {"elitism_count": 0},
            {"population_size": 2, "elitism_count": 3},
            {"crossover_rate": 1.1},
            {"mutation_rate": -0.1},
            {"max_offspring_attempts": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    GeneticAlgorithmConfig(**kwargs)

    def test_invalid_population_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithmConfig(population_size=0)


if __name__ == "__main__":
    unittest.main()
