import unittest

from hp_folding.genetic_algorithm import GeneticAlgorithmConfig
from hp_folding.hybrid_optimizer import HybridConfig, run_hybrid_optimizer
from hp_folding.simulated_annealing import SimulatedAnnealingConfig
from hp_folding.structure import validate_structure


class HybridOptimizerTests(unittest.TestCase):
    def make_config(self, **ga_overrides):
        ga_values = {
            "population_size": 10,
            "max_generations": 5,
            "tournament_size": 3,
            "elitism_count": 1,
            "max_offspring_attempts": 500,
            "seed": 27,
        }
        ga_values.update(ga_overrides)
        return HybridConfig(
            genetic=GeneticAlgorithmConfig(**ga_values),
            local_search=SimulatedAnnealingConfig(
                initial_temperature=2.0,
                cooling_rate=0.9,
                minimum_temperature=0,
                max_iterations=8,
                seed=27,
            ),
            local_search_frequency=1,
            local_search_fraction=0.3,
            seed=27,
        )

    def test_hybrid_returns_valid_structure_and_full_statistics(self):
        result = run_hybrid_optimizer("H" * 8, self.make_config())
        self.assertTrue(validate_structure(result.best_structure, expected_length=8).valid)
        self.assertEqual(len(result.best_individual), 7)
        self.assertEqual(result.best_energy, min(result.best_energy_history))
        self.assertEqual(len(result.best_energy_history), result.generations + 1)
        self.assertEqual(len(result.average_energy_history), result.generations + 1)
        self.assertEqual(len(result.pre_local_search_best_history), result.generations)
        self.assertEqual(len(result.diversity_history), result.generations + 1)
        self.assertEqual(
            len(result.energy_evaluations_history),
            len(result.best_energy_history),
        )
        self.assertEqual(
            result.energy_evaluations_history[-1], result.energy_evaluations
        )
        self.assertEqual(
            result.energy_evaluations,
            result.genetic_energy_evaluations
            + result.local_search_energy_evaluations,
        )
        self.assertEqual(
            result.generated_candidates,
            result.valid_candidates + result.invalid_candidates,
        )
        self.assertLessEqual(result.best_energy, result.initial_population_best_energy)
        self.assertGreaterEqual(result.local_search_calls, result.local_search_replacements)
        self.assertGreaterEqual(result.execution_time_seconds, 0)
        self.assertGreaterEqual(result.local_search_execution_time_seconds, 0)

    def test_local_search_frequency_controls_calls(self):
        config = self.make_config(max_generations=4)
        config = HybridConfig(
            genetic=config.genetic,
            local_search=config.local_search,
            local_search_frequency=2,
            local_search_fraction=0.5,
            seed=27,
        )
        result = run_hybrid_optimizer("H" * 8, config)
        self.assertEqual(result.local_search_calls, 5 * 2)

    def test_zero_local_search_fraction_disables_sa(self):
        base = self.make_config()
        config = HybridConfig(
            genetic=base.genetic,
            local_search=base.local_search,
            local_search_fraction=0,
            seed=27,
        )
        result = run_hybrid_optimizer("HPPHHPPH", config)
        self.assertEqual(result.local_search_calls, 0)
        self.assertEqual(result.local_search_energy_evaluations, 0)
        self.assertEqual(
            result.energy_evaluations, result.genetic_energy_evaluations
        )

    def test_partial_generation_preserves_valid_population_on_attempt_limit(self):
        config = HybridConfig(
            genetic=GeneticAlgorithmConfig(
                population_size=6,
                max_generations=5,
                tournament_size=2,
                elitism_count=1,
                max_offspring_attempts=1,
                seed=13,
            ),
            local_search=SimulatedAnnealingConfig(max_iterations=0),
            local_search_fraction=0,
            seed=13,
        )
        result = run_hybrid_optimizer("HPPHH", config)
        self.assertEqual(result.termination_reason, "offspring_attempt_limit")
        self.assertTrue(validate_structure(result.best_structure, expected_length=5).valid)
        self.assertLessEqual(
            result.best_energy, result.initial_population_best_energy
        )

    def test_local_search_frequency_and_fraction_are_validated(self):
        base = self.make_config()
        for frequency, fraction in ((0, 0.2), (1, -0.1), (1, 1.1)):
            with self.subTest(frequency=frequency, fraction=fraction):
                with self.assertRaises(ValueError):
                    HybridConfig(
                        genetic=base.genetic,
                        local_search=base.local_search,
                        local_search_frequency=frequency,
                        local_search_fraction=fraction,
                    )

    def test_fixed_seed_is_reproducible(self):
        config = self.make_config()
        first = run_hybrid_optimizer("HPPHHPPH", config)
        second = run_hybrid_optimizer("HPPHHPPH", config)
        self.assertEqual(first.best_individual, second.best_individual)
        self.assertEqual(first.best_energy_history, second.best_energy_history)
        self.assertEqual(first.average_energy_history, second.average_energy_history)

    def test_sa_improvements_replace_population_members(self):
        config = HybridConfig(
            genetic=GeneticAlgorithmConfig(
                population_size=6,
                max_generations=1,
                tournament_size=2,
                elitism_count=1,
                max_offspring_attempts=1000,
                seed=0,
            ),
            local_search=SimulatedAnnealingConfig(
                initial_temperature=2,
                cooling_rate=0.9,
                minimum_temperature=0,
                max_iterations=30,
            ),
            local_search_fraction=1,
            seed=0,
        )
        result = run_hybrid_optimizer("H" * 8, config)
        self.assertGreater(result.local_search_replacements, 0)
        self.assertLessEqual(result.best_energy, result.initial_population_best_energy)

    def test_target_energy_can_stop_before_generations(self):
        config = self.make_config(max_generations=10, target_energy=0)
        result = run_hybrid_optimizer("HHHH", config)
        self.assertEqual(result.generations, 0)
        self.assertEqual(result.termination_reason, "target_energy")

    def test_invalid_hp_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only H and P"):
            run_hybrid_optimizer("H-X", self.make_config())

    def test_hybrid_does_not_claim_to_outperform_baseline(self):
        result = run_hybrid_optimizer("HPPHHPPH", self.make_config())
        self.assertAlmostEqual(
            result.improvement_vs_initial_best,
            result.initial_population_best_energy - result.best_energy,
        )
        # Improvement is an observed statistic and is not constrained positive.
        self.assertIsInstance(result.improvement_vs_initial_best, float)


if __name__ == "__main__":
    unittest.main()
