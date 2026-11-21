import random
import math
import unittest

from hp_folding.simulated_annealing import (
    SimulatedAnnealingConfig,
    acceptance_probability,
    run_simulated_annealing,
)
from hp_folding.structure import ProteinStructure, validate_structure


class SimulatedAnnealingTests(unittest.TestCase):
    def test_acceptance_probability_matches_metropolis_rule(self):
        self.assertEqual(acceptance_probability(-2.0, 1.0), 1.0)
        self.assertEqual(acceptance_probability(0.0, 1.0), 1.0)
        self.assertAlmostEqual(acceptance_probability(2.0, 2.0), math.exp(-1))

    def test_better_candidate_is_accepted_and_best_is_tracked(self):
        # A straight chain has no contacts; the square has one H-H contact.
        initial = ProteinStructure.from_directions(["R", "R", "R"])
        result = run_simulated_annealing(
            "HHHH",
            initial,
            SimulatedAnnealingConfig(
                initial_temperature=2.0,
                cooling_rate=0.9,
                minimum_temperature=0.0,
                max_iterations=30,
                seed=17,
            ),
        )
        self.assertEqual(result.initial_energy, 0.0)
        self.assertLessEqual(result.best_energy, result.initial_energy)
        self.assertEqual(result.best_energy, min(result.best_energy_history))
        self.assertTrue(
            validate_structure(result.best_structure, expected_length=4).valid
        )

    def test_result_contains_consistent_counters_and_histories(self):
        initial = ProteinStructure.from_directions(["R", "R", "R"])
        result = run_simulated_annealing(
            "HHHH",
            initial,
            SimulatedAnnealingConfig(
                initial_temperature=1.0,
                cooling_rate=0.5,
                minimum_temperature=0.0,
                max_iterations=5,
                seed=2,
            ),
        )
        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.energy_evaluations, result.accepted_moves + result.rejected_moves + 1)
        self.assertEqual(len(result.current_energy_history), result.iterations + 1)
        self.assertEqual(len(result.best_energy_history), result.iterations + 1)
        self.assertEqual(
            len(result.energy_evaluations_history),
            len(result.best_energy_history),
        )
        self.assertEqual(
            result.energy_evaluations_history[-1], result.energy_evaluations
        )
        self.assertGreaterEqual(
            result.valid_candidate_structures,
            result.accepted_moves + result.rejected_moves,
        )
        self.assertGreaterEqual(result.invalid_candidate_structures, 0)
        self.assertEqual(
            result.accepted_moves + result.rejected_moves,
            result.energy_evaluations - 1,
        )
        self.assertGreaterEqual(result.execution_time_seconds, 0.0)

    def test_fixed_seed_is_reproducible_except_runtime(self):
        initial = ProteinStructure.from_directions(["R", "R", "R"])
        config = SimulatedAnnealingConfig(
            initial_temperature=3.0,
            cooling_rate=0.95,
            minimum_temperature=0.0,
            max_iterations=12,
            seed=81,
        )
        first = run_simulated_annealing("HHHH", initial, config)
        second = run_simulated_annealing("HHHH", initial, config)
        self.assertEqual(first.best_structure, second.best_structure)
        self.assertEqual(first.best_energy_history, second.best_energy_history)
        self.assertEqual(first.current_energy_history, second.current_energy_history)

    def test_target_energy_stops_immediately_when_initial_state_meets_it(self):
        initial = ProteinStructure.from_directions(["R", "R"])
        result = run_simulated_annealing(
            "HHH",
            initial,
            SimulatedAnnealingConfig(target_energy=0.0, max_iterations=20),
        )
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.termination_reason, "target_energy")

    def test_zero_iteration_budget_returns_initial_structure(self):
        initial = ProteinStructure.from_directions(["R", "R"])
        result = run_simulated_annealing(
            "HHH", initial, SimulatedAnnealingConfig(max_iterations=0)
        )
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.best_structure, initial)
        self.assertEqual(result.energy_evaluations, 1)
        self.assertEqual(result.termination_reason, "maximum_iterations")

    def test_invalid_initial_structure_is_rejected(self):
        invalid = ProteinStructure.from_directions(["R", "L"])
        with self.assertRaisesRegex(ValueError, "initial_structure must be valid"):
            run_simulated_annealing("HHH", invalid)

    def test_config_rejects_invalid_schedule_and_stopping_settings(self):
        for kwargs in (
            {"initial_temperature": 0.0},
            {"cooling_rate": 1.0},
            {"minimum_temperature": -1.0},
            {"max_iterations": -1},
            {"no_improvement_limit": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SimulatedAnnealingConfig(**kwargs)

    def test_acceptance_probability_rejects_invalid_temperature(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            acceptance_probability(1.0, 0.0)

    def test_optional_rng_is_supported(self):
        initial = ProteinStructure.from_directions(["R", "R", "R"])
        result = run_simulated_annealing(
            "HHHH",
            initial,
            SimulatedAnnealingConfig(max_iterations=3, minimum_temperature=0),
            rng=random.Random(4),
        )
        self.assertEqual(result.seed, None)
        self.assertEqual(result.iterations, 3)

    def test_no_valid_neighbors_terminates_without_scoring_invalid_candidates(self):
        initial = ProteinStructure(((0, 0),))
        result = run_simulated_annealing(
            "H",
            initial,
            SimulatedAnnealingConfig(max_iterations=20, minimum_temperature=0),
        )
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination_reason, "no_valid_neighbors")
        self.assertEqual(result.energy_evaluations, 1)
        self.assertEqual(result.best_structure, initial)

    def test_non_string_hp_sequence_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a string"):
            run_simulated_annealing(None, ProteinStructure(((0, 0),)))

    def test_restricted_move_types_are_recorded(self):
        initial = ProteinStructure.from_directions(["R", "R", "R"])
        result = run_simulated_annealing(
            "HHHH",
            initial,
            SimulatedAnnealingConfig(
                initial_temperature=2,
                cooling_rate=0.9,
                minimum_temperature=0,
                max_iterations=2,
                move_types=("end",),
                seed=1,
            ),
        )
        self.assertGreater(result.restricted_operator_opportunities, 0)
        self.assertGreaterEqual(result.duplicate_candidates_avoided, 0)
        self.assertLess(
            result.incremental_neighbor_checks,
            result.full_scan_neighbor_checks_estimate,
        )
        self.assertEqual(
            result.local_move_attempts,
            result.valid_candidate_structures
            + result.invalid_candidate_structures
            + result.duplicate_candidates_avoided,
        )

    def test_unknown_move_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "move_types"):
            SimulatedAnnealingConfig(move_types=("diagonal",))


if __name__ == "__main__":
    unittest.main()
