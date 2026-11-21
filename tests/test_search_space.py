import random
import unittest

from hp_folding.search_space import IncrementalEnergyTracker, symmetry_key
from hp_folding.genetic_algorithm import _Counters, _evaluate_genome
from benchmarks.benchmark_performance import _baseline_neighborhood
from hp_folding.local_moves import generate_neighborhood
from hp_folding.structure import (
    Direction,
    ProteinStructure,
    generate_random_valid_structure,
    validate_structure,
)


class SearchSpaceReductionTests(unittest.TestCase):
    def test_symmetry_key_matches_rotated_reflected_and_translated_walk(self):
        structure = ProteinStructure.from_directions(["R", "U", "R", "D"])
        transformed = ProteinStructure(
            tuple((4 - y, 7 + x) for x, y in structure.coordinates)
        )
        self.assertEqual(symmetry_key(structure), symmetry_key(transformed))

    def test_ga_symmetry_duplicate_skips_energy_evaluation(self):
        from hp_folding.energy import calculate_energy

        first_genome = (
            Direction.RIGHT,
            Direction.UP,
            Direction.RIGHT,
        )
        rotated_genome = (
            Direction.UP,
            Direction.LEFT,
            Direction.UP,
        )
        counters = _Counters()
        seen = {}
        first = _evaluate_genome(
            "HHHH", first_genome, 1.0, counters, seen, symmetry_reduction=True
        )
        evaluations_after_first = counters.evaluations
        second = _evaluate_genome(
            "HHHH", rotated_genome, 1.0, counters, seen, symmetry_reduction=True
        )
        self.assertIsNotNone(first)
        self.assertIs(second, first)
        self.assertEqual(counters.evaluations, evaluations_after_first)
        self.assertEqual(counters.duplicates, 1)
        self.assertEqual(counters.symmetry_duplicates, 1)
        self.assertEqual(
            first.energy, calculate_energy("HHHH", first.structure).energy
        )

    def test_symmetry_key_does_not_always_conflate_chain_reversal(self):
        first = ProteinStructure(
            ((0, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
        )
        reversed_chain = ProteinStructure(tuple(reversed(first.coordinates)))
        self.assertNotEqual(symmetry_key(first), symmetry_key(reversed_chain))

    def test_incremental_delta_matches_full_energy_for_local_moves(self):
        rng = random.Random(7)
        for length in (4, 8, 15):
            structure = generate_random_valid_structure(length, rng=rng)
            tracker = IncrementalEnergyTracker("H" * length, structure, epsilon=1.5)
            for _ in range(25):
                proposed = generate_random_valid_structure(length, rng=rng)
                delta = tracker.evaluate(proposed)
                from hp_folding.energy import calculate_energy

                expected = calculate_energy("H" * length, proposed, epsilon=1.5)
                self.assertAlmostEqual(delta.energy, expected.energy)
                self.assertEqual(delta.contact_count, expected.contact_count)
                tracker.commit(proposed, delta)
                self.assertAlmostEqual(tracker.energy, expected.energy)
                self.assertEqual(tracker.contact_count, expected.contact_count)
                tracker.full_energy()
                self.assertAlmostEqual(tracker.energy, expected.energy)

    def test_incremental_delta_matches_full_energy_with_polar_sequence(self):
        sequence = "HPPHHHPH"
        rng = random.Random(12)
        current = generate_random_valid_structure(len(sequence), rng=rng)
        tracker = IncrementalEnergyTracker(sequence, current)
        for _ in range(30):
            candidate = generate_random_valid_structure(len(sequence), rng=rng)
            delta = tracker.evaluate(candidate)
            from hp_folding.energy import calculate_energy

            expected = calculate_energy(sequence, candidate)
            self.assertEqual(delta.contact_count, expected.contact_count)
            tracker.commit(candidate, delta)

    def test_invalid_candidate_is_rejected_by_incremental_tracker(self):
        structure = ProteinStructure.from_directions(["R", "L"])
        tracker = IncrementalEnergyTracker("HHH", ProteinStructure.from_directions(["R", "R"]))
        with self.assertRaisesRegex(ValueError, "invalid candidate"):
            tracker.evaluate(structure)

    def test_tracker_rejects_stale_delta_commit(self):
        source = ProteinStructure.from_directions(["R", "R", "R"])
        tracker = IncrementalEnergyTracker("HHHH", source)
        first = ProteinStructure.from_directions(["R", "U", "L"])
        second = ProteinStructure.from_directions(["U", "R", "D"])
        delta = tracker.evaluate(first)
        with self.assertRaisesRegex(ValueError, "different candidate"):
            tracker.commit(second, delta)

    def test_optimized_neighborhood_matches_reference_for_seeded_structures(self):
        rng = random.Random(84)
        for length in (4, 8, 15, 25):
            structure = generate_random_valid_structure(length, rng=rng)
            reference = _baseline_neighborhood(structure)
            optimized = generate_neighborhood(structure)
            reference_keys = {symmetry_key(candidate) for candidate in reference}
            optimized_keys = {symmetry_key(candidate) for candidate in optimized}
            self.assertEqual(reference_keys, optimized_keys)


if __name__ == "__main__":
    unittest.main()
