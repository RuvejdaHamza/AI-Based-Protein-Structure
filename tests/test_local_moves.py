import random
import unittest

from hp_folding.local_moves import (
    MoveStatistics,
    corner_move,
    crankshaft_move,
    end_move,
    generate_neighborhood,
)
from hp_folding.structure import ProteinStructure, StructureError, validate_structure


class LocalMoveTests(unittest.TestCase):
    def assert_valid_candidate(self, candidate, expected_length):
        self.assertIsNotNone(candidate)
        self.assertTrue(
            validate_structure(candidate, expected_length=expected_length).valid
        )

    def test_end_move_repositions_terminal_residue(self):
        structure = ProteinStructure.from_directions(["R", "R"])
        candidate = end_move(structure, residue_index=2, target=(1, 1))
        self.assert_valid_candidate(candidate, 3)
        self.assertEqual(candidate.coordinates, ((0, 0), (1, 0), (1, 1)))

    def test_end_move_rejects_collision_and_non_neighbor(self):
        structure = ProteinStructure.from_directions(["R", "R"])
        self.assertIsNone(end_move(structure, 2, (0, 0)))
        self.assertIsNone(end_move(structure, 2, (4, 0)))

    def test_end_move_rejects_single_residue_structure_explicitly(self):
        structure = ProteinStructure(((0, 0),))
        with self.assertRaisesRegex(ValueError, "at least two"):
            end_move(structure, 0, (1, 0))

    def test_corner_move_flips_corner_to_other_square_vertex(self):
        structure = ProteinStructure.from_directions(["R", "U"])
        candidate = corner_move(structure, residue_index=1)
        self.assert_valid_candidate(candidate, 3)
        self.assertEqual(candidate.coordinates, ((0, 0), (0, 1), (1, 1)))

    def test_corner_move_rejects_when_alternate_vertex_is_occupied(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        self.assertIsNone(corner_move(structure, residue_index=1))

    def test_corner_move_requires_internal_residue(self):
        structure = ProteinStructure.from_directions(["R", "U"])
        with self.assertRaisesRegex(ValueError, "internal residue"):
            corner_move(structure, residue_index=0)

    def test_crankshaft_reconfigures_two_residue_segment(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        candidate = crankshaft_move(
            structure, start_index=1, rng=random.Random(3)
        )
        self.assert_valid_candidate(candidate, 4)
        self.assertNotEqual(candidate, structure)
        self.assertEqual(candidate.coordinates[0], structure.coordinates[0])
        self.assertEqual(candidate.coordinates[3], structure.coordinates[3])

    def test_crankshaft_rejects_when_no_alternative_path_exists(self):
        structure = ProteinStructure.from_directions(["R", "R", "R"])
        self.assertIsNone(crankshaft_move(structure, start_index=1))

    def test_invalid_source_is_not_silently_repaired(self):
        invalid = ProteinStructure.from_directions(["R", "L"])
        with self.assertRaisesRegex(StructureError, "invalid source structure"):
            corner_move(invalid, residue_index=1)

    def test_move_statistics_count_attempts_accepts_and_rejects(self):
        statistics = MoveStatistics()
        structure = ProteinStructure.from_directions(["R", "R"])
        accepted = end_move(structure, 2, (1, 1), statistics)
        rejected = end_move(structure, 2, (0, 0), statistics)
        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        self.assertEqual(
            statistics.snapshot(),
            {
                "attempted": 2,
                "accepted": 1,
                "rejected": 1,
                "duplicate_candidates": 0,
                "restricted_operator_opportunities": 0,
                "by_move": {
                    "end": {"attempted": 2, "accepted": 1, "rejected": 1}
                },
            },
        )

    def test_neighborhood_contains_only_unique_valid_structures(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        statistics = MoveStatistics()
        neighborhood = generate_neighborhood(structure, statistics)
        self.assertTrue(neighborhood)
        self.assertEqual(len(neighborhood), len({item.coordinates for item in neighborhood}))
        for candidate in neighborhood:
            self.assertTrue(validate_structure(candidate, expected_length=4).valid)
            self.assertNotEqual(candidate, structure)
        counts = statistics.snapshot()
        self.assertEqual(
            counts["attempted"], counts["accepted"] + counts["rejected"]
        )
        self.assertGreater(counts["attempted"], 0)

    def test_neighborhood_of_single_residue_is_empty(self):
        self.assertEqual(generate_neighborhood(ProteinStructure(((0, 0),))), ())

    def test_restricted_neighborhood_counts_omitted_operator_opportunities(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        statistics = MoveStatistics()
        neighborhood = generate_neighborhood(
            structure, statistics, move_types=("end",)
        )
        self.assertTrue(all(validate_structure(item).valid for item in neighborhood))
        snapshot = statistics.snapshot()
        self.assertEqual(snapshot["restricted_operator_opportunities"], 3)
        self.assertEqual(snapshot["duplicate_candidates"], 1)

    def test_neighborhood_avoids_symmetry_equivalent_states(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        statistics = MoveStatistics()
        neighborhood = generate_neighborhood(structure, statistics)
        keys = [tuple(item.coordinates) for item in neighborhood]
        self.assertEqual(len(keys), len(set(keys)))
        from hp_folding.search_space import symmetry_key

        self.assertNotIn(symmetry_key(structure), {symmetry_key(item) for item in neighborhood})


if __name__ == "__main__":
    unittest.main()
