import random
import unittest

from hp_folding.structure import (
    Direction,
    ProteinStructure,
    StructureError,
    ValidationStatistics,
    generate_random_valid_structure,
    validate_structure,
)


class StructureRepresentationTests(unittest.TestCase):
    def test_coordinates_are_generated_from_relative_directions(self):
        structure = ProteinStructure.from_directions(["R", "U", "R", "D"])
        self.assertEqual(
            structure.coordinates,
            ((0, 0), (1, 0), (1, 1), (2, 1), (2, 0)),
        )
        self.assertEqual(structure.directions, (
            Direction.RIGHT,
            Direction.UP,
            Direction.RIGHT,
            Direction.DOWN,
        ))

    def test_valid_self_avoiding_structure(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        result = validate_structure(structure, expected_length=4)
        self.assertTrue(result.valid)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.collisions, ())

    def test_collision_is_detected(self):
        structure = ProteinStructure.from_directions(["R", "L"])
        result = validate_structure(structure)
        self.assertFalse(result.valid)
        self.assertTrue(any(issue.startswith("collision:") for issue in result.issues))
        self.assertEqual(result.collisions, (((0, 0), (1, 3)),))

    def test_non_unit_consecutive_distance_fails_connectivity(self):
        structure = ProteinStructure(((0, 0), (2, 0)))
        result = validate_structure(structure)
        self.assertFalse(result.valid)
        self.assertTrue(
            any(issue.startswith("connectivity:") for issue in result.issues)
        )

    def test_expected_length_mismatch_is_invalid(self):
        structure = ProteinStructure.from_directions(["R"])
        result = validate_structure(structure, expected_length=3)
        self.assertFalse(result.valid)
        self.assertTrue(any(issue.startswith("length:") for issue in result.issues))

    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(StructureError, "Invalid direction"):
            ProteinStructure.from_directions(["N"])

    def test_invalid_start_coordinate_is_rejected(self):
        with self.assertRaisesRegex(StructureError, "two integers"):
            ProteinStructure.from_directions([], start=(0, 1.5))

    def test_boolean_coordinate_is_not_accepted_as_an_integer(self):
        structure = ProteinStructure(((False, 0), (1, 0)))
        result = validate_structure(structure)
        self.assertFalse(result.valid)
        self.assertTrue(any(issue.startswith("coordinate:") for issue in result.issues))

    def test_text_representation_includes_indices_and_coordinates(self):
        structure = ProteinStructure.from_directions(["U"])
        self.assertEqual(
            str(structure),
            "ProteinStructure(length=2, residues=[1:(0,0), 2:(0,1)])",
        )

    def test_occupancy_mapping_uses_one_based_indices(self):
        structure = ProteinStructure(((0, 0), (1, 0), (0, 0)))
        self.assertEqual(
            structure.coordinate_to_residues(),
            {(0, 0): (1, 3), (1, 0): (2,)},
        )

    def test_random_generation_is_valid_and_reproducible(self):
        first = generate_random_valid_structure(20, rng=random.Random(12))
        second = generate_random_valid_structure(20, rng=random.Random(12))
        self.assertEqual(first, second)
        self.assertTrue(validate_structure(first, expected_length=20).valid)

    def test_random_generation_supports_single_residue(self):
        structure = generate_random_valid_structure(1, rng=random.Random(1))
        self.assertEqual(structure.coordinates, ((0, 0),))
        self.assertTrue(validate_structure(structure, expected_length=1).valid)

    def test_random_generation_rejects_nonpositive_length(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            generate_random_valid_structure(0)

    def test_validation_statistics_track_results_and_issue_categories(self):
        statistics = ValidationStatistics()
        validate_structure(ProteinStructure.from_directions(["R"]), statistics=statistics)
        validate_structure(ProteinStructure.from_directions(["R", "L"]), statistics=statistics)
        self.assertEqual(
            statistics.snapshot(),
            {
                "total": 2,
                "valid": 1,
                "invalid": 1,
                "issue_counts": {"collision": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
