import unittest

from hp_folding.energy import EnergyResult, calculate_energy
from hp_folding.structure import ProteinStructure


class EnergyTests(unittest.TestCase):
    def test_square_loop_counts_single_endpoint_contact_once(self):
        # Residues 1 and 4 are adjacent in space but not in sequence.
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        self.assertEqual(
            calculate_energy("HHHH", structure),
            EnergyResult(energy=-1.0, contact_count=1),
        )

    def test_two_row_six_residue_chain_has_two_hh_contacts(self):
        # Contacts: (1, 6) and (2, 5). Pair (3, 4) is a chain bond.
        structure = ProteinStructure.from_directions(["R", "R", "U", "L", "L"])
        result = calculate_energy("HHHHHH", structure, epsilon=2.5)
        self.assertEqual(result.contact_count, 2)
        self.assertEqual(result.energy, -5.0)

    def test_polar_residues_do_not_contribute_to_contacts(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        self.assertEqual(
            calculate_energy("HPPH", structure),
            EnergyResult(energy=-1.0, contact_count=1),
        )
        self.assertEqual(
            calculate_energy("HPHP", structure),
            EnergyResult(energy=0.0, contact_count=0),
        )

    def test_consecutive_hydrophobic_neighbors_are_not_contacts(self):
        straight_chain = ProteinStructure.from_directions(["R", "R", "R"])
        self.assertEqual(
            calculate_energy("HHHH", straight_chain),
            EnergyResult(energy=0.0, contact_count=0),
        )

    def test_invalid_structure_is_rejected(self):
        overlapping = ProteinStructure.from_directions(["R", "L"])
        with self.assertRaisesRegex(ValueError, "invalid structure.*collision"):
            calculate_energy("HHH", overlapping)

    def test_structure_and_sequence_lengths_must_match(self):
        structure = ProteinStructure.from_directions(["R"])
        with self.assertRaisesRegex(ValueError, "expected 3 residues, got 2"):
            calculate_energy("HHH", structure)

    def test_invalid_hp_labels_are_rejected(self):
        structure = ProteinStructure.from_directions(["R"])
        with self.assertRaisesRegex(ValueError, "invalid label"):
            calculate_energy("HX", structure)

    def test_empty_hp_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            calculate_energy("", ProteinStructure(()))

    def test_epsilon_must_be_positive_and_finite(self):
        structure = ProteinStructure.from_directions(["R"])
        for epsilon in (0, -1, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    calculate_energy("HP", structure, epsilon=epsilon)


if __name__ == "__main__":
    unittest.main()
