import unittest

from hp_folding.sequence_processing import (
    POLAR_AMINO_ACIDS,
    HYDROPHOBIC_AMINO_ACIDS,
    ProcessedSequence,
    SequenceValidationError,
    convert_to_hp,
    process_sequence,
    validate_amino_acid_sequence,
)


class SequenceProcessingTests(unittest.TestCase):
    def test_classification_covers_exactly_twenty_standard_amino_acids(self):
        self.assertEqual(len(HYDROPHOBIC_AMINO_ACIDS), 11)
        self.assertEqual(len(POLAR_AMINO_ACIDS), 9)
        self.assertFalse(HYDROPHOBIC_AMINO_ACIDS & POLAR_AMINO_ACIDS)
        self.assertEqual(
            HYDROPHOBIC_AMINO_ACIDS | POLAR_AMINO_ACIDS,
            set("ACDEFGHIKLMNPQRSTVWY"),
        )

    def test_processes_valid_sequence(self):
        result = process_sequence("ACDE")
        self.assertEqual(
            result,
            ProcessedSequence(
                amino_acid_sequence="ACDE",
                hp_sequence="HHPP",
                length=4,
            ),
        )

    def test_accepts_lowercase_and_ignores_whitespace(self):
        result = process_sequence(" aC\n d ")
        self.assertEqual(result.amino_acid_sequence, "ACD")
        self.assertEqual(result.hp_sequence, "HHP")
        self.assertEqual(result.length, 3)

    def test_supports_different_sequence_lengths(self):
        self.assertEqual(process_sequence("V").length, 1)
        self.assertEqual(process_sequence("ARNDCQEGHILKMFPSTWYV").length, 20)

    def test_conversion_function_validates_and_converts(self):
        self.assertEqual(convert_to_hp("GPH"), "HHP")

    def test_empty_sequence_is_rejected(self):
        for sequence in ("", " \t\n "):
            with self.subTest(sequence=sequence):
                with self.assertRaisesRegex(
                    SequenceValidationError, "must not be empty"
                ):
                    process_sequence(sequence)

    def test_non_standard_symbols_are_reported_with_positions(self):
        with self.assertRaises(SequenceValidationError) as context:
            process_sequence("ACX-B")
        message = str(context.exception)
        self.assertIn("'X' at position 3", message)
        self.assertIn("'-' at position 4", message)
        self.assertIn("20 standard", message)

    def test_non_string_input_is_rejected(self):
        with self.assertRaisesRegex(SequenceValidationError, "must be a string"):
            process_sequence(None)

    def test_validation_returns_normalized_sequence(self):
        self.assertEqual(validate_amino_acid_sequence(" rN "), "RN")


if __name__ == "__main__":
    unittest.main()
