import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from hp_folding.cli import (
    load_constraints,
    main,
    parse_arguments,
    run_algorithm,
)
from hp_folding.local_moves import MoveStatistics, generate_neighborhood
from hp_folding.sequence_processing import process_sequence
from hp_folding.structure import ProteinStructure, StructureConstraints, validate_structure


class CliTests(unittest.TestCase):
    def test_argument_parsing_defaults_and_options(self):
        arguments = parse_arguments(
            [
                "--sequence",
                "ACDE",
                "--algorithm",
                "hybrid",
                "--max-evaluations",
                "17",
                "--time-limit",
                "2.5",
                "--seed",
                "9",
                "--hh-energy",
                "-2.0",
            ]
        )
        self.assertEqual(arguments.sequence, "ACDE")
        self.assertEqual(arguments.algorithm, "hybrid")
        self.assertEqual(arguments.max_evaluations, 17)
        self.assertEqual(arguments.time_limit, 2.5)
        self.assertEqual(arguments.seed, 9)
        self.assertEqual(arguments.hh_energy, -2.0)

        defaults = parse_arguments(["--sequence", "ACDE"])
        self.assertEqual(defaults.algorithm, "all")
        self.assertEqual(defaults.max_evaluations, 200)
        self.assertEqual(defaults.hh_energy, -1.0)

    def test_argument_parsing_rejects_nonpositive_limits_and_nonnegative_hh_energy(self):
        for option, value in (
            ("--max-evaluations", "0"),
            ("--time-limit", "nan"),
            ("--hh-energy", "1.0"),
        ):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_arguments(["--sequence", "ACDE", option, value])

    def test_json_constraints_are_loaded_and_validated(self):
        document = {
            "fixed_contacts": [[0, 3]],
            "forbidden_positions": [[3, 0]],
            "fixed_directions": {"0": "R"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "constraints.json"
            path.write_text(json.dumps(document), encoding="utf-8-sig")
            constraints = load_constraints(path, 4)

        structure = ProteinStructure.from_directions(["R", "U", "L"])
        self.assertIsNotNone(constraints)
        self.assertTrue(
            validate_structure(
                structure, expected_length=4, constraints=constraints
            ).valid
        )

    def test_all_optimizers_respect_constraints(self):
        sequence = process_sequence("ACGI")
        constraints = StructureConstraints(
            fixed_contacts=((0, 3),),
            forbidden_positions=frozenset({(3, 0)}),
            fixed_directions={0: "R"},
        )
        for algorithm in ("sa", "ga", "hybrid"):
            with self.subTest(algorithm=algorithm):
                record = run_algorithm(
                    algorithm,
                    sequence,
                    max_evaluations=12,
                    seed=10,
                    hh_energy=-1.5,
                    constraints=constraints,
                )
                self.assertEqual(record.status, "completed")
                self.assertIsNotNone(record.result)
                self.assertTrue(
                    validate_structure(
                        record.result.best_structure,
                        expected_length=sequence.length,
                        constraints=constraints,
                    ).valid
                )
                self.assertEqual(record.result.best_energy, -1.5)

    def test_constraints_reject_invalid_structures_and_local_moves(self):
        constraints = StructureConstraints(
            fixed_directions={0: "R"},
        )
        invalid_fixed_direction = ProteinStructure.from_directions(["U", "R"])
        self.assertFalse(
            validate_structure(
                invalid_fixed_direction, constraints=constraints
            ).valid
        )
        invalid_forbidden_position = ProteinStructure.from_directions(["R", "R"])
        self.assertFalse(
            validate_structure(
                invalid_forbidden_position,
                constraints=StructureConstraints(
                    forbidden_positions=frozenset({(2, 0)})
                ),
            ).valid
        )

        source = ProteinStructure.from_directions(["R", "R"])
        statistics = MoveStatistics()
        neighborhood = generate_neighborhood(
            source, statistics, constraints=constraints
        )
        self.assertTrue(neighborhood)
        self.assertTrue(
            all(
                validate_structure(candidate, constraints=constraints).valid
                for candidate in neighborhood
            )
        )
        self.assertGreater(statistics.snapshot()["rejected"], 0)

    def test_impossible_constraint_file_is_reported(self):
        document = {"fixed_contacts": [[0, 1]]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "constraints.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-consecutive"):
                load_constraints(path, 4)

    def test_each_algorithm_stops_cleanly_when_time_limit_is_already_hit(self):
        sequence = process_sequence("ACDE")
        constraints = StructureConstraints()
        for algorithm in ("sa", "ga", "hybrid"):
            with self.subTest(algorithm=algorithm):
                record = run_algorithm(
                    algorithm,
                    sequence,
                    max_evaluations=5,
                    time_limit=1e-12,
                    seed=3,
                    constraints=constraints,
                )
                self.assertEqual(record.status, "time_limit")
                self.assertIsNone(record.result)
                self.assertGreaterEqual(record.runtime_seconds, 0)
                self.assertGreaterEqual(record.peak_memory_bytes, 0)

    def test_all_mode_prints_comparison_and_structure(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "--sequence",
                    "ACDE",
                    "--algorithm",
                    "all",
                    "--max-evaluations",
                    "5",
                    "--seed",
                    "10",
                ]
            )
        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("=== Algorithm comparison ===", text)
        self.assertIn("Coordinates:", text)
        self.assertIn("ASCII fold:", text)
        self.assertIn("Energy evaluations:", text)


if __name__ == "__main__":
    unittest.main()
