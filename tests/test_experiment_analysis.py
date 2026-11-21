import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hp_folding.experiment_analysis import (
    analyze_budget_sensitivity_results,
    analyze_experiment_csv,
    analyze_experiment_results,
)


def _record(
    case_id,
    length,
    algorithm,
    final_energy,
    runtime,
    memory,
    generated,
    valid_candidates,
    invalid_candidates,
    evaluations,
    *,
    status="completed",
    valid_structure=True,
    best_history=None,
    evaluation_history=None,
):
    if best_history is None:
        best_history = (
            [final_energy + 1, final_energy] if final_energy is not None else []
        )
    if evaluation_history is None:
        evaluation_history = [1, evaluations] if evaluations > 0 else []
    return {
        "case_id": case_id,
        "protein_length": length,
        "algorithm": algorithm,
        "status": status,
        "valid_best_structure": valid_structure,
        "final_energy": final_energy,
        "best_energy": final_energy,
        "runtime_seconds": runtime,
        "cpu_seconds": runtime,
        "peak_memory_bytes": memory,
        "generated_candidates": generated,
        "valid_candidates": valid_candidates,
        "invalid_candidates": invalid_candidates,
        "energy_evaluations": evaluations,
        "convergence_json": json.dumps(
            {
                "best_energy_history": best_history,
                "energy_evaluations_history": evaluation_history,
            }
        ),
    }


class ExperimentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.runs = pd.DataFrame(
            [
                _record("case_N8", 8, "simulated_annealing", -2, 1, 100, 10, 8, 2, 5),
                _record("case_N8", 8, "simulated_annealing", -4, 3, 200, 20, 15, 5, 7),
                _record(
                    "case_N8",
                    8,
                    "simulated_annealing",
                    None,
                    0.1,
                    50,
                    0,
                    0,
                    0,
                    0,
                    status="failed",
                    valid_structure=False,
                ),
                _record(
                    "case_N8",
                    8,
                    "genetic_algorithm",
                    -3,
                    2,
                    300,
                    12,
                    9,
                    3,
                    6,
                    evaluation_history=[1, 4],
                ),
                _record("case_N12", 12, "hybrid_ga_sa", -5, 4, 400, 16, 11, 5, 8),
            ]
        )

    def test_summary_statistics_exclude_failed_energy_but_count_failures(self):
        with tempfile.TemporaryDirectory() as output_dir:
            analysis = analyze_experiment_results(self.runs, output_dir)

        sa = analysis.algorithm_summary.set_index("algorithm").loc[
            "simulated_annealing"
        ]
        self.assertEqual(sa["runs"], 3)
        self.assertEqual(sa["completed_runs"], 2)
        self.assertEqual(sa["failed_runs"], 1)
        self.assertAlmostEqual(sa["validity_percentage"], 200 / 3)
        self.assertEqual(sa["mean_final_energy"], -3)
        self.assertEqual(sa["best_final_energy"], -4)
        self.assertEqual(sa["worst_final_energy"], -2)
        self.assertAlmostEqual(sa["std_final_energy"], math.sqrt(2))
        self.assertEqual(sa["mean_energy_evaluations"], 4)
        self.assertAlmostEqual(sa["mean_runtime_seconds"], 4.1 / 3)
        self.assertAlmostEqual(sa["mean_cpu_seconds"], 4.1 / 3)
        self.assertAlmostEqual(sa["mean_peak_memory_bytes"], 350 / 3)
        self.assertAlmostEqual(sa["candidate_validity_percentage"], 23 / 30 * 100)

    def test_analysis_writes_all_requested_plots_and_tables(self):
        with tempfile.TemporaryDirectory() as output_dir:
            analysis = analyze_experiment_results(self.runs, output_dir)
            self.assertEqual(len(analysis.plot_paths), 6)
            self.assertTrue(all(path.is_file() for path in analysis.plot_paths))
            self.assertTrue(
                all(path.stat().st_size > 0 for path in analysis.plot_paths)
            )
            for name in (
                "algorithm_summary.csv",
                "length_algorithm_summary.csv",
                "case_algorithm_comparison.csv",
                "convergence_curves.csv",
            ):
                self.assertTrue((Path(output_dir) / name).is_file())
        self.assertIn("runs_observed", analysis.convergence.columns)
        self.assertTrue((analysis.convergence["runs_observed"] >= 1).all())
        ga_terminal = analysis.convergence.loc[
            (analysis.convergence["algorithm"] == "genetic_algorithm")
            & (analysis.convergence["energy_evaluations"] == 6)
        ].iloc[0]
        self.assertEqual(ga_terminal["median_best_energy"], -3)

    def test_csv_loader_requires_an_existing_recorded_file(self):
        with tempfile.TemporaryDirectory() as output_dir:
            missing_path = Path(output_dir) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                analyze_experiment_csv(missing_path, output_dir)

    def test_malformed_convergence_is_reported_instead_of_skipped(self):
        malformed = self.runs.iloc[[0]].copy()
        malformed.loc[:, "convergence_json"] = '{"best_energy_history":[-1]}'
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "Invalid convergence history"):
                analyze_experiment_results(malformed, output_dir)

    def test_budget_study_uses_best_energy_and_writes_three_plots(self):
        runs = self.runs.copy()
        runs["budget"] = 200
        runs.loc[
            runs["algorithm"].eq("simulated_annealing")
            & runs["status"].eq("completed"),
            "best_energy",
        ] = [-3, -5]
        with tempfile.TemporaryDirectory() as output_dir:
            analysis = analyze_budget_sensitivity_results(runs, output_dir)
            sa = analysis.budget_summary.loc[
                analysis.budget_summary["algorithm"].eq("simulated_annealing")
            ].iloc[0]
            self.assertEqual(sa["mean_best_energy"], -4)
            self.assertEqual(len(analysis.plot_paths), 3)
            self.assertTrue(all(path.is_file() for path in analysis.plot_paths))
            self.assertTrue(
                (Path(output_dir) / "budget_study_summary.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
