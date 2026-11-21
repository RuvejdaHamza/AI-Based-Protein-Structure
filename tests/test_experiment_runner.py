import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

from hp_folding.evaluation_budget import EvaluationBudget
from hp_folding.experiment_runner import (
    ALGORITHMS,
    BUDGET_STUDY_BUDGETS,
    BUDGET_STUDY_LENGTHS,
    DEFAULT_ROOT_SEEDS,
    ExperimentCase,
    build_budget_study_cases,
    build_experiment_cases,
    derive_seed,
    run_experiments,
    run_budget_sensitivity_study,
)
from hp_folding.genetic_algorithm import GeneticAlgorithmConfig
from hp_folding.hybrid_optimizer import HybridConfig, run_hybrid_optimizer
from hp_folding.simulated_annealing import SimulatedAnnealingConfig


class EvaluationBudgetTests(unittest.TestCase):
    def test_budget_never_exceeds_limit(self):
        budget = EvaluationBudget(2)
        self.assertTrue(budget.try_consume())
        self.assertTrue(budget.try_consume())
        self.assertFalse(budget.try_consume())
        self.assertEqual(budget.evaluations, 2)
        self.assertTrue(budget.exhausted)

    def test_hybrid_shares_budget_between_genetic_and_local_search(self):
        budget = EvaluationBudget(20)
        config = HybridConfig(
            genetic=GeneticAlgorithmConfig(
                population_size=5,
                max_generations=4,
                seed=31415,
            ),
            local_search=SimulatedAnnealingConfig(max_iterations=20),
            local_search_frequency=1,
            local_search_fraction=0.4,
            seed=31415,
        )

        result = run_hybrid_optimizer(
            "HPPHHPPH",
            config,
            evaluation_budget=budget,
        )

        self.assertGreater(result.local_search_calls, 0)
        self.assertGreater(result.local_search_energy_evaluations, 0)
        self.assertEqual(result.energy_evaluations, budget.evaluations)
        self.assertLessEqual(budget.evaluations, 20)


class ExperimentRunnerTests(unittest.TestCase):
    def test_case_matrix_has_synthetic_and_real_inputs_for_all_lengths(self):
        cases = build_experiment_cases()
        self.assertEqual(len(cases), 40)
        self.assertEqual(
            {case.length for case in cases},
            {8, 12, 16, 20, 30},
        )
        self.assertEqual(
            sum(case.sequence_type == "synthetic_hp" for case in cases), 20
        )
        self.assertEqual(
            sum(case.sequence_type == "real_amino_acid" for case in cases), 20
        )
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_budget_study_cases_include_requested_lengths_and_sources(self):
        self.assertEqual(BUDGET_STUDY_BUDGETS, (200, 1_000, 5_000, 20_000))
        cases = build_budget_study_cases()
        self.assertEqual(len(cases), 40)
        self.assertEqual({case.length for case in cases}, set(BUDGET_STUDY_LENGTHS))
        self.assertEqual(
            sum(case.sequence_type == "synthetic_hp" for case in cases), 20
        )
        self.assertEqual(
            sum(case.sequence_type == "real_amino_acid" for case in cases), 20
        )
        for case in cases:
            self.assertEqual(len(case.sequence), case.length)
            self.assertEqual(len(case.hp_sequence), case.length)

    def test_budget_study_smoke_mode_runs_and_exports_actual_results(self):
        with tempfile.TemporaryDirectory() as output_dir:
            runs, summary = run_budget_sensitivity_study(
                output_dir=output_dir,
                smoke_test=True,
            )
            self.assertEqual(len(runs), 2 * 2 * len(ALGORITHMS))
            self.assertEqual(set(runs["budget"]), {20, 40})
            self.assertEqual(set(runs["study_mode"]), {"smoke_test"})
            self.assertEqual(set(runs["algorithm"]), set(ALGORITHMS))
            self.assertTrue((runs["status"] == "completed").all())
            self.assertTrue((runs["energy_evaluations"] <= runs["budget"]).all())
            self.assertTrue(runs["valid_best_structure"].all())
            self.assertEqual(set(summary["budget"]), {20, 40})
            self.assertTrue((Path(output_dir) / "budget_study_runs.csv").is_file())
            for name in (
                "budget_study_summary.csv",
                "budget_study_length_summary.csv",
                "budget_study_case_summary.csv",
            ):
                self.assertTrue((Path(output_dir) / name).is_file())
            plots = (
                Path(output_dir)
                / "plots"
            )
            self.assertTrue((plots / "mean_best_energy_vs_budget.png").is_file())
            self.assertTrue(
                (plots / "mean_best_energy_vs_length_per_budget.png").is_file()
            )
            self.assertTrue((plots / "runtime_vs_length.png").is_file())

    def test_seed_derivation_is_stable_and_algorithm_specific(self):
        first = derive_seed(1101, "syn_H70_N8", "simulated_annealing", 1)
        self.assertEqual(
            first,
            derive_seed(1101, "syn_H70_N8", "simulated_annealing", 1),
        )
        self.assertNotEqual(
            first,
            derive_seed(1101, "syn_H70_N8", "genetic_algorithm", 1),
        )

    def test_runner_uses_shared_evaluation_budget_and_exports_csv(self):
        case = ExperimentCase(
            case_id="smoke_HP8",
            length=8,
            sequence="HPPHHPPH",
            hp_sequence="HPPHHPPH",
            sequence_type="synthetic_hp",
            source="unit-test",
        )
        with tempfile.TemporaryDirectory() as output_dir:
            runs, summary = run_experiments(
                cases=(case,),
                run_count=1,
                root_seeds=(DEFAULT_ROOT_SEEDS[0],),
                energy_budget=30,
                population_size=10,
                output_dir=output_dir,
            )
            self.assertEqual(len(runs), len(ALGORITHMS))
            self.assertEqual(set(runs["algorithm"]), set(ALGORITHMS))
            self.assertTrue((runs["status"] == "completed").all())
            self.assertTrue(runs["initial_energy"].notna().all())
            self.assertTrue(runs["runtime_seconds"].notna().all())
            self.assertTrue(runs["peak_memory_bytes"].notna().all())
            self.assertTrue(runs["generated_candidates"].notna().all())
            self.assertTrue(runs["convergence_evaluations_to_best"].notna().all())
            self.assertTrue((runs["peak_memory_bytes"] >= 0).all())
            self.assertTrue(
                (
                    runs["generated_candidates"]
                    == runs["valid_candidates"] + runs["invalid_candidates"]
                ).all()
            )
            self.assertTrue(runs["final_energy"].notna().all())
            self.assertTrue((runs["energy_budget_used"] <= 30).all())
            self.assertTrue((runs["energy_evaluations"] <= 30).all())
            self.assertTrue(runs["valid_best_structure"].all())
            self.assertFalse(tracemalloc.is_tracing())
            ga_details = json.loads(
                runs.loc[
                    runs["algorithm"] == "genetic_algorithm", "details_json"
                ].iloc[0]
            )
            self.assertIn("initial_population_average_energy", ga_details)
            self.assertEqual(len(summary), len(ALGORITHMS))
            self.assertIn("generated_candidates_mean", summary.columns)
            self.assertIn(
                "convergence_evaluations_to_best_mean", summary.columns
            )
            self.assertTrue(
                (Path(output_dir) / "experiment_runs.csv").is_file()
            )
            self.assertTrue(
                (Path(output_dir) / "experiment_summary.csv").is_file()
            )
            self.assertTrue(
                all(
                    isinstance(json.loads(value), dict)
                    for value in runs["convergence_json"]
                )
            )

    def test_runner_records_failures_and_continues_other_algorithms(self):
        case = ExperimentCase(
            case_id="failure_HP8",
            length=8,
            sequence="HPPHHPPH",
            hp_sequence="HPPHHPPH",
            sequence_type="synthetic_hp",
            source="unit-test",
        )
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "hp_folding.experiment_runner.run_simulated_annealing",
                side_effect=RuntimeError("deliberate test failure"),
            ):
                runs, _ = run_experiments(
                    cases=(case,),
                    run_count=1,
                    root_seeds=(1234,),
                    energy_budget=30,
                    population_size=10,
                    output_dir=output_dir,
                )
        sa_row = runs[runs["algorithm"] == "simulated_annealing"].iloc[0]
        self.assertEqual(sa_row["status"], "failed")
        self.assertEqual(sa_row["error_type"], "RuntimeError")
        self.assertEqual(
            sa_row["error_message"], "deliberate test failure"
        )
        self.assertEqual(
            int((runs["status"] == "completed").sum()), 2
        )
        self.assertFalse(tracemalloc.is_tracing())

    def test_existing_tracemalloc_session_is_not_modified(self):
        tracemalloc.start()
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                with self.assertRaisesRegex(
                    RuntimeError, "Stop existing tracemalloc"
                ):
                    run_experiments(
                        cases=(
                            ExperimentCase(
                                case_id="trace_HP8",
                                length=8,
                                sequence="HPPHHPPH",
                                hp_sequence="HPPHHPPH",
                                sequence_type="synthetic_hp",
                                source="unit-test",
                            ),
                        ),
                        run_count=1,
                        root_seeds=(1234,),
                        energy_budget=20,
                        population_size=10,
                        output_dir=output_dir,
                    )
            self.assertTrue(tracemalloc.is_tracing())
        finally:
            tracemalloc.stop()


if __name__ == "__main__":
    unittest.main()
