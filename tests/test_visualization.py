import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hp_folding.structure import ProteinStructure
from hp_folding.visualization import (
    find_hh_contacts,
    plot_algorithm_comparison,
    plot_protein_structure,
    select_algorithm_runs,
)


def _run(algorithm, energy, directions, replicate=1):
    return {
        "case_id": "test_case",
        "algorithm": algorithm,
        "replicate": replicate,
        "sequence": "ACGT",
        "hp_sequence": "HHHH",
        "status": "completed",
        "valid_best_structure": True,
        "best_energy": energy,
        "details_json": json.dumps({"best_directions": directions}),
    }


class VisualizationTests(unittest.TestCase):
    def test_hh_contacts_are_nonconsecutive_and_counted_once(self):
        structure = ProteinStructure.from_directions(["R", "U", "L"])
        self.assertEqual(find_hh_contacts("HHHH", structure), ((1, 4),))
        self.assertEqual(find_hh_contacts("HHHP", structure), ())

    def test_single_structure_plot_uses_equal_lattice_geometry_and_indices(self):
        import matplotlib.pyplot as plt

        structure = ProteinStructure.from_directions(["R", "U", "L"])
        figure, axis = plt.subplots()
        try:
            result = plot_protein_structure(
                "ACGT",
                "HHHH",
                structure,
                ax=axis,
            )
            self.assertEqual(result.energy, -1)
            self.assertEqual(result.contact_count, 1)
            self.assertEqual(axis.get_aspect(), 1.0)
            self.assertEqual(len(axis.texts), 4)
            self.assertEqual(axis.texts[0].get_text(), "1\nA")
        finally:
            plt.close(figure)

    def test_comparison_uses_saved_directions_and_selects_best_run_per_algorithm(self):
        import matplotlib.pyplot as plt

        runs = pd.DataFrame(
            [
                _run("simulated_annealing", 0, ["R", "R", "R"], replicate=1),
                _run("simulated_annealing", -1, ["R", "U", "L"], replicate=2),
                _run("genetic_algorithm", -1, ["R", "U", "L"]),
                _run("hybrid_ga_sa", -1, ["R", "U", "L"]),
            ]
        )
        selected = select_algorithm_runs(runs, "test_case")
        sa = selected.loc[
            selected["algorithm"] == "simulated_annealing"
        ].iloc[0]
        self.assertEqual(sa["replicate"], 2)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "comparison.png"
            figure, selection = plot_algorithm_comparison(
                runs,
                "test_case",
                output_path=output_path,
            )
            try:
                self.assertTrue(output_path.is_file())
                self.assertTrue(selection["recomputed_energy"].eq(-1).all())
                self.assertEqual(len(figure.axes), 3)
                self.assertTrue(
                    all(axis.get_aspect() == 1.0 for axis in figure.axes)
                )
                self.assertIn("E = -1", figure.axes[0].get_title())
                self.assertIn("H-H contacts = 1", figure.axes[0].get_title())
            finally:
                plt.close(figure)

    def test_comparison_rejects_missing_algorithm_or_energy_mismatch(self):
        runs = pd.DataFrame(
            [
                _run("simulated_annealing", -1, ["R", "U"]),
                _run("genetic_algorithm", -1, ["R", "U"]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "hybrid_ga_sa"):
            select_algorithm_runs(runs, "test_case")

        complete = pd.DataFrame(
            [
                _run("simulated_annealing", -1, ["R", "R", "R"]),
                _run("genetic_algorithm", -1, ["R", "U", "L"]),
                _run("hybrid_ga_sa", -1, ["R", "U", "L"]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not match recorded"):
            plot_algorithm_comparison(complete, "test_case")


if __name__ == "__main__":
    unittest.main()
