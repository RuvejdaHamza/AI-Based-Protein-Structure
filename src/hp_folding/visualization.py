"""Publication-oriented lattice plots for recorded HP folding structures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .energy import EnergyResult, calculate_energy
from .structure import ProteinStructure, validate_structure


_ALGORITHMS = {
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
    "hybrid_ga_sa": "Hybrid GA + SA",
}
_HYDROPHOBIC_COLOR = "#D55E00"
_POLAR_COLOR = "#0072B2"
_CONTACT_COLOR = "#CC79A7"
_REQUIRED_RUN_COLUMNS = {
    "case_id",
    "algorithm",
    "replicate",
    "sequence",
    "hp_sequence",
    "status",
    "valid_best_structure",
    "best_energy",
    "details_json",
}


def _is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    if value is None or pd.isna(value):
        return False
    raise ValueError(f"Unrecognized valid_best_structure value: {value!r}")


def _check_run_columns(runs: pd.DataFrame) -> None:
    if not isinstance(runs, pd.DataFrame):
        raise TypeError("runs must be a pandas DataFrame.")
    missing = sorted(_REQUIRED_RUN_COLUMNS - set(runs.columns))
    if missing:
        raise ValueError(
            "Run data is missing required columns: " + ", ".join(missing)
        )


def select_algorithm_runs(
    runs: pd.DataFrame,
    case_id: str,
    replicate: int | None = None,
) -> pd.DataFrame:
    """Select actual valid run records for a three-algorithm comparison.

    With no replicate specified, choose each algorithm's lowest-energy
    recorded run for the case. Otherwise, select the requested replicate for
    each algorithm, requiring it to exist and be valid.
    """
    _check_run_columns(runs)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string.")
    if replicate is not None and (type(replicate) is not int or replicate < 1):
        raise ValueError("replicate must be a positive integer or None.")

    case_runs = runs.loc[runs["case_id"].eq(case_id)].copy()
    if case_runs.empty:
        raise ValueError(f"No recorded runs were found for case {case_id!r}.")
    for column in ("best_energy", "replicate"):
        original = case_runs[column]
        converted = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & converted.isna()
        if malformed.any():
            raise ValueError(f"Column {column!r} contains non-numeric values.")
        case_runs[column] = converted
    valid_mask = (
        case_runs["status"].eq("completed")
        & case_runs["valid_best_structure"].map(_is_true)
        & case_runs["best_energy"].notna()
    )
    case_runs = case_runs.loc[valid_mask].copy()
    if replicate is not None:
        case_runs = case_runs.loc[case_runs["replicate"].eq(replicate)]
    else:
        case_runs = (
            case_runs.sort_values(
                ["algorithm", "best_energy", "replicate"],
                ascending=[True, True, True],
                kind="stable",
            )
            .groupby("algorithm", as_index=False, sort=True)
            .first()
        )

    selected_algorithms = set(case_runs["algorithm"])
    missing_algorithms = sorted(set(_ALGORITHMS) - selected_algorithms)
    if missing_algorithms:
        raise ValueError(
            f"Case {case_id!r} has no valid recorded run for: "
            + ", ".join(missing_algorithms)
        )
    return case_runs.sort_values("algorithm", kind="stable").reset_index(drop=True)


def _structure_from_run(row: Any) -> tuple[ProteinStructure, EnergyResult]:
    try:
        details = json.loads(row.details_json)
        directions = details["best_directions"]
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Run {row.algorithm!r}, replicate {row.replicate!r} does not "
            "contain saved best-direction data."
        ) from error
    if not isinstance(directions, list) or not all(
        isinstance(direction, str) for direction in directions
    ):
        raise ValueError("best_directions must be a list of direction strings.")

    try:
        structure = ProteinStructure.from_directions(directions)
        validation = validate_structure(
            structure, expected_length=len(row.hp_sequence)
        )
        if not validation.valid:
            raise ValueError("; ".join(validation.issues))
        energy = calculate_energy(row.hp_sequence, structure)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Saved structure for {row.algorithm!r}, replicate "
            f"{row.replicate!r} is invalid: {error}"
        ) from error

    recorded_energy = float(row.best_energy)
    if not math.isclose(energy.energy, recorded_energy, abs_tol=1e-12):
        raise ValueError(
            f"Recomputed energy {energy.energy} does not match recorded best "
            f"energy {recorded_energy} for {row.algorithm!r}, replicate "
            f"{row.replicate!r}."
        )
    return structure, energy


def find_hh_contacts(
    hp_sequence: str,
    structure: ProteinStructure,
) -> tuple[tuple[int, int], ...]:
    """Return one-based residue-index pairs for non-consecutive H-H contacts."""
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    hp_sequence = "".join(hp_sequence.split()).upper()
    calculate_energy(hp_sequence, structure)
    coordinate_indices = {
        coordinate: index
        for index, coordinate in enumerate(structure.coordinates)
    }
    contacts = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for index, (x, y) in enumerate(structure.coordinates):
        if hp_sequence[index] != "H":
            continue
        for dx, dy in neighbors:
            other_index = coordinate_indices.get((x + dx, y + dy))
            if (
                other_index is not None
                and other_index > index + 1
                and hp_sequence[other_index] == "H"
            ):
                contacts.append((index + 1, other_index + 1))
    return tuple(contacts)


def _legend_handles() -> tuple[Patch | Line2D, ...]:
    return (
        Patch(
            facecolor=_HYDROPHOBIC_COLOR,
            edgecolor="black",
            label="Hydrophobic (H)",
        ),
        Patch(
            facecolor=_POLAR_COLOR,
            edgecolor="black",
            label="Polar (P)",
        ),
        Line2D([0], [0], color="#666666", lw=1.8, label="Chain bond"),
        Line2D(
            [0],
            [0],
            color=_CONTACT_COLOR,
            lw=2,
            linestyle="--",
            label="Non-consecutive H-H contact",
        ),
    )


def plot_protein_structure(
    sequence: str,
    hp_sequence: str,
    structure: ProteinStructure,
    *,
    ax: Axes,
    epsilon: float = 1.0,
    limits: tuple[float, float, float, float] | None = None,
) -> EnergyResult:
    """Draw a validated structure on an equal-scale square-lattice grid."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string.")
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    sequence = "".join(sequence.split()).upper()
    hp_sequence = "".join(hp_sequence.split()).upper()
    energy = calculate_energy(hp_sequence, structure, epsilon)
    contacts = find_hh_contacts(hp_sequence, structure)
    if len(sequence) != structure.length:
        raise ValueError("sequence and structure lengths must match.")

    coordinates = np.asarray(structure.coordinates, dtype=float)
    for first, second in zip(coordinates, coordinates[1:]):
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color="#666666",
            linewidth=1.8,
            zorder=1,
        )

    for first_index, second_index in contacts:
        first = coordinates[first_index - 1]
        second = coordinates[second_index - 1]
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color=_CONTACT_COLOR,
            linestyle=(0, (3, 2)),
            linewidth=2.2,
            zorder=2,
        )

    colors = [
        _HYDROPHOBIC_COLOR if residue == "H" else _POLAR_COLOR
        for residue in hp_sequence
    ]
    marker_size = max(250, min(600, 3500 / max(1, structure.length)))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=marker_size,
        c=colors,
        edgecolors="black",
        linewidths=0.9,
        zorder=3,
    )
    label_size = 8 if structure.length <= 16 else 6
    for index, ((x, y), residue) in enumerate(
        zip(structure.coordinates, sequence), start=1
    ):
        ax.annotate(
            f"{index}\n{residue}",
            (x, y),
            ha="center",
            va="center",
            color="white",
            fontsize=label_size,
            fontweight="bold",
            zorder=4,
        )

    if limits is None:
        min_x, min_y = coordinates.min(axis=0)
        max_x, max_y = coordinates.max(axis=0)
        limits = (min_x - 1, max_x + 1, min_y - 1, max_y + 1)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    x_ticks = np.arange(math.floor(limits[0]), math.ceil(limits[1]) + 1)
    y_ticks = np.arange(math.floor(limits[2]), math.ceil(limits[3]) + 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, color="#D9DDE2", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Lattice x-coordinate")
    ax.set_ylabel("Lattice y-coordinate")
    ax.set_title(
        f"E = {energy.energy:g}    H-H contacts = {energy.contact_count}",
        fontsize=10,
    )
    return energy


def plot_algorithm_comparison(
    runs: pd.DataFrame,
    case_id: str,
    *,
    replicate: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Compare actual saved best structures for all algorithms on one case.

    By default, each algorithm's best valid run across replicates is selected.
    Set ``replicate`` to compare the recorded structures from that replicate.
    """
    selected = select_algorithm_runs(runs, case_id, replicate)
    structures_and_energies = [
        _structure_from_run(row) for row in selected.itertuples(index=False)
    ]
    sequences = selected["sequence"].astype(str).unique()
    hp_sequences = selected["hp_sequence"].astype(str).unique()
    if len(sequences) != 1 or len(hp_sequences) != 1:
        raise ValueError(
            f"Selected runs for {case_id!r} do not share the same sequence."
        )
    sequence = sequences[0]
    hp_sequence = hp_sequences[0]

    all_coordinates = np.asarray(
        [
            coordinate
            for structure, _ in structures_and_energies
            for coordinate in structure.coordinates
        ],
        dtype=float,
    )
    min_x, min_y = all_coordinates.min(axis=0)
    max_x, max_y = all_coordinates.max(axis=0)
    limits = (min_x - 1, max_x + 1, min_y - 1, max_y + 1)

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
        }
    ):
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(5.0 * len(selected), 5.2),
            squeeze=False,
        )
        for ax, row, (structure, energy) in zip(
            axes[0], selected.itertuples(index=False), structures_and_energies
        ):
            plot_protein_structure(
                sequence,
                hp_sequence,
                structure,
                ax=ax,
                limits=limits,
            )
            label = _ALGORITHMS.get(row.algorithm, row.algorithm)
            ax.set_title(
                f"{label}\nreplicate {int(row.replicate)}  |  "
                f"E = {energy.energy:g}  |  "
                f"H-H contacts = {energy.contact_count}",
                fontsize=10,
            )
        selection_label = (
            "best valid recorded run per algorithm"
            if replicate is None
            else f"recorded replicate {replicate}"
        )
        fig.suptitle(
            f"{case_id}: actual algorithm outputs\n{selection_label}",
            y=0.98,
        )
        fig.legend(
            handles=_legend_handles(),
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.90))
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=300, bbox_inches="tight")

    selection = selected[
        ["case_id", "algorithm", "replicate", "best_energy", "hp_sequence"]
    ].copy()
    selection["algorithm_label"] = selection["algorithm"].map(
        lambda value: _ALGORITHMS.get(value, value)
    )
    selection["recomputed_energy"] = [
        result.energy for _, result in structures_and_energies
    ]
    selection["hh_contact_count"] = [
        result.contact_count for _, result in structures_and_energies
    ]
    return fig, selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot actual saved HP-folding structures from run CSV data."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="results/experiment_runs.csv",
        help="Run-level CSV exported by hp_folding.experiment_runner.",
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="Case to compare, for example real_HBB_N20.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        help="Compare this recorded replicate instead of each algorithm's best run.",
    )
    parser.add_argument(
        "--output",
        help="PNG path; defaults to results/visualizations/<case-id>_comparison.png.",
    )
    args = parser.parse_args()

    source = Path(args.input_csv)
    if not source.is_file():
        raise FileNotFoundError(f"Run-level CSV does not exist: {source}")
    runs = pd.read_csv(source)
    output = args.output or str(
        Path("results")
        / "visualizations"
        / f"{args.case_id}_comparison.png"
    )
    figure, selection = plot_algorithm_comparison(
        runs,
        args.case_id,
        replicate=args.replicate,
        output_path=output,
    )
    plt.close(figure)
    print(selection.to_string(index=False))
    print(f"\nFigure saved to: {output}")


if __name__ == "__main__":
    main()
