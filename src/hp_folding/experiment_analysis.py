"""Statistical analysis and plots for recorded protein-folding experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "case_id",
    "protein_length",
    "algorithm",
    "status",
    "valid_best_structure",
    "final_energy",
    "best_energy",
    "runtime_seconds",
    "cpu_seconds",
    "peak_memory_bytes",
    "generated_candidates",
    "valid_candidates",
    "invalid_candidates",
    "energy_evaluations",
    "convergence_json",
}
_NUMERIC_COLUMNS = (
    "protein_length",
    "final_energy",
    "best_energy",
    "runtime_seconds",
    "cpu_seconds",
    "peak_memory_bytes",
    "generated_candidates",
    "valid_candidates",
    "invalid_candidates",
    "energy_evaluations",
)
_INTEGER_COLUMNS = (
    "protein_length",
    "peak_memory_bytes",
    "generated_candidates",
    "valid_candidates",
    "invalid_candidates",
    "energy_evaluations",
)


@dataclass(frozen=True)
class ExperimentAnalysis:
    """Analysis tables and generated plot paths."""

    algorithm_summary: pd.DataFrame
    length_summary: pd.DataFrame
    case_comparison: pd.DataFrame
    convergence: pd.DataFrame
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BudgetStudyAnalysis:
    """Budget-level and length-level summaries with sensitivity plots."""

    budget_summary: pd.DataFrame
    length_summary: pd.DataFrame
    case_summary: pd.DataFrame
    plot_paths: tuple[Path, ...]


def _parse_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Unrecognized valid_best_structure value: {value!r}")


def _prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(runs, pd.DataFrame):
        raise TypeError("runs must be a pandas DataFrame.")
    if runs.empty:
        raise ValueError("No experiment run records were provided.")
    missing = sorted(_REQUIRED_COLUMNS - set(runs.columns))
    if missing:
        raise ValueError(
            "Experiment results are missing required columns: "
            + ", ".join(missing)
        )

    prepared = runs.copy()
    for column in _NUMERIC_COLUMNS:
        original = prepared[column]
        converted = pd.to_numeric(original, errors="coerce")
        malformed = original.notna() & converted.isna()
        if malformed.any():
            raise ValueError(f"Column {column!r} contains non-numeric values.")
        prepared[column] = converted

    for column in _INTEGER_COLUMNS:
        observed = prepared[column].dropna()
        if (observed % 1 != 0).any():
            raise ValueError(f"Column {column!r} must contain whole numbers.")
    if prepared["protein_length"].isna().any():
        raise ValueError("protein_length must be present for every run.")
    if (prepared["protein_length"] < 1).any():
        raise ValueError("protein_length must be positive.")
    prepared["protein_length"] = prepared["protein_length"].astype(int)
    prepared["status"] = prepared["status"].astype(str)
    prepared["valid_best_structure"] = prepared[
        "valid_best_structure"
    ].map(_parse_boolean)
    prepared["valid_result"] = (
        prepared["status"].eq("completed")
        & prepared["valid_best_structure"]
        & prepared["final_energy"].notna()
    )

    nonnegative_columns = (
        "runtime_seconds",
        "cpu_seconds",
        "peak_memory_bytes",
        "generated_candidates",
        "valid_candidates",
        "invalid_candidates",
        "energy_evaluations",
    )
    for column in nonnegative_columns:
        observed = prepared[column].dropna()
        if (observed < 0).any():
            raise ValueError(f"Column {column!r} cannot contain negative values.")

    completed = prepared["status"].eq("completed")
    required_outcomes = (
        "final_energy",
        "best_energy",
        "runtime_seconds",
        "cpu_seconds",
        "peak_memory_bytes",
        "generated_candidates",
        "valid_candidates",
        "invalid_candidates",
        "energy_evaluations",
    )
    incomplete = prepared.loc[completed, list(required_outcomes)].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            "Completed runs must contain energy, timing, memory, and candidate "
            "measurements."
        )
    completed_counts = prepared.loc[completed]
    if (
        completed_counts["generated_candidates"]
        != completed_counts["valid_candidates"]
        + completed_counts["invalid_candidates"]
    ).any():
        raise ValueError(
            "Completed-run generated candidates must equal valid plus invalid "
            "candidate counts."
        )
    return prepared


def _summary_group(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    valid = group.loc[group["valid_result"]]
    generated = group["generated_candidates"].sum(min_count=1)
    valid_candidates = group["valid_candidates"].sum(min_count=1)
    candidate_validity = (
        float(valid_candidates / generated * 100)
        if pd.notna(generated) and generated > 0 and pd.notna(valid_candidates)
        else np.nan
    )

    return {
        **keys,
        "runs": int(len(group)),
        "completed_runs": int(group["status"].eq("completed").sum()),
        "failed_runs": int(group["status"].ne("completed").sum()),
        "valid_structure_runs": int(group["valid_result"].sum()),
        "validity_percentage": float(group["valid_result"].mean() * 100),
        "mean_final_energy": valid["final_energy"].mean(),
        "std_final_energy": valid["final_energy"].std(),
        "best_final_energy": valid["final_energy"].min(),
        "worst_final_energy": valid["final_energy"].max(),
        "mean_runtime_seconds": group["runtime_seconds"].mean(),
        "median_runtime_seconds": group["runtime_seconds"].median(),
        "max_runtime_seconds": group["runtime_seconds"].max(),
        "mean_cpu_seconds": group["cpu_seconds"].mean(),
        "mean_peak_memory_bytes": group["peak_memory_bytes"].mean(),
        "mean_peak_memory_mib": (
            group["peak_memory_bytes"].mean() / (1024**2)
        ),
        "mean_generated_candidates": group["generated_candidates"].mean(),
        "mean_valid_candidates": group["valid_candidates"].mean(),
        "mean_invalid_candidates": group["invalid_candidates"].mean(),
        "mean_energy_evaluations": group["energy_evaluations"].mean(),
        "total_energy_evaluations": group["energy_evaluations"].sum(min_count=1),
        "candidate_validity_percentage": candidate_validity,
    }


def _summarize(
    runs: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key, group in runs.groupby(list(group_columns), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        rows.append(
            _summary_group(
                group,
                dict(zip(group_columns, key_values)),
            )
        )
    return pd.DataFrame.from_records(rows)


def _convergence_runs(runs: pd.DataFrame) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    valid_runs = runs.loc[runs["valid_result"]]
    for row in valid_runs.itertuples(index=False):
        try:
            convergence = json.loads(row.convergence_json)
            energies = np.asarray(
                convergence["best_energy_history"], dtype=float
            )
            raw_evaluations = convergence["energy_evaluations_history"]
            if not isinstance(raw_evaluations, list) or any(
                type(value) is not int or value < 0
                for value in raw_evaluations
            ):
                raise ValueError("Evaluation checkpoints must be non-negative integers.")
            evaluations = np.asarray(raw_evaluations, dtype=np.int64)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid convergence history for case {row.case_id!r}, "
                f"algorithm {row.algorithm!r}."
            ) from error

        if (
            energies.ndim != 1
            or evaluations.ndim != 1
            or energies.size == 0
            or energies.size != evaluations.size
            or not np.isfinite(energies).all()
            or (evaluations < 0).any()
            or (np.diff(evaluations) < 0).any()
        ):
            raise ValueError(
                f"Inconsistent convergence checkpoints for case "
                f"{row.case_id!r}, algorithm {row.algorithm!r}."
            )
        if (np.diff(energies) > 1e-12).any():
            raise ValueError(
                f"Best-energy history increases for case {row.case_id!r}, "
                f"algorithm {row.algorithm!r}."
            )

        terminal_evaluations = int(row.energy_evaluations)
        if terminal_evaluations < evaluations[-1]:
            raise ValueError(
                f"Convergence history exceeds the evaluation count for case "
                f"{row.case_id!r}, algorithm {row.algorithm!r}."
            )
        terminal_best = float(row.best_energy)
        if not np.isfinite(terminal_best) or terminal_best > energies[-1] + 1e-12:
            raise ValueError(
                f"Final best energy is inconsistent with convergence history "
                f"for case {row.case_id!r}, algorithm {row.algorithm!r}."
            )
        if terminal_evaluations > evaluations[-1]:
            evaluations = np.append(evaluations, terminal_evaluations)
            energies = np.append(energies, terminal_best)
        elif not np.isclose(terminal_best, energies[-1], atol=1e-12):
            raise ValueError(
                f"Final best energy does not match the final checkpoint for "
                f"case {row.case_id!r}, algorithm {row.algorithm!r}."
            )

        # Collapse repeated evaluation counts to the best value observed there.
        unique_evaluations = np.unique(evaluations)
        best_at_checkpoint = np.asarray(
            [
                energies[evaluations == evaluation].min()
                for evaluation in unique_evaluations
            ],
            dtype=float,
        )
        histories.append(
            {
                "algorithm": row.algorithm,
                "case_id": row.case_id,
                "evaluations": unique_evaluations,
                "energies": best_at_checkpoint,
            }
        )
    return histories


def _convergence_table(histories: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    algorithms = sorted({str(history["algorithm"]) for history in histories})
    for algorithm in algorithms:
        algorithm_histories = [
            history
            for history in histories
            if history["algorithm"] == algorithm
        ]
        checkpoints = np.unique(
            np.concatenate(
                [history["evaluations"] for history in algorithm_histories]
            )
        )
        for checkpoint in checkpoints:
            observed = []
            for history in algorithm_histories:
                evaluations = history["evaluations"]
                if checkpoint < evaluations[0] or checkpoint > evaluations[-1]:
                    continue
                index = np.searchsorted(evaluations, checkpoint, side="right") - 1
                observed.append(history["energies"][index])
            if observed:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "energy_evaluations": int(checkpoint),
                        "median_best_energy": float(np.median(observed)),
                        "mean_best_energy": float(np.mean(observed)),
                        "runs_observed": len(observed),
                    }
                )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "algorithm",
            "energy_evaluations",
            "median_best_energy",
            "mean_best_energy",
            "runs_observed",
        ),
    )


def _save_empty_plot(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()


def _save_length_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    scale: float = 1.0,
    error_metric: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    if not summary.empty:
        for algorithm, group in summary.groupby("algorithm", sort=True):
            ordered = group.sort_values("protein_length")
            values = ordered[metric].to_numpy(dtype=float) / scale
            if np.isfinite(values).any():
                if error_metric is not None:
                    errors = ordered[error_metric].to_numpy(dtype=float) / scale
                else:
                    errors = np.asarray([], dtype=float)
                if errors.size and np.isfinite(errors).all():
                    ax.errorbar(
                        ordered["protein_length"],
                        values,
                        yerr=errors,
                        marker="o",
                        capsize=3,
                        label=str(algorithm),
                    )
                else:
                    ax.plot(
                        ordered["protein_length"],
                        values,
                        marker="o",
                        label=str(algorithm),
                    )
                has_data = True
    if has_data:
        ax.set_xlabel("Protein length (residues)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)
    else:
        _save_empty_plot(ax, "No valid recorded measurements")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_convergence_plot(
    convergence: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if convergence.empty:
        _save_empty_plot(ax, "No valid convergence histories")
    else:
        for algorithm, group in convergence.groupby("algorithm", sort=True):
            group = group.sort_values("energy_evaluations")
            ax.step(
                group["energy_evaluations"],
                group["median_best_energy"],
                where="post",
                label=str(algorithm),
            )
        ax.set_xlabel("Objective energy evaluations")
        ax.set_ylabel("Median best-so-far energy")
        ax.legend()
        ax.grid(True, alpha=0.25)
    ax.set_title("Convergence by objective-evaluation checkpoint")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_candidate_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if summary.empty or not summary[
        ["mean_valid_candidates", "mean_invalid_candidates"]
    ].notna().any().any():
        _save_empty_plot(ax, "No candidate-count measurements")
    else:
        labels = summary["algorithm"].astype(str).tolist()
        x = np.arange(len(labels), dtype=float)
        width = 0.38
        ax.bar(
            x - width / 2,
            summary["mean_valid_candidates"].to_numpy(dtype=float),
            width,
            label="Valid proposals",
        )
        ax.bar(
            x + width / 2,
            summary["mean_invalid_candidates"].to_numpy(dtype=float),
            width,
            label="Invalid proposals",
        )
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Mean candidate count per run")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    ax.set_title("Valid versus invalid candidate proposals")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_case_comparison(case_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, case_summary["algorithm"].nunique() * 2), 8)
    )
    if case_summary.empty or not case_summary["mean_final_energy"].notna().any():
        _save_empty_plot(ax, "No valid per-case energy measurements")
    else:
        cases = (
            case_summary[["case_id", "protein_length"]]
            .drop_duplicates()
            .sort_values(["protein_length", "case_id"])
        )
        algorithms = sorted(case_summary["algorithm"].unique())
        matrix = (
            case_summary.pivot(
                index="case_id",
                columns="algorithm",
                values="mean_final_energy",
            )
            .reindex(index=cases["case_id"], columns=algorithms)
            .to_numpy(dtype=float)
        )
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(algorithms)), algorithms, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(cases)), cases["case_id"])
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Test case")
        fig.colorbar(image, ax=ax, label="Mean final energy (lower is better)")
    ax.set_title("Algorithm comparison across test cases")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _budget_summary(
    runs: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in runs.groupby(list(group_columns), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        valid = group.loc[group["valid_result"] & group["best_energy"].notna()]
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                "runs": int(len(group)),
                "completed_runs": int(group["status"].eq("completed").sum()),
                "valid_structure_runs": int(group["valid_result"].sum()),
                "failed_runs": int(group["status"].ne("completed").sum()),
                "mean_best_energy": valid["best_energy"].mean(),
                "median_best_energy": valid["best_energy"].median(),
                "std_best_energy": valid["best_energy"].std(),
                "best_energy": valid["best_energy"].min(),
                "worst_energy": valid["best_energy"].max(),
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
                "median_runtime_seconds": group["runtime_seconds"].median(),
                "mean_energy_evaluations": group["energy_evaluations"].mean(),
                "total_energy_evaluations": group[
                    "energy_evaluations"
                ].sum(min_count=1),
            }
        )
    return pd.DataFrame.from_records(rows)


def _save_best_energy_vs_budget(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for algorithm, group in summary.groupby("algorithm", sort=True):
        ordered = group.sort_values("budget")
        values = ordered["mean_best_energy"].to_numpy(dtype=float)
        if np.isfinite(values).any():
            ax.plot(
                ordered["budget"],
                values,
                marker="o",
                label=str(algorithm),
            )
            has_data = True
    if has_data:
        ax.set_xscale("log")
        ax.set_xlabel("Objective-evaluation budget")
        ax.set_ylabel("Mean best-so-far energy (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.25)
    else:
        _save_empty_plot(ax, "No valid best-energy measurements")
    ax.set_title("Mean best energy versus evaluation budget (all cases)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_budget_length_panels(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    budgets = sorted(int(value) for value in summary["budget"].dropna().unique())
    if not budgets:
        fig, ax = plt.subplots(figsize=(8, 5))
        _save_empty_plot(ax, "No budget values recorded")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    column_count = min(2, len(budgets))
    row_count = (len(budgets) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 4.4 * row_count),
        squeeze=False,
    )
    for index, budget in enumerate(budgets):
        ax = axes[index // column_count][index % column_count]
        budget_rows = summary.loc[summary["budget"].eq(budget)]
        has_data = False
        for algorithm, group in budget_rows.groupby("algorithm", sort=True):
            ordered = group.sort_values("protein_length")
            values = ordered[metric].to_numpy(dtype=float)
            if np.isfinite(values).any():
                ax.plot(
                    ordered["protein_length"],
                    values,
                    marker="o",
                    label=str(algorithm),
                )
                has_data = True
        if has_data:
            ax.set_xlabel("Protein length (residues)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.25)
        else:
            _save_empty_plot(ax, "No valid recorded measurements")
        ax.set_title(f"Budget: {budget:,} evaluations")

    for index in range(len(budgets), row_count * column_count):
        axes[index // column_count][index % column_count].set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze_budget_sensitivity_results(
    runs: pd.DataFrame,
    output_dir: str | Path = "results/budget_study",
) -> BudgetStudyAnalysis:
    """Summarize measured budget-study runs and save the three requested plots."""
    prepared = _prepare_runs(runs)
    if "budget" not in prepared.columns:
        raise ValueError("Budget-study runs must contain a 'budget' column.")
    original_budget = prepared["budget"]
    converted_budget = pd.to_numeric(original_budget, errors="coerce")
    if converted_budget.isna().any() or (converted_budget % 1 != 0).any():
        raise ValueError("budget must contain positive whole-number values.")
    if (converted_budget < 1).any():
        raise ValueError("budget must contain positive whole-number values.")
    prepared["budget"] = converted_budget.astype(int)
    if "energy_budget_limit" in prepared.columns:
        recorded_limit = pd.to_numeric(
            prepared["energy_budget_limit"], errors="coerce"
        )
        mismatch = recorded_limit.notna() & recorded_limit.ne(prepared["budget"])
        if mismatch.any():
            raise ValueError("budget does not match energy_budget_limit.")

    budget_summary = _budget_summary(prepared, ("budget", "algorithm"))
    length_summary = _budget_summary(
        prepared, ("budget", "protein_length", "algorithm")
    )
    case_summary = _budget_summary(
        prepared, ("budget", "case_id", "protein_length", "algorithm")
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    budget_summary.to_csv(output / "budget_study_summary.csv", index=False)
    length_summary.to_csv(
        output / "budget_study_length_summary.csv", index=False
    )
    case_summary.to_csv(output / "budget_study_case_summary.csv", index=False)

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plot_dir / "mean_best_energy_vs_budget.png",
        plot_dir / "mean_best_energy_vs_length_per_budget.png",
        plot_dir / "runtime_vs_length.png",
    )
    _save_best_energy_vs_budget(budget_summary, plot_paths[0])
    _save_budget_length_panels(
        length_summary,
        "mean_best_energy",
        "Mean best-so-far energy (lower is better)",
        "Mean best energy versus protein length, by budget",
        plot_paths[1],
    )
    _save_budget_length_panels(
        length_summary,
        "mean_runtime_seconds",
        "Mean wall-clock runtime (seconds)",
        "Mean runtime versus protein length, by budget",
        plot_paths[2],
    )
    return BudgetStudyAnalysis(
        budget_summary=budget_summary,
        length_summary=length_summary,
        case_summary=case_summary,
        plot_paths=plot_paths,
    )


def analyze_budget_sensitivity_csv(
    csv_path: str | Path = "results/budget_study/budget_study_runs.csv",
    output_dir: str | Path = "results/budget_study",
) -> BudgetStudyAnalysis:
    """Analyze the budget-study CSV without filling missing observations."""
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"Budget-study run CSV does not exist: {source}")
    return analyze_budget_sensitivity_results(pd.read_csv(source), output_dir)


def analyze_experiment_results(
    runs: pd.DataFrame,
    output_dir: str | Path = "results/analysis",
) -> ExperimentAnalysis:
    """Summarize measured runs and save the six requested graphs.

    Quality statistics include only completed runs with independently valid
    best structures. Failure and validity rates use every recorded run.
    """
    prepared = _prepare_runs(runs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    algorithm_summary = _summarize(prepared, ("algorithm",))
    length_summary = _summarize(
        prepared, ("protein_length", "algorithm")
    )
    case_comparison = _summarize(
        prepared, ("case_id", "protein_length", "algorithm")
    )
    histories = _convergence_runs(prepared)
    convergence = _convergence_table(histories)

    algorithm_summary.to_csv(output / "algorithm_summary.csv", index=False)
    length_summary.to_csv(output / "length_algorithm_summary.csv", index=False)
    case_comparison.to_csv(output / "case_algorithm_comparison.csv", index=False)
    convergence.to_csv(output / "convergence_curves.csv", index=False)

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = (
        plot_dir / "final_energy_vs_length.png",
        plot_dir / "runtime_vs_length.png",
        plot_dir / "memory_vs_length.png",
        plot_dir / "convergence_curves.png",
        plot_dir / "valid_vs_invalid_candidates.png",
        plot_dir / "algorithm_comparison_by_case.png",
    )
    _save_length_plot(
        length_summary,
        "mean_final_energy",
        "Mean final energy (lower is better)",
        "Final energy versus protein length",
        plot_paths[0],
        error_metric="std_final_energy",
    )
    _save_length_plot(
        length_summary,
        "median_runtime_seconds",
        "Median wall-clock time (seconds)",
        "Median runtime versus protein length",
        plot_paths[1],
    )
    _save_length_plot(
        length_summary,
        "mean_peak_memory_bytes",
        "Mean peak traced memory (MiB)",
        "Peak traced memory versus protein length",
        plot_paths[2],
        scale=1024**2,
    )
    _save_convergence_plot(convergence, plot_paths[3])
    _save_candidate_plot(algorithm_summary, plot_paths[4])
    _save_case_comparison(case_comparison, plot_paths[5])

    return ExperimentAnalysis(
        algorithm_summary=algorithm_summary,
        length_summary=length_summary,
        case_comparison=case_comparison,
        convergence=convergence,
        plot_paths=plot_paths,
    )


def analyze_experiment_csv(
    csv_path: str | Path,
    output_dir: str | Path = "results/analysis",
) -> ExperimentAnalysis:
    """Load recorded run data from CSV and analyze it without inventing values."""
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"Experiment run CSV does not exist: {source}")
    return analyze_experiment_results(pd.read_csv(source), output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize recorded HP folding experiments and plot results."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Run-level CSV exported by the selected experiment runner.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary CSV files and plots.",
    )
    parser.add_argument(
        "--budget-study",
        action="store_true",
        help="Analyze budget-study data and create its three sensitivity plots.",
    )
    args = parser.parse_args(argv)
    if args.budget_study:
        input_csv = args.input_csv or "results/budget_study/budget_study_runs.csv"
        output_dir = args.output_dir or "results/budget_study"
        analysis = analyze_budget_sensitivity_csv(input_csv, output_dir)
        print(analysis.budget_summary.to_string(index=False))
    else:
        input_csv = args.input_csv or "results/experiment_runs.csv"
        output_dir = args.output_dir or "results/analysis"
        analysis = analyze_experiment_csv(input_csv, output_dir)
        print(analysis.algorithm_summary.to_string(index=False))
    print(f"\nAnalysis files saved to: {Path(output_dir)}")


if __name__ == "__main__":
    main()
