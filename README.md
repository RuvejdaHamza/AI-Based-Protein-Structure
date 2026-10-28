# AI-Based Protein Structure Prediction and Optimization

This project implements a 2D Hydrophobic-Polar (HP) lattice protein-folding
model and compares Simulated Annealing (SA), a Genetic Algorithm (GA), and a
hybrid GA + SA optimizer.

## Project layout

```text
.
├── benchmarks/             # Reproducible performance comparisons
├── docs/                   # Algorithm, model, architecture, and project notes
├── src/
│   └── hp_folding/         # Reusable Python package
├── tests/                  # Unit tests
└── README.md
```

Core functionality is organized by responsibility:

- `sequence_processing.py`: amino-acid validation and H/P conversion.
- `structure.py`: lattice coordinates, self-avoiding walk validation, and generation.
- `energy.py`: non-consecutive H-H contact energy.
- `local_moves.py`: end, corner, crankshaft, and neighborhood operators.
- `simulated_annealing.py`, `genetic_algorithm.py`, `hybrid_optimizer.py`:
  optimization algorithms.
- `search_space.py`: symmetry keys and incremental energy tracking.
- `experiment_runner.py`: reproducible runs with a shared evaluation budget.
- `experiment_analysis.py`: statistics, convergence analysis, and plots from
  recorded run-level CSV data.
- `visualization.py`: validated lattice plots and algorithm comparisons using
  the structures saved in experiment records.
- `cli.py`: command-line folding runs with per-algorithm evaluation/time limits,
  optional hard constraints, and structure reporting.

Install the analysis dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

## Run tests (PowerShell)

From the project root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Run the performance benchmark (PowerShell)

```powershell
$env:PYTHONPATH = "src"
python -m benchmarks.benchmark_performance
```

## Run the experiment framework

The runner evaluates all 40 synthetic and UniProt-derived cases with ten
independent seeded runs for each of SA, GA, and the hybrid (1,200 runs total).
It uses a shared cap of 200 objective evaluations per run; the hybrid shares
that cap between GA and nested SA. The cap counts initial energy scoring.

```powershell
$env:PYTHONPATH = "src"
python -m hp_folding.experiment_runner
```

## Run the command-line optimizer

The CLI validates the amino-acid sequence, converts it to H/P, and runs SA,
GA, the hybrid, or all three. Each algorithm receives its own objective
evaluation budget and optional wall-clock deadline. `--hh-energy` is the
negative energy contribution per H-H contact; its default is `-1.0`.

From PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m hp_folding.cli --sequence ACDEFGHIKLMNPQRSTVWY --algorithm all --max-evaluations 200 --seed 42
```

Run one optimizer with a time limit and save its best-so-far structure:

```powershell
python -m hp_folding.cli --sequence ACDEFGHIKLMNPQRSTVWY --algorithm hybrid --max-evaluations 200 --time-limit 30 --seed 42 --hh-energy -1.5 --png results\fold.png
```

An optional constraints JSON file uses zero-based residue and step indices.
Fixed contacts require the specified non-consecutive residues to occupy
neighboring lattice sites; forbidden positions are `(x, y)` lattice
coordinates, and fixed directions constrain the step starting at that
zero-based residue index. For example:

```json
{
  "fixed_contacts": [[0, 3]],
  "forbidden_positions": [[3, 0]],
  "fixed_directions": {"0": "R"}
}
```

Use the constraints with:

```powershell
python -m hp_folding.cli --sequence ACDE --algorithm all --constraints constraints.json
```

Structures that violate constraints are rejected during validation and
candidate generation. The CLI reports best/final energy, candidate counts,
objective evaluations, wall time, traced Python peak memory, and an ASCII
lattice drawing. In `all` mode it also prints a comparison table and writes a
separate PNG for each algorithm if `--png` is supplied. Time checks are
cooperative at candidate-generation/evaluation boundaries; a currently
executing Python operation can finish just after the requested deadline.

In a Kaggle Notebook:

```python
import sys
sys.path.insert(0, "/kaggle/working/daaprojekti1/src")

from hp_folding.experiment_runner import run_experiments
from hp_folding.experiment_analysis import analyze_experiment_csv

runs_df, summary_df = run_experiments(
    run_count=10,
    energy_budget=200,
    output_dir="/kaggle/working/results",
)
display(summary_df)
display(
    runs_df[
        [
            "case_id",
            "algorithm",
            "replicate",
            "runtime_seconds",
            "peak_memory_bytes",
            "generated_candidates",
            "valid_candidates",
            "invalid_candidates",
            "energy_evaluations",
            "convergence_evaluations_to_best",
            "final_energy",
        ]
    ].head()
)
```

Analyze the recorded data and generate summaries and six plots:

```python
analysis = analyze_experiment_csv(
    "/kaggle/working/results/experiment_runs.csv",
    output_dir="/kaggle/working/results/analysis",
)
display(analysis.algorithm_summary)
display(analysis.length_summary)

from IPython.display import Image, display

for plot_path in analysis.plot_paths:
    display(Image(filename=str(plot_path)))
```

The analysis writes algorithm-, length-, and case-level summary CSV files,
plus a convergence table and six PNG plots. Tables report mean final energy,
sample standard deviation, best/worst energy, mean wall and CPU time, mean
traced memory, validity percentages, and candidate/evaluation counts. The
runtime-versus-length plot uses median wall time so a single long scheduling
pause does not flatten the trends; raw mean, median, and maximum are retained
in the length summary. Quality statistics use only
completed runs with valid returned structures; validity and failure
percentages retain all run records. Standard deviation is the sample standard
deviation and remains undefined for a single valid run. Best energy is the
minimum (more negative is better); worst energy is the maximum.

Convergence curves show the median best-so-far energy at recorded
objective-evaluation checkpoints, including the measured terminal best at each
run's actual evaluation count. Between checkpoints they hold the latest
recorded value and do not extrapolate beyond termination. GA and hybrid
histories are generation-grained, so they cannot locate improvements within a
generation. Per-length means also combine distinct sequence cases and should
be interpreted alongside the case-level table.

The runner stores per-run convergence histories as JSON text in
`experiment_runs.csv` and writes `experiment_summary.csv` alongside it.
Equal objective-evaluation caps provide a consistent search budget; runtime
and peak Python-tracked memory are also recorded because algorithms have
different candidate-generation overhead. The complete input/seed protocol is
in [`docs/experimental-evaluation-protocol.md`](docs/experimental-evaluation-protocol.md).

Timing uses `time.perf_counter()` around each optimizer call (including SA
initial-structure generation, but excluding result validation and CSV work).
Each run gets a fresh `tracemalloc` session; the runner refuses to start while
another trace is active. `peak_memory_bytes` measures peak traced Python
allocations, not process RSS or native-library allocations. Candidate counts
cover all generated proposals (GA genomes and local-move attempts); valid and
invalid proposals are reported separately. `convergence_evaluations_to_best`
is the evaluation count at the first stored history checkpoint reaching the
best energy. It is exact for SA's per-iteration history but generation-grained
for GA and the hybrid, so it is a useful convergence indicator rather than an
exact timestamp for each improvement.
The analysis reports only values present in the input CSV; it does not fill
missing runs or infer which optimizer is best.

## Visualize recorded protein structures

The visualization module reconstructs structures from each run's saved
relative directions, validates chain connectivity and self-avoidance, and
recomputes energy before plotting. Comparisons default to each algorithm's
best valid recorded replicate for the selected case; pass `replicate` to show
the same replicate number instead. Lattice axes use equal scale, residue
indices are shown on the nodes, and non-consecutive H-H contacts are marked
with dashed links.

```python
from hp_folding.visualization import plot_algorithm_comparison

figure, selected_runs = plot_algorithm_comparison(
    runs_df,
    "real_HBB_N20",
    output_path="/kaggle/working/results/visualizations/real_HBB_N20.png",
)
display(selected_runs)
display(figure)
```

The returned table identifies the exact selected replicate and its recomputed
energy/contact count. No coordinates are generated or repaired for display.

From PowerShell, analyze an existing CSV with:

```powershell
$env:PYTHONPATH = "src"
python -m hp_folding.experiment_analysis results\experiment_runs.csv --output-dir results\analysis
```

Create an actual-structure comparison from a recorded run CSV:

```powershell
$env:PYTHONPATH = "src"
python -m hp_folding.visualization results\experiment_runs.csv --case-id real_HBB_N20
```

Benchmark values are workload- and machine-dependent; see
[`docs/search-space-reduction.md`](docs/search-space-reduction.md) for recorded
measurements and their limitations.

## Use in a Kaggle Notebook

After adding or cloning the project files into the Kaggle notebook environment,
add the `src` directory to Python's import path:

```python
import sys
sys.path.insert(0, "/kaggle/working/daaprojekti1/src")

from hp_folding.sequence_processing import process_sequence
from hp_folding.structure import generate_random_valid_structure
from hp_folding.simulated_annealing import run_simulated_annealing
```

Adjust the path to the directory where the project was placed in the notebook.
The folding algorithms use the Python standard library; analysis and plotting
use Pandas, NumPy, and Matplotlib from `requirements.txt`.

## Documentation

Start with [`docs/project-progress-summary.md`](docs/project-progress-summary.md)
for the project overview. The detailed theoretical bounds are in
[`docs/complexity-analysis.md`](docs/complexity-analysis.md).
The reproducible algorithm comparison plan is in
[`docs/experimental-evaluation-protocol.md`](docs/experimental-evaluation-protocol.md).
For the complete technical documentation and oral-defense questions, see
[`docs/project-defense-guide.md`](docs/project-defense-guide.md).
