# Results and Discussion

## 1. Experimental observations

The primary experiment contains **1,200 runs**: 40 sequence cases, 10
independent runs per case, and three algorithms. The cases include synthetic
HP sequences and amino-acid-derived HP prefixes at lengths 8, 12, 16, 20, and
30. Each run used the same budget of 200 energy evaluations. All runs
completed, and all 1,200 returned best structures passed the recorded
validity check. The analysis is based on the [run-level records](../results/experiment_runs.csv)
and the project's [aggregate results](../results/analysis/algorithm_summary.csv).

The objective was \(E=-C_{HH}\), with \(\epsilon=1\); therefore, more negative
energy means more non-consecutive hydrophobic contacts. No known global
optimum was used, so the best observed energies should not be interpreted as
proven optima.

## 2. Algorithm performance comparison

For solution-quality comparisons, the table below uses each run's
**best-so-far energy**, not necessarily its terminal-state energy.

| Algorithm | Mean best energy ± SD | Median best energy | Observed best–worst | Median wall time | Mean CPU time | Mean peak traced memory |
|---|---:|---:|---:|---:|---:|---:|
| Genetic Algorithm (GA) | −4.210 ± 3.329 | −4 | −16 to 0 | 0.025 s | 0.029 s | 0.0209 MiB |
| Hybrid GA + SA | −4.065 ± 3.228 | −3 | −15 to 0 | 0.316 s | 0.385 s | 0.0253 MiB |
| Simulated Annealing (SA) | −3.073 ± 2.410 | −3 | −13 to 0 | 0.569 s | 0.706 s | 0.0243 MiB |

Across all runs, GA had the most negative mean best energy and the fastest
median wall time. The hybrid's mean best energy was close to GA's, while SA's
was less favorable. The observed energy ranges include zero for each
algorithm, indicating that some runs found no non-consecutive H-H contact.

A per-case comparison of the **10-run mean best energy** gives a more
qualified picture: GA was better than the hybrid in 16 of 40 cases, tied in
18, and worse in 6. GA was better than SA in 28 cases and tied in 12; the
hybrid was better than SA in 27 and tied in 13. Thus, GA led descriptively
overall, but not on every case. The results do not include inferential tests
or confidence intervals, so these counts do not establish statistical
significance.

## 3. Solution quality

The best-energy results favor GA on average, but the difference between GA and
the hybrid was modest: 0.145 energy units. Because energy is discrete for this
model, that difference is an aggregate across runs and cases, not a
fractional contact in an individual structure. Both GA and the hybrid found
stronger best-so-far solutions than SA on average.

An important reporting distinction is that the run-level data separately
records `best_energy` and `final_energy`. SA may accept a worse move after
finding a strong structure, so its terminal state can be worse than its
best-so-far structure. In these records, SA's mean terminal energy is
**−1.090**, while its mean best-so-far energy is **−3.073**. Consequently,
best-so-far energy is the appropriate metric for comparing the best solutions
found; the aggregate CSV's `mean_final_energy` should not be described as
SA's best-found quality.

The overall standard deviations combine variation between sequence cases
with variation between runs. They therefore describe the mixed benchmark,
not just stochastic variability on one fixed input.

## 4. Computational efficiency

All algorithms used exactly 200 energy evaluations per run, or 80,000 per
algorithm. This equalizes objective calls, but not proposal-generation,
validation, or other computational work.

GA was fastest by median wall time: approximately **12.6 times faster than
the hybrid** and **22.7 times faster than SA**. The hybrid was faster than SA
by median wall time. Mean process CPU times show the same ordering. The peak
Python-traced memory means were all below 0.03 MiB; GA had the lowest mean,
and the hybrid the highest. These memory measurements are not process RSS and
exclude some native allocations.

Wall-time outliers make means less representative: the maximum hybrid run
took 7,087.7 seconds of wall time but 1.23 seconds of CPU time; the maximum SA
wall time was 115.9 seconds. The cause is not established by the records.
Medians and CPU times are therefore more informative than wall-time means
for this experiment, while the outliers still warrant caution.

Candidate counts also show different work profiles. Mean generated
candidates per run were about 925 for GA, 6,943 for the hybrid, and 11,542
for SA. Their mean invalid-candidate counts were about 502, 4,394, and 6,746,
respectively. These counts reflect distinct proposal processes and should
not be treated as equivalent units of work.

## 5. Scalability

Mean best energy became more negative with increasing protein length for all
three algorithms. This is consistent with longer sequences having more
opportunities for H-H contacts; it does not, by itself, indicate improved
biological prediction. The following table reports mean best energy and
median wall time from the [length-level results](../results/analysis/length_algorithm_summary.csv).

| Length | GA: mean best energy / median time | Hybrid: mean best energy / median time | SA: mean best energy / median time |
|---:|---:|---:|---:|
| 8 | −1.625 / 0.013 s | −1.613 / 0.112 s | −1.563 / 0.193 s |
| 12 | −2.900 / 0.022 s | −2.663 / 0.220 s | −2.313 / 0.408 s |
| 16 | −4.163 / 0.027 s | −4.113 / 0.317 s | −3.138 / 0.570 s |
| 20 | −5.013 / 0.026 s | −4.863 / 0.424 s | −3.338 / 0.816 s |
| 30 | −7.350 / 0.043 s | −7.075 / 0.839 s | −5.013 / 1.596 s |

Median time generally increased with length, although GA's values were not
strictly monotonic between lengths 16 and 20. The tested range ends at 30
residues; these measurements are insufficient to establish asymptotic scaling
or performance on longer proteins.

## 6. Search-space reduction effectiveness

In the primary runs, GA rejected an average of 502 candidates per run during
early collision checks and recorded about 223 duplicate candidates. SA
recorded about 2,079 symmetry-equivalent duplicate candidates avoided per
run. These figures document avoided or rejected work; they do **not**
quantify the total number of conformations never generated. The experiment
did not exhaustively enumerate the search space.

Separate controlled measurements in the project's
[search-space reduction report](search-space-reduction.md) qualify these
findings:

- In a five-seed GA comparison, enabling D4 symmetry keys produced no
  symmetry-only duplicates. Mean best energy and evaluation counts were
  unchanged, while mean runtime increased from 0.0130 to 0.0257 seconds. In
  that sample, canonicalization added cost without reducing evaluations.
- In an early-rejection benchmark, 566 of 807 proposals were invalid. Early
  rejection avoided approximately 2,871 direction-step checks, or 32.3% of
  the full-decode count. These are operation counts, not a measured runtime
  speedup.
- Restricting SA from end, corner, and crankshaft moves to end and corner
  moves reduced mean runtime by about 44%, but mean best energy worsened from
  −3.8 to −3.4. This illustrates a measured speed-quality trade-off for that
  small sample.
- Incremental contact evaluation reduced the reported contact-neighbor
  checks by about 86.5% against an estimated full-scan count. The report does
  not establish an equivalent runtime reduction; full proposal validation
  still has cost.

These component studies are small and should not be conflated with the
1,200-run primary matrix. In particular, GA symmetry reduction was disabled
in the main experiment, while SA's neighborhood generation applies symmetry
deduplication. This difference is a caveat when attributing outcomes solely
to the search strategy.

## 7. Convergence behavior

The [convergence data](../results/analysis/convergence_curves.csv) show the
mean best-so-far energy at selected evaluation checkpoints:

| Algorithm | At 20 evaluations | At 100 evaluations | At 200 evaluations |
|---|---:|---:|---:|
| GA | −3.568 | −4.033 | −4.210 |
| Hybrid GA + SA | −3.523 | −3.900 | −4.065 |
| SA | −2.208 | −2.798 | −3.073 |

All three methods improved their mean best-so-far score over the measured
checkpoints. GA had the most favorable mean at each listed checkpoint; the
hybrid was close initially but finished behind GA. SA also improved, though
its mean remained less favorable.

These checkpoints are not equally time-resolved: SA records per-iteration
history, whereas GA and the hybrid record generation-level history. The
comparison therefore describes progress at recorded evaluation checkpoints,
not the exact evaluation at which each improvement occurred.

## 8. Advantages and disadvantages

- **GA:** Best mean best-so-far energy, fastest median runtime, and lowest
  mean traced memory in this experiment. It still rejected many invalid
  candidates, and its symmetry-reduction option was disabled in the main
  comparison.
- **SA:** Improved best-so-far energy throughout the observed checkpoints and
  recorded substantial duplicate avoidance. It generated the most candidates
  and had higher median runtime than GA and the hybrid.
- **Hybrid:** Improved over SA in the per-case mean comparison for 27 cases
  and tied in 13. It did not consistently improve on GA and had greater
  runtime and mean traced memory. Its local search averaged one call per run,
  with 34 population replacements across 400 runs (0.085 per run), so the
  measured replacement benefit was limited under this configuration.

These observations apply to the tested settings and budget; they do not
establish a universal ranking of the algorithms.

## 9. Limitations of the 2D HP model and evaluation

The results concern a simplified 2D lattice objective, not native protein
structure prediction. The model reduces each amino acid to H or P and rewards
only non-consecutive H-H contacts. It does not represent three-dimensional
geometry, side-chain properties, explicit solvent, hydrogen bonds,
electrostatics, disulfide bonds, or other energetic and structural factors.
A low HP energy therefore means a favorable score under this model, not
demonstrated biological accuracy.

The benchmark also uses a limited set of synthetic patterns and N-terminal
prefixes from four real proteins. Prefixes from the same protein are nested
rather than independent biological samples, and the real-sequence cases are
not a representative sample of proteins. The records provide descriptive
statistics but no significance tests or confidence intervals. Finally,
equal energy-evaluation budgets do not imply equal CPU work, as the differing
proposal counts and runtimes demonstrate.

## 10. Potential future improvements

Future evaluation should report per-case uncertainty and use appropriately
chosen statistical tests with correction for multiple comparisons. Matched
initial conformations could support paired algorithm comparisons, provided
they are recorded and the comparison remains reproducible. Budget-sensitivity
experiments at multiple evaluation caps would show whether the observed
ranking changes with available search effort; those results should be
reported separately from this 200-evaluation matrix.

Further work could also benchmark harmonized symmetry handling, test longer
sequences and broader sequence sets, and run controlled ablations of local
moves and incremental evaluation. For biological relevance, the model would
need richer residue representations, more realistic energy terms, and
three-dimensional structural constraints. Such extensions should be
evaluated as new models rather than treated as conclusions supported by the
current 2D HP results.
