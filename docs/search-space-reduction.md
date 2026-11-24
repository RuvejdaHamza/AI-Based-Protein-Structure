# Search-space reduction in the 2D HP folding model

## Scope and interpretation

The optimizers search a combinatorial space of self-avoiding walks. The techniques below either remove equivalent representations, avoid repeating work, reject invalid prefixes early, restrict which moves can be proposed, or terminate a run sooner. Heuristic search still does **not** enumerate or mathematically eliminate all protein conformations.

The implementation distinguishes:

- **Generated candidates:** candidate direction sequences or local-move proposals attempted.
- **Invalid candidates:** candidates rejected because a walk collides or otherwise fails validation.
- **Duplicate candidates:** valid states equivalent to a state already represented in the current population/neighborhood; these are not energy-evaluated again.
- **Restricted operator opportunities:** local move slots omitted by a chosen move-type restriction. This counts omitted operator opportunities, not unique conformations or all states never generated.
- **Evaluated candidates:** structures for which energy evaluation was performed.

The exact number of possible conformations never generated is not measurable from a heuristic run without enumerating the full state space. No such total is claimed here.

## Techniques

| Technique | Principle and implementation point | Computational impact/statistics | Effect on solution quality |
|---|---|---|---|
| Symmetry reduction | `symmetry_key` maps translations and the eight square-lattice rotations/reflections to a canonical coordinate key. GA can use `symmetry_reduction=True`; GA and hybrid skip repeat energy evaluation for a key already present in the population. Local neighborhoods also omit equivalent candidates. | Key construction is \(O(N)\). Counted through `symmetry_duplicate_candidates` and `duplicate_candidates`; key computation itself has overhead. | Exact for this HP objective because lattice symmetries preserve connectivity, self-avoidance, and contacts. It removes equivalent representations, not distinct physical folds. |
| Duplicate-state detection | Hash direction genomes or canonical structural keys within each population/neighborhood. Keep a representative for population size, but do not reevaluate an equivalent candidate. | GA reports total and symmetry-specific duplicates; SA reports `duplicate_candidates_avoided`. | Does not remove a novel state. Prevents repeated evaluation; a duplicate can still occupy a GA slot as a copy of its representative. |
| Early collision rejection | GA decodes direction genes incrementally and stops as soon as a coordinate is already occupied. Local moves check occupancy before full structure validation. | GA reports `early_collision_rejections`, `coordinate_steps_checked`, and `coordinate_steps_avoided`; invalids remain separate from valid duplicates. | Exact feasibility check: only invalid self-intersecting walks are rejected. No valid structure is discarded by this test. |
| Incremental energy evaluation | `IncrementalEnergyTracker` stores contact pairs and coordinate occupancy. For each SA proposal it recalculates contacts incident to changed residues; accepted proposals update the tracker. Tests compare it with full energy recomputation. | SA reports incremental neighbor checks and an estimated full-scan count. Full structure validation and changed-index detection still take \(O(N)\), so the asymptotic proposal cost remains \(O(N)\) in this interface, but fewer contact-neighbor lookups are needed. | Exact energy; no intended change in ranking or solution quality. |
| Restricted local neighborhoods | SA accepts selected move types (`end`, `corner`, `crankshaft`). `generate_neighborhood` omits other move categories and counts the omitted operator opportunities. | `restricted_operator_opportunities`, local move attempts, valid/invalid proposals, and runtime are reported. | Can reduce work, but the graph of reachable states may be smaller or disconnected; it can miss useful folds and reduce solution quality. |
| Elitism | GA copies the best configured number of individuals into the next generation; hybrid additionally restores its global best if necessary. | `elite_individuals_copied`, best-energy history, and generation count make preservation measurable. | Guarantees the best-so-far result is not lost, but repeated elite propagation can reduce diversity and encourage premature convergence. It does not eliminate the underlying state space. |
| Early stopping | SA and GA/hybrid stop on configured iteration/generation limits, target energy, minimum temperature (SA), no improvement, or no valid neighbors (SA). | Results include actual iterations/generations and `termination_reason`; energy-evaluation counts show work saved relative to the configured maximum. | Saves computation but may stop before finding a better structure or the global optimum. |

## Complexity notes

Let \(N\) be the number of residues, \(P\) the GA population size, \(G\) the generation count, \(I\) the SA iteration count, \(k\) the number of changed residues, and \(H\) the number of hydrophobic residues.

- A symmetry key costs \(O(N)\) time and space. Hash lookup is expected \(O(1)\) after key construction.
- Early walk decoding costs \(O(j)\), where \(j\le N-1\) is the number of steps before collision or completion. Full decoding and validation is \(O(N)\).
- Full HP energy evaluation is \(O(N)\) using a coordinate map and four lattice-neighbor checks per hydrophobic residue.
- Incremental contact calculation examines up to \(4k\) neighboring lattice sites. Current structure validation and changed-index detection remain \(O(N)\); therefore the complete `evaluate` call remains \(O(N)\), although it avoids a full contact recount.
- GA cost is typically \(O(GPN)\), plus bounded retries, key construction, and invalid candidates. Memory is \(O(PN)\).
- An SA neighborhood has \(O(N)\) move opportunities; validating/materializing all candidates is \(O(N^2)\) in the current implementation. Incremental energy updates do not remove this neighborhood-generation cost.
- Restricting move types lowers the explored neighborhood and work, but is heuristic and may reduce reachability.

## Controlled experiment

The following measurements were obtained in the project environment using sequence `HPPHHPHHPPHH` and seeds `11, 23, 37, 41, 59`. Values are means across five runs unless stated otherwise. They are illustrative measurements, not universal performance guarantees. Wall time is sensitive to interpreter and machine load.

### GA symmetry-key overhead

Both settings used population 16, 15 generations, tournament size 3, and one elite. Exact duplicate detection remained active in both; the `False` setting used exact-genome keys, while `True` used D4 symmetry keys.

| Symmetry key | Mean best energy | Mean runtime (s) | Mean candidates generated | Mean invalid | Mean duplicate candidates | Mean symmetry-only duplicates | Mean energy evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Off (exact-genome dedup only) | -4.6 | 0.0130 | 807 | 566 | 99 | 0 | 142 |
| On (D4 canonical key) | -4.6 | 0.0257 | 807 | 566 | 99 | 0 | 142 |

This sample produced no additional symmetry-only duplicates, so canonicalization did not reduce energy evaluations and roughly doubled runtime. The separate unit test confirms that rotated/reflected representations share a key. For these settings, symmetry reduction is useful only when the population actually contains such equivalent states; the option should be benchmarked for other sequence lengths and workloads.

Early-rejection work was also measured in the GA runs: 566 of 807 generated candidates were invalid and rejected on collision. A full decode would inspect \(807\times11=8{,}877\) direction steps; early rejection inspected approximately \(6{,}006\), avoiding about 2,871 steps (32.3%) on average. These are step counts, not a claim of equivalent wall-time speedup.

### SA unrestricted versus restricted local moves

Both variants used 120 iterations, the same cooling schedule, and the same seeds. The restricted case allowed only end and corner moves, omitting crankshaft operator slots.

| Move set | Mean best energy | Best observed | Mean runtime (s) | Mean local proposals | Mean unique valid neighbors | Mean invalid proposals | Mean omitted operator opportunities |
|---|---:|---:|---:|---:|---:|---:|---:|
| End + corner + crankshaft | -3.8 | -5 | 0.1624 | 5,510 | 960 | 3,973 | 0 |
| End + corner | -3.4 | -4 | 0.0913 | 2,160 | 689 | 1,471 | 1,080 |

In this sample, restricting moves lowered runtime by about 44% but worsened mean and best-observed energy. It also removes crankshaft transitions, so it can exclude useful paths through the conformation graph. The 1,080 figure is the mean number of omitted move-start opportunities per run; it is not a count of unique folds.

### Incremental contact checks

For the unrestricted SA runs, the tracker performed an average of 456 contact-neighbor checks. A full scan of all hydrophobic residues for the initial energy plus 120 proposals would perform an estimated \(4H(121)=3{,}388\) such checks. This is approximately 86.5% fewer contact-neighbor checks. The estimate excludes validation and changed-index scans; therefore it does not imply an 86.5% runtime reduction.

## Implementation bottleneck benchmark

The project also benchmarks reconstructed reference paths against the optimized implementations in [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py). The reference paths are not historical Git versions (the workspace has no Git history). Component times below are medians of five runs in the current Python environment; peak allocation is measured with `tracemalloc`, not process RSS.

| Workload | Reference runtime / peak allocation | Optimized runtime / peak allocation | Equivalence and interpretation |
|---|---:|---:|---|
| Energy evaluation, 48-residue valid structure | 0.000080 s / 3,737 B | 0.000017 s / 3,376 B | The optimized internal path assumes the caller has already established validity; the public evaluator still validates. |
| Neighborhood generation | 0.010617 s / 29,664 B | 0.004161 s / 29,088 B | Both returned the same 20 canonical neighbors. |
| GA scoring, 250 candidates | 0.032100 s / 953,041 B | 0.007185 s / 953,016 B | Both scored 250 valid candidates with 250 energy evaluations and the same best energy. |
| Simulated Annealing, five seeds | 0.045399 s mean | 0.021842 s mean | Both had mean best energy -1.8, best observed -3, and 36 mean energy evaluations. |

The main runtime improvement comes from avoiding redundant full-structure validation and repeated candidate checks, not from changing the energy objective. Peak allocation for GA scoring changed by only 25 bytes in this run; memory improvement is therefore negligible there. Results are workload- and machine-dependent and should not be generalized as guaranteed speedups.

## Validation

Tests verify D4-invariant keys, skipping a symmetry duplicate's energy evaluation, early collision counters, restricted operator counters, incremental energies against full recalculation, and optimized/reference neighborhood parity. The complete project test command is:

```text
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The latest run completed 86 tests successfully. The performance experiment can be rerun with:

```text
$env:PYTHONPATH = "src"
python -m benchmarks.benchmark_performance
```
