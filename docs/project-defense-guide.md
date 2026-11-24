# AI-Based Protein Structure Prediction and Optimization

## Technical Documentation and University Project Defense Guide

> **Scope statement:** This project optimizes structures in a simplified
> two-dimensional Hydrophobic-Polar (HP) lattice model. It is an algorithmic
> folding-optimization project, not a validated predictor of experimentally
> determined native protein structures.

The descriptions below reflect the implementation in `src/hp_folding/` and
the recorded experiment files. The principal implementation references are
[`sequence_processing.py`](../src/hp_folding/sequence_processing.py),
[`structure.py`](../src/hp_folding/structure.py),
[`energy.py`](../src/hp_folding/energy.py),
[`local_moves.py`](../src/hp_folding/local_moves.py),
[`simulated_annealing.py`](../src/hp_folding/simulated_annealing.py),
[`genetic_algorithm.py`](../src/hp_folding/genetic_algorithm.py),
[`hybrid_optimizer.py`](../src/hp_folding/hybrid_optimizer.py),
[`search_space.py`](../src/hp_folding/search_space.py), and
[`experiment_runner.py`](../src/hp_folding/experiment_runner.py).

## 1. Project abstract

This project implements and experimentally compares three stochastic
optimization methods for protein folding in a two-dimensional HP lattice
model: Simulated Annealing (SA), a Genetic Algorithm (GA), and a hybrid GA
with SA local refinement. Amino-acid sequences are validated and converted
to a binary hydrophobic/polar sequence. A conformation is represented as a
self-avoiding walk on the square lattice, and is scored by the negative
number of non-consecutive hydrophobic-hydrophobic contacts. The software
includes explicit structure validation, local lattice moves, GA operators,
search-space reduction, a shared objective-evaluation budget, reproducible
seeds, and run-level performance measurement.

The primary experiment comprised 1,200 runs across 40 synthetic and
protein-derived sequence cases, five lengths from 8 to 30 residues, and ten
independent replicates per algorithm and case. All runs completed with a
valid best structure under a common budget of 200 energy evaluations. Across
the mixed test set, GA attained the most favorable mean best-so-far energy
and the shortest median runtime. The hybrid was close in mean solution
quality but incurred greater runtime; SA improved its best-so-far scores but
was slower and had a less favorable mean score. Because the model is
simplified and the evaluation set is limited, these are results about the
implemented HP optimization problem, not evidence of native biological
structure prediction.

## 2. Introduction

Proteins are chains of amino acids whose three-dimensional conformations
affect their biological roles. Searching all possible conformations is
computationally challenging, so this project uses a simplified lattice model
to study the algorithmic problem of finding low-energy structures.

The project has four practical goals:

1. Convert amino-acid sequences into a computational representation.
2. Maintain structurally valid candidate walks on a 2D square lattice.
3. Implement and compare SA, GA, and a hybrid optimizer without an
   optimization-library dependency.
4. Measure solution quality and computational work using reproducible,
   per-run records.

The implementation is organized as a Python package under `src/hp_folding/`
and can be imported into a Kaggle Notebook. The optimization core uses
standard Python; experiment analysis and figures use Pandas, NumPy, and
Matplotlib. The package separates sequence processing, structure, energy,
operators, optimizers, experiment measurement, analysis, and visualization.

## 3. Problem formulation and mathematical model

### 3.1 Sequence abstraction

Let an amino-acid sequence have length \(N\):

\[
S=(s_1,s_2,\ldots,s_N).
\]

The project uses a fixed binary mapping \(f\):

\[
q_i=f(s_i)\in\{H,P\},
\]

where \(H\) denotes hydrophobic and \(P\) denotes polar. In the implemented
mapping, the hydrophobic set is `ACGILMFPWYV`; the polar set is `RNDQEKSTH`.
Input is normalized to uppercase with whitespace removed and is checked
against the 20 standard one-letter amino-acid codes. Unknown symbols are
rejected rather than silently classified.

This mapping is a modeling convention. In particular, amino-acid proline
(`P`) maps to the hydrophobic HP label `H`, whereas HP label `P` means polar.

### 3.2 Lattice conformation and constraints

Each residue \(i\) is assigned an integer coordinate
\(x_i=(a_i,b_i)\in\mathbb Z^2\). The chain begins at \((0,0)\), and each of
its \(N-1\) relative movements is one of
\[
R=(1,0),\quad L=(-1,0),\quad U=(0,1),\quad D=(0,-1).
\]

Coordinates are accumulated in sequence order. A valid structure must
satisfy:

**Connectivity**
\[
|a_{i+1}-a_i|+|b_{i+1}-b_i|=1
\quad\text{for }1\le i<N.
\]

**Self-avoidance**
\[
x_i\ne x_j\quad\text{for all }i\ne j.
\]

No two distinct residues may occupy the same lattice site. The
implementation also checks expected length, integer coordinate shape, and
connected unit steps. The validator returns a structured result with issues
and collision details.

### 3.3 Contacts and energy

Two residues are lattice neighbors when their Manhattan distance is one.
Only hydrophobic pairs that are **not adjacent in the original sequence**
contribute:

\[
C_{HH}(X)=
\sum_{\substack{1\le i<j\le N\\|i-j|>1}}
\mathbf 1[q_i=H]\mathbf 1[q_j=H]
\mathbf 1[\|x_i-x_j\|_1=1].
\]

The energy is
\[
E(X)=-\epsilon C_{HH}(X),\qquad \epsilon>0.
\]

The negative sign makes more H-H contacts correspond to lower energy.
Contacts are counted once by considering a contact only when the neighboring
residue has a larger sequence index; sequence-consecutive pairs are excluded.
The public energy function rejects invalid structures and returns both
energy and contact count as an `EnergyResult`.

The optimization problem is therefore
\[
\min_{X} E(X)
\]
subject to lattice connectivity and self-avoidance. It is a discrete
combinatorial optimization problem over valid walks; the algorithms search
heuristically and do not guarantee a global optimum.

## 4. Algorithm design

### 4.1 Simulated Annealing

SA starts from a valid structure, records its initial energy, and repeatedly
forms the set of unique valid structures reachable by one enabled local
move. One candidate is selected uniformly from this neighborhood. For
\[
\Delta E=E_{\text{candidate}}-E_{\text{current}},
\]
the candidate is accepted automatically when \(\Delta E\le0\); otherwise,
it is accepted with probability
\[
P=\exp(-\Delta E/T).
\]

The temperature follows geometric cooling:
\[
T_i=T_0\alpha^i,\qquad 0<\alpha<1.
\]

The implementation tracks the current and best-so-far structures
separately. Therefore, a later accepted uphill move can make the terminal
current state worse than the best state already found. Invalid proposals are
not evaluated or repaired.

### 4.2 Genetic Algorithm

An individual is a tuple of \(N-1\) relative directions. The initial
population is created from valid random self-avoiding walks.

- **Selection:** tournament selection samples configured contestants and
  chooses the individual with the lowest energy.
- **Crossover:** single-point crossover concatenates a prefix of one parent's
  direction sequence with a suffix of the other parent's sequence.
- **Mutation:** each direction gene is independently considered for mutation;
  when selected, it is replaced by one of the other three lattice directions.
- **Elitism:** the configured best individuals are copied unchanged to the
  next generation.

Crossover and mutation do not guarantee self-avoidance. Each offspring is
decoded direction by direction; decoding stops at the first occupied lattice
coordinate. Invalid offspring are counted and rejected. Bounded retries are
used to fill the population; failure to do so is reported rather than hidden
by inserting an invalid structure or silently repairing one.

The GA uses energy directly as a minimization score; tournament selection
prefers the lower-energy candidate. Population histories include best and
average energy and diversity (fraction of distinct genomes).

### 4.3 Hybrid GA + SA

The hybrid uses the GA population for global exploration and periodically
refines promising individuals with the existing SA implementation. It
selects the lowest-energy fraction of the current population at the
configured frequency. An individual is replaced only if SA finds a lower
energy and the replacement does not duplicate another population member.
The global best is explicitly preserved.

The GA and nested SA share one `EvaluationBudget`; local refinement cannot
exceed the run's total objective-evaluation cap. This design balances
population-based exploration with local exploitation, but it does not
guarantee that the hybrid outperforms either component.

### 4.4 Valid local moves

The move module implements:

- **End move:** relocates a terminal residue to an unoccupied lattice
  neighbor of its fixed chain anchor.
- **Corner move:** moves an internal corner residue to the alternate vertex
  of the unit square formed by its two chain neighbors.
- **Crankshaft move:** reconfigures two consecutive internal residues between
  fixed anchors, selecting a valid alternative adjacent path.

Proposals are checked for collisions and connectivity. Neighborhood
generation retains unique valid structures and records attempted, accepted,
rejected, duplicate, and restricted-operator counts.

## 5. Data structures and software architecture

| Data structure | Role |
|---|---|
| `ProcessedSequence` | Holds normalized amino-acid sequence, HP sequence, and length. |
| `ProteinStructure` | Immutable ordered tuple of integer coordinate pairs; each coordinate corresponds to a sequence position. |
| Direction tuple | GA genome of \(N-1\) `Direction` values; also used to persist final structures. |
| `ValidationResult` | Validity flag, issue messages, and detected coordinate collisions. |
| Coordinate set / coordinate-to-index dictionary | Occupancy checking, coordinate lookup, and contact calculation. |
| `EnergyResult` | Energy and non-consecutive H-H contact count. |
| `IncrementalEnergyTracker` | Occupancy map, contact-pair set, current energy, and delta evaluation for SA moves. |
| Candidate / population records | Genome, validated structure, and energy for GA and hybrid population members. |
| Configuration and result dataclasses | Validated algorithm parameters, outputs, histories, and counters. |
| `EvaluationBudget` | Serial shared counter that prevents energy-evaluation calls from exceeding the configured limit. |
| Pandas DataFrames and CSV records | Run-level measurements, summaries, convergence histories, and experiment metadata. |

The dependency flow is from sequence processing and structure validation to
energy and local operators, then to the optimizers, experiment runner,
analysis, and visualization. Visualization reconstructs recorded directions,
validates them, and checks their energy independently before plotting.
Further module details are in [`software-architecture.md`](software-architecture.md).
The latest full test-suite run completed **101 tests successfully**; the
project README documents the command to reproduce it.

## 6. Search-space reduction

The direction representation has \(4^{N-1}\) possible raw direction strings.
Only a subset produce self-avoiding walks. A coarse upper bound on valid
self-avoiding walks for \(N\ge2\) is \(4\cdot3^{N-2}\), since after the first
step an immediate reversal is forbidden; other collisions reduce the count
further. The project does not enumerate all valid conformations.

Implemented or measured reduction mechanisms include:

1. **Early collision rejection:** GA decoding checks occupancy during
   construction and stops at the first collision.
2. **Duplicate detection:** repeated genomes or symmetry-equivalent
   structures can avoid repeated energy evaluations.
3. **Symmetry canonicalization:** the eight rotations/reflections of the
   square lattice are translated to canonical coordinates; chain order is
   preserved.
4. **Incremental energy:** SA updates contacts affected by moved residues
   rather than recounting all contacts for each accepted move. Changed-index
   discovery and validation still scan the chain.
5. **Restricted neighborhoods:** selected move types can reduce candidate
   generation but may remove useful paths from the reachable state graph.
6. **Elitism and stopping rules:** retain good solutions and cap work, but
   do not eliminate the global conformation space.

The primary experiment recorded approximately 502 invalid candidates and 223
duplicate candidates per GA run; GA's symmetry setting was off, so these
duplicates refer to duplicate-genome handling. SA recorded about 6,746
invalid proposals and 2,079 symmetry-equivalent duplicate proposals avoided
per run. These figures are distinct from the 200 energy evaluations per run.
Candidate-generation counters have different semantics across algorithms
and should not be treated as equal units of work.

Separate small component experiments found that enabling D4 canonicalization
produced no symmetry-only duplicates in the tested five-seed GA sample and
increased mean time from 0.0130 to 0.0257 seconds. A restricted SA move set
reduced mean runtime by about 44% but worsened mean best energy from −3.8 to
−3.4 in that experiment. Incremental contact evaluation reduced contact
neighbor checks by about 86.5% relative to an estimated full-scan count; this
is a count reduction, not a measured runtime speedup. Details and caveats are
documented in [`search-space-reduction.md`](search-space-reduction.md).

## 7. Complexity analysis

Let:

- \(N\): protein length;
- \(I\): maximum SA iterations;
- \(P\): GA population size;
- \(G\): maximum GA generations;
- \(K\): total SA local-search iterations performed across hybrid refinements;
- \(A_i\): initialization attempts allowed per individual;
- \(A_o\): offspring attempts allowed per generation;
- \(t\): tournament size;
- \(E_e\): number of elite individuals retained per generation.

Bounds below assume expected \(O(1)\) coordinate set/dictionary operations.

| Component | Time complexity | Additional space |
|---|---:|---:|
| Amino-acid/HP conversion | \(\Theta(N)\) | \(\Theta(N)\) |
| Direction decoding / coordinate construction | \(\Theta(N)\) for a valid walk; early collision may stop sooner | \(\Theta(N)\) |
| Structure validation | Expected \(\Theta(N)\); worst-case hash behavior can degrade | \(\Theta(N)\) |
| Full energy evaluation | Expected \(\Theta(N)\) | \(\Theta(N)\) |
| One materialized SA neighborhood with the default crankshaft move | Expected / worst-case \(O(N^2)\) | Worst-case \(O(N^2)\) |
| SA for \(I\) iterations | \(O(IN^2)\) for the current full-neighborhood implementation | \(O(N^2+I)\) including neighborhood and histories |
| GA, including bounded retries | \(O(PA_iN + G(PN + P\log(E_e+1) + A_o(N+t)))\) | \(O(PN+G)\) |
| Hybrid GA + SA | GA cost plus local-round selection/replacement checks and \(O(KN^2)\) SA work | \(O(PN+N^2+G+I)\) for sequential refinements |

Here \(E_e\) is the number of elites retained. The GA bound includes
bounded retries; it does not assume every crossover/mutation yields a valid
offspring. The SA bound reflects the implementation's materialization of a
complete one-move neighborhood before selecting a candidate. Incremental
contact evaluation reduces neighbor checks but does not change the
end-to-end \(O(N)\) changed-index/commit work, nor the \(O(N^2)\)
neighborhood-generation bound. These are theoretical bounds, not predictions
of elapsed seconds. The project's detailed assumptions and best/worst-case
notes are in [`complexity-analysis.md`](complexity-analysis.md).

## 8. Experimental methodology

### 8.1 Test cases

The primary matrix includes four synthetic HP patterns and four
protein-derived cases at each length \(N\in\{8,12,16,20,30\}\):

- synthetic: hydrophobic-rich block, polar-rich block, balanced blocks, and
  alternating H/P;
- protein-derived: N-terminal prefixes from hemoglobin beta (HBB, P68871),
  insulin precursor (INS, P01308), lysozyme C precursor (LYZ, P61626), and
  cytochrome c (CYCS, P99999).

The protein cases are prefix inputs, not complete native-protein folding
measurements. The exact sequences, mappings, and source versions are retained
in the experimental protocol and run records.

### 8.2 Replicates, budget, and configuration

Each of the 40 cases was run ten times with fixed root seeds
`1101, 2027, 3037, 4051, 5059, 6067, 7079, 8081, 9091, 10007`.
SHA-256 derives a stable 32-bit algorithm/case/replicate seed. The resulting
matrix is \(40\times10\times3=1{,}200\) runs.

The shared budget was **200 energy evaluations per run**, including initial
scoring. GA and hybrid used population size 20, tournament size 3,
crossover probability 0.8, mutation probability \(1/(N-1)\) per direction
gene, and one elite. GA symmetry reduction, target-energy stopping, and
no-improvement stopping were disabled in the main matrix. Initial population
attempts were capped at 100 per individual, offspring attempts at 2,000 per
generation, and the generation limit was 200.

SA used \(T_0=10\), cooling rate 0.995, minimum temperature 0, a maximum of
200 iterations, and enabled end, corner, and crankshaft moves. The hybrid
used the same GA and SA settings, scheduled local refinement every five
generations, and selected the best 20% for refinement. GA and nested SA
shared the same evaluation budget. The primary stopping condition was the
evaluation budget.

Wall time used `time.perf_counter`; CPU time used `time.process_time`; peak
Python-traced memory used an independent `tracemalloc` session per run.
Timing includes optimizer initialization (including SA's initial structure
generation) and excludes the independent result validation and CSV work.
Memory is Python-traced allocation, not process RSS or native-library
memory. Equal objective-evaluation counts do not imply equal CPU work.

## 9. Results and discussion

### 9.1 Completion and validity

All 1,200 runs completed, used 200 energy evaluations, and returned a valid
best structure. This is a 100% success/validity rate for the **returned best
structures**. It does not mean all generated proposals were valid: each
algorithm rejected many invalid candidates.

### 9.2 Best-so-far energy and efficiency

| Algorithm | Mean best energy ± SD | Median best energy | Observed best | Median wall time | Mean CPU time | Mean peak traced memory |
|---|---:|---:|---:|---:|---:|---:|
| GA | −4.210 ± 3.329 | −4 | −16 | 0.025 s | 0.029 s | 0.0209 MiB |
| Hybrid GA + SA | −4.065 ± 3.228 | −3 | −15 | 0.316 s | 0.385 s | 0.0253 MiB |
| SA | −3.073 ± 2.410 | −3 | −13 | 0.569 s | 0.706 s | 0.0243 MiB |

The worst observed best energy was 0 for each algorithm. GA had the strongest
aggregate best-so-far score and shortest median runtime. The hybrid was close
in mean energy but substantially slower. By per-case mean best energy, GA
beat the hybrid on 16 cases, tied on 18, and lost on 6; both GA and hybrid
were at least as good as SA in the 40-case comparisons. These are descriptive
comparisons, not significance tests.

Use the `best_energy` field when comparing best-found solutions. SA's mean
`final_energy` was −1.090, while its mean `best_energy` was −3.073, because
the terminal accepted state can be worse than the best-so-far state.

Mean generated proposals per run were approximately 925 for GA, 6,943 for
the hybrid, and 11,542 for SA. Mean invalid proposals were approximately
502, 4,394, and 6,746, respectively. Proposal definitions differ by method,
so equal energy budgets—not equal proposal counts—are the fair comparison
basis used here.

### 9.3 Results by length

Mean best energy became more negative with length, while median runtime
generally increased. The change in energy reflects more potential H-H
contacts and different sequence compositions; it is not evidence that longer
sequences were predicted more accurately.

| Length | GA: mean best energy / median time | Hybrid: mean best energy / median time | SA: mean best energy / median time |
|---:|---:|---:|---:|
| 8 | −1.625 / 0.013 s | −1.613 / 0.112 s | −1.563 / 0.193 s |
| 12 | −2.900 / 0.022 s | −2.663 / 0.220 s | −2.313 / 0.408 s |
| 16 | −4.163 / 0.027 s | −4.113 / 0.317 s | −3.138 / 0.570 s |
| 20 | −5.013 / 0.026 s | −4.863 / 0.424 s | −3.338 / 0.816 s |
| 30 | −7.350 / 0.043 s | −7.075 / 0.839 s | −5.013 / 1.596 s |

The experiment only reaches length 30, which is not sufficient to make
asymptotic runtime claims. The hybrid's maximum wall time was 7,087.7
seconds for one run while that run used 1.23 CPU seconds; SA also had a
115.9-second maximum wall-time record. The cause was not established.
Medians are more informative for typical elapsed time in this dataset.

### 9.4 Convergence

Mean best-so-far energy at selected evaluation checkpoints was:

| Algorithm | 20 evaluations | 100 evaluations | 200 evaluations |
|---|---:|---:|---:|
| GA | −3.568 | −4.033 | −4.210 |
| Hybrid GA + SA | −3.523 | −3.900 | −4.065 |
| SA | −2.208 | −2.798 | −3.073 |

GA had the most favorable mean at these checkpoints. SA improved steadily,
but remained less favorable on average. SA records per-iteration history;
GA and hybrid histories are generation-grained, so improvement times within
a generation are not observed exactly.

### 9.5 Search-space measurements

The primary records show that early collision rejection and duplicate
detection avoided work, while the component experiments show that reduction
techniques can also add overhead or limit solution quality. D4 symmetry
canonicalization had no measured duplicate benefit in its small GA
comparison. Restricted SA moves reduced runtime but worsened energy in the
tested sample. These findings do not establish which reduction method is
best for every sequence or length.

The results report invalid candidates, duplicate proposals where instrumented,
and objective evaluations. They do not measure the total number of valid
conformations never generated. No claim is made that heuristic techniques
mathematically remove all possible structures or find the global optimum.
The hybrid made an average of one local-search call per run and recorded 34
local-search replacements across 400 runs (0.085 per run), under the shared
budget and configured refinement schedule.

The aggregate CSVs and plots are under [`results/analysis/`](../results/analysis/);
the detailed analysis is in [`results-and-discussion.md`](results-and-discussion.md).

## 10. Limitations

### Model limitations

- The lattice is two-dimensional, while proteins fold in three dimensions.
- The 20-amino-acid alphabet is reduced to two classes, losing residue-specific
  chemistry.
- The objective accounts only for non-consecutive H-H contacts.
- It omits explicit hydrogen bonds, electrostatics, solvent, side-chain
  geometry, disulfide constraints, bond-angle preferences, and other physical
  terms.
- Lower HP energy is a score improvement within this abstraction, not proof
  of native structure or biological function.

### Experimental limitations

- Only four synthetic patterns and four protein-derived prefixes were used;
  prefixes of one protein are nested and are not independent biological
  samples.
- Lengths end at 30 residues.
- Ten stochastic runs per case provide descriptive evidence, but the
  reported analysis does not include inferential tests or confidence
  intervals.
- A common objective-evaluation budget does not equalize proposal generation,
  collision checking, or CPU instructions.
- SA, GA, and hybrid have different candidate-generation mechanics and
  different symmetry handling in the primary setup.
- Wall time contains extreme outliers; resource scheduling conditions were
  not established from the records.

## 11. Future improvements

1. Extend experiments to longer sequences and broader, preregistered sequence
   sets, while distinguishing complete proteins from selected prefixes.
2. Add confidence intervals, effect sizes, and suitable per-case statistical
   comparisons with multiple-comparison correction.
3. Run evaluation-budget sensitivity experiments and report them separately
   from the 200-evaluation primary matrix.
4. Harmonize symmetry handling across algorithms or use controlled ablations
   to isolate its impact.
5. Benchmark local moves, mutation/crossover policies, and local-search
   frequency under matched budgets.
6. Improve wall-time reproducibility with controlled hardware/runtime
   conditions and repeated timing measurements.
7. Explore richer energy functions and three-dimensional lattice models,
   validating each extension against an appropriate structural reference.
8. Continue testing incrementally optimized energy and neighborhood
   generation against the canonical implementations.

These are proposed directions; no results for them are claimed here.

## 12. Conclusion

The project provides a complete, reproducible implementation of SA, GA, and
hybrid GA + SA for a 2D HP lattice optimization problem. It represents
structures as self-avoiding walks, validates candidate conformations
explicitly, minimizes a transparent H-H contact energy, and records
per-run quality and performance measures.

Under the tested 200-evaluation budget, GA had the most favorable mean
best-so-far energy and shortest median runtime. The hybrid was competitive
in solution quality but incurred greater computational cost, while SA
showed improvement across checkpoints but had a weaker aggregate score and
longer median runtime. The findings are specific to the tested sequences,
settings, and 2D HP objective. They demonstrate optimization behavior in a
simplified model, not biological accuracy or universal algorithm
superiority.

## 13. Potential professor questions and technically grounded answers

### Q1. Why did you choose the HP model?

**Answer:** It converts a difficult folding problem into a manageable
combinatorial optimization task. The binary representation and contact
objective make candidate structures, validity, and algorithm behavior
transparent. It is appropriate for studying search methods, but it omits
many physical details and is not a full protein-structure predictor.

### Q2. Why use Simulated Annealing?

**Answer:** SA makes a sequence of local moves and can sometimes accept
worse-energy candidates with probability \(\exp(-\Delta E/T)\). This
temperature-dependent acceptance provides a mechanism to leave local
minima. The code also tracks the best-so-far structure separately from the
current state.

### Q3. Why use a Genetic Algorithm?

**Answer:** GA maintains a population of candidate walks and explores
multiple regions of the search space using selection, crossover, and
mutation. Tournament selection and elitism are straightforward to
implement and measure. Because its offspring can be invalid, the code
explicitly validates candidates rather than assuming genetic operators
preserve lattice feasibility.

### Q4. Why combine GA and SA?

**Answer:** GA supplies population-level exploration, while SA refines
selected individuals locally. The hybrid reuses the existing implementations
and shares one evaluation budget. It is a hypothesis about balancing
exploration and exploitation, not a guarantee of better results; in this
experiment it did not consistently outperform GA.

### Q5. How is energy calculated, and why is lower energy better?

**Answer:** The code counts each adjacent lattice pair of hydrophobic
residues once, excludes sequence neighbors with \(|i-j|=1\), and computes
\(E=-\epsilon C_{HH}\). With \(\epsilon>0\), more non-consecutive H-H
contacts make energy more negative. Invalid structures are rejected before
public energy evaluation.

### Q6. How do you prevent counting a contact twice?

**Answer:** For each residue, the energy implementation checks its lattice
neighbors and counts a contact only when the neighboring residue has a
larger sequence index. This gives one count per pair. Consecutive sequence
indices are excluded separately.

### Q7. How do you ensure a structure is valid?

**Answer:** Coordinates are integer lattice positions. The validator checks
expected length, unit Manhattan distance between consecutive residues, and
unique occupancy of coordinates. End, corner, and crankshaft moves return a
candidate only if its constraints hold. Genetic offspring are decoded
incrementally and rejected at a collision; they are not silently repaired.

### Q8. What does the GA chromosome represent?

**Answer:** It is a direction sequence of length \(N-1\), using `R`, `L`,
`U`, and `D`. Accumulating these steps from the origin produces the
coordinates of the \(N\)-residue chain.

### Q9. How do crossover and mutation work, and can they produce invalid walks?

**Answer:** Single-point crossover joins a prefix from one parent's direction
sequence to a suffix from the other. Mutation independently selects genes
and replaces a selected direction with one of the other three. Either
operator can produce self-intersections. The resulting direction sequence
is decoded and checked; invalid offspring are counted and discarded, with
bounded attempts to fill the next generation.

### Q10. How does selection favor better individuals?

**Answer:** Tournament selection samples a configured number of individuals
and selects the one with the lowest energy. The GA minimizes energy directly;
it does not transform energy into a separate reciprocal fitness measure.
Elitism copies the configured best individuals into the next generation.

### Q11. How is the hybrid's local search applied?

**Answer:** In the primary experiment, the best 20% of the population are
scheduled for refinement every five generations. SA starts from each
selected valid structure. A refined result replaces the individual only if
it has lower energy and does not duplicate another population member. The
global best is preserved.

### Q12. How did you compare the algorithms fairly?

**Answer:** Each case/algorithm combination used ten independent seeded
runs and a shared limit of 200 objective evaluations, including initial
scoring. The hybrid shares this limit between GA and nested SA. The same
budget does not mean equal CPU work, so runtime, CPU time, memory, invalid
proposals, and candidate counts were also recorded. Runtime outliers and
generation-grained histories are reported as measurement limitations.

### Q13. How do you measure search-space reduction?

**Answer:** The implementation records invalid candidates, duplicate
candidates avoided, early collision rejections, and energy evaluations.
These are separate counts: an invalid candidate is rejected for feasibility,
a duplicate is a repeated/equivalent valid state, and an evaluation is a
scored structure. The project does not claim to count every conformation
never generated or to eliminate the entire search space.

### Q14. What does the 100% validity result mean?

**Answer:** All returned best structures passed independent validation. It
does not mean all proposals were valid; the algorithms rejected many invalid
candidates during search.

### Q15. Which algorithm performed best?

**Answer:** Under this experiment's settings and budget, GA had the most
favorable mean best-so-far energy and shortest median runtime. It was not
best on every case: the hybrid beat it in six case-level mean comparisons
and tied it in eighteen. The evidence is descriptive, not a statistical
proof of universal superiority.

### Q16. Why can SA's final energy differ from its best energy?

**Answer:** SA can accept a higher-energy move to continue exploration.
Therefore, its current state at termination can be worse than a structure
it found earlier. The result object preserves both the terminal energy and
the best-so-far structure/energy.

### Q17. What are the main complexity bottlenecks?

**Answer:** Validating a structure and evaluating energy are expected
\(O(N)\). The current SA implementation materializes a full one-move
neighborhood; with crankshaft moves, that costs \(O(N^2)\) time and space in
the worst case per iteration, so \(I\) iterations cost \(O(IN^2)\). GA work
depends on population, generations, sequence length, and bounded invalid
offspring retries. Hybrid local SA adds that neighborhood cost.

### Q18. Why is the 2D HP model only a simplification?

**Answer:** It reduces each residue to H or P, uses a 2D lattice, and scores
only non-consecutive H-H contacts. It does not model three-dimensional
geometry, detailed residue chemistry, solvent, hydrogen bonds, or many other
physical interactions. Consequently, low energy is meaningful only within
this model and does not demonstrate a native protein fold.

### Q19. How are experiments reproducible?

**Answer:** The runner records exact inputs, configuration, replicate and
derived seeds, algorithm, status, energies, counters, and serialized best
directions. Seeds are derived deterministically with SHA-256, and the same
evaluation cap is applied per run. Runtime remains dependent on the machine
and its scheduling conditions.

### Q20. What does the memory measurement include?

**Answer:** `tracemalloc` reports peak traced Python allocations during a
run. It is not process RSS and does not include all native-library memory.
The method is started independently for each measured run.

### Q21. What is the most important conclusion to defend?

**Answer:** The implementation reliably optimizes and compares valid
structures under a clearly defined 2D HP objective. In the measured matrix,
GA gave the strongest overall combination of best-so-far energy and median
runtime. These conclusions are limited to the tested model and settings;
they should not be presented as validated biological protein prediction.

## Implementation and evidence references

- [Reproducible evaluation protocol](experimental-evaluation-protocol.md)
- [Theoretical complexity analysis](complexity-analysis.md)
- [Search-space reduction measurements](search-space-reduction.md)
- [Results and discussion](results-and-discussion.md)
- [Run-level experimental CSV](../results/experiment_runs.csv)
- [Algorithm summary CSV](../results/analysis/algorithm_summary.csv)
- [Length summary CSV](../results/analysis/length_algorithm_summary.csv)
- [Convergence CSV](../results/analysis/convergence_curves.csv)
- [Project README and run instructions](../README.md)
