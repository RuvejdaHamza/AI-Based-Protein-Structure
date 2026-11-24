# Theoretical Complexity Analysis

This analysis describes the current Python implementation, not an idealized
protein-folding algorithm. In particular, SA materializes a complete
one-move neighborhood, GA has bounded retry loops and explicit population
statistics, and the hybrid runs SA separately on selected individuals.

## Variables and analysis model

| Symbol | Meaning |
|---|---|
| \(N\) | Number of residues in the protein; a direction genome has \(N-1\) moves. |
| \(I\) | Maximum number of iterations in one Simulated Annealing (SA) run; actual iterations can be smaller due to stopping conditions. |
| \(P\) | Genetic Algorithm (GA) population size. |
| \(G\) | Maximum number of GA generations; early stopping may reduce the actual count. |
| \(K\) | Total number of SA local-search iterations performed across all hybrid refinements. One such iteration builds a neighborhood and proposes one move. |
| \(Q\) | Number of SA refinement calls made by the hybrid. This is separate from \(K\); each call executes at most \(I\) iterations, so \(K \le QI\). |
| \(A_i\) | Maximum initial-structure attempts per population member. |
| \(A_o\) | Maximum offspring-generation attempts per GA generation. |
| \(R\) | Maximum random-walk restarts in one call to the valid-structure generator. |
| \(B\) | Maximum candidate direction steps per random-walk restart. |
| \(E\) | Number of elites retained per generation. |
| \(t\) | Tournament size. |
| \(m\) | Number of changed residues in one incremental energy proposal; \(0 \le m \le N\). |
| \(F\) | Number of generations between hybrid local-search rounds. |
| \(L\) | Individuals refined in one local-search round; when enabled, \(L=\max(1,\lceil P f\rceil)\) for configured fraction \(f\). |
| \(H\) | Number of local-search rounds; \(H=\lfloor G/F\rfloor\) when the local-search fraction is positive. Thus \(Q=HL\). |

Unless stated otherwise, time bounds use a word-RAM model in which coordinate
arithmetic and expected dictionary/set lookup take \(O(1)\). Python dictionaries
and sets provide expected/amortized constant-time lookup, not a deterministic
worst-case guarantee; pathological hash collisions can make a sequence of
lookups quadratic. The bounds below call out that distinction. Coordinates of
a walk differ from its first coordinate by at most \(N-1\) in either axis.
When anchored at the origin, each coordinate therefore occupies
\(O(\log N)\) bits; with an arbitrary start coordinate \((x_0,y_0)\), absolute
coordinate storage also depends on the bit lengths of \(x_0,y_0\).

## Component analysis

### 1. Protein representation

`ProteinStructure.from_directions` processes each of the \(N-1\) directions
once and creates \(N\) coordinate pairs. The `ProteinStructure` object stores
the ordered coordinates; its tuple-preserving normalization still scans the
tuple to check its shape.

The amino-acid-to-HP conversion and validation of a length-\(N\) sequence
similarly take \(\Theta(N)\) expected time and \(O(N)\) space for the returned
normalized/HP strings. The amino-acid alphabet is fixed at 20 symbols, so
membership tests do not grow with \(N\).

| Operation | Time | Additional space |
|---|---:|---:|
| Build coordinates from directions | \(\Theta(N)\) | \(\Theta(N)\) for the returned coordinates |
| Recover directions | \(\Theta(N)\) | \(\Theta(N)\) for the returned directions |
| Build coordinate-to-residue mapping | \(\Theta(N)\) expected | \(\Theta(N)\) |
| Tuple-preserving post-initialization check | \(\Theta(N)\) | \(\Theta(1)\) beyond the input |
| Normalize non-tuple coordinates | \(\Theta(N)\) | \(\Theta(N)\) for the normalized tuple |

Under the word-RAM model, the stored representation is \(\Theta(N)\) words.
When anchored at the origin, counting coordinate bit lengths, it is
\(O(N\log N)\) bits. For arbitrary starting coordinates add their bit lengths
to the storage per coordinate. The sequence and direction genome each use
\(\Theta(N)\) symbols. Rendering a textual structure visits all residues; its
character output is \(O(N\log N)\) for origin-anchored walks, with additional
digits possible for arbitrary large starting coordinates.

These are representation costs only; they do not include self-avoidance
validation. The linear scan is also the best and worst asymptotic case for
fixed-length conversion/recovery operations: each direction or coordinate is
visited.

### 2. Structure validation and random valid structure generation

`validate_structure` makes one pass over all coordinates. It checks coordinate
types, neighboring Manhattan distances, and repeated coordinates while
collecting all applicable issues. It does not stop at the first malformed
coordinate, so even an early error does not make the traversal constant-time.

- **Expected time:** \(\Theta(N)\), assuming expected \(O(1)\) hash-table
  operations.
- **Best case:** still \(\Theta(N)\), because the implementation completes the
  coordinate traversal to collect all issues rather than returning on the
  first malformed coordinate.
- **Hash-table worst case:** \(O(N^2)\) if operations on the coordinate map
  degrade to \(O(N)\) each.
- **Additional space:** \(O(N)\) for the first-index map and, when needed,
  collision and issue details. A valid input still needs the first-index map.

`generate_random_valid_structure` uses randomized backtracking. Each restart
tries at most \(B\) candidate directions; each successful step can later be
undone at most once, so the total backtracking work per restart is also
\(O(B)\). Creating a complete walk and validating it costs \(O(N)\).

- **Bounded expected time:** \(O(R(B+N))\) under expected constant-time set
  operations. This is also the bounded asymptotic upper bound in the word-RAM
  model; the routine can fail after consuming its configured limits.
- **Best case:** \(O(N)\) when a complete valid walk is found with only
  \(O(N)\) candidate steps and no substantial backtracking.
- **Additional space:** \(O(N)\) for the partial coordinates, occupancy set,
  and per-depth direction choices.
- In the GA and hybrid initializers, each call sets \(R=1\) and
  \(B=\max(100,100N)=\Theta(N)\), making one generation attempt \(O(N)\)
  expected time and \(O(N)\) space.

These bounds depend on explicit search limits. They do not imply that
backtracking will find a valid walk within those limits for every input.

### 3. Energy evaluation

The public `calculate_energy` validates the HP labels and structure, builds a
coordinate-to-index dictionary, and checks at most four lattice neighbors per
residue. The private `_calculate_energy_validated` skips validation but still
builds the coordinate map and scans the residues.

- **Public expected time:** \(\Theta(N)\).
- **Trusted-valid expected time:** \(\Theta(N)\).
- **Best case on a non-empty valid input:** \(\Theta(N)\); even an all-polar
  sequence builds the coordinate map and traverses the structure.
- **Worst case with adversarial hash-table collisions:** \(O(N^2)\) for the
  map operations.
- **Additional space:** \(\Theta(N)\) for the coordinate map.

The four-neighbor check is constant work per residue; contacts are counted
once by index ordering, and sequence-adjacent pairs are excluded. The private
fast path is correct only when its caller has already established that the
coordinate list is valid, connected, self-avoiding, and the same length as
the HP sequence. The public API retains validation.

### Local move and neighborhood costs

The public `end_move`, `corner_move`, and `crankshaft_move` validate the source
and then validate a copied candidate when one is proposed. Each uses
\(\Theta(N)\) expected time and \(O(N)\) additional space. A crankshaft move
enumerates at most a constant number of ordered pairs (at most \(4\cdot4\))
after constructing occupancy information.

`generate_neighborhood` validates the source once and loops over \(O(N)\)
possible corner and crankshaft positions. The number of proposals per position
is bounded by a constant, but every retained candidate copies \(N\)
coordinates and computes a symmetry key across all eight square-lattice
transforms, also \(\Theta(N)\). In addition, for every crankshaft start it
constructs `fixed_occupied = occupied - moving`, a set difference that scans
\(O(N)\) coordinates even when no alternative is retained.

- **Expected best case with crankshaft enabled:** \(\Theta(N^2)\), due to the
  per-start set difference over \(\Theta(N)\) crankshaft starts. The default
  move set includes crankshaft, so this cost is present even in a sparse
  neighborhood.
- **Expected best case with only end and/or corner moves:** \(\Theta(N)\),
  including source validation/canonicalization and the \(O(N)\) corner scan;
  end-only has a constant number of proposals after source setup.
- **Individual local-move best case:** \(\Theta(N)\) for a valid source,
  because source validation scans the chain even when a proposal is rejected
  by a constant-time geometric check.
- **Expected worst case:** \(O(N^2)\), when \(\Theta(N)\) candidates are
  retained and each requires \(\Theta(N)\) copying and canonicalization.
- **Additional space:** \(O(N^2)\) worst case because up to \(O(N)\) structures
  and canonical keys of length \(N\) are materialized. The source occupancy
  set and temporary candidate use \(O(N)\).
- **Hash-table worst case:** membership and duplicate-detection operations can
  increase time beyond these expected bounds under pathological collisions.

Selecting a subset of move types reduces the constant and can reduce the
number of candidates. It does not necessarily change the asymptotic bound:
corner-only or crankshaft-only neighborhoods can still have \(O(N)\) proposals
and \(O(N^2)\) construction cost. End-only proposals are constant in number,
but source validation and structure copying still cost \(\Theta(N)\).

### 4. Simulated Annealing

SA validates its initial structure and creates an `IncrementalEnergyTracker`,
which validates again and builds occupancy/contact data, all in
\(\Theta(N)\) expected time and space. In each iteration it calls
`generate_neighborhood`, materializes the complete neighborhood, selects one
candidate, and updates energy.

Incremental energy evaluation is not \(O(m)\) end-to-end in this
implementation: it scans all \(N\) coordinate pairs to discover the changed
residues, then checks at most four neighbors per changed hydrophobic residue.
Commit also scans all \(N\) coordinate pairs before updating occupancy and
contact sets. Thus evaluation plus an accepted commit costs \(O(N+m)\)
expected time; it remains \(O(N)\) even for a move affecting only a few
residues. Its contact-neighbor work is \(O(m)\).

- **Worst-case time:** \(O(I N^2)\) expected for the current full-neighborhood
  implementation; neighborhood construction dominates the \(O(N+m)\)
  incremental score update. Initial validation/tracker construction adds
  \(O(N)\), and history recording adds \(O(I)\).
- **Best meaningful case:** \(O(N)\) when the initial target is already met,
  the iteration limit is zero, or the first neighborhood has no valid moves
  under a move set without crankshaft. With crankshaft enabled, an executed
  iteration costs \(\Theta(N^2)\) because of the per-start occupancy-set
  difference, even when few candidates are retained. For a move set without
  crankshaft and sparse candidate retention, \(I\) executed iterations can
  take \(O(IN)\).
- **Peak additional space:** \(O(N^2+I)\). The \(O(N^2)\) term is the
  materialized neighborhood; \(O(I)\) stores best/current energy histories.
  The tracker and current/best structures use \(O(N)\).

The bound assumes expected hash-table operations. Termination can occur before
\(I\) because of minimum temperature, target energy, no improvement, or no
valid neighbors; in formulas, replace \(I\) with the actual iteration count
for a particular run.

### 5. Genetic Algorithm

Let \(A_i\) and \(A_o\) denote the configured per-member initialization and
per-generation offspring-attempt caps. Initialization makes at most
\(P A_i\) attempts. Each attempt calls the bounded random structure generator
with \(R=1\), \(B=\Theta(N)\), then encodes and scores the resulting genome.
Its expected total cost is \(O(P A_i N)\).

For each generation:

- Elite selection uses `heapq.nsmallest(E, population)`, taking
  \(O(P\log E)\) time for \(E>1\), and \(O(P)\) for \(E=1\).
- A tournament selects \(t\) members and finds the minimum in \(O(t)\);
  two tournaments are run per offspring attempt.
- Crossover and mutation scan/copy at most \(N-1\) genes: \(O(N)\).
- Decoding a genome is \(O(N)\) worst case and can stop at an early collision.
  A valid walk necessarily consumes all \(N-1\) directions.
- Optional D4 symmetry canonicalization is \(O(N)\); exact-genome duplicate
  keys also require \(O(N)\) to hash/compare in the worst case.
- Energy scoring is \(O(N)\) expected. Population statistics and diversity
  require \(O(PN)\) expected time because genome tuples are hashed.

Consequently, a useful expected upper bound for a completed run is:

\[
O\!\left(
P A_i N
+ G\left(PN + P\log(E+1) + A_o(N+t)\right)
\right).
\]

The bound uses the configured retry cap \(A_o\), not an assumption that all
offspring attempts succeed. A generation may raise an error if its cap cannot
fill the population. Early stopping makes the actual generation count at most
\(G\). For \(t\le P\), the tournament work is covered by the displayed terms.
Hash-table worst cases can exceed this expected bound.

- **Best meaningful case:** after the required initial population is built,
  an initial target-energy hit or \(G=0\) avoids generation work. Initialization
  still costs at least \(\Omega(PN)\) because it constructs \(P\) structures.
- **Worst-case space:** \(O(PN+G)\). Current and next populations, duplicate
  keys, and candidate structures require \(O(PN)\); result histories require
  \(O(G)\). In-flight decoding uses \(O(N)\).

Elitism is implemented with bounded top-\(E\) selection, not a full sort unless
\(E\) is comparable to \(P\). The default disables D4 symmetry reduction, but
duplicate handling remains available; turning symmetry reduction on adds
linear canonical-key work per candidate.

### 6. Hybrid Genetic Algorithm + Simulated Annealing

The hybrid performs the GA work above, then at \(H=\lfloor G/F\rfloor\)
eligible generations selects \(L\) promising individuals and runs SA on each.
The number of SA calls is \(Q=HL\). Each call executes at most \(I\) iterations;
\(K\), the total number of executed local-search iterations, satisfies
\(K\le QI\).

Besides the GA cost, local rounds sort the population in \(O(HP\log P)\).
Each improved candidate can be checked against the population for duplicate
replacement in \(O(PN)\); in the worst case this occurs for all \(Q\) calls.
The local SA iterations cost \(O(KN^2)\) in total. Therefore an expected
upper-bound expression is:

\[
O\!\left(
P A_i N
+ G\left(PN + P\log(E+1) + A_o(N+t)\right)
+ HP\log P
+ QPN
+ KN^2
\right).
\]

Here \(QPN\) bounds worst-case duplicate-replacement comparisons, and also
covers the \(O(QN)\) initialization cost of the SA calls for \(P\ge1\). If no
local improvements occur, actual replacement checks are fewer. If the local
fraction is zero, \(Q=K=H=0\), and the algorithm reduces to GA plus hybrid
bookkeeping.

- **Best meaningful case:** a target reached by the initial population costs
  \(O(P A_i N)\) and runs no generations or local searches.
- **Worst-case space:** \(O(PN+N^2+G+I)\). The hybrid holds GA populations and
  histories, and one SA neighborhood at a time. SA histories from a call are
  released/replaced rather than accumulated across all \(Q\) calls.

The hybrid’s \(K N^2\) term is why a modest number of local refinements can
dominate the GA cost. Configuring frequency, fraction, and SA iteration limits
controls this overhead but does not guarantee better energy.

### 7. Search-space reduction techniques

These techniques affect work performed by the implemented heuristic; none
proves that every valid conformation has been enumerated or that the global
optimum is found.

| Technique | Current implementation cost/effect |
|---|---|
| Symmetry reduction | D4 canonicalization transforms \(N\) coordinates eight times: \(O(N)\) expected time and \(O(N)\) temporary/key space per candidate. It can avoid a repeated energy evaluation, itself \(O(N)\), but the key still has to be formed. It does not improve the worst-case asymptotic cost and can add overhead when there are no symmetry duplicates. |
| Duplicate-state detection | GA performs expected hash-table lookup after creating/hashing an \(O(N)\) genome or symmetry key. Across \(M\) attempted candidates, key work is \(O(MN)\); it can skip energy evaluation for duplicates but still decodes candidates first. A local neighborhood stores up to \(O(N)\) keys of length \(N\), included in its \(O(N^2)\) time/space bound. |
| Early collision rejection | Direction decoding checks occupancy as it builds coordinates. A collision at step \(j\) costs \(O(j)\), with \(O(1)\) best case and \(O(N)\) worst case; valid genomes still take \(\Theta(N)\). It reduces wasted work on invalid offspring, not worst-case time. |
| Incremental energy evaluation | Tracker initialization is \(O(N)\). A proposal scans \(N\) coordinate pairs to discover changes, then performs \(O(m)\) contact-neighbor checks; an accepted commit scans \(N\) again. Thus current end-to-end work is \(O(N+m)=O(N)\), not \(O(m)\), although it avoids a full contact scan. It does not reduce SA’s \(O(N^2)\) neighborhood-generation bound. |
| Restricted neighborhoods | Reduces proposal count/constants and may make a particular move set cheaper, but changes reachability and can remove useful paths. Crankshaft-enabled generation has \(\Theta(N^2)\) expected cost because it rebuilds a fixed-occupancy set for each crankshaft start. Corner-only has \(O(N^2)\) worst-case candidate construction; end-only generation is \(O(N)\). |
| Elitism | Copies \(E\) candidates and selects them in \(O(P\log E)\) time (or \(O(P)\) for one elite); it preserves a best-so-far member but does not prune the conformation space. |
| Early stopping | Caps actual work at configured \(I\) and \(G\); a run may stop earlier on target, temperature, no-improvement, or valid-neighbor conditions. It saves work only when a condition fires and may stop before a better fold is found. |

For context, the number of possible self-avoiding walks is not enumerated by
these optimizers. The coarse bound \(4\cdot3^{N-2}\) for \(N\ge2\) follows by
allowing four first steps and at most three non-immediate-reversal choices for
each later step; self-avoidance removes additional paths. This is only an upper
bound, not the exact number of conformations. GA and SA examine bounded
heuristic samples rather than exhaustively traversing this space.

## Theory versus measured runtime

Asymptotic bounds describe growth with \(N\), iteration counts, population
sizes, and retry limits under stated assumptions. They do not predict seconds
for a particular machine, Python runtime, workload, or random seed.

The separate [benchmark_performance.py](../benchmarks/benchmark_performance.py) compares
optimized paths with reconstructed reference paths in this workspace. In its
latest recorded run, examples included neighborhood generation at 0.010617 s
reference versus 0.004161 s optimized, and five-seed SA mean runtime at
0.045399 s versus 0.021842 s. These are measurements from specific workloads,
not complexity bounds or universal speedup claims. The benchmark notes that
its reference paths are reconstructed, not historical Git versions; peak
`tracemalloc` allocation is Python-tracked allocation rather than process RSS.

## Implementation references

- [structure.py](../src/hp_folding/structure.py): coordinate representation, validation, and
  bounded randomized backtracking.
- [energy.py](../src/hp_folding/energy.py): public validated and private trusted-valid energy
  evaluation.
- [local_moves.py](../src/hp_folding/local_moves.py): public local moves and materialized
  neighborhood generation.
- [simulated_annealing.py](../src/hp_folding/simulated_annealing.py): SA iteration and stopping
  logic.
- [genetic_algorithm.py](../src/hp_folding/genetic_algorithm.py): genome decoding, bounded
  population retries, selection, and scoring.
- [hybrid_optimizer.py](../src/hp_folding/hybrid_optimizer.py): periodic SA refinement and
  replacement.
- [search_space.py](../src/hp_folding/search_space.py): symmetry keys and incremental contact
  tracking.
