# Software Architecture

## AI-Based Protein Structure Prediction and Optimization

### 1. Scope and design principles

The project is implemented in one Kaggle Notebook, organized into ordered, independently testable sections. The notebook should separate configuration, domain representation, validation, objective evaluation, optimization, and experiment reporting even though the code runs in one kernel.

The system optimizes conformations in a two-dimensional square-lattice HP model. It reports model-valid lattice structures and optimization results; it does not claim to predict experimentally verified, native protein structures.

Design principles:

- Keep one canonical structure representation shared by all algorithms.
- Make validation and energy evaluation independent of the optimizer.
- Make stochastic behavior reproducible through explicit random generators.
- Use a common optimizer result format for fair experimental comparison.
- Validate structures after every operator that can change a conformation.
- Separate algorithm parameters from experiment data and measured results.

## 2. Notebook organization and dependency direction

Recommended notebook sections, in execution order:

1. Imports and reproducibility utilities
2. Configuration and parameter definitions
3. Amino-acid processing
4. HP model and data structures
5. Structure generation and validation
6. Energy calculation
7. Local search operators
8. Simulated Annealing
9. Genetic Algorithm
10. Hybrid Genetic Algorithm + Simulated Annealing
11. Visualization
12. Performance measurement and experiment runner
13. Results comparison and reporting
14. Unit/sanity checks and experiment execution

Dependency direction should flow toward foundational components:

`Configuration → Sequence processing → HP representation → Generation / validation / energy → Local operators → Optimizers → Experiment runner → Analysis / visualization`

Visualization and reporting consume results but are not dependencies of the optimization algorithms. Optimizers must not depend on plotting or notebook display state.

## 3. Shared data contracts

These are conceptual data structures, not implementation code.

### Sequence input

- `raw_sequence`: amino-acid string.
- `hp_sequence`: immutable sequence of `H`/`P` labels.
- `mapping_id`: documented mapping policy used to produce the HP sequence.

### Conformation

- `coordinates`: ordered collection of `N` integer coordinate pairs, one per sequence position.
- Coordinate at index `i` always corresponds to residue `i`.
- The first coordinate and initial direction may be canonicalized to remove translation and rotation duplicates.
- A conformation is a candidate until it passes the validator.

### Optimization result

A common result record should contain:

- algorithm name and parameter snapshot;
- best conformation and its energy/contact count;
- validity status and validation details;
- best-so-far history and optional current-state history;
- number of energy evaluations, accepted moves, and other algorithm-specific counters;
- elapsed wall time and peak-memory measurement when available;
- random seed and run identifier.

### Experiment result

An experiment record groups optimization results by sequence, algorithm, parameter set, and repetition. Keep raw per-run results; calculate summaries from them rather than discarding trial-level data.

## 4. Module specifications

### 4.1 Configuration and parameters

- **Responsibility:** Define validated, reproducible settings for sequences, algorithms, experiments, and output.
- **Input:** User-selected sequence(s), algorithm settings, experiment budget, and seed.
- **Output:** Validated configuration records and per-run random-generator seeds.
- **Important functions:** `default_config()`, `validate_config(config)`, `build_run_config(config, algorithm, repetition)`, `make_rng(seed)`.
- **Dependencies:** Standard Python; optionally NumPy for random generators and numeric summaries.
- **Data structures:** Immutable or clearly separated configuration records for common settings, SA settings, GA settings, hybrid settings, and experiment settings.
- **Notes:** Store parameter snapshots with results. Define budgets in comparable terms, preferably energy evaluations, and additionally record wall time.

### 4.2 Amino-acid processing

- **Responsibility:** Normalize and validate sequence input, then convert the amino-acid alphabet to the model's H/P alphabet.
- **Input:** Raw amino-acid sequence and an explicit mapping policy.
- **Output:** Validated normalized sequence and corresponding HP sequence.
- **Important functions:** `normalize_sequence(raw)`, `validate_amino_acids(sequence)`, `to_hp_sequence(sequence, mapping)`, `sequence_summary(hp_sequence)`.
- **Dependencies:** Configuration for the selected mapping; no optimizer dependency.
- **Data structures:** Strings or immutable tuples of one-letter amino-acid codes and H/P labels; mapping dictionary and mapping identifier.
- **Notes:** Treat mapping policy as part of the experiment definition. Do not silently classify unknown residues.

### 4.3 HP model representation

- **Responsibility:** Define the canonical model-level representation and lattice conventions.
- **Input:** HP sequence and lattice coordinate convention.
- **Output:** Model instance containing sequence length, residue labels, and lattice rules.
- **Important functions:** `create_hp_model(hp_sequence)`, `lattice_neighbors(coordinate)`, `is_sequence_neighbor(i, j)`.
- **Dependencies:** Amino-acid processing output; standard Python.
- **Data structures:** HP model record; coordinate type as a pair of integers; conformation as an ordered coordinate collection.
- **Notes:** Use Manhattan adjacency on the square lattice. Exclude consecutive sequence positions from nonbonded contact counting.

### 4.4 Protein structure generation

- **Responsibility:** Construct initial valid conformations and candidate conformations when an optimizer requires a restart or initial population.
- **Input:** HP model, generation strategy, random generator, and optional attempt limit.
- **Output:** A valid conformation or an explicit generation failure.
- **Important functions:** `generate_straight_chain(model)`, `generate_random_valid(model, rng, attempt_limit)`, `generate_initial_population(model, size, rng)`.
- **Dependencies:** HP representation, configuration, structure validation.
- **Data structures:** Conformation; occupancy set or coordinate-to-residue index map used while constructing a self-avoiding walk.
- **Notes:** Define a clear failure result if the attempt budget is exhausted. Avoid a success-shaped fallback that is not validated.

### 4.5 Structure validation

- **Responsibility:** Check whether a conformation satisfies model constraints and return actionable diagnostics.
- **Input:** HP model and candidate conformation.
- **Output:** Validation result with validity flag and violations, such as wrong length, non-integer coordinate, broken chain bond, or coordinate overlap.
- **Important functions:** `validate_structure(model, conformation)`, `is_valid_structure(model, conformation)`.
- **Dependencies:** HP representation; no energy or optimizer dependency.
- **Data structures:** Validation-result record; coordinate set or occupancy map.
- **Notes:** Connectivity requires Manhattan distance exactly one for consecutive residues; self-avoidance requires all coordinates to be distinct. Keep structural validity separate from biological accuracy.

### 4.6 Energy calculation

- **Responsibility:** Evaluate the HP objective for valid conformations and expose contact information.
- **Input:** HP model and validated conformation.
- **Output:** Energy and optionally the list/count of nonconsecutive H–H contacts.
- **Important functions:** `count_hh_contacts(model, conformation)`, `calculate_energy(model, conformation)`, `energy_delta(model, old, proposed, affected_indices)`.
- **Dependencies:** HP representation and validation contract.
- **Data structures:** Scalar energy, integer contact count, optional contact-pair list; coordinate-to-index map for efficient neighbor lookup.
- **Notes:** Use a documented sign convention, for example energy equals negative H–H contact count when contact strength is one. Reject invalid structures explicitly. An optimized incremental calculation must be tested against full recomputation.

### 4.7 Local search operators

- **Responsibility:** Propose local conformational changes for SA and local improvement in the hybrid.
- **Input:** Current conformation, HP model, move policy, and random generator.
- **Output:** Proposed conformation plus move metadata, or an explicit indication that no legal proposal was generated.
- **Important functions:** `propose_end_move(...)`, `propose_corner_flip(...)`, `propose_crankshaft_move(...)`, `propose_local_move(...)`.
- **Dependencies:** HP representation, structure validation, random generator; optionally energy-delta evaluation.
- **Data structures:** Conformation; move descriptor containing changed indices and old/new coordinates.
- **Notes:** Every proposal must preserve chain connectivity or be checked after generation. Reject overlaps before expensive evaluation. The chosen operator set affects reachable conformations and must be documented.

### 4.8 Simulated Annealing

- **Responsibility:** Search from a single conformation, accepting uphill moves with a temperature-dependent probability to escape local minima.
- **Input:** HP model, valid initial conformation, SA configuration, random generator, energy and move interfaces.
- **Output:** Common optimization result containing the best valid conformation, energy, counters, and history.
- **Important functions:** `run_simulated_annealing(...)`, `temperature_at(step, config)`, `accept_candidate(delta_energy, temperature, rng)`.
- **Dependencies:** Configuration, validation, energy calculation, local search operators.
- **Data structures:** Current and best conformations; scalar current/best energy; temperature schedule; compact best-so-far history and counters.
- **Notes:** Preserve the best-so-far state even if the final current state is worse. Define behavior for zero temperature and failed move proposals.

### 4.9 Genetic Algorithm

- **Responsibility:** Explore multiple conformations using selection, mutation, crossover, and replacement.
- **Input:** HP model, valid initial population, GA configuration, random generator, energy and validation interfaces.
- **Output:** Common optimization result with best individual, population-level history, and counters.
- **Important functions:** `run_genetic_algorithm(...)`, `select_parent(...)`, `mutate_conformation(...)`, `crossover_conformations(...)`, `build_next_generation(...)`.
- **Dependencies:** Configuration, structure generation, validation, energy calculation, local operators where reused.
- **Data structures:** Population of conformations; fitness/energy array; individual metadata; per-generation best and diversity summaries.
- **Notes:** Crossover is especially prone to violating lattice constraints. Prefer validity-preserving operators where practical; otherwise validate and use bounded repair or discard attempts. Avoid unbounded retry loops. Specify elitism, selection, and diversity handling.

### 4.10 Hybrid optimization

- **Responsibility:** Combine population-level exploration from GA with local refinement from SA.
- **Input:** HP model, hybrid configuration, random generator, and the shared GA/SA services.
- **Output:** Common optimization result plus hybrid-specific counters.
- **Important functions:** `run_hybrid_optimizer(...)`, `refine_individual_with_sa(...)`, `allocate_local_search_budget(...)`.
- **Dependencies:** GA, SA, configuration, validation, energy calculation.
- **Data structures:** GA population; selected individuals for refinement; SA state for each refinement; shared run counters.
- **Notes:** Specify when SA is applied (for example, to selected offspring), how often, and how its evaluation budget is charged. The hybrid's total work must be measured and controlled for fair comparison.

### 4.11 Visualization

- **Responsibility:** Display valid conformations and communicate sequence labels and hydrophobic contacts.
- **Input:** HP model, validated conformation, optional contact pairs, and plot settings.
- **Output:** Matplotlib figure or notebook display.
- **Important functions:** `plot_conformation(...)`, `plot_convergence(...)`, `plot_energy_comparison(...)`.
- **Dependencies:** Matplotlib; energy/contact data when contacts are shown.
- **Data structures:** Coordinates and contact-pair list; figure/axes objects.
- **Notes:** Draw sequence bonds separately from nonbonded H–H contacts. Label or color H and P consistently. Visualization must not mutate optimizer state.

### 4.12 Experimental evaluation

- **Responsibility:** Define benchmark cases, execute repeated runs, and retain reproducible trial-level data.
- **Input:** Validated experiment configuration, sequence set, algorithms, seeds, and resource budgets.
- **Output:** Collection of per-run result records and experiment metadata.
- **Important functions:** `run_experiment(...)`, `run_repetitions(...)`, `make_benchmark_cases(...)`, `save_results(...)`.
- **Dependencies:** Configuration, sequence processing, all optimizers, performance measurement.
- **Data structures:** Experiment configuration; benchmark-case records; list/table of per-run result records.
- **Notes:** Use the same input sequences and comparable evaluation/time budgets for each algorithm. Record software/library versions, seeds, and mapping policy. Keep failed runs and failure reasons visible.

### 4.13 Performance measurement

- **Responsibility:** Measure computational cost consistently without changing the optimization objective.
- **Input:** A callable run or experiment, resource budget, and measurement settings.
- **Output:** Runtime, evaluation counts, optional peak memory, and budget-exhaustion status.
- **Important functions:** `measure_run(...)`, `start_timer()`, `read_peak_memory()`, `record_evaluation()`.
- **Dependencies:** Standard Python timing and resource utilities; optionally `tracemalloc` for Python allocation measurements.
- **Data structures:** Measurement record; counters integrated into the common optimization result.
- **Notes:** State whether memory means Python-tracked allocations or process-level memory. Use the same measurement method for all algorithms. Avoid comparing notebook display or plotting time as optimizer time.

### 4.14 Results comparison

- **Responsibility:** Aggregate repeated runs, compare algorithms, and produce tables and plots for the report.
- **Input:** Per-run experiment results and selected grouping/summary criteria.
- **Output:** Summary tables, comparison figures, and concise textual findings.
- **Important functions:** `summarize_results(...)`, `compare_algorithms(...)`, `plot_energy_comparison(...)`, `format_report_table(...)`.
- **Dependencies:** Experiment-result schema; pandas and Matplotlib are optional conveniences.
- **Data structures:** DataFrame or list of summary records containing sequence, algorithm, repetition count, best/mean/standard deviation energy, runtime, evaluations, and validity rate.
- **Notes:** Report the number of repetitions and variability. Compare both solution quality and computational cost; do not infer significance from a single stochastic run.

## 5. End-to-end interaction

1. The configuration module validates the sequence, algorithm settings, budgets, and seed policy.
2. Amino-acid processing normalizes the sequence and converts it to an HP sequence under a recorded mapping.
3. The HP model stores the HP labels and lattice conventions used by all downstream modules.
4. Structure generation creates one valid starting conformation for SA, a valid population for GA, or both for the hybrid. The validator confirms each candidate before it is admitted.
5. The energy module evaluates valid conformations and returns energy/contact information. Optimizers request evaluations through this shared module and update their counters.
6. Local operators propose candidate moves. The validator checks changed structures; the energy module scores accepted-valid candidates.
7. SA, GA, and the hybrid run through a common experiment interface and return the shared optimization-result record.
8. Performance measurement records runtime, memory where supported, and evaluation counts. The experiment runner repeats runs with controlled seeds and budgets.
9. Results comparison aggregates the retained per-run records. Visualization renders selected valid structures and summary plots after optimization.

## 6. Cross-cutting correctness and testing checks

- A straight chain should validate and have no nonconsecutive contacts.
- A deliberately overlapping conformation and a broken-bond conformation should fail validation with distinct diagnostics.
- Hand-constructed small conformations should produce the expected H–H contact count and energy.
- Incremental energy deltas should match full energy recomputation on tested moves.
- Every generated, mutated, crossed, or locally modified conformation admitted to an optimizer must validate.
- With a fixed seed and environment, runs should be reproducible to the extent supported by the selected libraries.
- The best-so-far energy should never worsen over recorded best-so-far history.
- All algorithms should return the same result schema, including explicit failure or budget-exhaustion information.
- Experiment summaries should be computed from per-run results and should include trial counts and variability.
