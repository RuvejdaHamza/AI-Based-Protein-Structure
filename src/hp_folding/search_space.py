"""Search-space reduction helpers for the 2D HP lattice model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .energy import EnergyResult, calculate_energy
from .structure import Coordinate, ProteinStructure, validate_structure


def symmetry_key(structure: ProteinStructure) -> tuple[Coordinate, ...]:
    """Return a translation/rotation/reflection invariant key for a chain.

    All eight symmetries of the square (the dihedral group D4) are considered.
    Residue order is preserved, so chain reversal is not treated as equivalent.
    The key is useful for duplicate detection; it does not remove any physical
    conformation or prove that heuristic search explores the remaining space.
    """
    validation = validate_structure(structure)
    if not validation.valid:
        raise ValueError(
            "Cannot canonicalize an invalid structure: " + "; ".join(validation.issues)
        )
    return _symmetry_key_validated(structure)


def _symmetry_key_validated(
    structure: ProteinStructure,
) -> tuple[Coordinate, ...]:
    """Canonical key for a structure whose validity has already been checked."""
    if not structure.coordinates:
        return ()
    transforms = (
        (1, 0, 0, 1),
        (0, -1, 1, 0),
        (-1, 0, 0, -1),
        (0, 1, -1, 0),
        (-1, 0, 0, 1),
        (0, -1, -1, 0),
        (1, 0, 0, -1),
        (0, 1, 1, 0),
    )
    best: tuple[Coordinate, ...] | None = None
    for ax, bx, ay, by in transforms:
        points: list[Coordinate] = []
        min_x = min_y = 0
        for index, (x, y) in enumerate(structure.coordinates):
            tx, ty = ax * x + bx * y, ay * x + by * y
            if index == 0:
                min_x, min_y = tx, ty
            else:
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
            points.append((tx, ty))
        key = tuple((x - min_x, y - min_y) for x, y in points)
        if best is None or key < best:
            best = key
    return best if best is not None else ()


def direction_key(structure: ProteinStructure) -> tuple[Coordinate, ...]:
    """Compatibility alias for the canonical structure key."""
    return symmetry_key(structure)


@dataclass(frozen=True)
class EnergyDelta:
    energy: float
    contact_count: int
    delta_energy: float
    changed_residues: int
    neighbor_checks: int
    _affected_contacts: frozenset[tuple[int, int]]
    _removed_contacts: frozenset[tuple[int, int]]
    _source_coordinates: tuple[Coordinate, ...]
    _candidate_coordinates: tuple[Coordinate, ...]


class IncrementalEnergyTracker:
    """Track HP energy across accepted moves using changed-residue contacts.

    The tracker builds an occupancy map once. Each proposal checks only lattice
    neighbors of changed residues, then `commit` updates the map after an
    optimizer accepts the proposal. Structure validation and changed-index
    detection still cost O(N) with the current structure API; contact counting
    itself is O(k) for k moved residues instead of another full contact scan.
    """

    def __init__(
        self, hp_sequence: str, structure: ProteinStructure, epsilon: float = 1.0
    ) -> None:
        self.hp_sequence = "".join(hp_sequence.split()).upper()
        if not self.hp_sequence or set(self.hp_sequence) - {"H", "P"}:
            raise ValueError("hp_sequence must be non-empty and contain only H and P.")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("epsilon must be a positive finite number.")
        self.epsilon = float(epsilon)
        validation = validate_structure(
            structure, expected_length=len(self.hp_sequence)
        )
        if not validation.valid:
            raise ValueError(
                "Cannot track an invalid structure: " + "; ".join(validation.issues)
            )
        self.structure = structure
        self._occupancy = {
            coordinate: index
            for index, coordinate in enumerate(structure.coordinates)
        }
        self._contacts = self._find_all_contacts()
        self._energy = -self.epsilon * len(self._contacts)
        self.full_evaluations = 1
        self.incremental_evaluations = 0
        self.neighbor_checks = 0

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def contact_count(self) -> int:
        return len(self._contacts)

    def _find_all_contacts(self) -> set[tuple[int, int]]:
        contacts: set[tuple[int, int]] = set()
        for index, (x, y) in enumerate(self.structure.coordinates):
            if self.hp_sequence[index] != "H":
                continue
            for coordinate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                neighbor_index = self._occupancy.get(coordinate)
                if (
                    neighbor_index is not None
                    and neighbor_index > index + 1
                    and self.hp_sequence[neighbor_index] == "H"
                ):
                    contacts.add((index, neighbor_index))
        return contacts

    def evaluate(self, candidate: ProteinStructure) -> EnergyDelta:
        return self._evaluate(candidate, validate=True)

    def evaluate_validated(self, candidate: ProteinStructure) -> EnergyDelta:
        """Evaluate a candidate guaranteed valid by a trusted local operator."""
        return self._evaluate(candidate, validate=False)

    def _evaluate(
        self, candidate: ProteinStructure, *, validate: bool
    ) -> EnergyDelta:
        if validate:
            validation = validate_structure(
                candidate, expected_length=len(self.hp_sequence)
            )
            if not validation.valid:
                raise ValueError(
                    "Cannot evaluate an invalid candidate: "
                    + "; ".join(validation.issues)
                )
        elif len(candidate.coordinates) != len(self.hp_sequence):
            raise ValueError("Candidate length does not match the tracked sequence.")
        changed = {
            index
            for index, (old, new) in enumerate(
                zip(self.structure.coordinates, candidate.coordinates)
            )
            if old != new
        }
        if not changed:
            self.incremental_evaluations += 1
            return EnergyDelta(
                energy=self._energy,
                contact_count=len(self._contacts),
                delta_energy=0.0,
                changed_residues=0,
                neighbor_checks=0,
                _affected_contacts=frozenset(),
                _removed_contacts=frozenset(),
                _source_coordinates=self.structure.coordinates,
                _candidate_coordinates=candidate.coordinates,
            )

        old_affected: set[tuple[int, int]] = set()
        for index in changed:
            if self.hp_sequence[index] != "H":
                continue
            x, y = self.structure.coordinates[index]
            for coordinate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                neighbor_index = self._occupancy.get(coordinate)
                if (
                    neighbor_index is not None
                    and neighbor_index != index
                    and self.hp_sequence[neighbor_index] == "H"
                    and abs(neighbor_index - index) > 1
                ):
                    old_affected.add(tuple(sorted((index, neighbor_index))))
        new_affected: set[tuple[int, int]] = set()
        proposed_index = {
            candidate.coordinates[index]: index for index in changed
        }
        checks = 0

        for index in changed:
            if self.hp_sequence[index] != "H":
                continue
            x, y = candidate.coordinates[index]
            for coordinate in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                checks += 1
                neighbor_index = proposed_index.get(coordinate)
                if neighbor_index is None:
                    neighbor_index = self._occupancy.get(coordinate)
                    if neighbor_index in changed:
                        neighbor_index = None
                if (
                    neighbor_index is None
                    or neighbor_index == index
                    or self.hp_sequence[neighbor_index] != "H"
                    or abs(neighbor_index - index) <= 1
                ):
                    continue
                new_affected.add(tuple(sorted((index, neighbor_index))))

        contact_count = len(self._contacts) - len(old_affected) + len(new_affected)
        new_energy = -self.epsilon * contact_count
        self.incremental_evaluations += 1
        self.neighbor_checks += checks
        return EnergyDelta(
            energy=new_energy,
            contact_count=contact_count,
            delta_energy=new_energy - self._energy,
            changed_residues=len(changed),
            neighbor_checks=checks,
            _affected_contacts=frozenset(new_affected),
            _removed_contacts=frozenset(old_affected),
            _source_coordinates=self.structure.coordinates,
            _candidate_coordinates=candidate.coordinates,
        )

    def commit(self, candidate: ProteinStructure, delta: EnergyDelta) -> None:
        """Commit a previously evaluated candidate after optimizer acceptance."""
        if delta._source_coordinates != self.structure.coordinates:
            raise ValueError("EnergyDelta was calculated from a different tracker state.")
        if delta._candidate_coordinates != candidate.coordinates:
            raise ValueError("EnergyDelta was calculated for a different candidate.")
        changed = {
            index
            for index, (old, new) in enumerate(
                zip(self.structure.coordinates, candidate.coordinates)
            )
            if old != new
        }
        if len(changed) != delta.changed_residues:
            raise ValueError("EnergyDelta does not correspond to this candidate.")
        for index in changed:
            self._occupancy.pop(self.structure.coordinates[index])
        for index in changed:
            self._occupancy[candidate.coordinates[index]] = index

        self._contacts.difference_update(delta._removed_contacts)
        self._contacts.update(delta._affected_contacts)
        self.structure = candidate
        self._energy = delta.energy

    def full_energy(self) -> EnergyResult:
        """Recompute energy from scratch for verification or periodic refresh."""
        self.full_evaluations += 1
        result = calculate_energy(self.hp_sequence, self.structure, self.epsilon)
        return result
