"""Validity-preserving local moves for 2D square-lattice protein structures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
from threading import Lock
from typing import Iterable

from .structure import (
    Coordinate,
    DIRECTION_VECTORS,
    ProteinStructure,
    StructureConstraints,
    StructureError,
    _check_deadline,
    validate_structure,
)
from .search_space import _symmetry_key_validated


MoveName = str


@dataclass
class MoveStatistics:
    """Thread-safe attempted, accepted, and rejected move counters."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicate_candidates: int = 0
    restricted_operator_opportunities: int = 0
    by_move: dict[MoveName, Counter[str]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, move: MoveName, accepted: bool) -> None:
        with self._lock:
            self.attempted += 1
            outcome = "accepted" if accepted else "rejected"
            if accepted:
                self.accepted += 1
            else:
                self.rejected += 1
            self.by_move.setdefault(
                move, Counter({"attempted": 0, "accepted": 0, "rejected": 0})
            )["attempted"] += 1
            self.by_move[move][outcome] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "attempted": self.attempted,
                "accepted": self.accepted,
                "rejected": self.rejected,
                "duplicate_candidates": self.duplicate_candidates,
                "restricted_operator_opportunities": self.restricted_operator_opportunities,
                "by_move": {
                    move: dict(counts) for move, counts in self.by_move.items()
                },
            }

    def record_duplicate(self) -> None:
        with self._lock:
            self.duplicate_candidates += 1

    def record_restricted_opportunities(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative.")
        with self._lock:
            self.restricted_operator_opportunities += count


def _require_valid_source(
    structure: ProteinStructure,
    constraints: StructureConstraints | None = None,
) -> set[Coordinate]:
    result = validate_structure(structure, constraints=constraints)
    if not result.valid:
        raise StructureError(
            "Cannot modify an invalid source structure: " + "; ".join(result.issues)
        )
    return set(structure.coordinates)


def _candidate_or_reject(
    coordinates: list[Coordinate],
    move: MoveName,
    statistics: MoveStatistics | None,
    constraints: StructureConstraints | None = None,
) -> ProteinStructure | None:
    candidate = ProteinStructure(tuple(coordinates))
    result = validate_structure(candidate, constraints=constraints)
    if not result.valid:
        if statistics is not None:
            statistics.record(move, accepted=False)
        return None
    if statistics is not None:
        statistics.record(move, accepted=True)
    return candidate


def end_move(
    structure: ProteinStructure,
    residue_index: int,
    target: Coordinate,
    statistics: MoveStatistics | None = None,
    *,
    constraints: StructureConstraints | None = None,
) -> ProteinStructure | None:
    """Move one terminal residue to a specified free neighbor of its anchor.

    `residue_index` is zero-based and must identify the first or last residue.
    The target must be exactly one lattice step from the adjacent chain residue.
    Returns None for a validly formed but inapplicable/invalid proposal.
    """
    occupied = _require_valid_source(structure, constraints)
    if structure.length < 2:
        raise ValueError("An end move requires a structure of at least two residues.")
    if type(residue_index) is not int or residue_index not in (0, structure.length - 1):
        raise ValueError("residue_index must identify a terminal residue (0-based).")
    if (
        not isinstance(target, (tuple, list))
        or len(target) != 2
        or any(type(value) is not int for value in target)
    ):
        raise ValueError("target must contain exactly two integer coordinates.")
    target_coordinate = (target[0], target[1])

    anchor_index = 1 if residue_index == 0 else structure.length - 2
    anchor = structure.coordinates[anchor_index]
    distance = abs(target_coordinate[0] - anchor[0]) + abs(
        target_coordinate[1] - anchor[1]
    )
    if (
        target_coordinate == structure.coordinates[residue_index]
        or distance != 1
        or target_coordinate in occupied
    ):
        if statistics is not None:
            statistics.record("end", accepted=False)
        return None

    coordinates = list(structure.coordinates)
    coordinates[residue_index] = target_coordinate
    return _candidate_or_reject(coordinates, "end", statistics, constraints)


def corner_move(
    structure: ProteinStructure,
    residue_index: int,
    statistics: MoveStatistics | None = None,
    *,
    constraints: StructureConstraints | None = None,
) -> ProteinStructure | None:
    """Flip a 90-degree corner residue to the other corner of its unit square.

    `residue_index` is zero-based and must be an internal residue. The target is
    `x[i-1] + x[i+1] - x[i]`; it preserves both adjacent unit bonds. A collision
    or a non-corner index rejects the proposal.
    """
    occupied = _require_valid_source(structure, constraints)
    if (
        type(residue_index) is not int
        or not 1 <= residue_index < structure.length - 1
    ):
        raise ValueError("residue_index must identify an internal residue (0-based).")

    previous = structure.coordinates[residue_index - 1]
    current = structure.coordinates[residue_index]
    following = structure.coordinates[residue_index + 1]
    target = (
        previous[0] + following[0] - current[0],
        previous[1] + following[1] - current[1],
    )
    if target == current or target in occupied:
        if statistics is not None:
            statistics.record("corner", accepted=False)
        return None

    coordinates = list(structure.coordinates)
    coordinates[residue_index] = target
    return _candidate_or_reject(coordinates, "corner", statistics, constraints)


def _crankshaft_proposals(
    structure: ProteinStructure, start_index: int
) -> list[tuple[Coordinate, Coordinate]]:
    """Enumerate alternative two-residue paths between fixed chain anchors."""
    if (
        type(start_index) is not int
        or not 1 <= start_index < structure.length - 2
    ):
        raise ValueError(
            "start_index must identify the first of two internal residues "
            "(0-based)."
        )

    first_anchor = structure.coordinates[start_index - 1]
    second_anchor = structure.coordinates[start_index + 2]
    moving_coordinates = {
        structure.coordinates[start_index],
        structure.coordinates[start_index + 1],
    }
    fixed_occupied = set(structure.coordinates) - moving_coordinates
    proposals: list[tuple[Coordinate, Coordinate]] = []

    first_neighbors = [
        (first_anchor[0] + dx, first_anchor[1] + dy)
        for dx, dy in DIRECTION_VECTORS.values()
    ]
    second_neighbors = [
        (second_anchor[0] + dx, second_anchor[1] + dy)
        for dx, dy in DIRECTION_VECTORS.values()
    ]

    for first in first_neighbors:
        if first in fixed_occupied:
            continue
        for second in second_neighbors:
            if second == first or second in fixed_occupied:
                continue
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
                continue
            proposal = (first, second)
            if proposal != (
                structure.coordinates[start_index],
                structure.coordinates[start_index + 1],
            ):
                proposals.append(proposal)
    return proposals


def crankshaft_move(
    structure: ProteinStructure,
    start_index: int,
    rng: random.Random | None = None,
    statistics: MoveStatistics | None = None,
    *,
    constraints: StructureConstraints | None = None,
) -> ProteinStructure | None:
    """Reconfigure two consecutive internal residues between fixed anchors.

    For the two selected residues, enumerate alternative ordered pairs of
    adjacent lattice sites connecting the unchanged chain anchors. Select one
    valid alternative uniformly at random. If no alternative exists, reject.
    """
    _require_valid_source(structure, constraints)
    proposals = _crankshaft_proposals(structure, start_index)
    if not proposals:
        if statistics is not None:
            statistics.record("crankshaft", accepted=False)
        return None

    generator = rng if rng is not None else random.Random()
    first, second = generator.choice(proposals)
    coordinates = list(structure.coordinates)
    coordinates[start_index] = first
    coordinates[start_index + 1] = second
    return _candidate_or_reject(coordinates, "crankshaft", statistics, constraints)


def generate_neighborhood(
    structure: ProteinStructure,
    statistics: MoveStatistics | None = None,
    move_types: Iterable[str] | None = None,
    *,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
) -> tuple[ProteinStructure, ...]:
    """Return unique valid structures reachable by one supported local move.

    Every attempted endpoint target, internal corner flip, and alternative
    crankshaft pair contributes to the optional move statistics. Duplicate
    conformations reached through different proposals appear only once.
    """
    occupied = _require_valid_source(structure, constraints)
    supported_moves = {"end", "corner", "crankshaft"}
    selected_moves = supported_moves if move_types is None else set(move_types)
    invalid_moves = selected_moves - supported_moves
    if invalid_moves:
        raise ValueError(
            "Unknown local move type(s): " + ", ".join(sorted(invalid_moves))
        )
    if not selected_moves:
        raise ValueError("move_types must include at least one supported move.")
    candidates: dict[tuple[Coordinate, ...], ProteinStructure] = {}
    source_key = (
        _symmetry_key_validated(structure)
        if constraints is None
        else structure.coordinates
    )

    def retain(coordinates: list[Coordinate], move: MoveName) -> None:
        candidate = ProteinStructure(tuple(coordinates))
        validation = validate_structure(candidate, constraints=constraints)
        if not validation.valid:
            reject(move)
            return
        if statistics is not None:
            statistics.record(move, accepted=True)
        key = (
            _symmetry_key_validated(candidate)
            if constraints is None
            else candidate.coordinates
        )
        if key == source_key or key in candidates:
            if statistics is not None:
                statistics.record_duplicate()
            return
        candidates[key] = candidate

    def reject(move: MoveName) -> None:
        if statistics is not None:
            statistics.record(move, accepted=False)

    if structure.length >= 2 and "end" in selected_moves:
        for terminal_index, anchor_index in (
            (0, 1),
            (structure.length - 1, structure.length - 2),
        ):
            anchor_x, anchor_y = structure.coordinates[anchor_index]
            for dx, dy in DIRECTION_VECTORS.values():
                _check_deadline(deadline)
                target = (anchor_x + dx, anchor_y + dy)
                if (
                    target == structure.coordinates[terminal_index]
                    or target in occupied
                ):
                    reject("end")
                    continue
                coordinates = list(structure.coordinates)
                coordinates[terminal_index] = target
                retain(coordinates, "end")
    elif structure.length >= 2 and statistics is not None:
        statistics.record_restricted_opportunities(8)

    if "corner" in selected_moves:
        for residue_index in range(1, structure.length - 1):
            _check_deadline(deadline)
            previous = structure.coordinates[residue_index - 1]
            current = structure.coordinates[residue_index]
            following = structure.coordinates[residue_index + 1]
            target = (
                previous[0] + following[0] - current[0],
                previous[1] + following[1] - current[1],
            )
            if target == current or target in occupied:
                reject("corner")
                continue
            coordinates = list(structure.coordinates)
            coordinates[residue_index] = target
            retain(coordinates, "corner")
    elif statistics is not None:
        statistics.record_restricted_opportunities(max(0, structure.length - 2))

    if "crankshaft" not in selected_moves and statistics is not None:
        statistics.record_restricted_opportunities(max(0, structure.length - 3))
    for start_index in range(1, structure.length - 2) if "crankshaft" in selected_moves else ():
        _check_deadline(deadline)
        original_pair = (
            structure.coordinates[start_index],
            structure.coordinates[start_index + 1],
        )
        first_anchor = structure.coordinates[start_index - 1]
        second_anchor = structure.coordinates[start_index + 2]
        moving = set(original_pair)
        fixed_occupied = occupied - moving
        first_neighbors = tuple(
            (first_anchor[0] + dx, first_anchor[1] + dy)
            for dx, dy in DIRECTION_VECTORS.values()
        )
        second_neighbors = tuple(
            (second_anchor[0] + dx, second_anchor[1] + dy)
            for dx, dy in DIRECTION_VECTORS.values()
        )
        for first in first_neighbors:
            for second in second_neighbors:
                _check_deadline(deadline)
                if (
                    first == second
                    or abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
                    or (first, second) == original_pair
                ):
                    continue
                if first in fixed_occupied or second in fixed_occupied:
                    reject("crankshaft")
                    continue
                coordinates = list(structure.coordinates)
                coordinates[start_index] = first
                coordinates[start_index + 1] = second
                retain(coordinates, "crankshaft")

    return tuple(candidates.values())
