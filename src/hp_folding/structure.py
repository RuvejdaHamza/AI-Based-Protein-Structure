"""2D square-lattice protein structure representation and validation.

Directions are relative steps between consecutive residues:
R=(1, 0), L=(-1, 0), U=(0, 1), D=(0, -1).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import random
from threading import Lock
from time import perf_counter
from typing import Iterable, Mapping


Coordinate = tuple[int, int]


class Direction(str, Enum):
    RIGHT = "R"
    LEFT = "L"
    UP = "U"
    DOWN = "D"


DIRECTION_VECTORS: Mapping[Direction, Coordinate] = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, 1),
    Direction.DOWN: (0, -1),
}


class StructureError(ValueError):
    """Raised when a structure cannot be generated from its input."""


@dataclass(frozen=True)
class StructureConstraints:
    """Optional hard constraints for an ordered lattice chain.

    Residue indices and direction-step indices are zero-based. Fixed contacts
    require the selected, non-consecutive residues to be lattice neighbors.
    """

    fixed_contacts: tuple[tuple[int, int], ...] = ()
    forbidden_positions: frozenset[Coordinate] = frozenset()
    fixed_directions: Mapping[int, Direction | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_contacts: list[tuple[int, int]] = []
        for pair in self.fixed_contacts:
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(type(index) is not int or index < 0 for index in pair)
                or pair[0] == pair[1]
                or abs(pair[0] - pair[1]) <= 1
            ):
                raise ValueError(
                    "fixed_contacts must contain distinct, non-consecutive "
                    "zero-based residue-index pairs."
                )
            normalized_contacts.append(tuple(sorted(pair)))
        if len(set(normalized_contacts)) != len(normalized_contacts):
            raise ValueError("fixed_contacts cannot contain duplicate pairs.")

        normalized_positions: set[Coordinate] = set()
        for position in self.forbidden_positions:
            _validate_coordinate(position, "forbidden position")
            normalized_positions.add((position[0], position[1]))

        if not isinstance(self.fixed_directions, Mapping):
            raise TypeError("fixed_directions must map zero-based steps to directions.")
        normalized_directions: dict[int, Direction] = {}
        for index, direction in self.fixed_directions.items():
            if type(index) is not int or index < 0:
                raise ValueError("fixed direction indices must be non-negative integers.")
            try:
                normalized_directions[index] = Direction(direction)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid fixed direction at step {index}: {direction!r}."
                ) from error

        object.__setattr__(self, "fixed_contacts", tuple(normalized_contacts))
        object.__setattr__(self, "forbidden_positions", frozenset(normalized_positions))
        object.__setattr__(self, "fixed_directions", normalized_directions)

    def validate_for_length(self, length: int) -> None:
        """Reject constraints whose residue or step indices exceed the chain."""
        if type(length) is not int or length < 1:
            raise ValueError("Constraint validation requires a positive chain length.")
        if any(index >= length for pair in self.fixed_contacts for index in pair):
            raise ValueError("fixed_contacts contains a residue index outside the chain.")
        if any(index >= length - 1 for index in self.fixed_directions):
            raise ValueError("fixed_directions contains a step index outside the chain.")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("Optimization time limit reached.")


@dataclass(frozen=True)
class ProteinStructure:
    """Coordinates for an ordered protein chain; index i maps to residue i."""

    coordinates: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        if isinstance(self.coordinates, tuple) and all(
            isinstance(point, tuple) and len(point) == 2 for point in self.coordinates
        ):
            normalized = self.coordinates
        else:
            normalized = tuple(tuple(point) for point in self.coordinates)
        object.__setattr__(self, "coordinates", normalized)

    @classmethod
    def from_directions(
        cls,
        directions: Iterable[Direction | str],
        start: Coordinate = (0, 0),
    ) -> ProteinStructure:
        """Build coordinates cumulatively from N-1 relative directions."""
        _validate_coordinate(start, "start")
        coordinates = [start]
        for index, direction in enumerate(directions, start=1):
            try:
                parsed_direction = Direction(direction)
            except (TypeError, ValueError) as error:
                raise StructureError(
                    f"Invalid direction at step {index}: {direction!r}. "
                    "Use R, L, U, or D."
                ) from error
            dx, dy = DIRECTION_VECTORS[parsed_direction]
            x, y = coordinates[-1]
            coordinates.append((x + dx, y + dy))
        return cls(tuple(coordinates))

    @property
    def length(self) -> int:
        return len(self.coordinates)

    @property
    def directions(self) -> tuple[Direction, ...]:
        """Recover relative directions, raising if a bond is not a unit step."""
        result: list[Direction] = []
        vector_to_direction = {
            vector: direction for direction, vector in DIRECTION_VECTORS.items()
        }
        for index, (first, second) in enumerate(
            zip(self.coordinates, self.coordinates[1:]), start=1
        ):
            delta = (second[0] - first[0], second[1] - first[1])
            if delta not in vector_to_direction:
                raise StructureError(
                    f"Residues {index} and {index + 1} are not exactly "
                    "one lattice unit apart."
                )
            result.append(vector_to_direction[delta])
        return tuple(result)

    def coordinate_to_residues(self) -> dict[Coordinate, tuple[int, ...]]:
        """Map each occupied lattice coordinate to its 1-based residue indices."""
        indices: dict[Coordinate, list[int]] = {}
        for index, coordinate in enumerate(self.coordinates, start=1):
            indices.setdefault(coordinate, []).append(index)
        return {coordinate: tuple(items) for coordinate, items in indices.items()}

    def format_text(self) -> str:
        """Return a readable residue-index-to-coordinate representation."""
        if not self.coordinates:
            return "ProteinStructure(length=0, residues=[])"
        residues = ", ".join(
            f"{index}:({x},{y})"
            for index, (x, y) in enumerate(self.coordinates, start=1)
        )
        return f"ProteinStructure(length={self.length}, residues=[{residues}])"

    def __str__(self) -> str:
        return self.format_text()


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[str, ...]
    collisions: tuple[tuple[Coordinate, tuple[int, ...]], ...]


@dataclass
class ValidationStatistics:
    """Thread-safe counts of validation outcomes and issue categories."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    issue_counts: Counter[str] = field(default_factory=Counter)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, result: ValidationResult) -> None:
        with self._lock:
            self.total += 1
            if result.valid:
                self.valid += 1
            else:
                self.invalid += 1
            for issue in result.issues:
                category = issue.split(":", maxsplit=1)[0]
                self.issue_counts[category] += 1

    def snapshot(self) -> dict[str, int | dict[str, int]]:
        with self._lock:
            return {
                "total": self.total,
                "valid": self.valid,
                "invalid": self.invalid,
                "issue_counts": dict(self.issue_counts),
            }


def _validate_coordinate(coordinate: object, label: str) -> None:
    if (
        not isinstance(coordinate, (tuple, list))
        or len(coordinate) != 2
        or any(type(value) is not int for value in coordinate)
    ):
        raise StructureError(
            f"{label.capitalize()} coordinate must contain exactly two integers."
        )


def validate_structure(
    structure: ProteinStructure,
    expected_length: int | None = None,
    statistics: ValidationStatistics | None = None,
    *,
    constraints: StructureConstraints | None = None,
) -> ValidationResult:
    """Check coordinate shape, expected length, connectivity, and self-avoidance."""
    issues: list[str] = []
    coordinates = structure.coordinates

    if expected_length is not None:
        if type(expected_length) is not int or expected_length < 0:
            raise ValueError("expected_length must be a non-negative integer.")
        if len(coordinates) != expected_length:
            issues.append(
                f"length: expected {expected_length} residues, got {len(coordinates)}"
            )
    if constraints is not None:
        if not isinstance(constraints, StructureConstraints):
            raise TypeError("constraints must be a StructureConstraints instance or None.")
        constraint_length = expected_length if expected_length is not None else len(coordinates)
        if constraint_length > 0:
            constraints.validate_for_length(constraint_length)

    first_indices: dict[Coordinate, int] = {}
    collision_indices: dict[Coordinate, list[int]] = {}
    for index, coordinate in enumerate(coordinates, start=1):
        try:
            _validate_coordinate(coordinate, f"residue {index}")
        except StructureError as error:
            issues.append(f"coordinate: {error}")
            continue
        previous_index = first_indices.get(coordinate)
        if previous_index is None:
            first_indices[coordinate] = index
        else:
            collision_indices.setdefault(coordinate, [previous_index]).append(index)
        if index > 1:
            previous = coordinates[index - 2]
            if (
                isinstance(previous, tuple)
                and len(previous) == 2
                and all(type(value) is int for value in previous)
            ):
                distance = abs(previous[0] - coordinate[0]) + abs(
                    previous[1] - coordinate[1]
                )
                if distance != 1:
                    issues.append(
                        f"connectivity: residues {index - 1} and {index} have "
                        f"Manhattan distance {distance}, expected 1"
                    )

    collisions = [
        (coordinate, tuple(indices))
        for coordinate, indices in collision_indices.items()
    ]
    for coordinate, indices in collisions:
        issues.append(
            f"collision: coordinate {coordinate} is occupied by residues "
            f"{', '.join(map(str, indices))}"
        )

    if constraints is not None:
        for index, coordinate in enumerate(coordinates):
            if (
                isinstance(coordinate, tuple)
                and len(coordinate) == 2
                and all(type(value) is int for value in coordinate)
                and coordinate in constraints.forbidden_positions
            ):
                issues.append(
                    f"constraint: residue {index} occupies forbidden position {coordinate}"
                )
        for step_index, expected_direction in constraints.fixed_directions.items():
            if step_index + 1 >= len(coordinates):
                continue
            first = coordinates[step_index]
            second = coordinates[step_index + 1]
            if (
                isinstance(first, tuple)
                and isinstance(second, tuple)
                and len(first) == len(second) == 2
                and all(type(value) is int for value in (*first, *second))
            ):
                vector = (second[0] - first[0], second[1] - first[1])
                if vector != DIRECTION_VECTORS[expected_direction]:
                    issues.append(
                        f"constraint: step {step_index} must be "
                        f"{expected_direction.value}"
                    )
        for first_index, second_index in constraints.fixed_contacts:
            if second_index >= len(coordinates):
                continue
            first = coordinates[first_index]
            second = coordinates[second_index]
            if (
                isinstance(first, tuple)
                and isinstance(second, tuple)
                and len(first) == len(second) == 2
                and all(type(value) is int for value in (*first, *second))
                and abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
            ):
                issues.append(
                    f"constraint: residues {first_index} and {second_index} "
                    "must form a lattice contact"
                )

    result = ValidationResult(
        valid=not issues,
        issues=tuple(issues),
        collisions=tuple(collisions),
    )
    if statistics is not None:
        statistics.record(result)
    return result


def generate_random_valid_structure(
    length: int,
    rng: random.Random | None = None,
    max_restarts: int = 100,
    max_steps_per_restart: int = 10_000,
    statistics: ValidationStatistics | None = None,
    *,
    constraints: StructureConstraints | None = None,
    deadline: float | None = None,
) -> ProteinStructure:
    """Generate a self-avoiding walk using randomized depth-first backtracking.

    Raises StructureError with the search limits if it cannot construct a
    structure. The search is bounded and does not return an invalid fallback.
    """
    if type(length) is not int or length < 1:
        raise ValueError("length must be a positive integer.")
    if type(max_restarts) is not int or max_restarts < 1:
        raise ValueError("max_restarts must be a positive integer.")
    if type(max_steps_per_restart) is not int or max_steps_per_restart < 1:
        raise ValueError("max_steps_per_restart must be a positive integer.")
    if constraints is not None:
        if not isinstance(constraints, StructureConstraints):
            raise TypeError("constraints must be a StructureConstraints instance or None.")
        constraints.validate_for_length(length)
        if (0, 0) in constraints.forbidden_positions:
            raise StructureError(
                "The fixed chain origin (0, 0) is a forbidden position."
            )

    generator = rng if rng is not None else random.Random()
    if length == 1:
        structure = ProteinStructure(((0, 0),))
        result = validate_structure(
            structure,
            expected_length=length,
            statistics=statistics,
            constraints=constraints,
        )
        if not result.valid:
            raise StructureError(
                "Could not generate a structure satisfying constraints: "
                + "; ".join(result.issues)
            )
        return structure

    directions = tuple(DIRECTION_VECTORS)
    for _ in range(max_restarts):
        coordinates: list[Coordinate] = [(0, 0)]
        occupied: set[Coordinate] = {(0, 0)}
        choices: list[list[Direction]] = []
        attempts = 0

        while len(coordinates) < length and attempts < max_steps_per_restart:
            _check_deadline(deadline)
            if len(choices) < len(coordinates):
                candidates = list(directions)
                fixed = (
                    constraints.fixed_directions.get(len(coordinates) - 1)
                    if constraints is not None
                    else None
                )
                if fixed is not None:
                    candidates = [fixed]
                generator.shuffle(candidates)
                choices.append(candidates)

            if choices[-1]:
                direction = choices[-1].pop()
                attempts += 1
                dx, dy = DIRECTION_VECTORS[direction]
                x, y = coordinates[-1]
                candidate = (x + dx, y + dy)
                if (
                    candidate in occupied
                    or (
                        constraints is not None
                        and candidate in constraints.forbidden_positions
                    )
                ):
                    continue
                coordinates.append(candidate)
                occupied.add(candidate)
            else:
                choices.pop()
                if len(coordinates) == 1:
                    break
                occupied.remove(coordinates.pop())

        if len(coordinates) == length:
            structure = ProteinStructure(tuple(coordinates))
            result = validate_structure(
                structure,
                expected_length=length,
                statistics=statistics,
                constraints=constraints,
            )
            if result.valid:
                return structure
            if constraints is None:
                raise RuntimeError(
                    "Internal error: random walk generation produced an invalid "
                    "structure: " + "; ".join(result.issues)
                )
            occupied.remove(coordinates.pop())

    raise StructureError(
        f"Could not generate a valid structure of length {length} after "
        f"{max_restarts} restarts and at most {max_steps_per_restart} "
        "candidate steps per restart."
    )
