"""Energy evaluation for the 2D square-lattice HP protein model."""

from dataclasses import dataclass
import math

from .structure import ProteinStructure, validate_structure


@dataclass(frozen=True)
class EnergyResult:
    """Energy score and number of non-consecutive hydrophobic contacts."""

    energy: float
    contact_count: int


def calculate_energy(
    hp_sequence: str,
    structure: ProteinStructure,
    epsilon: float = 1.0,
) -> EnergyResult:
    """Calculate E = -epsilon * C_HH for a valid conformation.

    Each residue checks only its four lattice neighbors. The coordinate map
    makes contact counting O(N) for a structure with N residues, and requiring
    the neighbor index to be greater than the current index counts each pair
    at most once. Consecutive sequence neighbors are excluded.

    Args:
        hp_sequence: Non-empty sequence containing only ``H`` and ``P``.
        structure: Lattice coordinates indexed in the same order as the sequence.
        epsilon: Positive hydrophobic interaction strength.

    Raises:
        TypeError: If argument types are not supported.
        ValueError: If the HP sequence, epsilon, or structure is invalid.
    """
    if not isinstance(hp_sequence, str):
        raise TypeError("hp_sequence must be a string containing only H and P.")
    hp_sequence = "".join(hp_sequence.split()).upper()
    if not hp_sequence:
        raise ValueError("hp_sequence must not be empty.")
    invalid_labels = sorted(set(hp_sequence) - {"H", "P"})
    if invalid_labels:
        raise ValueError(
            "hp_sequence contains invalid label(s): "
            f"{', '.join(repr(label) for label in invalid_labels)}. "
            "Use only H and P."
        )

    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite number.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number.")

    validation = validate_structure(structure, expected_length=len(hp_sequence))
    if not validation.valid:
        raise ValueError(
            "Cannot calculate energy for an invalid structure: "
            + "; ".join(validation.issues)
        )

    return _calculate_energy_validated(hp_sequence, structure, float(epsilon))


def _calculate_energy_validated(
    hp_sequence: str,
    structure: ProteinStructure,
    epsilon: float,
) -> EnergyResult:
    """Evaluate an already validated structure without repeating validation."""
    coordinate_to_index = {
        coordinate: index
        for index, coordinate in enumerate(structure.coordinates)
    }

    contact_count = 0
    for index, (x, y) in enumerate(structure.coordinates):
        if hp_sequence[index] != "H":
            continue
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            neighbor_index = coordinate_to_index.get(neighbor)
            if (
                neighbor_index is not None
                and neighbor_index > index + 1
                and hp_sequence[neighbor_index] == "H"
            ):
                contact_count += 1

    return EnergyResult(
        energy=-epsilon * contact_count,
        contact_count=contact_count,
    )
