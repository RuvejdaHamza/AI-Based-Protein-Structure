"""Sequence validation and HP conversion for the 2D lattice model.

This project uses the following documented binary classification:

    Hydrophobic (H): A, C, G, I, L, M, F, P, W, Y, V
    Polar (P):       R, N, D, Q, E, K, S, T, H

This is a simplified modeling convention, not a universal biochemical
classification. Input is case-insensitive; whitespace is ignored. FASTA
headers, separators, and non-standard amino-acid symbols are not accepted.
"""

from dataclasses import dataclass


HYDROPHOBIC_AMINO_ACIDS = frozenset("ACGILMFPWYV")
POLAR_AMINO_ACIDS = frozenset("RNDQEKSTH")
STANDARD_AMINO_ACIDS = HYDROPHOBIC_AMINO_ACIDS | POLAR_AMINO_ACIDS

_AMINO_ACID_TO_HP = {
    **{amino_acid: "H" for amino_acid in HYDROPHOBIC_AMINO_ACIDS},
    **{amino_acid: "P" for amino_acid in POLAR_AMINO_ACIDS},
}


class SequenceValidationError(ValueError):
    """Raised when an input is not a valid standard amino-acid sequence."""


@dataclass(frozen=True)
class ProcessedSequence:
    """Validated sequence data used by the HP folding model."""

    amino_acid_sequence: str
    hp_sequence: str
    length: int


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and uppercase a sequence, rejecting non-string input."""
    if not isinstance(sequence, str):
        raise SequenceValidationError(
            f"Sequence must be a string; received {type(sequence).__name__}."
        )
    return "".join(sequence.split()).upper()


def validate_amino_acid_sequence(sequence: str) -> str:
    """Return a normalized sequence or raise an informative validation error.

    Positions in the error message are 1-based positions in the normalized
    sequence. Only the 20 standard one-letter amino-acid codes are accepted.
    """
    normalized = normalize_sequence(sequence)
    if not normalized:
        raise SequenceValidationError(
            "Sequence must not be empty or contain only whitespace."
        )

    invalid_residues = [
        (position, residue)
        for position, residue in enumerate(normalized, start=1)
        if residue not in STANDARD_AMINO_ACIDS
    ]
    if invalid_residues:
        details = ", ".join(
            f"{residue!r} at position {position}"
            for position, residue in invalid_residues
        )
        raise SequenceValidationError(
            f"Invalid amino-acid symbol(s): {details}. "
            "Use only the 20 standard one-letter codes."
        )
    return normalized


def convert_to_hp(sequence: str) -> str:
    """Validate and convert an amino-acid sequence to its H/P representation."""
    normalized = validate_amino_acid_sequence(sequence)
    return "".join(_AMINO_ACID_TO_HP[residue] for residue in normalized)


def process_sequence(sequence: str) -> ProcessedSequence:
    """Validate, normalize, convert, and measure a protein sequence."""
    normalized = validate_amino_acid_sequence(sequence)
    hp_sequence = "".join(_AMINO_ACID_TO_HP[residue] for residue in normalized)
    return ProcessedSequence(
        amino_acid_sequence=normalized,
        hp_sequence=hp_sequence,
        length=len(normalized),
    )
