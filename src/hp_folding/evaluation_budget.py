"""Shared objective-evaluation budget for optimization runs."""

from dataclasses import dataclass


@dataclass
class EvaluationBudget:
    """Count objective evaluations against a fixed, shareable budget.

    The budget is deliberately serial and not thread-safe. It is shared by
    the hybrid GA and its nested SA calls so their combined scoring cannot
    exceed the same limit as standalone methods.
    """

    limit: int
    evaluations: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("limit must be a positive integer.")
        if type(self.evaluations) is not int or not 0 <= self.evaluations <= self.limit:
            raise ValueError("evaluations must be an integer within the budget.")

    @property
    def exhausted(self) -> bool:
        return self.evaluations >= self.limit

    def try_consume(self) -> bool:
        """Reserve one evaluation, returning False when the budget is spent."""
        if self.exhausted:
            return False
        self.evaluations += 1
        return True
