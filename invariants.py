"""
RTL Mechanical Invariants for Zeckendorf Representations

This module formalizes the right-to-left (RTL) processing invariants that govern
Zeckendorf decomposition algorithms. These invariants ensure correctness, uniqueness,
and optimality of representations.

Mathematical Foundation:
-----------------------
The Zeckendorf theorem states that every positive integer has a unique representation
as a sum of non-consecutive Fibonacci numbers. This extends to generalized sequences.

RTL processing is fundamental because:
1. It ensures greedy optimality (largest terms first)
2. It maintains the non-adjacency constraint naturally
3. It produces the unique canonical representation

Invariant Classes:
-----------------
- SequenceInvariant: Properties that sequences must satisfy
- DecompositionInvariant: Properties that decompositions must maintain
- RTLProcessingInvariant: Guarantees about RTL traversal operations
"""

from functools import wraps
from itertools import takewhile
import sys


class InvariantViolation(Exception):
    """Raised when a mechanical invariant is violated."""
    pass


class SequenceInvariant:
    """
    Invariants that generalized Fibonacci-like sequences must satisfy.

    A valid sequence S for Zeckendorf representation must satisfy:
    1. Strict monotonicity: S[i] < S[i+1] for all i
    2. Completeness: Every positive integer has a representation
    3. Growth bound: S[i+1] <= 2*S[i] for standard Fibonacci (varies for generalized)
    """

    @staticmethod
    def is_strictly_monotonic(sequence):
        """
        Verify strict monotonicity: each term is strictly greater than the previous.

        This is essential for RTL processing - it guarantees that when we process
        from right (largest) to left (smallest), we never revisit larger values.

        Args:
            sequence: Iterable of numbers

        Returns:
            bool: True if strictly monotonic
        """
        seq_list = list(sequence)
        if len(seq_list) < 2:
            return True
        return all(seq_list[i] < seq_list[i+1] for i in range(len(seq_list)-1))

    @staticmethod
    def verify_monotonicity(sequence):
        """Verify and raise if not strictly monotonic."""
        if not SequenceInvariant.is_strictly_monotonic(sequence):
            raise InvariantViolation(
                "Sequence violates strict monotonicity invariant: "
                "each term must be strictly greater than the previous"
            )

    @staticmethod
    def is_complete_for_range(sequence, max_n):
        """
        Verify that the sequence can represent all integers in [1, max_n].

        Completeness ensures no "gaps" in representable values.

        Args:
            sequence: List of sequence values
            max_n: Maximum value to check

        Returns:
            bool: True if all values in range are representable
        """
        seq_list = [s for s in sequence if s <= max_n]

        # Check if 1 is representable (base case)
        if 1 not in seq_list and (not seq_list or seq_list[0] > 1):
            return False

        # For a complete sequence, each new term should be at most
        # 1 + sum of all previous terms
        cumsum = 0
        for val in seq_list:
            if val > cumsum + 1 and val != 1:
                return False
            cumsum += val

        return cumsum >= max_n

    @staticmethod
    def satisfies_growth_bound(sequence, k=1):
        """
        Check if sequence satisfies the generalized growth bound.

        For k-generalized Fibonacci: a_{n+1} = a_{n-1} + k*a_n
        This implies a_{n+1} <= (k+1)*a_n for sufficiently large n.

        Args:
            sequence: List of sequence values
            k: Generalization parameter

        Returns:
            bool: True if growth bound is satisfied
        """
        seq_list = list(sequence)
        if len(seq_list) < 3:
            return True

        bound = k + 1
        # Allow some tolerance for initial terms
        for i in range(2, len(seq_list)):
            if seq_list[i] > bound * seq_list[i-1] + 1:
                return False
        return True


class DecompositionInvariant:
    """
    Invariants that Zeckendorf decompositions must satisfy.

    A valid decomposition D of n must satisfy:
    1. Correctness: sum(D) == n
    2. Validity: All terms in D are from the sequence
    3. Non-adjacency: No two consecutive sequence terms appear (for standard Zeckendorf)
    4. Multiplier bounds: For generalized, multipliers in [1, k]
    5. Canonicity: The representation is unique (follows from greedy RTL)
    """

    @staticmethod
    def verify_sum(decomposition, target):
        """
        Verify that decomposition sums to target value.

        This is the fundamental correctness invariant.

        Args:
            decomposition: List of values or (value, multiplier) tuples
            target: Expected sum

        Returns:
            bool: True if sum matches target
        """
        total = 0
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                total += item[0] * item[1]  # value * multiplier
            else:
                total += item
        return total == target

    @staticmethod
    def verify_sum_strict(decomposition, target):
        """Verify sum and raise if incorrect."""
        if not DecompositionInvariant.verify_sum(decomposition, target):
            actual = sum(
                item[0] * item[1] if isinstance(item, (list, tuple)) else item
                for item in decomposition
            )
            raise InvariantViolation(
                f"Decomposition sum invariant violated: "
                f"expected {target}, got {actual}"
            )

    @staticmethod
    def is_non_adjacent(indexes, sequence_length=None):
        """
        Verify non-adjacency constraint for standard Zeckendorf.

        In a valid Zeckendorf representation, no two consecutive
        Fibonacci numbers appear. This is what makes the representation unique.

        The RTL greedy algorithm naturally maintains this because taking
        F_n prevents taking F_{n-1} (since F_n + F_{n-1} = F_{n+1}).

        Args:
            indexes: List of sequence indexes used in decomposition
            sequence_length: Optional bound for validation

        Returns:
            bool: True if no adjacent indexes
        """
        sorted_idx = sorted(indexes)
        for i in range(len(sorted_idx) - 1):
            if sorted_idx[i+1] - sorted_idx[i] == 1:
                return False
        return True

    @staticmethod
    def verify_non_adjacency(indexes):
        """Verify non-adjacency and raise if violated."""
        if not DecompositionInvariant.is_non_adjacent(indexes):
            raise InvariantViolation(
                f"Non-adjacency invariant violated: indexes {sorted(indexes)} "
                f"contain consecutive values"
            )

    @staticmethod
    def verify_multiplier_bounds(decomposition, k):
        """
        Verify multipliers are within bounds [1, k] for generalized Zeckendorf.

        Args:
            decomposition: List of (value, multiplier) or (index, multiplier)
            k: Maximum allowed multiplier
        """
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                multiplier = item[1]
                if not (1 <= multiplier <= k):
                    raise InvariantViolation(
                        f"Multiplier bound invariant violated: "
                        f"multiplier {multiplier} not in [1, {k}]"
                    )

    @staticmethod
    def is_descending_by_value(decomposition):
        """
        Check if decomposition is ordered by descending value.

        RTL processing naturally produces descending order since we
        process from largest to smallest sequence terms.

        Args:
            decomposition: List of values or (value, multiplier) tuples

        Returns:
            bool: True if in descending value order
        """
        values = []
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                values.append(item[0])
            else:
                values.append(item)

        return all(values[i] >= values[i+1] for i in range(len(values)-1))


class RTLProcessingInvariant:
    """
    Invariants specific to right-to-left processing operations.

    RTL processing is the mechanical foundation of Zeckendorf algorithms.
    These invariants capture the guarantees that RTL traversal provides.
    """

    @staticmethod
    def verify_rtl_order(original_sequence, processed_sequence):
        """
        Verify that processed_sequence is the RTL (reversed) order of original.

        Args:
            original_sequence: Sequence in ascending order
            processed_sequence: Sequence after RTL transformation
        """
        orig_list = list(original_sequence)
        proc_list = list(processed_sequence)

        if proc_list != list(reversed(orig_list)):
            raise InvariantViolation(
                "RTL order invariant violated: processed sequence "
                "is not the reverse of original"
            )

    @staticmethod
    def greedy_selection_valid(current_value, selected_term, remaining):
        """
        Verify that a greedy selection step is valid.

        In RTL greedy decomposition:
        - selected_term <= current_value (we can use it)
        - remaining = current_value - selected_term >= 0
        - No larger valid term was skipped

        Args:
            current_value: Value being decomposed
            selected_term: Term chosen from sequence
            remaining: Value remaining after selection

        Returns:
            bool: True if selection is valid
        """
        if selected_term > current_value:
            return False
        if remaining != current_value - selected_term:
            return False
        if remaining < 0:
            return False
        return True

    @staticmethod
    def verify_greedy_optimality(value, decomposition, sequence):
        """
        Verify that decomposition is the greedy optimal (canonical) representation.

        The greedy RTL algorithm always produces the unique Zeckendorf representation.
        This verifies that no larger terms could have been used.

        Args:
            value: Original value
            decomposition: Produced decomposition (descending order)
            sequence: Full sequence used (ascending order)
        """
        seq_list = list(sequence)
        decomp_values = []
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                decomp_values.append(item[0])
            else:
                decomp_values.append(item)

        # Simulate greedy RTL and compare
        remaining = value
        expected = []
        for term in reversed(seq_list):
            if term <= remaining:
                expected.append(term)
                remaining -= term

        if expected != decomp_values:
            raise InvariantViolation(
                f"Greedy optimality invariant violated: "
                f"expected {expected}, got {decomp_values}"
            )


class RTLMechanicalStructure:
    """
    Encapsulates the complete RTL mechanical structure for Zeckendorf operations.

    This class provides:
    1. RTL-aware sequence generation
    2. Invariant-preserving decomposition
    3. Verification at each step

    The mechanical structure ensures that all operations maintain the
    fundamental invariants required for correct Zeckendorf representations.
    """

    def __init__(self, generator_func, k=1, verify=True):
        """
        Initialize RTL mechanical structure.

        Args:
            generator_func: Sequence generator function
            k: Generalization parameter (1 for standard Fibonacci)
            verify: Whether to verify invariants (disable for performance)
        """
        self.generator = generator_func
        self.k = k
        self.verify = verify
        self._sequence_cache = {}

    def generate_sequence_up_to(self, n):
        """
        Generate sequence terms up to n (ascending order).

        Args:
            n: Upper bound

        Returns:
            list: Sequence terms in ascending order
        """
        if n in self._sequence_cache:
            return self._sequence_cache[n]

        seq = list(takewhile(lambda x: x <= n, self.generator()))

        if self.verify:
            SequenceInvariant.verify_monotonicity(seq)

        self._sequence_cache[n] = seq
        return seq

    def rtl_iterator(self, n):
        """
        Get RTL (right-to-left, descending) iterator for sequence terms up to n.

        This is the fundamental RTL operation that enables greedy decomposition.

        Args:
            n: Upper bound

        Yields:
            Sequence terms in descending order
        """
        seq = self.generate_sequence_up_to(n)
        for term in reversed(seq):
            yield term

    def decompose(self, n):
        """
        Decompose n using RTL greedy algorithm with invariant verification.

        Args:
            n: Value to decompose

        Returns:
            list: Decomposition terms in descending order
        """
        if n <= 0:
            raise ValueError("Can only decompose positive integers")

        original_n = n
        decomposition = []
        indexes_used = []

        seq = self.generate_sequence_up_to(n)

        for idx, term in enumerate(reversed(seq)):
            original_idx = len(seq) - 1 - idx
            if term <= n:
                if self.k > 1:
                    # Generalized: find maximum valid multiplier
                    for mult in reversed(range(1, self.k + 1)):
                        if term * mult <= n:
                            decomposition.append((term, mult))
                            n -= term * mult
                            indexes_used.append(original_idx)
                            break
                else:
                    # Standard: multiplier is always 1
                    decomposition.append(term)
                    n -= term
                    indexes_used.append(original_idx)

        # Verify invariants
        if self.verify:
            DecompositionInvariant.verify_sum_strict(decomposition, original_n)
            if self.k == 1:
                DecompositionInvariant.verify_non_adjacency(indexes_used)
            else:
                DecompositionInvariant.verify_multiplier_bounds(decomposition, self.k)

        return decomposition

    def decompose_to_indexes(self, n):
        """
        Decompose n to sequence indexes using RTL greedy algorithm.

        Args:
            n: Value to decompose

        Returns:
            list: Indexes in the sequence (or (index, multiplier) for generalized)
        """
        if n <= 0:
            raise ValueError("Can only decompose positive integers")

        original_n = n
        indexes = []

        seq = self.generate_sequence_up_to(n)

        for idx, term in enumerate(reversed(seq)):
            original_idx = len(seq) - 1 - idx
            if term <= n:
                if self.k > 1:
                    for mult in reversed(range(1, self.k + 1)):
                        if term * mult <= n:
                            indexes.append((original_idx, mult))
                            n -= term * mult
                            break
                else:
                    indexes.append(original_idx)
                    n -= term

        return indexes

    def verify_decomposition(self, n, decomposition):
        """
        Fully verify a decomposition against all invariants.

        Args:
            n: Original value
            decomposition: Decomposition to verify

        Returns:
            bool: True if all invariants satisfied

        Raises:
            InvariantViolation: If any invariant is violated
        """
        # Verify sum correctness
        DecompositionInvariant.verify_sum_strict(decomposition, n)

        # Extract values for further checks
        values = []
        indexes = []
        seq = self.generate_sequence_up_to(n)

        for item in decomposition:
            if isinstance(item, (list, tuple)):
                val, mult = item
                values.append(val)
                DecompositionInvariant.verify_multiplier_bounds([item], self.k)
            else:
                values.append(item)

        # Verify all values are from sequence
        for val in values:
            if val not in seq:
                raise InvariantViolation(
                    f"Value {val} not in sequence"
                )

        # Get indexes for non-adjacency check
        for val in values:
            indexes.append(seq.index(val))

        # Verify non-adjacency for standard Zeckendorf
        if self.k == 1:
            DecompositionInvariant.verify_non_adjacency(indexes)

        # Verify descending order
        if not DecompositionInvariant.is_descending_by_value(decomposition):
            raise InvariantViolation(
                "Decomposition not in descending order"
            )

        return True


def with_invariant_verification(invariant_check):
    """
    Decorator to add invariant verification to functions.

    Args:
        invariant_check: Function that verifies invariants on the result

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            invariant_check(result, *args, **kwargs)
            return result
        return wrapper
    return decorator


# Pre-built sequence generators

def fibonacci_generator():
    """
    Standard Fibonacci sequence generator for Zeckendorf: 1, 2, 3, 5, 8, ...

    Note: We skip the duplicate 1 at the start (F(1)=F(2)=1) to maintain
    strict monotonicity required for Zeckendorf representation.
    The Zeckendorf theorem uses F(2), F(3), F(4), ... = 1, 2, 3, 5, 8, ...
    """
    a, b = 1, 1
    yield 1  # First term
    while True:
        a, b = b, a + b
        yield b  # 2, 3, 5, 8, ...


def lucas_generator():
    """
    Lucas sequence generator for Zeckendorf-like representation: 1, 2, 3, 4, 7, 11, ...

    Note: The complete Lucas representational system uses L_1, L_0, L_2, L_3, ... = 1, 2, 3, 4, 7...
    We order as 1, 2, 3, 4, 7, 11, ... to maintain monotonicity.
    This allows representing all positive integers.
    """
    yield 1
    yield 2
    a, b = 1, 3
    while True:
        yield b
        a, b = b, a + b


def generalized_fibonacci_generator(k):
    """
    Generalized Fibonacci generator with parameter k.

    Recurrence: a_{n+1} = a_{n-1} + k * a_n

    Args:
        k: Generalization parameter

    Returns:
        Generator function
    """
    def gen():
        a, b = 1, 1
        while True:
            yield b
            a, b = b, a + b * k
    return gen


# Convenience instances
FibonacciRTL = RTLMechanicalStructure(fibonacci_generator, k=1)
LucasRTL = RTLMechanicalStructure(lucas_generator, k=1)


def create_generalized_rtl(k, verify=True):
    """
    Create an RTL mechanical structure for generalized Zeckendorf.

    Args:
        k: Generalization parameter
        verify: Whether to verify invariants

    Returns:
        RTLMechanicalStructure instance
    """
    return RTLMechanicalStructure(
        generalized_fibonacci_generator(k),
        k=k,
        verify=verify
    )


if __name__ == "__main__":
    # Demonstration of RTL mechanical invariants
    print("RTL Mechanical Invariants - Demonstration")
    print("=" * 50)

    # Standard Fibonacci
    print("\n1. Standard Fibonacci RTL Decomposition")
    print("-" * 40)

    for n in [10, 20, 50, 100]:
        decomp = FibonacciRTL.decompose(n)
        print(f"  {n} = {' + '.join(map(str, decomp))}")

    # Verify invariants explicitly
    print("\n2. Invariant Verification")
    print("-" * 40)

    seq = FibonacciRTL.generate_sequence_up_to(100)
    print(f"  Sequence (up to 100): {seq}")
    print(f"  Strictly monotonic: {SequenceInvariant.is_strictly_monotonic(seq)}")

    decomp = FibonacciRTL.decompose(50)
    indexes = FibonacciRTL.decompose_to_indexes(50)
    print(f"  50 decomposition: {decomp}")
    print(f"  50 indexes: {indexes}")
    print(f"  Non-adjacent indexes: {DecompositionInvariant.is_non_adjacent(indexes)}")
    print(f"  Sum correct: {DecompositionInvariant.verify_sum(decomp, 50)}")

    # Generalized Zeckendorf
    print("\n3. Generalized Zeckendorf (k=2)")
    print("-" * 40)

    gzeck2 = create_generalized_rtl(k=2)
    for n in [10, 20, 50]:
        decomp = gzeck2.decompose(n)
        terms = [f"{v}*{m}" for v, m in decomp]
        print(f"  {n} = {' + '.join(terms)}")
