"""
Verification Utilities for Zeckendorf Decompositions

This module provides utilities for:
1. Verifying decomposition correctness
2. Comparing different representation strategies
3. Analyzing representation efficiency
4. Debugging and diagnostics
"""

from collections import namedtuple
from itertools import takewhile
import math

from invariants import (
    SequenceInvariant,
    DecompositionInvariant,
    RTLProcessingInvariant,
    InvariantViolation,
    FibonacciRTL,
    LucasRTL,
    create_generalized_rtl,
)
from rtl_operations import (
    RTLSequence,
    RTLDecomposer,
    RTLEncoder,
    RTLDecoder,
    FibonacciSeq,
    LucasSeq,
    create_gzeck_sequence,
)


# Result types
VerificationResult = namedtuple('VerificationResult', [
    'valid',
    'n',
    'decomposition',
    'errors',
])

ComparisonResult = namedtuple('ComparisonResult', [
    'n',
    'representations',
    'bit_lengths',
    'most_efficient',
])

EfficiencyStats = namedtuple('EfficiencyStats', [
    'avg_bits',
    'max_bits',
    'min_bits',
    'total_bits',
    'compression_ratio',
])


class DecompositionVerifier:
    """
    Comprehensive verification of Zeckendorf decompositions.

    Checks:
    - Sum correctness
    - Term validity (all terms in sequence)
    - Non-adjacency (for standard Zeckendorf)
    - Multiplier bounds (for generalized)
    - Greedy optimality (canonical form)
    """

    def __init__(self, sequence, k=1):
        """
        Initialize verifier.

        Args:
            sequence: RTLSequence or RTLMechanicalStructure
            k: Maximum multiplier
        """
        if hasattr(sequence, 'generate_sequence_up_to'):
            self.structure = sequence
            self.sequence = None
        else:
            self.sequence = sequence
            self.structure = None
        self.k = k

    def _get_terms(self, n):
        """Get sequence terms up to n."""
        if self.structure:
            return self.structure.generate_sequence_up_to(n)
        return self.sequence.terms_up_to(n)

    def verify(self, n, decomposition):
        """
        Fully verify a decomposition.

        Args:
            n: Value that was decomposed
            decomposition: List of terms or (term, multiplier) pairs

        Returns:
            VerificationResult
        """
        errors = []

        # Extract values and check sum
        total = 0
        values = []
        multipliers = []

        for item in decomposition:
            if isinstance(item, (list, tuple)):
                val, mult = item
                values.append(val)
                multipliers.append(mult)
                total += val * mult
            else:
                values.append(item)
                multipliers.append(1)
                total += item

        # Check sum
        if total != n:
            errors.append(f"Sum mismatch: {total} != {n}")

        # Check all values are in sequence
        seq_terms = self._get_terms(n)
        for val in values:
            if val not in seq_terms:
                errors.append(f"Value {val} not in sequence")

        # Check non-adjacency for standard Zeckendorf
        if self.k == 1:
            indexes = [seq_terms.index(v) for v in values if v in seq_terms]
            if not DecompositionInvariant.is_non_adjacent(indexes):
                errors.append(f"Adjacent indexes detected: {sorted(indexes)}")

        # Check multiplier bounds for generalized
        if self.k > 1:
            for i, mult in enumerate(multipliers):
                if not (1 <= mult <= self.k):
                    errors.append(f"Multiplier {mult} out of bounds [1, {self.k}] at position {i}")

        # Check descending order
        if not DecompositionInvariant.is_descending_by_value(decomposition):
            errors.append("Decomposition not in descending order")

        return VerificationResult(
            valid=len(errors) == 0,
            n=n,
            decomposition=decomposition,
            errors=errors,
        )

    def verify_range(self, start, end):
        """
        Verify decompositions for a range of values.

        Args:
            start: Start of range (inclusive)
            end: End of range (exclusive)

        Yields:
            VerificationResult for each value
        """
        if self.structure:
            for n in range(start, end):
                decomp = self.structure.decompose(n)
                yield self.verify(n, decomp)
        else:
            decomposer = RTLDecomposer(self.sequence, self.k)
            for n in range(start, end):
                decomp = decomposer.decompose(n)
                yield self.verify(n, decomp)


class RepresentationComparator:
    """
    Compare different Zeckendorf representation strategies.

    Compares:
    - Standard Fibonacci
    - Lucas
    - Generalized with various k values
    """

    def __init__(self, strategies=None):
        """
        Initialize comparator.

        Args:
            strategies: Dict of {name: (sequence, k)} pairs
                       If None, uses defaults
        """
        if strategies is None:
            strategies = {
                'Fibonacci': (FibonacciSeq, 1),
                'Lucas': (LucasSeq, 1),
                'GZeck(2)': (create_gzeck_sequence(2), 2),
                'GZeck(3)': (create_gzeck_sequence(3), 3),
            }
        self.strategies = strategies
        self.decomposers = {
            name: RTLDecomposer(seq, k)
            for name, (seq, k) in strategies.items()
        }
        self.encoders = {
            name: RTLEncoder(k)
            for name, (seq, k) in strategies.items()
        }

    def compare(self, n):
        """
        Compare representations of n across strategies.

        Args:
            n: Value to represent

        Returns:
            ComparisonResult
        """
        representations = {}
        bit_lengths = {}

        for name, decomposer in self.decomposers.items():
            seq, k = self.strategies[name]
            encoder = self.encoders[name]

            decomp = decomposer.decompose(n)
            representations[name] = decomp

            # Calculate bit length
            indexes = []
            for item in decomp:
                if isinstance(item, (list, tuple)):
                    term, mult = item
                    idx = seq.index_of(term)
                    indexes.append((idx, mult))
                else:
                    idx = seq.index_of(item)
                    indexes.append((idx, 1))

            max_idx = max(idx for idx, _ in indexes)
            encoded = encoder.encode_decomposition(indexes, max_idx)
            bit_lengths[name] = len(encoded)

        # Find most efficient
        most_efficient = min(bit_lengths, key=bit_lengths.get)

        return ComparisonResult(
            n=n,
            representations=representations,
            bit_lengths=bit_lengths,
            most_efficient=most_efficient,
        )

    def compare_range(self, start, end):
        """
        Compare representations for a range.

        Yields:
            ComparisonResult for each value
        """
        for n in range(start, end):
            yield self.compare(n)


class EfficiencyAnalyzer:
    """
    Analyze efficiency of Zeckendorf representations.
    """

    def __init__(self, sequence, k=1):
        """
        Initialize analyzer.

        Args:
            sequence: RTLSequence instance
            k: Maximum multiplier
        """
        self.sequence = sequence
        self.k = k
        self.decomposer = RTLDecomposer(sequence, k)
        self.encoder = RTLEncoder(k)

    def bit_length(self, n):
        """
        Get bit length of compressed representation.

        Args:
            n: Value to compress

        Returns:
            int: Number of bits
        """
        decomp = self.decomposer.decompose(n)

        indexes = []
        for item in decomp:
            if isinstance(item, (list, tuple)):
                term, mult = item
                idx = self.sequence.index_of(term)
                indexes.append((idx, mult))
            else:
                idx = self.sequence.index_of(item)
                indexes.append((idx, 1))

        max_idx = max(idx for idx, _ in indexes)
        encoded = self.encoder.encode_decomposition(indexes, max_idx)
        return len(encoded)

    def analyze_range(self, start, end):
        """
        Analyze efficiency statistics for a range.

        Args:
            start: Start of range (inclusive)
            end: End of range (exclusive)

        Returns:
            EfficiencyStats
        """
        bit_lengths = []
        for n in range(start, end):
            bit_lengths.append(self.bit_length(n))

        total_bits = sum(bit_lengths)
        count = len(bit_lengths)

        # Compare to fixed-width encoding
        max_val = end - 1
        fixed_bits = math.ceil(math.log2(max_val + 1)) if max_val > 0 else 1
        fixed_total = fixed_bits * count

        return EfficiencyStats(
            avg_bits=total_bits / count,
            max_bits=max(bit_lengths),
            min_bits=min(bit_lengths),
            total_bits=total_bits,
            compression_ratio=fixed_total / total_bits if total_bits > 0 else 0,
        )


class DiagnosticPrinter:
    """
    Pretty-print diagnostics for debugging.
    """

    @staticmethod
    def print_decomposition(n, decomposition, sequence_name="Sequence"):
        """Print a decomposition in human-readable form."""
        if not decomposition:
            print(f"{n} = (empty decomposition)")
            return

        terms = []
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                val, mult = item
                if mult == 1:
                    terms.append(str(val))
                else:
                    terms.append(f"{val}*{mult}")
            else:
                terms.append(str(item))

        print(f"{n} = {' + '.join(terms)} ({sequence_name})")

    @staticmethod
    def print_rtl_trace(n, trace):
        """Print RTL decomposition trace."""
        print(f"\nRTL Decomposition Trace for {n}:")
        print("-" * 50)

        for i, (before, selection, after) in enumerate(trace):
            print(f"Step {i+1}:")
            print(f"  Before: remaining={before.remaining}")
            print(f"  Select: term={selection.term}, index={selection.index}, mult={selection.multiplier}")
            print(f"  After:  remaining={after.remaining}")
            print()

        print(f"Final decomposition: {after.decomposition}")

    @staticmethod
    def print_verification(result):
        """Print verification result."""
        status = "VALID" if result.valid else "INVALID"
        print(f"\nVerification of {result.n}: {status}")

        if result.valid:
            print(f"  Decomposition: {result.decomposition}")
        else:
            print(f"  Decomposition: {result.decomposition}")
            print("  Errors:")
            for error in result.errors:
                print(f"    - {error}")

    @staticmethod
    def print_comparison(result):
        """Print comparison result."""
        print(f"\nComparison for {result.n}:")
        print("-" * 40)

        for name, decomp in result.representations.items():
            bits = result.bit_lengths[name]
            marker = " *" if name == result.most_efficient else ""
            print(f"  {name}: {decomp} ({bits} bits){marker}")

        print(f"\nMost efficient: {result.most_efficient}")

    @staticmethod
    def print_efficiency_stats(stats, name=""):
        """Print efficiency statistics."""
        print(f"\nEfficiency Statistics{' for ' + name if name else ''}:")
        print("-" * 40)
        print(f"  Average bits: {stats.avg_bits:.2f}")
        print(f"  Min bits:     {stats.min_bits}")
        print(f"  Max bits:     {stats.max_bits}")
        print(f"  Total bits:   {stats.total_bits}")
        print(f"  Compression ratio: {stats.compression_ratio:.2f}x")


def verify_all_invariants(n, decomposition, sequence, k=1):
    """
    One-shot verification of all invariants.

    Args:
        n: Original value
        decomposition: Decomposition to verify
        sequence: RTLSequence instance
        k: Maximum multiplier

    Returns:
        tuple: (is_valid, list of error messages)
    """
    verifier = DecompositionVerifier(sequence, k)
    result = verifier.verify(n, decomposition)
    return result.valid, result.errors


def run_comprehensive_verification(max_n=1000):
    """
    Run comprehensive verification across all strategies.

    Args:
        max_n: Maximum value to verify

    Returns:
        dict: {strategy_name: (valid_count, invalid_count, errors)}
    """
    results = {}

    strategies = {
        'Fibonacci': (FibonacciRTL, 1),
        'Lucas': (LucasRTL, 1),
        'GZeck(2)': (create_generalized_rtl(2), 2),
        'GZeck(3)': (create_generalized_rtl(3), 3),
    }

    for name, (structure, k) in strategies.items():
        valid = 0
        invalid = 0
        errors = []

        for n in range(1, max_n + 1):
            try:
                decomp = structure.decompose(n)
                # Verify using the structure's own verification
                structure.verify_decomposition(n, decomp)
                valid += 1
            except InvariantViolation as e:
                invalid += 1
                errors.append((n, str(e)))
            except Exception as e:
                invalid += 1
                errors.append((n, f"Unexpected error: {e}"))

        results[name] = (valid, invalid, errors)

    return results


if __name__ == "__main__":
    print("Verification Utilities - Demonstration")
    print("=" * 60)

    # 1. Single decomposition verification
    print("\n1. Single Decomposition Verification")
    print("-" * 50)

    verifier = DecompositionVerifier(FibonacciSeq, k=1)

    # Valid decomposition
    result = verifier.verify(50, [34, 13, 3])
    DiagnosticPrinter.print_verification(result)

    # Invalid decomposition (wrong sum)
    result = verifier.verify(50, [34, 13, 2])
    DiagnosticPrinter.print_verification(result)

    # 2. RTL trace
    print("\n2. RTL Processing Trace")
    print("-" * 50)

    decomposer = RTLDecomposer(FibonacciSeq, k=1)
    trace = list(decomposer.decompose_with_trace(50))
    DiagnosticPrinter.print_rtl_trace(50, trace)

    # 3. Representation comparison
    print("\n3. Representation Comparison")
    print("-" * 50)

    comparator = RepresentationComparator()
    for n in [10, 50, 100]:
        result = comparator.compare(n)
        DiagnosticPrinter.print_comparison(result)

    # 4. Efficiency analysis
    print("\n4. Efficiency Analysis")
    print("-" * 50)

    for name, (seq, k) in [
        ('Fibonacci', (FibonacciSeq, 1)),
        ('GZeck(2)', (create_gzeck_sequence(2), 2)),
        ('GZeck(3)', (create_gzeck_sequence(3), 3)),
    ]:
        analyzer = EfficiencyAnalyzer(seq, k)
        stats = analyzer.analyze_range(1, 256)
        DiagnosticPrinter.print_efficiency_stats(stats, name)

    # 5. Comprehensive verification
    print("\n5. Comprehensive Verification (1-100)")
    print("-" * 50)

    results = run_comprehensive_verification(max_n=100)
    for name, (valid, invalid, errors) in results.items():
        print(f"  {name}: {valid} valid, {invalid} invalid")
        if errors:
            for n, err in errors[:3]:  # Show first 3 errors
                print(f"    n={n}: {err}")
