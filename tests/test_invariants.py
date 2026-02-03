"""
Test suite for RTL Mechanical Invariants.

This module tests that all invariant classes correctly identify
valid and invalid states, and that the RTL mechanical structure
properly maintains invariants during operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from invariants import (
    SequenceInvariant,
    DecompositionInvariant,
    RTLProcessingInvariant,
    RTLMechanicalStructure,
    InvariantViolation,
    fibonacci_generator,
    lucas_generator,
    generalized_fibonacci_generator,
    FibonacciRTL,
    LucasRTL,
    create_generalized_rtl,
)


class TestSequenceInvariant(unittest.TestCase):
    """Tests for SequenceInvariant class."""

    def test_strictly_monotonic_valid(self):
        """Valid strictly monotonic sequences should pass."""
        self.assertTrue(SequenceInvariant.is_strictly_monotonic([1, 2, 3, 5, 8]))
        self.assertTrue(SequenceInvariant.is_strictly_monotonic([1, 1000, 2000]))
        self.assertTrue(SequenceInvariant.is_strictly_monotonic([1]))
        self.assertTrue(SequenceInvariant.is_strictly_monotonic([]))

    def test_strictly_monotonic_invalid(self):
        """Non-monotonic sequences should fail."""
        self.assertFalse(SequenceInvariant.is_strictly_monotonic([1, 2, 2, 5]))
        self.assertFalse(SequenceInvariant.is_strictly_monotonic([1, 3, 2, 5]))
        self.assertFalse(SequenceInvariant.is_strictly_monotonic([5, 4, 3, 2, 1]))

    def test_verify_monotonicity_raises(self):
        """verify_monotonicity should raise InvariantViolation for invalid sequences."""
        with self.assertRaises(InvariantViolation):
            SequenceInvariant.verify_monotonicity([1, 2, 2, 3])

    def test_fibonacci_monotonicity(self):
        """Fibonacci sequence should be strictly monotonic."""
        fib = list(FibonacciRTL.generate_sequence_up_to(1000))
        self.assertTrue(SequenceInvariant.is_strictly_monotonic(fib))

    def test_complete_for_range_fibonacci(self):
        """Fibonacci sequence should be complete."""
        fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        self.assertTrue(SequenceInvariant.is_complete_for_range(fib, 100))

    def test_incomplete_sequence(self):
        """Sequence with gap should not be complete."""
        # Missing 1, can't represent 1
        self.assertFalse(SequenceInvariant.is_complete_for_range([2, 3, 5], 5))

    def test_growth_bound_fibonacci(self):
        """Standard Fibonacci should satisfy k=1 growth bound."""
        fib = list(FibonacciRTL.generate_sequence_up_to(1000))
        self.assertTrue(SequenceInvariant.satisfies_growth_bound(fib, k=1))


class TestDecompositionInvariant(unittest.TestCase):
    """Tests for DecompositionInvariant class."""

    def test_verify_sum_simple(self):
        """Simple sum verification should work."""
        self.assertTrue(DecompositionInvariant.verify_sum([5, 3, 2], 10))
        self.assertTrue(DecompositionInvariant.verify_sum([8, 2], 10))
        self.assertFalse(DecompositionInvariant.verify_sum([5, 3], 10))

    def test_verify_sum_with_multipliers(self):
        """Sum verification with multipliers should work."""
        # (value, multiplier) pairs
        self.assertTrue(DecompositionInvariant.verify_sum([(5, 2)], 10))
        self.assertTrue(DecompositionInvariant.verify_sum([(3, 2), (2, 2)], 10))
        self.assertFalse(DecompositionInvariant.verify_sum([(3, 2), (2, 1)], 10))

    def test_verify_sum_strict_raises(self):
        """verify_sum_strict should raise on mismatch."""
        with self.assertRaises(InvariantViolation):
            DecompositionInvariant.verify_sum_strict([5, 3], 10)

    def test_non_adjacent_valid(self):
        """Non-adjacent indexes should pass."""
        self.assertTrue(DecompositionInvariant.is_non_adjacent([0, 2, 4]))
        self.assertTrue(DecompositionInvariant.is_non_adjacent([1, 5, 10]))
        self.assertTrue(DecompositionInvariant.is_non_adjacent([0]))
        self.assertTrue(DecompositionInvariant.is_non_adjacent([]))

    def test_non_adjacent_invalid(self):
        """Adjacent indexes should fail."""
        self.assertFalse(DecompositionInvariant.is_non_adjacent([0, 1, 3]))
        self.assertFalse(DecompositionInvariant.is_non_adjacent([2, 3]))
        self.assertFalse(DecompositionInvariant.is_non_adjacent([5, 6, 7]))

    def test_verify_non_adjacency_raises(self):
        """verify_non_adjacency should raise for adjacent indexes."""
        with self.assertRaises(InvariantViolation):
            DecompositionInvariant.verify_non_adjacency([1, 2, 5])

    def test_multiplier_bounds_valid(self):
        """Valid multipliers should pass."""
        DecompositionInvariant.verify_multiplier_bounds([(5, 1), (3, 2)], k=3)
        DecompositionInvariant.verify_multiplier_bounds([(5, 3)], k=3)

    def test_multiplier_bounds_invalid(self):
        """Out-of-bounds multipliers should raise."""
        with self.assertRaises(InvariantViolation):
            DecompositionInvariant.verify_multiplier_bounds([(5, 4)], k=3)
        with self.assertRaises(InvariantViolation):
            DecompositionInvariant.verify_multiplier_bounds([(5, 0)], k=3)

    def test_descending_by_value(self):
        """Descending order detection should work."""
        self.assertTrue(DecompositionInvariant.is_descending_by_value([8, 5, 2]))
        self.assertTrue(DecompositionInvariant.is_descending_by_value([(8, 1), (5, 2)]))
        self.assertFalse(DecompositionInvariant.is_descending_by_value([2, 5, 8]))


class TestRTLProcessingInvariant(unittest.TestCase):
    """Tests for RTLProcessingInvariant class."""

    def test_verify_rtl_order_valid(self):
        """Valid RTL transformation should pass."""
        orig = [1, 2, 3, 5, 8]
        rtl = [8, 5, 3, 2, 1]
        RTLProcessingInvariant.verify_rtl_order(orig, rtl)

    def test_verify_rtl_order_invalid(self):
        """Invalid RTL transformation should raise."""
        orig = [1, 2, 3, 5, 8]
        wrong = [8, 5, 3, 1, 2]  # Wrong order
        with self.assertRaises(InvariantViolation):
            RTLProcessingInvariant.verify_rtl_order(orig, wrong)

    def test_greedy_selection_valid(self):
        """Valid greedy selections should pass."""
        self.assertTrue(
            RTLProcessingInvariant.greedy_selection_valid(
                current_value=10, selected_term=8, remaining=2
            )
        )
        self.assertTrue(
            RTLProcessingInvariant.greedy_selection_valid(
                current_value=5, selected_term=5, remaining=0
            )
        )

    def test_greedy_selection_invalid(self):
        """Invalid greedy selections should fail."""
        # Term too large
        self.assertFalse(
            RTLProcessingInvariant.greedy_selection_valid(
                current_value=5, selected_term=8, remaining=-3
            )
        )
        # Wrong remaining
        self.assertFalse(
            RTLProcessingInvariant.greedy_selection_valid(
                current_value=10, selected_term=8, remaining=5
            )
        )


class TestRTLMechanicalStructure(unittest.TestCase):
    """Tests for RTLMechanicalStructure class."""

    def test_fibonacci_decomposition_basic(self):
        """Basic Fibonacci decompositions should be correct."""
        for n in range(1, 101):
            decomp = FibonacciRTL.decompose(n)
            self.assertTrue(
                DecompositionInvariant.verify_sum(decomp, n),
                f"Failed for n={n}: {decomp}"
            )

    def test_fibonacci_decomposition_non_adjacent(self):
        """Fibonacci decompositions should have non-adjacent indexes."""
        for n in range(1, 101):
            indexes = FibonacciRTL.decompose_to_indexes(n)
            self.assertTrue(
                DecompositionInvariant.is_non_adjacent(indexes),
                f"Adjacent indexes for n={n}: {indexes}"
            )

    def test_fibonacci_decomposition_descending(self):
        """Fibonacci decompositions should be in descending order."""
        for n in [10, 50, 100, 500]:
            decomp = FibonacciRTL.decompose(n)
            self.assertTrue(
                DecompositionInvariant.is_descending_by_value(decomp),
                f"Not descending for n={n}: {decomp}"
            )

    def test_lucas_decomposition(self):
        """Lucas decompositions should be correct."""
        for n in range(1, 101):
            decomp = LucasRTL.decompose(n)
            self.assertTrue(
                DecompositionInvariant.verify_sum(decomp, n),
                f"Failed for n={n}: {decomp}"
            )

    def test_generalized_zeckendorf_k2(self):
        """Generalized Zeckendorf (k=2) should work correctly."""
        gzeck2 = create_generalized_rtl(k=2)

        for n in range(1, 51):
            decomp = gzeck2.decompose(n)
            self.assertTrue(
                DecompositionInvariant.verify_sum(decomp, n),
                f"Failed for n={n}: {decomp}"
            )
            # Verify multiplier bounds
            for term, mult in decomp:
                self.assertGreaterEqual(mult, 1)
                self.assertLessEqual(mult, 2)

    def test_generalized_zeckendorf_k3(self):
        """Generalized Zeckendorf (k=3) should work correctly."""
        gzeck3 = create_generalized_rtl(k=3)

        for n in range(1, 51):
            decomp = gzeck3.decompose(n)
            self.assertTrue(
                DecompositionInvariant.verify_sum(decomp, n),
                f"Failed for n={n}: {decomp}"
            )

    def test_rtl_iterator(self):
        """RTL iterator should produce descending sequence."""
        rtl_terms = list(FibonacciRTL.rtl_iterator(100))
        self.assertEqual(rtl_terms, sorted(rtl_terms, reverse=True))

    def test_verify_decomposition(self):
        """Full decomposition verification should work."""
        for n in [10, 50, 100]:
            decomp = FibonacciRTL.decompose(n)
            self.assertTrue(FibonacciRTL.verify_decomposition(n, decomp))

    def test_known_fibonacci_decompositions(self):
        """Test specific known Zeckendorf representations."""
        known = {
            1: [1],
            2: [2],
            3: [3],
            4: [3, 1],  # 4 = 3 + 1
            5: [5],
            6: [5, 1],  # 6 = 5 + 1
            7: [5, 2],  # 7 = 5 + 2
            10: [8, 2],  # 10 = 8 + 2
            20: [13, 5, 2],  # 20 = 13 + 5 + 2
            50: [34, 13, 3],  # 50 = 34 + 13 + 3
            100: [89, 8, 3],  # 100 = 89 + 8 + 3
        }

        for n, expected in known.items():
            decomp = FibonacciRTL.decompose(n)
            self.assertEqual(decomp, expected, f"n={n}: got {decomp}, expected {expected}")


class TestInvariantViolationDetection(unittest.TestCase):
    """Tests that invariant violations are properly detected."""

    def test_detect_invalid_decomposition_sum(self):
        """Should detect wrong sum in decomposition."""
        with self.assertRaises(InvariantViolation):
            FibonacciRTL.verify_decomposition(10, [8, 3])  # 11 != 10

    def test_detect_adjacent_indexes(self):
        """Should detect adjacent indexes in standard Zeckendorf."""
        # Create a decomposition with adjacent Fibonacci numbers
        # F_4=3, F_5=5 are adjacent
        with self.assertRaises(InvariantViolation):
            FibonacciRTL.verify_decomposition(8, [5, 3])  # 3 and 5 are adjacent Fibs


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_decompose_one(self):
        """Decomposing 1 should work."""
        decomp = FibonacciRTL.decompose(1)
        self.assertEqual(decomp, [1])

    def test_decompose_fibonacci_number(self):
        """Decomposing a Fibonacci number should return just that number."""
        fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for f in fibs:
            decomp = FibonacciRTL.decompose(f)
            self.assertEqual(decomp, [f])

    def test_decompose_zero_raises(self):
        """Decomposing 0 should raise."""
        with self.assertRaises(ValueError):
            FibonacciRTL.decompose(0)

    def test_decompose_negative_raises(self):
        """Decomposing negative should raise."""
        with self.assertRaises(ValueError):
            FibonacciRTL.decompose(-5)

    def test_large_number(self):
        """Should handle large numbers correctly."""
        n = 10000
        decomp = FibonacciRTL.decompose(n)
        self.assertTrue(DecompositionInvariant.verify_sum(decomp, n))

        indexes = FibonacciRTL.decompose_to_indexes(n)
        self.assertTrue(DecompositionInvariant.is_non_adjacent(indexes))


if __name__ == '__main__':
    unittest.main(verbosity=2)
