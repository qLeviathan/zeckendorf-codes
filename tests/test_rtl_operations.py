"""
Test suite for RTL Operations module.

Tests the RTL-aware sequence operations, decomposition, and encoding.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from rtl_operations import (
    RTLSequence,
    RTLDecomposer,
    RTLEncoder,
    RTLDecoder,
    RTLCompressionPipeline,
    FibonacciSeq,
    LucasSeq,
    create_gzeck_sequence,
    fibonacci_gen,
)


class TestRTLSequence(unittest.TestCase):
    """Tests for RTLSequence class."""

    def test_terms_up_to(self):
        """terms_up_to should return ascending sequence."""
        terms = FibonacciSeq.terms_up_to(50)
        expected = [1, 2, 3, 5, 8, 13, 21, 34]
        self.assertEqual(terms, expected)

    def test_rtl_terms(self):
        """rtl_terms should return descending sequence."""
        rtl = FibonacciSeq.rtl_terms(50)
        expected = [34, 21, 13, 8, 5, 3, 2, 1]
        self.assertEqual(rtl, expected)

    def test_rtl_enumerate(self):
        """rtl_enumerate should yield (index, term) in descending order."""
        result = list(FibonacciSeq.rtl_enumerate(20))
        # Sequence up to 20: [1, 2, 3, 5, 8, 13]
        # RTL enumerate: (5, 13), (4, 8), (3, 5), (2, 3), (1, 2), (0, 1)
        expected = [(5, 13), (4, 8), (3, 5), (2, 3), (1, 2), (0, 1)]
        self.assertEqual(result, expected)

    def test_index_of(self):
        """index_of should return correct index."""
        # Sequence: [1, 2, 3, 5, 8, 13, ...]
        self.assertEqual(FibonacciSeq.index_of(1), 0)
        self.assertEqual(FibonacciSeq.index_of(2), 1)
        self.assertEqual(FibonacciSeq.index_of(5), 3)
        self.assertEqual(FibonacciSeq.index_of(13), 5)

    def test_term_at(self):
        """term_at should return correct term."""
        # Sequence: [1, 2, 3, 5, 8, 13, ...]
        self.assertEqual(FibonacciSeq.term_at(0), 1)
        self.assertEqual(FibonacciSeq.term_at(1), 2)
        self.assertEqual(FibonacciSeq.term_at(3), 5)
        self.assertEqual(FibonacciSeq.term_at(5), 13)

    def test_lucas_sequence(self):
        """Lucas sequence should be correct."""
        terms = LucasSeq.terms_up_to(50)
        expected = [1, 2, 3, 4, 7, 11, 18, 29, 47]
        self.assertEqual(terms, expected)

    def test_gzeck_sequence(self):
        """Generalized Zeckendorf sequence should be correct."""
        gzeck2 = create_gzeck_sequence(2)
        terms = gzeck2.terms_up_to(50)
        # a_{n+1} = a_{n-1} + 2*a_n, starting 1, 1
        # 1, 3, 7, 17, 41, ... (note: we skip duplicate 1 for strict monotonicity)
        # Actually the gen yields: 1, then b where a,b = 1,1; next a,b = 1, 1+2*1=3, yield 3
        # then a,b = 3, 1+2*3=7, yield 7 etc.
        expected = [1, 3, 7, 17, 41]
        self.assertEqual(terms, expected)

    def test_verify_structure(self):
        """verify_structure should pass for valid sequences."""
        # Should not raise
        FibonacciSeq.verify_structure(1000)
        LucasSeq.verify_structure(1000)

    def test_caching(self):
        """Sequence should use caching for efficiency."""
        seq = RTLSequence(fibonacci_gen, "Test")

        # First call
        terms1 = seq.terms_up_to(100)
        # Second call should use cache
        terms2 = seq.terms_up_to(50)  # Subset of cached
        terms3 = seq.terms_up_to(100)  # Same as cached

        self.assertEqual(terms1, terms3)
        self.assertEqual(terms2, [t for t in terms1 if t <= 50])


class TestRTLDecomposer(unittest.TestCase):
    """Tests for RTLDecomposer class."""

    def setUp(self):
        self.decomposer = RTLDecomposer(FibonacciSeq, k=1)

    def test_initial_state(self):
        """initial_state should be correctly initialized."""
        state = self.decomposer.initial_state(50)
        self.assertEqual(state.value, 50)
        self.assertEqual(state.remaining, 50)
        self.assertEqual(state.position, 0)
        self.assertEqual(state.decomposition, [])

    def test_select_term_valid(self):
        """select_term should return valid selection."""
        state = self.decomposer.initial_state(50)
        selection = self.decomposer.select_term(state, 7, 34)

        self.assertIsNotNone(selection)
        self.assertEqual(selection.term, 34)
        self.assertEqual(selection.index, 7)
        self.assertEqual(selection.multiplier, 1)
        self.assertEqual(selection.remaining, 16)

    def test_select_term_too_large(self):
        """select_term should return None for term > remaining."""
        state = self.decomposer.initial_state(10)
        selection = self.decomposer.select_term(state, 6, 13)

        self.assertIsNone(selection)

    def test_apply_selection(self):
        """apply_selection should produce correct new state."""
        state = self.decomposer.initial_state(50)
        selection = self.decomposer.select_term(state, 7, 34)
        new_state = self.decomposer.apply_selection(state, selection)

        self.assertEqual(new_state.value, 50)
        self.assertEqual(new_state.remaining, 16)
        self.assertEqual(new_state.position, 1)
        self.assertEqual(new_state.decomposition, [(34, 1)])

    def test_decompose_basic(self):
        """decompose should produce correct results."""
        # 50 = 34 + 13 + 3
        decomp = self.decomposer.decompose(50)
        values = [t for t, m in decomp]
        self.assertEqual(values, [34, 13, 3])

    def test_decompose_fibonacci_number(self):
        """Decomposing Fibonacci number should return single term."""
        decomp = self.decomposer.decompose(13)
        self.assertEqual(decomp, [(13, 1)])

    def test_decompose_with_trace(self):
        """decompose_with_trace should yield all steps."""
        trace = list(self.decomposer.decompose_with_trace(10))

        # 10 = 8 + 2, so should have 2 steps
        self.assertEqual(len(trace), 2)

        # First step: select 8
        before1, sel1, after1 = trace[0]
        self.assertEqual(sel1.term, 8)
        self.assertEqual(before1.remaining, 10)
        self.assertEqual(after1.remaining, 2)

        # Second step: select 2
        before2, sel2, after2 = trace[1]
        self.assertEqual(sel2.term, 2)
        self.assertEqual(before2.remaining, 2)
        self.assertEqual(after2.remaining, 0)


class TestRTLDecomposerGeneralized(unittest.TestCase):
    """Tests for generalized Zeckendorf decomposition."""

    def test_gzeck_k2_decomposition(self):
        """k=2 generalized decomposition should work."""
        gzeck2 = create_gzeck_sequence(2)
        decomposer = RTLDecomposer(gzeck2, k=2)

        for n in range(1, 51):
            decomp = decomposer.decompose(n)
            total = sum(t * m for t, m in decomp)
            self.assertEqual(total, n, f"Failed for n={n}")

            # Verify multipliers in bounds
            for t, m in decomp:
                self.assertGreaterEqual(m, 1)
                self.assertLessEqual(m, 2)

    def test_gzeck_k3_decomposition(self):
        """k=3 generalized decomposition should work."""
        gzeck3 = create_gzeck_sequence(3)
        decomposer = RTLDecomposer(gzeck3, k=3)

        for n in range(1, 51):
            decomp = decomposer.decompose(n)
            total = sum(t * m for t, m in decomp)
            self.assertEqual(total, n, f"Failed for n={n}")


class TestRTLEncoder(unittest.TestCase):
    """Tests for RTLEncoder class."""

    def test_encode_standard(self):
        """Standard encoding (k=1) should work."""
        encoder = RTLEncoder(k=1)

        # Decomposition of 10: indexes 2 (value 2) and 5 (value 8)
        # Wait, let's think about this more carefully
        # Fib sequence: idx 0=1, 1=1, 2=2, 3=3, 4=5, 5=8
        # 10 = 8 + 2 = F_5 + F_2
        decomp = [(5, 1), (2, 1)]  # (index, multiplier)

        encoded = encoder.encode_decomposition(decomp, max_index=5)
        # Positions 0,1,2,3,4,5 -> should have 1s at positions 2 and 5
        expected = '001001'  # positions 2 and 5 are 1
        self.assertEqual(encoded, expected)

    def test_encode_generalized(self):
        """Generalized encoding (k>1) should work."""
        encoder = RTLEncoder(k=3)

        # bits_per_multiplier = ceil(log2(4)) = 2
        self.assertEqual(encoder.bits_per_multiplier, 2)

        # Example: positions 0 with mult 2, position 2 with mult 1
        decomp = [(0, 2), (2, 1)]
        encoded = encoder.encode_decomposition(decomp, max_index=2)
        # Position 0: mult 2 -> '10'
        # Position 1: mult 0 -> '00'
        # Position 2: mult 1 -> '01'
        expected = '100001'
        self.assertEqual(encoded, expected)

    def test_encode_term(self):
        """encode_term should return correct EncodedTerm."""
        encoder = RTLEncoder(k=1)
        et = encoder.encode_term(index=3, multiplier=1)

        self.assertEqual(et.index, 3)
        self.assertEqual(et.multiplier, 1)
        self.assertEqual(et.bits, '1')


class TestRTLDecoder(unittest.TestCase):
    """Tests for RTLDecoder class."""

    def test_decode_standard(self):
        """Standard decoding (k=1) should work."""
        decoder = RTLDecoder(FibonacciSeq, k=1, max_value=100)

        # Encode 10 = 8 + 2
        # Fib sequence: [1, 2, 3, 5, 8, 13, ...]
        # 10 = 8 (idx 4) + 2 (idx 1)
        bits = '010010'  # positions 1 and 4
        decoded = decoder.decode_binary(bits)
        self.assertEqual(decoded, 10)

    def test_decode_generalized(self):
        """Generalized decoding (k>1) should work."""
        gzeck2 = create_gzeck_sequence(2)
        decoder = RTLDecoder(gzeck2, k=2, max_value=100)

        # bits_per_multiplier = ceil(log2(3)) = 2 for k=2
        self.assertEqual(decoder.bits_per_multiplier, 2)

        # Encode with multipliers
        # Sequence: [1, 3, 7, ...]
        # Each position uses 2 bits for multiplier
        # bits = '0101' means position 0: mult 1, position 1: mult 1
        # Value: 1*1 + 3*1 = 4
        bits = '0101'
        decoded = decoder.decode_binary(bits)
        self.assertEqual(decoded, 4)


class TestRTLCompressionPipeline(unittest.TestCase):
    """Tests for RTLCompressionPipeline class."""

    def setUp(self):
        self.pipeline = RTLCompressionPipeline(FibonacciSeq, k=1)

    def test_compress_number(self):
        """compress_number should produce valid encoding."""
        for n in range(1, 101):
            compressed = self.pipeline.compress_number(n)
            # Should be binary string
            self.assertTrue(all(c in '01' for c in compressed))
            # Should be non-empty
            self.assertTrue(len(compressed) > 0)

    def test_roundtrip(self):
        """Compression should be reversible."""
        decoder = RTLDecoder(FibonacciSeq, k=1, max_value=255)

        for n in range(1, 101):
            compressed = self.pipeline.compress_number(n)
            decompressed = decoder.decode_binary(compressed)
            self.assertEqual(decompressed, n, f"Failed for n={n}")

    def test_compress_stream(self):
        """compress_stream should yield compressed numbers with delimiters."""
        numbers = [1, 5, 10]
        stream = list(self.pipeline.compress_stream(numbers))

        self.assertEqual(len(stream), 3)
        for item in stream:
            self.assertTrue(item.endswith('1'))  # Default delimiter


class TestRTLOperationsEdgeCases(unittest.TestCase):
    """Edge case tests for RTL operations."""

    def test_decompose_one(self):
        """Decomposing 1 should work."""
        decomposer = RTLDecomposer(FibonacciSeq, k=1)
        decomp = decomposer.decompose(1)
        self.assertEqual(decomp, [(1, 1)])

    def test_decompose_large(self):
        """Large number decomposition should work."""
        decomposer = RTLDecomposer(FibonacciSeq, k=1)
        n = 10000
        decomp = decomposer.decompose(n)

        total = sum(t * m for t, m in decomp)
        self.assertEqual(total, n)

    def test_empty_decomposition_impossible(self):
        """Positive integers should always have non-empty decomposition."""
        decomposer = RTLDecomposer(FibonacciSeq, k=1)

        for n in range(1, 101):
            decomp = decomposer.decompose(n)
            self.assertTrue(len(decomp) > 0)

    def test_pipeline_generalized(self):
        """Generalized pipeline should work."""
        gzeck3 = create_gzeck_sequence(3)
        pipeline = RTLCompressionPipeline(gzeck3, k=3)

        for n in range(1, 51):
            compressed = pipeline.compress_number(n)
            self.assertTrue(len(compressed) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
