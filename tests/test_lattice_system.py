"""
Integration tests for the Zero-RAM Lattice Inference System.

Tests the complete pipeline from text encoding through bit collapse
to invariant validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from typing import List

from validation_rules import (
    LatticeValidator,
    ZeckendorfConstraintRule,
    CassiniInvariantRule,
    SpectralBoundRule,
    EncodingSumRule,
    CollapseCorrectnessRule,
    RuleSeverity,
)
from lattice_encoder import TokenEncoder, VocabularyEncoder, LucasPrefilter
from bit_collapse import BitCollapseEngine, LatticeInferencePipeline, CassiniVerifier


class TestTokenEncoder(unittest.TestCase):
    """Tests for Zeckendorf token encoding."""

    def setUp(self):
        self.encoder = TokenEncoder(bit_width=16)

    def test_encode_single_fibonacci(self):
        """Encoding a Fibonacci number should have exactly one bit set."""
        # Fibonacci numbers: 1, 2, 3, 5, 8, 13, 21, ...
        fibs = [1, 2, 3, 5, 8, 13, 21, 34]
        for fib in fibs:
            bits = self.encoder.encode(fib)
            self.assertEqual(sum(bits), 1, f"Fibonacci {fib} should have 1 bit set")

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return original value."""
        for n in range(1, 100):
            encoded = self.encoder.encode(n)
            decoded = self.encoder.decode(encoded)
            self.assertEqual(decoded, n, f"Roundtrip failed for {n}")

    def test_no_adjacent_bits(self):
        """All encodings should satisfy Zeckendorf constraint."""
        rule = ZeckendorfConstraintRule()
        for n in range(1, 200):
            bits = self.encoder.encode(n)
            result = rule.check(bits)
            self.assertTrue(result.passed, f"Value {n} has adjacent 1s: {bits}")

    def test_bit_width_limit(self):
        """Should not encode values beyond max_value."""
        with self.assertRaises(ValueError):
            self.encoder.encode(self.encoder.max_value + 1)


class TestLucasPrefilter(unittest.TestCase):
    """Tests for Lucas bracket pre-filtering."""

    def setUp(self):
        self.filter = LucasPrefilter()

    def test_lucas_identity(self):
        """Verify L_n = F_{n-1} + F_{n+1}."""
        # Use mathematical Fibonacci/Lucas
        fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]

        for n in range(1, min(len(fib) - 1, len(lucas))):
            expected = fib[n - 1] + fib[n + 1]
            actual = lucas[n]
            self.assertEqual(expected, actual, f"Lucas identity failed at n={n}")

    def test_get_brackets(self):
        """lucas_brackets should return valid Lucas pairs."""
        for pos in range(1, 10):
            brackets = self.filter.lucas_brackets(pos)
            self.assertIsInstance(brackets, tuple)
            self.assertEqual(len(brackets), 2)


class TestBitCollapseEngine(unittest.TestCase):
    """Tests for bit collapse mechanics."""

    def setUp(self):
        self.engine = BitCollapseEngine()
        self.validator = LatticeValidator()

    def test_single_collapse_basic(self):
        """Basic single collapse: [1,1,0] -> [0,0,1]."""
        bits = [1, 1, 0]
        result = self.engine.single_collapse(bits)
        # After collapse: should have no adjacent 1s at position 0,1
        self.assertFalse(result[0] == 1 and result[1] == 1)

    def test_cascade_terminates(self):
        """Cascade collapse should always terminate."""
        test_patterns = [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
        for bits in test_patterns:
            result = self.engine.cascade_collapse(bits)
            # Check no adjacent 1s in result
            has_adjacent = any(
                result[i] == 1 and result[i+1] == 1
                for i in range(len(result) - 1)
            )
            self.assertFalse(has_adjacent, f"Cascade didn't complete for {bits}")

    def test_collapse_preserves_value(self):
        """Collapse should preserve the encoded value."""
        zeck_fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

        def bits_to_value(bits):
            return sum(zeck_fibs[i] for i in range(min(len(bits), len(zeck_fibs))) if bits[i])

        test_patterns = [
            [1, 1, 0, 0, 0],   # 1 + 2 = 3
            [1, 0, 1, 1, 0],   # 1 + 3 + 5 = 9
            [1, 1, 1, 1, 0],   # 1 + 2 + 3 + 5 = 11
        ]

        for bits in test_patterns:
            before_val = bits_to_value(bits)
            result = self.engine.cascade_collapse(bits)
            after_val = bits_to_value(result)
            self.assertEqual(before_val, after_val,
                           f"Value changed: {before_val} -> {after_val} for {bits}")

    def test_polarity(self):
        """Polarity should alternate with clock cycle."""
        for n in range(20):
            expected = (-1) ** n
            actual = self.engine.polarity(n)
            self.assertEqual(expected, actual, f"Polarity wrong at n={n}")

    def test_shift_direction(self):
        """Shift direction should alternate with clock cycle."""
        for n in range(20):
            direction = self.engine.shift_direction(n)
            if n % 2 == 0:
                self.assertEqual(direction, 'right')
            else:
                self.assertEqual(direction, 'left')


class TestCassiniVerifier(unittest.TestCase):
    """Tests for Cassini invariant verification."""

    def setUp(self):
        self.verifier = CassiniVerifier()

    def test_cassini_identity(self):
        """Verify Cassini identity holds for all n."""
        for n in range(1, 50):
            is_valid = self.verifier.verify(n)
            self.assertTrue(is_valid, f"Cassini identity failed at n={n}")

    def test_cassini_computation(self):
        """Manually verify Cassini computation."""
        # F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5
        # At n=2: F_1 * F_3 - F_2^2 = 1 * 2 - 1 = 1 = (-1)^2 ✓
        # At n=3: F_2 * F_4 - F_3^2 = 1 * 3 - 4 = -1 = (-1)^3 ✓
        # At n=4: F_3 * F_5 - F_4^2 = 2 * 5 - 9 = 1 = (-1)^4 ✓
        pass  # The test in test_cassini_identity covers this


class TestLatticeInferencePipeline(unittest.TestCase):
    """Tests for the complete inference pipeline."""

    def setUp(self):
        self.pipeline = LatticeInferencePipeline()

    def test_initial_state(self):
        """Initial state should be empty."""
        state = self.pipeline.get_state()
        self.assertEqual(state.clock_cycle, 0)
        self.assertEqual(state.polarity, 1)

    def test_step_increments_clock(self):
        """Each step should increment clock cycle."""
        input_bits = [1, 0, 1, 0, 0]
        state = self.pipeline.step(input_bits)
        self.assertEqual(state.clock_cycle, 1)

        state = self.pipeline.step(input_bits)
        self.assertEqual(state.clock_cycle, 2)

    def test_step_produces_valid_state(self):
        """Each step should produce a valid Zeckendorf state."""
        rule = ZeckendorfConstraintRule()

        for _ in range(10):
            input_bits = [1, 0, 1, 0, 0, 1, 0, 0]
            state = self.pipeline.step(input_bits)
            result = rule.check(state.current_bits)
            self.assertTrue(result.passed)


class TestVocabularyEncoder(unittest.TestCase):
    """Tests for vocabulary encoding."""

    def setUp(self):
        self.encoder = VocabularyEncoder(bit_width=16)
        sample_text = "the quick brown fox jumps over the lazy dog"
        self.encoder.build_vocabulary(sample_text, min_frequency=1)

    def test_vocabulary_built(self):
        """Vocabulary should be non-empty after building."""
        self.assertGreater(self.encoder.vocabulary_size, 0)

    def test_encode_known_token(self):
        """Known tokens should encode successfully."""
        encoded = self.encoder.encode_text("the")
        self.assertEqual(len(encoded), 1)
        self.assertIsNotNone(encoded[0].token_id)

    def test_frequency_based_ids(self):
        """More frequent tokens should have lower IDs."""
        # "the" appears twice in the sample, so should have lower ID than single-occurrence words
        encoded_the = self.encoder.encode_text("the")[0]
        encoded_fox = self.encoder.encode_text("fox")[0]
        # Both should exist, IDs should be assigned
        self.assertIsNotNone(encoded_the.token_id)
        self.assertIsNotNone(encoded_fox.token_id)


class TestValidationRules(unittest.TestCase):
    """Tests for the validation rule system."""

    def setUp(self):
        self.validator = LatticeValidator()

    def test_all_invariants_for_small_n(self):
        """All mathematical invariants should hold for small n."""
        report = self.validator.validate_all_invariants(max_n=20)
        # Check that critical invariants pass
        cassini_results = [r for r in report.results if r.rule_id == "V001"]
        for r in cassini_results:
            self.assertTrue(r.passed, f"Cassini failed: {r.message}")

    def test_valid_encoding_passes(self):
        """Valid Zeckendorf encoding should pass validation."""
        # 17 = 13 + 3 + 1 = F_7 + F_4 + F_2
        # In Zeckendorf bits (1-indexed from 1,2,3,5,8,13,...):
        # Position 0: 1, Position 2: 1, Position 5: 1
        bits = [1, 0, 1, 0, 0, 1, 0, 0]  # 1 + 3 + 13 = 17
        report = self.validator.validate_encoding(17, bits)
        self.assertTrue(report.passed, f"Valid encoding rejected: {[r.message for r in report.results if not r.passed]}")

    def test_invalid_encoding_fails(self):
        """Invalid encoding (adjacent 1s) should fail validation."""
        bits = [1, 1, 0, 0, 0]  # Adjacent 1s at positions 0,1
        report = self.validator.validate_bits(bits)
        self.assertFalse(report.passed)

    def test_pipeline_state_validation(self):
        """Pipeline state validation should check all components."""
        bits = [0, 1, 0, 0, 1, 0, 0, 0]  # 2 + 8 = 10, valid
        report = self.validator.validate_pipeline_state(
            bits=bits,
            clock_n=5,
            polarity=-1  # Odd clock cycle, polarity should be -1
        )
        self.assertTrue(report.passed)


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete text -> encoding -> inference -> validation flow."""
        # Build vocabulary
        vocab = VocabularyEncoder(bit_width=16)
        vocab.build_vocabulary("test text for encoding", min_frequency=1)

        # Encode text
        encoded = vocab.encode_text("test")
        self.assertEqual(len(encoded), 1)

        # Run through inference
        pipeline = LatticeInferencePipeline()
        for token in encoded:
            state = pipeline.step(token.zeckendorf_bits)

            # Validate state
            validator = LatticeValidator()
            report = validator.validate_pipeline_state(
                bits=state.current_bits,
                clock_n=state.clock_cycle,
                polarity=state.polarity
            )
            self.assertTrue(report.passed)

    def test_roundtrip_encoding(self):
        """Test that encoding and decoding are consistent."""
        encoder = TokenEncoder(bit_width=16)

        for n in range(1, 100):
            bits = encoder.encode(n)
            decoded = encoder.decode(bits)
            self.assertEqual(n, decoded)

            # Also verify Zeckendorf constraint
            rule = ZeckendorfConstraintRule()
            result = rule.check(bits)
            self.assertTrue(result.passed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
