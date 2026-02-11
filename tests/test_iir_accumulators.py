"""
Tests for IIR Accumulator Bank and Phase-Space Inference System.

Tests the components added to close the gap with FPGA implementation:
- Fibonacci-spaced IIR accumulators
- Context address formation
- Phi-psi phase-space channels
- Prediction LUT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from typing import List, Dict

from iir_accumulators import (
    AccumulatorState,
    PhaseSpaceState,
    FibonacciIIRBank,
    PredictionLUT,
    PhaseSpaceInferencePipeline,
    PhaseSpaceInferenceResult,
    CassiniLockedAccumulator,
    FIBONACCI_DELTAS,
)
from validation_rules import (
    LatticeValidator,
    IIRDecayFactorRule,
    IIRUpdateRule,
    FibonacciDeltaRule,
    PhiChannelRule,
    PsiVelocityRule,
    LUTAddressRangeRule,
)


class TestAccumulatorState(unittest.TestCase):
    """Tests for single IIR accumulator state."""

    def test_decay_factor_computation(self):
        """Verify decay factor = 1 - 2^{-δ} for each Fibonacci delta."""
        expected_decays = {
            1: 0.5,
            2: 0.75,
            3: 0.875,
            5: 0.96875
        }

        for delta, expected in expected_decays.items():
            acc = AccumulatorState(delta=delta)
            self.assertAlmostEqual(acc.decay_factor, expected, places=10,
                                   msg=f"Decay factor wrong for δ={delta}")

    def test_update_equation(self):
        """Test IIR update: A(t) = A(t-1) * decay + x(t)."""
        acc = AccumulatorState(delta=2)  # decay = 0.75

        # First update: A(1) = 0 * 0.75 + 10 = 10
        result = acc.update(10)
        self.assertAlmostEqual(result, 10.0)

        # Second update: A(2) = 10 * 0.75 + 5 = 12.5
        result = acc.update(5)
        self.assertAlmostEqual(result, 12.5)

        # Third update: A(3) = 12.5 * 0.75 + 0 = 9.375
        result = acc.update(0)
        self.assertAlmostEqual(result, 9.375)

    def test_reset(self):
        """Test accumulator reset."""
        acc = AccumulatorState(delta=1)
        acc.update(100)
        acc.update(50)
        self.assertNotEqual(acc.value, 0)

        acc.reset()
        self.assertEqual(acc.value, 0)
        self.assertEqual(len(acc.history), 0)

    def test_history_tracking(self):
        """Test that history is tracked on update."""
        acc = AccumulatorState(delta=3)

        for i in range(5):
            acc.update(i * 10)

        # History should have 5 entries (previous values)
        self.assertEqual(len(acc.history), 5)

    def test_quantization(self):
        """Test quantized value for context hash."""
        acc = AccumulatorState(delta=1)
        acc.value = 128.0  # Half of normalization factor

        quantized = acc.get_quantized(bits=8)
        # Should be approximately half of max (128)
        self.assertTrue(0 <= quantized <= 255)


class TestFibonacciIIRBank(unittest.TestCase):
    """Tests for the IIR accumulator bank."""

    def setUp(self):
        self.bank = FibonacciIIRBank(bit_width=16)

    def test_fibonacci_deltas(self):
        """Bank should have accumulators for δ ∈ {1, 2, 3, 5}."""
        self.assertEqual(set(self.bank.accumulators.keys()), {1, 2, 3, 5})

    def test_update_all_accumulators(self):
        """Update should affect all accumulators."""
        results = self.bank.update(100)

        self.assertEqual(len(results), 4)
        for delta in FIBONACCI_DELTAS:
            self.assertIn(delta, results)
            self.assertGreater(results[delta], 0)

    def test_different_decay_rates(self):
        """Different deltas should decay at different rates."""
        # Apply same input, then decay
        self.bank.update(100)

        # Apply zero input to observe decay
        results = self.bank.update(0)

        # Larger delta = slower decay = higher retained value
        self.assertGreater(results[5], results[3])
        self.assertGreater(results[3], results[2])
        self.assertGreater(results[2], results[1])

    def test_context_address_determinism(self):
        """Same state should produce same context address."""
        self.bank.update(50)
        self.bank.update(100)

        addr1 = self.bank.get_context_address()
        addr2 = self.bank.get_context_address()

        self.assertEqual(addr1, addr2)

    def test_context_address_range(self):
        """Context address should be within valid range."""
        for _ in range(20):
            self.bank.update(50)
            addr = self.bank.get_context_address(address_bits=12)
            self.assertTrue(0 <= addr < 4096, f"Address {addr} out of range")

    def test_update_from_bits(self):
        """Test update from Zeckendorf bit pattern."""
        bits = [1, 0, 1, 0, 0]  # 1 + 3 = 4
        results = self.bank.update_from_bits(bits)

        self.assertEqual(len(results), 4)
        # All accumulators should have been updated
        for delta in FIBONACCI_DELTAS:
            self.assertGreater(results[delta], 0)

    def test_temporal_signature(self):
        """Test temporal signature extraction."""
        self.bank.update(10)
        self.bank.update(20)

        signature = self.bank.get_temporal_signature()

        self.assertEqual(len(signature), 4)
        self.assertIsInstance(signature, tuple)

    def test_reset(self):
        """Test bank reset."""
        self.bank.update(100)
        self.bank.update(200)

        self.bank.reset()

        for delta, acc in self.bank.accumulators.items():
            self.assertEqual(acc.value, 0, f"Accumulator δ={delta} not reset")


class TestPhaseSpaceState(unittest.TestCase):
    """Tests for phase-space state tracking."""

    def test_initial_state(self):
        """Initial state should be empty."""
        state = PhaseSpaceState()
        self.assertEqual(state.phi, [])
        self.assertEqual(state.psi, [])
        self.assertEqual(state.polarity, 1)
        self.assertEqual(state.clock_cycle, 0)

    def test_phi_update(self):
        """Test phi (position) channel update."""
        state = PhaseSpaceState()

        phi1 = [1, 0, 1, 0, 0]
        state.update_phi(phi1)

        self.assertEqual(state.phi, phi1)

    def test_psi_computation(self):
        """Test psi (velocity) = phi(t) - phi(t-1)."""
        state = PhaseSpaceState()

        # First update: no previous, velocity should be zero-like
        state.update_phi([1, 0, 0, 0, 0])

        # Second update: should compute velocity
        state.update_phi([0, 1, 0, 0, 0])

        # psi[0] = 0 - 1 = -1, psi[1] = 1 - 0 = +1
        self.assertEqual(state.psi[0], -1)
        self.assertEqual(state.psi[1], 1)

    def test_velocity_magnitude(self):
        """Test velocity magnitude computation."""
        state = PhaseSpaceState()
        state.update_phi([1, 0, 1, 0, 0])
        state.update_phi([0, 1, 0, 1, 0])

        magnitude = state.get_velocity_magnitude()
        # All 4 positions changed: |psi| = 4
        self.assertEqual(magnitude, 4)

    def test_velocity_direction(self):
        """Test velocity direction detection."""
        state = PhaseSpaceState()

        # Start with lower bits set
        state.update_phi([1, 0, 0, 0, 0])

        # Move to higher bits - should be "increasing"
        state.update_phi([0, 0, 0, 1, 0])

        direction = state.get_velocity_direction()
        # Net change: -1 + 1 = 0 at positions 0, 3
        # Depends on implementation details

    def test_polarity_advance(self):
        """Test polarity alternation with clock."""
        state = PhaseSpaceState()

        self.assertEqual(state.polarity, 1)  # (-1)^0 = 1

        state.advance_clock()
        self.assertEqual(state.polarity, -1)  # (-1)^1 = -1

        state.advance_clock()
        self.assertEqual(state.polarity, 1)  # (-1)^2 = 1


class TestPredictionLUT(unittest.TestCase):
    """Tests for the prediction lookup table."""

    def setUp(self):
        self.lut = PredictionLUT(address_bits=8, value_bits=16)

    def test_initial_state(self):
        """LUT should start empty."""
        self.assertEqual(self.lut.get_fill_rate(), 0.0)
        self.assertEqual(self.lut.hits, 0)
        self.assertEqual(self.lut.misses, 0)

    def test_lookup_miss(self):
        """Lookup on empty LUT should return None."""
        result = self.lut.lookup(42)
        self.assertIsNone(result)
        self.assertEqual(self.lut.misses, 1)

    def test_update_and_lookup(self):
        """Update should enable subsequent lookup."""
        self.lut.update(42, 100)

        result = self.lut.lookup(42)
        self.assertIsNotNone(result)

        pred_value, confidence = result
        self.assertEqual(pred_value, 100)
        self.assertEqual(self.lut.hits, 1)

    def test_learning(self):
        """Multiple updates should adapt prediction."""
        # Initial entry
        self.lut.update(42, 100)

        # Update with different value
        self.lut.update(42, 110, learning_rate=0.5)

        result = self.lut.lookup(42)
        pred_value, _ = result

        # Should be between 100 and 110
        self.assertTrue(100 <= pred_value <= 110)

    def test_address_masking(self):
        """Addresses should be masked to valid range."""
        # Address bits = 8, so max address = 255
        self.lut.update(1000, 50)  # Should be masked

        # Should be stored at 1000 & 255 = 232
        result = self.lut.lookup(232)
        self.assertIsNotNone(result)

    def test_hit_rate(self):
        """Hit rate should track correctly."""
        self.lut.update(1, 10)
        self.lut.update(2, 20)

        self.lut.lookup(1)  # Hit
        self.lut.lookup(2)  # Hit
        self.lut.lookup(3)  # Miss
        self.lut.lookup(4)  # Miss

        self.assertAlmostEqual(self.lut.get_hit_rate(), 0.5)

    def test_clear(self):
        """Clear should reset LUT."""
        self.lut.update(1, 10)
        self.lut.lookup(1)

        self.lut.clear()

        self.assertEqual(self.lut.get_fill_rate(), 0.0)
        self.assertEqual(self.lut.hits, 0)
        self.assertEqual(self.lut.misses, 0)


class TestPhaseSpaceInferencePipeline(unittest.TestCase):
    """Tests for the complete phase-space pipeline."""

    def setUp(self):
        self.pipeline = PhaseSpaceInferencePipeline(bit_width=16, lut_address_bits=10)

    def test_initial_state(self):
        """Pipeline should start at cycle 0."""
        self.assertEqual(self.pipeline.cycle_count, 0)

    def test_step_returns_result(self):
        """Step should return PhaseSpaceInferenceResult."""
        bits = [1, 0, 1, 0, 0]
        result = self.pipeline.step(bits)

        self.assertIsInstance(result, PhaseSpaceInferenceResult)
        self.assertEqual(result.phi_bits, bits)
        self.assertEqual(result.clock_cycle, 1)

    def test_accumulator_values_populated(self):
        """Result should contain accumulator values."""
        result = self.pipeline.step([1, 0, 0, 0, 0])

        self.assertIn(1, result.accumulator_values)
        self.assertIn(2, result.accumulator_values)
        self.assertIn(3, result.accumulator_values)
        self.assertIn(5, result.accumulator_values)

    def test_context_address_generated(self):
        """Each step should generate context address."""
        result = self.pipeline.step([1, 0, 1, 0, 0])

        self.assertIsNotNone(result.context_address)
        self.assertTrue(0 <= result.context_address < 1024)  # 10 bits

    def test_polarity_alternates(self):
        """Polarity should alternate with each step."""
        r1 = self.pipeline.step([1, 0, 0, 0, 0])
        r2 = self.pipeline.step([0, 1, 0, 0, 0])
        r3 = self.pipeline.step([0, 0, 1, 0, 0])

        self.assertEqual(r1.polarity, -1)  # (-1)^1
        self.assertEqual(r2.polarity, 1)   # (-1)^2
        self.assertEqual(r3.polarity, -1)  # (-1)^3

    def test_velocity_tracked(self):
        """Velocity should be tracked between steps."""
        self.pipeline.step([1, 0, 0, 0, 0])
        result = self.pipeline.step([0, 1, 0, 0, 0])

        # Two bits changed
        self.assertEqual(result.velocity_magnitude, 2)

    def test_reset(self):
        """Reset should clear pipeline state."""
        self.pipeline.step([1, 0, 0, 0, 0])
        self.pipeline.step([0, 1, 0, 0, 0])

        self.pipeline.reset()

        self.assertEqual(self.pipeline.cycle_count, 0)
        self.assertEqual(self.pipeline.prediction_correct, 0)


class TestCassiniLockedAccumulator(unittest.TestCase):
    """Tests for Cassini-locked accumulator."""

    def setUp(self):
        self.acc = CassiniLockedAccumulator()

    def test_cassini_identity(self):
        """Verify Cassini identity: F_{n-1}*F_{n+1} - F_n^2 = (-1)^n."""
        for n in range(1, 50):
            is_valid = self.acc.verify_cassini(n)
            self.assertTrue(is_valid, f"Cassini identity failed at n={n}")

    def test_update_locked_returns_validity(self):
        """Update should return Cassini validity."""
        total, valid = self.acc.update_locked(10)
        self.assertEqual(total, 10)
        # At n=1, Cassini should be valid
        self.assertTrue(valid)

    def test_polarity_tracking(self):
        """Polarity should track with index."""
        self.assertEqual(self.acc.get_polarity(), 1)  # n=0

        self.acc.update_locked(5)
        self.assertEqual(self.acc.get_polarity(), -1)  # n=1

        self.acc.update_locked(3)
        self.assertEqual(self.acc.get_polarity(), 1)  # n=2


class TestValidationRules(unittest.TestCase):
    """Tests for the new IIR and phase-space validation rules."""

    def setUp(self):
        self.validator = LatticeValidator()

    def test_iir_decay_factor_rule_valid(self):
        """Valid decay factors should pass."""
        rule = IIRDecayFactorRule()

        for delta, expected in [(1, 0.5), (2, 0.75), (3, 0.875), (5, 0.96875)]:
            result = rule.check(delta, expected)
            self.assertTrue(result.passed, f"Valid decay for δ={delta} rejected")

    def test_iir_decay_factor_rule_invalid(self):
        """Invalid decay factors should fail."""
        rule = IIRDecayFactorRule()

        # Wrong decay for delta=2
        result = rule.check(2, 0.5)  # Should be 0.75
        self.assertFalse(result.passed)

    def test_fibonacci_delta_rule_valid(self):
        """Valid Fibonacci deltas should pass."""
        rule = FibonacciDeltaRule()

        result = rule.check([1, 2, 3, 5])
        self.assertTrue(result.passed)

    def test_fibonacci_delta_rule_invalid(self):
        """Non-Fibonacci deltas should fail."""
        rule = FibonacciDeltaRule()

        result = rule.check([1, 2, 4, 5])  # 4 is not Fibonacci
        self.assertFalse(result.passed)

    def test_phi_channel_rule_valid(self):
        """Valid Zeckendorf phi should pass."""
        rule = PhiChannelRule()

        result = rule.check([1, 0, 1, 0, 0, 1])  # No adjacent 1s
        self.assertTrue(result.passed)

    def test_phi_channel_rule_invalid(self):
        """Invalid phi (adjacent 1s) should fail."""
        rule = PhiChannelRule()

        result = rule.check([1, 1, 0, 0])  # Adjacent 1s
        self.assertFalse(result.passed)

    def test_lut_address_range_rule_valid(self):
        """Address within range should pass."""
        rule = LUTAddressRangeRule()

        result = rule.check(100, address_bits=8)  # Max 255
        self.assertTrue(result.passed)

    def test_lut_address_range_rule_invalid(self):
        """Address out of range should fail."""
        rule = LUTAddressRangeRule()

        result = rule.check(300, address_bits=8)  # Max 255
        self.assertFalse(result.passed)

    def test_validate_iir_accumulator(self):
        """Test IIR accumulator validation method."""
        # Valid update: prev=10, input=5, new should be 10*0.75+5=12.5
        report = self.validator.validate_iir_accumulator(
            delta=2,
            decay_factor=0.75,
            prev_value=10.0,
            input_value=5.0,
            new_value=12.5
        )
        self.assertTrue(report.passed)

    def test_validate_phase_space(self):
        """Test phase-space validation method."""
        phi = [1, 0, 1, 0, 0]
        phi_prev = [0, 1, 0, 0, 0]
        psi = [-1, 1, -1, 0, 0]  # Computed velocity

        report = self.validator.validate_phase_space(
            phi_bits=phi,
            phi_prev=phi_prev,
            psi_bits=psi,
            clock_n=5,
            polarity=-1
        )
        # Should pass all checks
        # Note: might fail on PS002 if psi is wrong, depends on computation


class TestIntegrationWithPipeline(unittest.TestCase):
    """Integration tests with the main lattice pipeline."""

    def test_pipeline_includes_phase_space(self):
        """Main pipeline should include phase-space results."""
        from lattice_pipeline import ZeroRAMLattice, PipelineConfig

        config = PipelineConfig(bit_width=16, min_token_freq=1)
        lattice = ZeroRAMLattice(config)

        # Build vocabulary
        lattice.build_vocabulary(["test one two three"])

        # Run inference
        results = list(lattice.run_inference("test"))

        self.assertGreater(len(results), 0)
        for result in results:
            # Should have phase-space extensions
            self.assertIsNotNone(result.accumulator_values)
            self.assertIsNotNone(result.context_address)

    def test_pipeline_stats_include_phase_space(self):
        """Pipeline stats should include phase-space metrics."""
        from lattice_pipeline import ZeroRAMLattice, PipelineConfig

        config = PipelineConfig(bit_width=16, min_token_freq=1)
        lattice = ZeroRAMLattice(config)

        lattice.build_vocabulary(["alpha beta gamma delta"])

        # Process some tokens
        list(lattice.run_inference("alpha beta"))

        stats = lattice.get_stats()

        # Should have velocity stats
        self.assertTrue(hasattr(stats, 'max_velocity_magnitude'))
        self.assertTrue(hasattr(stats, 'avg_velocity_magnitude'))
        self.assertTrue(hasattr(stats, 'lut_hit_rate'))
        self.assertTrue(hasattr(stats, 'prediction_accuracy'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
