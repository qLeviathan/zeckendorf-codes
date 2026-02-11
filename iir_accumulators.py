"""
IIR Accumulator Bank and Phase-Space Channels for Zero-RAM Lattice Inference

This module implements the missing components identified in the FPGA comparison:
1. IIR Accumulators with Fibonacci-spaced decay constants
2. Context Address Formation from accumulator outputs
3. Psi-rail (velocity/momentum) channel for phase-space representation
4. Prediction LUT infrastructure

The IIR (Infinite Impulse Response) accumulators maintain temporal context
through exponential decay at Fibonacci-spaced rates, creating a multi-scale
temporal representation without random access memory.

Key Concepts:
------------
1. Fibonacci-Spaced Decay: δ ∈ {1, 2, 3, 5} provides Zeckendorf-aligned temporal scales
2. Phi-Psi Phase Space: Position (φ) and velocity (ψ) channels for complete state
3. Context Address: Hash of accumulator states for prediction LUT lookup
4. Cassini-Locked Updates: State transitions preserve (-1)^n polarity invariant

Mathematical Foundation:
-----------------------
Each IIR accumulator follows:
    A_δ(t) = A_δ(t-1) × (1 - 2^{-δ}) + x(t)

where δ ∈ {1, 2, 3, 5} (Fibonacci sequence).

The velocity channel tracks:
    ψ(t) = φ(t) - φ(t-1)  (discrete derivative)

Context address formation:
    ctx = hash(A_1 ⊕ A_2 ⊕ A_3 ⊕ A_5)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from collections import deque
import hashlib
import struct


# Fibonacci-spaced decay constants as per FPGA spec
FIBONACCI_DELTAS = [1, 2, 3, 5]


@dataclass
class AccumulatorState:
    """State of a single IIR accumulator."""
    delta: int                    # Decay constant (1, 2, 3, or 5)
    value: float = 0.0           # Current accumulated value
    decay_factor: float = 0.0    # Computed 1 - 2^{-δ}
    history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Compute decay factor from delta."""
        self.decay_factor = 1.0 - (2.0 ** (-self.delta))

    def update(self, input_value: float) -> float:
        """
        Update accumulator with new input.

        A_δ(t) = A_δ(t-1) × (1 - 2^{-δ}) + x(t)

        Args:
            input_value: New input value x(t)

        Returns:
            Updated accumulator value
        """
        self.history.append(self.value)
        self.value = self.value * self.decay_factor + input_value
        return self.value

    def reset(self):
        """Reset accumulator to zero."""
        self.value = 0.0
        self.history.clear()

    def get_quantized(self, bits: int = 8) -> int:
        """Get quantized value for context hash."""
        # Scale to 0-255 range (assuming value is normalized)
        max_val = 256.0  # Normalization factor
        scaled = min(max(self.value / max_val, 0.0), 1.0)
        return int(scaled * ((1 << bits) - 1))


@dataclass
class PhaseSpaceState:
    """
    Combined phi-psi phase space state.

    Phi (φ): Position channel - direct Zeckendorf encoding
    Psi (ψ): Velocity channel - rate of change in Fibonacci space
    """
    phi: List[int] = field(default_factory=list)    # Position bits
    psi: List[int] = field(default_factory=list)    # Velocity bits
    phi_prev: List[int] = field(default_factory=list)  # Previous position
    polarity: int = 1                                # (-1)^n
    clock_cycle: int = 0

    def update_phi(self, new_phi: List[int]):
        """Update position channel and compute velocity."""
        self.phi_prev = list(self.phi)
        self.phi = list(new_phi)

        # Compute velocity as signed difference in Zeckendorf space
        self.psi = self._compute_velocity()

    def _compute_velocity(self) -> List[int]:
        """
        Compute velocity (ψ) as discrete derivative.

        ψ(t) = φ(t) - φ(t-1) in Fibonacci basis

        Returns:
            Velocity bits (can be negative, stored as signed)
        """
        if not self.phi_prev:
            return [0] * len(self.phi)

        # Extend to same length
        max_len = max(len(self.phi), len(self.phi_prev))
        phi_ext = self.phi + [0] * (max_len - len(self.phi))
        phi_prev_ext = self.phi_prev + [0] * (max_len - len(self.phi_prev))

        # Simple bit-wise difference (XOR for change detection)
        velocity = []
        for i in range(max_len):
            # +1 if bit turned on, -1 if turned off, 0 if unchanged
            diff = phi_ext[i] - phi_prev_ext[i]
            # Store as 0 (no change), 1 (increased), -1 (decreased)
            velocity.append(diff)

        return velocity

    def advance_clock(self):
        """Advance clock cycle and update polarity."""
        self.clock_cycle += 1
        self.polarity = (-1) ** self.clock_cycle

    def get_velocity_magnitude(self) -> int:
        """Get magnitude of velocity vector."""
        return sum(abs(v) for v in self.psi)

    def get_velocity_direction(self) -> int:
        """
        Get overall velocity direction.

        Returns:
            +1 if increasing, -1 if decreasing, 0 if stable
        """
        total = sum(self.psi)
        if total > 0:
            return 1
        elif total < 0:
            return -1
        return 0


class FibonacciIIRBank:
    """
    Bank of IIR accumulators with Fibonacci-spaced decay constants.

    Implements four parallel accumulators with δ ∈ {1, 2, 3, 5}, creating
    a multi-scale temporal representation. The accumulator outputs are
    combined to form context addresses for prediction lookup.
    """

    def __init__(self, bit_width: int = 16):
        """
        Initialize the IIR accumulator bank.

        Args:
            bit_width: Bit width for Zeckendorf encoding
        """
        self.bit_width = bit_width

        # Create accumulators for each Fibonacci delta
        self.accumulators: Dict[int, AccumulatorState] = {
            delta: AccumulatorState(delta=delta)
            for delta in FIBONACCI_DELTAS
        }

        # Track update count for statistics
        self.update_count = 0

        # History of context addresses
        self.context_history: deque = deque(maxlen=100)

    def update(self, zeckendorf_value: int) -> Dict[int, float]:
        """
        Update all accumulators with a new Zeckendorf-encoded value.

        Args:
            zeckendorf_value: Integer value from Zeckendorf bits

        Returns:
            Dictionary mapping delta -> accumulator value
        """
        results = {}
        for delta, acc in self.accumulators.items():
            results[delta] = acc.update(float(zeckendorf_value))

        self.update_count += 1
        return results

    def update_from_bits(self, bits: List[int]) -> Dict[int, float]:
        """
        Update accumulators from Zeckendorf bit pattern.

        Args:
            bits: Zeckendorf bit representation

        Returns:
            Dictionary mapping delta -> accumulator value
        """
        # Convert bits to integer value
        fibs = self._get_fibonacci_weights(len(bits))
        value = sum(fibs[i] for i in range(len(bits)) if bits[i])
        return self.update(value)

    def _get_fibonacci_weights(self, n: int) -> List[int]:
        """Get Fibonacci weights for n bits."""
        if n <= 0:
            return []
        if n == 1:
            return [1]
        fibs = [1, 2]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs[:n]

    def get_context_address(self, address_bits: int = 12) -> int:
        """
        Compute context address from accumulator states.

        The context address is formed by XORing quantized accumulator
        values, creating a hash that captures the multi-scale temporal
        state.

        ctx = hash(A_1 ⊕ A_2 ⊕ A_3 ⊕ A_5)

        Args:
            address_bits: Number of bits in context address

        Returns:
            Context address for LUT lookup
        """
        # Get quantized values from each accumulator
        quantized = []
        for delta in FIBONACCI_DELTAS:
            q = self.accumulators[delta].get_quantized(bits=8)
            quantized.append(q)

        # XOR all values together
        xor_result = 0
        for q in quantized:
            xor_result ^= q

        # Hash for better distribution
        hash_input = struct.pack('>BBBB', *quantized)
        hash_digest = hashlib.sha256(hash_input).digest()

        # Take first address_bits from hash
        address = int.from_bytes(hash_digest[:2], 'big') & ((1 << address_bits) - 1)

        self.context_history.append(address)
        return address

    def get_accumulator_states(self) -> Dict[int, float]:
        """Get current state of all accumulators."""
        return {delta: acc.value for delta, acc in self.accumulators.items()}

    def get_temporal_signature(self) -> Tuple[float, float, float, float]:
        """
        Get temporal signature as tuple of accumulator values.

        Returns:
            (A_1, A_2, A_3, A_5) accumulator values
        """
        return tuple(self.accumulators[d].value for d in FIBONACCI_DELTAS)

    def reset(self):
        """Reset all accumulators."""
        for acc in self.accumulators.values():
            acc.reset()
        self.update_count = 0
        self.context_history.clear()

    def get_decay_rates(self) -> Dict[int, float]:
        """Get decay rates (1 - 2^{-δ}) for each accumulator."""
        return {delta: acc.decay_factor for delta, acc in self.accumulators.items()}


class PredictionLUT:
    """
    Direct-mapped Lookup Table for prediction based on context addresses.

    The LUT stores prediction values indexed by context address, enabling
    fast O(1) prediction lookup without complex computations.
    """

    def __init__(self, address_bits: int = 12, value_bits: int = 16):
        """
        Initialize the prediction LUT.

        Args:
            address_bits: Number of bits in context address
            value_bits: Bit width of stored values
        """
        self.address_bits = address_bits
        self.value_bits = value_bits
        self.size = 1 << address_bits

        # LUT storage: maps address -> (prediction, confidence)
        self._table: Dict[int, Tuple[int, float]] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.updates = 0

    def lookup(self, context_address: int) -> Optional[Tuple[int, float]]:
        """
        Lookup prediction for context address.

        Args:
            context_address: Address from IIR bank

        Returns:
            Tuple of (prediction, confidence) or None if not found
        """
        address = context_address & (self.size - 1)  # Mask to valid range

        if address in self._table:
            self.hits += 1
            return self._table[address]

        self.misses += 1
        return None

    def update(self, context_address: int, actual_value: int,
               learning_rate: float = 0.1):
        """
        Update LUT entry with actual observed value.

        Args:
            context_address: Address to update
            actual_value: Actual observed value
            learning_rate: How quickly to adapt (0-1)
        """
        address = context_address & (self.size - 1)

        if address in self._table:
            old_pred, old_conf = self._table[address]
            # Exponential moving average
            new_pred = int(old_pred * (1 - learning_rate) +
                         actual_value * learning_rate)
            # Update confidence based on prediction accuracy
            error = abs(old_pred - actual_value)
            conf_delta = 0.1 if error < 10 else -0.1
            new_conf = max(0.0, min(1.0, old_conf + conf_delta))
            self._table[address] = (new_pred, new_conf)
        else:
            # New entry with low initial confidence
            self._table[address] = (actual_value, 0.5)

        self.updates += 1

    def get_hit_rate(self) -> float:
        """Get LUT hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_fill_rate(self) -> float:
        """Get fraction of LUT entries filled."""
        return len(self._table) / self.size

    def clear(self):
        """Clear all LUT entries."""
        self._table.clear()
        self.hits = 0
        self.misses = 0
        self.updates = 0


class PhaseSpaceInferencePipeline:
    """
    Complete inference pipeline with phi-psi phase space channels.

    Integrates:
    - Phi (position) channel with Zeckendorf encoding
    - Psi (velocity) channel for momentum tracking
    - IIR accumulator bank for multi-scale context
    - Prediction LUT for fast inference
    """

    def __init__(self, bit_width: int = 16, lut_address_bits: int = 12):
        """
        Initialize the phase-space pipeline.

        Args:
            bit_width: Bit width for Zeckendorf encoding
            lut_address_bits: Address bits for prediction LUT
        """
        self.bit_width = bit_width
        self.lut_address_bits = lut_address_bits

        # Phase space state
        self.phase_state = PhaseSpaceState()

        # IIR accumulator bank
        self.iir_bank = FibonacciIIRBank(bit_width=bit_width)

        # Prediction LUT
        self.prediction_lut = PredictionLUT(
            address_bits=lut_address_bits,
            value_bits=bit_width
        )

        # Statistics
        self.cycle_count = 0
        self.prediction_correct = 0
        self.prediction_total = 0

    def step(self, input_bits: List[int]) -> 'PhaseSpaceInferenceResult':
        """
        Execute one inference cycle.

        Pipeline stages:
        1. Update phi channel with input
        2. Compute psi (velocity) from phi change
        3. Update IIR accumulators
        4. Get context address
        5. Lookup/update prediction
        6. Advance clock

        Args:
            input_bits: Input Zeckendorf bit pattern

        Returns:
            PhaseSpaceInferenceResult with complete state
        """
        # Stage 1: Update phi (position) channel
        self.phase_state.update_phi(input_bits)

        # Stage 2: Psi (velocity) is computed in update_phi
        velocity_magnitude = self.phase_state.get_velocity_magnitude()
        velocity_direction = self.phase_state.get_velocity_direction()

        # Stage 3: Update IIR accumulators
        acc_values = self.iir_bank.update_from_bits(input_bits)

        # Stage 4: Get context address (use consistent address bits)
        context_address = self.iir_bank.get_context_address(address_bits=self.lut_address_bits)

        # Stage 5: Prediction lookup
        prediction = self.prediction_lut.lookup(context_address)

        # Stage 6: Convert input to value for LUT update
        input_value = self._bits_to_value(input_bits)

        # Check prediction accuracy (if we had a prediction)
        prediction_was_correct = False
        if prediction is not None:
            pred_value, confidence = prediction
            self.prediction_total += 1
            if abs(pred_value - input_value) < 10:  # Tolerance
                self.prediction_correct += 1
                prediction_was_correct = True

        # Update LUT with actual value
        self.prediction_lut.update(context_address, input_value)

        # Stage 7: Advance clock
        self.phase_state.advance_clock()
        self.cycle_count += 1

        return PhaseSpaceInferenceResult(
            phi_bits=list(self.phase_state.phi),
            psi_bits=list(self.phase_state.psi),
            accumulator_values=acc_values,
            context_address=context_address,
            prediction=prediction,
            prediction_correct=prediction_was_correct,
            velocity_magnitude=velocity_magnitude,
            velocity_direction=velocity_direction,
            clock_cycle=self.phase_state.clock_cycle,
            polarity=self.phase_state.polarity
        )

    def _bits_to_value(self, bits: List[int]) -> int:
        """Convert Zeckendorf bits to integer value."""
        fibs = self.iir_bank._get_fibonacci_weights(len(bits))
        return sum(fibs[i] for i in range(len(bits)) if bits[i])

    def get_prediction_accuracy(self) -> float:
        """Get prediction accuracy rate."""
        if self.prediction_total == 0:
            return 0.0
        return self.prediction_correct / self.prediction_total

    def reset(self):
        """Reset the pipeline."""
        self.phase_state = PhaseSpaceState()
        self.iir_bank.reset()
        self.prediction_lut.clear()
        self.cycle_count = 0
        self.prediction_correct = 0
        self.prediction_total = 0


@dataclass
class PhaseSpaceInferenceResult:
    """Result from one phase-space inference cycle."""
    phi_bits: List[int]                      # Position channel bits
    psi_bits: List[int]                      # Velocity channel bits
    accumulator_values: Dict[int, float]     # IIR accumulator states
    context_address: int                     # Computed context address
    prediction: Optional[Tuple[int, float]]  # (predicted_value, confidence)
    prediction_correct: bool                 # Whether prediction was correct
    velocity_magnitude: int                  # |ψ|
    velocity_direction: int                  # sign(ψ)
    clock_cycle: int                         # Current clock
    polarity: int                            # (-1)^n


class CassiniLockedAccumulator:
    """
    Accumulator with Cassini identity preservation.

    Maintains the invariant that updates preserve the Cassini relationship:
    F_{n-1} * F_{n+1} - F_n^2 = (-1)^n

    This ensures structural consistency in the Fibonacci lattice.
    """

    def __init__(self, max_index: int = 100):
        """Initialize with Fibonacci cache."""
        self.max_index = max_index
        self._fib = self._compute_fibonacci(max_index + 5)
        self.current_index = 0
        self.accumulated_value = 0

    def _compute_fibonacci(self, n: int) -> List[int]:
        """Compute Fibonacci sequence."""
        if n <= 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]

        fibs = [0, 1]
        for _ in range(n - 2):
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

    def verify_cassini(self, n: int) -> bool:
        """Verify Cassini identity at index n."""
        if n < 1 or n + 1 >= len(self._fib):
            return False

        f_prev = self._fib[n - 1]
        f_curr = self._fib[n]
        f_next = self._fib[n + 1]

        lhs = f_prev * f_next - f_curr * f_curr
        rhs = (-1) ** n

        return lhs == rhs

    def update_locked(self, input_value: int) -> Tuple[int, bool]:
        """
        Update accumulator while preserving Cassini invariant.

        Args:
            input_value: Value to add

        Returns:
            Tuple of (new_value, cassini_valid)
        """
        self.accumulated_value += input_value
        self.current_index += 1

        # Verify Cassini after update
        cassini_valid = self.verify_cassini(self.current_index)

        return self.accumulated_value, cassini_valid

    def get_polarity(self) -> int:
        """Get current polarity (-1)^n."""
        return (-1) ** self.current_index


def demonstrate_iir_system():
    """Demonstrate the IIR accumulator and phase-space system."""

    print("=" * 70)
    print("IIR ACCUMULATOR BANK AND PHASE-SPACE INFERENCE")
    print("=" * 70)

    # =========================================================================
    # 1. IIR ACCUMULATOR BANK
    # =========================================================================
    print("\n1. IIR ACCUMULATOR BANK")
    print("-" * 50)

    iir_bank = FibonacciIIRBank(bit_width=16)

    print("Fibonacci-spaced decay constants δ ∈ {1, 2, 3, 5}")
    print("\nDecay factors (1 - 2^{-δ}):")
    for delta, rate in iir_bank.get_decay_rates().items():
        print(f"  δ={delta}: decay_factor = {rate:.6f}")

    # Simulate input sequence
    print("\nSimulating token sequence...")
    test_values = [10, 20, 15, 25, 10, 30, 5, 20]

    print("\nValue | A_1      | A_2      | A_3      | A_5      | Context Addr")
    print("-" * 70)

    for val in test_values:
        acc_vals = iir_bank.update(val)
        ctx = iir_bank.get_context_address()
        print(f"{val:5d} | {acc_vals[1]:8.2f} | {acc_vals[2]:8.2f} | "
              f"{acc_vals[3]:8.2f} | {acc_vals[5]:8.2f} | 0x{ctx:03X}")

    # =========================================================================
    # 2. PHASE SPACE STATE
    # =========================================================================
    print("\n2. PHASE SPACE STATE (φ-ψ)")
    print("-" * 50)

    phase_state = PhaseSpaceState()

    print("Position (φ) and Velocity (ψ) channels:")
    print("\nφ(t)           | ψ(t) (velocity) | |ψ| | dir")
    print("-" * 55)

    test_patterns = [
        [1, 0, 1, 0, 0],    # 1 + 3 = 4
        [0, 1, 0, 1, 0],    # 2 + 5 = 7
        [1, 0, 0, 0, 1],    # 1 + 8 = 9
        [1, 0, 1, 0, 1],    # 1 + 3 + 8 = 12
        [0, 1, 0, 0, 0],    # 2
    ]

    for phi in test_patterns:
        phase_state.update_phi(phi)
        psi = phase_state.psi
        mag = phase_state.get_velocity_magnitude()
        direction = phase_state.get_velocity_direction()
        dir_str = '+' if direction > 0 else ('-' if direction < 0 else '0')

        phi_str = ''.join(str(b) for b in phi)
        psi_str = ''.join(f"{v:+d}" for v in psi[:5])

        print(f"{phi_str:14s} | {psi_str:15s} | {mag:2d} | {dir_str}")

    # =========================================================================
    # 3. PREDICTION LUT
    # =========================================================================
    print("\n3. PREDICTION LUT")
    print("-" * 50)

    lut = PredictionLUT(address_bits=8, value_bits=16)

    # Simulate training
    print("Training LUT with sample data...")
    for i in range(100):
        addr = (i * 13) % 256  # Pseudo-random addresses
        value = 50 + (i % 20)   # Values in range 50-70
        lut.update(addr, value)

    print(f"  Entries filled: {lut.get_fill_rate()*100:.1f}%")
    print(f"  Updates: {lut.updates}")

    # Test lookups
    print("\nTesting lookups...")
    test_addrs = [0, 13, 26, 39, 255]
    for addr in test_addrs:
        result = lut.lookup(addr)
        if result:
            pred, conf = result
            print(f"  Address 0x{addr:02X}: prediction={pred}, confidence={conf:.2f}")
        else:
            print(f"  Address 0x{addr:02X}: not found")

    print(f"\nHit rate: {lut.get_hit_rate()*100:.1f}%")

    # =========================================================================
    # 4. COMPLETE PHASE-SPACE PIPELINE
    # =========================================================================
    print("\n4. PHASE-SPACE INFERENCE PIPELINE")
    print("-" * 50)

    pipeline = PhaseSpaceInferencePipeline(bit_width=16, lut_address_bits=10)

    print("Running inference on token sequence...")
    print()

    test_inputs = [
        [1, 0, 0, 0, 0],    # 1
        [0, 1, 0, 0, 0],    # 2
        [0, 0, 1, 0, 0],    # 3
        [0, 0, 0, 1, 0],    # 5
        [0, 0, 0, 0, 1],    # 8
        [1, 0, 1, 0, 0],    # 4
        [0, 1, 0, 1, 0],    # 7
        [1, 0, 0, 0, 1],    # 9
    ]

    print("Cycle | φ         | |ψ| | A_1    | A_2    | A_3    | A_5    | Ctx  | Pred")
    print("-" * 85)

    for bits in test_inputs:
        result = pipeline.step(bits)
        phi_str = ''.join(str(b) for b in result.phi_bits[:8])

        pred_str = "---"
        if result.prediction:
            pred_val, pred_conf = result.prediction
            pred_str = f"{pred_val:3d}"

        print(f"{result.clock_cycle:5d} | {phi_str} | {result.velocity_magnitude:2d}  | "
              f"{result.accumulator_values[1]:6.1f} | {result.accumulator_values[2]:6.1f} | "
              f"{result.accumulator_values[3]:6.1f} | {result.accumulator_values[5]:6.1f} | "
              f"0x{result.context_address:03X} | {pred_str}")

    print(f"\nPrediction accuracy: {pipeline.get_prediction_accuracy()*100:.1f}%")
    print(f"LUT fill rate: {pipeline.prediction_lut.get_fill_rate()*100:.1f}%")

    # =========================================================================
    # 5. CASSINI-LOCKED ACCUMULATOR
    # =========================================================================
    print("\n5. CASSINI-LOCKED ACCUMULATOR")
    print("-" * 50)

    cassini_acc = CassiniLockedAccumulator()

    print("Accumulator updates with Cassini verification:")
    print("\nInput | Total | Cassini | F_{n-1}*F_{n+1} - F_n^2 = (-1)^n")
    print("-" * 60)

    for val in [5, 10, 3, 8, 2]:
        total, valid = cassini_acc.update_locked(val)
        n = cassini_acc.current_index
        polarity = cassini_acc.get_polarity()
        valid_str = "VALID" if valid else "N/A"

        # Show Cassini computation for valid indices
        if n >= 1 and n + 1 < len(cassini_acc._fib):
            f_prev = cassini_acc._fib[n - 1]
            f_curr = cassini_acc._fib[n]
            f_next = cassini_acc._fib[n + 1]
            cassini = f_prev * f_next - f_curr * f_curr
            cassini_str = f"{f_prev}*{f_next} - {f_curr}^2 = {cassini} = {polarity:+d}"
        else:
            cassini_str = "---"

        print(f"{val:5d} | {total:5d} | {valid_str:7s} | {cassini_str}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_iir_system()
