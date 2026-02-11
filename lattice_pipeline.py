"""
Zero-RAM Lattice Inference Pipeline

This module provides the complete end-to-end pipeline for:
1. Fetching text content (e.g., SICP)
2. Tokenizing and encoding with Zeckendorf representation
3. Running inference through bit collapse mechanics
4. Validating all invariants at each step

The pipeline demonstrates the core concept: memory recall and computation
are unified into a single act through Fibonacci-indexed bit patterns.
"""

from typing import List, Dict, Generator, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import json
import os

# Import our modules
from invariants import FibonacciRTL, create_generalized_rtl
from rtl_operations import FibonacciSeq, RTLDecomposer
from validation_rules import LatticeValidator, ValidationReport
from lattice_encoder import VocabularyEncoder, TokenEncoder, EncodedToken, LucasPrefilter
from bit_collapse import BitCollapseEngine, LatticeInferencePipeline, InferenceState, CassiniVerifier
from iir_accumulators import (
    FibonacciIIRBank,
    PhaseSpaceInferencePipeline,
    PhaseSpaceInferenceResult,
    PredictionLUT,
    FIBONACCI_DELTAS
)


@dataclass
class PipelineConfig:
    """Configuration for the lattice inference pipeline."""
    bit_width: int = 16
    validate_every_step: bool = True
    cache_dir: str = ".lattice_cache"
    max_vocab_size: int = 10000
    min_token_freq: int = 2


@dataclass
class InferenceResult:
    """Result of a single inference step."""
    input_text: str
    input_bits: List[int]
    output_bits: List[int]
    output_value: int
    clock_cycle: int
    polarity: int
    cascade_depth: int
    validation_passed: bool
    validation_report: Optional[ValidationReport] = None
    # Phase-space extensions
    psi_bits: Optional[List[int]] = None  # Velocity channel
    accumulator_values: Optional[Dict[int, float]] = None  # IIR states
    context_address: Optional[int] = None  # LUT address
    prediction: Optional[Tuple[int, float]] = None  # (value, confidence)
    velocity_magnitude: int = 0


@dataclass
class PipelineStats:
    """Statistics from running the pipeline."""
    total_tokens: int = 0
    total_collapses: int = 0
    max_cascade_depth: int = 0
    avg_cascade_depth: float = 0.0
    validation_failures: int = 0
    unique_patterns: int = 0
    # IIR accumulator stats
    max_velocity_magnitude: int = 0
    avg_velocity_magnitude: float = 0.0
    lut_hit_rate: float = 0.0
    prediction_accuracy: float = 0.0
    total_context_addresses: int = 0


class ZeroRAMLattice:
    """
    Complete Zero-RAM Lattice Inference System.

    This class orchestrates all components:
    - Vocabulary encoding
    - Zeckendorf bit pattern generation
    - Bit collapse inference
    - Invariant validation
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the lattice system."""
        self.config = config or PipelineConfig()
        self.vocab_encoder: Optional[VocabularyEncoder] = None
        self.token_encoder = TokenEncoder(bit_width=self.config.bit_width)
        self.lucas_filter = LucasPrefilter()
        self.collapse_engine = BitCollapseEngine()
        self.inference_pipeline = LatticeInferencePipeline()
        self.validator = LatticeValidator()
        self.cassini = CassiniVerifier()

        # IIR Accumulator Bank and Phase-Space Pipeline
        self.iir_bank = FibonacciIIRBank(bit_width=self.config.bit_width)
        self.phase_space_pipeline = PhaseSpaceInferencePipeline(
            bit_width=self.config.bit_width,
            lut_address_bits=12
        )

        self._stats = PipelineStats()
        self._pattern_cache: Dict[str, List[int]] = {}
        self._velocity_history: List[int] = []

        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def build_vocabulary(self, texts: List[str]) -> int:
        """
        Build vocabulary from corpus of texts.

        Args:
            texts: List of text strings

        Returns:
            Vocabulary size
        """
        self.vocab_encoder = VocabularyEncoder(
            bit_width=self.config.bit_width
        )
        self.vocab_encoder.build_vocabulary(
            " ".join(texts),
            min_frequency=self.config.min_token_freq
        )
        return self.vocab_encoder.vocabulary_size

    def encode_text(self, text: str) -> List[EncodedToken]:
        """
        Encode text into Zeckendorf bit patterns.

        Args:
            text: Input text string

        Returns:
            List of EncodedToken objects
        """
        if self.vocab_encoder is None:
            raise ValueError("Must build vocabulary first")

        return self.vocab_encoder.encode_text(text)

    def run_inference(
        self,
        text: str,
        return_trace: bool = False
    ) -> Generator[InferenceResult, None, None]:
        """
        Run inference on text, yielding results for each token.

        This is the core inference loop:
        1. Encode token to Zeckendorf bits
        2. Perturb state with input
        3. Apply Lucas pre-filter
        4. Execute bit collapse cascade
        5. Capture to registers
        6. Validate invariants
        7. Yield result

        Args:
            text: Input text
            return_trace: Whether to include full trace in results

        Yields:
            InferenceResult for each token
        """
        if self.vocab_encoder is None:
            raise ValueError("Must build vocabulary first")

        # Reset pipeline state
        self.inference_pipeline = LatticeInferencePipeline()

        encoded_tokens = self.vocab_encoder.encode_text(text)
        self._stats.total_tokens += len(encoded_tokens)

        for pos, enc_token in enumerate(encoded_tokens):
            input_bits = enc_token.zeckendorf_bits

            # Run one inference cycle (standard pipeline)
            state = self.inference_pipeline.step(input_bits)

            # Run phase-space inference (IIR + velocity)
            ps_result = self.phase_space_pipeline.step(input_bits)

            # Update stats
            self._stats.total_collapses += state.cascade_depth
            self._stats.max_cascade_depth = max(
                self._stats.max_cascade_depth,
                state.cascade_depth
            )

            # Update velocity stats
            self._velocity_history.append(ps_result.velocity_magnitude)
            self._stats.max_velocity_magnitude = max(
                self._stats.max_velocity_magnitude,
                ps_result.velocity_magnitude
            )
            self._stats.total_context_addresses += 1

            # Cache unique patterns
            pattern_key = str(state.current_bits)
            if pattern_key not in self._pattern_cache:
                self._pattern_cache[pattern_key] = state.current_bits.copy()
                self._stats.unique_patterns += 1

            # Validate if enabled
            validation_report = None
            validation_passed = True
            if self.config.validate_every_step:
                validation_report = self.validator.validate_pipeline_state(
                    bits=state.current_bits,
                    clock_n=state.clock_cycle,
                    polarity=state.polarity
                )
                validation_passed = validation_report.passed
                if not validation_passed:
                    self._stats.validation_failures += 1

            # Decode output value
            output_value = self._bits_to_value(state.current_bits)

            yield InferenceResult(
                input_text=enc_token.text,
                input_bits=input_bits,
                output_bits=state.current_bits,
                output_value=output_value,
                clock_cycle=state.clock_cycle,
                polarity=state.polarity,
                cascade_depth=state.cascade_depth,
                validation_passed=validation_passed,
                validation_report=validation_report,
                # Phase-space extensions
                psi_bits=ps_result.psi_bits,
                accumulator_values=ps_result.accumulator_values,
                context_address=ps_result.context_address,
                prediction=ps_result.prediction,
                velocity_magnitude=ps_result.velocity_magnitude
            )

    def _bits_to_value(self, bits: List[int]) -> int:
        """Convert bit pattern to integer value."""
        return self.token_encoder.decode(bits)

    def verify_cassini_range(self, max_n: int = 50) -> List[Tuple[int, bool]]:
        """
        Verify Cassini's identity for range of n.

        Args:
            max_n: Maximum n to check

        Returns:
            List of (n, is_valid) tuples
        """
        results = []
        for n in range(1, max_n + 1):
            is_valid = self.cassini.verify(n)
            results.append((n, is_valid))
        return results

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        if self._stats.total_collapses > 0:
            self._stats.avg_cascade_depth = (
                self._stats.total_collapses / self._stats.total_tokens
            )
        if self._velocity_history:
            self._stats.avg_velocity_magnitude = (
                sum(self._velocity_history) / len(self._velocity_history)
            )
        # Get LUT stats from phase-space pipeline
        self._stats.lut_hit_rate = self.phase_space_pipeline.prediction_lut.get_hit_rate()
        self._stats.prediction_accuracy = self.phase_space_pipeline.get_prediction_accuracy()
        return self._stats

    def reset_stats(self):
        """Reset statistics."""
        self._stats = PipelineStats()
        self._pattern_cache.clear()
        self._velocity_history.clear()
        self.phase_space_pipeline.reset()
        self.iir_bank.reset()

    def save_vocabulary(self, path: str):
        """Save vocabulary to file."""
        if self.vocab_encoder is None:
            raise ValueError("No vocabulary to save")

        data = {
            "token_to_id": self.vocab_encoder.token_to_id,
            "config": {
                "bit_width": self.config.bit_width,
                "max_vocab_size": self.config.max_vocab_size
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_vocabulary(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab_encoder = VocabularyEncoder(
            max_vocab_size=data["config"]["max_vocab_size"],
            bit_width=data["config"]["bit_width"]
        )
        self.vocab_encoder.token_to_id = data["token_to_id"]
        self.vocab_encoder.id_to_token = {
            v: k for k, v in data["token_to_id"].items()
        }


class TextProcessor:
    """Utilities for processing text before encoding."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    @staticmethod
    def tokenize_simple(text: str) -> List[str]:
        """Simple whitespace tokenization with punctuation separation."""
        import re
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:\'"()\[\]{}])', r' \1 ', text)
        return text.lower().split()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> Generator[str, None, None]:
        """Split text into chunks for processing."""
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])


def demonstrate_pipeline():
    """Demonstrate the full lattice inference pipeline."""
    print("=" * 70)
    print("ZERO-RAM LATTICE INFERENCE PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Sample texts for building vocabulary
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Structure and interpretation of computer programs.",
        "Programs must be written for people to read.",
        "The Fibonacci sequence appears throughout mathematics.",
        "Every positive integer has a unique Zeckendorf representation.",
        "Hardware inference requires no random access memory.",
    ]

    # Initialize pipeline
    config = PipelineConfig(
        bit_width=16,
        validate_every_step=True,
        min_token_freq=1
    )
    lattice = ZeroRAMLattice(config)

    # Build vocabulary
    print("\n1. BUILDING VOCABULARY")
    print("-" * 50)
    vocab_size = lattice.build_vocabulary(sample_texts)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Bit width: {config.bit_width}")
    print(f"   Max representable ID: {lattice.token_encoder.max_value}")

    # Run inference on test text
    print("\n2. RUNNING INFERENCE WITH PHASE-SPACE")
    print("-" * 50)
    test_text = "The Fibonacci sequence"

    print(f"   Input: \"{test_text}\"")
    print()
    print("   Token    | φ (position)  | |ψ| | A_1    | A_2    | Context | Pred")
    print("   " + "-" * 75)

    for result in lattice.run_inference(test_text):
        phi_str = ''.join(map(str, result.output_bits[:10]))

        # Get accumulator values
        a1 = result.accumulator_values.get(1, 0) if result.accumulator_values else 0
        a2 = result.accumulator_values.get(2, 0) if result.accumulator_values else 0

        ctx_str = f"0x{result.context_address:03X}" if result.context_address else "---"

        pred_str = "---"
        if result.prediction:
            pred_val, pred_conf = result.prediction
            pred_str = f"{pred_val:3d}"

        print(f"   {result.input_text:10s} | {phi_str}... | {result.velocity_magnitude:2d}  | "
              f"{a1:6.1f} | {a2:6.1f} | {ctx_str:7s} | {pred_str}")

    # Show stats
    print("\n3. PIPELINE STATISTICS")
    print("-" * 50)
    stats = lattice.get_stats()
    print(f"   Total tokens processed: {stats.total_tokens}")
    print(f"   Total collapse operations: {stats.total_collapses}")
    print(f"   Maximum cascade depth: {stats.max_cascade_depth}")
    print(f"   Average cascade depth: {stats.avg_cascade_depth:.2f}")
    print(f"   Unique bit patterns: {stats.unique_patterns}")
    print(f"   Validation failures: {stats.validation_failures}")
    print()
    print("   Phase-Space Statistics:")
    print(f"   Max velocity magnitude: {stats.max_velocity_magnitude}")
    print(f"   Avg velocity magnitude: {stats.avg_velocity_magnitude:.2f}")
    print(f"   LUT hit rate: {stats.lut_hit_rate*100:.1f}%")
    print(f"   Prediction accuracy: {stats.prediction_accuracy*100:.1f}%")

    # Verify Cassini identity
    print("\n4. CASSINI IDENTITY VERIFICATION")
    print("-" * 50)
    cassini_results = lattice.verify_cassini_range(15)
    all_valid = all(valid for _, valid in cassini_results)
    print(f"   Verified for n=1 to 15: {'ALL VALID' if all_valid else 'FAILURES DETECTED'}")

    # Demonstrate bit collapse
    print("\n5. BIT COLLAPSE DEMONSTRATION")
    print("-" * 50)

    collapse_engine = BitCollapseEngine()

    # Example with adjacent 1s
    test_patterns = [
        [1, 1, 0, 0, 0],      # 1 + 2 = 3 -> should collapse
        [1, 0, 1, 1, 0],      # 1 + 3 + 5 = 9, adjacent at 2,3
        [1, 1, 1, 1, 0],      # Multiple adjacencies
    ]

    for bits in test_patterns:
        # Use collapse_with_trace to get both result and depth
        from bit_collapse import CollapseCascadeResult
        trace_result = collapse_engine.collapse_with_trace(bits)
        result = trace_result.final_bits
        depth = trace_result.cascade_depth
        zeck_fibs = lattice.token_encoder._zeck_fibs
        before_val = sum(zeck_fibs[i] for i in range(len(bits)) if i < len(zeck_fibs) and bits[i])
        after_val = sum(zeck_fibs[i] for i in range(len(result)) if i < len(zeck_fibs) and result[i])

        print(f"   {bits} -> {result}")
        print(f"     Value: {before_val} -> {after_val}, Cascade depth: {depth}")

    print("\n" + "=" * 70)
    print("Pipeline demonstration complete!")


if __name__ == "__main__":
    demonstrate_pipeline()
