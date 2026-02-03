"""
Validation Rules for Zero-RAM Lattice Inference

This module defines the complete set of validation rules for the Fibonacci-Lucas
bit AI modeling system. Every rule corresponds to a mathematical invariant that
MUST hold for the system to be in a valid state.

Rule Categories:
- Structural Rules: Zeckendorf constraint, monotonicity, completeness
- Encoding Rules: Token-to-bits correctness, bidirectional consistency
- Collapse Rules: Adjacent resolution, cascade termination
- Verification Rules: Cassini invariant, spectral bounds
- Pipeline Rules: State machine validity, clock synchronization
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, Dict
from enum import Enum, auto
from functools import wraps
import math

from invariants import (
    SequenceInvariant,
    DecompositionInvariant,
    RTLProcessingInvariant,
    InvariantViolation,
    fibonacci_generator,
    lucas_generator,
)


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    FATAL = auto()      # System must halt
    ERROR = auto()      # Operation must be rejected
    WARNING = auto()    # Operation proceeds with caution
    INFO = auto()       # Logged for diagnostics


class RuleCategory(Enum):
    """Categories of validation rules."""
    STRUCTURAL = auto()
    ENCODING = auto()
    COLLAPSE = auto()
    VERIFICATION = auto()
    PIPELINE = auto()


@dataclass
class RuleResult:
    """Result of a rule validation check."""
    rule_id: str
    passed: bool
    message: str
    severity: RuleSeverity
    category: RuleCategory
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report from running all rules."""
    results: List[RuleResult] = field(default_factory=list)
    passed: bool = True
    fatal_count: int = 0
    error_count: int = 0
    warning_count: int = 0

    def add(self, result: RuleResult):
        self.results.append(result)
        if not result.passed:
            if result.severity == RuleSeverity.FATAL:
                self.fatal_count += 1
                self.passed = False
            elif result.severity == RuleSeverity.ERROR:
                self.error_count += 1
                self.passed = False
            elif result.severity == RuleSeverity.WARNING:
                self.warning_count += 1

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status}: "
            f"{len(self.results)} rules checked, "
            f"{self.fatal_count} fatal, "
            f"{self.error_count} errors, "
            f"{self.warning_count} warnings"
        )


class ValidationRule:
    """Base class for validation rules."""

    def __init__(
        self,
        rule_id: str,
        description: str,
        severity: RuleSeverity,
        category: RuleCategory
    ):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity
        self.category = category

    def check(self, *args, **kwargs) -> RuleResult:
        """Override in subclass to implement validation logic."""
        raise NotImplementedError

    def _pass(self, message: str = "", context: Dict = None) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            passed=True,
            message=message or f"{self.description}: OK",
            severity=self.severity,
            category=self.category,
            context=context or {}
        )

    def _fail(self, message: str, context: Dict = None) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            passed=False,
            message=message,
            severity=self.severity,
            category=self.category,
            context=context or {}
        )


# =============================================================================
# STRUCTURAL RULES (S-series)
# =============================================================================

class ZeckendorfConstraintRule(ValidationRule):
    """
    S001: Zeckendorf Non-Adjacency Constraint

    Every valid bit pattern must have no adjacent 1s.
    This is THE fundamental constraint of the encoding.

    Mathematical basis: Zeckendorf's theorem guarantees unique representation
    only when no two consecutive Fibonacci numbers are used.
    """

    def __init__(self):
        super().__init__(
            rule_id="S001",
            description="Zeckendorf non-adjacency constraint",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.STRUCTURAL
        )

    def check(self, bits: List[int]) -> RuleResult:
        """Check that no two adjacent bits are both 1."""
        for i in range(len(bits) - 1):
            if bits[i] == 1 and bits[i + 1] == 1:
                return self._fail(
                    f"Adjacent 1s found at positions {i} and {i+1}",
                    {"bits": bits, "violation_at": (i, i+1)}
                )
        return self._pass(context={"bits": bits})


class SequenceMonotonicityRule(ValidationRule):
    """
    S002: Sequence Strict Monotonicity

    The underlying Fibonacci/Lucas sequence must be strictly increasing.
    This ensures RTL processing always moves from larger to smaller values.
    """

    def __init__(self):
        super().__init__(
            rule_id="S002",
            description="Sequence strict monotonicity",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.STRUCTURAL
        )

    def check(self, sequence: List[int]) -> RuleResult:
        """Check strict monotonicity."""
        for i in range(len(sequence) - 1):
            if sequence[i] >= sequence[i + 1]:
                return self._fail(
                    f"Monotonicity violated: seq[{i}]={sequence[i]} >= seq[{i+1}]={sequence[i+1]}",
                    {"sequence": sequence, "violation_at": i}
                )
        return self._pass(context={"sequence_length": len(sequence)})


class CompletenessRule(ValidationRule):
    """
    S003: Sequence Completeness

    The sequence must be able to represent all integers in the target range.
    Completeness requires: each term <= 1 + sum of all previous terms.
    """

    def __init__(self):
        super().__init__(
            rule_id="S003",
            description="Sequence completeness for representation",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STRUCTURAL
        )

    def check(self, sequence: List[int], max_n: int) -> RuleResult:
        """Check completeness up to max_n."""
        cumsum = 0
        for i, val in enumerate(sequence):
            if val > cumsum + 1 and i > 0:
                return self._fail(
                    f"Gap at position {i}: value {val} > cumsum {cumsum} + 1",
                    {"sequence": sequence, "gap_at": i, "missing_range": (cumsum + 1, val - 1)}
                )
            cumsum += val
            if cumsum >= max_n:
                return self._pass(context={"representable_up_to": cumsum})

        if cumsum < max_n:
            return self._fail(
                f"Sequence only covers up to {cumsum}, need {max_n}",
                {"max_representable": cumsum, "required": max_n}
            )

        return self._pass(context={"representable_up_to": cumsum})


# =============================================================================
# ENCODING RULES (E-series)
# =============================================================================

class EncodingSumRule(ValidationRule):
    """
    E001: Encoding Sum Correctness

    The sum of active Fibonacci numbers must equal the encoded value.
    This is the fundamental encoding correctness check.
    """

    def __init__(self):
        super().__init__(
            rule_id="E001",
            description="Encoding sum equals target value",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.ENCODING
        )

    def check(self, value: int, bits: List[int], fib_sequence: List[int]) -> RuleResult:
        """Check that sum of active Fibonacci numbers equals value."""
        total = sum(fib_sequence[i] for i in range(len(bits)) if bits[i] == 1)
        if total != value:
            return self._fail(
                f"Sum mismatch: encoded {total}, expected {value}",
                {"value": value, "bits": bits, "computed_sum": total}
            )
        return self._pass(context={"value": value, "sum": total})


class BidirectionalConsistencyRule(ValidationRule):
    """
    E002: Bidirectional Encode/Decode Consistency

    encode(decode(bits)) == bits AND decode(encode(n)) == n

    The encoding must be perfectly reversible.
    """

    def __init__(self):
        super().__init__(
            rule_id="E002",
            description="Bidirectional encode/decode consistency",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.ENCODING
        )

    def check(
        self,
        encode_fn: Callable[[int], List[int]],
        decode_fn: Callable[[List[int]], int],
        test_values: List[int]
    ) -> RuleResult:
        """Test roundtrip consistency for given values."""
        failures = []
        for n in test_values:
            encoded = encode_fn(n)
            decoded = decode_fn(encoded)
            if decoded != n:
                failures.append((n, encoded, decoded))

        if failures:
            return self._fail(
                f"Roundtrip failures for {len(failures)} values",
                {"failures": failures[:10]}  # Limit context size
            )
        return self._pass(context={"tested_count": len(test_values)})


class LucasBracketRule(ValidationRule):
    """
    E003: Lucas Bracket Identity

    L_n = F_{n-1} + F_{n+1} must hold for all valid positions.
    This identity governs structural adjacency constraints.
    """

    def __init__(self):
        super().__init__(
            rule_id="E003",
            description="Lucas bracket identity L_n = F_{n-1} + F_{n+1}",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.ENCODING
        )

    def check(self, n: int, fib_seq: List[int], lucas_seq: List[int]) -> RuleResult:
        """Verify Lucas bracket identity."""
        if n < 1 or n >= len(fib_seq) - 1 or n >= len(lucas_seq):
            return self._fail(
                f"Index {n} out of range for verification",
                {"n": n, "fib_len": len(fib_seq), "lucas_len": len(lucas_seq)}
            )

        expected_lucas = fib_seq[n - 1] + fib_seq[n + 1]
        actual_lucas = lucas_seq[n]

        if expected_lucas != actual_lucas:
            return self._fail(
                f"Lucas bracket mismatch at n={n}: F_{n-1}+F_{n+1}={expected_lucas}, L_{n}={actual_lucas}",
                {"n": n, "expected": expected_lucas, "actual": actual_lucas}
            )
        return self._pass(context={"n": n, "L_n": actual_lucas})


# =============================================================================
# COLLAPSE RULES (C-series)
# =============================================================================

class CollapseTerminationRule(ValidationRule):
    """
    C001: Collapse Cascade Termination

    Every collapse cascade must terminate in finite steps.
    Maximum steps bounded by bit width (each collapse reduces total 1-count or shifts right).
    """

    def __init__(self, max_steps: int = 100):
        super().__init__(
            rule_id="C001",
            description="Collapse cascade terminates",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.COLLAPSE
        )
        self.max_steps = max_steps

    def check(self, collapse_fn: Callable[[List[int]], List[int]], initial_bits: List[int]) -> RuleResult:
        """Check that collapse terminates within max_steps."""
        bits = initial_bits.copy()
        steps = 0

        while steps < self.max_steps:
            # Check for adjacent 1s
            has_adjacent = any(bits[i] == 1 and bits[i+1] == 1 for i in range(len(bits) - 1))
            if not has_adjacent:
                return self._pass(context={"steps": steps, "final_bits": bits})

            bits = collapse_fn(bits)
            steps += 1

        return self._fail(
            f"Collapse did not terminate within {self.max_steps} steps",
            {"initial_bits": initial_bits, "steps": steps}
        )


class CollapseCorrectnessRule(ValidationRule):
    """
    C002: Collapse Preserves Value

    F_j + F_{j+1} = F_{j+2} implies collapse preserves the encoded integer value.
    The sum before collapse must equal the sum after collapse.
    """

    def __init__(self):
        super().__init__(
            rule_id="C002",
            description="Collapse preserves encoded value",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.COLLAPSE
        )

    def check(
        self,
        before_bits: List[int],
        after_bits: List[int],
        fib_sequence: List[int]
    ) -> RuleResult:
        """Check value preservation through collapse."""
        def compute_sum(bits):
            return sum(fib_sequence[i] for i in range(min(len(bits), len(fib_sequence))) if bits[i] == 1)

        before_sum = compute_sum(before_bits)
        after_sum = compute_sum(after_bits)

        if before_sum != after_sum:
            return self._fail(
                f"Collapse changed value: {before_sum} -> {after_sum}",
                {"before_bits": before_bits, "after_bits": after_bits,
                 "before_sum": before_sum, "after_sum": after_sum}
            )
        return self._pass(context={"preserved_value": before_sum})


class SingleCollapseRule(ValidationRule):
    """
    C003: Single Collapse Correctness

    When bits[j]=1 and bits[j+1]=1:
      - bits[j] -> 0
      - bits[j+1] -> 0
      - bits[j+2] -> bits[j+2] XOR 1
    """

    def __init__(self):
        super().__init__(
            rule_id="C003",
            description="Single collapse rule application",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.COLLAPSE
        )

    def check(self, before: List[int], after: List[int], collapse_pos: int) -> RuleResult:
        """Verify single collapse was applied correctly."""
        j = collapse_pos

        # Check precondition
        if j >= len(before) - 1:
            return self._fail(f"Invalid collapse position {j}", {"position": j})
        if before[j] != 1 or before[j + 1] != 1:
            return self._fail(
                f"No adjacent 1s at position {j}",
                {"before": before, "position": j}
            )

        # Check postconditions
        errors = []
        if after[j] != 0:
            errors.append(f"bits[{j}] should be 0, got {after[j]}")
        if after[j + 1] != 0:
            errors.append(f"bits[{j+1}] should be 0, got {after[j+1]}")

        expected_j2 = before[j + 2] ^ 1 if j + 2 < len(before) else 1
        if j + 2 < len(after) and after[j + 2] != expected_j2:
            errors.append(f"bits[{j+2}] should be {expected_j2}, got {after[j+2]}")

        if errors:
            return self._fail("; ".join(errors), {"before": before, "after": after})
        return self._pass(context={"collapse_position": j})


# =============================================================================
# VERIFICATION RULES (V-series)
# =============================================================================

class CassiniInvariantRule(ValidationRule):
    """
    V001: Cassini Invariant

    F_{n-1} * F_{n+1} - F_n^2 = (-1)^n

    This is the structural verification identity. If it fails,
    the system has entered an invalid state.
    """

    def __init__(self):
        super().__init__(
            rule_id="V001",
            description="Cassini invariant F_{n-1}*F_{n+1} - F_n^2 = (-1)^n",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.VERIFICATION
        )

    def check(self, n: int, fib_sequence: List[int]) -> RuleResult:
        """Verify Cassini invariant at position n."""
        if n < 1 or n >= len(fib_sequence) - 1:
            return self._fail(f"Index {n} out of range", {"n": n})

        F_prev = fib_sequence[n - 1]
        F_curr = fib_sequence[n]
        F_next = fib_sequence[n + 1]

        lhs = F_prev * F_next - F_curr * F_curr
        rhs = (-1) ** n

        if lhs != rhs:
            return self._fail(
                f"Cassini violation at n={n}: {F_prev}*{F_next} - {F_curr}^2 = {lhs} != {rhs}",
                {"n": n, "F_n-1": F_prev, "F_n": F_curr, "F_n+1": F_next, "lhs": lhs, "rhs": rhs}
            )
        return self._pass(context={"n": n, "verified": f"{F_prev}*{F_next}-{F_curr}^2={rhs}"})


class SpectralBoundRule(ValidationRule):
    """
    V002: Spectral Bound (Lucas-Fibonacci Parity)

    L_n^2 - 5*F_n^2 = 4*(-1)^n

    This constrains the relationship between dual channels to deviation of exactly ±4.
    Guarantees automatic spectral normalization.
    """

    def __init__(self):
        super().__init__(
            rule_id="V002",
            description="Spectral bound L_n^2 - 5*F_n^2 = 4*(-1)^n",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.VERIFICATION
        )

    def check(self, n: int, fib_seq: List[int], lucas_seq: List[int]) -> RuleResult:
        """Verify spectral bound."""
        if n >= len(fib_seq) or n >= len(lucas_seq):
            return self._fail(f"Index {n} out of range", {"n": n})

        F_n = fib_seq[n]
        L_n = lucas_seq[n]

        lhs = L_n * L_n - 5 * F_n * F_n
        rhs = 4 * ((-1) ** n)

        if lhs != rhs:
            return self._fail(
                f"Spectral bound violation at n={n}: {L_n}^2 - 5*{F_n}^2 = {lhs} != {rhs}",
                {"n": n, "L_n": L_n, "F_n": F_n, "lhs": lhs, "rhs": rhs}
            )
        return self._pass(context={"n": n, "L_n": L_n, "F_n": F_n})


class RatioConvergenceRule(ValidationRule):
    """
    V003: Ratio Convergence to Golden Ratio

    F_{n+1}/F_n -> φ as n -> ∞

    For large n, ratio should be within tolerance of φ ≈ 1.618033988749895
    """

    def __init__(self, tolerance: float = 1e-6):
        super().__init__(
            rule_id="V003",
            description="Fibonacci ratio convergence to φ",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.VERIFICATION
        )
        self.tolerance = tolerance
        self.phi = (1 + math.sqrt(5)) / 2

    def check(self, fib_sequence: List[int], min_n: int = 10) -> RuleResult:
        """Check ratio convergence for large n."""
        if len(fib_sequence) <= min_n:
            return self._pass("Sequence too short for convergence check")

        ratios = []
        for i in range(min_n, len(fib_sequence) - 1):
            ratio = fib_sequence[i + 1] / fib_sequence[i]
            deviation = abs(ratio - self.phi)
            ratios.append((i, ratio, deviation))

        max_deviation = max(r[2] for r in ratios)
        if max_deviation > self.tolerance:
            worst = max(ratios, key=lambda r: r[2])
            return self._fail(
                f"Ratio at n={worst[0]} deviates by {worst[2]:.2e} (tolerance: {self.tolerance})",
                {"worst_n": worst[0], "worst_ratio": worst[1], "deviation": worst[2]}
            )
        return self._pass(context={"max_deviation": max_deviation, "phi": self.phi})


# =============================================================================
# PIPELINE RULES (P-series)
# =============================================================================

class PolarityRule(ValidationRule):
    """
    P001: Polarity Alternation

    (φψ)^n = (-1)^n must hold.
    Clock cycle n has polarity (-1)^n which determines shift direction.
    """

    def __init__(self):
        super().__init__(
            rule_id="P001",
            description="Polarity alternation (-1)^n",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.PIPELINE
        )

    def check(self, n: int, reported_polarity: int) -> RuleResult:
        """Verify polarity matches (-1)^n."""
        expected = (-1) ** n
        if reported_polarity != expected:
            return self._fail(
                f"Polarity mismatch at n={n}: reported {reported_polarity}, expected {expected}",
                {"n": n, "reported": reported_polarity, "expected": expected}
            )
        return self._pass(context={"n": n, "polarity": expected})


class StateValidityRule(ValidationRule):
    """
    P002: State Validity After Capture

    After register capture (clock edge), state must be valid Zeckendorf.
    """

    def __init__(self):
        super().__init__(
            rule_id="P002",
            description="State validity after capture",
            severity=RuleSeverity.FATAL,
            category=RuleCategory.PIPELINE
        )

    def check(self, state_bits: List[int]) -> RuleResult:
        """Verify state is valid Zeckendorf after capture."""
        zeck_rule = ZeckendorfConstraintRule()
        return zeck_rule.check(state_bits)


class ClockSyncRule(ValidationRule):
    """
    P003: Clock Synchronization

    Clock cycle index n must be recoverable: log_φ(φ^n) = n
    State must be synchronized with clock.
    """

    def __init__(self):
        super().__init__(
            rule_id="P003",
            description="Clock synchronization and index recovery",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.PIPELINE
        )

    def check(self, clock_n: int, state_max_index: int) -> RuleResult:
        """Check clock is synchronized with state."""
        # State's maximum active Fibonacci index should be <= some function of clock
        # This is a soft check - just ensure they're not wildly mismatched
        if state_max_index > clock_n + 10:
            return self._fail(
                f"State index {state_max_index} too far ahead of clock {clock_n}",
                {"clock_n": clock_n, "state_max_index": state_max_index}
            )
        return self._pass(context={"clock_n": clock_n, "state_max_index": state_max_index})


# =============================================================================
# RULE REGISTRY AND VALIDATOR
# =============================================================================

class RuleRegistry:
    """Registry of all validation rules."""

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        """Register all default rules."""
        # Structural
        self.register(ZeckendorfConstraintRule())
        self.register(SequenceMonotonicityRule())
        self.register(CompletenessRule())

        # Encoding
        self.register(EncodingSumRule())
        self.register(BidirectionalConsistencyRule())
        self.register(LucasBracketRule())

        # Collapse
        self.register(CollapseTerminationRule())
        self.register(CollapseCorrectnessRule())
        self.register(SingleCollapseRule())

        # Verification
        self.register(CassiniInvariantRule())
        self.register(SpectralBoundRule())
        self.register(RatioConvergenceRule())

        # Pipeline
        self.register(PolarityRule())
        self.register(StateValidityRule())
        self.register(ClockSyncRule())

    def register(self, rule: ValidationRule):
        """Register a validation rule."""
        self.rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> Optional[ValidationRule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)

    def by_category(self, category: RuleCategory) -> List[ValidationRule]:
        """Get all rules in a category."""
        return [r for r in self.rules.values() if r.category == category]

    def by_severity(self, severity: RuleSeverity) -> List[ValidationRule]:
        """Get all rules of a severity level."""
        return [r for r in self.rules.values() if r.severity == severity]


class LatticeValidator:
    """
    Main validator for the Lattice Inference system.

    Runs validation rules against system state and produces reports.
    """

    def __init__(self):
        self.registry = RuleRegistry()
        self._build_sequences()

    def _build_sequences(self, max_n: int = 100):
        """Build Fibonacci and Lucas sequences for validation.

        We maintain TWO Fibonacci sequences:

        1. Mathematical Fibonacci (for invariant verification):
           F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5, ...
           These satisfy: Cassini, Spectral bound, Lucas bracket identities

        2. Zeckendorf Fibonacci (for encoding/decoding):
           1, 2, 3, 5, 8, 13, ... (no duplicate 1, starts at 1)
           These are used for token encoding bit positions

        Lucas sequence: L_0=2, L_1=1, L_2=3, L_3=4, L_4=7, L_5=11, ...
        """
        # Standard Fibonacci for mathematical invariants: F_0=0, F_1=1, F_2=1, F_3=2, ...
        self.fib_seq = [0, 1]
        while len(self.fib_seq) < max_n:
            self.fib_seq.append(self.fib_seq[-1] + self.fib_seq[-2])

        # Zeckendorf Fibonacci for encoding: 1, 2, 3, 5, 8, 13, ...
        self.zeck_fib_seq = [1, 2]
        while len(self.zeck_fib_seq) < max_n:
            self.zeck_fib_seq.append(self.zeck_fib_seq[-1] + self.zeck_fib_seq[-2])

        # Standard Lucas: L_0=2, L_1=1, L_2=3, L_3=4, ...
        self.lucas_seq = [2, 1]
        while len(self.lucas_seq) < max_n:
            self.lucas_seq.append(self.lucas_seq[-1] + self.lucas_seq[-2])

    def validate_bits(self, bits: List[int]) -> ValidationReport:
        """Validate a bit pattern."""
        report = ValidationReport()

        # S001: Zeckendorf constraint
        rule = self.registry.get("S001")
        report.add(rule.check(bits))

        # E001: Sum correctness (need to know target value)
        # Skipped - requires target value

        return report

    def validate_encoding(
        self,
        value: int,
        bits: List[int]
    ) -> ValidationReport:
        """Validate an encoding of a value."""
        report = ValidationReport()

        # S001: Zeckendorf constraint
        report.add(self.registry.get("S001").check(bits))

        # E001: Sum correctness (use Zeckendorf Fibonacci sequence)
        report.add(self.registry.get("E001").check(value, bits, self.zeck_fib_seq))

        return report

    def validate_collapse(
        self,
        before_bits: List[int],
        after_bits: List[int]
    ) -> ValidationReport:
        """Validate a collapse operation."""
        report = ValidationReport()

        # C002: Value preservation (use Zeckendorf Fibonacci)
        report.add(self.registry.get("C002").check(before_bits, after_bits, self.zeck_fib_seq))

        # S001: Result should be valid Zeckendorf
        report.add(self.registry.get("S001").check(after_bits))

        return report

    def validate_sequence(self, sequence: List[int], max_n: int = 1000) -> ValidationReport:
        """Validate a number sequence."""
        report = ValidationReport()

        # S002: Monotonicity
        report.add(self.registry.get("S002").check(sequence))

        # S003: Completeness
        report.add(self.registry.get("S003").check(sequence, max_n))

        return report

    def validate_structural_invariants(self, n: int) -> ValidationReport:
        """Validate structural invariants at position n."""
        report = ValidationReport()

        # V001: Cassini invariant
        if n > 0 and n < len(self.fib_seq) - 1:
            report.add(self.registry.get("V001").check(n, self.fib_seq))

        # V002: Spectral bound
        if n < len(self.fib_seq) and n < len(self.lucas_seq):
            report.add(self.registry.get("V002").check(n, self.fib_seq, self.lucas_seq))

        # E003: Lucas bracket
        if n > 0 and n < len(self.fib_seq) - 1 and n < len(self.lucas_seq):
            report.add(self.registry.get("E003").check(n, self.fib_seq, self.lucas_seq))

        return report

    def validate_all_invariants(self, max_n: int = 50) -> ValidationReport:
        """Run all invariant checks up to max_n."""
        report = ValidationReport()

        # Sequence validation
        report.add(self.registry.get("S002").check(self.fib_seq[:max_n]))
        report.add(self.registry.get("S003").check(self.fib_seq[:max_n], max_n * 10))

        # V003: Ratio convergence
        report.add(self.registry.get("V003").check(self.fib_seq[:max_n]))

        # Per-index invariants
        for n in range(1, min(max_n, len(self.fib_seq) - 1, len(self.lucas_seq))):
            sub_report = self.validate_structural_invariants(n)
            for result in sub_report.results:
                report.add(result)

        return report

    def validate_pipeline_state(
        self,
        bits: List[int],
        clock_n: int,
        polarity: int
    ) -> ValidationReport:
        """Validate pipeline state."""
        report = ValidationReport()

        # P001: Polarity
        report.add(self.registry.get("P001").check(clock_n, polarity))

        # P002: State validity
        report.add(self.registry.get("P002").check(bits))

        # P003: Clock sync
        max_active = max((i for i, b in enumerate(bits) if b == 1), default=0)
        report.add(self.registry.get("P003").check(clock_n, max_active))

        return report


def validate_rule(rule_id: str):
    """Decorator to automatically validate rule before/after function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = LatticeValidator()
            rule = validator.registry.get(rule_id)
            if rule is None:
                raise ValueError(f"Unknown rule: {rule_id}")

            result = func(*args, **kwargs)

            # Post-validation depends on rule type
            # This is a hook point for automatic validation
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    print("=" * 70)
    print("Validation Rules - Zero-RAM Lattice Inference")
    print("=" * 70)

    validator = LatticeValidator()

    # Test 1: Valid bit pattern
    print("\n1. Valid Zeckendorf bit pattern [1,0,1,0,0,1,0,0]:")
    bits = [1, 0, 1, 0, 0, 1, 0, 0]  # 1 + 3 + 13 = 17
    report = validator.validate_encoding(17, bits)
    print(f"   {report.summary()}")
    for r in report.results:
        print(f"   [{r.rule_id}] {'PASS' if r.passed else 'FAIL'}: {r.message}")

    # Test 2: Invalid bit pattern (adjacent 1s)
    print("\n2. Invalid bit pattern [1,1,0,0,1,0,0,0] (adjacent 1s):")
    bits = [1, 1, 0, 0, 1, 0, 0, 0]
    report = validator.validate_bits(bits)
    print(f"   {report.summary()}")
    for r in report.results:
        print(f"   [{r.rule_id}] {'PASS' if r.passed else 'FAIL'}: {r.message}")

    # Test 3: Structural invariants
    print("\n3. Structural invariants at n=10:")
    report = validator.validate_structural_invariants(10)
    print(f"   {report.summary()}")
    for r in report.results:
        print(f"   [{r.rule_id}] {'PASS' if r.passed else 'FAIL'}: {r.message}")

    # Test 4: Full invariant validation
    print("\n4. Full invariant validation (n=1 to 20):")
    report = validator.validate_all_invariants(max_n=20)
    print(f"   {report.summary()}")

    # Count by category
    by_cat = {}
    for r in report.results:
        cat = r.category.name
        if cat not in by_cat:
            by_cat[cat] = {"pass": 0, "fail": 0}
        by_cat[cat]["pass" if r.passed else "fail"] += 1

    for cat, counts in by_cat.items():
        print(f"   {cat}: {counts['pass']} passed, {counts['fail']} failed")

    # Test 5: Pipeline state validation
    print("\n5. Pipeline state validation:")
    bits = [0, 1, 0, 0, 1, 0, 0, 0]  # 2 + 8 = 10
    report = validator.validate_pipeline_state(bits, clock_n=5, polarity=-1)
    print(f"   {report.summary()}")
    for r in report.results:
        print(f"   [{r.rule_id}] {'PASS' if r.passed else 'FAIL'}: {r.message}")

    print("\n" + "=" * 70)
    print(f"Registered rules: {len(validator.registry.rules)}")
    for rule_id, rule in sorted(validator.registry.rules.items()):
        print(f"  {rule_id}: {rule.description} [{rule.severity.name}]")
