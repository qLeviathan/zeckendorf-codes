"""
RTL (Right-to-Left) Operations Module for Zeckendorf Representations

This module provides primitive operations that are aware of and preserve
the RTL mechanical structure fundamental to Zeckendorf decomposition.

Key Concepts:
------------
1. RTL Traversal: Processing sequences from largest to smallest
2. Greedy Selection: At each step, take the largest valid term
3. Residual Tracking: Maintain the remaining value to decompose
4. Index Mapping: Bidirectional mapping between positions and values

These primitives can be composed to build compression algorithms while
maintaining invariant guarantees at each step.
"""

from itertools import takewhile
from collections import namedtuple
import math

from invariants import (
    SequenceInvariant,
    DecompositionInvariant,
    RTLProcessingInvariant,
    InvariantViolation,
)


# Structured types for clarity
RTLState = namedtuple('RTLState', ['value', 'remaining', 'position', 'decomposition'])
SelectionResult = namedtuple('SelectionResult', ['term', 'index', 'multiplier', 'remaining'])
EncodedTerm = namedtuple('EncodedTerm', ['index', 'multiplier', 'bits'])


class RTLSequence:
    """
    A sequence wrapper that provides RTL-aware access patterns.

    This class wraps a sequence generator and provides methods that
    naturally express RTL operations while maintaining invariants.
    """

    def __init__(self, generator_func, name="Sequence"):
        """
        Initialize with a generator function.

        Args:
            generator_func: Function that yields sequence terms (ascending)
            name: Human-readable name for error messages
        """
        self.generator = generator_func
        self.name = name
        self._cache = []  # Memoized sequence terms
        self._max_cached = 0  # Largest value with complete cache

    def _ensure_cached_to(self, n):
        """Ensure sequence is cached up to at least n."""
        if n <= self._max_cached:
            return

        gen = self.generator()
        self._cache = []

        for term in gen:
            self._cache.append(term)
            if term > n:
                break

        self._max_cached = n if self._cache else 0

    def terms_up_to(self, n):
        """
        Get all sequence terms up to n (ascending order).

        Args:
            n: Upper bound (inclusive if term equals n)

        Returns:
            list: Terms in ascending order
        """
        self._ensure_cached_to(n)
        return [t for t in self._cache if t <= n]

    def rtl_terms(self, n):
        """
        Get sequence terms in RTL (descending) order.

        This is the fundamental RTL operation. Processing in this order
        enables greedy selection of largest valid terms.

        Args:
            n: Upper bound

        Returns:
            list: Terms in descending order
        """
        return list(reversed(self.terms_up_to(n)))

    def rtl_enumerate(self, n):
        """
        Enumerate terms in RTL order with their original (ascending) indexes.

        Yields:
            tuple: (original_index, term) pairs in descending term order
        """
        terms = self.terms_up_to(n)
        for i, term in enumerate(reversed(terms)):
            original_idx = len(terms) - 1 - i
            yield (original_idx, term)

    def index_of(self, term):
        """
        Get the index of a term in the ascending sequence.

        Args:
            term: Value to find

        Returns:
            int: Index in ascending sequence

        Raises:
            ValueError: If term not in sequence
        """
        self._ensure_cached_to(term)
        if term in self._cache:
            return self._cache.index(term)
        raise ValueError(f"{term} not found in {self.name}")

    def term_at(self, index):
        """
        Get the term at a specific index.

        Args:
            index: Position in ascending sequence

        Returns:
            Sequence term at that position
        """
        # Ensure we have enough terms
        while len(self._cache) <= index:
            self._ensure_cached_to(self._cache[-1] * 2 if self._cache else 100)

        return self._cache[index]

    def verify_structure(self, up_to_n):
        """Verify sequence satisfies structural invariants."""
        terms = self.terms_up_to(up_to_n)
        SequenceInvariant.verify_monotonicity(terms)
        return True


class RTLDecomposer:
    """
    Decomposes integers using RTL greedy algorithm.

    This class implements the core RTL decomposition with explicit
    state tracking at each step.
    """

    def __init__(self, sequence, k=1):
        """
        Initialize decomposer.

        Args:
            sequence: RTLSequence instance
            k: Maximum multiplier for generalized Zeckendorf (1 for standard)
        """
        self.sequence = sequence
        self.k = k

    def initial_state(self, n):
        """
        Create initial RTL processing state.

        Args:
            n: Value to decompose

        Returns:
            RTLState: Initial state
        """
        return RTLState(
            value=n,
            remaining=n,
            position=0,  # Position in RTL traversal
            decomposition=[]
        )

    def select_term(self, state, index, term):
        """
        Select a term in the RTL greedy algorithm.

        This is the atomic operation of RTL decomposition:
        - Check if term (with some multiplier) fits in remaining
        - Find maximum valid multiplier
        - Return selection result

        Args:
            state: Current RTLState
            index: Index of term in ascending sequence
            term: Term value

        Returns:
            SelectionResult or None if term doesn't fit
        """
        if term > state.remaining:
            return None

        # Find maximum valid multiplier
        for mult in reversed(range(1, self.k + 1)):
            if term * mult <= state.remaining:
                return SelectionResult(
                    term=term,
                    index=index,
                    multiplier=mult,
                    remaining=state.remaining - term * mult
                )

        return None

    def apply_selection(self, state, selection):
        """
        Apply a selection to produce new state.

        Args:
            state: Current RTLState
            selection: SelectionResult to apply

        Returns:
            RTLState: New state after selection
        """
        new_decomp = state.decomposition + [(selection.term, selection.multiplier)]

        return RTLState(
            value=state.value,
            remaining=selection.remaining,
            position=state.position + 1,
            decomposition=new_decomp
        )

    def decompose(self, n):
        """
        Fully decompose n using RTL greedy algorithm.

        Args:
            n: Value to decompose

        Returns:
            list: Decomposition as (term, multiplier) pairs
        """
        if n <= 0:
            raise ValueError("Can only decompose positive integers")

        state = self.initial_state(n)

        for index, term in self.sequence.rtl_enumerate(n):
            selection = self.select_term(state, index, term)
            if selection:
                state = self.apply_selection(state, selection)

                if state.remaining == 0:
                    break

        # Verify correctness
        DecompositionInvariant.verify_sum_strict(state.decomposition, n)

        return state.decomposition

    def decompose_with_trace(self, n):
        """
        Decompose with full trace of RTL processing steps.

        Useful for debugging and understanding the algorithm.

        Args:
            n: Value to decompose

        Yields:
            tuple: (state_before, selection, state_after) for each step
        """
        if n <= 0:
            raise ValueError("Can only decompose positive integers")

        state = self.initial_state(n)

        for index, term in self.sequence.rtl_enumerate(n):
            selection = self.select_term(state, index, term)
            if selection:
                old_state = state
                state = self.apply_selection(state, selection)
                yield (old_state, selection, state)

                if state.remaining == 0:
                    break


class RTLEncoder:
    """
    Encodes decompositions into binary representations.

    This handles the RTL-aware encoding where:
    - Indexes are from the ascending sequence
    - Output is ordered for efficient decoding
    """

    def __init__(self, k=1):
        """
        Initialize encoder.

        Args:
            k: Maximum multiplier (determines bits per index)
        """
        self.k = k
        self.bits_per_multiplier = int(math.ceil(math.log(k + 1, 2))) if k > 1 else 1

    def encode_decomposition(self, decomposition, max_index):
        """
        Encode a decomposition to binary string.

        For standard Zeckendorf (k=1):
            - Output has max_index+1 bits
            - Bit i is '1' if index i is used

        For generalized (k>1):
            - Output has (max_index+1) * bits_per_multiplier bits
            - Each position encodes the multiplier (0 if unused)

        Args:
            decomposition: List of (term, multiplier) or (index, multiplier)
            max_index: Maximum index in decomposition

        Returns:
            str: Binary encoded string
        """
        # Build index -> multiplier map
        index_mult = {}
        for item in decomposition:
            if isinstance(item, (list, tuple)):
                idx, mult = item[0], item[1]
                # If item looks like (term, mult), we need index conversion
                # Assume it's already (index, multiplier) for this encoder
                index_mult[idx] = mult
            else:
                index_mult[item] = 1

        if self.k == 1:
            # Standard: just mark used positions
            bits = ''
            for i in range(max_index + 1):
                bits += '1' if i in index_mult else '0'
            return bits
        else:
            # Generalized: encode multiplier at each position
            bits = ''
            for i in range(max_index + 1):
                mult = index_mult.get(i, 0)
                bits += format(mult, f'0{self.bits_per_multiplier}b')
            return bits

    def encode_term(self, index, multiplier):
        """
        Encode a single term.

        Args:
            index: Sequence index
            multiplier: Multiplier value

        Returns:
            EncodedTerm with encoding information
        """
        if self.k == 1:
            bits = '1'
        else:
            bits = format(multiplier, f'0{self.bits_per_multiplier}b')

        return EncodedTerm(index=index, multiplier=multiplier, bits=bits)


class RTLDecoder:
    """
    Decodes binary representations back to values.

    Maintains the decompression matrix for efficient decoding.
    """

    def __init__(self, sequence, k=1, max_value=255):
        """
        Initialize decoder.

        Args:
            sequence: RTLSequence instance
            k: Maximum multiplier
            max_value: Maximum value to prepare for
        """
        self.sequence = sequence
        self.k = k
        self.bits_per_multiplier = int(math.ceil(math.log(k + 1, 2))) if k > 1 else 1
        self.matrix = sequence.terms_up_to(max_value)

    def decode_binary(self, bits):
        """
        Decode binary string to integer value.

        Args:
            bits: Binary string from encoder

        Returns:
            int: Decoded value
        """
        value = 0

        if self.k == 1:
            # Standard: each bit indicates presence
            for i, bit in enumerate(bits):
                if bit == '1':
                    if i < len(self.matrix):
                        value += self.matrix[i]
        else:
            # Generalized: each chunk encodes multiplier
            chunks = [
                bits[i:i+self.bits_per_multiplier]
                for i in range(0, len(bits), self.bits_per_multiplier)
            ]
            for i, chunk in enumerate(chunks):
                mult = int(chunk, 2)
                if mult > 0 and i < len(self.matrix):
                    value += self.matrix[i] * mult

        return value


class RTLCompressionPipeline:
    """
    Complete RTL-aware compression pipeline.

    Combines sequence, decomposer, encoder for end-to-end compression.
    """

    def __init__(self, sequence, k=1, delimiter='1'):
        """
        Initialize pipeline.

        Args:
            sequence: RTLSequence instance
            k: Maximum multiplier
            delimiter: Bit pattern to separate encoded numbers
        """
        self.sequence = sequence
        self.decomposer = RTLDecomposer(sequence, k)
        self.encoder = RTLEncoder(k)
        self.k = k
        self.delimiter = delimiter

    def compress_number(self, n):
        """
        Compress a single number.

        Args:
            n: Number to compress

        Returns:
            str: Binary string representation
        """
        # Decompose using RTL greedy
        decomposition = self.decomposer.decompose(n)

        # Convert to indexes
        index_decomp = []
        for term, mult in decomposition:
            idx = self.sequence.index_of(term)
            index_decomp.append((idx, mult))

        # Find max index for encoding
        max_idx = max(idx for idx, _ in index_decomp)

        # Encode
        # Note: encoder expects indexes, but we have (index, mult)
        # Rebuild as index_mult for the encoder
        return self.encoder.encode_decomposition(index_decomp, max_idx)

    def compress_stream(self, numbers):
        """
        Compress a stream of numbers.

        Args:
            numbers: Iterable of numbers

        Yields:
            str: Compressed binary for each number (with delimiter)
        """
        for n in numbers:
            yield self.compress_number(n) + self.delimiter


# Pre-built sequence instances
def fibonacci_gen():
    """Fibonacci sequence for Zeckendorf: 1, 2, 3, 5, 8, ..."""
    a, b = 1, 1
    yield 1
    while True:
        a, b = b, a + b
        yield b

def lucas_gen():
    """Lucas sequence for Zeckendorf-like representation: 1, 2, 3, 4, 7, 11, ..."""
    yield 1
    yield 2
    a, b = 1, 3
    while True:
        yield b
        a, b = b, a + b

def make_gzeck_gen(k):
    def gen():
        a, b = 1, 1
        while True:
            yield b
            a, b = b, a + b * k
    return gen


FibonacciSeq = RTLSequence(fibonacci_gen, "Fibonacci")
LucasSeq = RTLSequence(lucas_gen, "Lucas")


def create_gzeck_sequence(k):
    """Create generalized Zeckendorf sequence with parameter k."""
    return RTLSequence(make_gzeck_gen(k), f"GZeck({k})")


if __name__ == "__main__":
    print("RTL Operations - Demonstration")
    print("=" * 50)

    # RTL Sequence operations
    print("\n1. RTL Sequence Access")
    print("-" * 40)

    print(f"  Fibonacci terms up to 50: {FibonacciSeq.terms_up_to(50)}")
    print(f"  RTL (descending) order:   {FibonacciSeq.rtl_terms(50)}")

    print("\n  RTL enumeration (index, term):")
    for idx, term in FibonacciSeq.rtl_enumerate(50):
        print(f"    Index {idx}: {term}")

    # RTL Decomposition with trace
    print("\n2. RTL Decomposition with Trace")
    print("-" * 40)

    decomposer = RTLDecomposer(FibonacciSeq, k=1)
    print("  Decomposing 50:")

    for before, selection, after in decomposer.decompose_with_trace(50):
        print(f"    State: remaining={before.remaining}")
        print(f"    Select: term={selection.term} at index={selection.index}")
        print(f"    New remaining: {after.remaining}")
        print()

    # Encoding
    print("\n3. RTL Encoding")
    print("-" * 40)

    pipeline = RTLCompressionPipeline(FibonacciSeq, k=1)

    for n in [10, 20, 50, 100]:
        compressed = pipeline.compress_number(n)
        print(f"  {n:3d} -> {compressed}")

    # Generalized Zeckendorf
    print("\n4. Generalized Zeckendorf (k=3)")
    print("-" * 40)

    gzeck3 = create_gzeck_sequence(3)
    decomposer_g = RTLDecomposer(gzeck3, k=3)

    print(f"  GZeck(3) terms up to 50: {gzeck3.terms_up_to(50)}")

    for n in [10, 20, 50]:
        decomp = decomposer_g.decompose(n)
        terms = [f"{t}*{m}" for t, m in decomp]
        print(f"  {n:3d} = {' + '.join(terms)}")
