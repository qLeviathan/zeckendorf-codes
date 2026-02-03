"""
Fibonacci-Lucas Token Encoding System for Zero-RAM Lattice Inference

This module implements a token encoding system based on Zeckendorf representations
and Lucas prefiltering, designed for efficient lattice-based inference operations.

The Zero-RAM Lattice Inference framework leverages:
1. Zeckendorf representations for compact, non-adjacent bit patterns
2. Lucas numbers for structural position filtering
3. The identity L_n = F_{n-1} + F_{n+1} for efficient bracket computation

Key Components:
--------------
- TokenEncoder: Converts token IDs to Zeckendorf bit patterns
- LucasPrefilter: Computes structural linking constraints via Lucas brackets
- EncodedToken: Dataclass holding complete encoding information
- VocabularyEncoder: Manages vocabulary with frequency-based ID assignment
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator, Dict
from collections import Counter
import re

from invariants import (
    DecompositionInvariant,
    RTLMechanicalStructure,
    fibonacci_generator,
)
from rtl_operations import (
    RTLSequence,
    RTLDecomposer,
    FibonacciSeq,
    LucasSeq,
)


# ============================================================================
# Fibonacci and Lucas Sequence Utilities
# ============================================================================

def fibonacci_up_to_index(max_index: int) -> List[int]:
    """
    Generate Fibonacci numbers up to a given index.

    The sequence starts: F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5, ...
    For Zeckendorf we typically use F_2, F_3, F_4, ... = 1, 2, 3, 5, 8, ...

    Args:
        max_index: Maximum Fibonacci index to generate

    Returns:
        List of Fibonacci numbers from F_0 to F_max_index
    """
    if max_index < 0:
        return []
    if max_index == 0:
        return [0]
    if max_index == 1:
        return [0, 1]

    fibs = [0, 1]
    for i in range(2, max_index + 1):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def lucas_up_to_index(max_index: int) -> List[int]:
    """
    Generate Lucas numbers up to a given index.

    The sequence starts: L_0=2, L_1=1, L_2=3, L_3=4, L_4=7, L_5=11, ...

    Args:
        max_index: Maximum Lucas index to generate

    Returns:
        List of Lucas numbers from L_0 to L_max_index
    """
    if max_index < 0:
        return []
    if max_index == 0:
        return [2]
    if max_index == 1:
        return [2, 1]

    lucas = [2, 1]
    for i in range(2, max_index + 1):
        lucas.append(lucas[-1] + lucas[-2])
    return lucas


def lucas_from_fibonacci(n: int, fibs: List[int]) -> int:
    """
    Compute Lucas number L_n using the identity L_n = F_{n-1} + F_{n+1}.

    This is the core identity used for Lucas prefiltering.

    Args:
        n: Index for Lucas number
        fibs: Precomputed Fibonacci sequence (must include F_{n+1})

    Returns:
        L_n = F_{n-1} + F_{n+1}
    """
    if n == 0:
        # L_0 = F_{-1} + F_1 = 1 + 1 = 2
        # (F_{-1} = 1 by extension of Fibonacci to negative indices)
        return 2
    if n + 1 >= len(fibs):
        raise ValueError(f"Fibonacci sequence too short for L_{n}")

    f_n_minus_1 = fibs[n - 1] if n > 0 else 1  # F_{-1} = 1
    f_n_plus_1 = fibs[n + 1]
    return f_n_minus_1 + f_n_plus_1


# ============================================================================
# EncodedToken Dataclass
# ============================================================================

@dataclass
class EncodedToken:
    """
    Represents a token with its complete Zeckendorf encoding and Lucas brackets.

    Attributes:
        token_id: Unique integer identifier for the token
        text: The original text (word/subword) of the token
        zeckendorf_bits: Binary coefficients c_i where value = sum(c_i * F_i)
        fibonacci_indices: List of indices i where c_i = 1 (active Fibonacci terms)
        lucas_brackets: Tuple (L_{n-1}, L_{n+1}) for position n in sequence
    """
    token_id: int
    text: str
    zeckendorf_bits: List[int] = field(default_factory=list)
    fibonacci_indices: List[int] = field(default_factory=list)
    lucas_brackets: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    def bit_pattern(self) -> str:
        """Return the Zeckendorf bits as a binary string."""
        return ''.join(str(b) for b in self.zeckendorf_bits)

    def to_int(self) -> int:
        """Convert bit pattern to integer (LSB first encoding)."""
        result = 0
        for i, bit in enumerate(self.zeckendorf_bits):
            if bit:
                result |= (1 << i)
        return result

    def verify_non_adjacency(self) -> bool:
        """Verify that no two adjacent Fibonacci indices are used."""
        return DecompositionInvariant.is_non_adjacent(self.fibonacci_indices)

    def __repr__(self) -> str:
        return (f"EncodedToken(id={self.token_id}, text='{self.text}', "
                f"bits={self.bit_pattern()}, indices={self.fibonacci_indices}, "
                f"lucas_brackets={self.lucas_brackets})")


# ============================================================================
# TokenEncoder Class
# ============================================================================

class TokenEncoder:
    """
    Encodes token IDs to Zeckendorf representations with configurable bit widths.

    Zeckendorf's theorem guarantees that every positive integer has a unique
    representation as a sum of non-consecutive Fibonacci numbers. This encoder
    converts token IDs to such representations, storing them as bit patterns.

    Supported bit widths:
    - 8-bit:  Tokens 1-54 (sum of first 9 Fibonacci: 1+2+3+5+8+13+21+34=87, but non-adjacent max ~54)
    - 16-bit: Tokens 1-2583 (first 17 Fibonacci)
    - 32-bit: Tokens 1-3524577 (first 33 Fibonacci)
    """

    # Maximum representable values for each bit width
    BIT_WIDTH_LIMITS = {
        8: 54,       # Max with 8 Fibonacci terms (non-adjacent)
        16: 2583,    # Max with 16 Fibonacci terms
        32: 3524577, # Max with 32 Fibonacci terms
    }

    def __init__(self, bit_width: int = 16):
        """
        Initialize the encoder with a specific bit width.

        Args:
            bit_width: One of 8, 16, or 32
        """
        if bit_width not in self.BIT_WIDTH_LIMITS:
            raise ValueError(f"bit_width must be one of {list(self.BIT_WIDTH_LIMITS.keys())}")

        self.bit_width = bit_width
        self.max_value = self.BIT_WIDTH_LIMITS[bit_width]

        # Precompute Fibonacci numbers for this bit width
        # We need enough Fibonacci numbers to cover the bit width
        self._fibonacci = fibonacci_up_to_index(bit_width + 2)

        # For Zeckendorf, we use F_2, F_3, ... (skipping F_0=0, F_1=1)
        # This gives us: 1, 2, 3, 5, 8, 13, 21, ...
        self._zeck_fibs = self._fibonacci[2:bit_width + 2]

        # Create RTL decomposer for the Zeckendorf sequence
        self._rtl_sequence = FibonacciSeq
        self._decomposer = RTLDecomposer(self._rtl_sequence, k=1)

    def encode(self, token_id: int) -> List[int]:
        """
        Convert a token ID to its Zeckendorf bit representation.

        Args:
            token_id: Positive integer token ID

        Returns:
            List of bits where position i corresponds to F_{i+2}

        Raises:
            ValueError: If token_id is out of range
        """
        if token_id < 1:
            raise ValueError("Token ID must be positive")
        if token_id > self.max_value:
            raise ValueError(f"Token ID {token_id} exceeds max {self.max_value} for {self.bit_width}-bit encoding")

        # Get Zeckendorf decomposition
        decomposition = self._decomposer.decompose(token_id)

        # Convert terms to bit positions
        # decomposition is [(term, multiplier), ...] where multiplier is always 1
        active_terms = {term for term, mult in decomposition}

        # Build bit array
        bits = []
        for i, fib in enumerate(self._zeck_fibs):
            bits.append(1 if fib in active_terms else 0)

        # Pad to bit_width
        while len(bits) < self.bit_width:
            bits.append(0)

        return bits[:self.bit_width]

    def get_fibonacci_indices(self, token_id: int) -> List[int]:
        """
        Get the indices of active Fibonacci terms in the Zeckendorf representation.

        Args:
            token_id: Positive integer token ID

        Returns:
            List of indices where the bit is 1
        """
        bits = self.encode(token_id)
        return [i for i, bit in enumerate(bits) if bit == 1]

    def decode(self, bits: List[int]) -> int:
        """
        Convert a Zeckendorf bit representation back to an integer.

        Args:
            bits: List of bits where position i corresponds to F_{i+2}

        Returns:
            The integer value represented
        """
        value = 0
        for i, bit in enumerate(bits):
            if bit and i < len(self._zeck_fibs):
                value += self._zeck_fibs[i]
        return value

    def pack_to_bytes(self, bits: List[int]) -> bytes:
        """
        Pack bit representation to bytes (little-endian bit order).

        Args:
            bits: Zeckendorf bit representation

        Returns:
            Bytes representation
        """
        num_bytes = (len(bits) + 7) // 8
        result = bytearray(num_bytes)

        for i, bit in enumerate(bits):
            if bit:
                byte_idx = i // 8
                bit_idx = i % 8
                result[byte_idx] |= (1 << bit_idx)

        return bytes(result)

    def unpack_from_bytes(self, data: bytes) -> List[int]:
        """
        Unpack bytes to bit representation.

        Args:
            data: Bytes to unpack

        Returns:
            List of bits
        """
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        return bits[:self.bit_width]


# ============================================================================
# LucasPrefilter Class
# ============================================================================

class LucasPrefilter:
    """
    Computes Lucas brackets for structural position filtering in lattice inference.

    The Lucas prefilter uses the identity L_n = F_{n-1} + F_{n+1} to determine
    which tokens can structurally link to a given position in a sequence.

    For a token at position n, the Lucas brackets (L_{n-1}, L_{n+1}) define
    the structural constraints for valid adjacent positions.
    """

    def __init__(self, max_position: int = 1000):
        """
        Initialize the prefilter with precomputed sequences.

        Args:
            max_position: Maximum position to support
        """
        self.max_position = max_position

        # Precompute Fibonacci and Lucas sequences
        # Need F up to max_position + 2 for Lucas computation
        self._fibonacci = fibonacci_up_to_index(max_position + 3)
        self._lucas = lucas_up_to_index(max_position + 2)

    def lucas_brackets(self, position: int) -> Tuple[int, int]:
        """
        Compute Lucas brackets (L_{n-1}, L_{n+1}) for a given position n.

        These brackets determine structural linking constraints:
        - L_{n-1} constrains what can precede position n
        - L_{n+1} constrains what can follow position n

        Args:
            position: Position in the sequence (0-indexed)

        Returns:
            Tuple of (L_{n-1}, L_{n+1})
        """
        if position < 0:
            raise ValueError("Position must be non-negative")
        if position > self.max_position:
            raise ValueError(f"Position {position} exceeds max {self.max_position}")

        # L_{n-1}
        if position == 0:
            l_prev = self._lucas[0]  # L_{-1} = L_0 by convention for position 0
        else:
            l_prev = self._lucas[position - 1]

        # L_{n+1}
        l_next = self._lucas[position + 1]

        return (l_prev, l_next)

    def lucas_at(self, n: int) -> int:
        """
        Get the Lucas number L_n.

        Args:
            n: Index for Lucas number

        Returns:
            L_n
        """
        if n < 0:
            # L_{-n} = (-1)^n * L_n for negative indices
            return ((-1) ** n) * self._lucas[abs(n)]
        return self._lucas[n]

    def verify_lucas_identity(self, n: int) -> bool:
        """
        Verify the identity L_n = F_{n-1} + F_{n+1}.

        Args:
            n: Index to verify

        Returns:
            True if identity holds
        """
        if n <= 0 or n + 1 >= len(self._fibonacci):
            return False

        l_n = self._lucas[n]
        f_computed = self._fibonacci[n - 1] + self._fibonacci[n + 1]
        return l_n == f_computed

    def can_link(self, from_position: int, to_position: int) -> bool:
        """
        Determine if a structural link is valid between two positions.

        Valid links satisfy the Lucas bracket constraints:
        - Adjacent positions (difference of 1) are always linkable
        - Non-adjacent links are constrained by the Lucas sum property

        Args:
            from_position: Source position
            to_position: Target position

        Returns:
            True if link is structurally valid
        """
        if from_position < 0 or to_position < 0:
            return False

        # Adjacent positions are always valid
        diff = abs(to_position - from_position)
        if diff == 1:
            return True

        # For non-adjacent, use Lucas constraint
        # The sum of Lucas brackets must satisfy certain properties
        # This is a simplified model; real constraints depend on token values
        if diff == 0:
            return True  # Same position (no link needed)

        # Non-adjacent links allowed only if gap is at least 2
        # (mirrors the Zeckendorf non-adjacency constraint)
        return diff >= 2

    def get_valid_link_targets(self, position: int, max_lookahead: int = 10) -> List[int]:
        """
        Get all valid positions that can be linked from a given position.

        Args:
            position: Current position
            max_lookahead: Maximum positions to look ahead

        Returns:
            List of valid target positions
        """
        targets = []
        for offset in range(1, max_lookahead + 1):
            target = position + offset
            if target <= self.max_position and self.can_link(position, target):
                targets.append(target)
        return targets


# ============================================================================
# VocabularyEncoder Class
# ============================================================================

class VocabularyEncoder:
    """
    Manages vocabulary with frequency-based ID assignment for Zeckendorf encoding.

    Tokens are assigned IDs based on frequency, with more frequent tokens
    receiving lower IDs. This results in smaller Zeckendorf representations
    for common tokens, optimizing overall encoding efficiency.
    """

    def __init__(self, bit_width: int = 16, tokenizer_pattern: str = r'\b\w+\b|\S'):
        """
        Initialize the vocabulary encoder.

        Args:
            bit_width: Bit width for Zeckendorf encoding (8, 16, or 32)
            tokenizer_pattern: Regex pattern for tokenization
        """
        self.bit_width = bit_width
        self.tokenizer_pattern = re.compile(tokenizer_pattern)

        self._token_encoder = TokenEncoder(bit_width)
        self._lucas_prefilter = LucasPrefilter()

        # Vocabulary mappings
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._token_frequencies: Counter = Counter()

        # Special tokens
        self._unk_token = "<UNK>"
        self._pad_token = "<PAD>"
        self._bos_token = "<BOS>"
        self._eos_token = "<EOS>"

        # Reserve IDs 1-4 for special tokens
        self._special_tokens = {
            self._pad_token: 1,
            self._unk_token: 2,
            self._bos_token: 3,
            self._eos_token: 4,
        }
        self._next_id = 5

        # Initialize special tokens
        for token, token_id in self._special_tokens.items():
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        return self.tokenizer_pattern.findall(text.lower())

    def build_vocabulary(self, corpus: str, min_frequency: int = 1) -> None:
        """
        Build vocabulary from a text corpus.

        Tokens are assigned IDs based on frequency, with more frequent tokens
        receiving lower IDs (after special tokens).

        Args:
            corpus: Text corpus to build vocabulary from
            min_frequency: Minimum frequency for a token to be included
        """
        # Count token frequencies
        tokens = self.tokenize(corpus)
        self._token_frequencies = Counter(tokens)

        # Sort by frequency (descending), then alphabetically for ties
        sorted_tokens = sorted(
            self._token_frequencies.items(),
            key=lambda x: (-x[1], x[0])
        )

        # Assign IDs (lower ID = more frequent = smaller Zeckendorf)
        max_vocab = self._token_encoder.max_value - self._next_id

        for token, freq in sorted_tokens:
            if freq < min_frequency:
                continue
            if token in self._token_to_id:
                continue
            if self._next_id > self._token_encoder.max_value:
                break

            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1

    def add_token(self, token: str) -> int:
        """
        Add a single token to the vocabulary.

        Args:
            token: Token to add

        Returns:
            Assigned token ID
        """
        if token in self._token_to_id:
            return self._token_to_id[token]

        if self._next_id > self._token_encoder.max_value:
            raise ValueError("Vocabulary full")

        token_id = self._next_id
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._next_id += 1
        return token_id

    def get_token_id(self, token: str) -> int:
        """
        Get the ID for a token, returning UNK ID if not in vocabulary.

        Args:
            token: Token to look up

        Returns:
            Token ID
        """
        return self._token_to_id.get(token, self._special_tokens[self._unk_token])

    def get_token(self, token_id: int) -> str:
        """
        Get the token for an ID.

        Args:
            token_id: Token ID to look up

        Returns:
            Token string
        """
        return self._id_to_token.get(token_id, self._unk_token)

    def encode_token(self, token: str, position: int = 0) -> EncodedToken:
        """
        Fully encode a token with Zeckendorf bits and Lucas brackets.

        Args:
            token: Token text
            position: Position in sequence for Lucas bracket computation

        Returns:
            EncodedToken with complete encoding information
        """
        token_id = self.get_token_id(token)
        zeck_bits = self._token_encoder.encode(token_id)
        fib_indices = self._token_encoder.get_fibonacci_indices(token_id)
        lucas_brack = self._lucas_prefilter.lucas_brackets(position)

        return EncodedToken(
            token_id=token_id,
            text=token,
            zeckendorf_bits=zeck_bits,
            fibonacci_indices=fib_indices,
            lucas_brackets=lucas_brack,
        )

    def encode_text(self, text: str) -> List[EncodedToken]:
        """
        Encode a full text into a list of EncodedTokens.

        Args:
            text: Input text

        Returns:
            List of EncodedToken objects
        """
        tokens = self.tokenize(text)
        encoded = []

        for position, token in enumerate(tokens):
            encoded.append(self.encode_token(token, position))

        return encoded

    def encode_streaming(self, text_stream: Iterator[str]) -> Iterator[EncodedToken]:
        """
        Streaming encoder for continuous text input.

        Args:
            text_stream: Iterator yielding text chunks

        Yields:
            EncodedToken objects for each token
        """
        position = 0

        for chunk in text_stream:
            tokens = self.tokenize(chunk)
            for token in tokens:
                yield self.encode_token(token, position)
                position += 1

    def decode_tokens(self, encoded_tokens: List[EncodedToken]) -> str:
        """
        Decode a list of EncodedTokens back to text.

        Args:
            encoded_tokens: List of EncodedToken objects

        Returns:
            Reconstructed text
        """
        return ' '.join(et.text for et in encoded_tokens)

    @property
    def vocabulary_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self._token_to_id)

    @property
    def max_vocabulary_size(self) -> int:
        """Return the maximum vocabulary size for the current bit width."""
        return self._token_encoder.max_value


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate():
    """Demonstrate the Fibonacci-Lucas token encoding system."""

    print("=" * 70)
    print("Fibonacci-Lucas Token Encoding System")
    print("Zero-RAM Lattice Inference Framework")
    print("=" * 70)

    # Sample corpus
    sample_corpus = """
    The quick brown fox jumps over the lazy dog.
    A quick brown dog runs through the park.
    The lazy fox sleeps under the tree.
    Quick thinking helps solve problems.
    The brown bear found a tree in the forest.
    """

    sample_text = "The quick brown fox jumps."

    # -------------------------------------------------------------------------
    print("\n1. TokenEncoder - Zeckendorf Representation")
    print("-" * 50)

    encoder = TokenEncoder(bit_width=16)

    print(f"Bit width: {encoder.bit_width}")
    print(f"Maximum token ID: {encoder.max_value}")
    print()

    # Show encoding for sample IDs
    sample_ids = [1, 2, 3, 5, 8, 13, 21, 50, 100]
    print("Token ID | Zeckendorf Bits         | Fibonacci Indices")
    print("-" * 60)

    for token_id in sample_ids:
        bits = encoder.encode(token_id)
        indices = encoder.get_fibonacci_indices(token_id)
        bit_str = ''.join(str(b) for b in bits[:12])  # Show first 12 bits
        print(f"{token_id:8d} | {bit_str}... | {indices}")

    # Verify round-trip
    print("\nRound-trip verification:")
    for token_id in sample_ids:
        bits = encoder.encode(token_id)
        decoded = encoder.decode(bits)
        status = "OK" if decoded == token_id else "FAIL"
        print(f"  {token_id} -> bits -> {decoded} [{status}]")

    # -------------------------------------------------------------------------
    print("\n2. LucasPrefilter - Structural Position Filtering")
    print("-" * 50)

    prefilter = LucasPrefilter()

    print("Position | Lucas Brackets (L_{n-1}, L_{n+1}) | Identity Check")
    print("-" * 65)

    for pos in range(10):
        brackets = prefilter.lucas_brackets(pos)
        identity_ok = prefilter.verify_lucas_identity(pos) if pos > 0 else "N/A"
        l_n = prefilter.lucas_at(pos)
        print(f"{pos:8d} | ({brackets[0]:4d}, {brackets[1]:4d})  L_{pos}={l_n:4d}  | {identity_ok}")

    print("\nLucas identity L_n = F_{n-1} + F_{n+1}:")
    fibs = fibonacci_up_to_index(12)
    for n in range(1, 10):
        l_n = prefilter.lucas_at(n)
        f_computed = fibs[n - 1] + fibs[n + 1]
        print(f"  L_{n} = {l_n} = F_{n-1} + F_{n+1} = {fibs[n-1]} + {fibs[n+1]} = {f_computed}")

    # -------------------------------------------------------------------------
    print("\n3. VocabularyEncoder - Full Text Encoding")
    print("-" * 50)

    vocab_encoder = VocabularyEncoder(bit_width=16)
    vocab_encoder.build_vocabulary(sample_corpus, min_frequency=1)

    print(f"Vocabulary size: {vocab_encoder.vocabulary_size}")
    print(f"Max vocabulary size: {vocab_encoder.max_vocabulary_size}")
    print()

    # Show frequency-based ID assignment
    print("Token frequency ranking (top 10):")
    freq_items = vocab_encoder._token_frequencies.most_common(10)
    for token, freq in freq_items:
        token_id = vocab_encoder.get_token_id(token)
        print(f"  '{token}': freq={freq}, id={token_id}")

    # -------------------------------------------------------------------------
    print(f"\n4. Encoding Sample Text: \"{sample_text}\"")
    print("-" * 50)

    encoded_tokens = vocab_encoder.encode_text(sample_text)

    print("\nPos | Token    | ID  | Zeckendorf Bits   | Fib Indices | Lucas Brackets")
    print("-" * 80)

    for i, et in enumerate(encoded_tokens):
        bit_str = et.bit_pattern()[:16]
        print(f"{i:3d} | {et.text:8s} | {et.token_id:3d} | {bit_str} | {et.fibonacci_indices} | {et.lucas_brackets}")

    # Verify non-adjacency
    print("\nNon-adjacency verification:")
    for et in encoded_tokens:
        valid = et.verify_non_adjacency()
        status = "VALID" if valid else "INVALID"
        print(f"  '{et.text}' (id={et.token_id}): {status}")

    # -------------------------------------------------------------------------
    print("\n5. Bit Width Comparison")
    print("-" * 50)

    print("Bit Width | Max Token ID | Example Encoding (ID=100)")
    print("-" * 55)

    for width in [8, 16, 32]:
        enc = TokenEncoder(bit_width=width)
        if 100 <= enc.max_value:
            bits = enc.encode(100)
            bit_str = ''.join(str(b) for b in bits[:16]) + ("..." if width > 16 else "")
            print(f"{width:9d} | {enc.max_value:12d} | {bit_str}")
        else:
            print(f"{width:9d} | {enc.max_value:12d} | (100 exceeds max)")

    # -------------------------------------------------------------------------
    print("\n6. Streaming Encoding Example")
    print("-" * 50)

    def text_stream():
        chunks = ["The quick ", "brown fox ", "jumps."]
        for chunk in chunks:
            yield chunk

    print("Streaming chunks: ['The quick ', 'brown fox ', 'jumps.']")
    print("\nStream output:")

    for encoded_token in vocab_encoder.encode_streaming(text_stream()):
        print(f"  Position {encoded_token.lucas_brackets}: '{encoded_token.text}' -> ID {encoded_token.token_id}")

    # -------------------------------------------------------------------------
    print("\n7. Byte Packing Example")
    print("-" * 50)

    encoder = TokenEncoder(bit_width=16)

    for token_id in [1, 13, 100]:
        bits = encoder.encode(token_id)
        packed = encoder.pack_to_bytes(bits)
        unpacked = encoder.unpack_from_bytes(packed)
        decoded = encoder.decode(unpacked)

        print(f"ID {token_id:3d}: bits={bits[:8]}... -> bytes={packed.hex()} -> decoded={decoded}")

    print("\n" + "=" * 70)
    print("Demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate()
