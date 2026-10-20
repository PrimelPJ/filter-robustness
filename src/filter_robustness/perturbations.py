"""Input perturbation strategies for adversarial robustness testing.

Each perturbation simulates a way an adversary (or an innocent user) might
rephrase text that a classifier saw during training.  The goal is not to
fool humans -- it is to find the gap between what a model learned and what
the underlying concept actually is.

Perturbation types, ordered by subtlety:

* **SynonymSwap** -- replace words with synonyms.  This is the canonical
  "rewording" attack: the meaning is preserved but the surface form changes.
* **CharacterNoise** -- typos, homoglyphs, inserted/deleted characters.
  Exploits the tokenizer's fragility rather than the model's semantics.
* **WordDeletion** -- randomly drop words.  Tests whether the model relies
  on a few anchor tokens or understands the full input.
* **WordReorder** -- shuffle word order.  Bag-of-words models are immune;
  sequence models may not be.
* **CasePerturb** -- random capitalization changes.  Cheap and surprisingly
  effective against models that did not lowercase during training.
* **ContractionFlip** -- expand or contract ("don't" <-> "do not").  A
  minimal surface change that alters tokenization significantly.

All perturbations are seeded for reproducibility.
"""

from __future__ import annotations

import random
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Curated synonym map -- small enough to ship without NLTK/WordNet, large
# enough to cover the token distributions that matter in spam, toxicity,
# and sentiment classifiers.
_SYNONYM_MAP: dict[str, list[str]] = {
    "good": ["great", "fine", "excellent", "nice", "positive"],
    "bad": ["terrible", "awful", "poor", "negative", "horrible"],
    "happy": ["glad", "pleased", "joyful", "cheerful", "delighted"],
    "sad": ["unhappy", "sorrowful", "miserable", "gloomy", "depressed"],
    "big": ["large", "huge", "enormous", "massive", "vast"],
    "small": ["tiny", "little", "miniature", "minute", "compact"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely", "plodding"],
    "buy": ["purchase", "acquire", "obtain", "get", "procure"],
    "sell": ["vend", "market", "trade", "offer", "peddle"],
    "free": ["complimentary", "gratis", "costless", "unpaid", "gratuitous"],
    "money": ["cash", "funds", "currency", "capital", "payment"],
    "click": ["press", "tap", "select", "hit", "activate"],
    "win": ["earn", "gain", "receive", "secure", "claim"],
    "offer": ["deal", "proposal", "promotion", "bargain", "opportunity"],
    "urgent": ["immediate", "pressing", "critical", "time-sensitive", "crucial"],
    "love": ["adore", "enjoy", "cherish", "appreciate", "treasure"],
    "hate": ["despise", "loathe", "detest", "abhor", "dislike"],
    "help": ["assist", "aid", "support", "facilitate", "guide"],
    "problem": ["issue", "trouble", "difficulty", "complication", "setback"],
    "easy": ["simple", "straightforward", "effortless", "trivial", "basic"],
    "hard": ["difficult", "tough", "challenging", "demanding", "complex"],
    "new": ["fresh", "recent", "modern", "novel", "latest"],
    "old": ["aged", "ancient", "outdated", "vintage", "former"],
    "start": ["begin", "commence", "initiate", "launch", "kick off"],
    "stop": ["halt", "cease", "end", "terminate", "discontinue"],
    "send": ["transmit", "deliver", "dispatch", "forward", "ship"],
    "get": ["receive", "obtain", "acquire", "retrieve", "fetch"],
    "make": ["create", "build", "produce", "construct", "generate"],
    "use": ["utilize", "employ", "apply", "leverage", "operate"],
    "amazing": ["incredible", "wonderful", "fantastic", "remarkable", "astonishing"],
    "terrible": ["dreadful", "appalling", "atrocious", "ghastly", "abysmal"],
    "important": ["crucial", "vital", "essential", "significant", "critical"],
    "safe": ["secure", "protected", "shielded", "guarded", "harmless"],
    "dangerous": ["hazardous", "risky", "perilous", "unsafe", "threatening"],
    "real": ["genuine", "authentic", "actual", "legitimate", "true"],
    "fake": ["counterfeit", "fraudulent", "bogus", "phony", "sham"],
}

# Homoglyph table for character-level attacks.  Maps ASCII to visually
# similar Unicode characters that survive copy-paste but break tokenizers.
_HOMOGLYPHS: dict[str, str] = {
    "a": "\u0430",  # Cyrillic а
    "e": "\u0435",  # Cyrillic е
    "o": "\u043e",  # Cyrillic о
    "p": "\u0440",  # Cyrillic р
    "c": "\u0441",  # Cyrillic с
    "x": "\u0445",  # Cyrillic х
    "i": "\u0456",  # Cyrillic і
    "s": "\u0455",  # Cyrillic ѕ
}

_CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "can't": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "let's": "let us",
}

# Reverse map: expanded -> contracted.
_EXPANSIONS: dict[str, str] = {v: k for k, v in _CONTRACTIONS.items()}


class Perturbation(ABC):
    """Base class for text perturbation strategies."""

    @abstractmethod
    def apply(self, text: str, rng: random.Random) -> str:
        """Return a perturbed copy of *text*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for reports."""


class SynonymSwap(Perturbation):
    """Replace words with synonyms from the built-in map.

    Parameters
    ----------
    rate : float
        Fraction of eligible words to replace (0.0 to 1.0).
    """

    def __init__(self, rate: float = 0.3) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return f"synonym_swap(rate={self.rate})"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        out: list[str] = []
        for w in words:
            key = w.lower().strip(string.punctuation)
            if key in _SYNONYM_MAP and rng.random() < self.rate:
                replacement = rng.choice(_SYNONYM_MAP[key])
                # Preserve original casing pattern.
                if w[0].isupper():
                    replacement = replacement.capitalize()
                if w.isupper():
                    replacement = replacement.upper()
                # Preserve trailing punctuation.
                trail = ""
                while w and w[-1] in string.punctuation:
                    trail = w[-1] + trail
                    w = w[:-1]
                out.append(replacement + trail)
            else:
                out.append(w)
        return " ".join(out)


class CharacterNoise(Perturbation):
    """Inject character-level noise: typos, homoglyphs, insertions.

    Parameters
    ----------
    rate : float
        Probability of perturbing each character.
    mode : str
        One of ``"typo"``, ``"homoglyph"``, ``"insert"``, ``"delete"``,
        or ``"mixed"`` (picks randomly per character).
    """

    MODES = ("typo", "homoglyph", "insert", "delete", "mixed")

    def __init__(self, rate: float = 0.05, mode: str = "mixed") -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.rate = rate
        self.mode = mode

    @property
    def name(self) -> str:
        return f"char_noise(rate={self.rate}, mode={self.mode})"

    def _pick_mode(self, rng: random.Random) -> str:
        if self.mode == "mixed":
            return rng.choice(["typo", "homoglyph", "insert", "delete"])
        return self.mode

    def apply(self, text: str, rng: random.Random) -> str:
        out: list[str] = []
        for ch in text:
            if ch == " " or rng.random() >= self.rate:
                out.append(ch)
                continue
            m = self._pick_mode(rng)
            if m == "typo" and ch.isalpha():
                # Swap with adjacent keyboard character.
                neighbors = _keyboard_neighbors(ch.lower())
                replacement = rng.choice(neighbors) if neighbors else ch
                out.append(replacement if ch.islower() else replacement.upper())
            elif m == "homoglyph" and ch.lower() in _HOMOGLYPHS:
                out.append(_HOMOGLYPHS[ch.lower()])
            elif m == "insert":
                out.append(ch)
                out.append(rng.choice(string.ascii_lowercase))
            elif m == "delete":
                pass  # drop the character
            else:
                out.append(ch)
        return "".join(out)


class WordDeletion(Perturbation):
    """Randomly delete words from the input.

    Parameters
    ----------
    rate : float
        Probability of deleting each word.
    """

    def __init__(self, rate: float = 0.15) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return f"word_deletion(rate={self.rate})"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        kept = [w for w in words if rng.random() >= self.rate]
        return " ".join(kept) if kept else words[0]


class WordReorder(Perturbation):
    """Shuffle word order within a sliding window.

    A window size of 2 produces adjacent transpositions (mild); the full
    sentence length produces a complete shuffle (severe).

    Parameters
    ----------
    window : int
        Number of consecutive words to shuffle at a time.
    """

    def __init__(self, window: int = 3) -> None:
        self.window = max(2, window)

    @property
    def name(self) -> str:
        return f"word_reorder(window={self.window})"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        out: list[str] = []
        for i in range(0, len(words), self.window):
            chunk = words[i : i + self.window]
            rng.shuffle(chunk)
            out.extend(chunk)
        return " ".join(out)


class CasePerturb(Perturbation):
    """Randomly change character casing.

    Parameters
    ----------
    rate : float
        Probability of flipping case for each alpha character.
    """

    def __init__(self, rate: float = 0.15) -> None:
        self.rate = rate

    @property
    def name(self) -> str:
        return f"case_perturb(rate={self.rate})"

    def apply(self, text: str, rng: random.Random) -> str:
        out: list[str] = []
        for ch in text:
            if ch.isalpha() and rng.random() < self.rate:
                out.append(ch.swapcase())
            else:
                out.append(ch)
        return "".join(out)


class ContractionFlip(Perturbation):
    """Expand contractions or contract expanded forms.

    "don't" becomes "do not" and vice versa.  A minimal surface change
    that significantly alters tokenization: "don't" is one or two tokens
    in most tokenizers, "do not" is always two.
    """

    @property
    def name(self) -> str:
        return "contraction_flip"

    def apply(self, text: str, rng: random.Random) -> str:
        lower = text.lower()
        # Try contracting first (longer patterns match first).
        for expanded, contracted in sorted(_EXPANSIONS.items(), key=lambda x: -len(x[0])):
            if expanded in lower:
                idx = lower.find(expanded)
                original_span = text[idx : idx + len(expanded)]
                # Preserve leading capitalization.
                if original_span[0].isupper():
                    replacement = contracted.capitalize()
                else:
                    replacement = contracted
                text = text[:idx] + replacement + text[idx + len(expanded) :]
                lower = text.lower()
        # Then expand any remaining contractions.
        for contracted, expanded in sorted(_CONTRACTIONS.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(re.escape(contracted), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                original = match.group()
                if original[0].isupper():
                    replacement = expanded.capitalize()
                else:
                    replacement = expanded
                text = text[: match.start()] + replacement + text[match.end() :]
        return text


@dataclass
class PerturbationPipeline:
    """Chain multiple perturbations and generate variants.

    Parameters
    ----------
    perturbations : list[Perturbation]
        Ordered list of perturbation strategies to apply.
    n_variants : int
        Number of perturbed variants to generate per input.
    seed : int
        Base random seed for reproducibility.
    """

    perturbations: list[Perturbation] = field(default_factory=lambda: [
        SynonymSwap(),
        CharacterNoise(),
        WordDeletion(),
    ])
    n_variants: int = 10
    seed: int = 42

    def generate(self, text: str) -> list[str]:
        """Generate ``n_variants`` perturbed versions of *text*."""
        variants: list[str] = []
        for i in range(self.n_variants):
            rng = random.Random(self.seed + i)
            result = text
            for p in self.perturbations:
                result = p.apply(result, rng)
            variants.append(result)
        return variants

    def generate_labeled(self, text: str) -> list[dict[str, str]]:
        """Generate variants with metadata about which perturbations were applied."""
        results: list[dict[str, str]] = []
        for i in range(self.n_variants):
            rng = random.Random(self.seed + i)
            result = text
            applied: list[str] = []
            for p in self.perturbations:
                before = result
                result = p.apply(result, rng)
                if result != before:
                    applied.append(p.name)
            results.append({
                "original": text,
                "perturbed": result,
                "perturbations": ", ".join(applied) if applied else "none",
                "variant_id": i,
            })
        return results

    def generate_per_strategy(self, text: str) -> dict[str, list[str]]:
        """Generate variants for each perturbation strategy independently.

        Returns a dict mapping perturbation name to a list of variants
        produced by that single strategy alone.  Useful for isolating
        which perturbation type causes the most damage to a given filter.
        """
        results: dict[str, list[str]] = {}
        for p in self.perturbations:
            variants: list[str] = []
            for i in range(self.n_variants):
                rng = random.Random(self.seed + i)
                variants.append(p.apply(text, rng))
            results[p.name] = variants
        return results


# ---------------------------------------------------------------- helpers

_KEYBOARD_LAYOUT = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etf", "t": "ryg",
    "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qwsz", "s": "weadxz", "d": "ersfc", "f": "rtdgv",
    "g": "tyfhb", "h": "yugjn", "j": "uihkm", "k": "iojl",
    "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def _keyboard_neighbors(ch: str) -> list[str]:
    return list(_KEYBOARD_LAYOUT.get(ch, ""))


class TextRepeat(Perturbation):
    """Repeat words or characters to evade classifiers.

    Attackers commonly stretch words ("freeee", "buyyy") to dodge
    exact-match and n-gram based filters while remaining human-readable.

    Parameters
    ----------
    rate : float
        Probability of repeating a character in each word.
    max_repeats : int
        Maximum number of extra character repetitions.
    """

    def __init__(self, rate: float = 0.1, max_repeats: int = 3) -> None:
        self.rate = rate
        self.max_repeats = max_repeats

    @property
    def name(self) -> str:
        return f"text_repeat(rate={self.rate})"

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        out: list[str] = []
        for w in words:
            if len(w) > 2 and rng.random() < self.rate:
                idx = rng.randint(1, len(w) - 1)
                repeats = rng.randint(1, self.max_repeats)
                w = w[:idx] + w[idx] * repeats + w[idx:]
            out.append(w)
        return " ".join(out)
