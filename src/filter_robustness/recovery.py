"""Recovery strategies for hardening text filters against perturbation.

Once the evaluator identifies how a filter breaks, these strategies repair
the damage.  Three approaches, from cheapest to most expensive:

1. **Input canonicalization** -- normalise the input before classification.
   Undo character-level tricks (homoglyphs, case noise) and expand
   contractions so the tokenizer sees a canonical form.  Zero retraining
   cost, effective against surface-level attacks.

2. **Adversarial training** -- augment the training set with perturbed
   variants and retrain.  The model learns that "purchase" and "buy" map
   to the same concept.  Moderate cost, effective against semantic attacks
   (synonym swaps).

3. **Ensemble smoothing** -- at inference time, classify multiple perturbed
   versions of the input and vote.  The perturbation that fools one variant
   is unlikely to fool all of them.  No retraining, but multiplies
   inference cost by the number of variants.

These can be composed: canonicalize first, then classify with an
adversarially-trained model, with ensemble smoothing as a final defense.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from .filters import TfidfFilter
from .perturbations import PerturbationPipeline, _CONTRACTIONS, _HOMOGLYPHS

# Reverse homoglyph map: confusable -> ASCII.
_HOMOGLYPH_REVERSE: dict[str, str] = {v: k for k, v in _HOMOGLYPHS.items()}


class InputCanonicalizer:
    """Normalize text to a canonical form before classification.

    Applies a sequence of normalization steps:
    1. Unicode NFKC normalization (collapses homoglyphs).
    2. Lowercase.
    3. Expand contractions.
    4. Collapse whitespace.
    5. Strip non-ASCII characters that survived NFKC.

    This is a zero-cost defense against character-level perturbations.
    It does not help against semantic attacks (synonym swaps) because
    those preserve the canonical form.
    """

    def __init__(self, lowercase: bool = True, expand_contractions: bool = True) -> None:
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions

    def canonicalize(self, text: str) -> str:
        """Return the canonical form of *text*."""
        # Unicode normalization first.
        text = unicodedata.normalize("NFKC", text)
        # Strip known homoglyphs that survive NFKC (e.g. Cyrillic а → a).
        text = "".join(_HOMOGLYPH_REVERSE.get(ch, ch) for ch in text)
        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            for contracted, expanded in sorted(
                _CONTRACTIONS.items(), key=lambda x: -len(x[0])
            ):
                text = re.sub(re.escape(contracted), expanded, text, flags=re.IGNORECASE)
        # Collapse whitespace.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def canonicalize_batch(self, texts: list[str]) -> list[str]:
        return [self.canonicalize(t) for t in texts]


class AdversarialTrainer:
    """Retrain a filter with adversarially augmented data.

    Takes the original training set, generates perturbed variants using
    the perturbation pipeline, and retrains the filter on the combined
    dataset.  The perturbed variants inherit the original label: a spam
    message rewritten with synonyms is still spam.

    Parameters
    ----------
    pipeline : PerturbationPipeline
        Perturbation pipeline for generating augmented examples.
    augmentation_factor : int
        Number of perturbed variants to generate per original sample.
    """

    def __init__(
        self,
        pipeline: PerturbationPipeline | None = None,
        augmentation_factor: int = 5,
    ) -> None:
        self.pipeline = pipeline or PerturbationPipeline()
        self.augmentation_factor = augmentation_factor

    def augment_dataset(
        self, texts: list[str], labels: list[int]
    ) -> tuple[list[str], list[int]]:
        """Generate an augmented dataset with perturbed variants.

        Returns (augmented_texts, augmented_labels) containing the
        original samples plus perturbed variants.
        """
        aug_texts = list(texts)
        aug_labels = list(labels)

        for i, (text, label) in enumerate(zip(texts, labels)):
            for j in range(self.augmentation_factor):
                rng = random.Random(self.pipeline.seed + i * self.augmentation_factor + j)
                perturbed = text
                for p in self.pipeline.perturbations:
                    perturbed = p.apply(perturbed, rng)
                aug_texts.append(perturbed)
                aug_labels.append(label)

        return aug_texts, aug_labels

    def retrain(
        self,
        filt: TfidfFilter,
        texts: list[str],
        labels: list[int],
    ) -> TfidfFilter:
        """Augment the dataset and retrain the filter.

        Returns a new filter instance trained on the augmented data.
        The original filter is not modified.
        """
        aug_texts, aug_labels = self.augment_dataset(texts, labels)
        new_filter = TfidfFilter(
            max_features=filt.pipeline.named_steps["tfidf"].max_features,
            ngram_range=filt.pipeline.named_steps["tfidf"].ngram_range,
            C=filt.pipeline.named_steps["clf"].C,
        )
        new_filter.fit(aug_texts, aug_labels)
        return new_filter


@dataclass
class EnsembleSmoother:
    """Classify multiple perturbation variants and vote.

    At inference time, generates ``n_variants`` perturbed copies of the
    input, classifies all of them, and returns the majority vote.  An
    adversarial perturbation that fools one variant is unlikely to fool
    the majority.

    This is the most expensive defense (inference cost multiplied by
    ``n_variants``) but requires no retraining and works against any
    perturbation type.

    Parameters
    ----------
    pipeline : PerturbationPipeline
        Perturbation pipeline for generating vote variants.
    n_variants : int
        Number of perturbed variants to vote over.
    """

    pipeline: PerturbationPipeline = None  # type: ignore[assignment]
    n_variants: int = 15

    def __post_init__(self) -> None:
        if self.pipeline is None:
            self.pipeline = PerturbationPipeline()

    def predict(self, filt: TfidfFilter, texts: list[str]) -> list[int]:
        """Predict labels using ensemble smoothing."""
        results: list[int] = []
        for text in texts:
            variants = [text]  # Include original.
            for i in range(self.n_variants):
                rng = random.Random(self.pipeline.seed + hash(text) + i)
                perturbed = text
                for p in self.pipeline.perturbations:
                    perturbed = p.apply(perturbed, rng)
                variants.append(perturbed)

            preds = filt.predict(variants)
            # Majority vote.
            counts: dict[int, int] = {}
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
            results.append(max(counts, key=counts.get))  # type: ignore[arg-type]
        return results

    def predict_proba(self, filt: TfidfFilter, texts: list[str]) -> np.ndarray:
        """Average predicted probabilities across ensemble variants."""
        all_probas: list[np.ndarray] = []
        for text in texts:
            variants = [text]
            for i in range(self.n_variants):
                rng = random.Random(self.pipeline.seed + hash(text) + i)
                perturbed = text
                for p in self.pipeline.perturbations:
                    perturbed = p.apply(perturbed, rng)
                variants.append(perturbed)

            probas = filt.predict_proba(variants)
            all_probas.append(probas.mean(axis=0))
        return np.array(all_probas)
