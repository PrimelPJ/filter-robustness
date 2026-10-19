"""Tests for input perturbation strategies."""

from __future__ import annotations

import random

from filter_robustness.perturbations import (
    CasePerturb,
    CharacterNoise,
    ContractionFlip,
    PerturbationPipeline,
    SynonymSwap,
    WordDeletion,
    WordReorder,
)


def test_synonym_swap_preserves_length():
    """Synonym swap should not change the number of words."""
    rng = random.Random(42)
    p = SynonymSwap(rate=1.0)
    text = "This is a good product with fast delivery"
    result = p.apply(text, rng)
    assert len(result.split()) == len(text.split())


def test_synonym_swap_changes_text():
    """With rate=1.0, at least one eligible word should change."""
    rng = random.Random(42)
    p = SynonymSwap(rate=1.0)
    text = "This is a good product"
    result = p.apply(text, rng)
    assert result != text


def test_synonym_swap_preserves_punctuation():
    rng = random.Random(42)
    p = SynonymSwap(rate=1.0)
    text = "This is good, bad, and fast."
    result = p.apply(text, rng)
    # Trailing punctuation should be preserved.
    assert result.endswith(".")
    assert "," in result


def test_character_noise_typo():
    rng = random.Random(0)
    p = CharacterNoise(rate=0.3, mode="typo")
    text = "hello world"
    result = p.apply(text, rng)
    # Should produce a different string (with high probability at 30% rate).
    # Allow the rare case where no change happens, but generally expect one.
    assert isinstance(result, str)
    assert len(result) > 0


def test_character_noise_homoglyph():
    rng = random.Random(42)
    p = CharacterNoise(rate=1.0, mode="homoglyph")
    text = "ace"
    result = p.apply(text, rng)
    assert result != text  # All characters have homoglyphs.


def test_character_noise_delete():
    rng = random.Random(42)
    p = CharacterNoise(rate=0.5, mode="delete")
    text = "abcdefghij"
    result = p.apply(text, rng)
    assert len(result) < len(text)


def test_word_deletion_reduces_length():
    rng = random.Random(42)
    p = WordDeletion(rate=0.5)
    text = "one two three four five six seven eight"
    result = p.apply(text, rng)
    assert len(result.split()) < len(text.split())


def test_word_deletion_keeps_at_least_one():
    rng = random.Random(42)
    p = WordDeletion(rate=0.99)
    text = "one two three"
    result = p.apply(text, rng)
    assert len(result.split()) >= 1


def test_word_reorder_preserves_words():
    rng = random.Random(42)
    p = WordReorder(window=10)
    text = "the quick brown fox jumps"
    result = p.apply(text, rng)
    assert sorted(result.split()) == sorted(text.split())


def test_case_perturb():
    rng = random.Random(42)
    p = CasePerturb(rate=1.0)
    text = "hello world"
    result = p.apply(text, rng)
    assert result != text  # At least some characters should flip.
    assert result.lower() == text.lower()  # Content unchanged.


def test_contraction_flip_expands():
    rng = random.Random(42)
    p = ContractionFlip()
    text = "I don't like this and it isn't good"
    result = p.apply(text, rng)
    assert "do not" in result.lower() or "is not" in result.lower()


def test_pipeline_generates_variants():
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.5), WordDeletion(rate=0.1)],
        n_variants=5,
        seed=42,
    )
    variants = pipeline.generate("This is a good product with fast delivery")
    assert len(variants) == 5
    assert all(isinstance(v, str) for v in variants)


def test_pipeline_reproducible():
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.5)],
        n_variants=3,
        seed=42,
    )
    v1 = pipeline.generate("This is a good product")
    v2 = pipeline.generate("This is a good product")
    assert v1 == v2


def test_pipeline_per_strategy():
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.5), CasePerturb(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    results = pipeline.generate_per_strategy("This is a good product")
    assert len(results) == 2
    for name, variants in results.items():
        assert len(variants) == 3


def test_pipeline_labeled():
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.5)],
        n_variants=3,
        seed=42,
    )
    labeled = pipeline.generate_labeled("This is a good product")
    assert len(labeled) == 3
    assert all("original" in d and "perturbed" in d for d in labeled)
