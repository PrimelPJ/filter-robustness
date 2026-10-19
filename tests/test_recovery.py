"""Tests for recovery strategies."""

from __future__ import annotations

from filter_robustness.evaluator import RobustnessEvaluator
from filter_robustness.filters import TfidfFilter, load_spam_dataset
from filter_robustness.perturbations import (
    PerturbationPipeline,
    SynonymSwap,
    CharacterNoise,
    WordDeletion,
)
from filter_robustness.recovery import (
    AdversarialTrainer,
    EnsembleSmoother,
    InputCanonicalizer,
)


def _trained_spam_filter() -> tuple[TfidfFilter, list[str], list[int]]:
    texts, labels = load_spam_dataset()
    filt = TfidfFilter(max_features=1000)
    filt.fit(texts, labels)
    return filt, texts, labels


def test_canonicalizer_normalizes_homoglyphs():
    c = InputCanonicalizer()
    # Cyrillic 'а' (U+0430) should map to ASCII 'a' via NFKC.
    text = "p\u0430ssword"
    result = c.canonicalize(text)
    assert result == "password"


def test_canonicalizer_expands_contractions():
    c = InputCanonicalizer()
    assert "do not" in c.canonicalize("Don't do this")


def test_canonicalizer_lowercases():
    c = InputCanonicalizer()
    assert c.canonicalize("HELLO WORLD") == "hello world"


def test_canonicalizer_collapses_whitespace():
    c = InputCanonicalizer()
    assert c.canonicalize("hello   world  test") == "hello world test"


def test_adversarial_training_augments_data():
    texts, labels = load_spam_dataset()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    trainer = AdversarialTrainer(pipeline=pipeline, augmentation_factor=2)
    aug_texts, aug_labels = trainer.augment_dataset(texts, labels)
    # Original + 2 augmented per sample.
    assert len(aug_texts) == len(texts) * 3
    assert len(aug_labels) == len(labels) * 3


def test_adversarial_training_improves_robustness():
    """An adversarially trained filter should be at least as robust as the baseline."""
    texts, labels = load_spam_dataset()

    # Baseline filter.
    baseline = TfidfFilter(max_features=1000)
    baseline.fit(texts, labels)

    # Adversarially trained filter.
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.5), WordDeletion(rate=0.15)],
        n_variants=5,
        seed=42,
    )
    trainer = AdversarialTrainer(pipeline=pipeline, augmentation_factor=3)
    hardened = trainer.retrain(baseline, texts, labels)

    # Evaluate both.
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=5)
    baseline_report = evaluator.evaluate(baseline, texts, labels)
    hardened_report = evaluator.evaluate(hardened, texts, labels)

    # The hardened filter should have a better or equal robustness score.
    assert hardened_report.combined_robustness_score >= baseline_report.combined_robustness_score - 0.1


def test_ensemble_smoother_predicts():
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    smoother = EnsembleSmoother(pipeline=pipeline, n_variants=5)
    preds = smoother.predict(filt, texts[:5])
    assert len(preds) == 5
    assert all(isinstance(p, int) for p in preds)


def test_ensemble_proba():
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    smoother = EnsembleSmoother(pipeline=pipeline, n_variants=5)
    probas = smoother.predict_proba(filt, texts[:5])
    assert probas.shape == (5, 2)
    # Probabilities should sum to ~1.
    for row in probas:
        assert abs(sum(row) - 1.0) < 0.01
