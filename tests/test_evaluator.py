"""Tests for the robustness evaluation harness."""

from __future__ import annotations

from filter_robustness.evaluator import RobustnessEvaluator
from filter_robustness.filters import TfidfFilter, load_spam_dataset
from filter_robustness.perturbations import (
    PerturbationPipeline,
    SynonymSwap,
    CharacterNoise,
    WordDeletion,
)


def _trained_spam_filter() -> tuple[TfidfFilter, list[str], list[int]]:
    texts, labels = load_spam_dataset()
    filt = TfidfFilter(max_features=1000)
    filt.fit(texts, labels)
    return filt, texts, labels


def test_evaluator_produces_report():
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=3)
    report = evaluator.evaluate(filt, texts, labels)

    assert 0.0 <= report.clean_accuracy <= 1.0
    assert 0.0 <= report.combined_perturbed_accuracy <= 1.0
    assert 0.0 <= report.combined_robustness_score <= 1.0
    assert len(report.per_strategy) == 1
    assert report.n_samples == len(texts)


def test_clean_accuracy_high():
    """A TF-IDF model trained and tested on the same data should be near-perfect."""
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=3)
    report = evaluator.evaluate(filt, texts, labels)
    assert report.clean_accuracy > 0.85


def test_perturbation_degrades_accuracy():
    """Perturbed accuracy should be lower than or equal to clean accuracy."""
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[
            SynonymSwap(rate=0.5),
            CharacterNoise(rate=0.1, mode="mixed"),
            WordDeletion(rate=0.2),
        ],
        n_variants=5,
        seed=42,
    )
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=5)
    report = evaluator.evaluate(filt, texts, labels)
    # Combined perturbed accuracy should not exceed clean (on average).
    # We allow a small margin for statistical noise on tiny datasets.
    assert report.combined_perturbed_accuracy <= report.clean_accuracy + 0.05


def test_report_summary_format():
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3)],
        n_variants=3,
        seed=42,
    )
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=3)
    report = evaluator.evaluate(filt, texts, labels)
    summary = report.summary()
    assert "ROBUSTNESS EVALUATION REPORT" in summary
    assert "Clean accuracy" in summary
    assert "Worst strategy" in summary


def test_multiple_strategies_independent():
    """Each strategy should be evaluated independently."""
    filt, texts, labels = _trained_spam_filter()
    pipeline = PerturbationPipeline(
        perturbations=[SynonymSwap(rate=0.3), CharacterNoise(rate=0.05)],
        n_variants=3,
        seed=42,
    )
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=3)
    report = evaluator.evaluate(filt, texts, labels)
    assert len(report.per_strategy) == 2
    names = {s.name for s in report.per_strategy}
    assert len(names) == 2
