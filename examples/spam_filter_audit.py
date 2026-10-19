"""End-to-end robustness audit of a spam filter.

Trains a TF-IDF spam classifier, evaluates its robustness against six
perturbation strategies, then hardens it with adversarial training and
ensemble smoothing, and compares before/after results.

Usage:
    python examples/spam_filter_audit.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from filter_robustness import (
    TfidfFilter,
    RobustnessEvaluator,
    PerturbationPipeline,
    SynonymSwap,
    CharacterNoise,
    WordDeletion,
    WordReorder,
    CasePerturb,
    ContractionFlip,
    AdversarialTrainer,
    EnsembleSmoother,
    InputCanonicalizer,
)
from filter_robustness.filters import load_spam_dataset
from filter_robustness.metrics import accuracy_drop, robustness_score


def main() -> None:
    print("=" * 60)
    print("SPAM FILTER ROBUSTNESS AUDIT")
    print("=" * 60)

    # Load and train.
    texts, labels = load_spam_dataset()
    filt = TfidfFilter(max_features=2000, ngram_range=(1, 2))
    filt.fit(texts, labels)
    print(f"\nTrained on {len(texts)} samples ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")

    # Define perturbation strategies.
    strategies = [
        SynonymSwap(rate=0.3),
        SynonymSwap(rate=0.6),
        CharacterNoise(rate=0.05, mode="mixed"),
        CharacterNoise(rate=0.1, mode="homoglyph"),
        WordDeletion(rate=0.15),
        WordReorder(window=3),
        CasePerturb(rate=0.2),
        ContractionFlip(),
    ]

    pipeline = PerturbationPipeline(
        perturbations=strategies,
        n_variants=10,
        seed=42,
    )

    # Evaluate baseline.
    print("\n--- Baseline Evaluation ---")
    evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=10)
    baseline_report = evaluator.evaluate(filt, texts, labels)
    print(baseline_report.summary())

    # Recovery 1: Input canonicalization.
    print("\n--- Recovery: Input Canonicalization ---")
    canon = InputCanonicalizer()
    canon_texts = canon.canonicalize_batch(texts)
    canon_filt = TfidfFilter(max_features=2000, ngram_range=(1, 2))
    canon_filt.fit(canon_texts, labels)
    # Evaluate with canonicalized inputs.
    canon_report = evaluator.evaluate(canon_filt, canon_texts, labels)
    print(f"Robustness score: {baseline_report.combined_robustness_score:.3f} -> "
          f"{canon_report.combined_robustness_score:.3f}")

    # Recovery 2: Adversarial training.
    print("\n--- Recovery: Adversarial Training ---")
    trainer = AdversarialTrainer(pipeline=pipeline, augmentation_factor=3)
    hardened = trainer.retrain(filt, texts, labels)
    hardened_report = evaluator.evaluate(hardened, texts, labels)
    print(f"Robustness score: {baseline_report.combined_robustness_score:.3f} -> "
          f"{hardened_report.combined_robustness_score:.3f}")

    # Recovery 3: Ensemble smoothing.
    print("\n--- Recovery: Ensemble Smoothing ---")
    smoother = EnsembleSmoother(pipeline=pipeline, n_variants=10)
    smooth_preds = smoother.predict(filt, texts)
    smooth_acc = sum(p == l for p, l in zip(smooth_preds, labels)) / len(labels)
    clean_acc = baseline_report.clean_accuracy
    print(f"Clean accuracy: {clean_acc:.1%}")
    print(f"Ensemble accuracy: {smooth_acc:.1%}")
    print(f"Robustness score: {robustness_score(clean_acc, smooth_acc):.3f}")

    # Summary comparison.
    print("\n" + "=" * 60)
    print("DEFENSE COMPARISON")
    print("=" * 60)
    print(f"{'Defense':<30} {'Robustness Score':>18}")
    print("-" * 60)
    print(f"{'Baseline (no defense)':<30} {baseline_report.combined_robustness_score:>17.3f}")
    print(f"{'Canonicalization':<30} {canon_report.combined_robustness_score:>17.3f}")
    print(f"{'Adversarial training':<30} {hardened_report.combined_robustness_score:>17.3f}")
    print(f"{'Ensemble smoothing':<30} {robustness_score(clean_acc, smooth_acc):>17.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
