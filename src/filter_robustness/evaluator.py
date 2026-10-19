"""Robustness evaluation harness.

The evaluator takes a fitted filter and a test set, generates perturbed
variants of every input, runs the filter on both clean and perturbed data,
and produces a structured report that isolates which perturbation types
cause which failure modes.

Usage::

    from filter_robustness import TfidfFilter, RobustnessEvaluator, PerturbationPipeline

    filt = TfidfFilter()
    filt.fit(train_texts, train_labels)

    evaluator = RobustnessEvaluator(
        pipeline=PerturbationPipeline(),
        n_variants=20,
    )
    report = evaluator.evaluate(filt, test_texts, test_labels)
    report.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import TfidfFilter
from .metrics import (
    accuracy_drop,
    confidence_shift,
    flip_rate,
    per_class_robustness,
    robustness_score,
    worst_case_accuracy,
)
from .perturbations import Perturbation, PerturbationPipeline


@dataclass
class StrategyReport:
    """Evaluation results for a single perturbation strategy."""
    name: str
    perturbed_accuracy: float
    accuracy_drop: float
    flip_rate: float
    confidence_shift: float
    robustness_score: float
    n_samples: int


@dataclass
class RobustnessReport:
    """Full evaluation report across all perturbation strategies."""
    clean_accuracy: float
    combined_perturbed_accuracy: float
    combined_robustness_score: float
    worst_strategy: str
    worst_accuracy: float
    per_strategy: list[StrategyReport] = field(default_factory=list)
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)
    n_samples: int = 0
    n_variants_per_sample: int = 0

    def summary(self) -> str:
        """Human-readable summary of the evaluation."""
        lines: list[str] = [
            "=" * 60,
            "ROBUSTNESS EVALUATION REPORT",
            "=" * 60,
            f"Samples: {self.n_samples}  |  Variants per sample: {self.n_variants_per_sample}",
            f"Clean accuracy:              {self.clean_accuracy:.1%}",
            f"Combined perturbed accuracy: {self.combined_perturbed_accuracy:.1%}",
            f"Robustness score:            {self.combined_robustness_score:.3f}",
            f"Worst strategy:              {self.worst_strategy} ({self.worst_accuracy:.1%})",
            "",
            "Per-strategy breakdown:",
            "-" * 60,
            f"{'Strategy':<35} {'Acc':>6} {'Drop':>6} {'Flip':>6} {'Score':>6}",
            "-" * 60,
        ]
        for s in sorted(self.per_strategy, key=lambda x: x.perturbed_accuracy):
            lines.append(
                f"{s.name:<35} {s.perturbed_accuracy:>5.1%} "
                f"{s.accuracy_drop:>5.1%} {s.flip_rate:>5.1%} "
                f"{s.robustness_score:>5.3f}"
            )
        if self.per_class:
            lines.append("")
            lines.append("Per-class breakdown:")
            lines.append("-" * 60)
            for cls in sorted(self.per_class):
                info = self.per_class[cls]
                lines.append(
                    f"  Class {cls}: clean={info['clean_acc']:.1%}  "
                    f"perturbed={info['perturbed_acc']:.1%}  "
                    f"flip_rate={info['flip_rate']:.1%}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


class RobustnessEvaluator:
    """Evaluate a text filter's robustness against input perturbations.

    Parameters
    ----------
    pipeline : PerturbationPipeline
        Perturbation pipeline to generate input variants.
    n_variants : int
        Number of perturbed variants per input per strategy.
    """

    def __init__(
        self,
        pipeline: PerturbationPipeline | None = None,
        n_variants: int = 10,
    ) -> None:
        self.pipeline = pipeline or PerturbationPipeline()
        self.n_variants = n_variants

    def evaluate(
        self,
        filt: TfidfFilter,
        texts: list[str],
        labels: list[int],
    ) -> RobustnessReport:
        """Run the full robustness evaluation.

        Evaluates the filter on clean data, then generates perturbed
        variants using each strategy independently and measures
        degradation.
        """
        # Clean evaluation.
        clean_preds = filt.predict(texts)
        clean_acc = sum(p == l for p, l in zip(clean_preds, labels)) / len(labels)
        clean_probas = filt.predict_proba(texts)

        # Per-strategy evaluation.
        strategy_reports: list[StrategyReport] = []
        all_perturbed_preds: list[int] = []
        all_perturbed_probas: list[np.ndarray] = []
        all_perturbed_labels: list[int] = []

        n_classes = len(set(labels))

        for perturbation in self.pipeline.perturbations:
            strat_preds, strat_probas, strat_labels = self._evaluate_strategy(
                filt, texts, labels, perturbation
            )
            strat_acc = sum(
                p == l for p, l in zip(strat_preds, strat_labels)
            ) / len(strat_labels)

            # Build clean predictions expanded to match perturbed count.
            expanded_clean_preds = []
            expanded_clean_probas = []
            for i in range(len(texts)):
                for _ in range(self.n_variants):
                    expanded_clean_preds.append(clean_preds[i])
                    expanded_clean_probas.append(clean_probas[i])

            strat_flip = flip_rate(expanded_clean_preds, strat_preds, strat_labels)
            strat_conf = confidence_shift(
                np.array(expanded_clean_probas),
                np.array(strat_probas) if strat_probas else np.zeros((0, n_classes)),
                strat_labels,
            )

            strategy_reports.append(StrategyReport(
                name=perturbation.name,
                perturbed_accuracy=strat_acc,
                accuracy_drop=accuracy_drop(clean_acc, strat_acc),
                flip_rate=strat_flip,
                confidence_shift=strat_conf,
                robustness_score=robustness_score(clean_acc, strat_acc),
                n_samples=len(strat_labels),
            ))

            all_perturbed_preds.extend(strat_preds)
            all_perturbed_labels.extend(strat_labels)
            if strat_probas:
                all_perturbed_probas.extend(strat_probas)

        # Combined metrics.
        if all_perturbed_labels:
            combined_acc = sum(
                p == l for p, l in zip(all_perturbed_preds, all_perturbed_labels)
            ) / len(all_perturbed_labels)
        else:
            combined_acc = 0.0

        per_strat_accs = {s.name: s.perturbed_accuracy for s in strategy_reports}
        worst_name, worst_acc = worst_case_accuracy(per_strat_accs)

        # Per-class breakdown using combined perturbation results.
        expanded_clean_all = []
        for i in range(len(texts)):
            for _ in range(self.n_variants * len(self.pipeline.perturbations)):
                expanded_clean_all.append(clean_preds[i])

        per_cls = per_class_robustness(
            expanded_clean_all, all_perturbed_preds,
            all_perturbed_labels, n_classes,
        )

        return RobustnessReport(
            clean_accuracy=clean_acc,
            combined_perturbed_accuracy=combined_acc,
            combined_robustness_score=robustness_score(clean_acc, combined_acc),
            worst_strategy=worst_name,
            worst_accuracy=worst_acc,
            per_strategy=strategy_reports,
            per_class=per_cls,
            n_samples=len(texts),
            n_variants_per_sample=self.n_variants,
        )

    def _evaluate_strategy(
        self,
        filt: TfidfFilter,
        texts: list[str],
        labels: list[int],
        perturbation: Perturbation,
    ) -> tuple[list[int], list[np.ndarray], list[int]]:
        """Evaluate a single perturbation strategy."""
        import random

        all_perturbed: list[str] = []
        all_labels: list[int] = []

        for i, (text, label) in enumerate(zip(texts, labels)):
            for j in range(self.n_variants):
                rng = random.Random(self.pipeline.seed + i * self.n_variants + j)
                perturbed = perturbation.apply(text, rng)
                all_perturbed.append(perturbed)
                all_labels.append(label)

        if not all_perturbed:
            return [], [], []

        preds = filt.predict(all_perturbed)
        probas = filt.predict_proba(all_perturbed).tolist()

        return preds, probas, all_labels
