"""Robustness metrics for evaluating filter degradation under perturbation.

Standard classification metrics (accuracy, F1) tell you how a model performs
on data drawn from the same distribution it was trained on.  The metrics here
measure what happens when that assumption breaks -- when an adversary or a
distribution shift rewrites the input while preserving its meaning.

The key quantities:

* **Accuracy drop** -- how much worse the model gets on perturbed vs. clean
  inputs.  A 5% drop from synonym swaps means the model memorised surface
  forms rather than learning the concept.

* **Worst-case accuracy** -- the lowest accuracy across all perturbation
  strategies.  This is the number that matters for security: an adversary
  only needs one strategy that works.

* **Robustness score** -- a single [0, 1] scalar that combines clean accuracy
  and perturbation resistance.  1.0 means the model is equally accurate on
  clean and perturbed inputs; 0.0 means it cannot classify perturbed inputs
  at all.

* **Flip rate** -- fraction of correctly-classified clean inputs whose
  prediction changes under perturbation.  A high flip rate on a specific
  perturbation type localises the fragility.

* **Confidence shift** -- mean change in predicted probability for the true
  class.  A model can maintain accuracy while losing confidence, which
  matters for systems that threshold on probability rather than argmax.
"""

from __future__ import annotations

import numpy as np


def accuracy_drop(clean_acc: float, perturbed_acc: float) -> float:
    """Absolute accuracy drop from clean to perturbed evaluation.

    Returns a non-negative float.  Higher means more fragile.
    """
    return max(clean_acc - perturbed_acc, 0.0)


def relative_accuracy_drop(clean_acc: float, perturbed_acc: float) -> float:
    """Relative accuracy drop, normalised by clean accuracy.

    Returns a value in [0, 1].  A 10% clean accuracy with 5% perturbed
    gives 0.5 (the model lost half its performance).
    """
    if clean_acc <= 0.0:
        return 0.0
    return max((clean_acc - perturbed_acc) / clean_acc, 0.0)


def worst_case_accuracy(per_strategy_accuracies: dict[str, float]) -> tuple[str, float]:
    """Find the perturbation strategy that causes the most damage.

    Returns (strategy_name, accuracy).
    """
    if not per_strategy_accuracies:
        return ("none", 0.0)
    worst = min(per_strategy_accuracies, key=per_strategy_accuracies.get)
    return worst, per_strategy_accuracies[worst]


def robustness_score(clean_acc: float, perturbed_acc: float) -> float:
    """Combined robustness score in [0, 1].

    Defined as ``perturbed_acc / clean_acc`` when clean_acc > 0,
    otherwise 0.  A score of 1.0 means no degradation; 0.0 means
    complete failure under perturbation.
    """
    if clean_acc <= 0.0:
        return 0.0
    return min(perturbed_acc / clean_acc, 1.0)


def flip_rate(
    clean_predictions: list[int],
    perturbed_predictions: list[int],
    true_labels: list[int],
) -> float:
    """Fraction of correctly-classified inputs whose label flips under perturbation.

    Only counts samples where the clean prediction was correct but the
    perturbed prediction is different (whether correct or not).  This
    isolates the perturbation's destabilising effect from the model's
    baseline error.
    """
    n_correct = 0
    n_flipped = 0
    for clean, perturbed, true in zip(clean_predictions, perturbed_predictions, true_labels):
        if clean == true:
            n_correct += 1
            if perturbed != clean:
                n_flipped += 1
    if n_correct == 0:
        return 0.0
    return n_flipped / n_correct


def confidence_shift(
    clean_probas: np.ndarray,
    perturbed_probas: np.ndarray,
    true_labels: list[int],
) -> float:
    """Mean change in confidence for the true class under perturbation.

    A negative value means the model is less confident on perturbed inputs.
    The magnitude indicates how much the probability mass shifted away
    from the correct class.
    """
    shifts: list[float] = []
    for i, label in enumerate(true_labels):
        if label < clean_probas.shape[1] and label < perturbed_probas.shape[1]:
            clean_conf = clean_probas[i, label]
            perturbed_conf = perturbed_probas[i, label]
            shifts.append(perturbed_conf - clean_conf)
    if not shifts:
        return 0.0
    return float(np.mean(shifts))


def per_class_robustness(
    clean_predictions: list[int],
    perturbed_predictions: list[int],
    true_labels: list[int],
    n_classes: int,
) -> dict[int, dict[str, float]]:
    """Per-class accuracy and flip rate breakdown.

    Returns a dict mapping class label to a dict with keys
    ``"clean_acc"``, ``"perturbed_acc"``, ``"flip_rate"``.
    """
    results: dict[int, dict[str, float]] = {}
    for cls in range(n_classes):
        indices = [i for i, l in enumerate(true_labels) if l == cls]
        if not indices:
            results[cls] = {"clean_acc": 0.0, "perturbed_acc": 0.0, "flip_rate": 0.0}
            continue
        clean_correct = sum(1 for i in indices if clean_predictions[i] == cls)
        pert_correct = sum(1 for i in indices if perturbed_predictions[i] == cls)
        flipped = sum(
            1 for i in indices
            if clean_predictions[i] == cls and perturbed_predictions[i] != cls
        )
        n = len(indices)
        results[cls] = {
            "clean_acc": clean_correct / n,
            "perturbed_acc": pert_correct / n,
            "flip_rate": flipped / max(clean_correct, 1),
        }
    return results
