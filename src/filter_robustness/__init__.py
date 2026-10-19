"""Adversarial robustness testing for text classification filters.

Measures how ML-based text filters degrade under input perturbations
(synonym swaps, character noise, word deletion, reordering) and provides
recovery strategies (adversarial training, input canonicalization,
ensemble smoothing) to harden them.
"""

from .perturbations import (
    PerturbationPipeline,
    CharacterNoise,
    SynonymSwap,
    WordDeletion,
    WordReorder,
    CasePerturb,
    ContractionFlip,
)
from .evaluator import RobustnessEvaluator, RobustnessReport
from .recovery import AdversarialTrainer, InputCanonicalizer, EnsembleSmoother
from .filters import TfidfFilter
from .metrics import accuracy_drop, worst_case_accuracy, robustness_score

__all__ = [
    "PerturbationPipeline",
    "CharacterNoise",
    "SynonymSwap",
    "WordDeletion",
    "WordReorder",
    "CasePerturb",
    "ContractionFlip",
    "RobustnessEvaluator",
    "RobustnessReport",
    "AdversarialTrainer",
    "InputCanonicalizer",
    "EnsembleSmoother",
    "TfidfFilter",
    "accuracy_drop",
    "worst_case_accuracy",
    "robustness_score",
]
