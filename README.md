# filter-robustness

Adversarial robustness testing for text classification filters. Measures how ML-based text classifiers degrade when inputs are reworded, and provides recovery strategies to harden them.

A spam filter that catches `"Buy cheap meds!"` should also catch `"Purchase affordable medications!"` — but most TF-IDF and bag-of-words classifiers don't, because they memorise surface tokens instead of learning the concept. This framework quantifies that gap and closes it.

## What it does

**Perturbation engine** — generates adversarial rewrites of input text through six strategies:

| Strategy | What it does | What it tests |
|---|---|---|
| Synonym swap | Replace words with synonyms | Semantic generalisation |
| Character noise | Typos, homoglyphs, insertions | Tokenizer fragility |
| Word deletion | Randomly drop words | Reliance on anchor tokens |
| Word reorder | Shuffle word order in windows | Bag-of-words assumption |
| Case perturbation | Random capitalisation changes | Preprocessing robustness |
| Contraction flip | "don't" ↔ "do not" | Tokenisation sensitivity |

**Evaluation harness** — runs a trained filter against clean and perturbed inputs, isolating which perturbation type causes which failure mode. Produces per-strategy accuracy drops, flip rates, confidence shifts, and per-class breakdowns.

**Recovery strategies** — three defenses, composable:

1. **Input canonicalisation** — NFKC normalisation, contraction expansion, case folding. Zero retraining cost. Effective against character-level attacks.
2. **Adversarial training** — augment the training set with perturbed variants and retrain. The model learns that "purchase" and "buy" map to the same class.
3. **Ensemble smoothing** — classify multiple perturbation variants and majority-vote. No retraining, but multiplies inference cost.

## Quick start

```bash
pip install -e .
```

Run the full spam filter audit:

```bash
python examples/spam_filter_audit.py
```

```
============================================================
SPAM FILTER ROBUSTNESS AUDIT
============================================================
Trained on 40 samples (20 spam, 20 ham)

--- Baseline Evaluation ---
Clean accuracy:              100.0%
Combined perturbed accuracy: 83.2%
Robustness score:            0.832
Worst strategy:              synonym_swap(rate=0.6) (72.5%)

--- Recovery: Adversarial Training ---
Robustness score: 0.832 -> 0.941

--- Recovery: Ensemble Smoothing ---
Clean accuracy: 100.0%
Ensemble accuracy: 97.5%
```

## Usage

```python
from filter_robustness import (
    TfidfFilter,
    RobustnessEvaluator,
    PerturbationPipeline,
    SynonymSwap,
    CharacterNoise,
    WordDeletion,
    AdversarialTrainer,
)
from filter_robustness.filters import load_spam_dataset

# Train a filter.
texts, labels = load_spam_dataset()
filt = TfidfFilter()
filt.fit(texts, labels)

# Evaluate robustness.
pipeline = PerturbationPipeline(
    perturbations=[SynonymSwap(rate=0.3), CharacterNoise(rate=0.05), WordDeletion(rate=0.15)],
    n_variants=20,
)
evaluator = RobustnessEvaluator(pipeline=pipeline, n_variants=20)
report = evaluator.evaluate(filt, texts, labels)
print(report.summary())

# Harden with adversarial training.
trainer = AdversarialTrainer(pipeline=pipeline, augmentation_factor=5)
hardened = trainer.retrain(filt, texts, labels)
hardened_report = evaluator.evaluate(hardened, texts, labels)
```

## Built-in datasets

Three datasets ship with the module for testing:

- `load_spam_dataset()` — 40 messages, binary (spam/ham)
- `load_toxicity_dataset()` — 30 comments, binary (toxic/clean)
- `load_sentiment_dataset()` — 30 reviews, ternary (positive/neutral/negative)

Each is small enough to fit in source and large enough to train a baseline that gets >90% on clean data but visibly degrades under perturbation. Plug in your own dataset with `filt.fit(your_texts, your_labels)`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project structure

```
src/filter_robustness/
├── perturbations.py   # Six perturbation strategies + pipeline
├── filters.py         # TF-IDF classifier + built-in datasets
├── evaluator.py       # Robustness evaluation harness
├── recovery.py        # Canonicalisation, adversarial training, ensemble smoothing
└── metrics.py         # Accuracy drop, flip rate, confidence shift, robustness score
```

## Stack

`Python` `scikit-learn` `NumPy`
