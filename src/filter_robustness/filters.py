"""Text classification filters used as evaluation targets.

Each filter wraps a scikit-learn pipeline and exposes a uniform interface:
``fit(texts, labels)`` and ``predict(texts) -> labels``.  The point is not
to build state-of-the-art classifiers -- it is to have realistic-enough
baselines that the robustness evaluation reveals genuine failure modes.

Three built-in datasets ship with the module:

* **spam** -- short messages, binary (spam / ham).
* **toxicity** -- social media comments, binary (toxic / clean).
* **sentiment** -- product reviews, ternary (positive / neutral / negative).

Each is small enough to fit in source and large enough to train a TF-IDF
logistic regression that gets >90% on clean data but visibly degrades
under perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class FilterResult:
    """Prediction result with confidence."""
    label: int
    confidence: float
    probabilities: np.ndarray


class TfidfFilter:
    """TF-IDF + Logistic Regression text classifier.

    This is the baseline filter architecture.  It is fast, interpretable,
    and representative of the classifiers actually deployed in production
    email filters, comment moderation, and content safety systems.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size for the TF-IDF vectorizer.
    ngram_range : tuple[int, int]
        N-gram range for feature extraction.
    C : float
        Regularization strength for logistic regression.
    lowercase : bool
        Whether to lowercase inputs during vectorization.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        lowercase: bool = True,
    ) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=lowercase,
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
            )),
        ])
        self._fitted = False

    def fit(self, texts: list[str], labels: list[int]) -> None:
        """Train the filter on labeled text data."""
        self.pipeline.fit(texts, labels)
        self._fitted = True

    def predict(self, texts: list[str]) -> list[int]:
        """Predict labels for a list of texts."""
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities for a list of texts."""
        return self.pipeline.predict_proba(texts)

    def predict_detail(self, text: str) -> FilterResult:
        """Predict a single text with full detail."""
        proba = self.pipeline.predict_proba([text])[0]
        label = int(np.argmax(proba))
        return FilterResult(
            label=label,
            confidence=float(proba[label]),
            probabilities=proba,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------- datasets

def load_spam_dataset() -> tuple[list[str], list[int]]:
    """Minimal spam/ham dataset for testing.

    Returns (texts, labels) where 1 = spam, 0 = ham.
    """
    data = [
        # Spam
        ("Congratulations! You've won a free iPhone. Click here to claim now!", 1),
        ("URGENT: Your account has been compromised. Send your password immediately.", 1),
        ("Buy cheap medications online! No prescription needed. Order today!", 1),
        ("You've been selected for a $1000 gift card. Act now before it expires!", 1),
        ("Make money fast! Work from home and earn $5000 per week guaranteed!", 1),
        ("Free trial offer! Sign up now and get premium access for 30 days!", 1),
        ("Your loan has been approved! Click the link to receive your funds.", 1),
        ("Hot singles in your area want to meet you tonight! Join free!", 1),
        ("Limited time offer: 90% off designer watches. Buy now!", 1),
        ("You are the lucky winner of our sweepstakes! Claim your prize!", 1),
        ("Lose weight fast with this one weird trick doctors hate!", 1),
        ("Get rich quick with cryptocurrency! Invest $100 and make millions!", 1),
        ("Your computer has a virus! Download our security software now!", 1),
        ("Exclusive deal just for you! Buy one get one free on everything!", 1),
        ("Final notice: your subscription is about to expire. Renew with discount!", 1),
        ("Double your money in 24 hours! No risk guaranteed returns!", 1),
        ("Free vacation package to Hawaii! Just complete a short survey!", 1),
        ("URGENT: IRS notification. You owe back taxes. Pay immediately to avoid arrest.", 1),
        ("Congratulations! You qualified for a low-interest credit card. Apply now!", 1),
        ("Secret method to earn passive income revealed! Click to learn more!", 1),
        # Ham
        ("Hey, are we still meeting for lunch tomorrow at noon?", 0),
        ("The quarterly report is attached. Let me know if you have questions.", 0),
        ("Can you pick up groceries on your way home? We need milk and bread.", 0),
        ("Great presentation today. The team really liked your analysis.", 0),
        ("I'll be 10 minutes late to the meeting. Traffic is bad.", 0),
        ("Did you see the game last night? What an incredible finish!", 0),
        ("The doctor's appointment is scheduled for Thursday at 3pm.", 0),
        ("Happy birthday! Hope you have a wonderful day.", 0),
        ("The project deadline has been moved to next Friday.", 0),
        ("Thanks for sending the files. I'll review them this afternoon.", 0),
        ("Reminder: team standup is at 9:30 tomorrow morning.", 0),
        ("Just finished reading that book you recommended. Really enjoyed it.", 0),
        ("The kids have soccer practice at 4. Can you drive them?", 0),
        ("Let's schedule a code review for the authentication module.", 0),
        ("I pushed the fix for the login bug. Can you test it?", 0),
        ("The conference call with the client is at 2pm EST.", 0),
        ("Dinner reservation is at 7:30 at the Italian place downtown.", 0),
        ("Could you review the pull request when you get a chance?", 0),
        ("The flight lands at 6:15pm. I'll take a cab from the airport.", 0),
        ("Meeting notes from today are in the shared drive. Please review.", 0),
    ]
    texts, labels = zip(*data)
    return list(texts), list(labels)


def load_toxicity_dataset() -> tuple[list[str], list[int]]:
    """Minimal toxicity dataset for testing.

    Returns (texts, labels) where 1 = toxic, 0 = clean.
    """
    data = [
        # Toxic
        ("You're an absolute idiot and everyone knows it.", 1),
        ("This is the stupidest thing I've ever read. Delete your account.", 1),
        ("What a pathetic loser. Nobody cares about your opinion.", 1),
        ("Shut up you moron. You don't know what you're talking about.", 1),
        ("This product is garbage and the people who made it are incompetent.", 1),
        ("You should be ashamed of yourself for writing this trash.", 1),
        ("Only a complete fool would believe this nonsense.", 1),
        ("Go away, nobody wants you here. You're worthless.", 1),
        ("This is disgusting. The author is clearly brain dead.", 1),
        ("Worst article ever written. The writer should be fired immediately.", 1),
        ("You're so dumb it actually hurts to read your comments.", 1),
        ("What an embarrassment. Absolutely pathetic attempt.", 1),
        ("Get lost you clown. Your ideas are laughably stupid.", 1),
        ("This is pure garbage. Anyone who agrees is an idiot.", 1),
        ("You have zero talent. Stop embarrassing yourself.", 1),
        # Clean
        ("I respectfully disagree with the main argument presented here.", 0),
        ("Great analysis! I learned something new from this article.", 0),
        ("Could you explain the methodology in more detail?", 0),
        ("I think there might be an error in the third paragraph.", 0),
        ("This is a well-researched piece. Thanks for sharing.", 0),
        ("I have a different perspective on this issue that I'd like to share.", 0),
        ("The data visualization in figure 3 is particularly effective.", 0),
        ("Has anyone tried implementing this approach in production?", 0),
        ("Interesting point. I hadn't considered that angle before.", 0),
        ("The conclusion seems strong but I'd like to see more evidence.", 0),
        ("Thanks for the detailed writeup. Very helpful for my research.", 0),
        ("I appreciate the balanced view presented in this discussion.", 0),
        ("This confirms what we observed in our own experiments.", 0),
        ("Would it be possible to share the source code for the benchmarks?", 0),
        ("Good contribution to the ongoing discussion about this topic.", 0),
    ]
    texts, labels = zip(*data)
    return list(texts), list(labels)


def load_sentiment_dataset() -> tuple[list[str], list[int]]:
    """Minimal sentiment dataset for testing.

    Returns (texts, labels) where 2 = positive, 1 = neutral, 0 = negative.
    """
    data = [
        # Positive
        ("Absolutely love this product! Best purchase I've made all year.", 2),
        ("The customer service was outstanding. They solved my issue immediately.", 2),
        ("This software is incredibly well designed and easy to use.", 2),
        ("Five stars! Exceeded all my expectations. Highly recommend.", 2),
        ("Amazing quality for the price. Will definitely buy again.", 2),
        ("The team did a fantastic job on this release. Very impressive.", 2),
        ("So happy with my purchase. It works exactly as advertised.", 2),
        ("This is the best restaurant in town. The food was phenomenal.", 2),
        ("Great experience from start to finish. Couldn't be happier.", 2),
        ("The new update is a huge improvement. Love the new features.", 2),
        # Neutral
        ("The product arrived on time and matches the description.", 1),
        ("It works as expected. Nothing special but gets the job done.", 1),
        ("Average quality. Some features are good, others need improvement.", 1),
        ("The service was adequate. No complaints but nothing stood out.", 1),
        ("Decent product for the price point. Has pros and cons.", 1),
        ("It's okay. Does what it says but I've seen better options.", 1),
        ("Standard shipping took the expected amount of time.", 1),
        ("The instructions were clear enough to follow without issues.", 1),
        ("Middle of the road experience. Neither great nor terrible.", 1),
        ("Functional but basic. Would consider other options next time.", 1),
        # Negative
        ("Terrible product. Broke after two days. Complete waste of money.", 0),
        ("The worst customer service I've ever experienced. Rude and unhelpful.", 0),
        ("Do not buy this. It's cheaply made and falls apart immediately.", 0),
        ("Extremely disappointed. Nothing like what was advertised.", 0),
        ("Regret this purchase. Poor quality and even worse support.", 0),
        ("The software crashes constantly. Completely unusable.", 0),
        ("One star. Arrived damaged and the company refused a refund.", 0),
        ("Awful experience. Long wait times and the problem wasn't fixed.", 0),
        ("Save your money. This is the worst product in its category.", 0),
        ("Frustrated beyond belief. Three attempts and still doesn't work.", 0),
    ]
    texts, labels = zip(*data)
    return list(texts), list(labels)
