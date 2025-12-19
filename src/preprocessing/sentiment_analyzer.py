"""
Sentiment Analysis Module - Analyzes sentiment of allergen mentions.
This helps distinguish helpful safety information from casual mentions.
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
from textblob import TextBlob

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    is_positive: bool
    is_negative: bool
    is_neutral: bool


class SentimentAnalyzer:
    """
    Sentiment analyzer for review text.

    Analyzes emotional tone to better understand if allergen mentions
    indicate safety or risk.
    """

    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("Initialized SentimentAnalyzer")

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        if not text:
            return SentimentResult(
                text="",
                polarity=0.0,
                subjectivity=0.0,
                is_positive=False,
                is_negative=False,
                is_neutral=True
            )

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        is_positive = polarity > 0.1
        is_negative = polarity < -0.1
        is_neutral = -0.1 <= polarity <= 0.1

        return SentimentResult(
            text=text,
            polarity=polarity,
            subjectivity=subjectivity,
            is_positive=is_positive,
            is_negative=is_negative,
            is_neutral=is_neutral
        )

    def analyze_allergen_context(self,
                                 text: str,
                                 allergen_mention_position: int,
                                 window: int = 50) -> Dict:
        """
        Analyze sentiment around a specific allergen mention.

        Args:
            text: Full review text
            allergen_mention_position: Position of allergen mention
            window: Characters to include on each side

        Returns:
            Dictionary with sentiment analysis
        """
        # Extract context around mention
        start = max(0, allergen_mention_position - window)
        end = min(len(text), allergen_mention_position + window)
        context = text[start:end]

        # Analyze context sentiment
        result = self.analyze_text(context)

        return {
            'context': context,
            'polarity': result.polarity,
            'subjectivity': result.subjectivity,
            'interpretation': self._interpret_allergen_sentiment(result.polarity)
        }

    def _interpret_allergen_sentiment(self, polarity: float) -> str:
        """
        Interpret what sentiment means for allergen safety.

        Positive sentiment around allergens can mean:
        - "Great gluten-free options" (SAFE)
        - "Love their bread" (UNSAFE)

        Negative sentiment can mean:
        - "No gluten-free options" (UNSAFE)
        - "They avoid cross-contamination" (SAFE)

        This requires contextual understanding!
        """
        if polarity > 0.3:
            return "POSITIVE_TONE"
        elif polarity < -0.3:
            return "NEGATIVE_TONE"
        else:
            return "NEUTRAL_TONE"

    def batch_analyze_reviews(self,
                              reviews: List[Dict]) -> Dict:
        """
        Analyze sentiment across multiple reviews.

        Args:
            reviews: List of review dictionaries with 'text' field

        Returns:
            Aggregated sentiment statistics
        """
        if not reviews:
            return {
                'average_polarity': 0.0,
                'average_subjectivity': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }

        polarities = []
        subjectivities = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for review in reviews:
            text = review.get('text', '')
            if not text:
                continue

            result = self.analyze_text(text)
            polarities.append(result.polarity)
            subjectivities.append(result.subjectivity)

            if result.is_positive:
                positive_count += 1
            elif result.is_negative:
                negative_count += 1
            else:
                neutral_count += 1

        return {
            'average_polarity': sum(polarities) / len(polarities) if polarities else 0.0,
            'average_subjectivity': sum(subjectivities) / len(subjectivities) if subjectivities else 0.0,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_analyzed': len(polarities)
        }

    def calculate_review_credibility(self,
                                     polarity: float,
                                     subjectivity: float) -> float:
        """
        Estimate review credibility based on sentiment characteristics.

        More objective reviews (low subjectivity) are generally more credible
        for factual allergen safety information.

        Args:
            polarity: Sentiment polarity
            subjectivity: Sentiment subjectivity

        Returns:
            Credibility score (0-1)
        """
        # Lower subjectivity = higher credibility for safety info
        objectivity_score = 1.0 - subjectivity

        # Extreme polarities might indicate emotional rather than factual reviews
        polarity_penalty = abs(polarity) * 0.2

        credibility = max(0.0, objectivity_score - polarity_penalty)

        return credibility
