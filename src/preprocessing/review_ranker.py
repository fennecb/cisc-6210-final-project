"""
Review Ranking Module - Ranks reviews by relevance to allergen queries.
Uses TF-IDF to identify most informative reviews for analysis.
"""
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ReviewRanker:
    """
    TF-IDF based review ranker.

    This is YOUR information retrieval implementation - demonstrating
    classical NLP techniques for document relevance.
    """

    def __init__(self, max_features: int = 100):
        """
        Initialize review ranker.

        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.is_fitted = False
        logger.info(f"Initialized ReviewRanker with max_features={max_features}")

    def rank_reviews(self,
                    reviews: List[Dict],
                    allergen_keywords: List[str],
                    top_k: int = None) -> List[Tuple[int, float, Dict]]:
        """
        Rank reviews by relevance to allergen keywords using TF-IDF.

        Args:
            reviews: List of review dictionaries with 'text' field
            allergen_keywords: Keywords to search for
            top_k: Optional limit on number of reviews to return

        Returns:
            List of tuples: (original_index, relevance_score, review_dict)
            Sorted by relevance score (highest first)
        """
        if not reviews or not allergen_keywords:
            return []

        # Extract review texts
        review_texts = [r.get('text', '') for r in reviews]
        review_texts = [t for t in review_texts if t]  # Remove empty

        if not review_texts:
            return []

        try:
            # Fit TF-IDF on review corpus
            tfidf_matrix = self.vectorizer.fit_transform(review_texts)
            self.is_fitted = True

            # Create query from allergen keywords
            query = " ".join(allergen_keywords)
            query_vector = self.vectorizer.transform([query])

            # Calculate cosine similarity between query and all reviews
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]

            # Create ranked list with original indices
            ranked = []
            for idx, score in enumerate(similarities):
                if idx < len(reviews):
                    ranked.append((idx, float(score), reviews[idx]))

            # Sort by score (descending)
            ranked.sort(key=lambda x: x[1], reverse=True)

            # Apply top_k limit if specified
            if top_k:
                ranked = ranked[:top_k]

            logger.info(f"Ranked {len(review_texts)} reviews, "
                       f"returning top {len(ranked)}")

            return ranked

        except Exception as e:
            logger.error(f"Error ranking reviews: {e}")
            # Return original order with neutral scores
            return [(i, 0.5, r) for i, r in enumerate(reviews)]

    def get_important_terms(self,
                           reviews: List[str],
                           top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract most important terms from reviews using TF-IDF scores.

        Args:
            reviews: List of review texts
            top_n: Number of top terms to return

        Returns:
            List of (term, score) tuples
        """
        if not reviews:
            return []

        try:
            # Fit TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(reviews)

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Calculate average TF-IDF score for each term
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Create term-score pairs
            term_scores = [(feature_names[i], avg_scores[i])
                          for i in range(len(feature_names))]

            # Sort by score and return top N
            term_scores.sort(key=lambda x: x[1], reverse=True)

            return term_scores[:top_n]

        except Exception as e:
            logger.error(f"Error extracting important terms: {e}")
            return []

    def filter_relevant_reviews(self,
                               reviews: List[Dict],
                               allergen_keywords: List[str],
                               min_relevance: float = 0.1,
                               max_reviews: int = 20) -> List[Dict]:
        """
        Filter reviews to only those relevant to allergen safety.

        Args:
            reviews: List of review dictionaries
            allergen_keywords: Keywords to filter by
            min_relevance: Minimum relevance score (0-1)
            max_reviews: Maximum number of reviews to return

        Returns:
            Filtered list of review dictionaries
        """
        ranked = self.rank_reviews(reviews, allergen_keywords)

        # Filter by minimum relevance and limit
        filtered = [
            review for idx, score, review in ranked
            if score >= min_relevance
        ][:max_reviews]

        logger.info(f"Filtered {len(reviews)} reviews down to {len(filtered)} "
                   f"relevant reviews (min_relevance={min_relevance})")

        return filtered

    def analyze_review_coverage(self,
                               reviews: List[Dict],
                               allergen_keywords: List[str]) -> Dict:
        """
        Analyze how well reviews cover allergen-related topics.

        Args:
            reviews: List of review dictionaries
            allergen_keywords: Keywords to check coverage for

        Returns:
            Coverage statistics
        """
        if not reviews:
            return {
                'total_reviews': 0,
                'relevant_reviews': 0,
                'coverage_percentage': 0.0,
                'average_relevance': 0.0
            }

        ranked = self.rank_reviews(reviews, allergen_keywords)

        # Count reviews with non-zero relevance
        relevant_count = sum(1 for _, score, _ in ranked if score > 0.1)

        # Calculate average relevance
        avg_relevance = np.mean([score for _, score, _ in ranked])

        return {
            'total_reviews': len(reviews),
            'relevant_reviews': relevant_count,
            'coverage_percentage': (relevant_count / len(reviews)) * 100,
            'average_relevance': float(avg_relevance),
            'top_relevance_score': float(ranked[0][1]) if ranked else 0.0
        }
