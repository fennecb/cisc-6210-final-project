"""
Review Ranking Module - Ranks reviews by relevance to allergen queries.
Uses hybrid approach: keyword-based pre-filtering + TF-IDF ranking.
"""
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ReviewRanker:
    """
    Hybrid review ranker combining keyword-based pre-filtering with TF-IDF.

    This is YOUR information retrieval implementation - demonstrating
    classical NLP techniques for document relevance.
    """

    def __init__(self, max_features: int = 100, allergen_type: str = "gluten"):
        """
        Initialize review ranker.

        Args:
            max_features: Maximum number of TF-IDF features
            allergen_type: Type of allergen to focus on
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.is_fitted = False
        self.allergen_type = allergen_type
        self.allergen_keywords = Config.ALLERGEN_KEYWORDS.get(allergen_type, [])
        self.safety_keywords = Config.SAFETY_KEYWORDS
        self.risk_keywords = Config.CROSS_CONTAMINATION_KEYWORDS
        logger.info(f"Initialized ReviewRanker with max_features={max_features}, "
                   f"allergen_type={allergen_type}")

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

    def compute_keyword_relevance_score(self, review_text: str) -> Tuple[float, Dict[str, int]]:
        """
        Compute keyword-based relevance score for a single review.

        Scoring factors (higher = more relevant):
        - Allergen mentions (+10 per mention)
        - Safety keywords (+15 per mention)
        - Risk keywords (+20 per mention, highest priority)
        - General allergy terms (+5)
        - Review length bonus (capped)

        Args:
            review_text: Review text to score

        Returns:
            Tuple of (relevance_score, keyword_counts_dict)
        """
        if not review_text:
            return 0.0, {}

        text_lower = review_text.lower()
        score = 0.0
        keyword_counts = {
            'allergen_mentions': 0,
            'safety_mentions': 0,
            'risk_mentions': 0,
            'general_allergy_mentions': 0
        }

        # Count allergen-specific keywords
        for keyword in self.allergen_keywords:
            count = text_lower.count(keyword.lower())
            keyword_counts['allergen_mentions'] += count
            score += count * 10

        # Count safety indicators
        for keyword in self.safety_keywords:
            count = text_lower.count(keyword.lower())
            keyword_counts['safety_mentions'] += count
            score += count * 15

        # Count risk indicators (highest value)
        for keyword in self.risk_keywords:
            count = text_lower.count(keyword.lower())
            keyword_counts['risk_mentions'] += count
            score += count * 20

        # General allergen awareness terms
        general_terms = ['allergy', 'allergic', 'allergen', 'celiac', 'intolerance', 'sensitivity']
        for term in general_terms:
            if term in text_lower:
                keyword_counts['general_allergy_mentions'] += 1
                score += 5

        # Length bonus (longer reviews often more detailed)
        length_bonus = min(len(review_text) / 100, 10)
        score += length_bonus

        return score, keyword_counts

    def prefilter_by_keywords(self,
                             reviews: List[Dict],
                             min_keyword_score: float = 1.0) -> List[Dict]:
        """
        Pre-filter reviews using keyword matching before TF-IDF ranking.
        This is useful for quickly identifying allergen-relevant reviews.

        Args:
            reviews: List of review dictionaries with 'text' field
            min_keyword_score: Minimum keyword score to pass filter

        Returns:
            Filtered list of reviews with keyword scores added
        """
        if not reviews:
            return []

        filtered_reviews = []
        for review in reviews:
            text = review.get('text', '')
            score, keyword_counts = self.compute_keyword_relevance_score(text)

            # Add metadata
            enhanced_review = review.copy()
            enhanced_review['keyword_relevance_score'] = score
            enhanced_review['keyword_counts'] = keyword_counts
            enhanced_review['is_keyword_relevant'] = score >= min_keyword_score

            # Only include if passes threshold
            if score >= min_keyword_score:
                filtered_reviews.append(enhanced_review)

        logger.info(f"Keyword pre-filter: {len(reviews)} reviews -> "
                   f"{len(filtered_reviews)} keyword-relevant reviews "
                   f"(min_score={min_keyword_score})")

        return filtered_reviews

    def hybrid_rank_reviews(self,
                           reviews: List[Dict],
                           use_keyword_prefilter: bool = True,
                           keyword_threshold: float = 1.0,
                           tfidf_weight: float = 0.6,
                           keyword_weight: float = 0.4,
                           top_k: int = None) -> List[Dict]:
        """
        Hybrid ranking combining keyword-based and TF-IDF scores.

        Process:
        1. Optional: Pre-filter by keyword relevance
        2. Compute TF-IDF scores
        3. Combine keyword and TF-IDF scores with weights
        4. Return top-k ranked reviews

        Args:
            reviews: List of review dictionaries
            use_keyword_prefilter: Whether to pre-filter by keywords
            keyword_threshold: Minimum keyword score for pre-filter
            tfidf_weight: Weight for TF-IDF score (0-1)
            keyword_weight: Weight for keyword score (0-1)
            top_k: Maximum number of reviews to return

        Returns:
            Ranked list of reviews with combined scores
        """
        if not reviews:
            return []

        # Step 1: Keyword pre-filtering
        if use_keyword_prefilter:
            working_reviews = self.prefilter_by_keywords(reviews, keyword_threshold)
            if not working_reviews:
                logger.warning("No reviews passed keyword filter, using all reviews")
                working_reviews = reviews
        else:
            # Still compute keyword scores for combination
            working_reviews = []
            for review in reviews:
                text = review.get('text', '')
                score, keyword_counts = self.compute_keyword_relevance_score(text)
                enhanced_review = review.copy()
                enhanced_review['keyword_relevance_score'] = score
                enhanced_review['keyword_counts'] = keyword_counts
                working_reviews.append(enhanced_review)

        # Step 2: TF-IDF ranking
        allergen_keywords = self.allergen_keywords + ['allergy', 'allergen', 'celiac']
        tfidf_ranked = self.rank_reviews(working_reviews, allergen_keywords)

        # Step 3: Combine scores
        combined_reviews = []
        for idx, tfidf_score, review in tfidf_ranked:
            keyword_score = review.get('keyword_relevance_score', 0)

            # Normalize keyword score to 0-1 range (assuming max ~100)
            normalized_keyword = min(keyword_score / 100.0, 1.0)

            # Combine with weights
            combined_score = (tfidf_weight * tfidf_score +
                            keyword_weight * normalized_keyword)

            enhanced_review = review.copy()
            enhanced_review['tfidf_score'] = tfidf_score
            enhanced_review['combined_relevance_score'] = combined_score
            enhanced_review['original_index'] = idx

            combined_reviews.append(enhanced_review)

        # Step 4: Sort by combined score
        combined_reviews.sort(key=lambda r: r['combined_relevance_score'], reverse=True)

        # Apply top_k limit
        if top_k:
            combined_reviews = combined_reviews[:top_k]

        logger.info(f"Hybrid ranking: {len(reviews)} total -> {len(combined_reviews)} ranked reviews "
                   f"(tfidf_weight={tfidf_weight}, keyword_weight={keyword_weight})")

        return combined_reviews

    def get_enhanced_review_summary(self, reviews: List[Dict]) -> Dict:
        """
        Generate comprehensive summary with both keyword and TF-IDF statistics.

        Args:
            reviews: List of reviews (can be raw or pre-processed)

        Returns:
            Dictionary with detailed review analysis statistics
        """
        if not reviews:
            return {
                'total_reviews': 0,
                'error': 'No reviews provided'
            }

        # Perform hybrid ranking
        ranked_reviews = self.hybrid_rank_reviews(reviews, use_keyword_prefilter=False)

        # Aggregate keyword counts
        total_allergen = sum(
            r.get('keyword_counts', {}).get('allergen_mentions', 0)
            for r in ranked_reviews
        )
        total_safety = sum(
            r.get('keyword_counts', {}).get('safety_mentions', 0)
            for r in ranked_reviews
        )
        total_risk = sum(
            r.get('keyword_counts', {}).get('risk_mentions', 0)
            for r in ranked_reviews
        )

        # Count highly relevant reviews
        highly_relevant = [r for r in ranked_reviews
                          if r.get('combined_relevance_score', 0) > 0.3]

        return {
            'total_reviews': len(reviews),
            'allergen_relevant_count': len(highly_relevant),
            'total_allergen_mentions': total_allergen,
            'total_safety_mentions': total_safety,
            'total_risk_mentions': total_risk,
            'average_combined_score': (
                np.mean([r.get('combined_relevance_score', 0) for r in ranked_reviews])
                if ranked_reviews else 0
            ),
            'average_tfidf_score': (
                np.mean([r.get('tfidf_score', 0) for r in ranked_reviews])
                if ranked_reviews else 0
            ),
            'average_keyword_score': (
                np.mean([r.get('keyword_relevance_score', 0) for r in ranked_reviews])
                if ranked_reviews else 0
            ),
            'top_5_reviews': [
                {
                    'text_preview': r.get('text', '')[:120] + '...',
                    'combined_score': r.get('combined_relevance_score', 0),
                    'tfidf_score': r.get('tfidf_score', 0),
                    'keyword_score': r.get('keyword_relevance_score', 0),
                    'rating': r.get('rating', 0)
                }
                for r in ranked_reviews[:5]
            ]
        }
