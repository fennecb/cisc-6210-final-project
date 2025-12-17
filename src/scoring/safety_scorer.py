"""
Scoring and Aggregation Module - YOUR ensemble algorithm.
Combines rule-based analysis with LLM reasoning to produce final safety scores.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

from config.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class SafetyAssessment:
    """Final safety assessment for a restaurant."""
    restaurant_name: str
    allergen_type: str

    # LLM-based scores
    overall_safety_score: float  # 0-100, lower is safer (from LLM)
    confidence_score: float  # 0-1 (from LLM)

    # Detailed findings (from LLM)
    risk_factors: List[str]
    safety_indicators: List[str]
    recommended_actions: List[str]
    safe_menu_items: List[str]

    # Metadata
    data_sources_used: List[str]
    reviews_analyzed: int
    menu_items_found: int
    relevant_review_excerpts: List[str]  # NEW: keyword-matched review snippets

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def get_rating(self) -> str:
        """Get human-readable safety rating."""
        if self.overall_safety_score < 20:
            return "VERY SAFE"
        elif self.overall_safety_score < 40:
            return "GENERALLY SAFE"
        elif self.overall_safety_score < 60:
            return "MODERATE RISK"
        elif self.overall_safety_score < 80:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"

class SafetyScorer:
    """
    LLM-based scoring system with review keyword search.
    Simplified to focus on LLM reasoning quality.
    """

    def __init__(self):
        """Initialize scorer."""
        logger.info("Initialized LLM-focused scorer")

    def search_review_keywords(self,
                               reviews: List[Dict],
                               keywords: List[str],
                               context_window: int = 100) -> List[str]:
        """
        Search reviews for specific keywords and extract context.

        Args:
            reviews: List of review dictionaries with 'text' field
            keywords: Keywords to search for (allergen-related terms)
            context_window: Characters of context around each match

        Returns:
            List of relevant review excerpts
        """
        excerpts = []

        for review in reviews:
            text = review.get('text', '')
            if not text:
                continue

            text_lower = text.lower()

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(keyword_lower, start)
                        if pos == -1:
                            break

                        # Extract context
                        context_start = max(0, pos - context_window)
                        context_end = min(len(text), pos + len(keyword) + context_window)
                        excerpt = text[context_start:context_end].strip()

                        # Add ellipsis if truncated
                        if context_start > 0:
                            excerpt = "..." + excerpt
                        if context_end < len(text):
                            excerpt = excerpt + "..."

                        excerpts.append(f"[{keyword}]: {excerpt}")
                        start = pos + 1

        return excerpts
    
    def aggregate_scores(self,
                        llm_response: Optional[object],
                        reviews: List[Dict],
                        menu_items: List[str],
                        restaurant_name: str,
                        allergen_type: str = "gluten") -> SafetyAssessment:
        """
        Create assessment based on LLM response only.
        Simplified to focus on LLM reasoning quality.

        Args:
            llm_response: LLMResponse object (or None)
            reviews: List of reviews for keyword search
            menu_items: List of menu items
            restaurant_name: Restaurant name
            allergen_type: Type of allergen

        Returns:
            SafetyAssessment object
        """
        logger.info(f"Creating LLM-based assessment for {restaurant_name}")

        # Get LLM score (or use neutral if unavailable)
        overall_score = 50.0
        confidence = 0.5
        risk_factors = []
        safety_indicators = []
        safe_alternatives = []
        recommendations = []

        if llm_response:
            overall_score = llm_response.safety_score
            confidence = llm_response.confidence
            risk_factors = llm_response.risk_factors
            safe_alternatives = llm_response.safe_alternatives

            # Generate safety indicators from LLM response
            # (assuming the LLM might provide these)
            if hasattr(llm_response, 'safety_indicators'):
                safety_indicators = llm_response.safety_indicators
            elif hasattr(llm_response, 'reasoning'):
                # Extract positive mentions from reasoning
                reasoning_lower = llm_response.reasoning.lower()
                if 'safe' in reasoning_lower or 'gluten-free' in reasoning_lower:
                    safety_indicators.append("LLM identified safe options")

            # Use LLM recommendations if available
            if hasattr(llm_response, 'recommendations'):
                recommendations = llm_response.recommendations
            else:
                # Generate basic recommendations based on score
                recommendations = self._generate_recommendations(
                    overall_score,
                    risk_factors,
                    safety_indicators,
                    allergen_type
                )
        else:
            logger.warning("LLM response not available - cannot generate assessment")
            recommendations = ["Unable to assess safety without LLM analysis"]

        # Search reviews for relevant keywords
        allergen_keywords = Config.ALLERGEN_KEYWORDS.get(allergen_type, [])
        safety_keywords = Config.SAFETY_KEYWORDS
        search_keywords = allergen_keywords + safety_keywords + ['allergen', 'allergy', 'celiac']

        review_excerpts = self.search_review_keywords(
            reviews,
            search_keywords,
            context_window=150
        )

        # Determine data sources used
        data_sources = []
        if llm_response:
            data_sources.append('llm_reasoning')
        if reviews:
            data_sources.append('reviews')
        if menu_items:
            data_sources.append('menu')

        # Create assessment
        assessment = SafetyAssessment(
            restaurant_name=restaurant_name,
            allergen_type=allergen_type,
            overall_safety_score=overall_score,
            confidence_score=confidence,
            risk_factors=risk_factors,
            safety_indicators=safety_indicators,
            recommended_actions=recommendations,
            safe_menu_items=safe_alternatives,
            data_sources_used=data_sources,
            reviews_analyzed=len(reviews),
            menu_items_found=len(menu_items),
            relevant_review_excerpts=review_excerpts[:10]  # Limit to 10 most relevant
        )

        logger.info(f"Final safety score (LLM): {overall_score:.1f}/100 ({assessment.get_rating()})")
        logger.info(f"Confidence (LLM): {confidence:.2f}")
        logger.info(f"Found {len(review_excerpts)} relevant review excerpts")

        return assessment
    
    def _generate_recommendations(self,
                                 safety_score: float,
                                 risk_factors: List[str],
                                 safety_indicators: List[str],
                                 allergen_type: str) -> List[str]:
        """
        Generate actionable recommendations based on score.
        
        Args:
            safety_score: Overall safety score
            risk_factors: List of identified risks
            safety_indicators: List of safety indicators
            allergen_type: Type of allergen
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if safety_score < 30:
            recommendations.append(f"This restaurant appears relatively safe for {allergen_type} allergies")
            if safety_indicators:
                recommendations.append("Look for menu items marked as allergen-free")
        
        elif safety_score < 60:
            recommendations.append(f"Exercise caution - moderate risk for {allergen_type} allergies")
            recommendations.append("Speak with server/manager about allergen handling")
            recommendations.append("Ask about dedicated preparation areas")
        
        else:
            recommendations.append(f"HIGH RISK - Not recommended for severe {allergen_type} allergies")
            recommendations.append("Consider alternative restaurants")
            
            if risk_factors:
                recommendations.append("Key concerns: " + ", ".join(risk_factors[:2]))
        
        # Add general recommendations
        if not safety_indicators:
            recommendations.append("No explicit allergen-free mentions found in reviews")
        
        return recommendations
    
    def export_assessment(self, assessment: SafetyAssessment, filepath: str) -> bool:
        """
        Export assessment to JSON file.
        
        Args:
            assessment: SafetyAssessment object
            filepath: Output file path
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(assessment.to_dict(), f, indent=2)
            logger.info(f"Exported assessment to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting assessment: {e}")
            return False
