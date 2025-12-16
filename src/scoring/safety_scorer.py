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
    
    # Final scores
    overall_safety_score: float  # 0-100, lower is safer
    confidence_score: float  # 0-1
    
    # Component scores
    rule_based_score: float
    llm_safety_score: float
    review_sentiment_score: float
    menu_analysis_score: float
    
    # Detailed findings
    risk_factors: List[str]
    safety_indicators: List[str]
    recommended_actions: List[str]
    safe_menu_items: List[str]
    
    # Metadata
    data_sources_used: List[str]
    reviews_analyzed: int
    menu_items_found: int
    
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
    Ensemble scoring system that combines multiple signals.
    This is YOUR algorithm - a key technical contribution.
    """
    
    def __init__(self):
        """Initialize scorer with configured weights."""
        self.weights = Config.WEIGHTS
        logger.info(f"Initialized scorer with weights: {self.weights}")
    
    def _normalize_score(self, score: float, reverse: bool = False) -> float:
        """
        Normalize score to 0-100 range.
        
        Args:
            score: Input score
            reverse: If True, reverse the scale (higher input = lower output)
        
        Returns:
            Normalized score
        """
        normalized = max(0, min(100, score))
        if reverse:
            normalized = 100 - normalized
        return normalized
    
    def calculate_review_sentiment_score(self, 
                                        review_summary: Dict,
                                        allergen_type: str) -> float:
        """
        Calculate score based on review sentiment and mentions.
        
        Args:
            review_summary: Summary from allergen detector
            allergen_type: Type of allergen
        
        Returns:
            Risk score (0-100, higher = more risk)
        """
        # Extract metrics
        total_mentions = review_summary.get('total_mentions', 0)
        safety_indicators = len(review_summary.get('safety_indicators', []))
        warning_indicators = len(review_summary.get('warning_indicators', []))
        avg_risk = review_summary.get('average_risk_score', 50)
        
        # Weight different signals
        mention_risk = min(40, total_mentions * 3)  # More mentions = more risk
        warning_risk = min(30, warning_indicators * 15)  # Warnings are serious
        safety_bonus = min(20, safety_indicators * 5)  # Safety mentions help
        
        # Combine
        score = (mention_risk + warning_risk + avg_risk - safety_bonus) / 2
        
        return self._normalize_score(score)
    
    def calculate_menu_analysis_score(self, 
                                     menu_items: List[str],
                                     allergen_type: str) -> float:
        """
        Calculate score based on menu analysis.
        
        Args:
            menu_items: List of menu items
            allergen_type: Type of allergen
        
        Returns:
            Risk score (0-100, higher = more risk)
        """
        if not menu_items:
            return 50.0  # Neutral score if no data
        
        # Get allergen keywords
        allergen_keywords = Config.ALLERGEN_KEYWORDS.get(allergen_type, [])
        
        # Count allergen mentions in menu
        allergen_count = 0
        for item in menu_items:
            item_lower = item.lower()
            for keyword in allergen_keywords:
                if keyword in item_lower:
                    allergen_count += 1
                    break
        
        # Calculate risk
        if len(menu_items) == 0:
            proportion = 0
        else:
            proportion = allergen_count / len(menu_items)
        
        # High proportion of allergen items = higher risk
        base_score = proportion * 60
        
        # Penalty for many allergen items (suggests limited safe options)
        if allergen_count > 10:
            base_score += 20
        elif allergen_count > 5:
            base_score += 10
        
        return self._normalize_score(base_score)
    
    def _calculate_confidence(self,
                            data_sources: List[str],
                            reviews_count: int,
                            menu_items_count: int,
                            llm_confidence: float) -> float:
        """
        Calculate overall confidence in the assessment.
        
        Args:
            data_sources: List of data sources used
            reviews_count: Number of reviews analyzed
            menu_items_count: Number of menu items found
            llm_confidence: LLM's self-reported confidence
        
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from data availability
        data_score = len(data_sources) / 4.0  # Assume max 4 sources
        
        # Review confidence (more reviews = higher confidence)
        review_score = min(1.0, reviews_count / 20.0)
        
        # Menu confidence (having menu data helps)
        menu_score = 1.0 if menu_items_count > 5 else 0.5
        
        # Average all confidence signals
        confidence = (data_score + review_score + menu_score + llm_confidence) / 4.0
        
        return min(1.0, max(0.0, confidence))
    
    def aggregate_scores(self,
                        rule_based_score: float,
                        llm_response: Optional[object],
                        review_summary: Dict,
                        menu_items: List[str],
                        restaurant_name: str,
                        allergen_type: str = "gluten") -> SafetyAssessment:
        """
        Aggregate all scores using ensemble method.
        
        This is the core of YOUR algorithm.
        
        Args:
            rule_based_score: Score from rule-based detector
            llm_response: LLMResponse object (or None)
            review_summary: Summary from allergen detection
            menu_items: List of menu items
            restaurant_name: Restaurant name
            allergen_type: Type of allergen
        
        Returns:
            SafetyAssessment object
        """
        logger.info(f"Aggregating scores for {restaurant_name}")
        
        # Calculate component scores
        review_sentiment_score = self.calculate_review_sentiment_score(
            review_summary, 
            allergen_type
        )
        
        menu_analysis_score = self.calculate_menu_analysis_score(
            menu_items,
            allergen_type
        )
        
        # Get LLM score (or use neutral if unavailable)
        llm_safety_score = 50.0
        llm_confidence = 0.5
        risk_factors = []
        safe_alternatives = []
        
        if llm_response:
            llm_safety_score = llm_response.safety_score
            llm_confidence = llm_response.confidence
            risk_factors = llm_response.risk_factors
            safe_alternatives = llm_response.safe_alternatives
        else:
            logger.warning("LLM response not available, using neutral score")
        
        # Weighted ensemble
        overall_score = (
            self.weights['rule_based_score'] * rule_based_score +
            self.weights['llm_reasoning'] * llm_safety_score +
            self.weights['review_sentiment'] * review_sentiment_score +
            self.weights['menu_analysis'] * menu_analysis_score
        )
        
        overall_score = self._normalize_score(overall_score)
        
        # Determine data sources used
        data_sources = ['rule_based_analysis']
        if llm_response:
            data_sources.append('llm_reasoning')
        if review_summary.get('reviews_analyzed', 0) > 0:
            data_sources.append('reviews')
        if menu_items:
            data_sources.append('menu')
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            data_sources,
            review_summary.get('reviews_analyzed', 0),
            len(menu_items),
            llm_confidence
        )
        
        # Compile risk factors
        all_risk_factors = risk_factors.copy()
        if review_summary.get('warning_indicators'):
            all_risk_factors.extend(review_summary['warning_indicators'][:3])
        
        # Compile safety indicators
        safety_indicators = review_summary.get('safety_indicators', [])[:5]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score,
            all_risk_factors,
            safety_indicators,
            allergen_type
        )
        
        # Create assessment
        assessment = SafetyAssessment(
            restaurant_name=restaurant_name,
            allergen_type=allergen_type,
            overall_safety_score=overall_score,
            confidence_score=confidence,
            rule_based_score=rule_based_score,
            llm_safety_score=llm_safety_score,
            review_sentiment_score=review_sentiment_score,
            menu_analysis_score=menu_analysis_score,
            risk_factors=all_risk_factors,
            safety_indicators=safety_indicators,
            recommended_actions=recommendations,
            safe_menu_items=safe_alternatives,
            data_sources_used=data_sources,
            reviews_analyzed=review_summary.get('reviews_analyzed', 0),
            menu_items_found=len(menu_items)
        )
        
        logger.info(f"Final safety score: {overall_score:.1f}/100 ({assessment.get_rating()})")
        logger.info(f"Confidence: {confidence:.2f}")
        
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
