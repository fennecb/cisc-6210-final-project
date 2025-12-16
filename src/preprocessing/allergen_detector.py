"""
Allergen detection module - YOUR NLP implementation.
This demonstrates domain-specific text analysis without relying on LLMs.
"""
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

from config.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class AllergenMention:
    """Represents an allergen mention in text."""
    allergen_type: str
    keyword: str
    context: str  # Surrounding text
    position: int
    confidence: float

@dataclass
class AllergenAnalysis:
    """Results of allergen analysis."""
    mentions: List[AllergenMention]
    allergen_counts: Dict[str, int]
    risk_score: float
    safety_indicators: List[str]
    warning_indicators: List[str]
    
class AllergenDetector:
    """
    Rule-based allergen detection system.
    This is YOUR technical contribution - no LLM required.
    """
    
    def __init__(self):
        """Initialize detector with keyword dictionaries."""
        self.allergen_keywords = Config.ALLERGEN_KEYWORDS
        self.cross_contamination_keywords = Config.CROSS_CONTAMINATION_KEYWORDS
        self.safety_keywords = Config.SAFETY_KEYWORDS
        
        # Build regex patterns for efficient matching
        self._build_patterns()
    
    def _build_patterns(self):
        """Build compiled regex patterns for each allergen type."""
        self.allergen_patterns = {}
        
        for allergen_type, keywords in self.allergen_keywords.items():
            # Create word boundary pattern for each keyword
            patterns = [rf'\b{re.escape(kw)}\w*\b' for kw in keywords]
            combined_pattern = '|'.join(patterns)
            self.allergen_patterns[allergen_type] = re.compile(
                combined_pattern, 
                re.IGNORECASE
            )
        
        # Cross-contamination pattern
        cc_patterns = [rf'{re.escape(kw)}' for kw in self.cross_contamination_keywords]
        self.cross_contamination_pattern = re.compile(
            '|'.join(cc_patterns),
            re.IGNORECASE
        )
        
        # Safety indicator pattern
        safety_patterns = [rf'{re.escape(kw)}' for kw in self.safety_keywords]
        self.safety_pattern = re.compile(
            '|'.join(safety_patterns),
            re.IGNORECASE
        )
    
    def _extract_context(self, text: str, position: int, window: int = 50) -> str:
        """Extract context around a match."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def detect_allergens(self, text: str, focus_allergen: str = None) -> AllergenAnalysis:
        """
        Detect allergen mentions in text.
        
        Args:
            text: Text to analyze
            focus_allergen: Optional specific allergen to focus on (e.g., "gluten")
        
        Returns:
            AllergenAnalysis object
        """
        if not text:
            return AllergenAnalysis([], {}, 0.0, [], [])
        
        text_lower = text.lower()
        mentions = []
        allergen_counts = defaultdict(int)
        
        # Analyze each allergen type
        if focus_allergen and focus_allergen in self.allergen_keywords:
            allergens_to_check = [(focus_allergen, self.allergen_keywords[focus_allergen])]
        else:
            allergens_to_check = list(self.allergen_keywords.items())
        
        for allergen_type, keywords in allergens_to_check:
            pattern = self.allergen_patterns[allergen_type]
            
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                position = match.start()
                context = self._extract_context(text, position)
                
                # Calculate confidence based on context
                confidence = self._calculate_confidence(context, allergen_type)
                
                mention = AllergenMention(
                    allergen_type=allergen_type,
                    keyword=matched_text,
                    context=context,
                    position=position,
                    confidence=confidence
                )
                mentions.append(mention)
                allergen_counts[allergen_type] += 1
        
        # Detect safety indicators
        safety_indicators = self._find_safety_indicators(text)
        
        # Detect warning indicators
        warning_indicators = self._find_warning_indicators(text)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            mentions, 
            safety_indicators, 
            warning_indicators
        )
        
        return AllergenAnalysis(
            mentions=mentions,
            allergen_counts=dict(allergen_counts),
            risk_score=risk_score,
            safety_indicators=safety_indicators,
            warning_indicators=warning_indicators
        )
    
    def _calculate_confidence(self, context: str, allergen_type: str) -> float:
        """
        Calculate confidence of allergen mention based on context.
        Higher confidence = more likely to be relevant.
        
        Args:
            context: Text context around mention
            allergen_type: Type of allergen
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence
        context_lower = context.lower()
        
        # Boost confidence for negative mentions
        negative_indicators = [
            'no ', 'without', 'free', 'avoid', 'allergen', 
            'contain', 'has ', 'includes'
        ]
        for indicator in negative_indicators:
            if indicator in context_lower:
                confidence += 0.2
                break
        
        # Boost for question context (people asking about allergens)
        if '?' in context:
            confidence += 0.15
        
        # Boost for explicit allergen discussion
        if 'allerg' in context_lower or 'celiac' in context_lower:
            confidence += 0.2
        
        # Lower confidence for casual mentions
        casual_phrases = ['love', 'delicious', 'yummy', 'great']
        for phrase in casual_phrases:
            if phrase in context_lower:
                confidence -= 0.1
                break
        
        return min(1.0, max(0.0, confidence))
    
    def _find_safety_indicators(self, text: str) -> List[str]:
        """Find positive safety indicators in text."""
        indicators = []
        for match in self.safety_pattern.finditer(text):
            indicators.append(match.group(0))
        return indicators
    
    def _find_warning_indicators(self, text: str) -> List[str]:
        """Find cross-contamination warnings in text."""
        warnings = []
        for match in self.cross_contamination_pattern.finditer(text):
            warnings.append(match.group(0))
        return warnings
    
    def _calculate_risk_score(self,
                             mentions: List[AllergenMention],
                             safety_indicators: List[str],
                             warning_indicators: List[str]) -> float:
        """
        Calculate overall risk score (0-100).
        
        Higher score = higher risk
        
        Args:
            mentions: List of allergen mentions
            safety_indicators: Positive safety mentions
            warning_indicators: Cross-contamination warnings
        
        Returns:
            Risk score (0-100)
        """
        if not mentions and not warning_indicators:
            return 0.0  # No allergen mentions
        
        # Start with base risk from number of mentions
        base_risk = min(50, len(mentions) * 10)
        
        # Add risk from high-confidence mentions
        high_conf_mentions = [m for m in mentions if m.confidence > 0.7]
        confidence_risk = min(30, len(high_conf_mentions) * 8)
        
        # Add risk from warning indicators
        warning_risk = min(40, len(warning_indicators) * 20)
        
        # Subtract safety bonus
        safety_bonus = min(30, len(safety_indicators) * 10)
        
        total_risk = base_risk + confidence_risk + warning_risk - safety_bonus
        
        return max(0.0, min(100.0, total_risk))
    
    def analyze_reviews(self, reviews: List[Dict], focus_allergen: str = None) -> Dict:
        """
        Analyze multiple reviews for allergen mentions.
        
        Args:
            reviews: List of review dictionaries with 'text' field
            focus_allergen: Optional allergen to focus on
        
        Returns:
            Aggregated analysis results
        """
        all_mentions = []
        total_allergen_counts = defaultdict(int)
        total_safety_indicators = []
        total_warning_indicators = []
        risk_scores = []
        
        for review in reviews:
            text = review.get('text', '')
            if not text or len(text) < Config.REVIEW_MIN_LENGTH:
                continue
            
            analysis = self.detect_allergens(text, focus_allergen)
            
            all_mentions.extend(analysis.mentions)
            for allergen, count in analysis.allergen_counts.items():
                total_allergen_counts[allergen] += count
            total_safety_indicators.extend(analysis.safety_indicators)
            total_warning_indicators.extend(analysis.warning_indicators)
            risk_scores.append(analysis.risk_score)
        
        # Calculate aggregate metrics
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        return {
            'total_mentions': len(all_mentions),
            'allergen_counts': dict(total_allergen_counts),
            'safety_indicators': total_safety_indicators,
            'warning_indicators': total_warning_indicators,
            'average_risk_score': avg_risk_score,
            'reviews_analyzed': len([r for r in reviews if len(r.get('text', '')) >= Config.REVIEW_MIN_LENGTH])
        }
