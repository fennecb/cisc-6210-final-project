"""
Unit tests for allergen detector.
Demonstrates YOUR algorithm's correctness.
"""
import pytest
from src.preprocessing.allergen_detector import AllergenDetector

@pytest.fixture
def detector():
    """Create detector instance for tests."""
    return AllergenDetector()

def test_basic_gluten_detection(detector):
    """Test detection of gluten-containing items."""
    text = "This restaurant has great bread and pasta dishes."
    analysis = detector.detect_allergens(text, focus_allergen="gluten")
    
    assert analysis.allergen_counts.get('gluten', 0) >= 2  # bread and pasta
    assert analysis.risk_score > 0

def test_safety_indicators(detector):
    """Test detection of safety indicators."""
    text = "They have a dedicated gluten-free kitchen and certified gluten-free menu items."
    analysis = detector.detect_allergens(text, focus_allergen="gluten")
    
    assert len(analysis.safety_indicators) >= 2
    assert analysis.risk_score < 50  # Lower risk with safety indicators

def test_cross_contamination_warnings(detector):
    """Test detection of cross-contamination warnings."""
    text = "Food prepared in shared fryer. May contain traces of wheat."
    analysis = detector.detect_allergens(text, focus_allergen="gluten")
    
    assert len(analysis.warning_indicators) >= 1
    assert analysis.risk_score > 30  # Higher risk with warnings

def test_confidence_calculation(detector):
    """Test confidence scoring."""
    # Explicit allergen question
    text1 = "Does this restaurant have gluten-free options?"
    analysis1 = detector.detect_allergens(text1, focus_allergen="gluten")
    
    # Casual mention
    text2 = "I love their bread, it's so delicious!"
    analysis2 = detector.detect_allergens(text2, focus_allergen="gluten")
    
    # Explicit mention should have higher confidence
    if analysis1.mentions and analysis2.mentions:
        assert analysis1.mentions[0].confidence > analysis2.mentions[0].confidence

def test_multiple_allergen_types(detector):
    """Test detection across multiple allergen types."""
    text = "They use butter, milk, and cheese in most dishes."
    analysis = detector.detect_allergens(text, focus_allergen="dairy")
    
    assert analysis.allergen_counts.get('dairy', 0) >= 3

def test_empty_text(detector):
    """Test handling of empty text."""
    analysis = detector.detect_allergens("", focus_allergen="gluten")
    
    assert len(analysis.mentions) == 0
    assert analysis.risk_score == 0.0

def test_review_batch_analysis(detector):
    """Test batch review analysis."""
    reviews = [
        {'text': 'Great gluten-free pizza!', 'rating': 5},
        {'text': 'Dedicated fryer for allergies.', 'rating': 5},
        {'text': 'Cross-contamination concerns here.', 'rating': 2}
    ]
    
    summary = detector.analyze_reviews(reviews, focus_allergen="gluten")
    
    assert summary['reviews_analyzed'] == 3
    assert summary['total_mentions'] > 0
    assert 'average_risk_score' in summary

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
