"""
Tests for NLP components - validates NLP implementations.
"""
import pytest
from src.preprocessing.sentiment_analyzer import SentimentAnalyzer
from src.preprocessing.review_ranker import ReviewRanker
from src.preprocessing.entity_extractor import EntityExtractor


@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()


@pytest.fixture
def review_ranker():
    return ReviewRanker()


@pytest.fixture
def entity_extractor():
    try:
        return EntityExtractor()
    except OSError:
        pytest.skip("spaCy model not installed")


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""

    def test_positive_sentiment(self, sentiment_analyzer):
        """Test detection of positive sentiment."""
        text = "This restaurant has amazing gluten-free options! Love it!"
        result = sentiment_analyzer.analyze_text(text)

        assert result.is_positive
        assert result.polarity > 0

    def test_negative_sentiment(self, sentiment_analyzer):
        """Test detection of negative sentiment."""
        text = "Terrible experience. No gluten-free options at all."
        result = sentiment_analyzer.analyze_text(text)

        assert result.is_negative
        assert result.polarity < 0

    def test_neutral_sentiment(self, sentiment_analyzer):
        """Test detection of neutral sentiment."""
        text = "The restaurant is located downtown."
        result = sentiment_analyzer.analyze_text(text)

        assert result.is_neutral

    def test_credibility_scoring(self, sentiment_analyzer):
        """Test review credibility calculation."""
        # Objective statement should have high credibility
        credibility_objective = sentiment_analyzer.calculate_review_credibility(
            polarity=0.1,
            subjectivity=0.2
        )

        # Subjective statement should have lower credibility
        credibility_subjective = sentiment_analyzer.calculate_review_credibility(
            polarity=0.8,
            subjectivity=0.9
        )

        assert credibility_objective > credibility_subjective

    def test_batch_analysis(self, sentiment_analyzer):
        """Test batch review analysis."""
        reviews = [
            {'text': 'Great gluten-free pizza!'},
            {'text': 'No safe options for celiacs.'},
            {'text': 'The atmosphere is nice.'}
        ]

        stats = sentiment_analyzer.batch_analyze_reviews(reviews)

        assert stats['total_analyzed'] == 3
        assert 'average_polarity' in stats
        assert stats['positive_count'] >= 0


class TestReviewRanker:
    """Test TF-IDF review ranking functionality."""

    def test_rank_reviews(self, review_ranker):
        """Test basic review ranking."""
        reviews = [
            {'text': 'This place has great gluten-free bread and pasta.'},
            {'text': 'The service was excellent.'},
            {'text': 'They offer dedicated gluten-free preparation.'}
        ]

        keywords = ['gluten', 'gluten-free', 'celiac']
        ranked = review_ranker.rank_reviews(reviews, keywords)

        assert len(ranked) == 3
        # First review should be most relevant (has most keywords)
        assert ranked[0][1] > ranked[1][1]  # Higher score for allergen-relevant

    def test_filter_relevant(self, review_ranker):
        """Test filtering of relevant reviews."""
        reviews = [
            {'text': 'Excellent gluten-free options available.'},
            {'text': 'Nice ambiance.'},
            {'text': 'Good parking.'},
            {'text': 'Dedicated gluten-free kitchen.'}
        ]

        keywords = ['gluten', 'gluten-free', 'allergen']
        filtered = review_ranker.filter_relevant_reviews(
            reviews,
            keywords,
            min_relevance=0.1
        )

        # Should filter out irrelevant reviews
        assert len(filtered) < len(reviews)
        assert len(filtered) >= 2  # Should keep the relevant ones

    def test_empty_reviews(self, review_ranker):
        """Test handling of empty review list."""
        ranked = review_ranker.rank_reviews([], ['gluten'])
        assert ranked == []

    def test_coverage_analysis(self, review_ranker):
        """Test review coverage analysis."""
        reviews = [
            {'text': 'Great gluten-free menu'},
            {'text': 'Love the gluten-free bread'},
            {'text': 'Nice decor'}
        ]

        keywords = ['gluten', 'gluten-free']
        coverage = review_ranker.analyze_review_coverage(reviews, keywords)

        assert coverage['total_reviews'] == 3
        assert coverage['relevant_reviews'] >= 2
        assert 0 <= coverage['coverage_percentage'] <= 100


class TestEntityExtractor:
    """Test named entity recognition functionality."""

    def test_extract_basic_entities(self, entity_extractor):
        """Test basic entity extraction."""
        text = "I ordered the chicken sandwich and a caesar salad."
        entities = entity_extractor.extract_entities(text)

        assert 'food_items' in entities
        assert 'ingredients' in entities

    def test_extract_menu_items(self, entity_extractor):
        """Test menu item extraction."""
        menu_text = """
        Grilled Chicken Sandwich - with lettuce and tomato
        Caesar Salad - romaine lettuce with parmesan
        Margherita Pizza - fresh mozzarella and basil
        """

        items = entity_extractor.extract_menu_items(menu_text)

        assert len(items) > 0
        assert all('name' in item for item in items)

    def test_extract_safety_mentions(self, entity_extractor):
        """Test extraction of safety-related terms."""
        text = "We have a dedicated gluten-free fryer and separate preparation area."

        safety = entity_extractor.extract_safety_mentions(text)

        assert safety['has_safety_info']
        assert len(safety['safety_terms_found']) > 0
        assert len(safety['equipment_mentions']) > 0

    def test_find_allergen_phrases(self, entity_extractor):
        """Test finding phrases containing allergen keywords."""
        text = "They have great gluten-free bread. The wheat pasta is also good."

        phrases = entity_extractor.find_allergen_phrases(
            text,
            ['gluten', 'wheat']
        )

        assert len(phrases) >= 2
        assert any('gluten' in p['phrase'] for p in phrases)

    def test_ingredient_patterns(self, entity_extractor):
        """Test ingredient pattern analysis."""
        menu_items = [
            "Grilled chicken with cheese",
            "Fried fish with cheese sauce",
            "Baked salmon with lemon"
        ]

        patterns = entity_extractor.analyze_ingredient_patterns(menu_items)

        assert patterns['total_items'] == 3
        assert len(patterns['common_ingredients']) > 0
        assert len(patterns['preparation_methods']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
