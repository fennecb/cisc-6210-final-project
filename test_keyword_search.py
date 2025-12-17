"""
Test script to demonstrate the new keyword search functionality.
"""
from src.scoring.safety_scorer import SafetyScorer
from config.config import Config

def test_keyword_search():
    """Test the keyword search on sample reviews."""

    scorer = SafetyScorer()

    # Sample reviews (you can replace with actual reviews)
    sample_reviews = [
        {
            'text': 'This restaurant has an excellent gluten-free menu! I have celiac disease and felt very safe eating here. The staff was knowledgeable about cross-contamination.',
            'rating': 5
        },
        {
            'text': 'Good food but be careful if you have allergies. They use wheat flour in many dishes and the kitchen is small so cross-contamination is possible.',
            'rating': 3
        },
        {
            'text': 'Amazing Thai food! The pad thai is delicious and they offer rice noodles as a gluten-free option. Highly recommend!',
            'rating': 5
        },
        {
            'text': 'I asked about allergen information and the server was very helpful. They brought out the chef who explained all ingredients.',
            'rating': 4
        },
        {
            'text': 'Great atmosphere and friendly service. The menu has lots of variety.',
            'rating': 4
        }
    ]

    print("=" * 70)
    print("KEYWORD SEARCH TEST")
    print("=" * 70)

    # Test different allergen types
    allergen_types = ['gluten', 'dairy', 'peanut']

    for allergen in allergen_types:
        print(f"\n[Allergen Type: {allergen.upper()}]")
        print("-" * 70)

        # Get keywords for this allergen
        allergen_keywords = Config.ALLERGEN_KEYWORDS.get(allergen, [])
        safety_keywords = Config.SAFETY_KEYWORDS
        search_keywords = allergen_keywords + safety_keywords + ['allergen', 'allergy', 'celiac']

        print(f"Searching for keywords: {', '.join(search_keywords[:10])}...")

        # Search reviews
        excerpts = scorer.search_review_keywords(
            sample_reviews,
            search_keywords,
            context_window=150
        )

        if excerpts:
            print(f"\nFound {len(excerpts)} relevant excerpts:")
            for i, excerpt in enumerate(excerpts, 1):
                print(f"  {i}. {excerpt}\n")
        else:
            print(f"\nNo relevant excerpts found for {allergen}")

    print("=" * 70)
    print("\n[Summary]")
    print("The keyword search extracts relevant context from reviews,")
    print("providing transparent evidence for the LLM's safety assessment.")
    print("\nThis helps users understand:")
    print("  • What allergen mentions exist in reviews")
    print("  • What customers actually said about safety")
    print("  • The context around allergen discussions")

if __name__ == "__main__":
    test_keyword_search()
