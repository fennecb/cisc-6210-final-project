"""
Sample Usage - User-friendly allergen safety checker.
This is what an end user would interact with.
"""
import sys
import os
import logging
import warnings

# Suppress warnings for cleaner user experience
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import AllergenSafetyPipeline
from config.config import Config

# Suppress verbose logging for cleaner user experience
logging.getLogger('src').setLevel(logging.WARNING)
logging.getLogger('config').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def print_header():
    """Print a clean header."""
    print("\n" + "=" * 70)
    print("🛡️  ALLERGEN SAFETY CHECKER")
    print("=" * 70)
    print("Get AI-powered allergen safety assessments for restaurants")
    print()


def get_user_input():
    """Get restaurant information from user."""
    print("Enter restaurant information:")
    restaurant_name = input("  Restaurant name: ").strip()
    location = input("  Location (city, state): ").strip()

    print("\nSelect allergen type:")
    print("  1. Gluten")
    print("  2. Dairy")
    print("  3. Nuts")
    print("  4. Shellfish")
    print("  5. Soy")
    print("  6. Eggs")

    allergen_choice = input("\nYour choice (1-6): ").strip()

    allergen_map = {
        "1": "gluten",
        "2": "dairy",
        "3": "nuts",
        "4": "shellfish",
        "5": "soy",
        "6": "eggs"
    }

    allergen_type = allergen_map.get(allergen_choice, "gluten")

    return restaurant_name, location, allergen_type


def print_assessment(assessment):
    """Print assessment in a user-friendly format."""

    print("\n" + "=" * 70)
    print(f"📍 ASSESSMENT FOR: {assessment.restaurant_name.upper()}")
    print("=" * 70)

    # Safety Score and Rating
    print(f"\n🎯 SAFETY RATING: {assessment.get_rating()}")
    print(f"   Safety Score: {assessment.overall_safety_score:.1f}/100")
    print(f"   (Higher scores are safer)")
    print(f"   Confidence: {assessment.confidence_score:.0%}")

    # Risk Factors
    if assessment.risk_factors:
        print("\n⚠️  RISK FACTORS:")
        for i, risk in enumerate(assessment.risk_factors[:5], 1):
            print(f"   {i}. {risk}")
    else:
        print("\n✅ No major risk factors identified")

    # Safety Indicators
    if assessment.safety_indicators:
        print("\n✨ POSITIVE INDICATORS:")
        for i, indicator in enumerate(assessment.safety_indicators[:5], 1):
            print(f"   {i}. {indicator}")

    # Safe Menu Items
    if assessment.safe_menu_items:
        print("\n🍽️  RECOMMENDED SAFE OPTIONS:")
        for i, item in enumerate(assessment.safe_menu_items[:5], 1):
            print(f"   {i}. {item}")

    # Relevant Review Excerpts
    if assessment.relevant_review_excerpts:
        print("\n💬 WHAT REVIEWERS ARE SAYING:")
        for i, excerpt in enumerate(assessment.relevant_review_excerpts[:3], 1):
            # Clean up the excerpt for display
            if len(excerpt) > 150:
                excerpt = excerpt[:147] + "..."
            print(f"   {i}. {excerpt}")

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(assessment.recommended_actions, 1):
        print(f"   {i}. {rec}")

    # Data sources used
    print(f"\n📊 Analysis based on:")
    sources_display = {
        'llm_reasoning': '✓ AI reasoning',
        'reviews': f'✓ {assessment.reviews_analyzed} reviews',
        'menu': f'✓ {assessment.menu_items_found} menu items'
    }
    for source in assessment.data_sources_used:
        if source in sources_display:
            print(f"   {sources_display[source]}")

    print("\n" + "=" * 70)
    print()


def main():
    """Run the user-friendly allergen safety checker."""

    print_header()

    # Check API keys first
    api_status = Config.validate_api_keys()

    if not api_status.get('google_places'):
        print("❌ ERROR: Google Places API key not configured!")
        print("   Please add GOOGLE_PLACES_API_KEY to your .env file")
        return

    if not (api_status.get('google_gemini') or api_status.get('openai') or api_status.get('anthropic')):
        print("❌ ERROR: No LLM API key configured!")
        print("   Please add at least one of: GOOGLE_GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
        return

    # Get user input
    restaurant_name, location, allergen_type = get_user_input()

    if not restaurant_name or not location:
        print("\n❌ Restaurant name and location are required!")
        return

    # Initialize pipeline (suppress verbose output)
    print("\n🔍 Analyzing restaurant safety...")
    print("   (This may take 30-60 seconds)")
    print()

    try:
        pipeline = AllergenSafetyPipeline(
            llm_provider="gemini",  # Change if needed
            use_cache=True
        )

        # Run analysis
        assessment = pipeline.analyze_restaurant(
            restaurant_name=restaurant_name,
            location=location,
            allergen_type=allergen_type,
            use_llm=True
        )

        if assessment:
            print_assessment(assessment)

            # Ask if user wants to save
            save = input("Would you like to save this assessment? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"data/assessments/{restaurant_name.replace(' ', '_')}_{allergen_type}.json"
                print(f"\n✅ Assessment saved to: {filename}")
        else:
            print("\n❌ Could not complete assessment. Restaurant may not be found.")
            print("   Try adjusting the restaurant name or location.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("   Please check your API keys and internet connection.")


if __name__ == "__main__":
    main()
