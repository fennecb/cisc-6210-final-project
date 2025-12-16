"""
Demo script for the Allergen Safety System.
Run this to test the entire pipeline.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import AllergenSafetyPipeline
from config.config import Config

def main():
    """Run demo analysis."""
    
    print("="*60)
    print("ALLERGEN SAFETY SYSTEM - DEMO")
    print("="*60)
    print()
    
    # Check API keys
    api_status = Config.validate_api_keys()
    print("API Configuration:")
    for service, available in api_status.items():
        status = "[OK]" if available else "[X]"
        print(f"  {status} {service}")
    print()
    
    if not any(api_status.values()):
        print("ERROR: No API keys configured!")
        print("Please copy .env.example to .env and add your API keys.")
        return
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = AllergenSafetyPipeline(
        llm_provider="gemini",  # Change to "openai" or "anthropic" if desired
        use_cache=True
    )
    print()
    
    # Example restaurants to analyze
    test_restaurants = [
        {
            "name": "Chipotle",
            "location": "New York, NY"
        },
        # Add more restaurants here for testing
    ]
    
    # Run analysis
    for restaurant in test_restaurants:
        print(f"\nAnalyzing: {restaurant['name']}...")
        print("-" * 60)
        
        try:
            assessment = pipeline.analyze_restaurant(
                restaurant_name=restaurant['name'],
                location=restaurant.get('location'),
                allergen_type="gluten",  # Focus on gluten for celiac disease
                use_llm=True  # Set to False to test pure rule-based approach
            )
            
            if assessment:
                print(f"\n[OK] Analysis complete for {restaurant['name']}")
                print(f"  Safety Score: {assessment.overall_safety_score:.1f}/100")
                print(f"  Rating: {assessment.get_rating()}")
            else:
                print(f"\n[X] Analysis failed for {restaurant['name']}")
        
        except Exception as e:
            print(f"\n[X] Error analyzing {restaurant['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nAssessment reports saved to: data/assessments/")
    print("Logs saved to: logs/pipeline.log")

if __name__ == "__main__":
    main()
