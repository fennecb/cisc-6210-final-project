"""
Configuration management for the Allergen Safety System.
"""
import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the system."""
    
    # API Keys
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    YELP_API_KEY = os.getenv("YELP_API_KEY", "")
    
    # System settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    
    # Paths
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Allergen keywords (YOUR domain knowledge)
    ALLERGEN_KEYWORDS = {
        "gluten": [
            "wheat", "barley", "rye", "gluten", "flour", "bread", "pasta",
            "couscous", "seitan", "malt", "beer", "soy sauce", "breadcrumb"
        ],
        "dairy": [
            "milk", "cheese", "butter", "cream", "yogurt", "whey", "casein",
            "lactose", "ghee", "paneer"
        ],
        "nuts": [
            "peanut", "almond", "walnut", "cashew", "pecan", "pistachio",
            "hazelnut", "macadamia", "nut"
        ],
        "shellfish": [
            "shrimp", "crab", "lobster", "clam", "mussel", "oyster", "scallop"
        ],
        "soy": [
            "soy", "soybean", "tofu", "tempeh", "edamame", "miso"
        ],
        "eggs": [
            "egg", "mayonnaise", "aioli", "meringue"
        ]
    }
    
    # Cross-contamination risk indicators (YOUR domain expertise)
    CROSS_CONTAMINATION_KEYWORDS = [
        "shared fryer", "shared equipment", "may contain traces",
        "processed in facility", "cross-contact", "same kitchen",
        "shared surfaces", "not guaranteed gluten-free"
    ]
    
    # Positive safety indicators
    SAFETY_KEYWORDS = [
        "gluten-free", "dedicated fryer", "separate preparation",
        "certified gluten-free", "celiac-safe", "allergen-free kitchen",
        "no cross-contamination", "dedicated equipment"
    ]
    
    # Data collection settings
    MAX_REVIEWS_PER_RESTAURANT = 50
    REVIEW_MIN_LENGTH = 20  # characters
    
    # Scoring weights (YOUR algorithm design)
    WEIGHTS = {
        "llm_reasoning": 0.35,
        "rule_based_score": 0.40,
        "review_sentiment": 0.15,
        "menu_analysis": 0.10
    }
    
    @classmethod
    def get_allergen_list(cls, allergen_type: str = None) -> List[str]:
        """Get list of allergen keywords."""
        if allergen_type:
            return cls.ALLERGEN_KEYWORDS.get(allergen_type, [])
        # Return all allergens flattened
        all_allergens = []
        for keywords in cls.ALLERGEN_KEYWORDS.values():
            all_allergens.extend(keywords)
        return all_allergens
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "google_places": bool(cls.GOOGLE_PLACES_API_KEY),
            "google_gemini": bool(cls.GOOGLE_GEMINI_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY),
            "yelp": bool(cls.YELP_API_KEY)
        }
