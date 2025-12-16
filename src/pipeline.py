"""
Main Pipeline - Orchestrates the entire allergen safety analysis.
This is YOUR system architecture in action.
"""
from typing import Optional, Dict, List
from pathlib import Path

from config.config import Config
from src.utils.logger import setup_logger
from src.data_collection.google_places import GooglePlacesCollector
from src.data_collection.review_scraper import ReviewScraper, YelpAPICollector
from src.preprocessing.allergen_detector import AllergenDetector
from src.preprocessing.menu_ocr import MenuOCR
from src.llm_reasoning.llm_reasoner import LLMReasoner
from src.scoring.safety_scorer import SafetyScorer, SafetyAssessment

logger = setup_logger(__name__, log_file="logs/pipeline.log")

class AllergenSafetyPipeline:
    """
    Main pipeline that orchestrates all components.
    
    This demonstrates YOUR system design and integration skills.
    """
    
    def __init__(self, 
                 llm_provider: str = "gemini",
                 use_cache: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            llm_provider: LLM provider to use
            use_cache: Whether to use caching
        """
        logger.info("Initializing Allergen Safety Pipeline")
        
        # Validate API keys
        api_status = Config.validate_api_keys()
        logger.info(f"API key status: {api_status}")
        
        # Initialize components
        self.google_places = GooglePlacesCollector(use_cache=use_cache) if api_status['google_places'] else None
        self.review_scraper = ReviewScraper(use_cache=use_cache)
        self.yelp_collector = YelpAPICollector(use_cache=use_cache) if api_status['yelp'] else None
        self.allergen_detector = AllergenDetector()
        self.menu_ocr = MenuOCR(use_cache=use_cache)
        self.llm_reasoner = LLMReasoner(provider=llm_provider, use_cache=use_cache)
        self.scorer = SafetyScorer()
        
        # Create output directory
        Path("data/assessments").mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline initialization complete")
    
    def analyze_restaurant(self,
                          restaurant_name: str,
                          location: Optional[str] = None,
                          allergen_type: str = "gluten",
                          use_llm: bool = True) -> Optional[SafetyAssessment]:
        """
        Perform complete allergen safety analysis for a restaurant.
        
        This is the main method that showcases YOUR entire system.
        
        Args:
            restaurant_name: Name of restaurant
            location: Optional location (e.g., "New York, NY")
            allergen_type: Type of allergen to check for
            use_llm: Whether to use LLM reasoning (set False for pure rule-based)
        
        Returns:
            SafetyAssessment object or None
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting analysis for: {restaurant_name}")
        logger.info(f"Location: {location or 'Not specified'}")
        logger.info(f"Allergen: {allergen_type}")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Collect restaurant data
        logger.info("STEP 1: Collecting restaurant data...")
        restaurant_data = None
        
        if self.google_places:
            restaurant_data = self.google_places.collect_restaurant_data(
                restaurant_name, 
                location
            )
        
        if not restaurant_data:
            logger.warning("Could not retrieve restaurant data from Google Places")
            # Could continue with other data sources, but for now return
            return None
        
        logger.info(f"[OK] Restaurant found: {restaurant_data.name}")
        logger.info(f"  Address: {restaurant_data.address}")
        logger.info(f"  Rating: {restaurant_data.rating} ({restaurant_data.total_ratings} reviews)")
        
        # Step 2: Collect reviews from multiple sources
        logger.info("\nSTEP 2: Collecting reviews...")
        all_reviews = []
        
        # Google Places reviews
        if restaurant_data.reviews:
            all_reviews.extend(restaurant_data.reviews)
            logger.info(f"[OK] Collected {len(restaurant_data.reviews)} reviews from Google Places")
        
        # Yelp reviews (if available)
        if self.yelp_collector:
            yelp_id = self.yelp_collector.search_business(restaurant_name, location)
            if yelp_id:
                yelp_reviews = self.yelp_collector.get_reviews(yelp_id)
                all_reviews.extend(yelp_reviews)
                logger.info(f"[OK] Collected {len(yelp_reviews)} reviews from Yelp")
        
        logger.info(f"Total reviews collected: {len(all_reviews)}")
        
        # Step 3: Rule-based allergen detection
        logger.info("\nSTEP 3: Running rule-based allergen detection...")
        review_summary = self.allergen_detector.analyze_reviews(
            all_reviews,
            focus_allergen=allergen_type
        )
        
        logger.info(f"[OK] Analyzed {review_summary['reviews_analyzed']} reviews")
        logger.info(f"  Total allergen mentions: {review_summary['total_mentions']}")
        logger.info(f"  Safety indicators: {len(review_summary['safety_indicators'])}")
        logger.info(f"  Warning indicators: {len(review_summary['warning_indicators'])}")
        logger.info(f"  Rule-based risk score: {review_summary['average_risk_score']:.1f}/100")
        
        # Step 4: Extract menu information
        logger.info("\nSTEP 4: Extracting menu information...")
        menu_items = []
        
        # Try OCR on photos if available
        if restaurant_data.photos and self.google_places:
            logger.info("Attempting OCR on menu photos...")
            for i, photo_ref in enumerate(restaurant_data.photos[:2]):  # Limit to 2 photos
                try:
                    # Download photo
                    photo_path = f"data/temp/menu_{i}.jpg"
                    Path("data/temp").mkdir(parents=True, exist_ok=True)
                    
                    if self.google_places.download_photo(photo_ref, photo_path):
                        # Try OCR
                        menu_text = self.menu_ocr.extract_text(photo_path)
                        if menu_text:
                            # Split into items (simple heuristic)
                            items = [line.strip() for line in menu_text.split('\n') 
                                   if len(line.strip()) > 10]
                            menu_items.extend(items)
                            logger.info(f"  [OK] Extracted {len(items)} items from photo {i+1}")
                except Exception as e:
                    logger.warning(f"Could not process photo {i}: {e}")
        
        # Try website menu extraction
        if restaurant_data.website and self.review_scraper:
            website_menu = self.review_scraper.extract_menu_text_from_page(
                restaurant_data.website
            )
            if website_menu:
                items = [line.strip() for line in website_menu.split('\n') 
                        if len(line.strip()) > 10]
                menu_items.extend(items)
                logger.info(f"  [OK] Extracted {len(items)} items from website")
        
        # Remove duplicates
        menu_items = list(set(menu_items))
        logger.info(f"Total unique menu items: {len(menu_items)}")
        
        # Step 5: LLM reasoning (optional)
        llm_response = None
        if use_llm and self.llm_reasoner:
            logger.info("\nSTEP 5: Running LLM reasoning...")
            try:
                llm_response = self.llm_reasoner.assess_safety(
                    menu_items,
                    review_summary,
                    allergen_type,
                    restaurant_data.name
                )
                
                if llm_response:
                    logger.info(f"[OK] LLM analysis complete")
                    logger.info(f"  LLM safety score: {llm_response.safety_score:.1f}/100")
                    logger.info(f"  LLM confidence: {llm_response.confidence:.2f}")
                else:
                    logger.warning("LLM analysis failed")
            except Exception as e:
                logger.error(f"Error in LLM reasoning: {e}")
        else:
            logger.info("\nSTEP 5: Skipping LLM reasoning (disabled or unavailable)")
        
        # Step 6: Score aggregation
        logger.info("\nSTEP 6: Aggregating scores...")
        assessment = self.scorer.aggregate_scores(
            rule_based_score=review_summary['average_risk_score'],
            llm_response=llm_response,
            review_summary=review_summary,
            menu_items=menu_items,
            restaurant_name=restaurant_data.name,
            allergen_type=allergen_type
        )
        
        # Print summary
        self._print_assessment_summary(assessment)
        
        # Save assessment
        output_file = f"data/assessments/{restaurant_name.replace(' ', '_')}_{allergen_type}.json"
        self.scorer.export_assessment(assessment, output_file)
        
        logger.info(f"\n{'='*60}")
        logger.info("Analysis complete!")
        logger.info(f"{'='*60}\n")
        
        return assessment
    
    def _print_assessment_summary(self, assessment: SafetyAssessment):
        """Print a formatted summary of the assessment."""
        print("\n" + "="*60)
        print(f"ALLERGEN SAFETY ASSESSMENT: {assessment.restaurant_name}")
        print("="*60)
        print(f"\nAllergen Type: {assessment.allergen_type.upper()}")
        print(f"\n[*] OVERALL SAFETY SCORE: {assessment.overall_safety_score:.1f}/100")
        print(f"   Rating: {assessment.get_rating()}")
        print(f"   Confidence: {assessment.confidence_score:.0%}")
        
        print(f"\n[SCORES] COMPONENT SCORES:")
        print(f"   Rule-based Analysis: {assessment.rule_based_score:.1f}/100")
        print(f"   LLM Reasoning: {assessment.llm_safety_score:.1f}/100")
        print(f"   Review Sentiment: {assessment.review_sentiment_score:.1f}/100")
        print(f"   Menu Analysis: {assessment.menu_analysis_score:.1f}/100")
        
        if assessment.risk_factors:
            print(f"\n[!] RISK FACTORS:")
            for factor in assessment.risk_factors[:5]:
                print(f"   • {factor}")
        
        if assessment.safety_indicators:
            print(f"\n[+] SAFETY INDICATORS:")
            for indicator in assessment.safety_indicators[:5]:
                print(f"   • {indicator}")
        
        if assessment.recommended_actions:
            print(f"\n[>] RECOMMENDATIONS:")
            for action in assessment.recommended_actions:
                print(f"   • {action}")
        
        if assessment.safe_menu_items:
            print(f"\n[FOOD] POTENTIALLY SAFE ITEMS:")
            for item in assessment.safe_menu_items[:5]:
                print(f"   • {item}")
        
        print(f"\n[DATA] DATA SOURCES:")
        print(f"   Reviews analyzed: {assessment.reviews_analyzed}")
        print(f"   Menu items found: {assessment.menu_items_found}")
        print(f"   Sources used: {', '.join(assessment.data_sources_used)}")
        
        print("="*60 + "\n")
    
    def batch_analyze(self,
                     restaurants: List[Dict],
                     allergen_type: str = "gluten") -> List[SafetyAssessment]:
        """
        Analyze multiple restaurants.
        
        Args:
            restaurants: List of dicts with 'name' and optional 'location'
            allergen_type: Type of allergen
        
        Returns:
            List of SafetyAssessment objects
        """
        assessments = []
        
        for restaurant in restaurants:
            name = restaurant.get('name')
            location = restaurant.get('location')
            
            if not name:
                continue
            
            try:
                assessment = self.analyze_restaurant(
                    restaurant_name=name,
                    location=location,
                    allergen_type=allergen_type
                )
                if assessment:
                    assessments.append(assessment)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
        
        return assessments
