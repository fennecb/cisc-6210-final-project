"""
Web scraper for collecting additional restaurant reviews.
This module demonstrates YOUR web scraping skills.
"""
import re
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote_plus

from src.utils.logger import setup_logger
from src.utils.cache import CacheManager

logger = setup_logger(__name__)

class ReviewScraper:
    """Generic review scraper for restaurant websites."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize scraper."""
        self.cache = CacheManager() if use_cache else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def search_yelp_reviews(self, restaurant_name: str, location: str = None) -> List[Dict]:
        """
        Scrape Yelp reviews (basic version - respects robots.txt).
        Note: For production, use Yelp Fusion API instead.
        
        Args:
            restaurant_name: Restaurant name
            location: Location string
        
        Returns:
            List of review dictionaries
        """
        cache_key = f"yelp_reviews:{restaurant_name}:{location}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached Yelp reviews for: {restaurant_name}")
                return cached
        
        reviews = []
        
        # In a real implementation, you would:
        # 1. Use Yelp Fusion API (recommended)
        # 2. Or implement proper Yelp scraping with robots.txt compliance
        # For now, return placeholder for demonstration
        
        logger.warning("Yelp scraping not implemented - use Yelp Fusion API")
        
        # Placeholder structure
        reviews = []
        
        if self.cache:
            self.cache.set(cache_key, reviews)
        
        return reviews
    
    def scrape_generic_reviews(self, url: str) -> List[Dict]:
        """
        Generic review scraper for restaurant websites.
        This demonstrates YOUR ability to parse HTML and extract structured data.
        
        Args:
            url: URL to scrape
        
        Returns:
            List of review dictionaries
        """
        cache_key = f"generic_scrape:{url}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached scrape for: {url}")
                return cached
        
        reviews = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Generic patterns to look for reviews
            # This is a simplified example - real implementation would be site-specific
            
            # Look for common review patterns
            review_containers = (
                soup.find_all('div', class_=re.compile(r'review|comment|rating')) +
                soup.find_all('article', class_=re.compile(r'review|comment'))
            )
            
            for container in review_containers[:20]:  # Limit to 20 reviews
                text = container.get_text(separator=' ', strip=True)
                
                # Extract rating if present
                rating_elem = container.find(class_=re.compile(r'rating|star'))
                rating = None
                if rating_elem:
                    rating_text = rating_elem.get_text()
                    rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))
                
                if len(text) > 50:  # Minimum length filter
                    reviews.append({
                        'text': self._clean_text(text),
                        'rating': rating,
                        'source': url
                    })
            
            logger.info(f"Scraped {len(reviews)} reviews from {url}")
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        if self.cache:
            self.cache.set(cache_key, reviews)
        
        return reviews
    
    def extract_menu_text_from_page(self, url: str) -> Optional[str]:
        """
        Extract menu text from a restaurant website.
        
        Args:
            url: Restaurant website URL
        
        Returns:
            Extracted menu text or None
        """
        cache_key = f"menu_text:{url}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached menu text for: {url}")
                return cached
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for menu sections
            menu_sections = (
                soup.find_all('div', class_=re.compile(r'menu|food|dish', re.I)) +
                soup.find_all('section', class_=re.compile(r'menu|food', re.I))
            )
            
            menu_text = ""
            for section in menu_sections:
                text = section.get_text(separator='\n', strip=True)
                menu_text += text + "\n"
            
            menu_text = self._clean_text(menu_text)
            
            if len(menu_text) > 100:
                logger.info(f"Extracted {len(menu_text)} chars of menu text")
                
                if self.cache:
                    self.cache.set(cache_key, menu_text)
                
                return menu_text
            
        except Exception as e:
            logger.error(f"Error extracting menu from {url}: {e}")
        
        return None

class YelpAPICollector:
    """
    Yelp Fusion API collector (proper way to get Yelp data).
    """
    
    BASE_URL = "https://api.yelp.com/v3"
    
    def __init__(self, api_key: str = None, use_cache: bool = True):
        """Initialize Yelp API collector."""
        self.api_key = api_key
        if not self.api_key:
            logger.warning("Yelp API key not configured")
        
        self.cache = CacheManager() if use_cache else None
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
    
    def search_business(self, name: str, location: str = None) -> Optional[str]:
        """
        Search for a business and return its ID.
        
        Args:
            name: Business name
            location: Location string
        
        Returns:
            Business ID or None
        """
        if not self.api_key:
            return None
        
        cache_key = f"yelp_search:{name}:{location}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        try:
            params = {'term': name}
            if location:
                params['location'] = location
            
            response = self.session.get(
                f"{self.BASE_URL}/businesses/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('businesses'):
                business_id = data['businesses'][0]['id']
                
                if self.cache:
                    self.cache.set(cache_key, business_id)
                
                return business_id
        
        except Exception as e:
            logger.error(f"Error searching Yelp: {e}")
        
        return None
    
    def get_reviews(self, business_id: str) -> List[Dict]:
        """
        Get reviews for a business.
        
        Args:
            business_id: Yelp business ID
        
        Returns:
            List of reviews
        """
        if not self.api_key:
            return []
        
        cache_key = f"yelp_reviews:{business_id}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/businesses/{business_id}/reviews",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            reviews = []
            for review in data.get('reviews', []):
                reviews.append({
                    'text': review.get('text', ''),
                    'rating': review.get('rating', 0),
                    'time_created': review.get('time_created', ''),
                    'source': 'yelp'
                })
            
            if self.cache:
                self.cache.set(cache_key, reviews)
            
            return reviews
        
        except Exception as e:
            logger.error(f"Error getting Yelp reviews: {e}")
            return []
