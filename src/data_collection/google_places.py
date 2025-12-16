"""
Google Places API data collector.
This module handles restaurant data collection from Google Places.
"""
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time

from src.utils.logger import setup_logger
from src.utils.cache import CacheManager
from config.config import Config

logger = setup_logger(__name__)

@dataclass
class RestaurantData:
    """Data structure for restaurant information."""
    name: str
    place_id: str
    address: str
    rating: float
    total_ratings: int
    phone: Optional[str] = None
    website: Optional[str] = None
    price_level: Optional[int] = None
    types: List[str] = None
    reviews: List[Dict] = None
    photos: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

class GooglePlacesCollector:
    """Collector for Google Places data."""
    
    BASE_URL = "https://maps.googleapis.com/maps/api/place"
    
    def __init__(self, api_key: str = None, use_cache: bool = True):
        """
        Initialize Google Places collector.
        
        Args:
            api_key: Google Places API key
            use_cache: Whether to use caching
        """
        self.api_key = api_key or Config.GOOGLE_PLACES_API_KEY
        if not self.api_key:
            raise ValueError("Google Places API key not configured")
        
        self.cache = CacheManager() if use_cache else None
        self.session = requests.Session()
    
    def search_restaurant(self, query: str, location: str = None) -> Optional[str]:
        """
        Search for a restaurant and return its place_id.
        
        Args:
            query: Restaurant name or search query
            location: Optional location (e.g., "New York, NY")
        
        Returns:
            place_id if found, None otherwise
        """
        cache_key = f"search:{query}:{location}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached search result for: {query}")
                return cached
        
        # Construct search query
        search_query = query
        if location:
            search_query = f"{query} in {location}"
        
        params = {
            'query': search_query,
            'key': self.api_key,
            'type': 'restaurant'
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/textsearch/json",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                place_id = data['results'][0]['place_id']
                
                # Cache result
                if self.cache:
                    self.cache.set(cache_key, place_id)
                
                logger.info(f"Found restaurant: {data['results'][0]['name']}")
                return place_id
            else:
                logger.warning(f"No results found for: {query}")
                return None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching for restaurant: {e}")
            return None
    
    def get_restaurant_details(self, place_id: str) -> Optional[RestaurantData]:
        """
        Get detailed information about a restaurant.
        
        Args:
            place_id: Google Places ID
        
        Returns:
            RestaurantData object or None
        """
        cache_key = f"details:{place_id}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached details for place_id: {place_id}")
                return RestaurantData(**cached)
        
        params = {
            'place_id': place_id,
            'key': self.api_key,
            'fields': 'name,place_id,formatted_address,rating,user_ratings_total,'
                     'formatted_phone_number,website,price_level,types,reviews,photos'
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/details/json",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'OK':
                result = data['result']
                
                # Extract photo references
                photos = []
                if 'photos' in result:
                    for photo in result['photos'][:5]:  # Limit to 5 photos
                        photos.append(photo.get('photo_reference', ''))
                
                # Extract reviews
                reviews = []
                if 'reviews' in result:
                    for review in result['reviews']:
                        reviews.append({
                            'author': review.get('author_name', ''),
                            'rating': review.get('rating', 0),
                            'text': review.get('text', ''),
                            'time': review.get('time', 0)
                        })
                
                restaurant_data = RestaurantData(
                    name=result.get('name', ''),
                    place_id=result.get('place_id', ''),
                    address=result.get('formatted_address', ''),
                    rating=result.get('rating', 0.0),
                    total_ratings=result.get('user_ratings_total', 0),
                    phone=result.get('formatted_phone_number'),
                    website=result.get('website'),
                    price_level=result.get('price_level'),
                    types=result.get('types', []),
                    reviews=reviews,
                    photos=photos
                )
                
                # Cache result
                if self.cache:
                    self.cache.set(cache_key, restaurant_data.to_dict())
                
                logger.info(f"Retrieved details for: {restaurant_data.name}")
                return restaurant_data
            else:
                logger.warning(f"Failed to get details: {data.get('status')}")
                return None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting restaurant details: {e}")
            return None
    
    def get_photo_url(self, photo_reference: str, max_width: int = 800) -> str:
        """
        Get URL for a photo.
        
        Args:
            photo_reference: Photo reference from Places API
            max_width: Maximum width of photo
        
        Returns:
            Photo URL
        """
        return (f"{self.BASE_URL}/photo?"
                f"maxwidth={max_width}&"
                f"photo_reference={photo_reference}&"
                f"key={self.api_key}")
    
    def download_photo(self, photo_reference: str, save_path: str) -> bool:
        """
        Download a photo from Google Places.
        
        Args:
            photo_reference: Photo reference
            save_path: Path to save the photo
        
        Returns:
            True if successful, False otherwise
        """
        try:
            url = self.get_photo_url(photo_reference)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded photo to: {save_path}")
            return True
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading photo: {e}")
            return False
    
    def collect_restaurant_data(self, 
                                restaurant_name: str, 
                                location: str = None) -> Optional[RestaurantData]:
        """
        Main method to collect all restaurant data.
        
        Args:
            restaurant_name: Name of the restaurant
            location: Optional location
        
        Returns:
            RestaurantData object or None
        """
        logger.info(f"Collecting data for: {restaurant_name}")
        
        # Search for restaurant
        place_id = self.search_restaurant(restaurant_name, location)
        if not place_id:
            return None
        
        # Get details
        restaurant_data = self.get_restaurant_details(place_id)
        
        return restaurant_data
