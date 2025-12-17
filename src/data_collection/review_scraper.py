"""
Web scraper for collecting additional restaurant reviews.
This module demonstrates YOUR web scraping skills.
"""
import re
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote_plus, urljoin
import io

from src.utils.logger import setup_logger
from src.utils.cache import CacheManager

logger = setup_logger(__name__)

# Try to import PDF parsing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF menu extraction disabled")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

class ReviewScraper:
    """Generic review scraper for restaurant websites."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize scraper."""
        self.cache = CacheManager() if use_cache else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Common menu platform patterns
        self.menu_platforms = {
            'toast': r'toast(tab|pos|order)',
            'square': r'square(up)?.*menu',
            'bentobox': r'bentobox|getbento',
            'menufy': r'menufy',
            'grubhub': r'grubhub\.com/restaurant',
            'doordash': r'doordash\.com/store',
            'ubereats': r'ubereats\.com/store',
            'chownow': r'chownow\.com',
            'olo': r'olo\.com',
            'spoton': r'spoton.*menu'
        }
    
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
    
    def detect_menu_platform(self, url: str) -> Optional[str]:
        """
        Detect if a URL belongs to a known menu hosting platform.

        Args:
            url: URL to check

        Returns:
            Platform name if detected, None otherwise
        """
        url_lower = url.lower()
        for platform, pattern in self.menu_platforms.items():
            if re.search(pattern, url_lower):
                logger.info(f"Detected menu platform: {platform}")
                return platform
        return None

    def extract_from_platform(self, url: str, platform: str) -> Optional[str]:
        """
        Extract menu using platform-specific methods.

        Args:
            url: Menu URL
            platform: Platform name

        Returns:
            Extracted menu text or None
        """
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            menu_text = ""

            # Platform-specific extraction
            if platform in ['toast', 'square', 'bentobox']:
                # These platforms often use structured JSON-LD or specific div classes
                menu_items = soup.find_all(['div', 'li'], class_=re.compile(r'item|product|dish', re.I))
                for item in menu_items:
                    name = item.find(['h3', 'h4', 'span'], class_=re.compile(r'name|title', re.I))
                    desc = item.find(['p', 'span'], class_=re.compile(r'desc|description', re.I))

                    if name:
                        menu_text += name.get_text(strip=True) + "\n"
                    if desc:
                        menu_text += desc.get_text(strip=True) + "\n"

            elif platform in ['grubhub', 'doordash', 'ubereats']:
                # Delivery platforms use specific structures
                menu_sections = soup.find_all(['div', 'section'], attrs={'data-testid': re.compile(r'menu|item', re.I)})
                if not menu_sections:
                    menu_sections = soup.find_all(['div'], class_=re.compile(r'menu.*item|menuItem', re.I))

                for section in menu_sections:
                    text = section.get_text(separator='\n', strip=True)
                    menu_text += text + "\n"

            elif platform in ['chownow', 'olo']:
                # Order platforms
                items = soup.find_all(['button', 'div'], attrs={'data-item': True})
                if not items:
                    items = soup.find_all(['div'], class_=re.compile(r'menu-item|order-item', re.I))

                for item in items:
                    menu_text += item.get_text(separator=' ', strip=True) + "\n"

            menu_text = self._clean_text(menu_text)

            if len(menu_text) > 100:
                logger.info(f"Extracted {len(menu_text)} chars from {platform} platform")
                return menu_text

        except Exception as e:
            logger.warning(f"Error extracting from {platform}: {e}")

        return None

    def find_menu_links(self, url: str) -> List[str]:
        """
        Find links to menu pages or PDFs on a restaurant website.

        Args:
            url: Restaurant website URL

        Returns:
            List of menu URLs (HTML pages or PDFs)
        """
        menu_urls = []

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text().lower()

                # Check if link text or href contains menu-related keywords
                if any(keyword in text for keyword in ['menu', 'food', 'our dishes', 'what we serve', 'order online']):
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    elif not href.startswith('http'):
                        continue

                    menu_urls.append(href)

                # Also check for direct PDF links
                elif href.endswith('.pdf') and any(kw in href.lower() for kw in ['menu', 'food']):
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    menu_urls.append(href)

                # Check for known menu platforms in href
                elif any(re.search(pattern, href.lower()) for pattern in self.menu_platforms.values()):
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    menu_urls.append(href)

            logger.info(f"Found {len(menu_urls)} potential menu URLs")

        except Exception as e:
            logger.error(f"Error finding menu links: {e}")

        return menu_urls

    def extract_pdf_menu(self, pdf_url: str) -> Optional[str]:
        """
        Extract text from a PDF menu.

        Args:
            pdf_url: URL to PDF menu

        Returns:
            Extracted text or None
        """
        cache_key = f"pdf_menu:{pdf_url}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached PDF menu for: {pdf_url}")
                return cached

        menu_text = None

        try:
            response = self.session.get(pdf_url, timeout=20)
            response.raise_for_status()

            pdf_content = io.BytesIO(response.content)

            # Try pdfplumber first (better for tables and structured menus)
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_content) as pdf:
                        text_parts = []
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                text_parts.append(text)

                        menu_text = '\n'.join(text_parts)
                        logger.info(f"Extracted {len(menu_text)} chars from PDF using pdfplumber")
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")

            # Fallback to PyPDF2
            if not menu_text and PDF_AVAILABLE:
                try:
                    pdf_content.seek(0)  # Reset stream position
                    pdf_reader = PyPDF2.PdfReader(pdf_content)
                    text_parts = []

                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                    menu_text = '\n'.join(text_parts)
                    logger.info(f"Extracted {len(menu_text)} chars from PDF using PyPDF2")
                except Exception as e:
                    logger.warning(f"PyPDF2 extraction failed: {e}")

            if menu_text:
                menu_text = self._clean_text(menu_text)

                if len(menu_text) > 100:
                    if self.cache:
                        self.cache.set(cache_key, menu_text)
                    return menu_text

        except Exception as e:
            logger.error(f"Error extracting PDF menu from {pdf_url}: {e}")

        return None

    def extract_menu_text_from_page(self, url: str) -> Optional[str]:
        """
        Extract menu text from a restaurant website.
        Enhanced with better detection patterns and fallbacks.

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
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Strategy 1: Look for menu sections by class
            menu_sections = (
                soup.find_all('div', class_=re.compile(r'menu|food|dish|item', re.I)) +
                soup.find_all('section', class_=re.compile(r'menu|food|dish', re.I)) +
                soup.find_all('article', class_=re.compile(r'menu|food|dish', re.I))
            )

            # Strategy 2: Look for menu sections by ID
            menu_sections += (
                soup.find_all('div', id=re.compile(r'menu|food|dish', re.I)) +
                soup.find_all('section', id=re.compile(r'menu|food|dish', re.I))
            )

            # Strategy 3: Look for semantic HTML elements
            menu_sections += soup.find_all(['nav', 'aside'], attrs={'aria-label': re.compile(r'menu', re.I)})

            # Strategy 4: Look for common menu container patterns
            menu_sections += (
                soup.find_all('ul', class_=re.compile(r'menu|food|items', re.I)) +
                soup.find_all('div', attrs={'data-menu': True}) +
                soup.find_all('div', attrs={'data-section': re.compile(r'menu', re.I)})
            )

            menu_text = ""
            for section in menu_sections:
                text = section.get_text(separator='\n', strip=True)
                menu_text += text + "\n"

            menu_text = self._clean_text(menu_text)

            # If we found substantial menu text, return it
            if len(menu_text) > 100:
                logger.info(f"Extracted {len(menu_text)} chars of menu text from main page")

                if self.cache:
                    self.cache.set(cache_key, menu_text)

                return menu_text

            # Fallback: Try to find dedicated menu pages
            logger.info("Insufficient menu text on main page, searching for menu links...")
            menu_links = self.find_menu_links(url)

            for menu_url in menu_links[:5]:  # Try up to 5 menu links
                try:
                    if menu_url.endswith('.pdf'):
                        # Handle PDF menus
                        logger.info(f"Found PDF menu: {menu_url}")
                        pdf_text = self.extract_pdf_menu(menu_url)
                        if pdf_text and len(pdf_text) > 200:
                            menu_text = pdf_text
                            break
                        continue

                    # Check if it's a known menu platform
                    platform = self.detect_menu_platform(menu_url)
                    if platform:
                        logger.info(f"Trying platform-specific extraction for: {platform}")
                        platform_text = self.extract_from_platform(menu_url, platform)
                        if platform_text and len(platform_text) > 200:
                            menu_text = platform_text
                            break

                    logger.info(f"Trying menu page: {menu_url}")
                    menu_response = self.session.get(menu_url, timeout=10)
                    menu_response.raise_for_status()
                    menu_soup = BeautifulSoup(menu_response.content, 'html.parser')

                    # Extract all visible text from menu page
                    # Remove script and style elements
                    for script in menu_soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()

                    page_text = menu_soup.get_text(separator='\n', strip=True)
                    page_text = self._clean_text(page_text)

                    if len(page_text) > 200:
                        logger.info(f"Extracted {len(page_text)} chars from menu page: {menu_url}")
                        menu_text = page_text
                        break

                except Exception as e:
                    logger.warning(f"Could not extract from {menu_url}: {e}")
                    continue

            if len(menu_text) > 100:
                if self.cache:
                    self.cache.set(cache_key, menu_text)
                return menu_text

        except Exception as e:
            logger.error(f"Error extracting menu from {url}: {e}")

        return None

    def search_menu_via_google(self, restaurant_name: str, location: str = None) -> List[str]:
        """
        Use Google search to find menu URLs for a restaurant.
        This is a fallback when the restaurant website doesn't have an easily parseable menu.

        Args:
            restaurant_name: Name of the restaurant
            location: Optional location

        Returns:
            List of potential menu URLs
        """
        menu_urls = []

        try:
            # Construct search query
            query = f"{restaurant_name}"
            if location:
                query += f" {location}"
            query += " menu"

            # Try to find menu links through simple search
            # Note: This is a basic implementation. For production, consider using Google Custom Search API
            logger.info(f"Searching for menu via query: {query}")

            # For now, we'll rely on the existing methods
            # A full implementation would use Google Custom Search API or similar

        except Exception as e:
            logger.error(f"Error searching for menu: {e}")

        return menu_urls

    def extract_menu_with_retry(self, url: str, max_retries: int = 2) -> Optional[str]:
        """
        Extract menu with retry logic for better reliability.

        Args:
            url: URL to extract menu from
            max_retries: Maximum number of retry attempts

        Returns:
            Extracted menu text or None
        """
        for attempt in range(max_retries + 1):
            try:
                menu_text = self.extract_menu_text_from_page(url)
                if menu_text:
                    return menu_text

                if attempt < max_retries:
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {url}")
                    time.sleep(2)  # Wait before retry

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"All retry attempts failed for {url}: {e}")

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
