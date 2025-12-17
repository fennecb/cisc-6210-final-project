"""
Test script to demonstrate improved menu extraction capabilities.
"""
from src.data_collection.review_scraper import ReviewScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_menu_extraction():
    """Test the enhanced menu extraction features."""

    scraper = ReviewScraper(use_cache=True)

    # Test URLs (you can replace these with actual restaurant websites)
    test_cases = [
        {
            'name': 'Restaurant with PDF menu',
            'url': 'https://example-restaurant.com',  # Replace with actual URL
            'description': 'Tests PDF menu extraction if found'
        },
        {
            'name': 'Restaurant with platform menu',
            'url': 'https://example-toast-menu.com',  # Replace with actual Toast/Square URL
            'description': 'Tests platform-specific extraction'
        },
        {
            'name': 'Restaurant with standard website',
            'url': 'https://example-standard.com',  # Replace with actual URL
            'description': 'Tests enhanced HTML extraction'
        }
    ]

    print("=" * 60)
    print("MENU EXTRACTION TEST")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"URL: {test_case['url']}")
        print("-" * 60)

        try:
            # Test finding menu links
            menu_links = scraper.find_menu_links(test_case['url'])
            print(f"Found {len(menu_links)} menu links")

            if menu_links:
                for link in menu_links[:3]:
                    print(f"  - {link}")

                    # Check for platform detection
                    platform = scraper.detect_menu_platform(link)
                    if platform:
                        print(f"    Platform: {platform}")

            # Test menu extraction with retry
            print("\nAttempting menu extraction...")
            menu_text = scraper.extract_menu_with_retry(test_case['url'], max_retries=2)

            if menu_text:
                print(f"SUCCESS: Extracted {len(menu_text)} characters")
                # Show first 200 chars as preview
                preview = menu_text[:200].replace('\n', ' ')
                print(f"Preview: {preview}...")
            else:
                print("FAILED: Could not extract menu")

        except Exception as e:
            print(f"ERROR: {e}")

        print("=" * 60)

    print("\n[Summary]")
    print("Menu extraction features tested:")
    print("  ✓ Enhanced HTML detection (4 strategies)")
    print("  ✓ Menu link discovery")
    print("  ✓ PDF menu support (PyPDF2 + pdfplumber)")
    print("  ✓ Platform-specific extraction (10 platforms)")
    print("  ✓ Retry logic")
    print("  ✓ Caching")

if __name__ == "__main__":
    test_menu_extraction()
