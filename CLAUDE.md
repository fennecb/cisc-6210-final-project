# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Allergen Safety System** - a Python-based NLP/LLM project for analyzing restaurant allergen safety using multi-source data aggregation, rule-based NLP, and strategic LLM reasoning. The system collects data from Google Places and Yelp APIs, performs rule-based allergen detection, uses LLM reasoning for complex safety assessment, and produces scored safety reports.

**Key architecture**: Multi-stage pipeline → Data Collection → Preprocessing → LLM Reasoning → Scoring → Assessment Output

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional for advanced NLP)
python -m spacy download en_core_web_sm
```

### Configuration
```bash
# Copy environment template and add API keys
cp .env.example .env
# Edit .env - at minimum need GOOGLE_PLACES_API_KEY and one LLM key
```

### Running the System
```bash
# Run demo with test restaurants
python demo.py

# Run specific tests
pytest tests/test_allergen_detector.py

# Run all tests
pytest tests/
```

## Code Architecture

### Pipeline Flow (src/pipeline.py)
The main orchestrator `AllergenSafetyPipeline` coordinates all components through a 6-step process:

1. **Data Collection** (src/data_collection/):
   - `google_places.py`: Google Places API integration with caching
   - `review_scraper.py`: Web scraping for menus, Yelp API integration

2. **Preprocessing** (src/preprocessing/):
   - `allergen_detector.py`: Rule-based NLP using regex/keyword matching for allergen detection
   - `menu_ocr.py`: OCR extraction from menu images (Tesseract/EasyOCR)

3. **LLM Reasoning** (src/llm_reasoning/):
   - `llm_reasoner.py`: Multi-provider LLM integration (Gemini/OpenAI/Anthropic) with prompt engineering for structured safety assessments

4. **Scoring** (src/scoring/):
   - `safety_scorer.py`: LLM-focused scoring system with keyword search for relevant review excerpts

5. **Output**: JSON assessment reports saved to `data/assessments/`

### Key Design Patterns

- **Caching System** (src/utils/cache.py): Hash-based caching for all API calls to minimize costs
- **Configuration** (config/config.py): Centralized config with allergen keywords, scoring weights, API keys
- **Logging** (src/utils/logger.py): Structured logging to both console and files

### Important Implementation Details

- **LLM-Centric Scoring**: The system was simplified to use LLM reasoning as the primary scoring mechanism, removing earlier rule-based ensemble scoring
- **Keyword Search**: Reviews are searched for allergen keywords and relevant excerpts are extracted for context
- **Multi-Provider Support**: LLM reasoner supports Gemini, OpenAI, and Anthropic APIs
- **Menu Extraction Strategy**: Attempts both OCR on photos (up to 5 photos) and website scraping with retry logic
- **Error Handling**: Components fail gracefully - analysis continues even if menu extraction or some data sources fail

## Testing Strategy

- **Unit Tests** (tests/): Focus on AllergenDetector rule-based algorithms
- **Manual Testing** (test_*.py files): Test files for keyword search and menu extraction
- **Demo Script** (demo.py): End-to-end testing with real restaurants

## Data Storage

```
data/
├── cache/           # Cached API responses (JSON, hashed filenames)
├── assessments/     # Output safety assessment reports
└── temp/            # Temporary files (e.g., downloaded menu images)
```

## API Keys and Costs

Required APIs:
- **Google Places API**: Required for restaurant data and reviews
- **LLM API**: One of Gemini (free tier recommended), OpenAI, or Anthropic
- **Yelp Fusion API**: Optional, adds more review data

Cost optimization via aggressive caching (7-day TTL for LLM responses).

## Common Development Patterns

### Adding a New Allergen Type
Add to `config/config.py` in `ALLERGEN_KEYWORDS` dictionary with keyword list.

### Modifying LLM Prompts
Edit prompt templates in `src/llm_reasoning/llm_reasoner.py` - the system uses structured prompts to get JSON-formatted responses.

### Adjusting Scoring Logic
Primary scoring happens in `src/scoring/safety_scorer.py` via `aggregate_scores()` method - currently passes through LLM score directly.

## Important Notes

- **Python Version**: Python 3.8+ (developed with 3.13)
- **Virtual Environment**: Always use `.venv` to avoid dependency conflicts
- **API Rate Limits**: System respects rate limits via caching; avoid disabling cache during development
- **Menu Extraction**: OCR quality varies; some restaurants may require website scraping fallback
- **LLM Provider Selection**: Gemini Flash recommended for free tier during development/testing
- **Output Semantics**: Safety scores are 0-100 where LOWER is SAFER (inverse scoring)
