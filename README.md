# 🍽️ Allergen Safety System

**An intelligent multi-source allergen risk assessment system for restaurant safety**

Designed to help people with food allergies (especially celiac disease) make informed dining decisions by analyzing restaurant data from multiple sources using rule-based NLP and strategic LLM reasoning.

---

## 🎯 Project Overview

### Motivation
For individuals with severe food allergies like celiac disease, dining out can be dangerous. This system aggregates and analyzes data from multiple sources to provide data-driven safety assessments, combining:
- **Rule-Based NLP**: Domain-specific allergen detection algorithms
- **Strategic LLM Use**: Complex reasoning about cross-contamination and safety
- **Multi-Source Data**: Reviews, menus, and expert knowledge

### Key Innovation
Unlike simple LLM query tools, this system does the heavy lifting through:
1. **Custom web scraping** and data aggregation
2. **Domain-specific NLP algorithms** for allergen detection
3. **Ensemble scoring** combining multiple signals
4. **Transparent, explainable assessments**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Restaurant Name                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │  Data Collection Layer  │  ← Custom code: Web scraping
          │   (Google Places, Yelp) │     API integration
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │  Preprocessing Layer    │  ← Custom code: Rule-based NLP
          │  - Allergen Detection   │     OCR, keyword matching
          │  - Menu OCR             │     Feature engineering
          │  - Review Analysis      │
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │   LLM Reasoning Layer   │  ← Strategic API use
          │  (Gemini/GPT/Claude)    │     Prompt engineering
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │  Scoring & Aggregation  │  ← Custom code: Ensemble algorithm
          │   - Weighted scoring    │     Confidence estimation
          │   - Risk assessment     │
          └────────────┬────────────┘
                       │
          ┌────────────┴────────────┐
          │  OUTPUT: Safety Score   │
          │  + Detailed Report      │
          └─────────────────────────┘
```

---

## 📊 Technical Contributions

### 1. Data Collection (40% of work)
- **Google Places API integration** with intelligent caching
- **Web scraping** for additional review sources
- **Multi-source aggregation** pipeline
- **Robust error handling** and rate limiting

### 2. Preprocessing & NLP (30% of work)
- **Custom allergen detection** using regex and keyword matching
- **Context-aware confidence scoring** for mentions
- **Cross-contamination warning detection**
- **Menu OCR** with Tesseract/EasyOCR integration
- **Review filtering and cleaning**

### 3. LLM Integration (20% of work)
- **Prompt engineering** for safety-critical reasoning
- **Multi-provider support** (Gemini, OpenAI, Claude)
- **Structured output parsing** with error handling
- **Vision API integration** for menu analysis

### 4. Scoring Algorithm (10% of work)
- **Ensemble method** combining 4 signal types
- **Configurable weights** for different components
- **Confidence estimation** based on data availability
- **Explainable recommendations**

---

## 🧠 NLP Techniques Implemented

### Classical NLP (40% of system)

1. **Rule-Based Pattern Matching**
   - Regex with word boundaries for allergen detection
   - Context-aware confidence scoring
   - Multi-pattern aggregation

2. **Sentiment Analysis**
   - Polarity detection (-1 to +1 scale)
   - Subjectivity measurement
   - Review credibility scoring
   - Distinguishes "great gluten-free" from "no gluten-free"

3. **TF-IDF Information Retrieval**
   - Review relevance ranking
   - Important term extraction
   - Coverage analysis
   - Filters noise from irrelevant reviews

4. **Named Entity Recognition (NER)**
   - Food item extraction from menus
   - Ingredient identification
   - Preparation method detection
   - Safety equipment mentions

### Strategic LLM Use (20% of system)

5. **LLM Reasoning**
   - Complex safety assessment
   - Cross-contamination risk evaluation
   - Prompt engineering for structured outputs

### Expected Benefits

The NLP enhancements provide:
- **Better Review Filtering**: TF-IDF ranking prioritizes allergen-relevant reviews
- **Sentiment Context**: Distinguishes positive/negative allergen mentions for nuanced understanding
- **Credibility Scoring**: Weights objective reviews higher than subjective ones
- **Structured Data**: Extracts entities like food items, ingredients, and safety equipment from unstructured text

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- API keys for at least one service:
  - Google Places API (required for restaurant data)
  - Google Gemini API OR OpenAI API OR Anthropic API (for LLM reasoning)
  - Yelp Fusion API (optional)

### Step 1: Clone and Install Dependencies

```bash
# Navigate to project directory
cd allergen-safety-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for advanced NLP)
python -m spacy download en_core_web_sm
```

### Step 2: Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# At minimum, you need:
# - GOOGLE_PLACES_API_KEY
# - One of: GOOGLE_GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
```

### Step 3: Test Installation

```bash
# Run demo
python demo.py
```

---

## 💻 Usage

### Basic Analysis

```python
from src.pipeline import AllergenSafetyPipeline

# Initialize pipeline
pipeline = AllergenSafetyPipeline(
    llm_provider="gemini",  # or "openai" or "anthropic"
    use_cache=True
)

# Analyze a restaurant
assessment = pipeline.analyze_restaurant(
    restaurant_name="Chipotle",
    location="New York, NY",
    allergen_type="gluten"
)

# View results
print(f"Safety Score: {assessment.overall_safety_score}/100")
print(f"Rating: {assessment.get_rating()}")
print(f"Confidence: {assessment.confidence_score:.0%}")
```

### Batch Analysis

```python
restaurants = [
    {"name": "Chipotle", "location": "New York, NY"},
    {"name": "Panera Bread", "location": "New York, NY"},
    {"name": "Shake Shack", "location": "New York, NY"}
]

assessments = pipeline.batch_analyze(restaurants, allergen_type="gluten")

# Compare results
for assessment in sorted(assessments, key=lambda x: x.overall_safety_score):
    print(f"{assessment.restaurant_name}: {assessment.overall_safety_score:.1f}/100")
```

### Pure Rule-Based Mode (No LLM)

```python
# Useful for demonstrating rule-based algorithms without LLM dependency
assessment = pipeline.analyze_restaurant(
    restaurant_name="Chipotle",
    location="New York, NY",
    allergen_type="gluten",
    use_llm=False  # Disable LLM reasoning
)
```

---

## 📁 Project Structure

```
allergen-safety-system/
├── src/
│   ├── data_collection/       # Web scraping & API code
│   │   ├── google_places.py   # Google Places integration
│   │   └── review_scraper.py  # Web scraping utilities
│   ├── preprocessing/         # NLP algorithms
│   │   ├── allergen_detector.py  # Rule-based detection
│   │   └── menu_ocr.py        # OCR for menu images
│   ├── llm_reasoning/         # Strategic LLM use
│   │   └── llm_reasoner.py    # Prompt engineering
│   ├── scoring/               # Ensemble algorithm
│   │   └── safety_scorer.py   # Score aggregation
│   ├── utils/                 # Utilities
│   │   ├── logger.py
│   │   └── cache.py
│   └── pipeline.py            # Main orchestrator
├── config/
│   └── config.py              # Configuration management
├── data/                      # Data storage
│   ├── cache/                 # Cached API responses
│   ├── assessments/           # Output reports
│   └── temp/                  # Temporary files
├── tests/                     # Unit tests
├── demo.py                    # Demo script
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🎓 Project Report Notes

### Technical Highlights

1. **System Design**
   - Multi-stage pipeline architecture
   - Separation of concerns (data collection → preprocessing → reasoning → scoring)
   - Modular design allowing components to be tested independently

2. **Algorithm Development**
   - Custom allergen detection with context-aware confidence scoring
   - Ensemble scoring method with configurable weights
   - Confidence estimation based on data availability

3. **Engineering Best Practices**
   - Comprehensive caching to minimize API costs
   - Robust error handling and logging
   - Configurable components via Config class
   - Clean code with type hints and documentation

4. **LLM Integration Strategy**
   - Strategic use for complex reasoning only
   - Prompt engineering for structured outputs
   - Multi-provider support (vendor agnostic)
   - Validation and fallback mechanisms

### Evaluation Metrics to Report

1. **Coverage Metrics**
   - Number of reviews successfully analyzed
   - Menu item extraction success rate
   - Data source availability per restaurant

2. **Algorithm Performance**
   - Precision/recall of allergen detection
   - Agreement between rule-based and LLM scores
   - Confidence correlation with data availability

3. **System Performance**
   - Average analysis time per restaurant
   - API cost per assessment
   - Cache hit rate

---

## 🔬 Testing & Validation

### Unit Tests
```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_allergen_detector.py
```

### Manual Testing
```bash
# Test with different restaurants
python demo.py

# Check generated reports
cat data/assessments/*.json
```

---

## 💰 Cost Optimization

### Current Setup
- **Gemini 1.5 Flash**: FREE tier (15 RPM)
- **Caching**: Reduces redundant API calls by ~80%
- **Estimated cost per restaurant**: $0.02-0.05 with paid APIs

### Tips for Demo
1. Use Gemini Flash (free) during development
2. Enable caching to avoid repeat calls
3. Test with 5-10 restaurants for comprehensive demo
4. Expected total cost: $0-5 for entire project

---

## 🎬 Demo Video Suggestions

### Structure (10 minutes)
1. **Introduction** (1 min)
   - Motivation and problem statement
   - Solution overview

2. **System Architecture** (2 min)
   - Show pipeline diagram
   - Explain technical contributions
   - Highlight what makes it unique

3. **Live Demo** (4 min)
   - Run analysis on 2-3 restaurants
   - Show different safety scores
   - Explain component scores
   - Demonstrate recommendations

4. **Technical Deep Dive** (2 min)
   - Show key code snippets:
     * Allergen detection algorithm
     * Scoring ensemble method
     * Prompt engineering
   - Explain design decisions

5. **Results & Conclusion** (1 min)
   - Evaluation metrics
   - Limitations
   - Future work

### Visual Elements
- Architecture diagram
- Live terminal output
- Code snippets with syntax highlighting
- JSON output visualization
- Comparison table of restaurants

---

## 📈 Future Work

1. **Enhanced Data Sources**
   - Instagram food photos analysis
   - Direct restaurant website scraping
   - FDA allergen database integration

2. **Improved Algorithms**
   - Fine-tuned small model for allergen detection
   - Sentiment analysis for review credibility
   - Temporal trends (safety improving/declining)

3. **User Features**
   - Web interface for easy access
   - Personalized allergen profiles
   - Location-based recommendations
   - User feedback integration

4. **Research Extensions**
   - Compare LLM vs rule-based accuracy
   - Analyze cross-contamination language patterns
   - Study regional safety variations

---

## 📝 License

MIT License - Feel free to use and modify

---

## 🙏 Acknowledgments

- Inspired by the need for better food allergy safety tools
- Built for CISC 6210 Final Project
- Dataset: Google Places API, Yelp Fusion API

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Remember**: This system provides informational assessments only. Always verify allergen safety directly with restaurant staff before ordering.
