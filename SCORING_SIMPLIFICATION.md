# Scoring System Simplification

## Summary
The scoring system has been simplified to focus exclusively on **LLM-based reasoning**, removing all rule-based scoring metrics. The system now also includes **keyword search functionality** for finding relevant review excerpts.

## What Changed

### 1. **Removed Rule-Based Scoring Components**

#### Before
The system used a weighted ensemble of 4 different scores:
- Rule-based score (from AllergenDetector pattern matching)
- LLM reasoning score
- Review sentiment score (calculated from mentions)
- Menu analysis score (proportion of allergen items)

These were combined using configured weights:
```python
overall_score = (
    0.20 * rule_based_score +
    0.50 * llm_reasoning +
    0.15 * review_sentiment +
    0.15 * menu_analysis
)
```

#### After
The system now uses **only the LLM safety score**:
```python
overall_score = llm_response.safety_score  # Direct from LLM
confidence = llm_response.confidence        # Direct from LLM
```

### 2. **Simplified SafetyAssessment Dataclass**

#### Removed Fields
- `rule_based_score`
- `review_sentiment_score`
- `menu_analysis_score`

#### Added Fields
- `relevant_review_excerpts: List[str]` - Keyword-matched review snippets

#### Before (src/scoring/safety_scorer.py:14-56)
```python
@dataclass
class SafetyAssessment:
    # ... restaurant info ...

    # Component scores
    rule_based_score: float
    llm_safety_score: float
    review_sentiment_score: float
    menu_analysis_score: float

    # ... other fields ...
```

#### After (src/scoring/safety_scorer.py:14-51)
```python
@dataclass
class SafetyAssessment:
    # ... restaurant info ...

    # LLM-based scores only
    overall_safety_score: float  # From LLM
    confidence_score: float      # From LLM

    # ... other fields ...
    relevant_review_excerpts: List[str]  # NEW
```

### 3. **Added Review Keyword Search**

#### New Method: `search_review_keywords()`
**Location**: src/scoring/safety_scorer.py:63-111

Searches reviews for allergen-related keywords and extracts context:

```python
def search_review_keywords(self,
                          reviews: List[Dict],
                          keywords: List[str],
                          context_window: int = 100) -> List[str]:
    """
    Search reviews for specific keywords and extract context.

    Returns relevant excerpts like:
    "[gluten]: ...the restaurant has a dedicated gluten-free menu..."
    """
```

**Features**:
- Searches for allergen keywords (e.g., "gluten", "wheat", "celiac")
- Searches for safety keywords (e.g., "gluten-free", "allergen-friendly")
- Extracts 100-character context window around each match
- Adds ellipsis for truncated text
- Returns formatted excerpts with keyword labels

**Example Output**:
```
[gluten]: ...they have a great gluten-free menu with many options...
[celiac]: ...the staff is very knowledgeable about celiac disease...
[allergen]: ...they take allergen concerns seriously here...
```

### 4. **Updated Pipeline Flow**

#### Step 3: Removed Rule-Based Analysis
**Before** (pipeline.py:119-130):
```python
# Step 3: Rule-based allergen detection
logger.info("\nSTEP 3: Running rule-based allergen detection...")
review_summary = self.allergen_detector.analyze_reviews(
    all_reviews,
    focus_allergen=allergen_type
)
logger.info(f"  Rule-based risk score: {review_summary['average_risk_score']:.1f}/100")
```

**After** (pipeline.py:119-120):
```python
# Step 3: Store reviews for LLM analysis and keyword search
logger.info(f"\nSTEP 3: Prepared {len(all_reviews)} reviews for LLM analysis")
```

#### Step 5: Pass Review Texts Directly to LLM
**Before** (pipeline.py:172-194):
```python
llm_response = self.llm_reasoner.assess_safety(
    menu_items,
    review_summary,  # Pre-processed summary with risk scores
    allergen_type,
    restaurant_data.name
)
```

**After** (pipeline.py:178-195):
```python
# Create simple review summary with actual text
review_texts = [r.get('text', '') for r in all_reviews if r.get('text')]
simple_review_summary = {
    'reviews_analyzed': len(review_texts),
    'review_texts': review_texts[:20]  # Pass up to 20 raw reviews
}

llm_response = self.llm_reasoner.assess_safety(
    menu_items,
    simple_review_summary,  # Raw review texts for LLM
    allergen_type,
    restaurant_data.name
)
```

**Key Change**: Instead of passing pre-computed risk scores, we now pass **raw review texts** to the LLM for direct analysis.

#### Step 6: Simplified Score Aggregation
**Before** (pipeline.py:196-204):
```python
assessment = self.scorer.aggregate_scores(
    rule_based_score=review_summary['average_risk_score'],
    llm_response=llm_response,
    review_summary=review_summary,
    menu_items=menu_items,
    restaurant_name=restaurant_data.name,
    allergen_type=allergen_type
)
```

**After** (pipeline.py:209-217):
```python
assessment = self.scorer.aggregate_scores(
    llm_response=llm_response,  # Only LLM response needed
    reviews=all_reviews,         # For keyword search
    menu_items=menu_items,
    restaurant_name=restaurant_data.name,
    allergen_type=allergen_type
)
```

### 5. **Updated Output Display**

#### Before (pipeline.py:219-260)
Showed all component scores:
```
[SCORES] COMPONENT SCORES:
   Rule-based Analysis: 45.2/100
   LLM Reasoning: 32.1/100
   Review Sentiment: 38.5/100
   Menu Analysis: 55.0/100
```

#### After (pipeline.py:232-273)
Shows only LLM score and adds review excerpts:
```
[*] LLM SAFETY SCORE: 32.1/100
   Rating: GENERALLY SAFE
   LLM Confidence: 85%

[REVIEWS] RELEVANT REVIEW EXCERPTS:
   1. [gluten]: ...they have a dedicated gluten-free menu...
   2. [celiac]: ...staff is knowledgeable about celiac...
   3. [allergen]: ...very careful with allergen preparation...
```

## Technical Details

### Files Modified
1. **src/scoring/safety_scorer.py**
   - Removed: `calculate_review_sentiment_score()`, `calculate_menu_analysis_score()`, `_calculate_confidence()`
   - Added: `search_review_keywords()`
   - Simplified: `aggregate_scores()` - now only uses LLM response
   - Modified: `SafetyAssessment` dataclass

2. **src/pipeline.py**
   - Removed: Rule-based allergen detection step
   - Modified: LLM receives raw review texts instead of processed summaries
   - Updated: Output display to show only LLM scores
   - Added: Display of keyword-matched review excerpts

### Keyword Search Details

**Keywords Searched** (from Config):
- Allergen-specific: Based on `allergen_type` (e.g., for gluten: "gluten", "wheat", "barley", "rye", "celiac")
- Safety keywords: "gluten-free", "dairy-free", "allergen-friendly", etc.
- General: "allergen", "allergy", "celiac"

**Search Parameters**:
- Context window: 150 characters (configurable)
- Max excerpts returned: 10 most relevant
- Case-insensitive matching
- Handles multiple occurrences of same keyword

### Data Flow

```
Restaurant Search
    ↓
Collect Reviews (Google + Yelp)
    ↓
Extract Menu Items (Photos + Website + PDF)
    ↓
Pass to LLM:
  - Raw review texts (up to 20)
  - Menu items
  - Allergen type
    ↓
LLM Analysis
    ↓
Create Assessment:
  - LLM safety score
  - LLM confidence
  - LLM risk factors
  - LLM recommendations
  - Keyword-searched review excerpts
    ↓
Display Results
```

## Benefits

### 1. **Simpler Architecture**
- No complex weight tuning
- No ensemble calculations
- Single source of truth (LLM)

### 2. **Better Transparency**
- Clear that scores come from LLM reasoning
- No "black box" weighted combinations
- Review excerpts provide evidence

### 3. **More Flexible**
- Can improve scores by improving LLM prompts
- Can add more reviews without recalculating weights
- Easier to understand what drives the score

### 4. **Review Evidence**
- Keyword search provides concrete examples
- Users can see actual review mentions
- Transparent basis for LLM conclusions

## Future Enhancements

### Potential Improvements:
1. **Enhanced Keyword Search**
   - Fuzzy matching for misspellings
   - Synonym detection
   - Sentiment analysis on excerpts

2. **More Review Data**
   - Increase from 20 to 50+ reviews passed to LLM
   - Support for longer context windows
   - Batch processing for many reviews

3. **LLM Prompt Tuning**
   - Specialized prompts per allergen type
   - Chain-of-thought reasoning
   - Few-shot examples

4. **Review Ranking**
   - Rank excerpts by relevance
   - Prioritize recent reviews
   - Weight verified purchasers higher

## Usage

No changes needed to your existing code:

```python
from src.pipeline import AllergenSafetyPipeline

pipeline = AllergenSafetyPipeline(llm_provider="gemini", use_cache=True)

assessment = pipeline.analyze_restaurant(
    restaurant_name="Restaurant Name",
    location="City, State",
    allergen_type="gluten",
    use_llm=True  # Now effectively required
)

# Assessment now contains:
# - overall_safety_score (from LLM)
# - confidence_score (from LLM)
# - risk_factors (from LLM)
# - recommended_actions (from LLM)
# - relevant_review_excerpts (keyword search)
```

## Output Example

```
============================================================
ALLERGEN SAFETY ASSESSMENT: Thai Spice Restaurant
============================================================

Allergen Type: GLUTEN

[*] LLM SAFETY SCORE: 28.5/100
   Rating: GENERALLY SAFE
   LLM Confidence: 82%

[!] RISK FACTORS (from LLM):
   • Some menu items contain soy sauce (contains wheat)
   • Shared fryer for some dishes
   • Cross-contamination risk in small kitchen

[+] SAFETY INDICATORS (from LLM):
   • Restaurant offers gluten-free menu
   • Staff trained on allergen awareness
   • Separate prep area available upon request

[>] RECOMMENDATIONS (from LLM):
   • Ask for gluten-free menu
   • Inform server about celiac disease
   • Request dishes be prepared in separate area

[FOOD] POTENTIALLY SAFE ITEMS (from LLM):
   • Pad Thai (with rice noodles)
   • Green Curry (verify sauce)
   • Fresh Spring Rolls
   • Mango Sticky Rice

[REVIEWS] RELEVANT REVIEW EXCERPTS:
   1. [gluten-free]: ...they have a separate gluten-free menu with many options...
   2. [celiac]: ...I have celiac and the staff was very accommodating...
   3. [allergen]: ...they take allergen concerns seriously here...

[DATA] DATA SOURCES:
   Reviews analyzed: 25
   Menu items found: 67
   Keyword matches in reviews: 8
   Sources used: llm_reasoning, reviews, menu

============================================================
```

## Migration Notes

### For Future Development
- The `AllergenDetector` class is still available if needed for other purposes
- Rule-based scoring methods are removed but the underlying NLP patterns remain
- All existing cache keys work the same
- JSON export format includes new `relevant_review_excerpts` field

### Backward Compatibility
- Old assessment JSON files won't have `relevant_review_excerpts`
- Missing component scores in new assessments (expected)
- API key requirements unchanged
