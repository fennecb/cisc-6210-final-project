# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Get API Keys (Free)

#### Google Places API (Required)
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project
3. Enable "Places API"
4. Create credentials → API Key
5. Copy the key

#### Gemini API (Required for LLM - Easiest & Free)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Create in new project or existing
4. Copy the key

That's it! You can skip OpenAI/Anthropic/Yelp for now.

### Step 2: Setup Project

```bash
# Navigate to project directory
cd allergen-safety-system

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure

```bash
# Create .env file
copy .env.example .env

# Edit .env and add:
GOOGLE_PLACES_API_KEY=your_key_here
GOOGLE_GEMINI_API_KEY=your_key_here
```

### Step 4: Run Demo

```bash
python demo.py
```

That's it! 🎉

---

## 📝 Testing Different Scenarios

### Test 1: Pure Rule-Based (No LLM cost)
```python
from src.pipeline import AllergenSafetyPipeline

pipeline = AllergenSafetyPipeline(use_cache=True)

# Analyze without LLM
assessment = pipeline.analyze_restaurant(
    restaurant_name="Chipotle",
    location="New York, NY",
    allergen_type="gluten",
    use_llm=False  # No API cost!
)
```

### Test 2: With LLM Reasoning (Small cost)
```python
# Same as above but use_llm=True
assessment = pipeline.analyze_restaurant(
    restaurant_name="Chipotle",
    location="New York, NY",
    allergen_type="gluten",
    use_llm=True  # ~$0.02
)
```

### Test 3: Compare Multiple Restaurants
```python
restaurants = [
    {"name": "Chipotle", "location": "New York, NY"},
    {"name": "Panera Bread", "location": "New York, NY"},
    {"name": "Five Guys", "location": "New York, NY"}
]

assessments = pipeline.batch_analyze(restaurants)

# Sort by safety
for a in sorted(assessments, key=lambda x: x.overall_safety_score):
    print(f"{a.restaurant_name}: {a.overall_safety_score:.1f}/100 - {a.get_rating()}")
```

---

## 🎥 Recording Demo Video

### Suggested Screen Recording Flow

1. **Open Terminal** - Show project structure
   ```bash
   tree -L 2  # or ls -R
   ```

2. **Show config.py** - Highlight allergen keyword dictionaries
   ```python
   # Display the configured allergen keywords
   ```

3. **Show allergen_detector.py** - Explain the detection algorithm
   ```python
   # Walk through detect_allergens() method
   ```

4. **Run demo.py** - Live analysis
   ```bash
   python demo.py
   ```

5. **Show output JSON** - Explain each component score
   ```bash
   cat data/assessments/Chipotle_gluten.json
   ```

6. **Show test passing** - Demonstrate correctness
   ```bash
   pytest tests/test_allergen_detector.py -v
   ```

### Key Points to Highlight

- Rule-based detection algorithm implementation
- Ensemble scoring method design
- Strategic LLM use for complex reasoning
- Caching system to minimize costs

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
# Make sure in project root directory
cd allergen-safety-system

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### API key not working
```bash
# Test Google Places directly
python -c "from config.config import Config; print(Config.validate_api_keys())"
```

### No restaurants found
- Check your location string: "City, State" format
- Try a well-known chain first (Chipotle, Panera, etc.)
- Check API quota: https://console.cloud.google.com

---

## 💡 Pro Tips

1. **Start Small**: Test with 1-2 restaurants first
2. **Use Cache**: Saves money and time during development
3. **Check Logs**: `logs/pipeline.log` has detailed info
4. **Test Without LLM**: Verify rule-based algorithms work independently
5. **Compare Scores**: Run same restaurant with/without LLM to show difference

---

## 📊 Expected Results

For a typical restaurant, you should see:
- **Data collection**: 5-20 reviews found
- **Allergen detection**: 0-10 mentions found
- **Menu extraction**: 5-50 items (if successful)
- **Final score**: 0-100 with clear rating
- **Processing time**: 5-15 seconds per restaurant
- **Cost**: $0.02-0.05 with paid APIs, $0 with Gemini free tier

---

## ✅ Checklist Before Demo

- [ ] API keys configured
- [ ] Demo runs without errors
- [ ] At least 3 restaurants analyzed
- [ ] Output JSON files generated
- [ ] Tests pass
- [ ] Logs show reasonable data
- [ ] Screenshots/recordings prepared
- [ ] Can explain each component

---

Good luck! 🚀
