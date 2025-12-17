"""
LLM Reasoning Layer - Strategic use of LLM APIs.
This module handles complex reasoning that benefits from LLM capabilities.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from src.utils.logger import setup_logger
from src.utils.cache import CacheManager
from config.config import Config

logger = setup_logger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM response."""
    safety_score: float  # 0-100
    reasoning: str
    confidence: float  # 0-1
    risk_factors: List[str]
    safe_alternatives: List[str]
    raw_response: str

class LLMReasoner:
    """
    LLM-based reasoning for complex allergen safety assessment.
    Uses prompt engineering to get structured, reliable outputs.
    """
    
    def __init__(self, provider: str = "gemini", use_cache: bool = True):
        """
        Initialize LLM reasoner.
        
        Args:
            provider: LLM provider ('gemini', 'openai', 'anthropic')
            use_cache: Whether to cache responses
        """
        self.provider = provider.lower()
        self.cache = CacheManager(ttl_hours=168) if use_cache else None  # 7-day cache
        
        # Initialize appropriate client
        self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=Config.GOOGLE_GEMINI_API_KEY)
                # print("Available Gemini models:")
                # import pprint
                # for model in genai.list_models():
                #     pprint.pprint(model)
                self.client = genai.GenerativeModel('gemini-flash-lite-latest')
                logger.info("Initialized Gemini client")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.client = None
        
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.client = None
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                logger.info("Initialized Anthropic client")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                self.client = None
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _build_prompt(self, 
                     menu_items: List[str],
                     review_summary: Dict,
                     allergen_type: str = "gluten",
                     restaurant_name: str = "") -> str:
        """
        Build a carefully engineered prompt for allergen safety assessment.
        
        This is YOUR prompt engineering work - a key technical contribution.
        """
        prompt = f"""You are an expert food safety analyst specializing in allergen risk assessment for people with {allergen_type} allergies/sensitivities.

**Restaurant**: {restaurant_name}

**Task**: Assess the safety of this restaurant for someone with celiac disease (severe gluten intolerance) based on the provided data.

**Menu Items Detected** (from OCR/web scraping):
{chr(10).join(f"- {item}" for item in menu_items[:20]) if menu_items else "No menu data available"}

**Review Analysis Summary** (from rule-based detector):
- Total allergen mentions: {review_summary.get('total_mentions', 0)}
- Safety indicators found: {len(review_summary.get('safety_indicators', []))}
- Warning indicators found: {len(review_summary.get('warning_indicators', []))}
- Average risk score (rule-based): {review_summary.get('average_risk_score', 0):.1f}/100

**Key Safety Indicators Mentioned**:
{chr(10).join(f"- {ind}" for ind in review_summary.get('safety_indicators', [])[:10]) or "None found"}

**Warning Indicators Mentioned**:
{chr(10).join(f"- {warn}" for warn in review_summary.get('warning_indicators', [])[:10]) or "None found"}

**Instructions**:
1. Analyze the menu for {allergen_type}-containing items and cross-contamination risk
2. Consider the review mentions and safety/warning indicators
3. Assess whether the restaurant has dedicated {allergen_type}-free preparation
4. Evaluate staff knowledge based on review mentions
5. Consider the risk of cross-contact in the kitchen

**Output Format** (respond ONLY with valid JSON):
{{
  "safety_score": <number 0-100, where 0=completely safe, 100=extremely dangerous>,
  "confidence": <number 0-1, your confidence in this assessment>,
  "reasoning": "<brief explanation of your assessment>",
  "risk_factors": ["<list of specific risk factors>"],
  "safe_alternatives": ["<list of safe menu items or alternatives if any>"]
}}

**Important**:
- Be conservative (err on the side of caution for safety)
- Base your assessment on EVIDENCE from the data provided
- If data is insufficient, reflect lower confidence
- Focus specifically on {allergen_type} safety
"""
        return prompt
    
    def _parse_llm_response(self, raw_response: str) -> Optional[LLMResponse]:
        """
        Parse LLM response into structured format.
        
        Args:
            raw_response: Raw text from LLM
        
        Returns:
            LLMResponse object or None
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds markdown formatting
            json_str = raw_response
            
            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            # Parse JSON
            data = json.loads(json_str.strip())
            
            return LLMResponse(
                safety_score=float(data.get('safety_score', 50)),
                reasoning=data.get('reasoning', ''),
                confidence=float(data.get('confidence', 0.5)),
                risk_factors=data.get('risk_factors', []),
                safe_alternatives=data.get('safe_alternatives', []),
                raw_response=raw_response
            )
        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {raw_response}")
            return None
    
    def assess_safety(self,
                     menu_items: List[str],
                     review_summary: Dict,
                     allergen_type: str = "gluten",
                     restaurant_name: str = "") -> Optional[LLMResponse]:
        """
        Get LLM assessment of restaurant safety for allergen.
        
        Args:
            menu_items: List of menu items (from OCR or scraping)
            review_summary: Summary from rule-based analysis
            allergen_type: Type of allergen
            restaurant_name: Restaurant name
        
        Returns:
            LLMResponse object or None
        """
        # Create cache key
        cache_key = f"llm_assess:{restaurant_name}:{allergen_type}:{hash(str(menu_items))}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached LLM response for {restaurant_name}")
                return LLMResponse(**cached)
        
        if not self.client:
            logger.error("LLM client not initialized")
            return None
        
        # Build prompt
        prompt = self._build_prompt(menu_items, review_summary, allergen_type, restaurant_name)
        
        try:
            # Call LLM based on provider
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                raw_response = response.text
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Cheaper model for testing
                    messages=[
                        {"role": "system", "content": "You are a food safety expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3  # Lower temperature for more consistent outputs
                )
                raw_response = response.choices[0].message.content
            
            elif self.provider == "anthropic":
                message = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Cheaper model
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_response = message.content[0].text
            
            else:
                return None
            
            # Parse response
            llm_response = self._parse_llm_response(raw_response)
            
            if llm_response and self.cache:
                # Cache the parsed response
                cache_data = {
                    'safety_score': llm_response.safety_score,
                    'reasoning': llm_response.reasoning,
                    'confidence': llm_response.confidence,
                    'risk_factors': llm_response.risk_factors,
                    'safe_alternatives': llm_response.safe_alternatives,
                    'raw_response': llm_response.raw_response
                }
                self.cache.set(cache_key, cache_data)
            
            logger.info(f"LLM assessment complete: score={llm_response.safety_score if llm_response else 'N/A'}")
            return llm_response
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None
    
    def analyze_menu_image(self, 
                          image_path: str,
                          allergen_type: str = "gluten") -> Optional[Dict]:
        """
        Use LLM vision capabilities to analyze menu image directly.
        This is a fallback when OCR fails or for better accuracy.
        
        Args:
            image_path: Path to menu image
            allergen_type: Allergen to check for
        
        Returns:
            Analysis dict or None
        """
        # Only works with vision-capable models
        if self.provider not in ["gemini", "openai"]:
            logger.warning(f"{self.provider} doesn't support vision")
            return None
        
        cache_key = f"llm_vision:{image_path}:{allergen_type}"
        
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        try:
            from PIL import Image
            
            prompt = f"""Analyze this restaurant menu image for {allergen_type} safety.

Extract:
1. All menu items visible
2. Any allergen warnings or notes
3. Items that contain or may contain {allergen_type}
4. Any indication of {allergen_type}-free options

Respond with JSON:
{{
  "menu_items": ["list of items"],
  "contains_allergen": ["items with {allergen_type}"],
  "allergen_warnings": ["any warnings found"],
  "safe_options": ["potentially safe items"]
}}
"""
            
            if self.provider == "gemini":
                import google.generativeai as genai
                image = Image.open(image_path)
                response = self.client.generate_content([prompt, image])
                result = self._parse_llm_response(response.text)
            
            elif self.provider == "openai":
                # OpenAI vision requires base64 encoding
                import base64
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }]
                )
                result = response.choices[0].message.content
            
            if self.cache and result:
                self.cache.set(cache_key, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing menu image: {e}")
            return None
