"""
Entity Extraction Module - Extracts food items, ingredients, and safety terms.
Uses spaCy NER to identify structured entities in menu and review text.
"""
from typing import Dict, List, Set, Tuple
import spacy
from collections import Counter

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EntityExtractor:
    """
    Named Entity Recognition for food and allergen contexts.

    Extracts structured information from unstructured text
    using linguistic analysis.
    """

    def __init__(self):
        """Initialize entity extractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

        # Custom food-related patterns
        self.food_indicators = {
            'dish', 'item', 'option', 'menu', 'entree', 'appetizer',
            'dessert', 'sandwich', 'salad', 'soup', 'plate', 'bowl'
        }

        self.preparation_methods = {
            'fried', 'grilled', 'baked', 'roasted', 'steamed', 'sauteed',
            'boiled', 'raw', 'fresh', 'prepared', 'cooked'
        }

        self.safety_terms = {
            'dedicated', 'separate', 'certified', 'free', 'safe',
            'cross-contamination', 'allergen', 'gluten-free', 'dairy-free'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of entity types and their values
        """
        if not text:
            return {
                'products': [],
                'organizations': [],
                'food_items': [],
                'ingredients': []
            }

        doc = self.nlp(text.lower())

        products = []
        organizations = []
        food_items = []
        ingredients = []

        # Extract spaCy entities
        for ent in doc.ents:
            if ent.label_ == "PRODUCT":
                products.append(ent.text)
            elif ent.label_ == "ORG":
                organizations.append(ent.text)

        # Extract food-related nouns (custom heuristic)
        for token in doc:
            if token.pos_ == "NOUN":
                # Check if noun is preceded by food indicator
                if any(ind in token.text for ind in self.food_indicators):
                    food_items.append(token.text)

                # Check for direct object or prepositional object
                if token.dep_ in ["dobj", "pobj"]:
                    # Check if parent verb is food-related
                    if token.head.pos_ == "VERB" and token.head.lemma_ in {
                        'eat', 'serve', 'order', 'cook', 'prepare', 'make'
                    }:
                        ingredients.append(token.text)

        return {
            'products': list(set(products)),
            'organizations': list(set(organizations)),
            'food_items': list(set(food_items)),
            'ingredients': list(set(ingredients))
        }

    def extract_menu_items(self, menu_text: str) -> List[Dict]:
        """
        Extract structured menu items with ingredients and preparation.

        Args:
            menu_text: Menu text (from OCR or scraping)

        Returns:
            List of menu item dictionaries
        """
        if not menu_text:
            return []

        menu_items = []

        # Split into lines (assuming each line is a menu item)
        lines = [line.strip() for line in menu_text.split('\n') if line.strip()]

        for line in lines:
            if len(line) < 10:  # Skip very short lines
                continue

            doc = self.nlp(line.lower())

            # Extract item name (typically a noun phrase at the start)
            item_name = ""
            for chunk in doc.noun_chunks:
                item_name = chunk.text
                break

            # Extract ingredients
            ingredients = []
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ in ["dobj", "pobj", "conj"]:
                    ingredients.append(token.text)

            # Extract preparation method
            preparation = ""
            for token in doc:
                if token.text in self.preparation_methods:
                    preparation = token.text
                    break

            if item_name:
                menu_items.append({
                    'name': item_name,
                    'full_text': line,
                    'ingredients': list(set(ingredients)),
                    'preparation': preparation
                })

        logger.info(f"Extracted {len(menu_items)} menu items")
        return menu_items

    def extract_safety_mentions(self, text: str) -> Dict:
        """
        Extract allergen safety-related mentions.

        Args:
            text: Review or description text

        Returns:
            Dictionary with safety-related information
        """
        if not text:
            return {
                'safety_terms_found': [],
                'has_safety_info': False,
                'preparation_mentions': [],
                'equipment_mentions': []
            }

        doc = self.nlp(text.lower())

        safety_terms_found = []
        preparation_mentions = []
        equipment_mentions = []

        for token in doc:
            # Safety terms
            if token.text in self.safety_terms:
                safety_terms_found.append(token.text)

            # Preparation methods
            if token.text in self.preparation_methods:
                preparation_mentions.append(token.text)

            # Equipment mentions (look for specific patterns)
            if token.text in {'fryer', 'grill', 'oven', 'surface', 'utensil', 'pan'}:
                # Check for modifiers
                modifiers = [child.text for child in token.children
                           if child.dep_ == "amod"]
                if modifiers:
                    equipment_mentions.append(f"{' '.join(modifiers)} {token.text}")
                else:
                    equipment_mentions.append(token.text)

        return {
            'safety_terms_found': list(set(safety_terms_found)),
            'has_safety_info': len(safety_terms_found) > 0,
            'preparation_mentions': list(set(preparation_mentions)),
            'equipment_mentions': list(set(equipment_mentions))
        }

    def find_allergen_phrases(self,
                             text: str,
                             allergen_keywords: List[str]) -> List[Dict]:
        """
        Find complete phrases containing allergen keywords.

        Args:
            text: Text to search
            allergen_keywords: Keywords to look for

        Returns:
            List of phrase dictionaries
        """
        if not text or not allergen_keywords:
            return []

        doc = self.nlp(text.lower())
        phrases = []

        for sent in doc.sents:
            # Check if sentence contains any allergen keyword
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in allergen_keywords):

                # Extract the noun phrase containing the allergen
                for chunk in sent.noun_chunks:
                    if any(keyword in chunk.text for keyword in allergen_keywords):
                        phrases.append({
                            'phrase': chunk.text,
                            'sentence': sent.text,
                            'root': chunk.root.text,
                            'modifiers': [token.text for token in chunk
                                        if token.dep_ == "amod"]
                        })

        return phrases

    def analyze_ingredient_patterns(self,
                                   menu_items: List[str]) -> Dict:
        """
        Analyze patterns in ingredient usage across menu.

        Args:
            menu_items: List of menu item texts

        Returns:
            Dictionary with ingredient statistics
        """
        if not menu_items:
            return {
                'common_ingredients': [],
                'preparation_methods': [],
                'total_items': 0
            }

        all_ingredients = []
        all_preparations = []

        for item in menu_items:
            doc = self.nlp(item.lower())

            # Extract nouns (potential ingredients)
            for token in doc:
                if token.pos_ == "NOUN":
                    all_ingredients.append(token.text)

                if token.text in self.preparation_methods:
                    all_preparations.append(token.text)

        # Count frequencies
        ingredient_counts = Counter(all_ingredients)
        preparation_counts = Counter(all_preparations)

        return {
            'common_ingredients': ingredient_counts.most_common(10),
            'preparation_methods': preparation_counts.most_common(5),
            'total_items': len(menu_items),
            'unique_ingredients': len(set(all_ingredients))
        }
