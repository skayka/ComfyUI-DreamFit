"""
Prompt enhancement utilities for DreamFit
"""

import re
from typing import List, Optional, Dict, Tuple


class PromptEnhancer:
    """
    Enhance text prompts for garment-centric generation
    """
    
    # Garment-related keywords for better understanding
    GARMENT_TYPES = [
        "shirt", "t-shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat",
        "dress", "gown", "skirt", "pants", "jeans", "shorts", "trousers", "leggings",
        "suit", "vest", "cardigan", "pullover", "tank top", "crop top"
    ]
    
    GARMENT_ATTRIBUTES = [
        "cotton", "silk", "wool", "leather", "denim", "lace", "velvet", "satin",
        "striped", "plaid", "floral", "solid", "patterned", "embroidered",
        "fitted", "loose", "oversized", "slim", "relaxed", "tailored"
    ]
    
    STYLE_KEYWORDS = [
        "casual", "formal", "elegant", "sporty", "vintage", "modern", "classic",
        "bohemian", "minimalist", "streetwear", "business", "evening", "summer"
    ]
    
    def __init__(self):
        """Initialize the prompt enhancer"""
        self.garment_pattern = re.compile(
            r'\b(' + '|'.join(self.GARMENT_TYPES) + r')\b',
            re.IGNORECASE
        )
        
    def enhance_prompt(
        self,
        prompt: str,
        garment_description: Optional[str] = None,
        style_preference: Optional[str] = None,
        use_lmm: bool = False
    ) -> str:
        """
        Enhance a text prompt for better garment generation
        
        Args:
            prompt: Original text prompt
            garment_description: Optional description of the garment
            style_preference: Optional style preference
            use_lmm: Whether to use LMM enhancement (placeholder)
            
        Returns:
            Enhanced prompt
        """
        enhanced_parts = []
        
        # Add garment description if provided
        if garment_description:
            enhanced_parts.append(garment_description)
        
        # Process original prompt
        prompt_lower = prompt.lower()
        
        # Check if prompt already contains garment information
        has_garment = bool(self.garment_pattern.search(prompt))
        
        if not has_garment and not garment_description:
            # Add generic garment context
            enhanced_parts.append("fashionable garment")\n        
        # Add original prompt
        enhanced_parts.append(prompt)
        
        # Add style preference if provided
        if style_preference:
            enhanced_parts.append(f"in {style_preference} style")
        
        # Join parts
        enhanced_prompt = ", ".join(enhanced_parts)
        
        # Clean up redundancies
        enhanced_prompt = self._clean_prompt(enhanced_prompt)
        
        # Placeholder for LMM enhancement
        if use_lmm:
            enhanced_prompt = self._enhance_with_lmm(enhanced_prompt)
        
        return enhanced_prompt
    
    def _clean_prompt(self, prompt: str) -> str:
        """
        Clean up redundancies and format prompt
        
        Args:
            prompt: Prompt to clean
            
        Returns:
            Cleaned prompt
        """
        # Remove duplicate words
        words = prompt.split()
        seen = set()
        cleaned_words = []
        
        for word in words:
            word_lower = word.lower().strip(',.;:')
            if word_lower not in seen or word_lower in ['a', 'an', 'the', 'in', 'on', 'with']:
                cleaned_words.append(word)
                seen.add(word_lower)
        
        # Rejoin and clean up punctuation
        cleaned = ' '.join(cleaned_words)
        cleaned = re.sub(r'\s+([,.;:])', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _enhance_with_lmm(self, prompt: str) -> str:
        """
        Placeholder for LMM-based prompt enhancement
        
        Args:
            prompt: Prompt to enhance
            
        Returns:
            Enhanced prompt (currently returns original)
        """
        # TODO: Integrate with actual LMM service
        # For now, just add some generic enhancements
        enhancements = [
            "high quality",
            "detailed",
            "professional photography"
        ]
        
        # Check if quality indicators already exist
        quality_indicators = ["quality", "detailed", "professional", "hd", "4k", "8k"]
        has_quality = any(indicator in prompt.lower() for indicator in quality_indicators)
        
        if not has_quality:
            prompt = f"{prompt}, {', '.join(enhancements)}"
        
        return prompt
    
    def extract_garment_info(self, prompt: str) -> Dict[str, List[str]]:
        """
        Extract garment-related information from prompt
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            Dictionary with extracted information
        """
        info = {
            "garment_types": [],
            "attributes": [],
            "styles": []
        }
        
        prompt_lower = prompt.lower()
        
        # Extract garment types
        for garment in self.GARMENT_TYPES:
            if garment in prompt_lower:
                info["garment_types"].append(garment)
        
        # Extract attributes
        for attr in self.GARMENT_ATTRIBUTES:
            if attr in prompt_lower:
                info["attributes"].append(attr)
        
        # Extract styles
        for style in self.STYLE_KEYWORDS:
            if style in prompt_lower:
                info["styles"].append(style)
        
        return info
    
    def suggest_improvements(self, prompt: str) -> List[str]:
        """
        Suggest improvements for a prompt
        
        Args:
            prompt: Original prompt
            
        Returns:
            List of suggestions
        """
        suggestions = []
        info = self.extract_garment_info(prompt)
        
        # Check for missing elements
        if not info["garment_types"]:
            suggestions.append("Consider specifying the type of garment (e.g., shirt, dress, jacket)")
        
        if not info["attributes"]:
            suggestions.append("Add material or pattern details (e.g., cotton, striped, floral)")
        
        if not info["styles"]:
            suggestions.append("Include style preference (e.g., casual, formal, vintage)")
        
        # Check prompt length
        word_count = len(prompt.split())
        if word_count < 5:
            suggestions.append("Consider adding more descriptive details")
        elif word_count > 50:
            suggestions.append("Consider simplifying the prompt for better results")
        
        return suggestions
    
    def create_negative_prompt(self, positive_prompt: str) -> str:
        """
        Create a negative prompt based on the positive prompt
        
        Args:
            positive_prompt: The positive prompt
            
        Returns:
            Suggested negative prompt
        """
        # Common negative elements for garment generation
        negative_elements = [
            "low quality",
            "blurry",
            "distorted",
            "deformed",
            "wrinkled fabric",
            "bad proportions",
            "unrealistic"
        ]
        
        # Add specific negatives based on prompt content
        info = self.extract_garment_info(positive_prompt)
        
        if info["garment_types"]:
            negative_elements.append("wrong garment type")
        
        if "elegant" in positive_prompt.lower() or "formal" in positive_prompt.lower():
            negative_elements.extend(["casual", "sloppy"])
        elif "casual" in positive_prompt.lower():
            negative_elements.extend(["formal", "overdressed"])
        
        return ", ".join(negative_elements)