"""
Text Cleaning Service for Evidence Extraction
Makes raw PDF text human-readable
"""

import re
from typing import str

class TextCleaner:
    """Clean extracted PDF text for human readability"""
    
    def clean_medical_evidence(self, text: str, max_length: int = 200) -> str:
        """
        Clean extracted medical evidence text
        - Remove formatting codes
        - Fix spacing
        - Extract complete sentences
        - Make human-readable
        """
        
        if not text:
            return ""
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')
        text = text.replace('\t', ' ')
        text = text.replace('\r', '')
        
        # Fix multiple newlines
        text = re.sub(r'\n+', ' ', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'=== Page \d+ of \d+ ===', '', text)
        text = re.sub(r'Page \d+', '', text)
        
        # Remove date codes that don't make sense
        text = re.sub(r'\d{2}/\d{2}/\d{4}(?=\s*[A-Z]{2,})', '', text)
        
        # Fix common OCR/PDF issues
        text = text.replace('•', '')
        text = text.replace('◦', '')
        
        # Clean up
        text = text.strip()
        
        # Try to extract complete sentences
        sentences = self._extract_complete_sentences(text)
        
        if sentences:
            # Take first 1-2 complete sentences
            result = '. '.join(sentences[:2])
            if len(result) > max_length:
                result = result[:max_length] + "..."
            return result
        
        # Fallback: just clean and truncate
        if len(text) > max_length:
            # Try to end at a word boundary
            text = text[:max_length]
            last_space = text.rfind(' ')
            if last_space > max_length * 0.7:  # At least 70% of desired length
                text = text[:last_space]
            text += "..."
        
        return text
    
    def _extract_complete_sentences(self, text: str) -> list:
        """Extract complete sentences from text"""
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Filter out incomplete or very short sentences
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences that are between 20-300 chars and start with capital
            if 20 <= len(sentence) <= 300 and sentence and sentence[0].isupper():
                complete_sentences.append(sentence)
        
        return complete_sentences
    
    def clean_filename_for_display(self, filename: str) -> str:
        """Clean filename for better display"""
        
        # Remove common prefixes
        filename = filename.replace('bailey_mahoney_', '')
        filename = filename.replace('Bailey Mahoney ', '')
        filename = filename.replace('Mahoney_Bailey_', '')
        
        # Capitalize properly
        filename = filename.replace('_', ' ')
        
        return filename

# Global instance
text_cleaner = TextCleaner()

