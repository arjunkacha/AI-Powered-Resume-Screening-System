import spacy
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
import re
from typing import Dict, List, Optional

class ResumeParser:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            return pdf_extract_text(file_path)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name using NLP."""
        doc = self.nlp(text)
        # Look for proper nouns at the beginning of the text
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text, including comma-separated lists under a 'Skills' section."""
        # Try to find a 'Skills' section and extract comma-separated skills
        skills = set()
        skills_section = re.search(r"Skills\s*([\s\S]+?)(?:\n\s*[A-Z][a-z]+|\n\s*[A-Z]{2,}|$)", text)
        if skills_section:
            skills_text = skills_section.group(1)
            # Split by comma and newlines
            for skill in re.split(r",|\n", skills_text):
                skill = skill.strip().lower()
                if skill and len(skill) > 1:
                    skills.add(skill)
        # Also use the original technical skill patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust)\b',
            r'\b(?:HTML|CSS|React|Angular|Vue|Node\.js|Django|Flask|Spring|Express)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Cassandra|Oracle)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|CI/CD)\b',
            r'\b(?:Machine Learning|Deep Learning|NLP|Computer Vision|Data Science)\b'
        ]
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            skills.update(match.group() for match in matches)
        return list(skills)
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text, including blocks under 'Experience' or date patterns."""
        experiences = []
        # Try to find blocks under 'Experience' section
        exp_section = re.search(r"Experience\s*([\s\S]+?)(?:\n\s*[A-Z][a-z]+|\n\s*[A-Z]{2,}|$)", text)
        if exp_section:
            exp_text = exp_section.group(1)
            # Split by double newlines or job changes
            for block in re.split(r"\n\s*\n", exp_text):
                block = block.strip()
                if block:
                    # Try to find a date range in the block
                    date_match = re.search(r'(\d{4}\s*-\s*(?:present|\d{4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\s*-\s*(?:present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}))', block, re.IGNORECASE)
                    date_range = date_match.group(1) if date_match else ''
                    experiences.append({
                        'date_range': date_range,
                        'description': block
                    })
        # Fallback to original date-based extraction if nothing found
        if not experiences:
            experience_pattern = r'(?i)(\d{4}\s*-\s*(?:present|\d{4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\s*-\s*(?:present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}))'
            matches = re.finditer(experience_pattern, text)
            for match in matches:
                date_range = match.group(1)
                start_pos = match.start()
                end_pos = text.find('\n\n', start_pos)
                if end_pos == -1:
                    end_pos = len(text)
                experience_text = text[start_pos:end_pos].strip()
                experiences.append({
                    'date_range': date_range,
                    'description': experience_text
                })
        return experiences
    
    def parse_resume(self, file_path: str) -> Dict:
        """Parse resume and extract structured information."""
        # Determine file type and extract text
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide PDF or DOCX file.")
        
        # Extract information
        return {
            'name': self.extract_name(text),
            'skills': self.extract_skills(text),
            'experience': self.extract_experience(text),
            'raw_text': text
        } 