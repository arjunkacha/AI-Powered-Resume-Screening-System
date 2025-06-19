import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ResumeAnalyzer:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using spaCy."""
        doc = self.nlp(text.lower())
        # Remove stop words and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using spaCy."""
        doc = self.nlp(text)
        # Common technical skills
        skills = set()
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                skills.add(token.text.lower())
        return list(skills)
    
    def calculate_similarity(self, job_description: str, resume_text: str) -> float:
        """Calculate similarity between job description and resume."""
        # Preprocess texts
        job_processed = self.preprocess_text(job_description)
        resume_processed = self.preprocess_text(resume_text)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([job_processed, resume_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    
    def analyze_resume(self, job_description: str, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resume against job description."""
        # Extract skills from job description
        job_skills = set(self.extract_skills(job_description))
        
        # Get resume skills
        resume_skills = set(resume_data.get("skills", []))
        
        # Calculate matching and missing skills
        matching_skills = list(job_skills.intersection(resume_skills))
        missing_skills = list(job_skills - resume_skills)
        
        # Calculate overall similarity
        similarity_score = self.calculate_similarity(
            job_description,
            resume_data.get("raw_text", "")
        )
        
        return {
            "match_score": similarity_score,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "total_skills_required": len(job_skills),
            "skills_matched": len(matching_skills)
        }

# Create a singleton instance
analyzer = ResumeAnalyzer() 