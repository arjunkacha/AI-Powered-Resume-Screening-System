import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple
import spacy

class SimilarityModel:
    def __init__(self, model_type: str = "tfidf"):
        """
        Initialize the similarity model.
        
        Args:
            model_type: Type of model to use ("tfidf", "bert", or "spacy")
        """
        self.model_type = model_type
        
        if model_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )
        elif model_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        elif model_type == "spacy":
            self.nlp = spacy.load("en_core_web_sm")
        else:
            raise ValueError("Unsupported model type. Choose from: tfidf, bert, spacy")
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings[0]
    
    def _get_spacy_embedding(self, text: str) -> np.ndarray:
        """Generate spaCy embeddings for text."""
        doc = self.nlp(text)
        return doc.vector
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for input text based on model type."""
        if self.model_type == "tfidf":
            return self.vectorizer.transform([text]).toarray()[0]
        elif self.model_type == "bert":
            return self._get_bert_embedding(text)
        elif self.model_type == "spacy":
            return self._get_spacy_embedding(text)
    
    def fit(self, texts: List[str]) -> None:
        """Fit the model on training texts (only needed for TF-IDF)."""
        if self.model_type == "tfidf":
            self.vectorizer.fit(texts)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Reshape for cosine similarity
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        return float(cosine_similarity(emb1, emb2)[0][0])
    
    def rank_resumes(self, job_description: str, resumes: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Rank resumes based on similarity to job description.
        
        Args:
            job_description: Job description text
            resumes: List of resume dictionaries containing 'raw_text' key
            
        Returns:
            List of tuples containing (resume_dict, similarity_score)
        """
        # Get job description embedding
        job_embedding = self.get_embedding(job_description)
        job_embedding = job_embedding.reshape(1, -1)
        
        # Calculate similarities
        ranked_resumes = []
        for resume in resumes:
            resume_embedding = self.get_embedding(resume['raw_text'])
            resume_embedding = resume_embedding.reshape(1, -1)
            
            similarity = float(cosine_similarity(job_embedding, resume_embedding)[0][0])
            ranked_resumes.append((resume, similarity))
        
        # Sort by similarity score in descending order
        ranked_resumes.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_resumes 