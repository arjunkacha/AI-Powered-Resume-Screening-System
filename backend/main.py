from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.resume_parser import ResumeParser
from models.similarity_model import SimilarityModel
import os
from typing import List, Dict
import json
from pydantic import BaseModel

app = FastAPI(
    title="Resume Screening System API",
    description="API for automated resume screening and candidate ranking",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
resume_parser = ResumeParser()
similarity_model = SimilarityModel(model_type="bert")  # Using BERT for better accuracy

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class JobDescription(BaseModel):
    text: str

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    """Parse a resume and extract structured information."""
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse resume
        result = resume_parser.parse_resume(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rank-resumes")
async def rank_resumes(
    job_description: JobDescription,
    resumes: List[Dict]
):
    """Rank resumes based on similarity to job description."""
    try:
        # Rank resumes
        ranked_resumes = similarity_model.rank_resumes(
            job_description.text,
            resumes
        )
        
        # Format response
        result = [
            {
                "resume": resume,
                "similarity_score": float(score)
            }
            for resume, score in ranked_resumes
        ]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to Resume Screening System API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 