from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from backend.models.similarity import analyzer
from typing import List, Dict
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.resume_parser import ResumeParser
from models.similarity import analyzer

router = APIRouter()
parser = ResumeParser()

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and parse a single resume file.
    """
    try:
        # Create a temporary file to store the upload
        temp_file = Path("temp") / file.filename
        temp_file.parent.mkdir(exist_ok=True)
        
        # Save the uploaded file
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse the resume
        result = parser.parse_resume(temp_file)
        
        # Clean up
        temp_file.unlink()
        
        return {
            "status": "success",
            "message": "Resume parsed successfully",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-multiple")
async def upload_multiple_resumes(files: List[UploadFile] = File(...)):
    """
    Upload and parse multiple resume files.
    """
    results = []
    errors = []
    
    for file in files:
        try:
            # Create a temporary file to store the upload
            temp_file = Path("temp") / file.filename
            temp_file.parent.mkdir(exist_ok=True)
            
            # Save the uploaded file
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Parse the resume
            result = parser.parse_resume(temp_file)
            results.append(result)
            
            # Clean up
            temp_file.unlink()
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "status": "success",
        "message": f"Processed {len(results)} resumes, {len(errors)} errors",
        "results": results,
        "errors": errors
    }

@router.post("/analyze")
async def analyze_resume(
    job_description: str = Body(...),
    resume_data: Dict = Body(...)
):
    """
    Analyze a resume against a job description.
    """
    try:
        analysis_result = analyzer.analyze_resume(job_description, resume_data)
        return {
            "status": "success",
            "message": "Analysis completed",
            "data": analysis_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-multiple")
async def analyze_multiple_resumes(
    job_description: str = Body(...),
    resumes_data: List[Dict] = Body(...)
):
    """
    Analyze multiple resumes against a job description.
    """
    try:
        results = []
        for resume_data in resumes_data:
            analysis = analyzer.analyze_resume(job_description, resume_data)
            results.append({
                "filename": resume_data.get("filename", "unknown"),
                "analysis": analysis
            })
        
        # Sort results by match score
        results.sort(key=lambda x: x["analysis"]["match_score"], reverse=True)
        
        return {
            "status": "success",
            "message": f"Analyzed {len(results)} resumes",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 